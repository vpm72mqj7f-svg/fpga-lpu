#!/usr/bin/env python3
"""
prefill_coordinator.py — Three-Tier Prefill Coordinator

Tier 1: CPU Prefill (Dual Xeon GNR / EPYC Turin)
  - Short prompts (< 1K tokens): full prefill on CPU
  - Agent incremental turns: chunked prefill on CPU
  - Free — CPUs are already in the server for PCIe/host duties

Tier 2: FPGA Chunked Prefill
  - Long prompts (> 1K tokens): FPGA does chunked prefill
  - FPGA pauses decode for 85ms per 512-token chunk
  - DSP array reconfigured from DECODE to PREFILL mode

Tier 3: GPU Prefill (optional)
  - Ultra-low TTFT (< 50ms) for latency-critical applications
  - Requires additional hardware (A100/H100/B200)

Decision logic:
  if prompt_len < CPU_SHORT_THRESHOLD:
      → Tier 1 (CPU, full prefill)
  elif CPU can sustain prefill rate > decode drain rate:
      → Tier 1 (CPU, chunked prefill, background)
  elif prompt_len < FPGA_CHUNK_MAX:
      → Tier 2 (FPGA chunked prefill)
  else:
      → Tier 3 (GPU, if available) or Tier 2 with larger chunks
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import time


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PrefillConfig:
    """Prefill coordinator configuration."""

    # Model parameters (DeepSeek V4 Pro fp4)
    hidden_dim:        int = 7168
    intermediate_dim:  int = 3072
    kv_latent_dim:     int = 512
    num_experts:       int = 384
    top_k:             int = 6
    num_layers:        int = 61

    # CPU prefill
    cpu_effective_tflops:   float = 10.5   # Dual Xeon GNR 6980P
    cpu_chunk_size:         int   = 128    # tokens per CPU prefill chunk
    cpu_short_threshold:    int   = 512    # below this, CPU does full prefill
    cpu_kv_production_rate: float = 323.0  # tok/s (CPU prefill throughput)

    # FPGA prefill
    fpga_chunk_size:        int   = 512    # tokens per FPGA prefill chunk
    fpga_chunk_latency_us:  float = 85_000 # 85ms per 512-token chunk
    fpga_decode_rate:       float = 660.0  # tok/s (FPGA decode throughput)

    # GPU prefill (optional)
    gpu_available:          bool  = False
    gpu_effective_tflops:   float = 187.0  # 1x A100 fp16

    # Thresholds
    ttft_target_us:         float = 500_000  # target TTFT (500ms)
    ttft_acceptable_us:     float = 2_000_000  # acceptable TTFT (2s)


class PrefillTier(Enum):
    CPU = auto()       # Tier 1: Host CPU
    FPGA = auto()      # Tier 2: FPGA chunked prefill
    GPU = auto()       # Tier 3: Dedicated GPU


# ═══════════════════════════════════════════════════════════════════════════
# Prefill Latency Model
# ═══════════════════════════════════════════════════════════════════════════

def estimate_prefill_macs(prompt_tokens: int, cfg: PrefillConfig) -> float:
    """Estimate total MACs for prefill of `prompt_tokens` tokens."""
    H, I, KL, NE, TK, NL = (cfg.hidden_dim, cfg.intermediate_dim,
                             cfg.kv_latent_dim, cfg.num_experts,
                             cfg.top_k, cfg.num_layers)
    P = prompt_tokens

    # Per-layer MACs (batched GEMM)
    # MLA QKV
    mla_q_macs   = P * H * H                     # Q projection [P,H]×[H,H]
    mla_k_macs   = P * H * KL                    # K latent
    mla_v_macs   = P * H * KL                    # V latent
    mla_kup_macs = P * KL * H                    # K up-projection
    mla_vup_macs = P * KL * H                    # V up-projection

    # Attention (causal, over growing prefix)
    # For chunked prefill with cached prefix of S tokens:
    # attention = P * (S + P/2) * H (average over triangular mask)
    # Simplified: treat as P * (P/2) * H for first chunk
    attn_macs    = P * (P / 2) * H

    # Shared Expert FFN
    shared_macs  = 3 * P * H * I   # gate + up + down

    # Routed Experts (top-K)
    routed_macs  = TK * 3 * P * H * I

    # Router
    router_macs  = P * H * NE

    per_layer = (mla_q_macs + mla_k_macs + mla_v_macs +
                 mla_kup_macs + mla_vup_macs + attn_macs +
                 shared_macs + routed_macs + router_macs)

    return per_layer * NL


def estimate_cpu_prefill_latency(prompt_tokens: int, cfg: PrefillConfig) -> float:
    """Estimate CPU prefill latency in microseconds."""
    total_macs = estimate_prefill_macs(prompt_tokens, cfg)
    effective_tflops = cfg.cpu_effective_tflops * 1e12
    latency_s = total_macs / effective_tflops
    return latency_s * 1e6  # microseconds


def estimate_fpga_prefill_latency(prompt_tokens: int, cfg: PrefillConfig) -> float:
    """Estimate FPGA chunked prefill latency in microseconds."""
    chunk_size = cfg.fpga_chunk_size
    num_chunks = math.ceil(prompt_tokens / chunk_size)
    # First chunk determines TTFT
    # Each chunk: chunk_latency_us
    return cfg.fpga_chunk_latency_us  # TTFT = first chunk


def estimate_gpu_prefill_latency(prompt_tokens: int, cfg: PrefillConfig) -> float:
    """Estimate GPU prefill latency in microseconds."""
    if not cfg.gpu_available:
        return float('inf')
    total_macs = estimate_prefill_macs(prompt_tokens, cfg)
    effective_tflops = cfg.gpu_effective_tflops * 1e12
    latency_s = total_macs / effective_tflops
    return latency_s * 1e6


def cpu_can_sustain(prompt_tokens: int, cfg: PrefillConfig) -> bool:
    """Check if CPU prefill rate can keep up with FPGA decode rate."""
    # CPU produces KV entries at cpu_kv_production_rate tok/s
    # FPGA consumes KV entries at fpga_decode_rate tok/s
    # If CPU rate >= FPGA rate, CPU can prefill faster than FPGA drains
    return cfg.cpu_kv_production_rate >= cfg.fpga_decode_rate


def cpu_kv_drain_gap(prompt_tokens: int, cfg: PrefillConfig) -> float:
    """How many seconds before FPGA catches up to CPU prefill."""
    cpu_rate = cfg.cpu_kv_production_rate
    fpga_rate = cfg.fpga_decode_rate
    if cpu_rate >= fpga_rate:
        return 0.0  # CPU keeps up, no gap
    # Gap: time for FPGA to drain what CPU produced
    return prompt_tokens / fpga_rate - prompt_tokens / cpu_rate


# ═══════════════════════════════════════════════════════════════════════════
# Coordinator Decision Logic
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PrefillDecision:
    """Result of prefill tier selection."""
    tier: PrefillTier
    ttft_us: float
    total_prefill_us: float
    num_chunks: int
    chunk_size: int
    reasoning: str
    # For CPU tier: can it sustain?
    cpu_sustainable: bool = False
    cpu_drain_gap_s: float = 0.0


def decide_prefill_tier(prompt_tokens: int, prefix_cached_tokens: int,
                        cfg: PrefillConfig) -> PrefillDecision:
    """
    Decide which prefill tier to use.

    Args:
        prompt_tokens: Total tokens to prefill (including system prompt)
        prefix_cached_tokens: Tokens already in KV cache (from prior turns)
        cfg: Prefill configuration

    Returns:
        PrefillDecision with tier, TTFT, and reasoning.
    """
    new_tokens = prompt_tokens - prefix_cached_tokens
    if new_tokens <= 0:
        return PrefillDecision(
            tier=PrefillTier.CPU, ttft_us=0, total_prefill_us=0,
            num_chunks=0, chunk_size=0,
            reasoning="All tokens already cached (prefix reuse)"
        )

    # ── Tier 1: CPU ───────────────────────────────────────
    cpu_lat = estimate_cpu_prefill_latency(new_tokens, cfg)
    cpu_sustainable = cpu_can_sustain(new_tokens, cfg)

    # CPU full prefill (no chunking): viable for short prompts
    if new_tokens <= cfg.cpu_short_threshold:
        if cpu_lat < cfg.ttft_acceptable_us:
            return PrefillDecision(
                tier=PrefillTier.CPU, ttft_us=cpu_lat,
                total_prefill_us=cpu_lat,
                num_chunks=1, chunk_size=new_tokens,
                reasoning=f"Short prompt ({new_tokens} tok), CPU full prefill, "
                          f"TTFT={cpu_lat/1000:.0f}ms",
                cpu_sustainable=True, cpu_drain_gap_s=0
            )

    # CPU chunked prefill: viable for medium prompts
    chunk_size = cfg.cpu_chunk_size
    num_chunks = math.ceil(new_tokens / chunk_size)
    cpu_chunk_lat = estimate_cpu_prefill_latency(chunk_size, cfg)
    cpu_total = cpu_chunk_lat * num_chunks
    drain_gap = cpu_kv_drain_gap(new_tokens, cfg)

    # CPU chunked prefill is viable if:
    #   1. TTFT is under target (500ms), AND
    #   2. Total prefill time is not absurd (< 10s), AND
    #   3. KV drain gap is manageable (< 2s stall)
    cpu_viable = (cpu_chunk_lat < cfg.ttft_target_us and
                  cpu_total < 10_000_000 and      # 10s max total prefill
                  drain_gap < 2.0)                 # 2s max stall

    if cpu_viable:
        return PrefillDecision(
            tier=PrefillTier.CPU, ttft_us=cpu_chunk_lat,
            total_prefill_us=cpu_total,
            num_chunks=num_chunks, chunk_size=chunk_size,
            reasoning=f"CPU chunked prefill, P={chunk_size}x{num_chunks}, "
                      f"TTFT={cpu_chunk_lat/1000:.0f}ms, total={cpu_total/1e6:.1f}s",
            cpu_sustainable=cpu_sustainable,
            cpu_drain_gap_s=drain_gap
        )

    # ── Tier 2: FPGA Chunked Prefill ──────────────────────
    fpga_lat = estimate_fpga_prefill_latency(new_tokens, cfg)
    fpga_chunks = math.ceil(new_tokens / cfg.fpga_chunk_size)

    if fpga_lat < cfg.ttft_acceptable_us:
        return PrefillDecision(
            tier=PrefillTier.FPGA, ttft_us=fpga_lat,
            total_prefill_us=cfg.fpga_chunk_latency_us * fpga_chunks,
            num_chunks=fpga_chunks, chunk_size=cfg.fpga_chunk_size,
            reasoning=f"FPGA chunked prefill, P={cfg.fpga_chunk_size}×{fpga_chunks}, "
                      f"TTFT={fpga_lat/1000:.0f}ms (first chunk), "
                      f"CPU would take {cpu_lat/1000:.0f}ms"
        )

    # ── Tier 3: GPU (if available) ────────────────────────
    if cfg.gpu_available:
        gpu_lat = estimate_gpu_prefill_latency(new_tokens, cfg)
        return PrefillDecision(
            tier=PrefillTier.GPU, ttft_us=gpu_lat,
            total_prefill_us=gpu_lat,
            num_chunks=1, chunk_size=new_tokens,
            reasoning=f"GPU prefill, TTFT={gpu_lat/1000:.0f}ms"
        )

    # ── Fallback: CPU with large chunks ───────────────────
    return PrefillDecision(
        tier=PrefillTier.CPU, ttft_us=cpu_chunk_lat,
        total_prefill_us=cpu_chunk_lat * num_chunks,
        num_chunks=num_chunks, chunk_size=chunk_size,
        reasoning=f"Fallback: CPU chunked (no better option), "
                  f"TTFT={cpu_chunk_lat/1000:.0f}ms"
    )


# ═══════════════════════════════════════════════════════════════════════════
# KV Cache Coordination
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KVCacheCoordinator:
    """
    Manages KV cache across CPU prefill and FPGA decode.

    CPU prefill produces KV entries → DMA to FPGA HBM.
    FPGA decode consumes KV entries from HBM → generates tokens.
    """

    fpga_hbm_kv_capacity: int = 4096 * 4  # tokens (4K slots × 4 cards?)

    def cpu_to_fpga_transfer(self, layer_idx: int, start_pos: int,
                             num_tokens: int) -> float:
        """
        Transfer CPU-produced KV cache entries to FPGA HBM.

        PCIe 5.0 x16 bandwidth: ~28 GB/s effective.
        KV entry size: K_latent(512) + V_latent(512) = 1024 fp8 = 1 KB.
        For `num_tokens` tokens: `num_tokens` KB per layer.

        Returns: transfer time in microseconds.
        """
        kv_bytes_per_token = 1024  # K_latent + V_latent in fp8
        total_bytes = num_tokens * kv_bytes_per_token
        pcie_bw_gbps = 28  # GB/s
        transfer_time_us = total_bytes / (pcie_bw_gbps * 1e3) * 1e6
        return transfer_time_us

    def fpga_hbm_available(self, current_kv_tokens: int) -> int:
        """How many more KV entries FPGA HBM can hold."""
        return max(0, self.fpga_hbm_kv_capacity - current_kv_tokens)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_prefill_pipeline(prompt_tokens: int, output_tokens: int,
                              cfg: PrefillConfig, kv_coord: KVCacheCoordinator):
    """
    Simulate the full prefill → decode pipeline.

    1. Decide prefill tier
    2. Execute prefill (CPU/FPGA/GPU)
    3. Transfer KV cache to FPGA HBM (if CPU prefill)
    4. Run decode on FPGA
    5. Report timeline
    """
    cached_prefix = 0  # assume cold start
    decision = decide_prefill_tier(prompt_tokens, cached_prefix, cfg)

    timeline = []
    t = 0.0  # microseconds

    # Step 1: Prefill
    timeline.append((t, "PREFILL_START", f"Tier={decision.tier.name}, "
                     f"tokens={prompt_tokens}, chunks={decision.num_chunks}"))

    for chunk in range(decision.num_chunks):
        chunk_start = t
        if decision.tier == PrefillTier.CPU:
            chunk_lat = estimate_cpu_prefill_latency(decision.chunk_size, cfg)
        elif decision.tier == PrefillTier.FPGA:
            chunk_lat = cfg.fpga_chunk_latency_us
        else:  # GPU
            chunk_lat = estimate_gpu_prefill_latency(decision.chunk_size, cfg)

        t += chunk_lat
        timeline.append((chunk_start, f"PREFILL_CHUNK_{chunk}",
                        f"{decision.chunk_size} tokens, {chunk_lat/1000:.1f}ms"))

        # Transfer KV to FPGA if CPU prefill
        if decision.tier == PrefillTier.CPU:
            xfer_us = kv_coord.cpu_to_fpga_transfer(0, chunk * decision.chunk_size,
                                                     decision.chunk_size)
            t += xfer_us
            timeline.append((chunk_start + chunk_lat, "KV_TRANSFER",
                            f"PCIe DMA, {xfer_us:.0f}us"))

    # First chunk done → TTFT
    first_chunk_lat = (estimate_cpu_prefill_latency(decision.chunk_size, cfg)
                       if decision.tier == PrefillTier.CPU
                       else cfg.fpga_chunk_latency_us)
    timeline.append((first_chunk_lat, "TTFT_READY",
                    f"First token visible at {first_chunk_lat/1000:.1f}ms"))

    # Step 2: Decode (overlaps with remaining prefill chunks)
    decode_lat_per_token = 1e6 / cfg.fpga_decode_rate  # microseconds
    for tok in range(output_tokens):
        t += decode_lat_per_token
    timeline.append((t - output_tokens * decode_lat_per_token, "DECODE",
                    f"{output_tokens} tokens @ {cfg.fpga_decode_rate} tok/s, "
                    f"{output_tokens * decode_lat_per_token / 1000:.1f}ms"))

    timeline.append((t, "DONE", "Pipeline complete"))

    return decision, sorted(timeline, key=lambda x: x[0])


# ═══════════════════════════════════════════════════════════════════════════
# Main: Test scenarios
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = PrefillConfig()
    kv = KVCacheCoordinator()

    scenarios = [
        ("Short chatbot", 200, 1000),
        ("Agent incremental", 3000, 500),
        ("Customer service", 1000, 500),
        ("RAG query", 4000, 500),
        ("Code review", 16000, 2000),
        ("Long document", 32000, 1000),
        ("128K ultra-long", 128000, 2000),
    ]

    print("=" * 90)
    print(" Three-Tier Prefill Coordinator — Scenario Analysis")
    print(f" CPU: Dual Xeon GNR 6980P ({cfg.cpu_effective_tflops} TFLOPS eff)")
    print(f" FPGA: {cfg.fpga_decode_rate} tok/s decode, "
          f"{cfg.fpga_chunk_latency_us/1000:.0f}ms/chunk")
    print(f" GPU: {'Available' if cfg.gpu_available else 'Not available'}")
    print("=" * 90)

    for name, prompt_len, output_len in scenarios:
        decision, timeline = simulate_prefill_pipeline(
            prompt_len, output_len, cfg, kv)

        print(f"\n── {name} (prompt={prompt_len}, output={output_len}) ──")
        print(f"  Tier:    {decision.tier.name}")
        print(f"  TTFT:    {decision.ttft_us/1000:.0f} ms")
        print(f"  Total:   {decision.total_prefill_us/1000:.0f} ms")
        print(f"  Chunks:  {decision.num_chunks} × {decision.chunk_size}")
        print(f"  Reason:  {decision.reasoning}")
        if decision.cpu_drain_gap_s > 0:
            print(f"  Warning: FPGA will stall for {decision.cpu_drain_gap_s:.1f}s "
                  f"waiting for CPU prefill")
