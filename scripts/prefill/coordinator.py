#!/usr/bin/env python3
"""
prefill_coordinator.py — CPU Prefill Coordinator

All prefill runs on host CPU (Xeon AMX / EPYC Turin). KV cache produced on CPU
is transferred via PCIe DMA to FPGA HBM. FPGA handles decode only.

FPGA-side prefill and GPU prefill are reserved for future extension.
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

    # Thresholds
    ttft_target_us:         float = 500_000  # target TTFT (500ms)
    ttft_acceptable_us:     float = 2_000_000  # acceptable TTFT (2s)


class PrefillTier(Enum):
    CPU = auto()


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
    """Estimate FPGA chunked prefill TTFT in microseconds.

    TTFT = all chunks completed + first decode step.
    NOT just first chunk time — decode cannot start until full KV cache is ready.
    """
    return float('inf')  # FPGA prefill reserved for future


def estimate_gpu_prefill_latency(prompt_tokens: int, cfg: PrefillConfig) -> float:
    """Estimate GPU prefill latency in microseconds."""
    return float('inf')  # GPU prefill reserved for future


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
    return PrefillDecision(
        tier=PrefillTier.CPU, ttft_us=0, total_prefill_us=0,
        num_chunks=1, chunk_size=new_tokens,
        reasoning="CPU prefill only"
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
        transfer_time_us = total_bytes / (pcie_bw_gbps * 1e3)
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
    Simulate the full prefill -> decode pipeline.

    1. Decide prefill tier (always CPU)
    2. Execute CPU prefill
    3. Transfer KV cache to FPGA HBM
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
        chunk_lat = estimate_cpu_prefill_latency(decision.chunk_size, cfg)

        t += chunk_lat
        timeline.append((chunk_start, f"PREFILL_CHUNK_{chunk}",
                        f"{decision.chunk_size} tokens, {chunk_lat/1000:.1f}ms"))

        # Transfer KV to FPGA
        xfer_us = kv_coord.cpu_to_fpga_transfer(0, chunk * decision.chunk_size,
                                                 decision.chunk_size)
        t += xfer_us
        timeline.append((chunk_start + chunk_lat, "KV_TRANSFER",
                        f"PCIe DMA, {xfer_us:.0f}us"))

    # TTFT = all chunks completed + first decode step
    # Decode can only start after full KV cache is available (all chunks done)
    decode_lat_first_token = 1e6 / 660.0  # FPGA decode rate (tok/s)
    ttft_us = t + decode_lat_first_token
    timeline.append((t, "TTFT_READY",
                    f"All {decision.num_chunks} chunks done, "
                    f"TTFT={ttft_us/1000:.1f}ms (prefill={t/1000:.1f}ms + decode={decode_lat_first_token/1000:.1f}ms)"))

    # Step 2: Decode (starts after all chunks are in KV cache)
    for tok in range(output_tokens):
        t += decode_lat_first_token
    timeline.append((t - output_tokens * decode_lat_first_token, "DECODE",
                    f"{output_tokens} tokens @ 660.0 tok/s, "
                    f"{output_tokens * decode_lat_first_token / 1000:.1f}ms"))

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
    print(" CPU Prefill Coordinator — Scenario Analysis")
    print(f" CPU: Dual Xeon GNR 6980P ({cfg.cpu_effective_tflops} TFLOPS eff)")
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
