#!/usr/bin/env python3
"""
vllm_prefill.py — vLLM Scheduler Integration with CPU Prefill

Extends vLLM's scheduler to route prefill to CPU.
Drop-in module — no changes to vLLM core needed.

Usage:
    from vllm_prefill import FPGAPrefillRouter
    router = FPGAPrefillRouter(cfg)
    router.route(seq_group)  → tier decision

Integration points:
    1. scheduler.py: _schedule_prefills() → call router.route()
    2. model_runner.py: execute_model() → check prefill tier
    3. worker.py: CPU prefill worker thread

Architecture:

    vLLM Scheduler
         │
         ▼
    FPGAPrefillRouter.route(seq)
         │
         └─ CPU prefill worker thread
         │
         ▼
    KV Cache Coordinator
         │
         ├─ CPU→FPGA DMA transfer
         ├─ Double-buffer swap
         └─ FPGA decode
"""

import asyncio
import threading
import time
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, List, Tuple
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
# vLLM-compatible data structures
# ═══════════════════════════════════════════════════════════════════════════

class PrefillTier(Enum):
    CPU = auto()


@dataclass
class PrefillRequest:
    """Prefill request from vLLM scheduler."""
    seq_id: int
    prompt_tokens: int
    cached_prefix_tokens: int = 0
    arrived_at: float = 0.0

    # Assigned by router
    tier: PrefillTier = PrefillTier.CPU
    chunk_size: int = 128
    num_chunks: int = 1
    ttft_estimate_ms: float = 0.0
    total_estimate_ms: float = 0.0


@dataclass
class RouterConfig:
    """Prefill router configuration."""

    # Model
    hidden_dim: int = 7168
    num_layers: int = 61

    # CPU prefill
    cpu_effective_tflops: float = 10.5
    cpu_chunk_size: int = 128
    cpu_short_threshold: int = 512
    cpu_max_total_s: float = 10.0       # max total CPU prefill time

    # TTFT targets
    ttft_target_ms: float = 500.0
    ttft_acceptable_ms: float = 2000.0



# ═══════════════════════════════════════════════════════════════════════════
# MAC Estimator (matches coordinator.py)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_prefill_macs(prompt_tokens: int, cfg: RouterConfig) -> float:
    H, I, KL, NE, TK, NL = (cfg.hidden_dim, 3072, 512, 384, 6, cfg.num_layers)
    P = prompt_tokens
    mla  = P * H * H + 4 * P * H * KL
    attn = P * (P / 2) * H
    shared = 3 * P * H * I
    routed = TK * 3 * P * H * I
    router = P * H * NE
    return (mla + attn + shared + routed + router) * NL


def cpu_prefill_latency_ms(prompt_tokens: int, cfg: RouterConfig) -> float:
    macs = estimate_prefill_macs(prompt_tokens, cfg)
    return macs / (cfg.cpu_effective_tflops * 1e12) * 1000


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════

class FPGAPrefillRouter:
    """
    Prefill routing logic for vLLM scheduler integration.

    API compatible with vLLM's scheduler — call route() before
    _schedule_prefills() to assign prefill tier.
    """

    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.stats: Dict[str, int] = {"cpu": 0}

    def route(self, req: PrefillRequest) -> PrefillRequest:
        """Always route to CPU prefill. FPGA prefill reserved for future."""
        new_tokens = req.prompt_tokens - req.cached_prefix_tokens

        if new_tokens <= 0:
            req.tier = PrefillTier.CPU
            req.chunk_size = 0
            req.num_chunks = 0
            req.ttft_estimate_ms = 0
        else:
            num = math.ceil(new_tokens / self.cfg.cpu_chunk_size)
            chunk_ms = cpu_prefill_latency_ms(self.cfg.cpu_chunk_size, self.cfg)
            req.tier = PrefillTier.CPU
            req.chunk_size = self.cfg.cpu_chunk_size
            req.num_chunks = num
            req.ttft_estimate_ms = chunk_ms
            req.total_estimate_ms = chunk_ms * num

        self.stats["cpu"] += 1
        return req


# ═══════════════════════════════════════════════════════════════════════════
# CPU Prefill Worker (runs in background thread)
# ═══════════════════════════════════════════════════════════════════════════

class CPUPrefillWorker:
    """
    Background thread that executes CPU prefill.

    Communicates with FPGA decode via the KV cache coordinator.
    """

    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.queue: deque[PrefillRequest] = deque()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.kv_buffer_ready = threading.Event()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)

    def submit(self, req: PrefillRequest):
        self.queue.append(req)

    def _run(self):
        """Main worker loop."""
        while self.running:
            if not self.queue:
                time.sleep(0.001)
                continue

            req = self.queue.popleft()
            chunk_size = req.chunk_size

            for chunk in range(req.num_chunks):
                tokens_this_chunk = min(chunk_size,
                                       req.prompt_tokens - chunk * chunk_size)

                # Call CPU prefill (in real impl: cpu_prefill.c via ctypes)
                lat_ms = cpu_prefill_latency_ms(tokens_this_chunk, self.cfg)
                time.sleep(lat_ms / 1000)  # simulated latency

                # Signal KV buffer ready
                self.kv_buffer_ready.set()

    def pending_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.queue)


# ═══════════════════════════════════════════════════════════════════════════
# vLLM Scheduler Monkey-Patch
# ═══════════════════════════════════════════════════════════════════════════

def patch_vllm_scheduler(scheduler, router: FPGAPrefillRouter,
                         cpu_worker: CPUPrefillWorker):
    """
    Monkey-patch vLLM's scheduler to use FPGA-aware prefill routing.

    Call this once during vLLM initialization, before starting the server.

    Example:
        from vllm.core.scheduler import Scheduler
        patch_vllm_scheduler(scheduler, router, cpu_worker)
    """

    original_schedule = scheduler._schedule_prefills

    def patched_schedule_prefills(*args, **kwargs):
        """Wrapped _schedule_prefills with FPGA routing."""
        result = original_schedule(*args, **kwargs)

        # Route each scheduled sequence group
        for sg in getattr(scheduler, 'waiting', []):
            req = PrefillRequest(
                seq_id=sg.request_id,
                prompt_tokens=sg.get_token_count(),
                cached_prefix_tokens=getattr(sg, 'cached_prefix_len', 0),
                arrived_at=time.time()
            )
            req = router.route(req)

            if req.tier == PrefillTier.CPU:
                cpu_worker.submit(req)

        return result

    scheduler._schedule_prefills = patched_schedule_prefills
    return scheduler


# ═══════════════════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = RouterConfig()
    router = FPGAPrefillRouter(cfg)

    test_requests = [
        ("Short chatbot", 200, 0),
        ("Agent incremental", 3000, 0),
        ("RAG query", 4000, 0),
        ("Code review", 16000, 0),
        ("Long doc", 32000, 0),
        ("128K", 128000, 0),
        ("Cached agent", 3000, 2500),   # 2.5K cached prefix
    ]

    print("=" * 80)
    print(" FPGA Prefill Router — vLLM Integration Test")
    print(f" CPU: {cfg.cpu_effective_tflops} TFLOPS, "
          f"chunk={cfg.cpu_chunk_size}")
    print("=" * 80)

    for name, prompt, cached in test_requests:
        req = PrefillRequest(seq_id=hash(name) & 0xFFFF,
                            prompt_tokens=prompt,
                            cached_prefix_tokens=cached)
        req = router.route(req)
        new_tok = prompt - cached
        print(f"\n{name}: {prompt}tok (cached={cached}, new={new_tok})")
        print(f"  Tier:    {req.tier.name}")
        print(f"  Chunks:  {req.num_chunks} x {req.chunk_size}")
        print(f"  TTFT:    {req.ttft_estimate_ms:.0f}ms")
        print(f"  Total:   {req.total_estimate_ms:.0f}ms")

    print(f"\nRouter stats: {router.stats}")
