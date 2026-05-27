#!/usr/bin/env python3
"""
prefill_scheduler.py — Concurrent CPU Prefill + FPGA Decode Scheduler (P1)

Double-buffered KV cache coordination:
  Buffer A: FPGA reads for decode
  Buffer B: CPU prefill writes → DMA to FPGA HBM
  Swap when B ready, A drained

Concurrency model:
  Timeline:
    [CPU prefill chunk N  ][CPU prefill chunk N+1]...
    [FPGA decode tokens 0..P][FPGA decode tokens P+1..]...

  CPU and FPGA run concurrently:
  - CPU prefill produces KV entries in background
  - FPGA decode consumes KV entries in foreground
  - DMA transfers happen asynchronously
  - Buffer swap occurs atomically at chunk boundaries

States:
  INIT:  CPU prefill first chunk → DMA to HBM → swap → FPGA starts
  STEADY: CPU prefill chunk N+1 || FPGA decode chunk N
  DRAIN: CPU prefill done, FPGA drains remaining tokens
  IDLE:  All done
"""

import threading
import time
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SchedulerConfig:
    """Concurrent scheduler configuration."""

    # CPU prefill
    cpu_chunk_size:   int = 128
    cpu_chunk_lat_us: float = 395_000  # 395ms for P=128 on GNR

    # FPGA decode
    fpga_decode_tok_s: float = 660.0
    fpga_decode_lat_us: float = 1_500  # 1.5ms per token

    # DMA
    dma_kv_entry_bytes: int = 1024
    dma_bw_gbps: float = 28.0  # PCIe 5.0 x16

    # KV cache
    kv_hbm_slots: int = 4096
    kv_slots_per_buffer: int = 2048  # half for double buffering

    # Timing
    sim_speedup: float = 1.0  # >1 for faster-than-realtime simulation


class SchedulerState(Enum):
    INIT = auto()
    STEADY = auto()
    DRAIN = auto()
    IDLE = auto()


# ═══════════════════════════════════════════════════════════════════════════
# KV Cache Buffer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KVBuffer:
    """One half of the double-buffered KV cache."""
    capacity: int
    tokens: int = 0
    ready: bool = False

    @property
    def available(self) -> int:
        return self.capacity - self.tokens

    def fill(self, n: int):
        self.tokens = min(self.capacity, self.tokens + n)
        self.ready = True

    def consume(self, n: int) -> int:
        consumed = min(self.tokens, n)
        self.tokens -= consumed
        if self.tokens == 0:
            self.ready = False
        return consumed


# ═══════════════════════════════════════════════════════════════════════════
# DMA Engine (simulated)
# ═══════════════════════════════════════════════════════════════════════════

class DMAEngine:
    """Simulated PCIe DMA engine for CPU→FPGA KV cache transfer."""

    def __init__(self, cfg: SchedulerConfig):
        self.cfg = cfg
        self.busy = False
        self.bytes_transferred = 0

    def transfer_time_us(self, num_tokens: int) -> float:
        """Time to DMA `num_tokens` KV entries to FPGA HBM."""
        total_bytes = num_tokens * self.cfg.dma_kv_entry_bytes
        return total_bytes / (self.cfg.dma_bw_gbps * 1e3)  # microseconds

    def start_transfer(self, num_tokens: int) -> float:
        """Start DMA transfer, return completion time in us."""
        self.busy = True
        return self.transfer_time_us(num_tokens)

    def complete(self, tokens: int):
        self.busy = False
        self.bytes_transferred += tokens * self.cfg.dma_kv_entry_bytes


# ═══════════════════════════════════════════════════════════════════════════
# Scheduler
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SchedulerEvent:
    """Timeline event."""
    time_us: float
    event_type: str
    detail: str


class ConcurrentScheduler:
    """
    Manages concurrent CPU prefill and FPGA decode.

    Double-buffered KV cache:
      Buffer A → FPGA reads for decode
      Buffer B → CPU prefill writes → DMA → ready for swap
    """

    def __init__(self, cfg: SchedulerConfig,
                 cpu_prefill_fn: Callable[[int], float],
                 fpga_decode_fn: Callable[[int], float]):
        self.cfg = cfg
        self.cpu_prefill_fn = cpu_prefill_fn
        self.fpga_decode_fn = fpga_decode_fn

        self.buf_a = KVBuffer(cfg.kv_slots_per_buffer)
        self.buf_b = KVBuffer(cfg.kv_slots_per_buffer)
        self.active_buf = 'A'  # which buffer FPGA reads
        self.dma = DMAEngine(cfg)

        self.state = SchedulerState.INIT
        self.events: list[SchedulerEvent] = []
        self.clock_us = 0.0

    def log(self, event_type: str, detail: str):
        self.events.append(SchedulerEvent(self.clock_us, event_type, detail))

    def run(self, total_prompt_tokens: int, total_decode_tokens: int):
        """
        Execute the full prefill → decode pipeline with concurrency.

        Returns: list of SchedulerEvent for timeline analysis.
        """
        cfg = self.cfg
        chunk_size = cfg.cpu_chunk_size
        num_chunks = math.ceil(total_prompt_tokens / chunk_size)

        # ── Phase 1: Initial Prefill ──────────────────────
        self.log("START", f"Prompt={total_prompt_tokens}, Decode={total_decode_tokens}")

        # Prefill first chunk → TTFT
        first_chunk = min(chunk_size, total_prompt_tokens)
        self.clock_us += self.cpu_prefill_fn(first_chunk)
        self.log("PREFILL_CHUNK_0", f"{first_chunk} tokens, TTFT={self.clock_us/1000:.1f}ms")
        self.log("TTFT", f"First token ready at {self.clock_us/1000:.1f}ms")

        # DMA to FPGA
        dma_time = self.dma.start_transfer(first_chunk)
        self.clock_us += dma_time
        self.dma.complete(first_chunk)
        self.log("DMA_0", f"{first_chunk} KV entries, {dma_time:.0f}us")

        # Fill buffer A, FPGA starts decode
        self.buf_a.fill(first_chunk)
        self.state = SchedulerState.STEADY
        remaining_prefill = total_prompt_tokens - first_chunk
        remaining_decode = total_decode_tokens
        chunk_idx = 1

        # ── Phase 2: Steady State (CPU prefill || FPGA decode) ──
        stall_count = 0
        while remaining_prefill > 0 or remaining_decode > 0:
            made_progress = False

            # CPU: prefill next chunk if tokens remain
            if remaining_prefill > 0:
                next_chunk = min(chunk_size, remaining_prefill)
                self.clock_us += self.cpu_prefill_fn(next_chunk)
                remaining_prefill -= next_chunk
                self.log(f"PREFILL_CHUNK_{chunk_idx}",
                        f"{next_chunk} tokens")

                # DMA to idle buffer
                target_buf = self.buf_b if self.active_buf == 'A' else self.buf_a
                dma_t = self.dma.start_transfer(next_chunk)
                self.clock_us += dma_t
                self.dma.complete(next_chunk)
                target_buf.fill(next_chunk)
                self.log(f"DMA_{chunk_idx}", f"{next_chunk} KV entries → buf "
                        f"{'B' if self.active_buf == 'A' else 'A'}")
                chunk_idx += 1
                made_progress = True

            # FPGA: decode tokens from active buffer
            if remaining_decode > 0:
                active_buf = self.buf_a if self.active_buf == 'A' else self.buf_b
                idle_buf = self.buf_b if self.active_buf == 'A' else self.buf_a

                # Check if swap needed
                if idle_buf.ready and active_buf.tokens == 0:
                    self.active_buf = 'B' if self.active_buf == 'A' else 'A'
                    self.log("BUFFER_SWAP", f"→ {self.active_buf}")
                    active_buf = self.buf_a if self.active_buf == 'A' else self.buf_b

                if active_buf.tokens > 0:
                    consume = min(remaining_decode, active_buf.tokens)
                    active_buf.consume(consume)
                    decode_time = consume * (1e6 / cfg.fpga_decode_tok_s)
                    self.clock_us += decode_time
                    remaining_decode -= consume
                    self.log(f"DECODE", f"{consume} tokens, "
                            f"buf_{self.active_buf} left={active_buf.tokens}")
                    made_progress = True

                # If active buffer empty and idle not ready and no more prefill → done
                if active_buf.tokens == 0 and not idle_buf.ready and remaining_prefill == 0:
                    remaining_decode = 0  # no more data to decode
                    self.log("DRAIN", "No more KV entries, decode ending")
                    made_progress = True

            # Safety: prevent infinite loop
            if not made_progress:
                stall_count += 1
                if stall_count > 10:
                    self.log("STALL", "Deadlock detected, breaking")
                    break
                self.clock_us += 100  # advance time slightly
            else:
                stall_count = 0

        # ── Phase 3: Drain ──
        self.state = SchedulerState.DRAIN
        active_buf = self.buf_a if self.active_buf == 'A' else self.buf_b
        if active_buf.tokens > 0:
            self.log("DRAIN", f"Final {active_buf.tokens} tokens")
            self.clock_us += active_buf.tokens * (1e6 / cfg.fpga_decode_tok_s)

        self.state = SchedulerState.IDLE
        self.log("DONE", f"Total time: {self.clock_us/1000:.1f}ms "
                 f"({self.clock_us/1e6:.2f}s)")

        return self.events


# ═══════════════════════════════════════════════════════════════════════════
# CPU Prefill Stub (real impl uses cpu_prefill.c via ctypes)
# ═══════════════════════════════════════════════════════════════════════════

def make_cpu_prefill_fn(cfg: SchedulerConfig):
    """Create a CPU prefill function with realistic latency."""
    def prefill_fn(num_tokens: int) -> float:
        """Return latency in microseconds for `num_tokens`."""
        # Model: linear scaling from the calibrated P=128 point
        chunks = math.ceil(num_tokens / cfg.cpu_chunk_size)
        return chunks * cfg.cpu_chunk_lat_us
    return prefill_fn


# ═══════════════════════════════════════════════════════════════════════════
# Main: Run scenarios
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = SchedulerConfig()
    cpu_prefill = make_cpu_prefill_fn(cfg)
    fpga_decode = lambda n: n * cfg.fpga_decode_lat_us

    scenarios = [
        ("Short chatbot",        200,  1000),
        ("Agent incremental",   3000,   500),
        ("RAG query",           4000,   500),
        ("Code review",        16000,  2000),
    ]

    print("=" * 80)
    print(" Concurrent CPU Prefill + FPGA Decode Scheduler")
    print(f" CPU: {cfg.cpu_chunk_lat_us/1000:.0f}ms per P={cfg.cpu_chunk_size} chunk")
    print(f" FPGA: {cfg.fpga_decode_tok_s:.0f} tok/s decode")
    print(f" DMA: {cfg.dma_bw_gbps:.0f} GB/s PCIe")
    print("=" * 80)

    for name, prompt_len, output_len in scenarios:
        print(f"\n{'='*60}")
        print(f" {name}: prompt={prompt_len}, output={output_len}")
        print(f"{'='*60}")

        sched = ConcurrentScheduler(cfg, cpu_prefill, fpga_decode)
        events = sched.run(prompt_len, output_len)

        # Show timeline summary
        for evt in events:
            marker = ""
            if evt.event_type == "TTFT":
                marker = " ← TTFT"
            elif evt.event_type == "BUFFER_SWAP":
                marker = " ← SWAP"
            elif evt.event_type == "DONE":
                marker = " ← DONE"
            print(f"  {evt.time_us/1000:8.1f}ms  {evt.event_type:20s} {evt.detail}{marker}")
