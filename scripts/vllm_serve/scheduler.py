"""
vllm_serve/scheduler.py — Continuous batching scheduler.

State machine: WAITING → PREFILL → DECODE → FINISHED

Key behaviors:
  - Prefill-first scheduling (minimize TTFT)
  - Decode batch formed from all active decode requests
  - Batch size limited by MAX_DECODE_BATCH
  - Throughput-aware: uses PipelineEngine.throughput_model() to optimize batch sizing
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import deque
import numpy as np

from .types import Request, RequestState, Batch, BatchType, SchedulerStats
from .config import (
    MAX_BATCH_SIZE, MAX_SEQ_LEN,
    MIN_DECODE_BATCH, MAX_DECODE_BATCH, MAX_PREFILL_TOKENS,
    BLOCK_SIZE, KV_BLOCK_TOKENS, KV_BYTES_PER_TOKEN,
    PREFILL_PRIORITY,
)
from fpga_arch.config import CPU_PREFILL, CPU_FP8_TFLOPS, CPU_PCIE_LATENCY_US


class ContinuousBatchingScheduler:
    """Continuous batching scheduler with prefill priority.

    On each scheduling tick:
      1. Admit waiting requests (up to capacity)
      2. Form prefill batch (priority, up to MAX_PREFILL_TOKENS)
      3. Form decode batch (all active decode requests, up to MAX_DECODE_BATCH)
      4. Return batches for execution
    """

    def __init__(self, num_chips: int = 32, max_decode_batch: int = MAX_DECODE_BATCH):
        self.num_chips = num_chips
        self.max_decode_batch = max_decode_batch

        # Queues
        self.waiting_queue: deque[Request] = deque()
        self.active_decode: List[Request] = []   # requests in decode phase
        self.prefilling: List[Request] = []       # requests currently in prefill

        # All requests
        self.all_requests: Dict[int, Request] = {}

        # Stats
        self.stats = SchedulerStats()

        # Batch counter
        self._batch_counter = 0

        # Chip ids (for KV cache allocation) — simplified: use all chips
        self.chip_ids = list(range(num_chips))

    def submit_request(self, req: Request):
        """Submit a new request to the waiting queue."""
        self.all_requests[req.request_id] = req
        self.waiting_queue.append(req)
        self.stats.total_requests += 1

    def schedule(self, current_time_us: float,
                 kv_manager, model_runner) -> List[Batch]:
        """Run one scheduling tick. Returns list of batches to execute.

        Forms MIXED batches when both prefill and decode work is available
        (prefill-first, then fill remaining slots with decode).
        Returns empty list if no work to do.
        """
        batches = []

        # 1. Admit waiting requests
        admitted = self._admit_waiting(current_time_us, kv_manager)

        # 2. Collect active decode requests (up to limit)
        decode_candidates = list(self.active_decode[:self.max_decode_batch])

        if admitted and decode_candidates:
            # MIXED: prefill + decode in same batch
            mixed_batch = self._form_mixed_batch(current_time_us, admitted,
                                                  decode_candidates)
            if mixed_batch:
                batches.append(mixed_batch)
        elif admitted:
            # Pure PREFILL batch
            prefill_batch = self._form_prefill_batch(current_time_us, admitted)
            if prefill_batch:
                batches.append(prefill_batch)

            # Still form decode batch if active
            if self.active_decode:
                decode_batch = self._form_decode_batch(current_time_us)
                if decode_batch:
                    batches.append(decode_batch)
        elif self.active_decode:
            # Pure DECODE batch
            decode_batch = self._form_decode_batch(current_time_us)
            if decode_batch:
                batches.append(decode_batch)

        return batches

    def _admit_waiting(self, current_time_us: float, kv_manager) -> List[Request]:
        """Admit waiting requests to prefill if capacity allows.

        All prefill runs on CPU — no FPGA DSP bottleneck for attention,
        so admission capacity is higher than FPGA-prefill mode.
        """
        admitted = []
        # CPU prefill: no FPGA chip 0 bottleneck → doubled admission cap.
        effective_max_batch = MAX_BATCH_SIZE * 2

        # Rough capacity check: KV cache must have space
        while self.waiting_queue and len(self.prefilling) < effective_max_batch:
            req = self.waiting_queue.popleft()
            # Check if KV cache can accommodate
            blocks_needed = max(1, req.prompt_len // KV_BLOCK_TOKENS)
            if kv_manager.total_blocks_free >= blocks_needed * len(self.chip_ids) * 0.1:
                req.state = RequestState.PREFILL
                req.scheduled_time_us = current_time_us
                self.prefilling.append(req)
                self.stats.total_accepted += 1
                admitted.append(req)
                # Record prefill admission wait time (S2.5)
                admission_wait = current_time_us - req.arrival_time_us
                self.stats.admission_waits.append(admission_wait)
                self.stats.scheduling_latencies.append(admission_wait)
            else:
                # Reject due to OOM
                req.state = RequestState.REJECTED
                self.stats.total_rejected += 1
        return admitted

    def _form_prefill_batch(self, current_time_us: float,
                            admitted: List[Request]) -> Optional[Batch]:
        """Form a prefill batch from admitted requests."""
        if not admitted:
            return None

        # Limit total prefill tokens
        total_tokens = sum(r.prompt_len for r in admitted)
        selected = admitted
        if total_tokens > MAX_PREFILL_TOKENS:
            # Select subset
            selected = []
            tokens = 0
            for r in admitted:
                if tokens + r.prompt_len <= MAX_PREFILL_TOKENS:
                    selected.append(r)
                    tokens += r.prompt_len
                else:
                    # Put back to waiting
                    r.state = RequestState.WAITING
                    self.waiting_queue.appendleft(r)

        if not selected:
            return None

        batch = Batch(
            batch_id=self._batch_counter,
            batch_type=BatchType.PREFILL,
            requests=list(selected),
            batch_size_tokens=sum(r.prompt_len for r in selected),
            created_time_us=current_time_us,
        )
        self._batch_counter += 1
        self.stats.total_prefill_batches += 1
        return batch

    def _form_mixed_batch(self, current_time_us: float,
                           admitted: List[Request],
                           decode_candidates: List[Request]) -> Optional[Batch]:
        """Form a MIXED batch: prefill requests + decode requests.

        Prefill gets priority. Remaining batch slots (up to max_decode_batch)
        are filled with decode requests.
        """
        if not admitted and not decode_candidates:
            return None

        # Limit total prefill tokens
        total_tokens = sum(r.prompt_len for r in admitted)
        selected_prefill = admitted
        if total_tokens > MAX_PREFILL_TOKENS:
            selected_prefill = []
            tokens = 0
            for r in admitted:
                if tokens + r.prompt_len <= MAX_PREFILL_TOKENS:
                    selected_prefill.append(r)
                    tokens += r.prompt_len
                else:
                    r.state = RequestState.WAITING
                    self.waiting_queue.appendleft(r)

        # Fill remaining batch slots with decode requests
        remaining_slots = self.max_decode_batch - len(selected_prefill)
        selected_decode = decode_candidates[:max(0, remaining_slots)]

        if not selected_prefill and not selected_decode:
            return None

        all_requests = selected_prefill + selected_decode
        prefill_tokens = sum(r.prompt_len for r in selected_prefill)

        batch = Batch(
            batch_id=self._batch_counter,
            batch_type=BatchType.MIXED,
            requests=all_requests,
            batch_size_tokens=prefill_tokens + len(selected_decode),
            created_time_us=current_time_us,
        )
        self._batch_counter += 1
        self.stats.total_prefill_batches += 1
        return batch

    def _form_decode_batch(self, current_time_us: float) -> Optional[Batch]:
        """Form a decode batch from all active decode requests."""
        active = list(self.active_decode)
        if not active:
            return None

        # Limit batch size
        if len(active) > self.max_decode_batch:
            active = active[:self.max_decode_batch]

        batch = Batch(
            batch_id=self._batch_counter,
            batch_type=BatchType.DECODE,
            requests=active,
            batch_size_tokens=len(active),  # 1 token each per decode step
            created_time_us=current_time_us,
        )
        self._batch_counter += 1
        self.stats.total_decode_batches += 1
        return batch

    def on_prefill_complete(self, batch: Batch, current_time_us: float):
        """Transition requests from prefill to decode.

        In MIXED batches, decode-phase requests are already in DECODE state
        and should not be re-transitioned — only prefill-phase requests.
        """
        for req in batch.requests:
            if req.state == RequestState.PREFILL:
                req.tokens_processed = req.prompt_len  # all prompt tokens processed
                req.first_token_time_us = current_time_us  # first token "generated" at prefill end
                req.state = RequestState.DECODE
                self.active_decode.append(req)
                if req in self.prefilling:
                    self.prefilling.remove(req)

    def on_mixed_batch_complete(self, batch: Batch, current_time_us: float):
        """Handle MIXED batch completion: prefill transition + decode step.

        1. Prefill-phase requests: transition to DECODE, set TTFT
        2. All requests (including newly prefilled): advance one decode token
        3. Remove finished requests from active_decode
        """
        # Stage 1: prefill transition
        self.on_prefill_complete(batch, current_time_us)

        # Stage 2: decode step for all requests in batch
        self.on_decode_step(batch, current_time_us)

    def on_decode_step(self, batch: Batch, current_time_us: float):
        """Advance decode by one token for each request in batch."""
        finished = []
        for req in batch.requests:
            # Skip if this request already finished in a prior (overlapping) batch.
            # In microbatch / pipeline-clone modes a session can be in multiple
            # in-flight batches simultaneously; we only count its completion once.
            if req.state == RequestState.FINISHED:
                continue
            req.tokens_generated += 1
            if req.is_finished:
                req.state = RequestState.FINISHED
                req.finished_time_us = current_time_us
                finished.append(req)
                self.stats.record_finished(req)

        # Remove finished from active_decode
        for req in finished:
            if req in self.active_decode:
                self.active_decode.remove(req)

    # ── Stats ──

    @property
    def active_count(self) -> int:
        return len(self.active_decode)

    @property
    def waiting_count(self) -> int:
        return len(self.waiting_queue)

    @property
    def prefill_count(self) -> int:
        return len(self.prefilling)

    @property
    def finished_count(self) -> int:
        return self.stats.total_finished

    @property
    def rejected_count(self) -> int:
        return self.stats.total_rejected

    def summary(self) -> str:
        s = self.stats
        lines = [
            f"Scheduler: {s.total_requests} submitted, {s.total_accepted} accepted, {s.total_rejected} rejected",
            f"  Active: {self.active_count} decode, {self.prefill_count} prefill, {self.waiting_count} waiting",
            f"  Finished: {s.total_finished}, tokens: {s.total_tokens_input} in / {s.total_tokens_output} out",
            f"  Avg TTFT: {s.avg_ttft_ms:.1f} ms, Avg TPOT: {s.avg_tpot_ms:.1f} ms",
            f"  Accept rate: {s.accept_rate:.1%}",
        ]
        return "\n".join(lines)
