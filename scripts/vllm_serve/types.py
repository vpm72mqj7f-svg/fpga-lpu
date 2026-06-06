"""
vllm_serve/types.py — Request, Batch, Session, and state enums.

Core data types for the vLLM serving simulator.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum, auto
import time


class RequestState(Enum):
    """Lifecycle of a single inference request."""
    WAITING   = auto()  # queued, not yet scheduled
    PREFILL   = auto()  # prefill phase (process prompt tokens)
    DECODE    = auto()  # decode phase (generate output tokens)
    FINISHED  = auto()  # generation complete (eos or max_len)
    REJECTED  = auto()  # denied due to overload


class BatchType(Enum):
    PREFILL = auto()
    DECODE  = auto()
    MIXED   = auto()   # prefill + decode in same batch


@dataclass
class Request:
    """A single inference request with full lifecycle tracking."""
    request_id: int
    arrival_time_us: float       # simulation time when request arrived
    prompt_len: int              # number of input tokens
    max_output_len: int          # max tokens to generate

    # Lifecycle state
    state: RequestState = RequestState.WAITING

    # Timing (microseconds)
    scheduled_time_us: float = 0.0     # when first scheduled
    first_token_time_us: float = 0.0   # when first output token generated
    finished_time_us: float = 0.0      # when generation completed

    # Token counters
    tokens_generated: int = 0
    tokens_processed: int = 0          # prefill tokens done so far

    # KV cache blocks allocated
    kv_block_ids: List[int] = field(default_factory=list)

    # Last token
    last_token_id: int = -1

    @property
    def ttft_us(self) -> float:
        """Time-to-first-token in microseconds."""
        if self.first_token_time_us == 0:
            return 0.0
        return self.first_token_time_us - self.arrival_time_us

    @property
    def ttft_ms(self) -> float:
        return self.ttft_us / 1000.0

    @property
    def total_latency_us(self) -> float:
        """Total end-to-end latency."""
        if self.finished_time_us == 0:
            return 0.0
        return self.finished_time_us - self.arrival_time_us

    @property
    def total_latency_ms(self) -> float:
        return self.total_latency_us / 1000.0

    @property
    def tpot_us(self) -> float:
        """Time-per-output-token (decode latency / tokens generated)."""
        if self.tokens_generated <= 1:
            return 0.0
        decode_time = self.finished_time_us - self.first_token_time_us
        return decode_time / (self.tokens_generated - 1)

    @property
    def tpot_ms(self) -> float:
        return self.tpot_us / 1000.0

    @property
    def is_finished(self) -> bool:
        return self.tokens_generated >= self.max_output_len

    @property
    def prefilled(self) -> bool:
        """Whether all prompt tokens have been processed."""
        return self.tokens_processed >= self.prompt_len


@dataclass
class Batch:
    """A batch of requests processed together."""
    batch_id: int
    batch_type: BatchType
    requests: List[Request] = field(default_factory=list)
    batch_size_tokens: int = 0          # total tokens to process

    # Timing
    created_time_us: float = 0.0
    estimated_duration_us: float = 0.0  # estimated processing time

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def max_prompt_len(self) -> int:
        if not self.requests:
            return 0
        return max(r.prompt_len for r in self.requests)


@dataclass
class Session:
    """A logical session grouping multiple requests (unused in basic sim)."""
    session_id: int
    requests: List[Request] = field(default_factory=list)
    created_time_us: float = 0.0


@dataclass
class AgentSession:
    """Multi-turn agent session with KV cache persistence across turns.

    Turn 1: full prefill (P_init tokens).
    Turns 2..N: delta prefill (P_delta tokens), KV cache reused from prior turns.
    KV blocks are pinned between turns, freed when session completes.
    """
    session_id: int
    total_turns: int
    prompt_init_tokens: int            # prompt length for first turn
    prompt_delta_tokens: int           # new tokens per subsequent turn
    output_tokens_per_turn: int        # output length per turn
    thinking_time_us: int              # gap between turns (user/agent think time)

    turns_completed: int = 0
    turn_request_ids: List[int] = field(default_factory=list)
    kv_block_ids: List[int] = field(default_factory=list)  # pinned across turns
    accumulated_tokens: int = 0        # total KV cached (prompt + output history)
    next_turn_at_us: float = 0.0
    created_time_us: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.turns_completed < self.total_turns

    @property
    def is_first_turn(self) -> bool:
        return self.turns_completed == 0

    def effective_prompt_len(self) -> int:
        """Prompt length for the NEXT turn. Turn 1 = full init, later = delta only."""
        if self.turns_completed == 0:
            return self.prompt_init_tokens
        return self.prompt_delta_tokens


@dataclass
class SchedulerStats:
    """Accumulated scheduler statistics."""
    total_requests: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_finished: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_prefill_batches: int = 0
    total_decode_batches: int = 0

    # Latency accumulators (for averaging)
    ttft_sum_us: float = 0.0
    tpot_sum_us: float = 0.0
    latency_sum_us: float = 0.0

    # Histograms (raw values for percentile computation)
    ttfts: List[float] = field(default_factory=list)
    tpots: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)

    # Scheduling metrics (S2.5)
    scheduling_latencies: List[float] = field(default_factory=list)  # arrival -> first batch inclusion
    admission_waits: List[float] = field(default_factory=list)       # arrival -> prefill admission
    batch_formation_times: List[float] = field(default_factory=list) # batch assembly duration
    decode_queue_depths: List[int] = field(default_factory=list)     # decode queue depth samples

    def record_finished(self, req: Request):
        self.total_finished += 1
        self.total_tokens_input += req.prompt_len
        self.total_tokens_output += req.tokens_generated

        if req.ttft_us > 0:
            self.ttft_sum_us += req.ttft_us
            self.ttfts.append(req.ttft_us)
        if req.tpot_us > 0:
            self.tpot_sum_us += req.tpot_us
            self.tpots.append(req.tpot_us)
        if req.total_latency_us > 0:
            self.latency_sum_us += req.total_latency_us
            self.latencies.append(req.total_latency_us)

    @property
    def avg_ttft_ms(self) -> float:
        return (self.ttft_sum_us / self.total_finished / 1000.0) if self.total_finished > 0 else 0

    @property
    def avg_tpot_ms(self) -> float:
        return (self.tpot_sum_us / self.total_finished / 1000.0) if self.total_finished > 0 else 0

    @property
    def avg_latency_ms(self) -> float:
        return (self.latency_sum_us / self.total_finished / 1000.0) if self.total_finished > 0 else 0

    @property
    def throughput_tps(self) -> float:
        """Output tokens per second."""
        total_time_s = self.total_finished  # approximate
        return self.total_tokens_output / max(total_time_s, 0.001)

    @property
    def accept_rate(self) -> float:
        return self.total_accepted / max(self.total_requests, 1)
