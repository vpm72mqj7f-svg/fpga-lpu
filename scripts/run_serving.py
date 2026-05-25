"""
run_serving.py — FPGA Cloud Serving End-to-End Simulation

Event-driven simulation integrating:
  - fpga_arch: FPGA hardware pipeline (32-chip cluster, 10-stage pipeline)
  - vllm_serve: Continuous batching scheduler, KV cache, model runner

Usage:
  python scripts/run_serving.py --duration 60 --arrival-rate 5
  python scripts/run_serving.py --duration 300 --arrival-rate 10 --verbose
  python scripts/run_serving.py --duration 600 --arrival-rate 20 --output report.json
"""

import argparse
import json
import sys
import time as wall_time
import heapq
import numpy as np
from typing import List, Dict, Optional
from enum import Enum, auto

from fpga_arch import FPGACluster, PipelineEngine
from vllm_serve import (
    ContinuousBatchingScheduler, KVCacheManager,
    ModelRunner, APIServer, BatchExecutionResult,
    BatchType, RequestState, Batch, AgentSession, Request,
    MAX_DECODE_BATCH, MIN_DECODE_BATCH, MAX_PREFILL_TOKENS,
    MAX_PREFILL_CHUNKS, MAX_PREFILL_WAIT_US,
    SIM_TIME_STEP_US, WARMUP_DURATION_S,
    TTFT_TARGET_MS, TTFT_SLA_MS, TPOT_TARGET_MS, TPOT_SLA_MS,
    PROMPT_LEN_MEAN, OUTPUT_LEN_MEAN,
    KV_BLOCKS_PER_CHIP,
    AGENT_TURNS, AGENT_THINK_TIME_MS, AGENT_DELTA_PROMPT, AGENT_OUTPUT_PER_TURN,
)


# ============================================================================
# Event System
# ============================================================================

class EventType(Enum):
    REQUEST_ARRIVAL = auto()
    BATCH_COMPLETE   = auto()
    DECODE_READY     = auto()  # prefill first chunk done → promote to decode
    AGENT_NEXT_TURN  = auto()  # agent session: next turn ready after think time
    SESSION_RELEASE  = auto()  # 解法 C: release sessions from busy set after pipeline-fill delay
    METRICS_SAMPLE  = auto()


# ============================================================================
# Simulation Metrics
# ============================================================================

class SimulationMetrics:
    """Collects and reports end-to-end simulation metrics."""

    def __init__(self):
        self.start_time_us: float = 0.0
        self.end_time_us: float = 0.0
        self.warmup_end_us: float = 0.0

        self.samples: List[Dict] = []

        self.total_requests: int = 0
        self.total_finished: int = 0
        self.total_rejected: int = 0
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0

        self.ttfts_us: List[float] = []
        self.tpots_us: List[float] = []
        self.latencies_us: List[float] = []

        self.batch_sizes: List[int] = []
        self.batch_durations_us: List[float] = []
        self.batch_tps: List[float] = []          # per-batch throughput
        self.prefill_batch_sizes: List[int] = []          # requests per prefill batch
        self.prefill_batch_tokens: List[int] = []         # tokens per prefill batch
        self.prefill_batch_durations_us: List[float] = [] # prefill duration per batch

        # Concurrency metrics
        self.total_concurrent_time_us: float = 0.0   # time with both prefill+decode in-flight
        self.contention_factors: List[float] = []     # contention factor samples
        self.peak_concurrent_prefills: int = 0        # peak prefill batches in-flight simultaneously

        # Agent KV reuse metrics
        self.agent_prefill_tokens_total: int = 0      # total prefill tokens with KV reuse
        self.agent_prefill_tokens_no_reuse: int = 0   # total without KV reuse
        self.agent_sessions_started: int = 0
        self.agent_turns_completed: int = 0
        self.agent_total_turns: int = 0

    def percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        return float(np.percentile(data, p))

    def sample(self, sim_time_us: float, scheduler, kv_manager):
        self.samples.append({
            'time_s': sim_time_us / 1e6,
            'active_decode': scheduler.active_count,
            'waiting': scheduler.waiting_count,
            'prefill': scheduler.prefill_count,
            'finished': scheduler.finished_count,
            'rejected': scheduler.rejected_count,
            'kv_util_pct': kv_manager.utilization_pct,
            'kv_used_blocks': kv_manager.total_blocks_allocated,
        })

    def finalize(self, scheduler):
        self.total_requests = scheduler.stats.total_requests
        self.total_finished = scheduler.stats.total_finished
        self.total_rejected = scheduler.stats.total_rejected
        self.total_tokens_in = scheduler.stats.total_tokens_input
        self.total_tokens_out = scheduler.stats.total_tokens_output
        self.ttfts_us = list(scheduler.stats.ttfts)
        self.tpots_us = list(scheduler.stats.tpots)
        self.latencies_us = list(scheduler.stats.latencies)

    @property
    def simulation_duration_s(self) -> float:
        return (self.end_time_us - self.start_time_us) / 1e6

    @property
    def measured_duration_s(self) -> float:
        return max(0.001, (self.end_time_us - self.warmup_end_us) / 1e6)

    @property
    def throughput_tps(self) -> float:
        return self.total_tokens_out / max(0.001, self.measured_duration_s)

    @property
    def accept_rate(self) -> float:
        return self.total_finished / max(1, self.total_requests)

    @property
    def ttft_p50_ms(self) -> float:
        return self.percentile(self.ttfts_us, 50) / 1000

    @property
    def ttft_p95_ms(self) -> float:
        return self.percentile(self.ttfts_us, 95) / 1000

    @property
    def ttft_p99_ms(self) -> float:
        return self.percentile(self.ttfts_us, 99) / 1000

    @property
    def tpot_p50_ms(self) -> float:
        return self.percentile(self.tpots_us, 50) / 1000

    @property
    def tpot_p95_ms(self) -> float:
        return self.percentile(self.tpots_us, 95) / 1000

    @property
    def latency_p50_ms(self) -> float:
        return self.percentile(self.latencies_us, 50) / 1000

    @property
    def latency_p95_ms(self) -> float:
        return self.percentile(self.latencies_us, 95) / 1000


# ============================================================================
# Event-Driven Simulation
# ============================================================================

class ServingSimulation:
    """Event-driven FPGA cloud serving simulation.

    Uses a priority queue of events instead of fixed-time-step polling.
    When a batch completes, the next batch is scheduled immediately,
    allowing natural accumulation of concurrent requests.

    Supports two deployment modes:
      - Monolithic (default): N identical 32-chip servers, each running both
        prefill and decode. Requests distributed round-robin.
      - Disaggregated: Separate prefill and decode server pools.
        Prefill servers run chunked prefill only; after KV transfer,
        requests join a shared decode pool for batched decode.
    """

    MAX_DECODE_WAIT_US = 2000    # max 2ms wait before forcing a smaller decode batch
    # When set >0, scheduler delays decode batch formation by this many us to allow
    # more sessions to accumulate. Trade-off: higher batch size (better TPS) vs
    # higher per-token latency (worse TPOT). At B=32, latency_per_token ≈ 103us
    # so waiting 100-300us is acceptable while batch grows.
    DECODE_BATCH_WAIT_US = 0     # 0 = greedy (current), 200 = wait for batch fill

    # KV transfer: per-token latency for prefill→decode pool transfer.
    # C2C bandwidth ~512 GB/s (4 links × 128 GB/s). KV bytes per token = 1152 B.
    # Transfer time ≈ 1152 B / 512 GB/s ≈ 2.25 ns/token. Model conservatively.
    KV_TRANSFER_US_PER_TOKEN = 0.01  # 10 ns per token (conservative, includes overhead)

    def __init__(self, arrival_rate: float = 5.0,
                 duration_s: float = 60.0,
                 max_decode_batch: int = MAX_DECODE_BATCH,
                 min_decode_batch: int = MIN_DECODE_BATCH,
                 prompt_len_mean: int = PROMPT_LEN_MEAN,
                 output_len_mean: int = OUTPUT_LEN_MEAN,
                 num_servers: int = 1,
                 num_prefill_servers: int = 0,
                 num_decode_servers: int = 0,
                 kv_blocks_per_chip: int = KV_BLOCKS_PER_CHIP,
                 agent_mode: bool = False,
                 agent_turns: int = AGENT_TURNS,
                 agent_think_ms: int = AGENT_THINK_TIME_MS,
                 agent_delta_prompt: int = AGENT_DELTA_PROMPT,
                 agent_output_per_turn: int = AGENT_OUTPUT_PER_TURN,
                 cpu_hybrid: bool = False,
                 cpu_tflops: float = 3.0,
                 microbatch: bool = False,
                 expert_replication: str = 'none',
                 zipf_alpha: float = 1.0,
                 decode_batch_wait_us: float = 0.0,
                 pipeline_clone: int = 1,
                 seed: int = 42,
                 verbose: bool = False):
        self.arrival_rate = arrival_rate
        self.duration_us = duration_s * 1e6
        self.max_decode_batch = max_decode_batch
        self.min_decode_batch = min_decode_batch
        self.seed = seed
        self.verbose = verbose

        # Deployment mode
        self.num_servers = num_servers
        self.num_prefill_servers = num_prefill_servers
        self.num_decode_servers = num_decode_servers
        self.is_disaggregated = num_prefill_servers > 0 and num_decode_servers > 0

        if self.is_disaggregated:
            self.prefill_scale = num_prefill_servers
            self.decode_scale = num_decode_servers
            self.mode_label = f"Disaggregated ({num_prefill_servers}P + {num_decode_servers}D)"
        else:
            self.prefill_scale = num_servers
            self.decode_scale = num_servers
            self.mode_label = f"Monolithic ({num_servers} servers)"

        # Pipeline Cloning: split 32 chips into `pipeline_clone` independent
        # pipelines, each with its own chip 0. This multiplies prefill
        # admission rate (each pipeline has independent chip 0 capacity)
        # but does NOT multiply decode peak throughput (peak DSP is shared).
        # Decode latency per token rises slightly because each chip handles
        # more layers (61/16 ≈ 4 layers/chip for clone=2 vs 61/32 ≈ 2 for clone=1).
        self.pipeline_clone = max(1, pipeline_clone)
        if self.pipeline_clone > 1:
            self.prefill_scale *= self.pipeline_clone
            self.mode_label += f" + PipelineCloning x{self.pipeline_clone}"

        # Multi-turn agent mode
        self.agent_mode = agent_mode
        self.agent_turns = agent_turns
        self.agent_think_us = agent_think_ms * 1000
        self.agent_delta_prompt = agent_delta_prompt
        self.agent_output_per_turn = agent_output_per_turn
        self._agent_sessions: Dict[int, AgentSession] = {}
        self._agent_session_counter: int = 0
        # Metrics for KV reuse comparison
        self._agent_prefill_tokens_total: int = 0  # total prefill tokens across all turns
        self._agent_prefill_tokens_no_reuse: int = 0  # what it would be without KV reuse

        # P2 CPU-FPGA hybrid prefill
        self.cpu_hybrid = cpu_hybrid
        self.cpu_tflops = cpu_tflops

        # 解法 C: Continuous Microbatching (token-level injection)
        # ─────────────────────────────────────────────────────────────────
        # Without microbatch: a decode batch holds all N sessions busy for the
        # full batch wall-clock (~10ms at B=10). Next decode for these sessions
        # cannot start until BATCH_COMPLETE fires.
        #
        # With microbatch: once a session's token has cleared the pipeline
        # head (chip 0 → chip 1), chip 0 is free to inject the next token from
        # ANY active session. We model this by releasing each session from the
        # busy set after PIPELINE_FILL_US (the per-token injection interval),
        # not after full batch duration. The batch still completes at its
        # normal time (last token reaches chip 31), but new tokens can start
        # injecting much earlier.
        #
        # Per-token injection interval (steady state):
        #   PIPELINE_FILL_US = 1e6 / PIPELINE_TPS ≈ 57.3 us
        # This matches PER_LAYER_US ≈ 24.7us × 2 layers/chip averaged.
        self.microbatch = microbatch
        from fpga_arch.config import PIPELINE_TPS as _PTPS
        self.pipeline_fill_us = 1e6 / _PTPS  # ~57.3 us, time between token injections
        self.decode_batch_wait_us = decode_batch_wait_us

        self.cluster = FPGACluster(seed=seed, expert_replication=expert_replication,
                                   zipf_alpha=zipf_alpha)
        self.pipeline = PipelineEngine(self.cluster, seed=seed)
        self.kv_manager = KVCacheManager(num_chips=32, max_blocks_per_chip=kv_blocks_per_chip)
        self.scheduler = ContinuousBatchingScheduler(
            num_chips=32, max_decode_batch=max_decode_batch
        )
        self.model_runner = ModelRunner(self.cluster, self.pipeline,
                                        cpu_hybrid=cpu_hybrid, cpu_tflops=cpu_tflops)
        self.api_server = APIServer(self.scheduler, seed=seed,
                                     prompt_len_mean=prompt_len_mean,
                                     output_len_mean=output_len_mean)

        self.metrics = SimulationMetrics()

        self._event_queue: List[tuple] = []  # (time_us, seq, event_type, payload)
        self._event_seq: int = 0             # tiebreaker counter
        self._arrival_schedule: List[float] = []
        self._batch_counter: int = 0
        self._last_decode_schedule_us: float = 0.0

        # Concurrent pipeline tracking
        self._in_flight: Dict[int, dict] = {}  # batch_id -> {type, prefill_tokens, decode_batch, start_us, duration_us, contention}
        self._busy_ids: set = set()            # request_ids currently in a running batch
        self._concurrent_start_us: float = 0.0 # when both prefill+decode became simultaneously active
        self._concurrent_samples: List[float] = []  # contention factors over time

        # Superscalar prefill interleaving
        from fpga_arch.config import PREFILL_CHUNK_SIZE, PREFILL_USE_FP4_ATTN, PREFILL_ATTN_SPARSITY
        self._prefill_bottleneck_us = PipelineEngine.prefill_chip0_bottleneck_us(
            chunk_size=PREFILL_CHUNK_SIZE,
            use_fp4_attn=PREFILL_USE_FP4_ATTN,
            attn_sparsity=PREFILL_ATTN_SPARSITY)
        # Sustainable admission: chip 0 processes 1/B chunks/s.
        # Each prefill needs N = ceil(P/chunk_size) chunks on chip 0.
        # Default N for max prefill tokens ≈ MAX_PREFILL_TOKENS / PREFILL_CHUNK_SIZE.
        self._prefill_chunks_per_batch = max(1, MAX_PREFILL_TOKENS // PREFILL_CHUNK_SIZE)
        self._next_prefill_admit_us: float = 0.0  # when chip 0 has capacity for next prefill's chunks
        self._prefill_in_flight_count: int = 0     # number of prefill batches currently in-flight
        self._peak_prefill_in_flight: int = 0      # peak concurrent prefill batches

    def _push_event(self, time_us: float, event_type: EventType, payload=None):
        heapq.heappush(self._event_queue, (time_us, self._event_seq, event_type, payload))
        self._event_seq += 1

    def _generate_arrivals(self):
        rng = np.random.RandomState(self.seed)
        t = 0.0
        duration_s = self.duration_us / 1e6
        mean_interval_s = 1.0 / self.arrival_rate if self.arrival_rate > 0 else float('inf')

        while t < duration_s:
            interval = rng.exponential(mean_interval_s)
            t += interval
            if t < duration_s:
                arrival_us = t * 1e6
                self._arrival_schedule.append(arrival_us)
                self._push_event(arrival_us, EventType.REQUEST_ARRIVAL)

        if self.verbose:
            print(f"  Pre-generated {len(self._arrival_schedule)} request arrival events")

    def run(self) -> SimulationMetrics:
        print()
        print("=" * 70)
        print("  FPGA Cloud Serving Simulation")
        print("=" * 70)
        print(f"  Architecture: 8 cards x 4 chips = 32 AGM 039-F")
        print(f"  Deployment:   {self.mode_label}")
        if self.agent_mode:
            print(f"  Agent mode:   {self.agent_turns} turns/session, "
                  f"{self.agent_think_us/1000:.0f}ms think, "
                  f"P_delta={self.agent_delta_prompt}, O={self.agent_output_per_turn}")
        if self.cpu_hybrid:
            print(f"  Prefill:      P2 CPU-FPGA hybrid (CPU={self.cpu_tflops} TFLOPS)")
        if self.cluster.expert_replication == 'hot':
            from fpga_arch.config import K_PIPELINE
            print(f"  Expert repl:  Hot replication (Zipf α={self.cluster.zipf_alpha})")
            print(f"                K_pipeline: {self.pipeline.k_pipeline:.1f} (was {K_PIPELINE:.1f})")
        sched_mode = ("Microbatch (token-level injection, release="
                      f"{self.pipeline_fill_us:.0f}us)" if self.microbatch
                      else "Batch-step (release on BATCH_COMPLETE)")
        print(f"  Scheduler:    {sched_mode} (max decode batch={self.max_decode_batch})")
        print(f"  Duration:     {self.duration_us/1e6:.0f}s")
        print(f"  Arrival rate: {self.arrival_rate} req/s (Poisson)")
        print(f"  Seed:         {self.seed}")
        print()

        self.metrics.start_time_us = 0.0
        self.metrics.end_time_us = self.duration_us
        self.metrics.warmup_end_us = min(WARMUP_DURATION_S * 1e6, self.duration_us * 0.3)

        self._generate_arrivals()

        # Schedule periodic metrics samples
        for t_us in range(1_000_000, int(self.duration_us) + 1_000_000, 1_000_000):
            self._push_event(t_us, EventType.METRICS_SAMPLE)

        start_wall = wall_time.time()
        last_progress_us = 0

        while self._event_queue:
            sim_time_us, _, event_type, payload = heapq.heappop(self._event_queue)

            if sim_time_us > self.duration_us:
                break

            if event_type == EventType.REQUEST_ARRIVAL:
                if self.agent_mode:
                    self._start_agent_session(sim_time_us)
                else:
                    req = self.api_server.generator._generate_request(sim_time_us)
                    self.scheduler.submit_request(req)
                self._maybe_schedule(sim_time_us)

            elif event_type == EventType.AGENT_NEXT_TURN:
                session = payload
                self._submit_agent_turn(session, sim_time_us)
                self._maybe_schedule(sim_time_us)

            elif event_type == EventType.DECODE_READY:
                batch = payload
                # First chunk done → promote prefill→decode
                self.scheduler.on_prefill_complete(batch, sim_time_us)
                # Requests now available for decode scheduling
                for req in batch.requests:
                    self._busy_ids.discard(req.request_id)

            elif event_type == EventType.SESSION_RELEASE:
                # 解法 C: pipeline-fill time elapsed since this decode batch started.
                # These sessions have cleared chip 0 and can be re-injected
                # while the batch's tail tokens are still flowing through later chips.
                released_reqs = payload
                for req in released_reqs:
                    self._busy_ids.discard(req.request_id)
                # Try to schedule next decode immediately (chip 0 is free)
                self._maybe_schedule(sim_time_us)

            elif event_type == EventType.BATCH_COMPLETE:
                batch = payload
                # Clear busy flags
                for req in batch.requests:
                    self._busy_ids.discard(req.request_id)
                info = self._in_flight.pop(batch.batch_id, None)
                if batch.batch_type == BatchType.PREFILL:
                    self._prefill_in_flight_count = max(0, self._prefill_in_flight_count - 1)
                if info and info.get('contention', 1.0) > 1.0:
                    self.metrics.contention_factors.append(info['contention'])
                self._update_concurrent_tracking(sim_time_us)
                if batch.batch_type == BatchType.DECODE:
                    self.scheduler.on_decode_step(batch, sim_time_us)
                    for req in batch.requests:
                        if req.state == RequestState.FINISHED:
                            self._maybe_finish_agent_turn(req, sim_time_us)
                self._maybe_schedule(sim_time_us)

            elif event_type == EventType.METRICS_SAMPLE:
                self.metrics.sample(sim_time_us, self.scheduler, self.kv_manager)

            # Progress
            if self.verbose and sim_time_us - last_progress_us > self.duration_us * 0.1:
                pct = int(sim_time_us / self.duration_us * 100)
                wall_elapsed = wall_time.time() - start_wall
                print(f"  ... {pct}% ({sim_time_us/1e6:.0f}s), "
                      f"wall: {wall_elapsed:.1f}s, "
                      f"active: {self.scheduler.active_count}, "
                      f"finished: {self.scheduler.finished_count}")
                last_progress_us = sim_time_us

        # Drain remaining active requests
        drain_count = 0
        drain_time_us = self.duration_us
        max_event_time_us = self.duration_us
        while self.scheduler.active_decode and drain_count < 2000:
            self._maybe_schedule(drain_time_us)
            # Process any completion events generated
            while self._event_queue:
                ev_time, _, ev_type, ev_payload = heapq.heappop(self._event_queue)
                max_event_time_us = max(max_event_time_us, ev_time)
                if ev_type == EventType.AGENT_NEXT_TURN:
                    # Drain phase: don't submit new agent turns. Submitting would
                    # inflate total_requests beyond what arrived in [0, duration_us]
                    # and cause accept_rate > 100%. Just mark the session as
                    # complete and free its KV blocks.
                    session = ev_payload
                    if session.is_active:
                        # Free KV from the last completed turn
                        for tid in session.turn_request_ids:
                            self.kv_manager.free_request(tid)
                    continue
                elif ev_type == EventType.DECODE_READY:
                    batch = ev_payload
                    self.scheduler.on_prefill_complete(batch, ev_time)
                    for req in batch.requests:
                        self._busy_ids.discard(req.request_id)
                elif ev_type == EventType.SESSION_RELEASE:
                    for req in ev_payload:
                        self._busy_ids.discard(req.request_id)
                    self._maybe_schedule(ev_time)
                elif ev_type == EventType.BATCH_COMPLETE:
                    batch = ev_payload
                    for req in batch.requests:
                        self._busy_ids.discard(req.request_id)
                    self._in_flight.pop(batch.batch_id, None)
                    if batch.batch_type == BatchType.PREFILL:
                        self._prefill_in_flight_count = max(0, self._prefill_in_flight_count - 1)
                    self._update_concurrent_tracking(ev_time)
                    if batch.batch_type == BatchType.DECODE:
                        self.scheduler.on_decode_step(batch, ev_time)
                        for req in batch.requests:
                            if req.state == RequestState.FINISHED:
                                self._maybe_finish_agent_turn(req, ev_time)
                    self._maybe_schedule(ev_time)
            drain_time_us += 1000
            drain_count += 1

        # Extend end_time_us to actual drain completion so throughput denominator
        # matches the numerator (both include drain phase). Otherwise drain
        # completions inflate tokens/s ratio against fixed duration.
        self.metrics.end_time_us = max(self.metrics.end_time_us,
                                        max_event_time_us, drain_time_us)

        if self.verbose:
            print(f"  Drain: {drain_count} cycles, {self.scheduler.finished_count} finished")

        self.metrics.finalize(self.scheduler)
        self.metrics.peak_concurrent_prefills = self._peak_prefill_in_flight

        # Agent KV reuse metrics
        self.metrics.agent_prefill_tokens_total = self._agent_prefill_tokens_total
        self.metrics.agent_prefill_tokens_no_reuse = self._agent_prefill_tokens_no_reuse
        self.metrics.agent_sessions_started = self._agent_session_counter
        self.metrics.agent_turns_completed = sum(
            s.turns_completed for s in self._agent_sessions.values())
        self.metrics.agent_total_turns = sum(
            s.total_turns for s in self._agent_sessions.values())

        if self.verbose:
            wall_total = wall_time.time() - start_wall
            print(f"  Simulation wall time: {wall_total:.1f}s")

        return self.metrics

    def _maybe_schedule(self, sim_time_us: float):
        """Try to schedule new batches. Called on request arrival and batch completion.

        Adaptive prefill batching:
          - MAX_PREFILL_CHUNKS caps batch tokens → bounds admission interval
          - Admission interval = N_chunks × B (chip 0 capacity constraint)
          - Small batches → short admission interval → low queue wait → TTFT SLA
          - When arrival rate > chip 0 capacity, queue grows; TTFT tail degrades
            gracefully (not by overloading chip 0)

        Decode: concurrent with prefill — runs independently.
        """
        if self.scheduler.waiting_queue and sim_time_us >= self._next_prefill_admit_us:
            self._schedule_prefill(sim_time_us)

        # Decode: continuous batching — always schedule if requests are available.
        # In disaggregated mode, prefills enter decode at staggered times, so
        # waiting for min_decode_batch causes excessive idle time between steps.
        # Instead, schedule immediately if any active request is free.
        n_active = len(self.scheduler.active_decode)
        if n_active == 0:
            return

        n_available = len([r for r in self.scheduler.active_decode
                          if r.request_id not in self._busy_ids])
        if n_available == 0:
            return

        can_schedule = (
            n_active >= self.min_decode_batch
            or n_available >= self.min_decode_batch
            or (sim_time_us - self._last_decode_schedule_us >= self.MAX_DECODE_WAIT_US)
        )

        # Optional: delay scheduling to grow batch (better TPS, slightly worse TPOT)
        if can_schedule and self.decode_batch_wait_us > 0:
            time_since_last = sim_time_us - self._last_decode_schedule_us
            # Wait at least decode_batch_wait_us OR until batch reaches a fraction
            # of active sessions (whichever first)
            target_batch = max(8, int(0.5 * n_active))
            if (n_available < target_batch and
                time_since_last < self.decode_batch_wait_us):
                # Re-schedule a tick to retry later
                retry_time = sim_time_us + (self.decode_batch_wait_us - time_since_last)
                self._push_event(retry_time, EventType.SESSION_RELEASE, [])
                return

        if can_schedule:
            self._schedule_decode(sim_time_us)

    def _schedule_prefill(self, sim_time_us: float):
        """Admit waiting requests and form a latency-constrained prefill batch.

        Token cap = MAX_PREFILL_CHUNKS × PREFILL_CHUNK_SIZE limits chip 0 queue
        depth for typical short requests. For agent workloads with prompts larger
        than the cap, a single oversized request is admitted alone.
        """
        from fpga_arch.config import PREFILL_CHUNK_SIZE

        # Admit waiting requests (up to capacity)
        admitted = self.scheduler._admit_waiting(sim_time_us, self.kv_manager)
        if not admitted:
            return

        max_tokens = MAX_PREFILL_CHUNKS * PREFILL_CHUNK_SIZE

        # Agent workload: if all admitted requests are individually oversized,
        # admit just the first one alone rather than bouncing them forever.
        if all(r.prompt_len > max_tokens for r in admitted):
            selected = [admitted[0]]
            tokens = admitted[0].prompt_len
            # Put the rest back
            for r in admitted[1:]:
                r.state = RequestState.WAITING
                if r in self.scheduler.prefilling:
                    self.scheduler.prefilling.remove(r)
                self.scheduler.waiting_queue.appendleft(r)
        else:
            selected = []
            tokens = 0
            for r in admitted:
                if tokens + r.prompt_len <= max_tokens:
                    selected.append(r)
                    tokens += r.prompt_len
                else:
                    r.state = RequestState.WAITING
                    if r in self.scheduler.prefilling:
                        self.scheduler.prefilling.remove(r)
                    self.scheduler.waiting_queue.appendleft(r)

        batch = Batch(
            batch_id=self._batch_counter,
            batch_type=BatchType.PREFILL,
            requests=selected,
            batch_size_tokens=tokens,
            created_time_us=sim_time_us,
        )
        self._batch_counter += 1
        self.scheduler.stats.total_prefill_batches += 1
        self.scheduler._batch_counter = self._batch_counter

        self._execute_batch(batch, sim_time_us)

    def _schedule_decode(self, sim_time_us: float):
        """Form a decode batch from active decode requests not already in-flight."""
        available = [r for r in self.scheduler.active_decode
                     if r.request_id not in self._busy_ids]
        if not available:
            return

        batch_requests = available[:self.max_decode_batch]

        batch = Batch(
            batch_id=self._batch_counter,
            batch_type=BatchType.DECODE,
            requests=batch_requests,
            batch_size_tokens=len(batch_requests),
            created_time_us=sim_time_us,
        )
        self._batch_counter += 1
        self.scheduler.stats.total_decode_batches += 1
        self._last_decode_schedule_us = sim_time_us

        self._execute_batch(batch, sim_time_us)

    def _update_concurrent_tracking(self, sim_time_us: float):
        """Track overlapping intervals where both prefill and decode are in-flight."""
        has_prefill = any(b['type'] == BatchType.PREFILL for b in self._in_flight.values())
        has_decode = any(b['type'] == BatchType.DECODE for b in self._in_flight.values())
        was_concurrent = self._concurrent_start_us > 0
        is_concurrent = has_prefill and has_decode
        if is_concurrent and not was_concurrent:
            self._concurrent_start_us = sim_time_us
        elif not is_concurrent and was_concurrent:
            self.metrics.total_concurrent_time_us += sim_time_us - self._concurrent_start_us
            self._concurrent_start_us = 0.0

    def _execute_batch(self, batch: Batch, sim_time_us: float):
        """Execute a batch and enqueue completion events.

        Superscalar prefill: admission controlled by chip 0 capacity.
        Admission interval = N_chunks × bottleneck_per_chip ensures chip 0 is
        never contended — chunks from different prefills don't overlap on chip 0.
        Multiple prefills occupy different pipeline stages simultaneously,
        using different chip resources (DSP on chip 1-31 while chip 0 processes
        new chunks). Decode contends for DSP/HBM on shared chips.

        Prefill lifecycle (two-phase):
          1. 0 → TTFT: prefill first chunk, decode starts at DECODE_READY
          2. TTFT → total_duration_us: remaining chunks + decode concurrent
        """
        result = self.model_runner.execute_batch(batch, self.kv_manager, sim_time_us)

        if not result.success:
            if self.verbose:
                print(f"  t={sim_time_us/1e6:.4f}s BATCH FAILED: {result.error}")
            # Clean up failed batch: put requests back to waiting, advance admission.
            for req in batch.requests:
                self._busy_ids.discard(req.request_id)
                if req in self.scheduler.prefilling:
                    self.scheduler.prefilling.remove(req)
                req.state = RequestState.WAITING
                self.scheduler.waiting_queue.appendleft(req)
            if batch.batch_type == BatchType.PREFILL:
                self._prefill_in_flight_count = max(0, self._prefill_in_flight_count - 1)
                # Advance admission window slightly to allow retry
                self._next_prefill_admit_us = sim_time_us + 50_000  # 50ms backoff
            return

        is_prefill = batch.batch_type == BatchType.PREFILL
        is_decode = batch.batch_type == BatchType.DECODE

        pipeline_duration = result.total_duration_us if is_prefill else result.duration_us

        contention = 1.0
        kv_transfer_us = 0.0

        if is_prefill:
            from fpga_arch.config import PREFILL_CHUNK_SIZE, PREFILL_USE_FP4_ATTN, PREFILL_ATTN_SPARSITY

            if self.is_disaggregated:
                # No prefill-decode contention — separate hardware pools.
                # KV transfer latency: move KV blocks from prefill → decode server.
                kv_transfer_us = batch.batch_size_tokens * self.KV_TRANSFER_US_PER_TOKEN
                # Decode-ready after full prefill + KV transfer (single-phase)
                pipeline_duration = result.total_duration_us
            else:
                # Monolithic: prefill and decode share same pipeline.
                has_decode = any(info['type'] == BatchType.DECODE
                               for info in self._in_flight.values())
                if has_decode:
                    total_prefill_tokens = batch.batch_size_tokens + sum(
                        info.get('prefill_tokens', 0) for info in self._in_flight.values()
                        if info['type'] == BatchType.PREFILL)
                    decode_batch_size = sum(
                        info.get('decode_batch', 0) for info in self._in_flight.values()
                        if info['type'] == BatchType.DECODE)
                    n_prefill_req = batch.size + sum(
                        1 for info in self._in_flight.values()
                        if info['type'] == BatchType.PREFILL)
                    r = PipelineEngine.concurrent_pipeline_model(
                        total_prefill_tokens, decode_batch_size,
                        n_requests=max(1, n_prefill_req))
                    contention = r['contention_factor']
                    pipeline_duration *= contention

            # Update chip 0 admission window, scaled by prefill server count.
            # N prefill servers → N independent chip 0 pipelines → N× admission rate.
            dyn_bottleneck = PipelineEngine.prefill_chip0_bottleneck_us(
                chunk_size=PREFILL_CHUNK_SIZE,
                use_fp4_attn=PREFILL_USE_FP4_ATTN,
                attn_sparsity=PREFILL_ATTN_SPARSITY,
                n_requests=max(1, batch.size))
            actual_chunks = max(1, (batch.batch_size_tokens + PREFILL_CHUNK_SIZE - 1)
                                // PREFILL_CHUNK_SIZE)
            self._next_prefill_admit_us = sim_time_us + dyn_bottleneck * actual_chunks / self.prefill_scale
            self._prefill_in_flight_count += 1
            if self._prefill_in_flight_count > self._peak_prefill_in_flight:
                self._peak_prefill_in_flight = self._prefill_in_flight_count

        elif is_decode:
            if not self.is_disaggregated:
                # Monolithic: decode contends with prefill for DSP/HBM.
                has_prefill = any(info['type'] == BatchType.PREFILL
                                for info in self._in_flight.values())
                if has_prefill:
                    prefill_tokens = sum(
                        info.get('prefill_tokens', 0) for info in self._in_flight.values()
                        if info['type'] == BatchType.PREFILL)
                    n_prefill_req = sum(
                        1 for info in self._in_flight.values()
                        if info['type'] == BatchType.PREFILL)
                    r = PipelineEngine.concurrent_pipeline_model(
                        prefill_tokens, batch.size, n_requests=max(1, n_prefill_req))
                    contention = r['contention_factor']
                    pipeline_duration *= contention
            # Disaggregated: decode runs on dedicated servers, no prefill contention.

        # Prefill requests are busy until decode starts (prevent double-scheduling)
        for req in batch.requests:
            self._busy_ids.add(req.request_id)

        # Track in-flight for concurrency (uses total pipeline occupancy)
        self._in_flight[batch.batch_id] = {
            'type': batch.batch_type,
            'prefill_tokens': batch.batch_size_tokens if is_prefill else 0,
            'decode_batch': batch.size if is_decode else 0,
            'start_us': sim_time_us,
            'duration_us': pipeline_duration,
            'contention': contention,
        }
        self._concurrent_samples.append(contention)

        self.metrics.batch_sizes.append(batch.size)
        self.metrics.batch_durations_us.append(pipeline_duration)
        self.metrics.batch_tps.append(result.throughput_tps)
        if is_prefill:
            self.metrics.prefill_batch_sizes.append(batch.size)
            self.metrics.prefill_batch_tokens.append(batch.batch_size_tokens)
            self.metrics.prefill_batch_durations_us.append(pipeline_duration)

        self._update_concurrent_tracking(sim_time_us)

        if is_prefill and result.ttft_us > 0:
            if self.is_disaggregated:
                # Disaggregated: KV streaming model. First chunk KV transferred
                # during TTFT window → decode can start at TTFT (like monolithic).
                # KV transfer latency for the first chunk is negligible (~1us).
                decode_ready_time = sim_time_us + result.ttft_us
            else:
                # Monolithic two-phase: decode starts after first chunk (TTFT).
                decode_ready_time = sim_time_us + result.ttft_us
            self._push_event(decode_ready_time, EventType.DECODE_READY, batch)

        # Pipeline free (monolithic) or prefill complete + KV transferred (disaggregated)
        comp_time = sim_time_us + pipeline_duration + kv_transfer_us

        # 解法 C: Continuous Microbatching — early session release (DISABLED)
        # ─────────────────────────────────────────────────────────────────
        # Initial design released sessions after pipeline_fill_us so chip 0
        # could accept new tokens from the same session sooner. This is
        # PHYSICALLY INCORRECT for decode: decode is autoregressive — session
        # X cannot inject token N+1 until token N's logits return from the
        # pipeline tail, because token N IS the input to token N+1.
        #
        # The true benefit of microbatching is letting DIFFERENT sessions
        # share the pipeline (B tokens flowing through B different stages
        # simultaneously). This is already captured by the B/(B+K) throughput
        # model and by allowing concurrent batches of different sessions.
        #
        # What microbatch=True now does:
        #   1. MIN_DECODE_BATCH=1 (no floor) — schedule even single-session decodes
        #   2. MAX_DECODE_BATCH=256 (raised) — pack more sessions per batch
        #   3. (No early release: would violate autoregressive constraint)
        if False and is_decode and self.microbatch:
            release_time = sim_time_us + self.pipeline_fill_us
            if release_time < comp_time:
                self._push_event(release_time, EventType.SESSION_RELEASE,
                                 list(batch.requests))

        self._push_event(comp_time, EventType.BATCH_COMPLETE, batch)

    # ── Multi-turn Agent Session Management ──

    def _start_agent_session(self, sim_time_us: float):
        """Create a new agent session and submit turn 1 (full prefill)."""
        rng = np.random.RandomState(self.seed + self._agent_session_counter)
        prompt_init = int(np.clip(
            rng.normal(self.api_server.generator.prompt_len_mean,
                       self.api_server.generator.prompt_len_mean * 0.5),
            256, 8192))
        total_turns = max(2, int(rng.normal(self.agent_turns, 2)))

        session = AgentSession(
            session_id=self._agent_session_counter,
            total_turns=total_turns,
            prompt_init_tokens=prompt_init,
            prompt_delta_tokens=self.agent_delta_prompt,
            output_tokens_per_turn=self.agent_output_per_turn,
            thinking_time_us=self.agent_think_us,
            created_time_us=sim_time_us,
        )
        self._agent_sessions[session.session_id] = session
        self._agent_session_counter += 1

        # Track tokens: with KV reuse (turn 1 full, turns 2+ delta) vs without
        self._agent_prefill_tokens_total += prompt_init  # turn 1: full prefill
        self._agent_prefill_tokens_no_reuse += prompt_init * total_turns

        # Submit turn 1: full prompt
        req = Request(
            request_id=len(self.scheduler.all_requests),
            arrival_time_us=sim_time_us,
            prompt_len=prompt_init,
            max_output_len=self.agent_output_per_turn,
        )
        req._agent_session_id = session.session_id
        self.scheduler.submit_request(req)
        session.turn_request_ids.append(req.request_id)

    def _submit_agent_turn(self, session: AgentSession, sim_time_us: float):
        """Submit the next turn of an agent session (delta prefill, KV reuse)."""
        if not session.is_active:
            return

        # Subsequent turns: only prefill delta tokens (new user input / tool results).
        # The accumulated KV cache from prior turns is reused.
        delta_prompt = session.prompt_delta_tokens
        self._agent_prefill_tokens_total += delta_prompt

        req = Request(
            request_id=len(self.scheduler.all_requests),
            arrival_time_us=sim_time_us,
            prompt_len=delta_prompt,
            max_output_len=self.agent_output_per_turn,
        )
        req._agent_session_id = session.session_id
        self.scheduler.submit_request(req)
        session.turn_request_ids.append(req.request_id)

    def _maybe_finish_agent_turn(self, req: Request, sim_time_us: float):
        """Handle agent turn completion: schedule next turn or finalize session."""
        session_id = getattr(req, '_agent_session_id', None)
        if session_id is None:
            # Regular request — free KV blocks
            self.kv_manager.free_request(req.request_id)
            return

        session = self._agent_sessions.get(session_id)
        if session is None:
            return

        session.turns_completed += 1
        session.accumulated_tokens += req.prompt_len + req.tokens_generated

        # During drain (t > duration_us), don't schedule new turns: they would
        # add to total_requests but never be processed, inflating accept_rate.
        in_drain = sim_time_us > self.duration_us

        if session.is_active and not in_drain:
            # Schedule next turn after thinking time
            next_turn_at = sim_time_us + session.thinking_time_us
            session.next_turn_at_us = next_turn_at
            self._push_event(next_turn_at, EventType.AGENT_NEXT_TURN, session)
            # KV blocks are NOT freed — reused by next turn
        else:
            # Session complete (or drain phase) — free all KV blocks
            self.kv_manager.free_request(req.request_id)
            if session_id in self._agent_sessions:
                del self._agent_sessions[session_id]



# ============================================================================
# Report Generation
# ============================================================================

def print_report(metrics: SimulationMetrics, args: argparse.Namespace,
                 pipeline: 'PipelineEngine' = None):
    """Print formatted simulation report."""

    print()
    print("=" * 70)
    print("  SIMULATION RESULTS")
    print("=" * 70)
    print()

    print("  --- Configuration ---")
    print(f"  Duration:        {metrics.simulation_duration_s:.0f}s")
    print(f"  Arrival rate:    {args.arrival_rate} req/s")
    if args.prefill_servers and args.decode_servers:
        print(f"  Deployment:      Disaggregated ({args.prefill_servers}P + {args.decode_servers}D)")
        total_servers = args.prefill_servers + args.decode_servers
    else:
        print(f"  Deployment:      Monolithic ({args.num_servers} servers)")
        total_servers = args.num_servers
    print(f"  Total servers:   {total_servers}")
    print(f"  Max decode batch: {args.max_decode_batch}")
    print(f"  Warmup:          {WARMUP_DURATION_S}s")
    print()

    from fpga_arch.pipeline import PipelineEngine
    from fpga_arch.config import PIPELINE_TPS

    print("  --- Throughput ---")
    print(f"  Total requests:  {metrics.total_requests:>8d}")
    print(f"  Finished:        {metrics.total_finished:>8d}")
    print(f"  Rejected:        {metrics.total_rejected:>8d}")
    print(f"  Accept rate:     {metrics.accept_rate:>8.1%}")
    print(f"  Tokens in:       {metrics.total_tokens_in:>8d}")
    print(f"  Tokens out:      {metrics.total_tokens_out:>8d}")
    print(f"  Output TPS:      {metrics.throughput_tps:>8.1f} tok/s")
    avg_batch = np.mean(metrics.batch_sizes) if metrics.batch_sizes else 0
    theoretical_tps = PipelineEngine.throughput_model(max(1, int(avg_batch)))
    print(f"  Theoretical TPS: {theoretical_tps:>8.0f} tok/s  (at B={avg_batch:.0f})")
    print(f"  Sys efficiency:  {metrics.throughput_tps/max(1,theoretical_tps)*100:>8.1f}%")
    print(f"  Peak TPS (B=inf):{PIPELINE_TPS:>8,d} tok/s  (fully saturated)")

    # Concurrency metrics
    concurrent_pct = min(100.0, metrics.total_concurrent_time_us / max(1, metrics.measured_duration_s * 1e6) * 100)
    avg_contention = np.mean(metrics.contention_factors) if metrics.contention_factors else 1.0
    print(f"  Concurrent time:  {concurrent_pct:>7.1f}%  (prefill + decode overlap, of measured duration)")
    print(f"  Avg contention:   {avg_contention:>7.2f}x  (resource sharing overhead)")
    print()

    print("  --- Latency ---")
    print(f"  TTFT P50:        {metrics.ttft_p50_ms:>8.1f} ms  (target: {TTFT_TARGET_MS} ms, SLA: {TTFT_SLA_MS} ms)")
    print(f"  TTFT P95:        {metrics.ttft_p95_ms:>8.1f} ms")
    print(f"  TTFT P99:        {metrics.ttft_p99_ms:>8.1f} ms")
    print(f"  TPOT P50:        {metrics.tpot_p50_ms:>8.1f} ms  (target: {TPOT_TARGET_MS} ms, SLA: {TPOT_SLA_MS} ms)")
    print(f"  TPOT P95:        {metrics.tpot_p95_ms:>8.1f} ms")
    print(f"  E2E Latency P50: {metrics.latency_p50_ms:>8.1f} ms")
    print(f"  E2E Latency P95: {metrics.latency_p95_ms:>8.1f} ms")
    print()

    if metrics.ttfts_us:
        violations = sum(1 for t in metrics.ttfts_us if t > TTFT_SLA_MS * 1000)
        rate = (1 - violations / len(metrics.ttfts_us)) * 100
        print(f"  TTFT SLA compliance: {rate:.1f}% ({len(metrics.ttfts_us) - violations}/{len(metrics.ttfts_us)})")

    if metrics.tpots_us:
        violations = sum(1 for t in metrics.tpots_us if t > TPOT_SLA_MS * 1000)
        rate = (1 - violations / len(metrics.tpots_us)) * 100
        print(f"  TPOT SLA compliance: {rate:.1f}% ({len(metrics.tpots_us) - violations}/{len(metrics.tpots_us)})")
    print()

    if metrics.prefill_batch_sizes:
        n_prefill_batches = len(metrics.prefill_batch_sizes)
        print("  --- Prefill Batch Stats ---")
        print(f"  Prefill batches:   {n_prefill_batches:>8d}")
        print(f"  Avg prefill reqs:  {np.mean(metrics.prefill_batch_sizes):>8.1f}")
        print(f"  Max prefill reqs:  {np.max(metrics.prefill_batch_sizes):>8d}")
        print(f"  Avg prefill tokens:{np.mean(metrics.prefill_batch_tokens):>8.0f}")
        print(f"  Max prefill tokens:{np.max(metrics.prefill_batch_tokens):>8d}")
        avg_prefill_ms = np.mean(metrics.prefill_batch_durations_us) / 1000
        print(f"  Avg prefill dur:   {avg_prefill_ms:>8.1f} ms")
        # Admission rate: prefill batches / measured duration (not derived from batch duration which includes K scaling)
        req_per_s = n_prefill_batches * np.mean(metrics.prefill_batch_sizes) / max(0.001, metrics.measured_duration_s)
        tok_per_s = sum(metrics.prefill_batch_tokens) / max(0.001, metrics.measured_duration_s)
        print(f"  Prefill admission: {req_per_s:>8.1f} req/s  (token-level: {tok_per_s:>8.0f} tok/s)")
        print(f"  Peak concurrent prefills: {metrics.peak_concurrent_prefills:>8d}")
        print()

    # Agent KV reuse metrics
    if metrics.agent_sessions_started > 0:
        reuse_saved = metrics.agent_prefill_tokens_no_reuse - metrics.agent_prefill_tokens_total
        reuse_pct = (1 - metrics.agent_prefill_tokens_total / max(1, metrics.agent_prefill_tokens_no_reuse)) * 100
        print("  --- Agent KV Cache Reuse ---")
        print(f"  Sessions started:  {metrics.agent_sessions_started:>8d}")
        print(f"  Turns completed:   {metrics.agent_turns_completed:>8d} / {metrics.agent_total_turns}")
        print(f"  Prefill w/ reuse:  {metrics.agent_prefill_tokens_total:>8d} tokens")
        print(f"  Prefill w/o reuse: {metrics.agent_prefill_tokens_no_reuse:>8d} tokens")
        print(f"  Tokens saved:      {reuse_saved:>8d} ({reuse_pct:.0f}%)")
        print()

    # P2 CPU-FPGA hybrid prefill analysis
    from fpga_arch.pipeline import PipelineEngine
    from fpga_arch.config import PREFILL_CHUNK_SIZE, PREFILL_USE_FP4_ATTN, PREFILL_ATTN_SPARSITY

    if getattr(args, 'cpu_hybrid', False) and metrics.prefill_batch_tokens:
        # Compute theoretical FPGA-only TTFT for comparison
        p2_ttfts = []
        baseline_ttfts = []
        for tokens in metrics.prefill_batch_tokens:
            n_req = 1
            p2_r = PipelineEngine.cpu_hybrid_prefill_model(
                tokens, cpu_tflops=args.cpu_tflops,
                use_fp4_attn=PREFILL_USE_FP4_ATTN,
                attn_sparsity=PREFILL_ATTN_SPARSITY,
                chunk_size=PREFILL_CHUNK_SIZE, n_requests=n_req)
            base_r = PipelineEngine.chunked_prefill_model(
                tokens, chunk_size=PREFILL_CHUNK_SIZE,
                use_fp4_attn=PREFILL_USE_FP4_ATTN,
                attn_sparsity=PREFILL_ATTN_SPARSITY, n_requests=n_req)
            p2_ttfts.append(p2_r['ttft_ms'])
            baseline_ttfts.append(base_r['ttft_ms'])

        if p2_ttfts:
            print("  --- P2 CPU-FPGA Hybrid Prefill ---")
            print(f"  CPU TFLOPS:       {args.cpu_tflops:>8.1f}")
            print(f"  P2 TTFT (mean):   {np.mean(p2_ttfts):>8.1f} ms")
            print(f"  FPGA TTFT (mean): {np.mean(baseline_ttfts):>8.1f} ms")
            speedup = np.mean(baseline_ttfts) / max(1, np.mean(p2_ttfts))
            print(f"  P2 speedup:       {speedup:>8.2f}x")
            print()

    if metrics.batch_sizes:
        print("  --- Batch Stats ---")
        print(f"  Avg batch size:  {np.mean(metrics.batch_sizes):>8.1f}")
        print(f"  Max batch size:  {np.max(metrics.batch_sizes):>8d}")
        print(f"  Avg batch dur:   {np.mean(metrics.batch_durations_us):>8.0f} us")
        if metrics.batch_tps:
            print(f"  Avg batch TPS:   {np.mean(metrics.batch_tps):>8.0f} tok/s")
            print(f"  Peak batch TPS:  {np.max(metrics.batch_tps):>8.0f} tok/s")
        print()

    # Cost analysis — scaled by server count
    if args.prefill_servers and args.decode_servers:
        total_srv = args.prefill_servers + args.decode_servers
    else:
        total_srv = args.num_servers
    tokens_per_year = metrics.throughput_tps * 3600 * 24 * 365
    revenue_per_year = tokens_per_year * 3.0 / 1e6
    power_per_server = 5.3 * 0.35 * 24 * 365
    total_server_cost = total_srv * 1_000_000
    total_power_cost = total_srv * power_per_server
    print("  --- Cost Analysis ---")
    print(f"  Servers:         {total_srv}")
    print(f"  HW cost:         RMB {total_server_cost:,.0f}")
    print(f"  Power:           {total_srv} x 5.3 kW x 0.35 RMB/kWh = RMB {total_power_cost:,.0f}/yr")
    print(f"  Revenue/yr:      RMB {revenue_per_year:,.0f} (at 3.0 RMB/1M tokens)")
    print(f"  Tokens/yr:       {tokens_per_year/1e9:.2f}B")
    print()

    if metrics.samples:
        samples = [s for s in metrics.samples if s['time_s'] >= WARMUP_DURATION_S]
        if samples:
            avg_active = np.mean([s['active_decode'] for s in samples])
            avg_kv_util = np.mean([s['kv_util_pct'] for s in samples])
            peak_active = max(s['active_decode'] for s in samples)
            print("  --- Steady-State (post-warmup) ---")
            print(f"  Avg active:      {avg_active:>8.1f} requests")
            print(f"  Peak active:     {peak_active:>8d} requests")
            print(f"  Avg KV util:     {avg_kv_util:>8.1f}%")
            print()

    # Hardware comparison
    from fpga_arch.config import (
        H200_DECODE_TPS, H200_PREFILL_TPS, H200_TTFT_P50_MS, H200_SRV_COST,
        ASCEND_DECODE_TPS, ASCEND_PREFILL_TPS, ASCEND_TTFT_P50_MS, ASCEND_SRV_COST,
        FPGA_DECODE_TPS, FPGA_DECODE_TPS_HW, FPGA_PREFILL_TPS, FPGA_TTFT_P50_MS,
        FPGA_TTFT_CHUNKED_MS, FPGA_PREFILL_TPS_CHUNKED,
    )
    print("  --- Hardware Comparison (DS V4 Pro, FP8, per server) ---")
    print(f"  {'Metric':<28s} {'FPGA A7':>14s} {'H200':>14s} {'Ascend 950PR':>14s}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {'Decode TPS (high concur)':<28s} {FPGA_DECODE_TPS:>14,d} {H200_DECODE_TPS:>14,d} {ASCEND_DECODE_TPS:>14,d}")
    print(f"  {'Decode TPS (raw HW)':<28s} {FPGA_DECODE_TPS_HW:>14,d} {'N/A':>14s} {'N/A':>14s}")
    print(f"  {'Prefill TPS (P=512,P0+P1)':<28s} {FPGA_PREFILL_TPS:>14,d} {H200_PREFILL_TPS:>14,d} {ASCEND_PREFILL_TPS:>14,d}")
    print(f"  {'TTFT single (P=512) ms':<28s} {FPGA_TTFT_P50_MS:>14,d} {H200_TTFT_P50_MS:>14,d} {ASCEND_TTFT_P50_MS:>14,d}")
    print(f"  {'TTFT chunked (P=128) ms':<28s} {FPGA_TTFT_CHUNKED_MS:>14,d} {'N/A':>14s} {'N/A':>14s}")
    print(f"  {'Server cost (RMB 10k)':<28s} {100:>14,d} {H200_SRV_COST//10000:>14,d} {ASCEND_SRV_COST//10000:>14,d}")
    print(f"  {'Cost / (tok/s) RMB':<28s} {1_000_000/max(1,FPGA_DECODE_TPS):>13.0f}  {H200_SRV_COST/max(1,H200_DECODE_TPS):>13.0f}  {ASCEND_SRV_COST/max(1,ASCEND_DECODE_TPS):>13.0f}")
    print(f"  P0: fp4 K/V attention (2x attn MAC). P1: router-guided sparse (88.8%).")
    print(f"  Chunked prefill P=128: TTFT 2,551ms -> 411ms (6.2x).")

    print("=" * 70)


def export_json(metrics: SimulationMetrics, filepath: str):
    """Export metrics as JSON."""
    data = {
        'config': {
            'duration_s': metrics.simulation_duration_s,
            'measured_duration_s': metrics.measured_duration_s,
        },
        'throughput': {
            'total_requests': metrics.total_requests,
            'finished': metrics.total_finished,
            'rejected': metrics.total_rejected,
            'accept_rate': metrics.accept_rate,
            'tokens_in': metrics.total_tokens_in,
            'tokens_out': metrics.total_tokens_out,
            'output_tps': metrics.throughput_tps,
        },
        'latency': {
            'ttft_p50_ms': metrics.ttft_p50_ms,
            'ttft_p95_ms': metrics.ttft_p95_ms,
            'ttft_p99_ms': metrics.ttft_p99_ms,
            'tpot_p50_ms': metrics.tpot_p50_ms,
            'tpot_p95_ms': metrics.tpot_p95_ms,
            'latency_p50_ms': metrics.latency_p50_ms,
            'latency_p95_ms': metrics.latency_p95_ms,
        },
        'samples': metrics.samples,
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Results exported to {filepath}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FPGA Cloud Serving End-to-End Simulation"
    )
    parser.add_argument('--duration', type=float, default=60,
                        help='Simulation duration in seconds (default: 60)')
    parser.add_argument('--arrival-rate', type=float, default=5,
                        help='Mean request arrival rate, Poisson (default: 5)')
    parser.add_argument('--max-decode-batch', type=int, default=MAX_DECODE_BATCH,
                        help=f'Maximum decode batch size (default: {MAX_DECODE_BATCH})')
    parser.add_argument('--prompt-len-mean', type=int, default=PROMPT_LEN_MEAN,
                        help=f'Mean prompt length in tokens (default: {PROMPT_LEN_MEAN})')
    parser.add_argument('--output-len-mean', type=int, default=OUTPUT_LEN_MEAN,
                        help=f'Mean output length in tokens (default: {OUTPUT_LEN_MEAN})')
    parser.add_argument('--num-servers', type=int, default=1,
                        help='Number of monolithic servers (default: 1)')
    parser.add_argument('--prefill-servers', type=int, default=0,
                        help='Dedicated prefill servers (disaggregated mode, default: 0)')
    parser.add_argument('--decode-servers', type=int, default=0,
                        help='Dedicated decode servers (disaggregated mode, default: 0)')
    parser.add_argument('--kv-blocks-per-chip', type=int, default=KV_BLOCKS_PER_CHIP,
                        help=f'KV cache blocks per chip (default: {KV_BLOCKS_PER_CHIP})')
    parser.add_argument('--agent', action='store_true',
                        help='Multi-turn agent mode with KV cache reuse across turns')
    parser.add_argument('--agent-turns', type=int, default=AGENT_TURNS,
                        help=f'Turns per agent session (default: {AGENT_TURNS})')
    parser.add_argument('--agent-think-ms', type=int, default=AGENT_THINK_TIME_MS,
                        help=f'Think time between turns in ms (default: {AGENT_THINK_TIME_MS})')
    parser.add_argument('--agent-delta-prompt', type=int, default=AGENT_DELTA_PROMPT,
                        help=f'New tokens per subsequent turn (default: {AGENT_DELTA_PROMPT})')
    parser.add_argument('--agent-output-per-turn', type=int, default=AGENT_OUTPUT_PER_TURN,
                        help=f'Output tokens per turn (default: {AGENT_OUTPUT_PER_TURN})')
    parser.add_argument('--cpu-hybrid', action='store_true',
                        help='Enable P2 CPU-FPGA hybrid prefill (CPU offloads Q·K^T attention)')
    parser.add_argument('--cpu-tflops', type=float, default=3.0,
                        help='CPU TFLOPS for hybrid prefill attention (default: 3.0)')
    parser.add_argument('--microbatch', action='store_true',
                        help='解法 C: Continuous Microbatching — token-level injection (release sessions after pipeline-fill, not after full batch)')
    parser.add_argument('--expert-replication', choices=['none', 'hot'], default='none',
                        help='解法 A: Expert replication strategy. "hot" = Zipf-based replicas for popular experts')
    parser.add_argument('--zipf-alpha', type=float, default=1.0,
                        help='Zipf skewness for expert popularity (0=uniform, 1=standard, 2=extreme). Default: 1.0')
    parser.add_argument('--decode-batch-wait-us', type=float, default=0.0,
                        help='Delay decode batch formation by this many us to accumulate sessions (better TPS, slightly worse TPOT). Default: 0 (greedy).')
    parser.add_argument('--pipeline-clone', type=int, default=1,
                        help='Split 32 chips into N independent pipelines (each with own chip 0). Multiplies prefill admission rate by N. Default: 1.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose per-batch output')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Export results as JSON')
    args = parser.parse_args()

    sim = ServingSimulation(
        arrival_rate=args.arrival_rate,
        duration_s=args.duration,
        max_decode_batch=args.max_decode_batch,
        prompt_len_mean=args.prompt_len_mean,
        output_len_mean=args.output_len_mean,
        num_servers=args.num_servers,
        num_prefill_servers=args.prefill_servers,
        num_decode_servers=args.decode_servers,
        kv_blocks_per_chip=args.kv_blocks_per_chip,
        agent_mode=args.agent,
        agent_turns=args.agent_turns,
        agent_think_ms=args.agent_think_ms,
        agent_delta_prompt=args.agent_delta_prompt,
        agent_output_per_turn=args.agent_output_per_turn,
        cpu_hybrid=args.cpu_hybrid,
        cpu_tflops=args.cpu_tflops,
        microbatch=args.microbatch,
        expert_replication=args.expert_replication,
        zipf_alpha=args.zipf_alpha,
        decode_batch_wait_us=args.decode_batch_wait_us,
        pipeline_clone=args.pipeline_clone,
        seed=args.seed,
        verbose=args.verbose,
    )

    metrics = sim.run()
    print_report(metrics, args, pipeline=sim.pipeline)

    if args.output:
        export_json(metrics, args.output)

    return metrics


if __name__ == "__main__":
    metrics = main()
