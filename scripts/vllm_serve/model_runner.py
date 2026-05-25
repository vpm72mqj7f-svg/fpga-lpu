"""
vllm_serve/model_runner.py — Bridge between scheduler and FPGA pipeline engine.

Translates Batch → PipelineEngine.execute_batch() calls.
Handles prefill vs decode timing and KV cache interactions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from .types import Request, Batch, BatchType
from .config import (
    BLOCK_SIZE, KV_BLOCK_TOKENS, KV_BYTES_PER_TOKEN,
    MAX_PREFILL_TOKENS,
)
from fpga_arch.pipeline import PipelineEngine
from fpga_arch.cluster import FPGACluster
from fpga_arch.config import (
    PREFILL_USE_FP4_ATTN, PREFILL_USE_SPARSE_ATTN, PREFILL_ATTN_SPARSITY,
    PREFILL_CHUNK_SIZE, PREFILL_USE_CHUNKED,
)


@dataclass
class BatchExecutionResult:
    """Result of executing a batch on the FPGA pipeline.

    For prefill:
      - duration_us: TTFT (first chunk done → decode can start)
      - total_duration_us: actual wall-clock for full batch to clear pipeline
    For decode:
      - duration_us == total_duration_us (single step)
    """
    batch: Batch
    duration_us: float
    throughput_tps: float
    dsp_util_pct: float
    total_duration_us: float = 0.0  # for prefill: full pipeline occupancy
    ttft_us: float = 0.0            # for prefill: first-token latency
    success: bool = True
    error: str = ""


class ModelRunner:
    """Bridges the vLLM scheduler to the FPGA pipeline engine.

    Responsibilities:
      - Estimate batch execution time using throughput_model()
      - Allocate/release KV cache blocks via KVCacheManager
      - Track per-request KV cache state
      - Support P2 CPU-FPGA hybrid prefill (CPU offloads Q·K^T attention)
    """

    def __init__(self, cluster: FPGACluster, pipeline: PipelineEngine,
                 cpu_hybrid: bool = False, cpu_tflops: float = 3.0):
        self.cluster = cluster
        self.pipeline = pipeline
        self.cpu_hybrid = cpu_hybrid
        self.cpu_tflops = cpu_tflops

    def execute_batch(self, batch: Batch, kv_manager,
                      current_time_us: float) -> BatchExecutionResult:
        """Execute a batch on the FPGA pipeline.

        For prefill:
          - Process all prompt tokens in parallel
          - Allocate KV blocks for the full sequence
          - Duration = latency_model(prompt_len) for one layer × 61 layers / pipeline depth

        For decode:
          - Process one token per request in the batch
          - Allocate one new KV block per request
          - Duration = latency_model(batch_size) per token
        """
        try:
            if batch.batch_type == BatchType.PREFILL:
                duration_us, total_duration_us = self._execute_prefill(batch, kv_manager, current_time_us)
            else:
                duration_us = self._execute_decode(batch, kv_manager, current_time_us)
                total_duration_us = duration_us

            # Use analytical throughput model for the batch
            tps = PipelineEngine.throughput_model(max(1, batch.batch_size_tokens))

            return BatchExecutionResult(
                batch=batch,
                duration_us=duration_us,
                throughput_tps=tps,
                dsp_util_pct=60.0,  # from fpga_cloud_serving analysis
                total_duration_us=total_duration_us,
                ttft_us=duration_us if batch.batch_type == BatchType.PREFILL else 0.0,
                success=True,
            )
        except RuntimeError as e:
            return BatchExecutionResult(
                batch=batch, duration_us=0, throughput_tps=0,
                dsp_util_pct=0, success=False, error=str(e),
            )

    def _execute_prefill(self, batch: Batch, kv_manager,
                         current_time_us: float) -> float:
        """Execute prefill: process all prompt tokens, allocate KV blocks.

        With chunked prefill: splits prompt into PREFILL_CHUNK_SIZE chunks.
        First chunk determines TTFT (decode can start after first chunk).
        Subsequent chunks are pipelined across chips (overlap with decode).

        P2 CPU-FPGA hybrid: CPU offloads Q·K^T + A·V attention via PCIe,
        FPGA handles Q/K/V projections + FFN. Reduces FPGA DSP bottleneck.

        Returns TTFT in microseconds (first chunk latency).
        """
        total_prompt_tokens = batch.batch_size_tokens

        # Allocate KV blocks on representative chips (each chip stores KV for its layers)
        chip_ids = [c.global_id for c in self.cluster.chips[:8]]
        for req in batch.requests:
            try:
                blocks = kv_manager.allocate_prefill(
                    req.request_id, req.prompt_len,
                    chip_ids, current_time_us,
                )
                req.kv_block_ids = blocks
            except RuntimeError:
                raise

        use_fp4 = PREFILL_USE_FP4_ATTN
        sparsity = PREFILL_ATTN_SPARSITY if PREFILL_USE_SPARSE_ATTN else 0.0

        n_req = len(batch.requests)

        if self.cpu_hybrid:
            # P2: CPU-FPGA hybrid prefill — CPU handles attention, FPGA handles FFN
            chunk_result = PipelineEngine.cpu_hybrid_prefill_model(
                total_prompt_tokens,
                cpu_tflops=self.cpu_tflops,
                use_fp4_attn=use_fp4,
                attn_sparsity=sparsity,
                chunk_size=PREFILL_CHUNK_SIZE,
                n_requests=n_req,
            )
            ttft_us = chunk_result['ttft_ms'] * 1000.0
            total_us = chunk_result['total_prefill_ms'] * 1000.0
            return ttft_us, total_us
        elif PREFILL_USE_CHUNKED and total_prompt_tokens > PREFILL_CHUNK_SIZE:
            chunk_result = PipelineEngine.chunked_prefill_model(
                total_prompt_tokens, chunk_size=PREFILL_CHUNK_SIZE,
                use_fp4_attn=use_fp4, attn_sparsity=sparsity,
                n_requests=n_req,
            )
            ttft_us = chunk_result['ttft_ms'] * 1000.0
            total_us = chunk_result['total_prefill_ms'] * 1000.0
            return ttft_us, total_us
        else:
            lat = PipelineEngine.prefill_latency_model(
                total_prompt_tokens,
                use_fp4_attn=use_fp4,
                attn_sparsity=sparsity,
                n_requests=n_req,
            )
            return lat, lat

    def _execute_decode(self, batch: Batch, kv_manager,
                        current_time_us: float) -> float:
        """Execute one decode step for all requests in batch.

        Decode is memory-bound: one token per request, shared weight loads.
        Returns duration per token in microseconds.
        """
        batch_size = len(batch.requests)

        # Allocate one new KV block per request (every BLOCK_SIZE steps)
        for req in batch.requests:
            try:
                new_blocks = kv_manager.allocate_decode(
                    req.request_id, req.tokens_generated,
                    [c.global_id for c in self.cluster.chips[:8]],
                    current_time_us,
                )
                req.kv_block_ids.extend(new_blocks)
            except RuntimeError:
                raise

        # Decode latency: memory-bound, one token per request, shared weight loads.
        return PipelineEngine.decode_latency_model(batch_size)

    def free_request(self, req: Request, kv_manager):
        """Release KV cache blocks for a finished request."""
        kv_manager.free_request(req.request_id)

    @property
    def cluster_summary(self) -> str:
        return self.cluster.cluster_report()
