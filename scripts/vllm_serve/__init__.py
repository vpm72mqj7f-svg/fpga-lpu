"""
vllm_serve — vLLM software serving stack for FPGA inference.

Functional simulation of:
  - Continuous batching scheduler (state machine: WAITING → PREFILL → DECODE → FINISHED)
  - PagedAttention KV cache block manager (LRU eviction)
  - Model runner (bridge to fpga_arch pipeline)
  - API server (Poisson request generation)
"""

from .config import *
from .types import (
    Request, Batch, Session, AgentSession, SchedulerStats,
    RequestState, BatchType,
)
from .scheduler import ContinuousBatchingScheduler
from .kv_cache import KVCacheManager
from .model_runner import ModelRunner, BatchExecutionResult
from .api_server import APIServer, RequestGenerator
from .weight_layout import WeightLayoutCompiler, LayoutReport, ChipLayout, HBMRegion
