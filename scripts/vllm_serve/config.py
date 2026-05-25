"""
vllm_serve/config.py — Scheduler and KV cache constants.

Scheduling targets and limits for the continuous batching scheduler.
"""

import numpy as np

# ============================================================================
# Scheduling targets
# ============================================================================
MAX_BATCH_SIZE         = 256      # max concurrent sequences in one batch (raised for D+C)
MAX_SEQ_LEN            = 8192     # max sequence length (tokens)
BLOCK_SIZE             = 16       # tokens per KV cache block
MAX_NUM_BLOCKS         = MAX_SEQ_LEN // BLOCK_SIZE  # 512: blocks per sequence at max length

# ============================================================================
# SLA targets (FPGA-specific, calibrated to chunked prefill + P0/P1)
# ============================================================================
# FPGA chunked prefill TTFT floor: 411ms at P=128 (P0+P1).
# GPU-class targets (50ms TTFT) are unachievable on FPGA; targets set
# slightly above the chunked prefill floor.
TTFT_TARGET_MS         = 450      # time-to-first-token target (chunked P=128)
TTFT_SLA_MS            = 500      # TTFT SLA threshold
TPOT_TARGET_MS         = 10       # time-per-output-token target (decode)
TPOT_SLA_MS            = 30       # TPOT SLA threshold

# ============================================================================
# Batch scheduling (FPGA-aware)
# ============================================================================
# FPGA decode is memory-bound and scales well with batch size:
# TPS(B) = PIPELINE_TPS * B / (B + K), K=25.4.
# Larger batches → higher throughput per request → lower cost.
# Max decode batch is generous because FPGA HBM BW (920 GB/s per chip)
# easily handles many concurrent KV reads.
PREFILL_PRIORITY       = True     # prefill gets priority over decode
MAX_PREFILL_TOKENS     = 16384    # max tokens to prefill in one batch (chunked pipeline: O(log N) time)
MIN_DECODE_BATCH       = 1        # no batching floor (was 4): admit decode as soon as any session ready
MAX_DECODE_BATCH       = 256      # max decode sequences in one batch (raised for D+C)

# ============================================================================
# KV Cache (解法 D: 扩容)
# ============================================================================
# Sizing rationale:
#   HBM per chip:           32 GB
#   Weights resident:       ~0.7 GB (fp4 experts streamed on-demand, see WEIGHT_GB_PER_CHIP)
#   Safety margin (act/buf): ~9   GB
#   → KV region:            ~22  GB  (22528 blocks × 16 tok × 1152 B ≈ 0.4 GB raw —
#                                      the per-block bytes is small; blocks-per-chip is
#                                      the real ceiling on concurrent sessions)
# Concurrent session ceiling (at P=4096):
#   blocks/session = 4096/16 = 256
#   sessions ≈ 22528 / 256 ≈ 88 sessions/chip (host distributes across chips)
KV_BLOCK_TOKENS        = 16       # tokens per block
KV_BLOCKS_PER_CHIP     = 22528    # raised from 4096 (was the limiter, not HBM bytes)
KV_BYTES_PER_TOKEN     = 576 * 2  # K + V, FP8 (from MLA_KV_BYTES * 2)
KV_GB_PER_BLOCK        = KV_BLOCK_TOKENS * KV_BYTES_PER_TOKEN / (1024**3)
KV_MAX_GB_PER_SEQ      = MAX_SEQ_LEN * KV_BYTES_PER_TOKEN / (1024**3)

# ============================================================================
# Prefill batching (adaptive)
# ============================================================================
# Max chunks per prefill batch limits chip 0 queue depth.
# Chip 0 processes ~91 chunks/s; 8 chunks → max ~88ms admission interval.
# TTFT = queue_wait + 411ms ≤ 88ms + 411ms = 499ms (within 500ms SLA).
# Smaller batches = lower latency, same token throughput (chip 0 always busy).
MAX_PREFILL_CHUNKS      = 12       # max chunks per prefill batch (×128 = 1536 tokens, ~3 reqs)
MAX_PREFILL_WAIT_US     = 100_000  # max wait before urgent admission (100ms)

# ============================================================================
# Simulation
# ============================================================================
SIM_TIME_STEP_US       = 100      # 100 us simulation granularity
WARMUP_DURATION_S      = 30       # warmup period before metrics collection

# ============================================================================
# Request generation (Poisson process)
# ============================================================================
PROMPT_LEN_MEAN        = 512      # mean prompt length (tokens)
PROMPT_LEN_MIN         = 16       # minimum prompt length
PROMPT_LEN_MAX         = 4096     # maximum prompt length
OUTPUT_LEN_MEAN        = 256      # mean output length (tokens)
OUTPUT_LEN_MAX         = 2048     # maximum output length

# ============================================================================
# Multi-turn Agent (KV cache reuse across turns)
# ============================================================================
AGENT_TURNS            = 10       # turns per agent session
AGENT_THINK_TIME_MS    = 500      # think time between turns (ms)
AGENT_DELTA_PROMPT     = 256      # new tokens per subsequent turn
AGENT_OUTPUT_PER_TURN  = 512      # output tokens per turn
