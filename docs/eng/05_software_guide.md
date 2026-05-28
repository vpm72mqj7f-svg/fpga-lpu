# FPGA LLM Inference Cluster -- Software & Simulation Guide

**Audience**: Host software team (3 C/software engineers)

**Version**: 1.3 -- May 2026

**Scope**: Driver, scheduler, API server, weight compiler, and Python simulation framework. This document is the authoritative reference for the host software stack. The Python simulation (`scripts/fpga_arch` + `scripts/vllm_serve`) IS the executable specification for the hardware. Every number in the RTL must match the simulation output.

---

## Table of Contents

1. [Software Architecture Overview](#1-software-architecture-overview)
2. [Python Simulation Stack (The Design Tool)](#2-python-simulation-stack-the-design-tool)
3. [Running Simulations (Hands-On)](#3-running-simulations-hands-on)
4. [Key Design Concepts (What the Simulation Models)](#4-key-design-concepts-what-the-simulation-models)
5. [Host Software Components (C Runtime)](#5-host-software-components-c-runtime)
6. [Software Work Breakdown (from Proposal Section 8.4.3)](#6-software-work-breakdown)
7. [C Reference Models](#7-c-reference-models)
8. [Development Workflow for Software Engineers](#8-development-workflow-for-software-engineers)
9. [Key APIs and Protocols](#9-key-apis-and-protocols)
10. [Testing Strategy for Software](#10-testing-strategy-for-software)

---

## 1. Software Architecture Overview

### 1.1 Three Layers

```
+------------------------------------------------------+
|  Python Simulation  (scripts/)                        |
|  Design Tool + Executable Specification               |
|  fpga_arch/ + vllm_serve/ + simulation/ + prefill/   |
+------------------------------------------------------+
          |  defines timing models, protocols,
          |  data formats, scheduler behavior
          v
+------------------------------------------------------+
|  Host Runtime (C)  (c_ref/)                           |
|  Production software: driver, scheduler, API server    |
|  libfpga.so + fpga_serve + weight_compiler            |
+------------------------------------------------------+
          |  MMIO/DMA commands, weight files
          v
+------------------------------------------------------+
|  FPGA RTL  (rtl/)                                     |
|  Verilog/SystemVerilog, 32-chip cluster                |
|  Bit-exact with C reference models                     |
+------------------------------------------------------+
```

### 1.2 Data Flow Through the Stack

```
API Request (HTTP, OpenAI-compatible)
  --> APIServer receives request, assigns request_id
  --> ContinuousBatchingScheduler: state WAITING->PREFILL->DECODE->FINISHED
  --> ModelRunner translates Batch -> PipelineEngine.execute_batch()
  --> PipelineEngine computes timing: latency_model(B) or throughput_model(B)
  --> FPGACluster routes tokens through 32 chips, 61 layers
  --> Results flow back: tokens -> API response (SSE streaming)
```

### 1.3 Key Files at a Glance

| Layer | File | Purpose |
|-------|------|---------|
| **Config** | `scripts/fpga_arch/config.py` | Single source of truth for ALL hardware constants |
| **HW Model** | `scripts/fpga_arch/chip.py` | FPGAChip with SRAMBank, HBMBank, DSPArray, KV blocks |
| **HW Model** | `scripts/fpga_arch/cluster.py` | 32-chip assembly, layer/weight placement |
| **HW Model** | `scripts/fpga_arch/pipeline.py` | 10-stage timing engine, TPS formula |
| **HW Model** | `scripts/fpga_arch/interconnect.py` | C2C Dual Ring + PCIe P2P with Dijkstra |
| **SW Stack** | `scripts/vllm_serve/scheduler.py` | Continuous batching, state machine |
| **SW Stack** | `scripts/vllm_serve/kv_cache.py` | PagedAttention block allocator with LRU |
| **SW Stack** | `scripts/vllm_serve/model_runner.py` | Bridge from scheduler to pipeline |
| **SW Stack** | `scripts/vllm_serve/api_server.py` | Poisson request generator |
| **SW Stack** | `scripts/vllm_serve/weight_layout.py` | Weight-to-HBM mapping compiler |
| **SW Stack** | `scripts/vllm_serve/types.py` | Request, Batch, Session, AgentSession |
| **SW Stack** | `scripts/vllm_serve/config.py` | Scheduler constants, SLA targets |
| **Sim Entry** | `scripts/run_serving.py` | End-to-end event-driven simulation |
| **FP4 Ref** | `scripts/simulation/fp4_utils.py` | FP4 E2M1 quantize/dequantize/GEMM |
| **Prefill** | `scripts/prefill/scheduler.py` | CPU-FPGA concurrent prefill scheduler |
| **C Ref** | `c_ref/src/fp4_ref.c` | Bit-exact FP4 reference for RTL verification |
| **C Ref** | `c_ref/prefill/cpu_prefill.c` | CPU prefill engine (AMX/AVX-512) |

---

## 2. Python Simulation Stack (The Design Tool)

The Python simulation is the executable specification. Modify `config.py`, re-run `run_serving.py`, and observe the impact on TPS, TTFT, TPOT, and cost. This is your primary design loop.

### 2.1 `fpga_arch/config.py` -- The Single Source of Truth

This file defines every hardware constant. When you change a value here, every module that imports it updates automatically.

**Critical constants you will modify:**

```python
# DSP compute -- the heart of TPS calculation
DSP_COUNT = 12_300          # DSP slices per chip
DSP_FREQ_MHZ = 450          # operating frequency
DSP_MAC_PER_CYCLE = 2       # fp4 x fp8 = 2 MAC/cycle
DSP_TMACS = 11.07           # per-chip theoretical throughput

# Cluster
TOTAL_CHIPS = 32
NUM_LAYERS = 61
NUM_EXPERTS = 384
TOP_K_EXPERTS = 6

# Model dimensions (DeepSeek V4 Pro)
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 3072
KV_LORA_RANK = 512
Q_LORA_RANK = 1536
O_LORA_RANK = 1024

# Pipeline performance -- the magic numbers
PIPELINE_TPS = 17_445       # saturated decode TPS
BATCH1_TPS = 660            # single-batch TPS
K_PIPELINE = 25.4           # batch efficiency coefficient
TOKEN_LATENCY_US = 1_510    # single-token latency through 32 chips
```

**How K_PIPELINE is computed:**

```
TPS(B) = PIPELINE_TPS * B / (B + K_PIPELINE)
K = PIPELINE_TPS / BATCH1_TPS - 1 = 17445/660 - 1 ~= 25.4
```

This captures pipeline fill overhead. At B=32, efficiency = 32/(32+25.4) = 55.7%. At B=256, efficiency = 91.0%.

### 2.2 `fpga_arch/chip.py` -- FPGAChip Model

Each `FPGAChip` instance models a single AGM 039-F FPGA:

```python
chip = FPGAChip(chip_id=0, card_id=0)

# Resource banks
chip.sram    # SRAMBank: 32.5 MB total, tracks deterministic/scratch/buffer
chip.hbm     # HBMBank: 32 GB total, tracks weight_storage/kv_cache/misc
chip.dsp     # DSPArray: 12,300 units, tracks busy time + utilization

# Layer and expert assignment
chip.assigned_layers     # [0, 1] for C0.0
chip.assigned_experts    # [0, 1, ..., 11] for C0.0

# KV cache block management (PagedAttention)
chip.allocate_kv_blocks(4)     # allocate 4 blocks, returns [block_ids]
chip.free_kv_blocks([bid1])    # free block
chip.access_kv_block(bid)      # touch LRU
```

**SRAMBank consumers:**

| Consumer | Size | Purpose |
|----------|------|---------|
| deterministic | ~21 MB per chip | Double-buffered attention + shared expert + router weights |
| kv_scratch | ~1 MB | KV cache working set during attention |
| expert_buffer | ~2 MB | Streaming buffer for expert weights from HBM |

**HBMBank consumers:**

| Consumer | Size | Purpose |
|----------|------|---------|
| weight_storage | ~0.7 GB | FP4 expert + attention + router weights |
| kv_cache | 0-31 GB | Per-token KV cache blocks (PagedAttention) |
| misc | small | Residuals, router tables |

### 2.3 `fpga_arch/cluster.py` -- 32-Chip Cluster Assembly

`FPGACluster` creates 32 chips across 8 cards and assigns:

- **Layers**: 61 layers distributed as 29 chips x 2 layers + 3 chips x 1 layer
- **Experts**: 384 experts, 12 per chip baseline (uniform), or Zipf-based hot replication
- **Weights**: placed in HBM and SRAM per chip

```python
cluster = FPGACluster(seed=42, expert_replication='none')
# Or with hot expert replication:
cluster = FPGACluster(seed=42, expert_replication='hot', zipf_alpha=1.0)

# Key methods
chip = cluster.get_chip_for_layer(42)          # which chip handles layer 42?
expert_chip = cluster.get_chips_for_expert(250) # all chips hosting expert 250
closest = cluster.closest_replica(chip, 250)   # closest replica from chip's view
grouped = cluster.dispatch_experts(chip, [10,20,30])  # {chip_id: [expert_ids]}
n_local = cluster.count_local_experts(chip, [10,20,30,40,50,60])  # how many local

# C2C/PCIe transfer timing
t_us = cluster.c2c_transfer_time_us(src_chip, dst_chip, payload_bytes=7168)

print(cluster.cluster_report())  # Human-readable resource summary
```

**Layer-to-chip mapping** (excerpt):

```
C0.0: L00-01, experts E00-11    C0.1: L02-03, experts E12-23
C0.2: L04-05, experts E24-35    C0.3: L06-07, experts E36-47
C1.0: L08-09, experts E48-59    C1.1: L10-11, experts E60-71
...
C7.2 (C7.2): L57-58, experts E360-371
C7.3 (C7.3): L59-60, experts E372-383
```

Single-layer chips: C1.3 (L14), C3.3 (L30), C5.3 (L46).

### 2.4 `fpga_arch/pipeline.py` -- 10-Stage Pipeline Engine

This is the most important file for understanding performance. The pipeline processes each token through 10 stages across all 61 layers.

**The 10 pipeline stages:**

```
[1] WEIGHT_PREFETCH   -- SRAM deterministic weights (0.1 us)
[2] MLA_ATTENTION     -- DSP attention: Q·K^T, A·V, projections
[3] ATTN_NORM         -- DSP RMSNorm
[4] MOE_ROUTER        -- Select top-6 experts from 384
[5] MOE_DISPATCH      -- C2C send tokens to expert chips
[6] SHARED_EXPERT     -- Always-active shared expert (DSP)
[7] ROUTED_EXPERT     -- Top-6 routed experts (DSP + HBM weight load)
[8] MOE_REDUCE        -- C2C receive expert output
[9] FFN_NORM          -- DSP RMSNorm
[10] PIPELINE_FWD     -- C2C forward hidden state to next layer
```

**Key PipelineEngine methods you will call:**

```python
eng = PipelineEngine(cluster, seed=42)

# Decode TPS for a given batch size (O(1) analytical model):
tps = PipelineEngine.throughput_model(32)       # ~9,720 tok/s at B=32
lat = PipelineEngine.decode_latency_model(32)    # ~103 us per token at B=32

# Prefill latency for given prompt length:
lat = PipelineEngine.prefill_latency_model(512)  # microseconds, P=512

# Chunked prefill:
result = PipelineEngine.chunked_prefill_model(512, chunk_size=128)
# result['ttft_ms'] -> 483 ms (true TTFT)
# result['num_chunks'] -> 4
# result['effective_tps'] -> 1,064 tok/s

# Concurrent prefill + decode:
r = PipelineEngine.concurrent_pipeline_model(prefill_tokens=512, decode_batch=32)
# r['contention_factor'] -> ~1.08

# CPU-FPGA hybrid prefill (P2):
r = PipelineEngine.cpu_hybrid_prefill_model(512, cpu_tflops=3.0)

# Detailed pipeline simulation (Monte Carlo):
sim = eng.simulate_decode(batch_size=32, num_tokens=50)
print_pipeline_result(sim)

# Execute a batch (fast analytical path for scheduler):
result = eng.execute_batch(batch_size=32, is_prefill=False)
```

**Analytical TPS formula (decode):**

```
TPS(B) = 17445 * B / (B + 25.4)

B=1      660 tok/s     (pipeline mostly idle)
B=4    2,441 tok/s
B=8    4,177 tok/s
B=16   6,741 tok/s
B=32   9,720 tok/s     (55.7% efficiency)
B=64  12,492 tok/s
B=128 14,565 tok/s
B=256 15,872 tok/s     (91.0% efficiency)
B=inf 17,445 tok/s     (theoretical maximum)
```

### 2.5 `vllm_serve/scheduler.py` -- ContinuousBatchingScheduler

State machine: **WAITING -> PREFILL -> DECODE -> FINISHED**

```
[WAITING] --admit--> [PREFILL] --first_token--> [DECODE] --done--> [FINISHED]
    |                    |                           |                    |
    +--- rejected (OOM)--+                           +--- (per step) -----+
```

On each scheduling tick:
1. Admit waiting requests if KV cache has space
2. Form prefill batch (priority, up to MAX_PREFILL_TOKENS=16384)
3. Form decode batch (all active decode requests, up to MAX_DECODE_BATCH=256)
4. Submit batches to ModelRunner

```python
scheduler = ContinuousBatchingScheduler(num_chips=32, max_decode_batch=256)

scheduler.submit_request(req)
batches = scheduler.schedule(current_time_us, kv_manager, model_runner)
scheduler.on_prefill_complete(batch, current_time_us)
scheduler.on_decode_step(batch, current_time_us)

print(scheduler.summary())
# Scheduler: 1000 submitted, 995 accepted, 5 rejected
#   Active: 32 decode, 2 prefill, 3 waiting
#   Finished: 960, tokens: 480000 in / 245760 out
#   Avg TTFT: 450.0 ms, Avg TPOT: 8.5 ms
```

### 2.6 `vllm_serve/kv_cache.py` -- KVCacheManager (PagedAttention)

Block-based KV cache with per-chip allocation and LRU eviction.

```python
kv_manager = KVCacheManager(num_chips=32, max_blocks_per_chip=22528)

# Prefill: allocate blocks for the full prompt
blocks = kv_manager.allocate_prefill(
    request_id=42, prompt_len=512,
    chip_ids=[0,1,2,3,4,5,6,7],
    current_time_us=1_000_000
)

# Decode: allocate one new block per KV_BLOCK_TOKENS steps
new_blocks = kv_manager.allocate_decode(
    request_id=42, decode_step=17,
    chip_ids=[0,1,2,3,4,5,6,7],
    current_time_us=5_000_000
)

# Release on finish
kv_manager.free_request(42)

# Stats
print(kv_manager.utilization_pct)     # e.g. 45.2%
print(kv_manager.stats_summary())
```

**Block parameters:**

| Parameter | Value |
|-----------|-------|
| Tokens per block | 16 |
| Bytes per token (K+V, FP8) | 1,152 |
| Bytes per block | 18,432 |
| Blocks per chip | 22,528 |
| GB per chip (KV) | ~0.4 GB |
| HBM per chip | 32 GB (weight ~0.7 GB, KV ~0.4 GB, headroom ~31 GB) |

### 2.7 `vllm_serve/model_runner.py` -- Bridge to Pipeline

Translates `Batch` -> `PipelineEngine` calls and handles KV cache allocation.

```python
runner = ModelRunner(cluster, pipeline, cpu_hybrid=False)

result = runner.execute_batch(batch, kv_manager, current_time_us)
# result.duration_us     -> batch wall-clock
# result.ttft_us         -> TTFT for prefill batches
# result.throughput_tps  -> analytical TPS at this batch size
# result.success         -> True/False

# For chunked prefill: first chunk completes at TTFT, decode starts immediately
# For P2 CPU hybrid: CPU attention + FPGA FFN, timing from cpu_hybrid_prefill_model()
```

### 2.8 `vllm_serve/api_server.py` -- Request Generator

```python
server = APIServer(scheduler, seed=42)

# Generate Poisson arrivals for a time window:
requests = server.generate_and_submit(
    arrival_rate=5.0,      # 5 req/s (Poisson)
    duration_us=60_000_000  # 60 seconds
)

# Manual submission (testing):
req = server.submit_manual(
    prompt_len=512, max_output=256, arrival_time_us=0
)
```

### 2.9 `vllm_serve/weight_layout.py` -- Weight Layout Compiler

Maps logical weights to physical HBM regions. Supports pipeline cloning and hot expert replication.

```python
# Baseline: 1 pipeline, 12 experts/chip uniform
wlc = WeightLayoutCompiler(pipeline_clones=1, replication='none')
layout = wlc.compile()
print(layout.summary())

# Pipeline x2 + hot replication
wlc = WeightLayoutCompiler(pipeline_clones=2, replication='hot',
                           zipf_alpha=1.0, kv_reserve_gb=22.0)
layout = wlc.compile()

# Layout attributes:
layout.pipeline_clones     # 2
layout.chip_layouts[0]     # ChipLayout: layers, experts, HBM regions
layout.max_used_gb         # max HBM usage across all chips
layout.min_free_gb         # min free HBM across all chips
layout.total_weight_gb     # aggregate weight storage
```

### 2.10 `scripts/simulation/fp4_utils.py` -- FP4 Reference

```python
from simulation.fp4_utils import quantize_fp4_e2m1, dequantize_fp4_e2m1, fp4_gemm_simulate

# Quantize
weight_fp4_idx, weight_scales = quantize_fp4_e2m1(weight_f32, group_size=128)

# Dequantize
weight_f32 = dequantize_fp4_e2m1(weight_fp4_idx, weight_scales, group_size=128)

# GEMM simulation (models FPGA DSP path: fp4 weight x fp8 activation -> fp32 accumulate)
output = fp4_gemm_simulate(weight_fp4_idx, weight_scales, activation, group_size=128)

# Compare with bf16 reference
from simulation.fp4_utils import compute_output_diff
diff = compute_output_diff(fp4_output, bf16_ref)
# diff['mean_cosine'] ~= 0.998
# diff['mean_relative_error'] ~= 0.004
```

---

## 3. Running Simulations (Hands-On)

### 3.1 Quick Start

```bash
# Basic 30-second simulation at 5 req/s
python scripts/run_serving.py --duration 30 --arrival-rate 5

# Longer run with JSON output
python scripts/run_serving.py --duration 300 --arrival-rate 10 --output results.json

# Verbose mode -- see per-batch events
python scripts/run_serving.py --duration 60 --arrival-rate 5 --verbose
```

### 3.2 Key CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--duration` | Simulation time (seconds) | 60 |
| `--arrival-rate` | Poisson arrival rate (req/s) | 5 |
| `--max-decode-batch` | Maximum decode batch size | 256 |
| `--prompt-len-mean` | Mean prompt tokens | 512 |
| `--output-len-mean` | Mean output tokens | 256 |
| `--num-servers` | Monolithic server count | 1 |
| `--prefill-servers` | Disaggregated prefill servers | 0 |
| `--decode-servers` | Disaggregated decode servers | 0 |
| `--agent` | Enable multi-turn agent mode | False |
| `--cpu-hybrid` | Enable P2 CPU-FPGA hybrid prefill | False |
| `--microbatch` | Enable continuous microbatching | False |
| `--expert-replication` | `none` or `hot` | none |
| `--pipeline-clone` | Split 32 chips into N pipelines | 1 |
| `--seed` | Random seed | 42 |

### 3.3 Running Various Deployment Modes

```bash
# Monolithic (single 32-chip pipeline)
python scripts/run_serving.py --duration 120 --arrival-rate 8 --num-servers 1

# Disaggregated (separate prefill + decode pools)
python scripts/run_serving.py --duration 120 --arrival-rate 15 \
    --prefill-servers 2 --decode-servers 4

# With pipeline cloning (x2 prefill admission rate)
python scripts/run_serving.py --duration 120 --arrival-rate 12 \
    --num-servers 1 --pipeline-clone 2

# Multi-turn agent mode (KV cache reuse)
python scripts/run_serving.py --duration 180 --arrival-rate 3 --agent \
    --agent-turns 10 --agent-think-ms 500

# CPU-FPGA hybrid prefill (P2)
python scripts/run_serving.py --duration 60 --arrival-rate 5 --cpu-hybrid --cpu-tflops 3.0

# Hot expert replication
python scripts/run_serving.py --duration 60 --arrival-rate 8 \
    --expert-replication hot --zipf-alpha 1.0

# Full combination
python scripts/run_serving.py --duration 120 --arrival-rate 20 \
    --prefill-servers 2 --decode-servers 4 --pipeline-clone 2 \
    --expert-replication hot --cpu-hybrid --agent --output full_test.json
```

### 3.4 Reading the Output Report

```
======================================================================
  SIMULATION RESULTS
======================================================================

  --- Configuration ---
  Duration:        60s
  Arrival rate:    5 req/s
  Deployment:      Monolithic (1 servers)
  Total servers:   1
  Max decode batch: 256

  --- Throughput ---
  Total requests:  302
  Finished:        298
  Rejected:        4
  Accept rate:     98.7%
  Tokens in:       152,064
  Tokens out:      76,288
  Output TPS:      1,688 tok/s          <-- This is your key performance number
  Theoretical TPS: 9,720 tok/s          <-- At avg batch size B=...
  Sys efficiency:  17.4%               <-- Scheduling + queuing overhead
  Peak TPS (B=inf): 17,445 tok/s       <-- Hardware theoretical max

  --- Latency ---
  TTFT P50:        451 ms               <-- Time-to-first-token median
  TTFT P95:        489 ms
  TTFT P99:        498 ms
  TPOT P50:        8.2 ms               <-- Time-per-output-token median
  TPOT P95:        12.1 ms
  E2E Latency P50: 2,580 ms
  E2E Latency P95: 4,210 ms

  TTFT SLA compliance: 96.2% (287/298)  <-- Target: 500ms
  TPOT SLA compliance: 98.7% (294/298)  <-- Target: 30ms

  --- Hardware Comparison (DS V4 Pro, FP8, per server) ---
  Metric                      FPGA A7           H200    Ascend 950PR
  Decode TPS (high concur)      17,445          2,000           1,500
  Prefill TPS (P=512,P0+P1)      6,122          8,000           6,000
  TTFT full (P=512) ms           2,550            120             160
  TTFT chunked (true) ms           483            N/A             N/A
  Server cost (RMB 10k)            100            300             130
```

**Interpreting the numbers:**

- **Output TPS**: Actual measured throughput. Lower than theoretical because of scheduling gaps and queuing.
- **System efficiency**: `Output TPS / Theoretical TPS`. At low load, much of the pipeline is idle. At high load, should approach 80-90%.
- **TTFT**: Dominated by prefill computation. Chunked prefill reduces 2,550ms -> 483ms. Still higher than GPU (120ms) because of compute-bound attention.
- **TPOT**: Per-token decode latency. FPGA advantage here (8.2ms vs ~15ms GPU) because memory bandwidth scales better.

### 3.5 Module-Level Smoke Tests

```bash
# FPGA architecture smoke test (cluster + pipeline)
python -c "
from fpga_arch import FPGACluster, PipelineEngine
cluster = FPGACluster(seed=42)
print(cluster.cluster_report())
eng = PipelineEngine(cluster)
print('TPS B=32:', PipelineEngine.throughput_model(32))
sim = eng.simulate_decode(32, num_tokens=50)
from fpga_arch.pipeline import print_pipeline_result
print_pipeline_result(sim)
"

# Weight layout compiler test
python -c "
from vllm_serve.weight_layout import demo_layouts
print(demo_layouts())
"

# FP4 quantization test
python -c "
from simulation.fp4_utils import quantize_fp4_e2m1, dequantize_fp4_e2m1
import numpy as np
w = np.random.randn(4096, 7168).astype(np.float32) * 0.1
idx, sc = quantize_fp4_e2m1(w, group_size=128)
w_rec = dequantize_fp4_e2m1(idx, sc, group_size=128)
print('Max abs error:', np.max(np.abs(w - w_rec[:, :7168])))
"

# Prefill scheduler smoke test
python scripts/prefill/scheduler.py
```

### 3.6 Generating Test Vectors for RTL

```bash
python scripts/simulation/gen_tb_vectors.py
```

This produces bit-exact test vectors that the C reference (`c_ref/`) and RTL testbench both consume for verification.

---

## 4. Key Design Concepts (What the Simulation Models)

### 4.1 Pipeline TPS Formula

```
TPS(B) = PIPELINE_TPS * B / (B + K_PIPELINE)
       = 17445 * B / (B + 25.4)
```

This is an empirically calibrated model. The detailed pipeline simulation (`simulate_pipeline()`) produces raw hardware TPS of ~12-14K tok/s. The analytical model applies K_PIPELINE=25.4 to capture system overheads (scheduling gaps, KV cache management, queuing), matching the effective TPS observed in end-to-end simulation.

**When K_PIPELINE changes**: With hot expert replication, more experts are local to each chip, reducing C2C dispatch/reduce overhead. The `PipelineEngine.__init__` recomputes K from Monte Carlo sampling of the actual cluster layout:

```python
# Inside PipelineEngine.__init__:
if cluster.expert_replication == 'hot':
    self._k_pipeline = self._recompute_k_with_replicas()
```

### 4.2 B=1 Token Latency

A single token through all 61 layers on 32 chips: **~1,510 microseconds**. This is the minimum possible decode latency. As batch size grows, tokens are pipelined, and per-token latency drops:

```
B=1:   1,510 us   (sequential, one token at a time)
B=32:    103 us   (pipelined, 32 tokens in-flight)
B=256:    63 us   (nearly saturated pipeline)
```

### 4.3 Chunked Prefill

Standard vLLM approach. Instead of processing all P prompt tokens at once (O(P^2) attention), split into chunks of 128 tokens.

```
P=512 with chunked prefill:
  Chunk 0: tokens 0-127   -> partial context only (411ms)
  Chunk 1: tokens 128-255 -> accumulates
  Chunk 2: tokens 256-383 -> accumulates
  Chunk 3: tokens 384-511 -> full KV cache ready
  True TTFT = 411ms (chunk 0) + 70ms (remaining 3 chunks pipelined) = 483ms
```

Without chunking: TTFT = 2,551ms. With chunking: TTFT = 483ms (5.3x speedup).

**Important**: The "first chunk" result (411ms) is partial context only -- the model sees only the first 128 tokens. The true TTFT (decodable output) is 483ms when all 4 chunks complete.

### 4.4 Expert Hit Probabilities

With 384 experts distributed across 32 chips (12 per chip, uniform):

```
P(0 local experts out of 6)  = (1 - 12/384)^6              = 82.7%  <- most common
P(1 local expert out of 6)   = C(6,1) * 12/384 * (1-P)^5   = 16.5%
P(2+ local experts out of 6) = 1 - P(0) - P(1)              = 0.8%
```

82.7% of the time, ALL 6 selected experts are remote -- tokens must be dispatched to other chips via C2C. This is the fundamental driver of C2C traffic.

**With hot expert replication**: Popular experts get replicas. The effective local hit rate increases, K_PIPELINE drops, and throughput improves.

### 4.5 KV Cache (PagedAttention)

| Parameter | Value |
|-----------|-------|
| Block size | 16 tokens |
| Bytes per token | 576 bytes x 2 (K+V, FP8) = 1,152 bytes |
| Bytes per block | 18,432 bytes (~18 KB) |
| Blocks per chip | 22,528 |
| Total blocks (32 chips) | 720,896 |

Each request allocates blocks proportional to its sequence length across all chips. At P=4096: 256 blocks per session. Concurrent session ceiling: ~88 sessions per chip.

### 4.6 Weight Placement

**SRAM (32.5 MB per chip, deterministic, double-buffered):**

| Component | Size per layer | Notes |
|-----------|---------------|-------|
| Shared expert weights | ~15 MB | Always active, always local |
| Router table | 2.6 MB | FP8, 384 x 7168 |
| RMSNorm params | 0.03 MB | FP16 |
| **Total per layer** | **13.2 MB** | Two layers = 26.4 MB per chip |

**HBM (32 GB per chip, streamed on demand):**

| Component | Size | Notes |
|-----------|------|-------|
| Expert weights | 33 MB per expert | FP4, 10.5MB gate + 10.5MB up + 10.5MB down + overhead |
| Attention weights | ~44.5 MB per layer | FP4, TP=2 shared, per-chip fraction |
| **Total per chip** | **~0.7 GB** | 12 experts x 33MB + 2 layers x 44.5MB + router |

### 4.7 C2C Interconnect

```
C2C Dual Ring (per card, 4 chips):
  Ring A:  0 -> 1 -> 2 -> 3 -> 0      (clockwise)
  Ring B:  0 <-> 2, 1 <-> 3           (cross links)

  Link BW:   128 Gbps per link
  Hop latency: 50 ns
  Payload:    4,088 bytes per frame
  Frame overhead: 24 bytes

PCIe P2P (cross-card):
  Bandwidth: 64 GB/s
  Latency:   400 ns
```

The C2C contention model (`C2CContentionModel`) computes:
- Same-card: parallel if different ring links, serialized if sharing a link
- Cross-card: all serial through shared PCIe bus
- Same-card and cross-card can overlap (different physical interfaces)

### 4.8 P2 CPU-FPGA Hybrid Prefill

CPU handles Q-K^T + A-V attention (AMX matmul on Intel Xeon / AVX-512 BF16 on AMD EPYC). FPGA handles all projections + FFN (fp8 x fp4 DSP).

Pipeline model:
```
FPGA path:   proj + FFN   (per layer, pipelined)
CPU path:    PCIe + attn  (per layer, sequential)
Bottleneck:  max(fpga_path, cpu_path)
```

With 3.0 CPU TFLOPS for FP8 matmul, the bottleneck is typically the FPGA FFN path for small prompts and the CPU attention path for large prompts.

---

## 5. Host Software Components (C Runtime)

### 5.1 Driver (`libfpga.so`)

The driver provides the low-level interface to the AGM 039-F FPGA:

- **VFIO / UIO**: Maps FPGA BARs into userspace via `/dev/vfio/` or `/dev/uio*`
- **MMIO registers**: Control/status registers at known BAR offsets
- **MSI-X interrupts**: Batch-complete, error, DMA-done notifications
- **DMA buffer management**: Allocate/free/map DMA buffers in host RAM

```c
// Conceptual driver API (production interface)
typedef struct fpga_dev fpga_dev_t;

fpga_dev_t* fpga_open(int card_id, int chip_id);
void        fpga_close(fpga_dev_t *dev);

// MMIO
uint32_t    fpga_read32(fpga_dev_t *dev, uint32_t offset);
void        fpga_write32(fpga_dev_t *dev, uint32_t offset, uint32_t value);

// DMA
int         fpga_dma_alloc(fpga_dev_t *dev, size_t size, void **host_ptr, uint64_t *phys_addr);
void        fpga_dma_free(fpga_dev_t *dev, void *host_ptr);
int         fpga_dma_submit(fpga_dev_t *dev, uint64_t phys_addr, size_t size, int dir);
int         fpga_dma_wait(fpga_dev_t *dev, int timeout_ms);
```

### 5.2 Weight Loader

Loads FP4-quantized model weights into FPGA HBM at startup.

```c
// Conceptual API
typedef struct weight_loader weight_loader_t;

weight_loader_t* wl_create(fpga_dev_t **devs, int num_devs);
int  wl_load_layer_weights(weight_loader_t *wl, int layer_idx,
                           const uint8_t *weight_fp4, const float *scales,
                           size_t weight_bytes);
int  wl_load_expert_weights(weight_loader_t *wl, int expert_idx,
                            const uint8_t *weight_fp4, const float *scales,
                            size_t weight_bytes);
int  wl_commit(weight_loader_t *wl);  // signal FPGA to use new weight set
void wl_destroy(weight_loader_t *wl);
```

### 5.3 Inference Protocol

**Command (host -> FPGA, via MMIO/DMA):**

```
Offset  | Field            | Description
--------|------------------|----------------------------------
0x0000  | cmd_code         | 0x01=PREFILL, 0x02=DECODE
0x0004  | batch_id         | Monotonically increasing batch counter
0x0008  | batch_size       | Number of tokens in this batch
0x000C  | input_addr_hi/lo | DMA address of input tokens (FP8)
0x0014  | kv_base_hi/lo    | DMA address where KV cache lives
0x001C  | exc_reg_base_hi/lo| Expert register file base address
0x0024  | seq_lens_addr_hi/lo | DMA address of per-request seq lengths (prefill only)
0x002C  | flags            | Bit 0: is_chunked, Bit 1: use_fp4_attn
```

**Completion (FPGA -> host, via MSI-X + DMA):**

```
Offset  | Field            | Description
--------|------------------|----------------------------------
0x0000  | batch_id         | Echo of submitted batch_id
0x0004  | status           | 0=OK, 1=error
0x0008  | tokens_completed | Tokens generated (decode) or processed (prefill)
0x000C  | output_addr_hi/lo| DMA address of output tokens (FP8 logits)
0x0014  | debug[0..3]      | Per-chip status: DSP util, HBM B/W, errors
```

### 5.4 Scheduler (C Production)

Ports the Python `ContinuousBatchingScheduler` to C. Key differences from the Python prototype:

- Real KV cache management with physical addresses
- Real DMA submission instead of simulated timing
- Real interrupt handling instead of event queue simulation
- Multi-threaded: one thread per card for DMA polling

### 5.5 API Server (C Production)

OpenAI-compatible REST API with SSE streaming:

```
POST /v1/chat/completions
{
  "model": "deepseek-v4-pro",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 256,
  "stream": true
}

Response (SSE):
data: {"id":"req_1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"}}]}
data: {"id":"req_1","object":"chat.completion.chunk","choices":[{"delta":{"content":" there"}}]}
...
data: [DONE]
```

### 5.6 Weight Layout Compiler (C Production)

Reads model weights (PyTorch safetensors or numpy format), runs FP4 quantization, produces `.fp4w` files for each chip:

```
File format (.fp4w):
  Header (64 bytes):
    magic:      "FP4W" (4 bytes)
    version:    uint32 (1)
    chip_id:    uint32 (0-31)
    num_regions: uint32
    reserved:   48 bytes
  Regions (variable):
    region_name[32]: char
    region_kind[16]: char  ("attention", "expert", "router", "runtime")
    start_mb:     float32
    size_mb:      float32
    weight_data:  (size_mb * 1MB) bytes of FP4 packed weights
```

### 5.7 DMA Buffer Layout

```
Input buffer (FP8 activations):
  [token_B, hidden_7168]  packed as uint8

Output buffer (FP8 logits):
  [token_B, vocab_N]  packed as uint8

KV cache (FP8, per chip):
  [layer_fraction, block_N, token_16, kv_512+64]  packed as uint8
```

---

## 6. Software Work Breakdown

From the architecture proposal, Section 8.4.3. Total: ~15.5 person-months for 3 engineers over ~10 months.

| Component | PM | Owner | Notes |
|-----------|----|----|-------|
| Inference Engine Core | 3.0 | E1 | Driver (VFIO/MMIO/DMA/MSI-X), command/response protocol, weight loader. `libfpga.so` |
| Scheduler | 4.0 | E2 | Session management, KV cache allocation, continuous batching. Port of `vllm_serve/scheduler.py` |
| API Service Layer | 3.0 | E3 | OpenAI-compatible REST API, SSE streaming, request routing |
| Ecosystem Adaptation | 2.0 | E3 | Integration with vLLM, SGLang, or custom serving framework |
| Testing & Stability | 2.0 | All | Unit/integration/perf tests, 72h+ soak, correctness vs GPU ref |
| Weight Layout Compiler | 1.5 | E1 | FP4 quantization, `.fp4w` file generation, hot expert replication support |
| **Total** | **15.5** | -- | Within 30 PM budget |

---

## 7. C Reference Models

### 7.1 `c_ref/src/fp4_ref.c` -- FP4 E2M1 Reference

Bit-exact reference implementation for RTL verification. The FPGA DSP datapath must produce output identical to this C code within the precision bounds defined by FP4 quantization.

**Key functions:**

```c
// Dequantize a single FP4 code (0-15) to float
float fp4_e2m1_dequant_code(uint8_t code);

// Quantize a float to nearest FP4 code
uint8_t fp4_e2m1_quant_value(float x);

// Quantize [rows x cols] float tensor with per-group scaling
void fp4_quantize_grouped(const float *src, uint8_t *codes, float *scales,
                          size_t rows, size_t cols, size_t group_size);

// Dequantize back to float
void fp4_dequantize_grouped(const uint8_t *codes, const float *scales,
                            float *dst, size_t rows, size_t cols, size_t group_size);

// Reference GEMM: weight(FP4) x activation(float) -> output(float)
void fp4_gemm_ref(const uint8_t *weight_codes, const float *weight_scales,
                  const float *activation, float *out,
                  size_t m, size_t k, size_t n, size_t group_size);

// Quality metrics
float fp4_cosine_similarity(const float *a, const float *b, size_t n);
float fp4_relative_l2_error(const float *a, const float *b, size_t n);
```

**FP4 E2M1 values**: `{0, +/-0.25, +/-0.5, +/-0.75, +/-1.0, +/-1.5, +/-2.0, +/-3.0}`

**Building:**

```bash
cd c_ref && make
# Produces: build/fp4_ref_test, build/cpu_prefill_test
```

### 7.2 `c_ref/prefill/cpu_prefill.c` -- CPU Prefill Engine

Reference for the P2 CPU-FPGA hybrid prefill pipeline. The CPU handles Q-K^T + A-V attention using Intel AMX or AMD AVX-512 BF16 instructions.

**Backend selection** (compile-time):

| Macro | Backend | Platform |
|-------|---------|----------|
| `__AMX_TILE__` | Intel AMX (TDPBF16PS) | Xeon Granite Rapids |
| `__AVX512BF16__` | AVX-512 BF16 (VDPBF16PS) | AMD EPYC Turin |
| (none) | Scalar fallback (OpenMP) | Any x86-64 |

**Key functions:**

```c
// FP8 GEMM (AMX/AVX-512 accelerated, returns GFLOPS achieved)
double cpu_gemm_fp8(int M, int K, int N,
                    const uint8_t *A, const uint8_t *B, float *C,
                    const float *scale_A, const float *scale_B);

// Batched GEMV (decode) -- multiple independent matvecs
double cpu_batched_gemv_fp8(int batch, int K, int N, ...);

// Full prefill through one layer (MLA + Shared Expert + Routed Experts + RMSNorm)
double cpu_prefill_layer(const cpu_prefill_config_t *cfg,
                         int layer_idx, int chunk_size,
                         const uint8_t *hidden_state, uint8_t *output_state,
                         uint8_t *kv_cache_k, uint8_t *kv_cache_v);

// Full prefill through ALL layers
double cpu_prefill_all_layers(const cpu_prefill_config_t *cfg,
                              int chunk_size,
                              const uint8_t *input_tokens, uint8_t *output_state,
                              uint8_t *kv_cache_k_all, uint8_t *kv_cache_v_all);

// Weight cache (pinned in RAM for reuse across requests)
int cpu_weight_cache_load(const cpu_prefill_config_t *cfg,
                          int layer_idx, const void *data, size_t size);
void cpu_weight_cache_unload_all(void);
```

### 7.3 How C Code Relates to RTL

The C reference establishes the **golden output** for each computation. The verification flow is:

```
1. Python generates random test inputs -> saves as .npy
2. C reference processes inputs -> saves golden_output.npy
3. RTL testbench reads same inputs -> produces rtl_output.npy
4. Comparison: assert(rtl_output == golden_output) within FP4 tolerance
```

---

## 8. Development Workflow for Software Engineers

### Step 1: Understand the Model in Python Simulation

Start with `fpga_arch/config.py`. Read every constant. Run `cluster.cluster_report()` to see the resource layout. Run `PipelineEngine.throughput_model()` for various batch sizes to understand the performance envelope.

### Step 2: Prototype Algorithm in Python (Fast Iteration)

The Python simulation runs a 60-second workload in ~1-2 seconds of wall time. This is your design loop. Example: to test a new scheduling policy -- modify `scheduler.py`, run `run_serving.py`, check the metrics.

### Step 3: Port to C for Production

Once the algorithm is validated in Python, port it to C. Use the Python output as the golden reference for your C implementation.

### Step 4: Test C Against Python Golden Output

Write a test harness that feeds the same inputs to both Python and C, then compares outputs. The FP4 GEMM and CPU prefill functions must produce bit-identical results.

### Step 5: Integrate with FPGA Hardware

When FPGA hardware is available:
1. Load weights via `weight_loader`
2. Submit inference commands via MMIO
3. Verify FPGA output matches C reference output

### General Rules

- **`config.py` is truth.** If you change a constant, change it there. Never hardcode numbers elsewhere.
- **Test at module boundaries.** Each file in `vllm_serve/` should be runnable standalone.
- **Compare against GPU.** Every model-level change should be validated against a PyTorch/HuggingFace reference for token logprobs.
- **Version the simulation.** Tag the Python simulation version that corresponds to each RTL release.

---

## 9. Key APIs and Protocols

### 9.1 Inference Command Format (to FPGA)

```
struct inference_cmd {
    uint32_t cmd_code;          // 0x01=PREFILL, 0x02=DECODE
    uint32_t batch_id;
    uint32_t batch_size;
    uint64_t input_addr;        // DMA physical address
    uint64_t kv_base_addr;      // KV cache physical base
    uint64_t expert_reg_base;   // Expert register file base
    uint64_t seq_lens_addr;     // Per-request seq len buffer (prefill only)
    uint32_t flags;             // Bit 0: chunked, Bit 1: fp4_attn
    uint32_t reserved;
};
// Total: 56 bytes, written to MMIO BAR0 offset 0x100
```

### 9.2 Inference Completion Format (from FPGA)

```
struct inference_cmpl {
    uint32_t batch_id;          // Echo
    uint32_t status;            // 0=OK
    uint32_t tokens_completed;
    uint64_t output_addr;       // DMA address of output
    uint32_t debug[4];          // DSP util, HBM BW, error count, C2C stalls
};
// Total: 40 bytes, read from MMIO BAR0 offset 0x200, signaled via MSI-X vector 0
```

### 9.3 DMA Buffer Layout

```
Input activation buffer (FP8):
  Layout: [tokens, hidden_dim] row-major, uint8
  Size:   batch_size * 7168 bytes

Output logit buffer (FP8):
  Layout: [tokens, vocab_size] row-major, uint8
  Size:   batch_size * vocab_size bytes

KV cache (per chip, PagedAttention):
  Layout: [blocks, tokens_per_block, kv_dim * 2]
  Block:  18,432 bytes (16 tokens * 576 bytes * 2 K+V)
```

### 9.4 Weight File Format (`.fp4w`)

```
Offset  | Size   | Field
--------|--------|----------------------------------
0x00    | 4      | Magic: "FP4W"
0x04    | 4      | Version: uint32 (1)
0x08    | 4      | chip_id: uint32 (0-31)
0x0C    | 4      | num_regions: uint32
0x10    | 48     | Reserved
0x40    | var    | Region 0 header (64 bytes)
        |        |   name[32]: char
        |        |   kind[16]: char
        |        |   start_mb: float32
        |        |   size_mb:  float32
        |        |   reserved: 8 bytes
0x80    | var    | Region 0 data (size_mb * 1,048,576 bytes, uint8 packed FP4)
...     | ...    | Region 1, 2, ...
```

### 9.5 KV Cache Addressing

```
Physical address on chip C:
  kv_base + block_id * BLOCK_BYTES + token_offset * KV_BYTES_PER_TOKEN * 2
  where BLOCK_BYTES = 18432, KV_BYTES_PER_TOKEN = 576

Logical address (scheduler view):
  request_id -> [block_id_0, block_id_1, ...]  (one per chip)
  Each block holds 16 tokens of K+V for a fraction of layers
```

---

## 10. Testing Strategy for Software

### 10.1 Unit Tests

Each `vllm_serve/` module should have a standalone test:

```python
# tests/test_scheduler.py
def test_state_machine():
    """Request flows through WAITING -> PREFILL -> DECODE -> FINISHED"""
    ...

def test_continuous_batching():
    """Multiple requests batched together, prefill priority"""
    ...

# tests/test_kv_cache.py
def test_allocate_free():
    """Allocate blocks, verify ref counting, free and reuse"""
    ...

def test_lru_eviction():
    """LRU evicts unused blocks when cache is full"""
    ...

# tests/test_weight_layout.py
def test_baseline_placement():
    """384 experts, 12 per chip, no overflow"""
    ...

def test_hot_replication():
    """Zipf-based replication respects HBM budget"""
    ...
```

### 10.2 Integration Tests

```bash
# End-to-end simulation (integration test)
python scripts/run_serving.py --duration 10 --arrival-rate 3 --verbose

# All deployment modes
python scripts/run_serving.py --duration 10 --arrival-rate 3 --num-servers 2
python scripts/run_serving.py --duration 10 --arrival-rate 5 --prefill-servers 2 --decode-servers 4
python scripts/run_serving.py --duration 10 --arrival-rate 3 --agent
python scripts/run_serving.py --duration 10 --arrival-rate 5 --cpu-hybrid
python scripts/run_serving.py --duration 10 --arrival-rate 5 --expert-replication hot
python scripts/run_serving.py --duration 10 --arrival-rate 12 --pipeline-clone 2
```

### 10.3 Performance Tests

```bash
# Throughput vs batch size curve
for B in 1 2 4 8 16 32 64 128 256; do
    python -c "
from fpga_arch.pipeline import PipelineEngine
tps = PipelineEngine.throughput_model($B)
lat = PipelineEngine.decode_latency_model($B)
print(f'B=$B: TPS={tps:.0f} tok/s, Latency={lat:.0f} us/tok')
"
done

# TTFT vs prompt length curve
for P in 128 256 512 1024 2048 4096; do
    python -c "
from fpga_arch.pipeline import PipelineEngine
r = PipelineEngine.chunked_prefill_model($P, chunk_size=128)
print(f'P=$P: TTFT={r[\"ttft_ms\"]:.0f} ms, eff_TPS={r[\"effective_tps\"]:.0f}')
"
done

# Load test: find saturation point
for rate in 3 5 8 12 16 20 24 28; do
    python scripts/run_serving.py --duration 120 --arrival-rate $rate --output /dev/null 2>&1 | \
        grep "Output TPS"
done
```

### 10.4 Correctness Tests

```python
# Token logprobs vs GPU reference
# Run the same prompt through PyTorch/HuggingFace and compare output tokens
def test_token_agreement():
    prompt = "The capital of France is"
    gpu_tokens = run_gpu_inference(prompt, max_tokens=10)
    fpga_tokens = run_simulation_inference(prompt, max_tokens=10)
    # With FP4 quantization, expect >99% token agreement
    agreement = sum(a == b for a, b in zip(gpu_tokens, fpga_tokens)) / len(gpu_tokens)
    assert agreement > 0.99, f"Token agreement {agreement:.2%} below threshold"

# FP4 cosine similarity
def test_fp4_quality():
    from simulation.fp4_utils import quantize_fp4_e2m1, dequantize_fp4_e2m1, compute_output_diff
    weight = np.random.randn(4096, 7168).astype(np.float32) * 0.02
    idx, sc = quantize_fp4_e2m1(weight)
    weight_rec = dequantize_fp4_e2m1(idx, sc)
    # Per-channel cosine should be > 0.995
    cos_sim = np.mean([np.dot(weight[i], weight_rec[i]) /
                       (np.linalg.norm(weight[i]) * np.linalg.norm(weight_rec[i]) + 1e-8)
                       for i in range(len(weight))])
    assert cos_sim > 0.995, f"FP4 cosine similarity {cos_sim:.4f} below threshold"
```

### 10.5 Stability Tests

```bash
# Long-duration stability run (72h equivalent at simulation speed)
# The simulation runs ~1000x faster than real-time, so 260 seconds ~= 72 hours
python scripts/run_serving.py --duration 260 --arrival-rate 8 \
    --output stability_72h.json

# Check for:
# - Memory leaks (monitor simulation memory usage)
# - KV cache fragmentation (monitor kv_util_pct over time)
# - Queue buildup (monitor waiting_count over time)
# - Degraded TPS over time (compare first vs last quartile)
```

### 10.6 C Reference Tests

```bash
cd c_ref && make test

# Tests cover:
# - FP4 quantize/dequantize roundtrip (max error < 0.5 LSB)
# - GEMM vs float reference (cosine similarity > 0.998)
# - CPU prefill vs Python golden output (bit-exact)
# - Edge cases: subnormals, max values, zero weights
```

---

## Appendix A: Quick Reference Card

```python
# The numbers you will need daily
from fpga_arch.config import *

DSP_TMACS          # 11.07 per chip
TOTAL_CHIPS        # 32
NUM_LAYERS         # 61
NUM_EXPERTS        # 384
TOP_K_EXPERTS      # 6
PIPELINE_TPS       # 17,445 (saturated decode)
BATCH1_TPS         # 660
K_PIPELINE         # 25.4
TOKEN_LATENCY_US   # 1,510 (B=1)
HIDDEN_SIZE        # 7,168
INTERMEDIATE_SIZE  # 3,072
KV_LORA_RANK       # 512
Q_LORA_RANK        # 1,536
O_LORA_RANK        # 1,024
P_0_HIT            # 0.8265
P_1_HIT            # 0.1653
P_2P_HIT           # 0.0082
PREFILL_CHUNK_SIZE # 128
KV_BLOCK_TOKENS    # 16
FPGA_TTFT_CHUNKED_MS  # 483 (P=512 true TTFT)
FPGA_PREFILL_TPS      # 6,122 (P=512 P0+P1)
FPGA_DECODE_TPS       # 17,445
```

## Appendix B: File Index

```
scripts/
  fpga_arch/
    __init__.py           Package init, re-exports
    config.py             ALL hardware constants (read this first)
    chip.py               FPGAChip, SRAMBank, HBMBank, DSPArray, KVBlock
    cluster.py            FPGACluster, ClusterStats, 32-chip assembly
    pipeline.py           PipelineEngine, 10-stage timing, TPS formulas
    interconnect.py       C2CDualRing, PCIeFabric, Dijkstra routing
    expert_popularity.py  Zipf-based expert popularity for hot replication

  vllm_serve/
    __init__.py           Package init, re-exports
    types.py              Request, Batch, Session, AgentSession, SchedulerStats
    config.py             Scheduler constants, KV cache sizing, SLA targets
    scheduler.py          ContinuousBatchingScheduler, state machine
    kv_cache.py           KVCacheManager, PagedAttention, LRU eviction
    model_runner.py       ModelRunner, bridge to pipeline
    api_server.py         APIServer, RequestGenerator
    weight_layout.py      WeightLayoutCompiler, HBMRegion, ChipLayout, LayoutReport

  simulation/
    fp4_utils.py          FP4 E2M1 quantize/dequantize/GEMM in NumPy

  prefill/
    scheduler.py          Concurrent CPU-FPGA prefill scheduler

  run_serving.py          Main entry point: event-driven end-to-end simulation

c_ref/
  src/
    fp4_ref.h             FP4 C API header
    fp4_ref.c             FP4 E2M1 reference (bit-exact for RTL verification)
  prefill/
    cpu_prefill.h         CPU prefill C API header
    cpu_prefill.c         CPU prefill engine (AMX/AVX-512)
  Makefile                Build fp4_ref_test and cpu_prefill_test

Makefile                  Top-level: docker build, FPGA synthesis, cloud deployment, lint
```

---

*End of Software & Simulation Guide. Questions to the architecture team; changes should start as a PR against `scripts/fpga_arch/config.py` with a companion simulation run.*
