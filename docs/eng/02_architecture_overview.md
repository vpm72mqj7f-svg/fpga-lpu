# FPGA LLM Inference Cluster -- Architecture Overview

> **Target audience:** Overseas RTL, verification, and software engineers joining the project.
> **Version:** v1.0 (2026/05)
> **Hardware:** 32 Agilex 7 M-Series chips (8 cards x 4 chips/card)
> **Target model:** DeepSeek V4 Pro (61-layer, 384-expert MoE with MLA)
> **Status:** RTL optimization complete, Python simulation validated, CPU-prefill, FPGA decode-only architecture.

-------------------------------------------------------------------------------

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware Platform](#2-hardware-platform)
3. [Model Architecture Mapping](#3-model-architecture-mapping)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Compute Data Flow](#5-compute-data-flow)
6. [Memory Hierarchy](#6-memory-hierarchy)
7. [Interconnect Architecture](#7-interconnect-architecture)
8. [Software Stack](#8-software-stack)
9. [RTL Hierarchy](#9-rtl-hierarchy)
10. [Bring-Up vs Production](#10-bring-up-vs-production)
11. [Key Design Decisions](#11-key-design-decisions)

-------------------------------------------------------------------------------

## 1. System Overview

### 1.1 What We Are Building

We are building an FPGA-based inference accelerator cluster that serves a **DeepSeek V4 Pro** (7168 hidden, 61 layers, 384 MoE experts, 128 attention heads with MLA) model at scale. The system is **one 4U server** containing **8 FPGA accelerator cards**, each with **4 Intel Agilex 7 M-Series chips** (32 chips total). The chips are interconnected via on-board C2C SerDes dual rings within each card and PCIe 5.0 P2P across cards. No external switch, no Ethernet, no RoCE -- all communication happens within a single chassis.

The cluster achieves:
- **B=1 decode:** approximately 660 tokens/second
- **Saturated batch decode:** approximately 14,000-17,500 tokens/second
- **Primary Prefill: AMD EPYC Turin + Flash model** — 192-core CPU delivers ~1s TTFT @ P=512
- **Fallback Prefill: L20 GPU + Flash model** — ~40ms TTFT for latency-critical deployments
- **High-concurrency agent serving:** approximately 5,800-8,500 tokens/second (post-optimizations)

### Heterogeneous Prefill/Decode Architecture

```
Primary (CPU):
  Token → [AMD EPYC 9755: Flash Prefill ~1.0s] → KV Cache (27 layers)
             ↓ PCIe DMA (174μs) + KV Layer Mapping (27→61)
          [FPGA LPU: Full Decode 17,445 TPS] → Output tokens

Fallback (GPU, low-latency):
  Token → [L20 GPU: Flash Prefill ~40ms] → KV Cache (27 layers)
             ↓ PCIe DMA
          [FPGA LPU: Full Decode 17,445 TPS] → Output tokens
```

The Flash model (285B, ~27 layers) and Full model (671B, 61 layers) share identical
hidden dimensions (HIDDEN=7168, K_LATENT=512, V_LATENT=512). Flash model prefill
compute is only 44% of full model (7.9T vs 17.9T MACs @ P=512).

| Prefill Hardware | FP8 TFLOPS | Flash TTFT @ P=512 | Cost | Use Case |
|------|:---:|:---:|:---:|------|
| AMD EPYC 9755 (192C) | 8.0 | ~1.0s | ~15万 RMB | **Primary** — best CPU throughput |
| AMD EPYC 9965 (128C) | 6.0 | ~1.3s | ~10万 RMB | Balanced |
| Intel Xeon 6980P (MR-AMX) | 5.0 | ~1.6s | ~12万 RMB | Alternative |
| NVIDIA L20 GPU (Fallback) | 200 | ~40ms | ~5万 RMB | Low-latency SLA

### 1.2 Why This Exists

The primary motivation is hardware availability. NVIDIA H100/B200 GPUs are embargoed for the China market. Domestically sourced Ascend NPUs face production capacity constraints (SMIC 7nm + CoWoS). Agilex 7 FPGAs from Intel/Altera (now an independent company) have a diversified supply chain not subject to the same restrictions, and FPGA reconfigurability allows us to harden the specific arithmetic patterns (fp4 + MLA) that GPUs must emulate in software.

### 1.3 What Makes This Architecturally Unique

1. **Native fp4 compute throughout** -- weights stored at 4 bits (E2M1), activations at 8 bits (E4M3), DSPs multiply at 2 MAC/cycle for fp4xfp8 mode. No decompression overhead.
2. **MLA hardened into a single pipeline** -- KV decompression, Q-K dot product, softmax, A-V dot product, and O recompression flow through dedicated RTL without repeated HBM round-trips.
3. **Weight-stationary systolic arrays** -- the 12,300 DSPs per chip are organized as 2D weight-stationary systolic arrays. Weights are pre-loaded into the array; activations stream through. This is fundamentally different from GPU-style weight-streaming.
4. **SRAM-resident deterministic weights** -- attention Q/KV/O projections, shared expert, router table, and RMSNorm parameters are double-buffered in on-chip SRAM. In 81.6% of layers (zero local expert hits), HBM bandwidth is completely idle during compute.
5. **Hardware KV cache addressing** -- KV entries are addressed as `{session_id, layer_id, seq_pos}` through combinational logic. No software page table walk, no CPU involvement.

### 1.4 System at a Glance

```
                          +----------------------------+
                          |     Client Applications    |
                          |  OpenAI REST API /v1/...   |
                          +-------------+--------------+
                                        |
                          +-------------+--------------+
                          |    Inference Service       |
                          |  (vLLM-style scheduler,     |
                          |   FastAPI, tokenizer,       |
                          |   KV cache manager)         |
                          +-------------+--------------+
                                        | PCIe 5.0 x16
          +-----------------------------+-----------------------------+
          |                        4U Server                         |
          |                                                          |
          |   +----------+  +----------+        +----------+         |
          |   |  Card 0  |  |  Card 1  |  ...   |  Card 7  |         |
          |   | 4x A7 M  |  | 4x A7 M  |        | 4x A7 M  |         |
          |   | L00-L07  |  | L08-L14  |        | L53-L60  |         |
          |   +----+-----+  +----+-----+        +----+-----+         |
          |        |              |                    |              |
          |        +--- PCIe 5.0 Backplane (P2P) ----+              |
          |                                                          |
          |   Intra-card: C2C SerDes Dual Ring (128 Gbps/link)       |
          +----------------------------------------------------------+
```

-------------------------------------------------------------------------------

## 2. Hardware Platform

### 2.1 Agilex 7 M-Series (AGM 039-F)

| Parameter | Value | Notes |
|-----------|-------|-------|
| DSP blocks | 12,300 | With AI Tensor Block |
| DSP frequency | 450 MHz | |
| fp4xfp8 MAC | 2 MAC/cycle/DSP | Native mode |
| fp8xfp8 MAC | 1 MAC/cycle/DSP | Conservative mode |
| Total fp4 compute | 11.07 TMACs/chip | 12,300 x 2 x 450M |
| Total fp8 compute | 5.54 TMACs/chip | = 11.07 TFLOPS (GPU-equivalent) |
| HBM2e capacity | 32 GB | Per chip |
| HBM2e bandwidth | 920 GB/s | |
| On-chip SRAM (M20K) | 29.2 MB usable | 75% of 38.9 MB physical |
| On-chip SRAM (MLAB) | 3.3 MB usable | Distributed logic SRAM |
| Total usable SRAM | 32.5 MB | Per chip |
| Process node | Intel 10nm SuperFin | |
| Package | R47A (56mm x 66mm) | EMIB-attached HBM2e |

### 2.2 Single Card Topology

Each accelerator card carries **4 Agilex 7 M-Series chips** connected via **C2C dual ring**:

```
            +===============================================+
            |                 Ring A (clockwise)              |
            |   Chip0 <-----------> Chip1                    |
            |     ^                   ^                       |
            |     |                   |                       |
            |     v                   v                       |
            |   Chip2 <-----------> Chip3                    |
            +===============================================+

            +===============================================+
            |              Ring B (redundant)                 |
            |   Chip0 <-----------> Chip2                    |
            |   Chip1 <-----------> Chip3                    |
            +===============================================+
```

**Chip0 (PCIe Master):** R-Tile PCIe 5.0 x16 endpoint. The only chip on each card with a PCIe link to the host. Chips 1-3 communicate with the host and other cards exclusively via C2C-to-PCIe proxy through Chip0.

**Chips 1-3 (Slaves):** No PCIe logic. Same compute pipeline RTL as Chip0 (`chip_top.sv` with `IS_PCIE_MASTER=0`). This saves approximately 15-20% logic area per slave.

### 2.3 8-Card Cluster Topology

```
    Card 0                Card 1                Card 7
  +---------+           +---------+           +---------+
  | C0 C1   |           | C0 C1   |           | C0 C1   |
  | C2 C3   |           | C2 C3   |           | C2 C3   |
  +----+----+           +----+----+           +----+----+
       |                      |                      |
       +---- PCIe 5.0 x16 ----+---- ... ----+      |
       |        P2P           |              |      |
  +----+----------------------+--------------+------+----+
  |              PCIe 5.0 Backplane (dual CPU RC)         |
  +-------------------------------------------------------+
                          |
                  Dual Xeon Host CPU
              (SPR or GNR, VFIO driver)
```

### 2.4 Key Parameters (Aggregate)

| Parameter | Per Chip | Per Card (4 chips) | Cluster (8 cards) |
|-----------|----------|---------------------|--------------------|
| DSP count | 12,300 | 49,200 | 393,600 |
| fp4 TMACs | 11.07 | 44.28 | 354.2 |
| HBM capacity | 32 GB | 128 GB | 1 TB |
| HBM bandwidth | 920 GB/s | 3.68 TB/s | 29.4 TB/s |
| SRAM | 32.5 MB | 130 MB | 1.04 GB |
| C2C links | 2/chip | 8 links | 64 links |
| PCIe links | 1 (chip 0) | 1 per card | 8 |

-------------------------------------------------------------------------------

## 3. Model Architecture Mapping

### 3.1 DeepSeek V4 Pro Parameters

| Parameter | Value |
|-----------|-------|
| Hidden size | 7,168 |
| Intermediate size (FFN) | 3,072 |
| Layers | 61 |
| Attention heads | 128 |
| KV heads (MLA) | 1 (latent vector) |
| KV latent rank | 512 + 64 (rope) = 576 bytes/token |
| Q latent rank | 1,536 |
| O latent rank | 1,024 |
| Total experts | 384 (routed) + 1 (shared) |
| Top-K routing | 6 |
| Vocab size | 129,280 |
| Max position embedding | 1,048,576 |
| Sliding window | 128 |
| Weight format | fp4 (E2M1) |
| Activation format | fp8 (E4M3) |

### 3.2 Layer Assignment (61 Layers Across 32 Chips)

```
  29 chips host 2 layers each (58 layers)
   3 chips host 1 layer each  (3 layers)
  ---
  61 layers total

  Card 0:  C0.0 = L00-01  (+Embedding)   C0.1 = L02-03
           C0.2 = L04-05                  C0.3 = L06-07
  Card 1:  C1.0 = L08-09                  C1.1 = L10-11
           C1.2 = L12-13                  C1.3 = L14      (single-layer chip)
  Card 2:  C2.0 = L15-16                  C2.1 = L17-18
           C2.2 = L19-20                  C2.3 = L21-22
  Card 3:  C3.0 = L23-24                  C3.1 = L25-26
           C3.2 = L27-28                  C3.3 = L29      (single-layer chip)
  Card 4:  C4.0 = L30-31                  C4.1 = L32-33
           C4.2 = L34-35                  C4.3 = L36-37
  Card 5:  C5.0 = L38-39                  C5.1 = L40-41
           C5.2 = L42-43                  C5.3 = L44      (single-layer chip)
  Card 6:  C6.0 = L45-46                  C6.1 = L47-48
           C6.2 = L49-50                  C6.3 = L51-52
  Card 7:  C7.0 = L53-54                  C7.1 = L55-56
           C7.2 = L57-58                  C7.3 = L59-60  (+lm_head, +MTP)
```

### 3.3 Expert Distribution (384 Experts Across 32 Chips)

Each chip hosts **12 contiguous experts** in local HBM.

```
  C0.0: E000-E011    C0.1: E012-E023    C0.2: E024-E035    C0.3: E036-E047
  C1.0: E048-E059    C1.1: E060-E071    C1.2: E072-E083    C1.3: E084-E095
  ...
  C7.0: E336-E347    C7.1: E348-E359    C7.2: E360-E371    C7.3: E372-E383
```

**Expert hit probability** (per layer, 6 top-K experts, 12/384 = 3.125% per chip):

| Scenario | Probability | Description |
|----------|-------------|-------------|
| 0 local hits | 82.7% | All 6 selected experts are remote -- SRAM-only compute, zero HBM weight load |
| 1 local hit | 16.5% | One expert local, 5 remote -- moderate HBM weight load |
| 2+ local hits | 0.8% | Two or more experts local -- heaviest HBM weight load |

Expert selection follows a **Zipf distribution** -- the top 10 experts account for approximately 50% of token selections. Hot expert replication (a deployment-time optimization) can reduce C2C dispatch overhead by placing replicas of popular experts on additional chips.

### 3.4 Per-Token Per-Layer Compute Breakdown

```
  MLA Attention:
    Q compression (LoRA down):      7,168 x 1,536 = 11.01M MAC
    KV compression (latent):        7,168 x 512   =  3.67M MAC
    KV compression (rope):          7,168 x 64    =  0.46M MAC
    Q * K^T (nope + rope):                     ~= 29.88M MAC
    A * V (nope against c_KV):                ~= 29.36M MAC
    O decompression (LoRA):       128x512x1024  = 67.11M MAC
    O decompression (to model dim):1024x7168    =  7.34M MAC
    ---------------------------------------------------------
    MLA subtotal:                                148.83M MAC

  MoE FFN (per expert, SwiGLU):
    Gate projection:    7,168 x 3,072 = 22.02M MAC
    Up projection:      7,168 x 3,072 = 22.02M MAC
    Down projection:    3,072 x 7,168 = 22.02M MAC
    ---------------------------------------------------------
    Per expert:                           66.06M MAC

  6 routed + 1 shared expert:          462.4M MAC

  Total per layer per token:           ~611M MAC
```

-------------------------------------------------------------------------------

## 4. Pipeline Architecture

### 4.1 10-Stage Pipeline per Layer

The pipeline is defined in `scripts/fpga_arch/pipeline.py` and implemented in the RTL FSM in `rtl/layer/full_transformer_layer.sv`. Each layer decomposes into 10 stages:

```
   Stage 1       Stage 2        Stage 3       Stage 4       Stage 5
  +---------+  +----------+  +-----------+  +----------+  +-----------+
  | Weight  |  |   MLA    |  |  Attn     |  |   MoE    |  |    MoE    |
  | Prefetch|->| Attention|->|  RMSNorm  |->|  Router  |->|  Dispatch |
  | (0.1us) |  |  (DSP)   |  |  (DSP)    |  |(DSP+SRAM)|  |   (C2C)   |
  +---------+  +----------+  +-----------+  +----------+  +-----+-----+
                                                                  |
  +---------+  +----------+  +-----------+  +----------+  +-----v-----+
  |Pipeline |  |   FFN    |  |    MoE    |  |  Routed  |  |  Shared   |
  | Forward |<-|  RMSNorm |<-|   Reduce  |<-|  Expert  |<-|  Expert   |
  |  (C2C)  |  |  (DSP)   |  |   (C2C)   |  |(DSP+HBM) |  |(DSP+SRAM) |
  +---------+  +----------+  +-----------+  +----------+  +-----------+
   Stage 10      Stage 9        Stage 8       Stage 7       Stage 6
```

**Stage descriptions:**

1. **Weight Prefetch** -- Deterministic weights loaded from SRAM double-buffer. Expert weights prefetched from HBM if a local hit is expected.
2. **MLA Attention** -- Q/KV projections, Q-K dot product, softmax, A-V dot product, O decompression. Hardware pipeline in `rtl/attention/mla_attention_v2.sv`.
3. **Attention RMSNorm** -- Normalize attention output. `rtl/activation/rms_norm.sv`, single-cycle throughput.
4. **MoE Router** -- Compute top-2 expert scores. `rtl/moe/router_topk.sv`, 3-stage pipeline.
5. **MoE Dispatch** -- Send activation vector to chips hosting the selected experts. C2C dual ring, parallel dispatch.
6. **Shared Expert** -- Always-local expert FFN. Weight-stationary in SRAM. `rtl/moe/expert_ffn_engine_fp4_down.sv`.
7. **Routed Experts** -- Selected experts (0/1/2 local hits). Weights streamed from HBM. `rtl/moe/expert_ffn_engine_fp4_down.sv`.
8. **MoE Reduce** -- Receive expert outputs, weighted sum. C2C dual ring, parallel reduce.
9. **FFN RMSNorm** -- Normalize MoE output. Same module as stage 3.
10. **Pipeline Forward** -- Forward hidden state (7,168 bytes FP8) to the next chip in the pipeline. C2C dual ring.

### 4.2 Prefill vs Decode Paths

**Decode (autoregressive, B tokens):**
- Q/KV projections scale linearly with B.
- Q-K dot product scales as O(B x seq_len). For B=1, this is a vector-matrix product.
- KV cache is read, not written.
- Dominant constraint: HBM bandwidth for expert weight loading (16.9% of layers).

**Prefill:**
All prefill runs on the host CPU (Xeon AMX). KV cache produced on CPU is transferred via PCIe DMA to FPGA HBM. FPGA handles decode only.

### 4.3 Throughput Model

The pipeline performance follows a saturating curve (defined in `scripts/fpga_arch/config.py`):

```
  TPS(B) = PIPELINE_TPS * B / (B + K)

    PIPELINE_TPS = 17,445  (saturated batch throughput)
    K            = 25.4    (pipeline overhead coefficient)
    B            = batch size

  B=1:   17,445 * 1 / 26.4  =  660 tok/s
  B=8:   17,445 * 8 / 33.4  = 4,178 tok/s
  B=32:  17,445 * 32 / 57.4 = 9,725 tok/s
  B=128: 17,445 * 128 / 153.4 = 14,556 tok/s
```

Pipeline depth = 32. At any moment, up to 32 tokens are simultaneously in flight across the 32-chip pipeline, each at a different layer.

-------------------------------------------------------------------------------

## 5. Compute Data Flow

### 5.1 Single Token Through the 32-Chip Pipeline

```
  Request arrives (Poisson, rate = lambda req/s)
      |
      v
  +--------------+
  |  Scheduler   |  Admit request, assign to prefill batch
  +------+-------+
         |
         v
  +--------------+
  | Chip 0       |  Layer 00-01 (Embedding + full transformer layers)
  | (C0.0, Card0)|  +----------+    +----------+
  |              |  |Embedding |--->| L00 MLA  |--->| L00 MoE  |---> ...
  |              |  |  (HBM)   |    | (SRAM)   |    | (HBM/C2C)|
  +------+-------+  +----------+    +----------+
         | C2C Pipeline Forward (7,168 B FP8, ~250ns)
         v
  +--------------+
  | Chip 1       |  Layer 02-03
  | (C0.1, Card0)|
  +------+-------+
         | C2C Pipeline Forward (~250ns)
         v
       ... repeat for chips 2 through 30 ...
         |
         v
  +--------------+
  | Chip 31      |  Layer 59-60 (+ lm_head + MTP)
  | (C7.3, Card7)|  +----------+    +----------+
  |              |  |  L60 MoE |--->| lm_head  |---> token_id -> Host
  |              |  |          |    | (HBM)    |
  +--------------+  +----------+    +----------+

  Total single-token latency (B=1, decode): ~1,510 us
  Per-layer average: ~24.7 us
```

### 5.2 MoE Token Routing Within a Layer

```
  Token at layer L, processed by Chip A (e.g., C0.0)
      |
      v
  +--------------+
  | MoE Router   |  Compute top-6 expert scores: [E50, E120, E200, E300, E350, E5]
  | (SRAM, FPGA) |  latency: ~0.5 us
  +------+-------+
         |
         | Expert resolution:
         |   E5   -> LOCAL  (Chip C0.0, expert range E000-E011)
         |   E50  -> C0.1   (same card, C2C Ring A, 1 hop)
         |   E120 -> C1.0   (cross-card, C2C + PCIe P2P)
         |   E200 -> C4.0   (cross-card, C2C + PCIe P2P)
         |   E300 -> C7.1   (cross-card, C2C + PCIe P2P)
         |   E350 -> C7.2   (cross-card, C2C + PCIe P2P)
         v
  +------+------------------+
  | MoE Dispatch (parallel) |  Send hidden state to each expert's host chip
  | C2C + PCIe P2P          |  250ns dispatch latency (parallel across 6 destinations)
  +------+------------------+
         |
         v
  +------+------------------+
  | Expert FFN Computation  |  1 shared expert (always local, SRAM weights)
  | (6 experts in parallel) |  0-2 routed experts (local, HBM weights)
  |                         |  4-6 routed experts (remote, computed on host chips)
  +------+------------------+
         |
         v
  +------+------------------+
  | MoE Reduce (parallel)   |  Collect expert outputs, weighted sum
  | C2C + PCIe P2P          |  250ns reduce latency
  +-------------------------+
         |
         v
  Pipeline Forward -> Next chip
```

### 5.3 Pipeline Concurrency (32 Tokens In-Flight)

```
  Time --->
  
  Chip 0:  [T0:L00] [T0:L01] [T1:L00] [T1:L01] [T2:L00] [T2:L01] ...
  Chip 1:           [T0:L02] [T0:L03] [T1:L02] [T1:L03] [T2:L02] ...
  ...
  Chip 31:                              ... [T0:L59] [T0:L60] [T1:L59] ...

  Pipeline fill: 32 stages must fill before first token emerges.
  Steady state: new token result emerges every ~1,510/32 = ~47 us (B=1).
  With batch B>1: throughput scales sub-linearly per the TPS(B) model above.
```

-------------------------------------------------------------------------------

## 6. Memory Hierarchy

### 6.1 Three-Tier Memory

```
  +--------------------------------------------------------------+
  |                      HBM2e (32 GB/chip)                       |
  |                     920 GB/s bandwidth                        |
  |                                                               |
  |  +---------------------------+  +---------------------------+ |
  |  |   Weight Zone (~0.7 GB)   |  |   Runtime Zone (~31.3 GB)| |
  |  |                           |  |                           | |
  |  |  * 12 routed experts fp4 |  |  * KV cache (FP8)         | |
  |  |    (~33 MB/expert, 396 MB)|  |  * Activation buffers     | |
  |  |  * 1 shared expert fp4   |  |  * C2C RX/TX ring buffers | |
  |  |    (~33 MB)              |  |  * Expert replica storage  | |
  |  |  * Attention weights fp4 |  |  * CPU prefill KV staging  | |
  |  |    (~89 MB across TP=2)  |  |                           | |
  |  |  * Router table fp8      |  |  KV per token: 576 B FP8  | |
  |  |    (~2.6 MB)             |  |  128K ctx x 2 layers:     | |
  |  |  * RMSNorm params fp16   |  |    128K x 2 x 576B = 147MB| |
  |  |    (~0.03 MB)            |  |                           | |
  |  +---------------------------+  +---------------------------+ |
  +--------------------------------------------------------------+
  
  +--------------------------------------------------------------+
  |                    On-Chip SRAM (32.5 MB)                     |
  |                                                               |
  |  M20K (29.2 MB usable):                                       |
  |  +----------------------------------------------------------+|
  |  | Deterministic weights (double-buffered): 18.6 MB          ||
  |  |   * Shared expert fp4:                         4.4 MB     ||
  |  |   * Attention Q/KV/O fp4 (TP-split):          4.4 MB     ||
  |  |   * Router table fp8:                          0.37 MB    ||
  |  |   + Prefetch buffer (next layer):             ~9.3 MB     ||
  |  +----------------------------------------------------------+|
  |  | Systolic array weight stationary:              2.0 MB     ||
  |  | Expert weight streaming prefetch buffer:       4.0 MB     ||
  |  | KV cache hot window key index:                 2.0 MB     ||
  |  | Router routing table (all layers):             2.0 MB     ||
  |  | Fragmentation margin:                          0.8 MB     ||
  |  +----------------------------------------------------------+|
  |                                                               |
  |  MLAB (3.3 MB usable):                                       |
  |  +----------------------------------------------------------+|
  |  | Session table + KV address generation:         1.0 MB     ||
  |  | C2C/PCIe packet buffers:                       1.0 MB     ||
  |  | Systolic accumulator partial sums:             1.0 MB     ||
  |  | FSM control state:                             0.3 MB     ||
  |  +----------------------------------------------------------+|
  +--------------------------------------------------------------+
  
  +--------------------------------------------------------------+
  |                    DSP Array (12,300 units)                    |
  |                                                               |
  |  Weight-stationary 2D systolic array:                         |
  |    M_ROWS = 32 rows (output parallelism)                      |
  |    LANES  = 128 columns (K-direction parallelism)             |
  |    Cells  = 4,096 (per array)                                 |
  |    Configurable: 8 parallel arrays for MoE experts            |
  |                                                               |
  |  Each cell: fp4_systolic_cell (rtl/dsp/fp4_systolic_cell.sv)  |
  |    * Stores 4-bit E2M1 weight locally                         |
  |    * Accepts 8-bit E4M3 activation per cycle                  |
  |    * Accumulates in FP32                                      |
  |    * Pre-decoded fp8 scale factor (12-bit)                    |
  +--------------------------------------------------------------+
```

### 6.2 Weight Placement Strategy

This is one of the most important architectural decisions. The guiding principle:

| Weight Type | Precision | Location | Size per Layer | Why |
|-------------|-----------|----------|----------------|-----|
| Attention Q/KV/O | fp4 | SRAM (double-buffered) | ~9.3 MB | Needed every layer, deterministic |
| Shared Expert | fp4 | SRAM (double-buffered) | ~4.4 MB | Needed every layer, deterministic |
| Router table | fp8 | SRAM | ~0.37 MB | Sensitivity to quantization error |
| RMSNorm params | fp16 | SRAM | ~0.03 MB | Tiny, needed every layer |
| Routed Experts | fp4 | HBM | 33 MB each | 384 experts, too large for SRAM, only accessed on hit |
| Embedding | fp16 | HBM (Chip 0 only) | ~1.85 GB | Too large; table lookup, not matrix multiply |
| lm_head | fp16 | HBM (Chip 31 only) | ~1.85 GB | Too large; only accessed on final decode step |

**Why SRAM double-buffering:** The "current layer" deterministic weights are in one buffer while the "next layer" weights prefetch into the other. In 81.6% of layers (zero local expert hits), HBM is completely idle -- all compute uses SRAM-resident weights. This is the core reason FPGA achieves approximately 50% DSP utilization at B=1 vs GPU's approximately 2-5%.

### 6.3 KV Cache Organization

KV entries are stored in HBM as a flat array addressed by combinational hardware:

```
  Physical address = KV_BASE
                   + session_id * session_stride
                   + layer_id   * layer_stride
                   + seq_pos    * KV_BYTES_PER_TOKEN

  KV_BYTES_PER_TOKEN = 576 bytes (K_latent 512 + rope 64, all FP8)
```

The KV cache hardware manager (`rtl/attention/mla_kv_cache.sv` for on-chip cache, HBM for bulk storage) operates a ring buffer with sliding window eviction (window=128 positions). This is fundamentally different from GPU PagedAttention: no page table, no block list traversal, no CPU involvement in address translation.

KV entries arrive via PCIe DMA from the CPU prefill pipeline through `rtl/chip/kv_dma_bridge.sv`, which provides double-buffered HBM banks -- one bank for active FPGA decode reads, one bank for incoming CPU prefill writes, with atomic buffer swap at decode step boundaries.

-------------------------------------------------------------------------------

## 7. Interconnect Architecture

### 7.1 Three Communication Layers

```
  +==============================================================+
  | Layer 3: Inter-Node (Phase 2)                                |
  |   RoCE v2 RDMA over 200GbE (F-Tile)                          |
  |   Multi-server scale-out                                      |
  |   (Not implemented in Phase 1 -- single server only)         |
  +==============================================================+
  +==============================================================+
  | Layer 2: Inter-Card                                           |
  |   PCIe 5.0 x16 P2P DMA                                        |
  |   64 GB/s unidirectional                                       |
  |   ~400ns end-to-end latency                                    |
  |   Used for: MoE cross-card dispatch/reduce,                   |
  |             pipeline forward (when layers span cards)          |
  +==============================================================+
  +==============================================================+
  | Layer 1: Intra-Card (C2C Dual Ring)                           |
  |   F-Tile SerDes, 32 Gbps NRZ per lane                          |
  |   4 lanes per link = 128 Gbps unidirectional                   |
  |   2 links per chip (Ring A + Ring B)                           |
  |   ~50ns per hop                                                |
  |   Used for: MoE intra-card dispatch/reduce,                    |
  |             pipeline forward, PCIe proxy traffic               |
  +==============================================================+
```

### 7.2 C2C Dual Ring Details

**Physical layer:**
- F-Tile transceivers, NRZ 32 Gbps (no PAM4, no DSP overhead)
- 4-lane bonded per link, AC-coupled, on-chip termination
- PCB trace < 200mm between chips
- Link training: TS1/TS2 sequence (PCIe-like), <1ms

**Ring A (clockwise, primary):** C0-C1-C2-C3-C0
**Ring B (cross-connect, redundant):** C0-C2, C1-C3

**Routing:** Dijkstra shortest path on Ring A, computed at compile time per chip. Maximum 2 hops, approximately 100ns.

**Frame format (defined in `c2c_packet.svh`):**
```
  +-- SOP (8B) --+-- Header (8B) --+-- Payload (0-4088B) --+-- CRC32 --+-- EOP (4B) --+
  
  Header fields:
    Type (4b):   MoE_Dispatch / MoE_Reduce / Pipeline_Fwd / PCIe_Proxy /
                 Credit_Update / Weight_Broadcast / Heartbeat
    Priority (2b): 0=Credit/Mgmt, 1=Pipeline, 2=MoE, 3=PCIe_Proxy
    VC (2b):       0=Control, 1=Data_HP, 2=Data_Bulk, 3=Management
    SrcChip (5b), DstChip (5b), SeqNum (8b), FrameLen (12b)
```

**Credit-based flow control:** RX buffer per VC per port. TX decrements credits per frame; RX returns credits after consumption. No congestion -- dedicated physical links, no shared medium.

**Error handling:** CRC32 per frame. Single error -> NAK + retransmit. 3 consecutive errors -> switch to Ring B. Both rings down -> interrupt host.

### 7.3 PCIe 5.0 P2P

**Physical:** R-Tile PCIe 5.0 x16 hard IP. Only Chip 0 per card connects to PCIe.

**Chip 0 BAR4 layout (64 MB):**
```
  0x0000_0000 - 0x00FF_FFFF: Chip 0 (local registers + DMA)
  0x0100_0000 - 0x01FF_FFFF: Chip 1 (via C2C proxy)
  0x0200_0000 - 0x02FF_FFFF: Chip 2 (via C2C proxy)
  0x0300_0000 - 0x03FF_FFFF: Chip 3 (via C2C proxy)
```

**Cross-card data flow:**
```
  Card A Chip X -> Card A Chip 0 (C2C) -> PCIe MWr -> Card B Chip 0 (R-Tile)
                -> Card B Chip 0 -> Card B Chip Y (C2C)
```

**P2P Bandwidth:** PCIe 5.0 x16 = ~64 GB/s. Cross-socket bottleneck is UPI 2.0 at ~20 GB/s. Mitigation: prefer same-socket expert allocation (experts 0-191 on CPU0 cards, 192-383 on CPU1 cards), halving cross-socket traffic.

### 7.4 Communication Bandwidth Budget (200 tok/s decode)

```
  Intra-card C2C (MoE dispatch/reduce):    65 MB/s   -> <0.1% of 128 Gbps
  Same-socket PCIe P2P:                    193 MB/s  -> 3% of 64 GB/s
  Cross-socket UPI:                        769 MB/s  -> 3.8% of 20 GB/s
  Pipeline layer forwarding:               43 MB/s   -> <0.1%
  
  All communication paths < 5% utilization at 200 tok/s.
  Communication latency: C2C 250ns, PCIe 400ns vs compute ~3us per layer.
  -> Communication is not the bottleneck.
```

-------------------------------------------------------------------------------

## 8. Software Stack

### 8.1 Three-Layer Architecture

```
  +============================================================+
  |  Layer 3: run_serving.py                                   |
  |  Event-driven simulation & orchestration                     |
  |                                                              |
  |  - Poisson request arrival generator                         |
  |  - Batch completion event handler                            |
  |  - Metrics sampling (1 Hz): TTFT, TPOT, P50/P95/P99 latency |
  |  - Disaggregated deployment coordinator (prefill + decode)   |
  |  - Drain phase handling                                      |
  +============================================================+
           |  Batch requests + KV block allocations
           v
  +============================================================+
  |  Layer 2: vllm_serve/                                      |
  |  vLLM-style serving framework                                |
  |                                                              |
  |  api_server.py    - Request generator, FastAPI mock          |
  |  scheduler.py     - Continuous batching scheduler            |
  |                     State machine: WAITING -> PREFILL ->     |
  |                     DECODE -> FINISHED                        |
  |  model_runner.py  - Batch -> hardware pipeline bridge        |
  |                     Prefill/decode latency estimation        |
  |  kv_cache.py      - Block-based KV cache manager             |
  |                     16 tokens/block, LRU eviction            |
  |                     Per-chip sharded storage                  |
  |  weight_layout.py - Weight-to-chip mapping compiler          |
  |  config.py        - Serving configuration                    |
  +============================================================+
           |  Hardware model calls
           v
  +============================================================+
  |  Layer 1: fpga_arch/                                       |
  |  FPGA hardware architecture model                             |
  |                                                              |
  |  config.py        - Unified hardware constants               |
  |                     (DSP counts, model dims, C2C params,     |
  |                      MAC counts, expert probabilities)       |
  |  chip.py          - Single FPGA chip model                   |
  |                     DSPArray, SRAMBank, HBMBank,              |
  |                     KV block manager                         |
  |  cluster.py       - 32-chip cluster assembly                 |
  |                     Layer assignment, expert distribution,   |
  |                     hot expert replication                   |
  |  interconnect.py  - C2C Dual Ring + PCIe P2P fabric         |
  |                     Dijkstra routing, transfer time models  |
  |  pipeline.py      - 10-stage pipeline engine                 |
  |                     Dual-path timing (fast throughput model  |
  |                     + detailed stage-by-stage)               |
  |  expert_popularity.py - Zipf-distributed expert sampling     |
  +============================================================+
```

### 8.2 Key Files and Their Roles

| File | Purpose |
|------|---------|
| `scripts/run_serving.py` | Top-level event-driven serving simulation. Configures server topology (disaggregated/cloned), dispatches requests, collects metrics. |
| `scripts/vllm_serve/scheduler.py` | `ContinuousBatchingScheduler` -- prefill-priority scheduling, decode batch formation, KV block allocation orchestration. |
| `scripts/vllm_serve/model_runner.py` | Bridges the gap between the scheduler and the hardware model. Calls `PipelineEngine.execute_batch()` for both prefill and decode. |
| `scripts/vllm_serve/kv_cache.py` | `KVCacheManager` -- block-based allocation (16 tokens/block), per-chip sharding, LRU eviction. |
| `scripts/fpga_arch/config.py` | **Single source of truth** for all hardware constants. DSP counts, model dimensions, MAC counts, C2C parameters, weight sizes, expert probabilities, calibrated pipeline performance. |
| `scripts/fpga_arch/pipeline.py` | `PipelineEngine` with dual-path timing: fast `throughput_model(B)` for scheduler use, detailed `_per_layer_timing()` for architectural analysis. |
| `scripts/fpga_arch/cluster.py` | `FPGACluster` -- assembles 32 chip objects, assigns layers, distributes experts, creates C2C ring fabric per card. |
| `scripts/fpga_arch/interconnect.py` | `C2CDualRing` and `PCIE2PFabric` -- Dijkstra routing, frame-level transfer time modeling. |

### 8.3 Driver Model

The FPGA kernel driver uses Linux VFIO (userspace device control), not a custom kernel module:

```
  Linux Kernel:
    PCIe Subsystem (VFIO)
    /dev/vfio/N              <- userspace direct FPGA control (1 device per card)
    MSI-X interrupts         <- inference done / error / heartbeat
    IOMMU                    <- DMA address isolation
    PCIe P2P                 <- drivers/pci/p2p.c (native kernel support)

  Userspace (libfpga.so):
    VFIO mmap                <- HBM address space mapping
    MMIO register access     <- inference command dispatch
    MSI-X event handling     <- completion interrupt processing
    DMA buffer management    <- weight loading, KV cache I/O
```

P2P is configured once at boot:
```
  echo 1 > /sys/bus/pci/devices/0000:01:00.0/p2pmem/enable  # Card A
  echo 1 > /sys/bus/pci/devices/0000:02:00.0/p2pmem/enable  # Card B
```

No kernel module compilation, no version-locked driver -- the FPGA appears as a standard PCIe endpoint to the OS.

### 8.4 OpenAI API Compatibility

The inference service exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models` via FastAPI. Any OpenAI client SDK works without modification. This includes LangChain, Dify, Open WebUI, and custom frontends.

-------------------------------------------------------------------------------

## 9. RTL Hierarchy

### 9.1 Top-Level Modules (`hw/src/`)

```
  +---------------------------+
  |    top_master.sv          |  Master FPGA (Chip 0 of each card)
  |                           |  - PCIe 5.0 R-Tile host interface
  |                           |  - KV DMA Engine (host SSD -> HBM)
  |                           |  - chip_top instantiation (IS_PCIE_MASTER=1)
  |    Key sub-modules:       |
  |      kv_dma_engine        |  <- rtl/chip/kv_dma_engine.sv
  |      chip_top #(.IS=1)    |  <- rtl/chip/chip_top.sv
  +---------------------------+

  +---------------------------+
  |    top_slave.sv           |  Slave FPGA (Chips 1-3 of each card)
  |                           |  - Identical compute pipeline as Master
  |                           |  - No PCIe IP (saves ~15-20% logic)
  |                           |  - C2C passthrough mode (bring-up)
  |                           |  - chip_top instantiation (IS_PCIE_MASTER=0)
  +---------------------------+

  +---------------------------+
  |  top_full_stack.sv        |  Full integration test (12-layer pipeline)
  |  (hw/src/full_stack/)     |  - Weight loading from HBM/PCIe
  |                           |  - Test token injection
  |                           |  - Output verification vs C golden model
  |                           |  - LED/UART debug interface
  +---------------------------+

  +-------------------++-------------------+
  | top_bringup.sv    || top_hbm_char.sv  |  (hw/src/bringup/, hw/src/hbm_char/)
  | (bring-up config) || (HBM test)       |  Parameterized for early validation
  +-------------------++-------------------+
```

### 9.2 Chip-Level Module (`rtl/chip/`)

```
  chip_top.sv
  Parameters: CHIP_ID, CARD_ID, LAYER_START, LAYER_END, IS_PCIE_MASTER
  
  - C2C dual ring ports (rx_a, tx_a, rx_b, tx_b)
  - PCIe DMA ports (master only)
  - C2C proxy bridge (cross-card forwarding)
  - Pipeline token ingress/egress (pipe_in_*, pipe_out_*)
  - MoE dispatch/reduce interfaces
  - Config registers (layer range, expert bitmap)
  
  Instantiates:
    full_transformer_layer u_layer  -- 1 or 2 layers per chip
    C2C routing logic               -- Dijkstra shortest path
    PCIe DMA engine (master only)   -- descriptor ring + H2D/D2H streams
```

### 9.3 Layer-Level Module (`rtl/layer/`)

```
  full_transformer_layer.sv
  Parameters: HIDDEN, K_LATENT, V_LATENT, NUM_SLOTS, MAX_POS, WEIGHT_W, DATA_W
  
  Main FSM (10 states): S_IDLE -> S_R1 -> S_ATTN -> S_R2 -> S_RTR
                        -> S_FFN_LD1 -> S_FFN_LD2 -> S_FFN -> S_R3 -> S_OUT
  
  Instantiates:
    rms_norm                x3   (rtl/activation/rms_norm.sv)
    mla_attention_v2        x1   (rtl/attention/mla_attention_v2.sv)
    router_topk             x1   (rtl/moe/router_topk.sv)
    expert_ffn_engine_fp4_down x1 (rtl/moe/expert_ffn_engine_fp4_down.sv)
    q12_to_fp8_e4m3         x8   (rtl/activation/q12_to_fp8_e4m3.sv)
```

### 9.4 Attention Subsystem (`rtl/attention/`)

```
  mla_attention_v2.sv
  Full MLA pipeline: hidden -> QKV proj -> RoPE -> KV cache -> attention -> output
  
  Sub-modules:
    mla_qkv_proj.sv      - Low-rank Q, K, V projections (7168->1536/512/64)
    mla_rope.sv           - Decoupled RoPE (64-dim rope part only)
    mla_kv_cache.sv       - Ring buffer KV cache (M20K BRAM + MLAB valid bits)
  
  FSM (9 states): S_IDLE -> S_QKV_PROJ -> S_ROPE -> S_CACHE_WR
                  -> S_CACHE_RD_INIT -> S_CACHE_RD -> S_ATTN_SCORE
                  -> S_SOFTMAX -> S_OUTPUT
```

### 9.5 DSP Subsystem (`rtl/dsp/`)

```
  fp4_gemm_engine.sv          - Top-level GEMM engine (production)
  fp4_systolic_2d.sv           - 2D weight-stationary systolic array
  fp4_systolic_cell.sv         - Single systolic cell (fp4 weight + fp8 activ in, FP32 accum)
  fp4_mac.sv                   - fp4 x fp8 multiply-accumulate unit
  fp4_scale_reader.sv          - Scale factor decode and apply
  fp4_prefill_engine.sv        - Batched prefill GEMM (P tokens, shared weights) (reserved for future FPGA-side prefill)
```

The 2D systolic array (`fp4_systolic_2d.sv`) is the core compute unit:
- **M_ROWS** rows (output parallelism, typically 32)
- **LANES** columns (K-direction parallelism, typically 128)
- Each cell stores a 4-bit fp4 weight and 12-bit pre-decoded scale
- Activations broadcast per column, accumulate along rows
- After all K-beats, per-row adder tree reduces LANES results into one output
- Weights are loaded externally (from SRAM or HBM) via `wt_wr_en`
- See `rtl/dsp/fp4_systolic_2d.sv` for the complete microarchitecture

### 9.6 MoE Subsystem (`rtl/moe/`)

```
  router_topk.sv               - Top-2 expert selection (3-stage pipeline)
    Pipeline: latch -> partials -> reduce + top-2
    Weights: EXPERT x HIDDEN, stored in registers or M20K
  
  expert_ffn_engine_fp4_down.sv - Expert FFN (gate/up/down with SwiGLU)
    FSM: S_IDLE -> S_RUN_GU -> S_MID -> S_LOAD_DOWN -> S_RUN_DOWN -> S_DONE
    Instantiates: fp4_linear_engine x3 (gate, up, down)
                  silu_q12_lut        x INTER (SiLU activation)
                  q12_to_fp8_e4m3     (precision conversion)
```

### 9.7 Remaining Subsystems

| Module | File | Purpose |
|--------|------|---------|
| RMSNorm | `rtl/activation/rms_norm.sv` | RMS normalization, single-cycle, fully pipelined |
| SiLU | `rtl/activation/silu_q12_lut.sv` | SiLU activation via Q12 LUT |
| Q12-FP8 | `rtl/activation/q12_to_fp8_e4m3.sv` | Fixed-point to FP8 E4M3 conversion |
| MTP Head | `rtl/head/mtp_head.sv` | Multi-token prediction (2-4 future tokens) |
| MTP Verify | `rtl/head/mtp_verify.sv` | MTP hypothesis verification |
| MHC Mixer | `rtl/layer/mhc_mixer.sv` | Hyper-connection mixing matrix |
| KV DMA Bridge | `rtl/chip/kv_dma_bridge.sv` | CPU prefill KV -> FPGA HBM DMA |
| KV DMA Engine | `rtl/chip/kv_dma_engine.sv` | DMA descriptor engine |
| SRAM Cache | `rtl/engram/sram_cache.sv` | On-chip SRAM cache controller |
| Hash Unit | `rtl/engram/hash_unit.sv` | Hardware hash for KV prefix matching |
| Lookup Engine | `rtl/engram/lookup_engine.sv` | Generic lookup table engine |
| UART Debug | `rtl/debug/uart_debug.sv` | UART-based debug output |
| DSP Stress | `rtl/debug/dsp_stress_test.sv` | DSP utilization stress test |
| HBM BW Test | `rtl/debug/hbm_bw_test.sv` | HBM bandwidth characterization |

### 9.8 Testbenches (`rtl/sim/`)

All 20+ testbenches in `rtl/sim/` follow a consistent pattern: generate weight data, feed activation input, capture output, compare against golden model.

| Testbench | Tests |
|-----------|-------|
| `tb_fp4_mac.sv` | Single fp4 x fp8 MAC cell |
| `tb_fp4_systolic_cell.sv` | Single systolic cell with scale reader |
| `tb_fp4_systolic_tile.sv` | Tile-level systolic tests |
| `tb_fp4_systolic_2d.sv` | Full 2D systolic array |
| `tb_fp4_systolic_array.sv` | Multiple array configuration |
| `tb_fp4_gemm_engine.sv` | Top-level GEMM engine |
| `tb_fp4_linear_engine.sv` | Linear projection engine |
| `tb_fp4_prefill_engine.sv` | Prefill-mode batched GEMM (reserved for future FPGA prefill) |
| `tb_fp4_scale_reader.sv` | Scale factor decoding |
| `tb_expert_ffn_engine.sv` | Expert FFN (gate/up/down) |
| `tb_expert_ffn_engine_fp4_down.sv` | Expert FFN with fp4 down projection |
| `tb_rms_norm.sv` | RMSNorm accuracy |
| `tb_router_topk.sv` | Router top-K selection |
| `tb_silu_q12_lut.sv` | SiLU LUT accuracy |
| `tb_mla_attention.sv` | MLA attention end-to-end |
| `tb_mla_attention_v2.sv` | MLA attention v2 (with RoPE + KV cache) |
| `tb_mla_qkv.sv` | QKV projection |
| `tb_layer_compute_engine.sv` | Layer compute engine |
| `tb_full_transformer_layer.sv` | Full transformer layer |
| `tb_chip_12layer.sv` | 12-layer chip stack |
| `tb_cluster_384.sv` | 384-expert cluster |
| `tb_c2c_ring.sv` | C2C ring topology |
| `tb_kv_dma.sv` | KV DMA bridge |
| `tb_mtp_head.sv` | Multi-token prediction head |
| `tb_mhc_mixer.sv` | MHC mixing matrix |
| `tb_lookup_engine.sv` | Lookup engine |

-------------------------------------------------------------------------------

## 10. Bring-Up vs Production

### 10.1 Configuration Architecture

The codebase uses `lpu_config.svh` (included by all RTL modules) and `scripts/fpga_arch/config.py` (used by all Python models) as the **single source of truth** for hardware parameters. The SystemVerilog side uses the `` `define FPGA_LPU_PRODUCTION `` flag to select between two configurations:

```
  Bring-Up (FPGA_LPU_PRODUCTION not defined):
    HIDDEN      = 8      (minimal hidden dim)
    K_LATENT    = 4      
    V_LATENT    = 4      
    INTER       = 4      (minimal FFN intermediate)
    EXPERTS     = 4      
    KV_SLOTS    = 64     (limited SRAM)
    MAX_POS     = 64
    WEIGHT_W    = 16     (Q12 fixed point)
    DATA_W      = 32
    LANES       = 8      (small systolic array)
    M_ROWS      = 4

  Production (FPGA_LPU_PRODUCTION defined):
    HIDDEN      = 7168
    K_LATENT    = 512
    V_LATENT    = 512
    INTER       = 3072
    EXPERTS     = 384
    KV_SLOTS    = configurable (HBM-backed, up to 128K tokens)
    MAX_POS     = 1_048_576
    WEIGHT_W    = 4       (fp4 E2M1)
    DATA_W      = 8       (fp8 E4M3)
    LANES       = 128     (full systolic array)
    M_ROWS      = 32
```

### 10.2 Bring-Up Top-Level Modules

| File | Purpose |
|------|---------|
| `hw/src/bringup/top_bringup.sv` | Minimal bring-up: LED blink, UART echo, basic systolic test |
| `hw/src/hbm_char/top_hbm_char.sv` | HBM characterization: read/write bandwidth test |
| `hw/src/dsp_char/top_dsp_char.sv` | DSP characterization: fp4 MAC accuracy sweep |
| `hw/src/c2c_test/top_c2c_test.sv` | C2C link test: loopback, BER test, latency measurement |
| `hw/src/pcie_test/top_pcie_test.sv` | PCIe endpoint test: BAR access, DMA read/write |

### 10.3 Production Deployment Path

The `top_full_stack.sv` module in `hw/src/full_stack/` is the final integration test before production deployment. Its sequencer:

```
  S_INIT          -> Wait for HBM calibration + PCIe link up + C2C link up
  S_LOAD_WEIGHTS  -> DMA weights from host to HBM (or pre-loaded via JTAG)
  S_RUN_PIPELINE  -> Iterate layers LAYER_START..LAYER_END, run full transformer
  S_VERIFY        -> Compare output against golden C model values
  S_DONE / S_FAIL -> Report pass/fail
```

The same `chip_top.sv` RTL is used for both bring-up and production. The `IS_PCIE_MASTER` parameter controls whether PCIe IP is instantiated (master = Chip 0) or gated (slaves = Chips 1-3). Top-level modules differ only in which QSYS-generated IP (PCIe, HBM, C2C) is connected and which test sequencer wraps the chip core.

-------------------------------------------------------------------------------

## 11. Key Design Decisions

### 11.1 Why fp4 (E2M1) Weight Format?

**Decision:** Store all model weights at 4 bits (E2M1: 1 sign + 2 exponent + 1 mantissa), with per-128-group fp8 scaling factors. Exceptions: Router weights remain fp8 (sensitive to quantization error in softmax scoring). RMSNorm parameters remain fp16 (negligible size).

**Rationale:**
- DeepSeek V4 Pro was trained with QAT (Quantization-Aware Training) in fp4 during the last ~5% of pre-training. The weights are already optimized for fp4 inference.
- Effective bandwidth advantage: 920 GB/s HBM / 0.5 bytes per fp4 parameter = 1.84 trillion parameters/second. This is 10% higher than H100's 3.35 TB/s / 2 bytes per BF16 parameter = 1.68 trillion parameters/second.
- FPGA DSPs natively support fp4 x fp8 MAC at 2 operations/cycle. No decompression overhead -- weights arrive from SRAM at 4 bits and feed directly into the systolic array.
- Python simulation validates fp4 precision: cosine similarity >= 0.9955 vs fp8 baseline (group_size=16, smoothing alpha=1.0). Perplexity degradation < 0.5%.

### 11.2 Why Pipeline Parallelism (Not Tensor-Parallel Dominant)?

**Decision:** 32 chips form a 61-layer depth pipeline. Each chip computes 1-2 complete layers (MLA attention + MoE FFN). TP=2 is used only for attention weights within a layer.

**Rationale:**
- Pipeline parallelism minimizes inter-chip communication. The only data crossing chip boundaries is the hidden state (7,168 bytes FP8), forwarded approximately every 250ns. Total per-token pipeline forwarding bandwidth is approximately 43 MB/s at 200 tok/s -- trivial.
- MoE routing is naturally distributed across the pipeline. Each chip hosts 12 experts, and the router output determines which chips are contacted for expert computation. This scattering would be communication-heavy under pure TP.
- Pipeline concurrency is high: 32 tokens are simultaneously in-flight, each at a different layer. This naturally amortizes pipeline fill/drain overhead.
- GPU inference also uses pipeline parallelism for large models (e.g., DeepSeek V3/R1 uses PP=8-16 across nodes). This is a well-established pattern for MoE models.

**Trade-off:** Single-token latency is higher than TP-dominant designs (~1,510 us for 61 layers). For latency-sensitive applications, Pipeline Cloning (splitting 32 chips into 2 or 4 independent pipelines) reduces TTFT by trading off per-pipeline throughput.

### 11.3 Why C2C Ring (Not Crossbar)?

**Decision:** Within each 4-chip card, chips are connected by a dual ring (Ring A clockwise, Ring B with cross-connects), not a full crossbar.

**Rationale:**
- Crossbar: 3 bidirectional links per chip (full connectivity). Ring: 2 bidirectional links per chip. Saves 4 F-Tile lanes per chip (17% of the 48-lane F-Tile budget).
- Ring maximum hop count: 2 hops, approximately 100ns. Per-layer MoE compute time: approximately 3 us. Communication overhead of 100ns vs 3,000ns = 3.3% -- negligible.
- Ring B provides redundancy. Any single link failure -> all traffic routed on ring B. Zero frame loss.
- Saved F-Tile lanes are reserved for debug ILA (Signal Tap remote capture) and future expansion (e.g., QSFP-DD for multi-server scale-out in Phase 2).

### 11.4 Why SRAM-Resident Deterministic Weights?

**Decision:** Attention Q/KV/O projections, shared expert, router table, and RMSNorm parameters (collectively ~13.2 MB per layer) are double-buffered in on-chip M20K SRAM.

**Rationale:**
- These weights are needed every layer. Loading them from HBM every layer would consume approximately 9.3 MB/layer / 920 GB/s = 10.1 us per layer of HBM bandwidth, reducing effective throughput by approximately 40%.
- In 81.6% of layers (zero local expert hits), the only weights needed are these deterministic weights. With SRAM residency, HBM is completely idle during compute in these layers, and DSP utilization reaches nearly 100%.
- Without SRAM cache: weighted per-layer latency approximately 17.2 us. With SRAM cache: approximately 9.99 us. A 42% improvement.
- SRAM utilization is 75.6% (M20K) and 80% (MLAB) -- within the routable, timing-closable range for 450 MHz operation on Agilex 7 M-Series.
- The router table specifically must be fp8 (not fp4) to avoid functional correctness issues in expert selection. At 0.37 MB/layer, SRAM residency costs nothing in bandwidth.

### 11.5 Why Single Server (Not Multi-Node)?

**Decision:** The entire 32-chip cluster fits in one 4U server. No ToR switch, no QSFP-DD cables, no RoCE v2 fabric in Phase 1.

**Rationale:**
- Eliminates approximately 1.25M RMB in switch + cable + RoCE IP costs.
- PCIe 5.0 backplane P2P latency (~400ns) vs Ethernet RoCE (~2-5 us) = 4-10x faster.
- Single failure domain reduces operational complexity. No cross-server network partition risk.
- Phase 2 (multi-server scale-out via 200GbE RoCE v2) is planned but not required for the initial 32-chip deployment.
- Physical constraint: 8 FPGA cards (FHFL extended) fit in standard 4U GPU server chassis (e.g., Inspur NF5688M7, Lenovo SR670 V3).

### 11.6 Why No Hot Sparing (Chip Self-Healing Instead)?

**Decision:** All 32 chips are active. Failed chips are recovered by redistributing their experts to sibling chips on the same card.

**Rationale:**
- Each chip's HBM (32 GB) stores only approximately 0.7 GB of weights. Sibling chips have approximately 31 GB of headroom to absorb a failed chip's expert weights.
- Single chip failure: 3 sibling chips each take 4 additional experts. Recovery time < 100ms. Zero throughput degradation.
- Card-level failure (4 chips simultaneously): 7-card operation, approximately 12.5% throughput degradation. Manual replacement restores full throughput.
- Hot-spare cards would waste 12.5% of hardware budget for a failure mode with MTBF > 50,000 hours per chip.

### 11.7 Why FPGA (Not GPU, Not ASIC)?

**Decision:** FPGA prototype validation -> FPGA volume production + eASIC -> full-custom ASIC. This is a three-phase roadmap, not an either-or choice.

**Rationale:**
- FPGA reconfigurability allows architectural validation before committing to silicon. fp4+MLA inference has no prior silicon-level validation. Architecture changes cost a recompilation (8 hours, $0), not a mask revision ($10-20M).
- FPGA supply chain is diversified (Intel/Altera fabs + SK Hynix/Samsung HBM + mainland China PCB + global PCIe standard). Not subject to GPU compute export controls or CoWoS advanced packaging sanctions.
- RTL code is approximately 60% portable to ASIC. The fp4 MAC, MLA pipeline, KV cache manager, and MoE router are written in standard SystemVerilog with no FPGA-specific primitives in the core data path.
- FPGA B=1 decode utilization (~50%) is >10x higher than GPU (~2-5%). In the single-user, private-deployment scenarios this architecture targets, FPGA's DSP-to-HBM ratio naturally matches the workload better than GPU's massive-but-mostly-idle Tensor Cores.

-------------------------------------------------------------------------------

## Appendix A: File Reference Index

### Python Simulation (`scripts/`)

| File | Description |
|------|-------------|
| `scripts/fpga_arch/config.py` | HW constants (single source of truth) |
| `scripts/fpga_arch/chip.py` | Single FPGA chip model |
| `scripts/fpga_arch/cluster.py` | 32-chip cluster assembly |
| `scripts/fpga_arch/pipeline.py` | 10-stage pipeline engine |
| `scripts/fpga_arch/interconnect.py` | C2C ring + PCIe fabric models |
| `scripts/fpga_arch/expert_popularity.py` | Zipf-distributed expert sampling |
| `scripts/vllm_serve/scheduler.py` | Continuous batching scheduler |
| `scripts/vllm_serve/model_runner.py` | Batch-to-hardware bridge |
| `scripts/vllm_serve/kv_cache.py` | KV cache block manager |
| `scripts/vllm_serve/api_server.py` | API server + request generator |
| `scripts/run_serving.py` | Event-driven serving simulation |
| `scripts/ARCHITECTURE.txt` | Architecture diagram (this document's source) |

### RTL Source (`rtl/`)

| Directory | Key Files | Purpose |
|-----------|-----------|---------|
| `rtl/dsp/` | `fp4_gemm_engine.sv`, `fp4_systolic_2d.sv`, `fp4_systolic_cell.sv`, `fp4_mac.sv`, `fp4_scale_reader.sv`, `fp4_prefill_engine.sv` | fp4 DSP compute |
| `rtl/attention/` | `mla_attention_v2.sv`, `mla_qkv_proj.sv`, `mla_rope.sv`, `mla_kv_cache.sv` | MLA attention |
| `rtl/moe/` | `router_topk.sv`, `expert_ffn_engine_fp4_down.sv` | MoE routing + FFN |
| `rtl/layer/` | `full_transformer_layer.sv`, `layer_compute_engine.sv`, `mhc_mixer.sv` | Layer integration |
| `rtl/chip/` | `chip_top.sv`, `kv_dma_engine.sv`, `kv_dma_bridge.sv` | Chip-level + KV DMA |
| `rtl/activation/` | `rms_norm.sv`, `silu_q12_lut.sv`, `q12_to_fp8_e4m3.sv` | Activation functions |
| `rtl/head/` | `mtp_head.sv`, `mtp_verify.sv` | Multi-token prediction |
| `rtl/engram/` | `sram_cache.sv`, `hash_unit.sv`, `lookup_engine.sv` | Memory/cache infrastructure |
| `rtl/debug/` | `uart_debug.sv`, `dsp_stress_test.sv`, `hbm_bw_test.sv` | Debug + characterization |
| `rtl/sim/` | `tb_*.sv` (20+ testbenches) | Verification |

### Hardware Top-Level (`hw/src/`)

| File | Purpose |
|------|---------|
| `hw/src/top_master.sv` | Master FPGA (PCIe host interface) |
| `hw/src/top_slave.sv` | Slave FPGA (C2C compute pipeline) |
| `hw/src/full_stack/top_full_stack.sv` | Full integration test |
| `hw/src/bringup/top_bringup.sv` | Minimal bring-up |
| `hw/src/hbm_char/top_hbm_char.sv` | HBM characterization |
| `hw/src/dsp_char/top_dsp_char.sv` | DSP characterization |
| `hw/src/c2c_test/top_c2c_test.sv` | C2C link test |
| `hw/src/pcie_test/top_pcie_test.sv` | PCIe endpoint test |

### Documentation (`docs/`)

| File | Description |
|------|-------------|
| `docs/fpga_inference_cluster_proposal_en.md` | Full proposal document (v1.3) |
| `docs/eng/02_architecture_overview.md` | This document |

-------------------------------------------------------------------------------

## Appendix B: Quick Reference -- Key Constants

```
  CLUSTER:
    NUM_CARDS          = 8
    CHIPS_PER_CARD     = 4
    TOTAL_CHIPS        = 32
    NUM_LAYERS         = 61
    NUM_EXPERTS        = 384
    EXPERTS_PER_CHIP   = 12
    TOP_K_EXPERTS      = 6
    SHARED_EXPERT      = True

  CHIP (AGM 039-F):
    DSP_COUNT          = 12,300
    DSP_FREQ_MHZ       = 450
    DSP_MAC_PER_CYCLE  = 2 (fp4xfp8)
    DSP_TMACS          = 11.07
    HBM_SIZE_GB        = 32
    HBM_BW_GBPS        = 920
    SRAM_TOTAL_MB      = 32.5

  MODEL (DeepSeek V4 Pro):
    HIDDEN_SIZE        = 7168
    INTERMEDIATE_SIZE  = 3072
    NUM_ATTN_HEADS     = 128
    KV_LORA_RANK       = 512
    Q_LORA_RANK        = 1536
    O_LORA_RANK        = 1024
    QK_ROPE_HEAD_DIM   = 64
    QK_NOPE_HEAD_DIM   = 448
    V_HEAD_DIM         = 128
    MLA_KV_BYTES       = 576 (FP8 per token)

  INTERCONNECT:
    C2C_LINK_BW_GBPS   = 128 (per link, 4 lanes x 32 Gbps)
    C2C_HOP_LATENCY_NS = 50
    C2C_DISPATCH_NS    = 250
    PCIE_P2P_BW_GBPS   = 64
    PCIE_P2P_LATENCY_NS= 400

  PERFORMANCE:
    PIPELINE_TPS       = 17,445 (saturated batch)
    BATCH1_TPS         = 660
    K_PIPELINE         = 25.4
    TOKEN_LATENCY_US   = 1,510 (B=1, full 61 layers)
    PER_LAYER_US       = 24.7
    WEIGHT_GB_PER_CHIP = 0.7
```

-------------------------------------------------------------------------------

*End of Architecture Overview.*
