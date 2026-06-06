# DeepSeek V4 Pro — LLM Inference Hardware Solution

## FPGA Prototype Validation → ASIC Tape-Out Mass Production

> Version: v1.3 (2026/05)
> Status: RTL optimization complete, 2D systolic array validation passed, three-tier prefill architecture ready
>        Bring-up/Production code separated, Docker cloud build environment available
> Confidentiality: Internal
>
> Latest updates (§11.A.2, §13.4, §4.9.6):
> - Competitive narrative restructured: from "cheaper" to three orders-of-magnitude architectural advantages (effective bandwidth utilization 83x,
>   switch latency 1000x, KV address resolution 1000x), new §11.A.2 Orders-of-Magnitude Architectural Advantages section added
> - §11.A Step 7 economics restructured: $/token is a projection of architectural bandwidth efficiency, not a pricing strategy
> - ASIC endgame positioning restructured: architectural advantages solidified + economies of scale, not "equally fast but 75% cheaper"
> - Prior updates: Three-tier prefill architecture, P0/P1 code ready, CPU prefill audit,
>   Agent scenario analysis, BOM pricing correction, vLLM integration verification

---

## Table of Contents

1. [Strategic Positioning and Necessity](#1-strategic-positioning-and-necessity)
   - 1.4 [Research Direction Justification](#14-research-direction-justification)
2. [Architecture Overview](#2-architecture-overview)
3. [DeepSeek V4 Pro Architecture Parameters](#3-deepseek-v4-pro-architecture-parameters)
4. [Compute Allocation and Resource Accounting](#4-compute-allocation-and-resource-accounting)
   - 4.4.1 [SRAM Cache and HBM Bandwidth Analysis](#441-sram-cache-hierarchy--quantitative-analysis-and-routing-feasibility)
   - 4.5.1 [HBM Capacity Limit Analysis](#451-hbm-capacity-limit-analysis)
   - 4.6.1 [Architecture-Level Optimization Paths for Concurrency Ceiling](#461-architecture-level-optimization-paths-for-concurrency-ceiling)
     - 4.6.1.7 [End-to-End Validation (18 Configuration Matrix)](#4617-end-to-end-validation-18-configuration-matrix)
   - 4.7 [fp4 Precision Justification](#47-fp4-precision-justification)
   - 4.8 [Prefill Performance Analysis and Scheduling Strategy](#48-prefill-performance-analysis-and-scheduling-strategy)
   - 4.8.x [Chip 0 Prefill Entry Bottleneck Analysis](#48x-chip-0-prefill-entry-bottleneck-analysis)
   - 4.9 [Agent Scenario Adaptation Analysis](#49-agent-scenario-adaptation-analysis)
   - 4.10 [Embedding / lm_head Bottleneck Analysis](#410-embedding--lm_head-serial-bottleneck-analysis)
5. [RTL Module Partitioning](#5-rtl-module-partitioning)
   - 5.3 [Weight Conversion and Deployment Toolchain](#53-weight-conversion-and-deployment-toolchain)
6. [Network Topology and Communication Scheme](#6-network-topology-and-communication-scheme)
7. [Server Platform and Physical Form Factor](#7-server-platform-and-physical-form-factor)
   - 7.5 [Power Analysis and Cooling Solution](#75-power-analysis-and-cooling-solution)
8. [Software Ecosystem and Inference Service Layer](#8-software-ecosystem-and-inference-service-layer)
   - 8.0 [Software Stack "Starting from Scratch" Response](#80-software-stack-starting-from-scratch--explained-in-one-diagram)
   - 8.3 [Deployment and Operations Characteristics](#83-deployment-and-operations-characteristics)
   - 8.4 [Inference Service Feature Matrix](#84-inference-service-feature-matrix)
9. [Development Roadmap](#9-development-roadmap)
   - 9.2 [FPGA Validation Strategy and Development Cadence](#92-fpga-validation-strategy-and-development-cadence)
   - 9.3 [Development Board Empirical Validation Plan](#93-rebuttal-a-response-development-board-empirical-validation-plan)
10. [Cost Analysis](#10-cost-analysis)
   - 10.3 [Hardware Pricing and Gross Margin](#103-hardware-pricing-and-gross-margin)
   - 10.4 [R&D Investment (IP Assets)](#104-rd-investment-ip-assets)
   - 10.5 [Three-Tier Customer Delivery Pricing](#105-three-tier-customer-delivery-pricing)
   - 10.6 [Comparison and Conclusions](#106-comparison-and-conclusions)
   - 10.7 [Corrected Cost Baseline (Post §4.6.1+§4.8.x Optimizations)](#107-corrected-cost-baseline-post-461461--48x-optimizations)
11. [Competitive Analysis](#11-competitive-analysis)
   - 11.1 [Benchmarking Matrix (Two Phases)](#111-benchmarking-matrix-two-phases)
   - 11.3 [Ascend 910C In-Depth Comparison](#113-ascend-910c-in-depth-comparison-analysis)
   - 11.4 [Target Market Size (TAM) Estimation](#114-target-market-size-tam-estimation)
   - 11.5 [Ascend 950PR Comparison Analysis](#115-ascend-950pr-comparison-analysis)
   - 11.6 [Performance Data Methodology Notes](#116-performance-data-methodology-notes481-481--48x-post-optimization)
12. [Risk Assessment and Mitigation](#12-risk-assessment-and-mitigation)
13. [Endgame Roadmap: FPGA Validation → ASIC Tape-Out](#13-endgame-roadmap-fpga-validation--asic-tape-out)
13. [Endgame Roadmap: FPGA Validation → ASIC Tape-Out](#13-endgame-roadmap-fpga-validation--asic-tape-out)
14. [Appendix: Key Technical Parameters Quick Reference](#14-appendix-key-technical-parameters-quick-reference)

---

## 1. Strategic Positioning and Necessity

### 1.1 The Core Problem

China's large models (represented by the DeepSeek series) face a structural contradiction:

```
  Model capability globally leading (DeepSeek V4 Pro 1.6T MoE)
         ↕
  Inference deployment hardware constrained by supply restrictions
         ↕
  ┌─────────────────────────────────────────┐
  │ Available Hardware   Constraining Factor         │
  ├─────────────────────────────────────────┤
  │ NVIDIA H100/B200     U.S. export control (3A090) │
  │ AMD MI300X           Equivalent restrictions      │
  │ Huawei Ascend 910C   SMIC 7nm capacity + CoWoS   │
  │                      Advanced packaging sanctions + international blockade │
  │ Cambricon/Hygon/Biren Different combinations of supply/ecosystem issues │
  └─────────────────────────────────────────┘
```

### 1.2 The Unique Advantages of the FPGA Path

```
Intel Agilex 7 M supply chain dispersed globally:

  FPGA chip:         Intel/Altera Fab (U.S./Ireland/Israel)
  HBM2e Stack:       SK Hynix / Samsung (South Korea, dual-source)
  EMIB packaging:     Intel Malaysia / Vietnam
  PCB manufacturing:  Mainland China / Taiwan
  PCIe standard:      Globally universal, not restricted
  Quartus tools:      Globally available (non-weapons-grade software)

  Altera independent operation (2024):
    Spun off from Intel as independent company → financially independent, FPGA as sole business
    → Not dragged down by Intel foundry losses, PE invested, IPO in preparation
    → Agilex product line lifecycle 10-15 years (industry norm)
    → 32-unit order qualifies as routine customer, lead time 8-12 weeks

  Key characteristics:
  ┌────────────────────────────────────────────┐
  │ ✓ Not dependent on SMIC capacity                         │
  │ ✓ Not subject to GPU compute export controls (TPP far below threshold)     │
  │ ✓ Not subject to advanced packaging sanctions (EMIB is not CoWoS)           │
  │ ✓ HBM supply from South Korea (not exclusively U.S.-controlled)              │
  │ ✓ Deployment location unrestricted (can deploy in China, Middle East, Southeast Asia)     │
  │ ✓ Standard PCIe interface, compatible with any server             │
  │ ✓ Altera independently operated, not an Intel subsidiary         │
  │ ✓ RTL code ~60% portable to Xilinx/Achronix    │
  └────────────────────────────────────────────┘
```

### 1.3 Strategic Positioning: Three-Phase Roadmap (FPGA → eASIC → ASIC)

> **Phase 1 — FPGA Prototype Validation (Now):** Low-risk, low-NRE validation of fp4+MLA architecture, serving seed customers.
> **Phase 2 — FPGA Volume Production + eASIC Conversion:** FPGA volume production and delivery + eASIC structured cost reduction (same process, metal layers only).
> **Phase 3 — Full-Custom ASIC (Long-term):** After market validation of >1,000 units/year, HBM3@7nm ultimate cost reduction.

```
Three-Phase Business Logic:

  Phase 1: FPGA Prototype Validation (Month 1-12)
    → ¥6.75M RTL R&D (IP assets), 8-card × 4-chip validation rig
    → 3-5 seed customers, validate fp4+MLA on real workloads

  Phase 2a: FPGA Volume Delivery (Month 12+)
    → 10-100 FPGA cluster units, priced ¥2.8M→¥2.2M (at 100 units)
    → $5.9/M token, already superior to H100 ($12-20) and 950PR ($16-25)

  Phase 2b: eASIC Cost Reduction (Month 18-24, optional)
    → NRE $2-5M (metal-layer masks only, reuse HBM2e PHY + EMIB)
    → 4 FPGAs consolidated into 1 eASIC, priced $100-130K/unit
    → $3.5-4.5/M token, 30-40% further reduction vs. FPGA volume production

  Phase 3: Full-Custom ASIC (Long-term, post market validation)
    → HBM3@7nm, NRE $20-30M
    → Priced $70-80K/unit, $2.5-3.5/M token

  Core insight:
    FPGA validation → eASIC cost reduction → ASIC ultimate harvest.
    eASIC is the pragmatic bridge to avoid the HBM3 NRE quagmire — reuse validated HBM2e+EMIB,
    NRE is only 20% of full-custom ASIC, yet cost reduction already reaches 60%.
```

```
            Market Coverage (FPGA and ASIC both globally deployable):

              NVIDIA GPU    Ascend        This Solution (FPGA→ASIC)
            ┌───────────┐ ┌───────────┐ ┌───────────┐
  China domest. │ ✗ restricted │ ✓ salable     │ ✓ salable     │
  SE Asia       │ △ downgraded │ △ limited export │ ✓ salable     │
  Middle East   │ △ restricted │ ✗ export difficult │ ✓ salable     │
  Europe        │ ✓ salable    │ ✗ export difficult │ ✓ salable     │
  LatAm/Africa  │ ✓ salable    │ ✗ no channel  │ ✓ salable     │
            └───────────┘ └───────────┘ └───────────┘
```

### 1.4 Technical Roadmap Justification: Why FPGA→ASIC Two-Stage?

**Review Question: "Is the research direction correct? Why choose FPGA instead of directly going ASIC / GPU optimization / waiting for Chiplet?"**

Answer: It is not "FPGA or ASIC"; it is "validate with FPGA first, then tape out ASIC for volume production." Both are two stages of the same roadmap.

**I. Matrix Comparison of Four Alternative Paths**

```
┌────────────────────┬──────────────┬────────────┬──────────┬──────────┐
│                     │ FPGA→ASIC    │ Direct ASIC│ GPU Opt.   │ Wait Chiplet│
│                     │ (This Plan)  │ (In-House) │ (CUDA)    │          │
├────────────────────┼──────────────┼────────────┼──────────┼──────────┤
│ Phase 1 Investment  │ ~¥7M (FPGA)  │ ~¥50-200M  │ ~¥5M      │ ~¥3M     │
│ Time to Deployable  │ ~10 months   │ 18-24 months│ 6-12 months│ Uncertain│
│ Arch. Validation Risk│ Low (FPGA reconfig.)│ High (silicon-frozen)│ Low       │ High     │
│ Endgame HW Cost     │ ~$150-190K/unit│ ~$50K/unit│ ~$280K     │ Unknown   │
│ Native fp4          │ ✓            │ ✓ hardenable│ ✗         │ △        │
│ MLA Hardened        │ ✓            │ ✓ hardenable│ ✗         │ ✗        │
│ Globally Deployable │ ✓            │ ✓ (in-house)│ ✗ restricted│ ✓       │
│ Model Evolution Adapt.│ ✓ FPGA reconfig.│ ✗ frozen   │ ✓ flexible│ ✓ flexible│
│ Multi-Source Supply │ ✓            │ △ single    │ ✗ TSMC-only│ ✗ TSMC-only│
└────────────────────┴──────────────┴────────────┴──────────┴──────────┘

Key differences:
  Direct ASIC: Single tape-out bet → failure = $50-200M down the drain
  FPGA→ASIC:   FPGA runs first → validated with real data → tape-out risk extremely low
               RTL reuse rate >70% → ASIC design cycle shrinks from 18 months to 12 months
```

**II. Why Not Direct ASIC?**

```
Direct ASIC is infeasible at this stage:

1. Architectural risk not yet discharged:
   ● fp4+MLA inference architecture has never been validated at the silicon level
   ● DeepSeek V5/V6 may change MoE Top-K, MLA compression algorithm
   ● Direct tape-out → architecture change = $10-20M revision cost
   ● FPGA first: architecture change = recompile (8h, $0)

2. No workload data:
   ● ASIC design requires real workload data for microarchitecture optimization
   ● Without seed customers running → ASIC design relies on guesswork → performance and area not at optimal point
   ● FPGA runs 10 seed customer units first → accumulate data → guide ASIC design

3. Capital pacing:
   ● FPGA stage ¥6.75M → affordable
   ● Phase 3 seed customer revenue → can partially cover ASIC NRE
   ● Not a single bet of $50M+, but risk released in stages
```
   → The 10-month FPGA validation window fits precisely between "needed now" and "future uncertain"

3. Not mutually exclusive:
   ● FPGA RTL code ~60% directly usable for ASIC design
   ● If market validation succeeds (100+ units), FPGA→ASIC is the natural evolution
   ● If market validation fails, ¥7M R&D sunk vs. ¥200M ASIC NRE
   → FPGA is the "market-validation front-end" for ASIC, not an "either-or"

Academic analogy: FPGA inference != "abandoning ASIC",
            FPGA inference = "using reconfigurable hardware to validate architectural hypotheses, reducing pre-tape-out ASIC risk"
            Similar to Google using FPGAs to validate TPU architecture before tape-out.
```

**III. Why Not "Just Optimize GPU CUDA Kernels"?**

```
This is the most common objection. But it overlooks three critical facts:

1. GPUs are unavailable — this is not a cost problem, it is a zero-supply problem:
   ● This document is not comparing "who is better" against GPUs — GPUs are certainly more mature
   ● The scenario here is "GPU supply is zero; do you have an alternative?"
   ● Optimizing CUDA kernels presupposes a GPU to run on — target customers lack this prerequisite

2. fp4 is emulated on GPUs:
   ● NVIDIA has no fp4 Tensor Core (H100/B200 minimum granularity INT8/FP8)
   ● Running fp4 on GPU requires loading fp4 → decompress to FP8 → Tensor Core compute
     Decompression overhead ~10-15% bandwidth + latency
   ● FPGA computes natively in fp4 directly on the DSP chain, zero decompression
   → Even with a GPU, fp4 inference is not the GPU's optimal path

3. MLA is a software implementation on GPUs:
   ● GPU running MLA requires 4 CUDA kernels: Q projection → KV decompress →
     attention → O projection, multiple VRAM round-trips in between
   ● FPGA hardens MLA into a single pipeline, KV never leaves on-chip SRAM
   → This is not "optimizing kernels"; this is "eliminating data movement between kernels"
   → GPU's ceiling is constrained by HBM bandwidth; FPGA's ceiling is DSP compute throughput

Analogy: Discussing CUDA optimization under GPU supply restrictions
      is like discussing "where to buy discounted H100s" while on an embargo list —
      the problem is not optimization, it is access.
```

**IV. Why Not Wait for Chiplet / Compute-in-Memory / Optical Computing?**

```
These are technology directions worth watching, but their timelines and maturity are unsuitable for current decisions:

  Chiplet AI:
    ● UCIe standard still evolving; production-grade Chiplet AI chips (e.g., d-Matrix) sampling only in 2026
    ● Requires TSMC CoWoS or similar packaging — equally subject to advanced packaging sanctions
    → May mature in 3-5 years; cannot wait now

  Compute-in-Memory (PIM / CIM):
    ● Samsung HBM-PIM announced but no public purchasing channel
    ● Programming model entirely different from existing AI frameworks; adaptation cost unknown
    → Laboratory stage, unsuitable for scenarios requiring deployment within 2 years

  Optical Computing (Lightmatter / Lightelligence):
    ● So far only demonstrated matrix multiplication prototypes; far from full LLM inference
    ● Precision limitations (analog computing); precision loss similar to fp4 but no mature calibration scheme
    → 5-10 years before practical deployment

FPGA is the only option that "can be purchased today, deployed in 10 months, and is not restricted."
This is not dismissing cutting-edge directions; rather, FPGA happens to be the intersection of cutting-edge
(fp4+MLA hardening) and practical (procurable).
```

**V. Academic/Industrial Positioning of This Research Direction**

```
This proposal has independent research contributions at three levels:

1. System Architecture Level:
   ● First proposal of "MoE Layer-Wise Pipeline over Ethernet" for FPGA clusters
   ● Not "porting GPU architecture to FPGA," but leveraging FPGA reconfigurability
     to redesign the inference data path (Weight Stationary + Systolic Array)

2. Numerical Computing Level:
   ● Implementing native fp4 inference on FPGA (no publicly disclosed complete solution in industry)
   ● Validating end-to-end fp4 precision under MoE+MLA architecture (vs. BF16 baseline)
   → If validation passes, will establish FPGA's unique advantage in low-precision inference

3. Hardware-Algorithm Co-Design Level:
   ● MLA KV compression → directly mapped to FPGA matrix multiplication pipeline
   ● MoE Expert Routing → mapped to FPGA distributed routing logic
   ● Router fp8 exception → mapped to FPGA mixed-precision compute chain
   → This is not "optimizing known algorithms"; it is "redesigning the inference path from hardware characteristics"

Research significance: If successful, this is not an incremental "FPGA accelerates LLM" paper,
         but establishes a new inference paradigm —
         "a reconfigurable, exportable inference hardware route precisely adapted to a single model family."
         This paradigm has systemic value for China's AI industry under GPU embargo.
```

---

## 2. Architecture Overview

### 2.1 Two-Layer Physical Architecture

A single 4U server handles all inference, with no cross-machine networking required.

```
┌──────────────────────────────────────────────────────────────┐
│ [User Access Layer]                                           │
│   OpenAI REST API-compatible endpoint                        │
│   (Any framework supporting OpenAI client, zero-cost integration) │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│ [Inference Service Layer — Same 4U Server]                    │
│   Token encoding / sampling control / session management      │
│   Inference command orchestration / result assembly           │
│   PCIe 5.0 P2P direct to FPGA accelerator cards               │
└────────────────────────┬─────────────────────────────────────┘
                         │ PCIe 5.0 x16 (P2P)
┌────────────────────────▼─────────────────────────────────────┐
│ [FPGA Compute Engine — 8 Cards × 4 AGM 039/Card = 32 Chips]   │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Card 0  │  │  Card 1  │  │  Card 2  │  │  Card 3  │    │
│  │ 4×AGM039 │  │ 4×AGM039 │  │ 4×AGM039 │  │ 4×AGM039 │    │
│  │Layer 0-7 │  │Layer 8-14│  │Layer15-22│  │Layer23-29│    │
│  │+Embedding│  │          │  │          │  │          │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │              │              │              │         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Card 4  │  │  Card 5  │  │  Card 6  │  │  Card 7  │    │
│  │ 4×AGM039 │  │ 4×AGM039 │  │ 4×AGM039 │  │ 4×AGM039 │    │
│  │Layer30-37│  │Layer38-44│  │Layer45-52│  │Layer53-60│    │
│  │          │  │          │  │          │  │+lm_head  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘         │
│                     PCIe 5.0 Backplane                        │
│                  (Dual CPU Root Complex)                      │
│                                                              │
│   Inter-Card: PCIe 5.0 x16 P2P DMA                           │
│   Intra-Card: Chip-to-Chip SerDes Dual Ring (F-Tile, 4 lane/link) │
└──────────────────────────────────────────────────────────────┘
```

**Key changes (vs. old proposal):**
- ~~4 × 2U server × 8 cards~~ → **1 × 4U server × 8 cards × 4 chips**
- ~~F-Tile 200GbE + ToR Switch + RoCE v2~~ → **PCIe 5.0 backplane P2P + intra-card C2C SerDes**
- ~~AGM 032 (9,375 DSPs)~~ → **AGM 039 (12,300 DSPs)**, +31% compute per chip
- Total chip count unchanged (32), card count reduced from 32 to 8, servers from 4 to 1

### 2.2 Single-FPGA Internal RTL Module Block Diagram

```
External: PCIe 5.0 x16 CEM edge connector (only Card's Chip0 faces externally; Chip1/2/3 use intra-chip C2C)
        │
┌───────▼─────────────────────────────────────────────────┐
│  PCIe 5.0 EP Hard IP + P2P DMA Engine (Chip0 only)      │
│  · PCIe MWr/MRd generator                               │
│  · BAR4 64MB (mapped for 4 chips per card)              │
│  · C2C → PCIe proxy bridge                              │
└───────┬─────────────────────────────────────────────────┘
        │
┌───────▼──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│fp4 Systolic │ MLA  │ RoPE │RMSNorm│ KV   │MoE   │C2C   │
│ Array ×8    │Attn  │Hard  │ Hard  │Cache │Router│SerDes│
│(12,300 DSPs)│Pipe  │Unit  │ Unit  │Mgr   │Dispatch│Link │
│             │line  │      │       │(HBM) │      │×2    │
└───────┬──────┴──────┴──────┴──────┴──────┴──────┴──────┘
        │
┌───────▼─────────────────────────────────────────────────┐
│              HBM2e Controller (Avalon-MM)                 │
│              32 GB HBM2e @ ~920 GB/s                      │
│   ┌─────────────────────┬─────────────────────┐         │
│   │  Weight Area ≤24 GB │  Runtime Area ≤8 GB │         │
│   │  · 12 routed experts fp4  │  · KV Cache FP8      │         │
│   │  · 1 shared expert fp4    │  · Activation Buffer  │         │
│   │  · Attention weights      │  · C2C RX/TX Ring     │         │
│   │  · Router weights         │  · Micro-batch intermediate activations │         │
│   └─────────────────────┴─────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

**Chip0 vs. Chip1/2/3 differences:**
- Chip0: PCIe EP + P2P DMA Engine + C2C Proxy Bridge + full RTL modules
- Chip1/2/3: No PCIe logic; host interaction via C2C → Chip0 → PCIe
- All chips have C2C SerDes Link ×2 (Dual Ring A/B)
- Chip0's BAR4 maps the 4 chips' register space into a unified 64MB PCIe BAR

---

## 3. DeepSeek V4 Pro Architecture Parameters

> Source: DeepSeek V4 Pro open-source repository `config.json` (verified)

| Parameter | Value | Notes |
|------|-----|------|
| `hidden_size` | 7,168 | |
| `num_hidden_layers` | 61 | |
| `n_routed_experts` | 384 | |
| `n_shared_experts` | 1 | |
| `num_experts_per_tok` | 6 | top-6 routing |
| `moe_intermediate_size` | 3,072 | SwiGLU FFN |
| `num_attention_heads` | 128 | |
| `num_key_value_heads` | 1 | **MLA (Multi-head Latent Attention)** |
| `head_dim` | 512 | nope=448 + rope=64 |
| `q_lora_rank` | 1,536 | Q compression rank |
| `kv_lora_rank` | 512 | KV compression rank |
| `qk_rope_head_dim` | 64 | Decoupled RoPE |
| `o_lora_rank` | 1,024 | O compression rank |
| `o_groups` | 16 | |
| `expert_dtype` | `fp4` | E2M1 format |
| `activation_dtype` | FP8 (E4M3) | quantization_config |
| `vocab_size` | 129,280 | |
| `max_position_embeddings` | 1,048,576 | 1M context |
| `sliding_window` | 128 | Local attention window |
| `num_nextn_predict_layers` | 1 | MTP prediction layer |
| `routed_scaling_factor` | 2.5 | |
| `scoring_func` | `sqrtsoftplus` | |
| `topk_method` | `noaux_tc` | Token choice, no auxiliary loss |

### 3.1 Core MLA Characteristics

```
Standard MHA (e.g., LLaMA):
  Q/K/V: 128 heads × 128 dim each = 16,384 dim per head group
  KV Cache per token: 2 × 128 × 128 × FP16 = 32 KB

DeepSeek MLA:
  KV compressed to 1 latent vector (512 dim) + decoupled RoPE (64 dim)
  KV Cache per token: (512 + 64) × FP8 = 576 Bytes

  Compression ratio: 32 KB → 576 B = 56× compression
  This is the core foundation of DeepSeek's long-context capability
```

---
