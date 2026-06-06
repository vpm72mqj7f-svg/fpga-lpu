### 4.10 Embedding / lm_head Serial Bottleneck Analysis

Reviewer question: §4.1 places Embedding (1.85 GB) on Node 0 and lm_head (1.85 GB) on Node 3. These two largest single tensors fall outside the DSP systolic array's coverage — Embedding is a table lookup, and lm_head is a 129K-vocab dense matrix multiply. Could they become pipeline bottlenecks?

**4.10.1 Embedding: Negligible Overhead**

```
Embedding lookup (token_id → 7168-dim vector):

  Single token:
    HBM address = base + token_id × 7168 × 2B (fp16)
    Read: 7168 × 2B = 14 KB, one sequential burst
    HBM latency: ~100ns (tRC) + 14KB/920GB/s ≈ 15ns = 115ns

  Prefill 4K tokens:
    4K × 115ns ≈ 0.46 ms
    In practice: many duplicates among 4K tokens; duplicate tokens are cacheable
    0.46ms / 800ms TTFT = 0.06% — negligible

  Decode 1 token per step:
    115ns / 160μs per-token decode = 0.07% — negligible

Conclusion: Embedding is not a matrix multiply, but it does not need to be.
      Pure HBM burst read; overhead is negligible in all scenarios.
      DSP is not involved, nor does it need to be.
```

**4.10.2 lm_head: A Real Bottleneck Candidate**

```
lm_head = Linear(7168 → 129,280), 926M params = 1.85 GB fp16

  Node 3 single chip (TP=7):
    HBM:  1.85 GB / 7 / 920 GB/s = 287 μs
    DSP:  129,280 × 7,168 / 7 / 8.44T = 15.7 μs
    → lm_head per token: ~290 μs (HBM-bound)

Placed in pipeline:
  Node 3: [L45-60: ~150μs] + [lm_head: 290μs] = 440 μs/token
  Node 2: [L30-44: ~150μs]
  Node 1: [L15-29: ~150μs]
  Node 0: [L0-14:  ~150μs]

  → Node 3 is 290μs slower than other nodes
  → Pipeline throughput governed by slowest node: 1/440μs ≈ 2,270 tok/s
  → This is not "27 chips waiting idly"; pipeline stages are paced by the slowest stage
```

**4.10.3 Mitigation: lm_head Overlapped with Next Token Pipeline**

```
  t=0:    Token 1 L45-60 @ Node 3 (150μs)
  t=150:  Token 1 lm_head @ Node 3 (290μs)
          Token 2 L0-14 @ Node 0 (150μs)  ← already started, not waiting!
  t=300:  Token 2 L15-29 @ Node 1
  t=440:  Token 1 lm_head complete, first token visible
          Token 2 L30-44 @ Node 2
  t=590:  Token 2 L45-60 @ Node 3
          Token 2 lm_head @ Node 3 (overlapped with Token 3 L0-14)
  ...

Pipeline bubbles:
  Node 3: 440μs/token, utilization 150/440 = 34%
  Node 0-2: waiting for Node 3 to complete, utilization ~34%

Compare to GPU 8×H100 pipeline parallelism:
  Same bubbles (uneven stages); this is not an FPGA-specific problem.
```

**4.10.4 MTP (Multi-Token Prediction): An Unexpected Savior for lm_head**

```
V4 Pro MTP predicts 2-4 subsequent tokens in one shot.

Under MTP, lm_head goes from vector-matrix → matrix-matrix:

  B=4 (4 hidden states):
    DSP:  4 × 129,280 × 7,168 / 7 / 8.44T = 62.8 μs
    HBM:  1.85 GB / 7 / 920 GB/s = 287 μs  (weights read once, shared across 4 ways)
    Total latency: ~350 μs (only 60μs more than B=1)

  Per-token amortized: 350/4 = 87.5 μs (vs 290 μs for B=1)
  → MTP reduces lm_head per-token cost to ~30%
  → Pipeline bottleneck drops from 440 → 238 μs/token
  → Throughput rises from ~2,270 → ~4,200 tok/s

MTP does not aggravate the lm_head bottleneck — it alleviates it.
Large-batch lm_head is exactly the scenario where DSP utilization can improve.
```

**4.10.5 More Aggressive Option: Distributed lm_head (Phase 3+)**

```
Split lm_head across all 30 chips:

  vocab 129,280 split 30 ways: ~4,309 tokens per chip (27 MB fp16)
  Each chip runs local 4,309 × 7168 matmul → top-k argmax
  → 30-chip All-Reduce to consolidate top-k results

  Latency: 4,309 × 7,168 / 8.44T = 3.7 μs (DSP)
           + HBM 27 MB / 920 GB/s = 29 μs
           + All-Reduce ~10 μs
           ≈ 43 μs

  → 6.7× faster than Node 3 exclusive (290μs)
  → Requires Phase 3 implementation (extra RTL + custom All-Reduce)

  Cost: 27 MB extra lm_head weights per chip (HBM headroom is ample)
```

**4.10.6 Candid Gap Assessment**

```
┌──────────────────────┬──────────────────┬──────────────────┐
│ lm_head (decode B=1) │ 30 FPGA          │ 8×H100           │
├──────────────────────┼──────────────────┼──────────────────┤
│ Bottleneck            │ HBM (Node 3 only)│ HBM (but GPU larger)│
│ Per-token latency     │ ~290 μs          │ ~550 μs (HBM)     │
│ Mitigation            │ MTP batch → 88μs │ larger batch → lower│
│ Distributed (all-chip)│ Phase 3, ~43μs   │ native TP         │
└──────────────────────┴──────────────────┴──────────────────┘

GPU's advantage in lm_head comes from larger HBM bandwidth + native TP.
H100 8-card total HBM bandwidth 26.8 TB/s vs FPGA 30-chip 27.6 TB/s —
but GPU weights need only one copy (8 chips share HBM), while FPGA stores its own per chip.
lm_head 1.85 GB on GPU at TP=8 is 0.23 GB per chip;
HBM read takes only 0.23/3.35 = 69 μs, faster than FPGA's 287 μs.

Candid admission: lm_head in non-distributed mode is an FPGA weakness.
But under MTP (batch≥4), per-token amortized cost already approaches GPU.
Distributed lm_head (Phase 3) can fully eliminate this disadvantage.
```

---

### 5.0 IP Reuse Strategy and Workload Accounting

Reviewer questioned whether 13 modules can be delivered with only 50 person-months. Key response: **most infrastructure comes from Intel hard IP or external procurement; only inference-specific data paths require in-house development.**

```
┌───────────────────────────────┬──────────────────┬──────────┬──────────┐
│ Module                        │ Implementation   │ Source   │ In-house PM│
├───────────────────────────────┼──────────────────┼──────────┼──────────┤
│ PCIe 5.0 x16 Endpoint         │ R-Tile Hard IP   │ Intel    │ 0        │
│ PCIe DMA (Scatter-Gather)     │ Intel DMA IP     │ Intel    │ 0.5      │
│ HBM2e Controller (2048-bit)   │ Avalon-MM HBM IP │ Intel    │ 1.0      │
│ F-Tile 200GbE MAC/PCS/FEC     │ F-Tile Hard IP   │ Intel    │ 0        │
│ RoCE v2 RDMA + DCQCN + PFC    │ Outsourced       │ 3rd-party│ 0 (¥1M)  │
│ Inference Payload Codec (RDMA)│ In-house RTL     │ —        │ 1.0      │
│ fp4×fp8 Systolic Array (×8)   │ In-house RTL     │ —        │ 10.0     │
│ MLA Attention Pipeline        │ In-house RTL     │ —        │ 12.0     │
│ Decoupled RoPE Unit           │ In-house RTL     │ —        │ 1.0      │
│ MoE Router Gating + Dispatch  │ In-house RTL     │ —        │ 4.0      │
│ Shared Expert Unit            │ In-house RTL     │ —        │ 1.0      │
│ KV Cache Manager (HW address) │ In-house RTL     │ —        │ 6.0      │
│ Chip2Chip Router (RoCE v2)    │ In-house RTL     │ —        │ 3.0      │
│ RMSNorm Unit                  │ In-house RTL     │ —        │ 0.5      │
│ Inference Control FSM         │ In-house RTL     │ —        │ 2.0      │
│ ILA / Perf Counters / Debug   │ Intel Debug IP   │ Intel    │ 1.0      │
│ Token Embedding LUT           │ In-house RTL     │ —        │ 1.0      │
│ lm_head + MTP                 │ In-house RTL     │ —        │ 2.0      │
├───────────────────────────────┼──────────────────┼──────────┼──────────┤
│ Subtotal                      │                  │          │ 46.0     │
│ Integration + System Bring-up │                  │          │ 8.0      │
│ (20% margin)                  │                  │          │          │
│ TOTAL                         │                  │          │ 54.0 PM  │
└───────────────────────────────┴──────────────────┴──────────┴──────────┘

Key reuse:
  ● Intel Hard IP (zero RTL): PCIe EP, F-Tile MAC/PCS/FEC — hardened in silicon, config-only
  ● Intel IP (minor customization): HBM controller, DMA Engine — Avalon-MM standard interface, minimal adaptation
  ● External procurement: RoCE v2 protocol stack — mature FPGA IP market, ~¥1M, saves 36-54 PM in-house
  ● Parametric replication: 8 Systolic Arrays share the same RTL, only top-level wiring parameters change
  ● Shared Expert reuse: directly instantiate Systolic Array (only 1 lane needed)
```

**Alignment with staffing budget:**

The proposal staffing budget is 5 FPGA RTL × 10 months = 50 person-months. In-house module accounting is 55 PM (including integration margin). Deviation is within 10%, with the following flexibility:

- RoCE v2 outsourcing releases substantial risk (in-house development would have consumed 36+ PM)
- Systolic Array and MLA Pipeline are the only two >10 PM modules; the rest are 0.5-6 PM
- If team FPGA experience is sufficient (≥5 years), compressible to 50 PM; otherwise adjust to 6 people × 12 months = 72 PM (`+¥2.1M`)

### 5.1 Top-Level Module Hierarchy

```
ds_v4_fpga_top
│
├── pcie_cxl_ep_wrapper        # R-Tile PCIe 5.0 x16 Endpoint + CXL
│   ├── r_tile_hard_ip          # Intel PCIe 5.0 Hard IP (zero LUT)
│   ├── tlp_to_rdma_cmd         # TLP → inference payload (RDMA payload)
│   └── rdma_cmd_to_tlp         # inference completion → TLP
│
├── f_tile_eth_wrapper          # F-Tile 200GbE Hard MAC + custom RoCE
│   ├── f_tile_hard_mac         # Intel 200G/400G Ethernet Hard IP
│   ├── roce_v2_subsystem       # RoCE v2 RDMA (outsourced IP ¥1M)
│   ├── rdma_payload_codec       # inference payload encode/decode (on RDMA)
│   └── credit_flow_ctrl        # credit-based backpressure flow control
│
├── inference_ctrl_fsm          # global inference pipeline state machine
│   ├── layer_counter           # layer count 0~60
│   ├── pipeline_handshake      # cross-stage handshake
│   └── prefill_decode_mode     # Prefill/Decode mode switching
│
├── mla_attention_pipeline      # MLA Attention full pipeline
│   ├── q_compress_unit         # Q: 7168 → 1536 (LoRA)
│   ├── kv_compress_unit        # KV: 7168 → 576 (512+64)
│   ├── qk_dot_product_unit     # Q·K^T (128 heads, nope×c_KV + rope×k_R)
│   ├── online_softmax_unit     # Online Safe Softmax (FP32 acc)
│   ├── av_dot_product_unit     # A·V (nope against c_KV latent)
│   └── o_decompress_unit       # O: 128×512 → 1024 → 7168 (LoRA)
│
├── rope_hardware_unit          # Decoupled RoPE (64-dim rope part only)
│
├── moe_expert_core             # fp4×fp8 mixed-precision MoE inference core
│   ├── systolic_array_128x128  # 8 parallel systolic arrays
│   ├── fp4_multiplier_unit     # fp4 × fp8 multiplier (200 LUTs)
│   ├── swiglu_hard_unit        # SiLU + element-wise multiply
│   ├── router_gating_unit      # Hash routing + top-6 selection
│   └── expert_dispatch_unit    # Expert dispatch to target FPGA
│
├── shared_expert_unit          # Shared expert FFN
│
├── kv_cache_manager            # KV Cache hardware management
│   ├── kv_addr_generator       # {session, layer, seq} → HBM addr
│   ├── sliding_window_ctrl     # sliding_window=128 window management
│   └── kv_fp8_compress_unit    # FP8 quantize/dequantize
│
├── hbm_memory_controller       # HBM2e controller (2048-bit interface)
│
├── chip2chip_router            # inter-chip communication engine (RoCE v2)
│   ├── all2all_scheduler       # MoE All-to-All scheduler
│   └── roce_qp_ctrl            # RoCE QP management / flow control
│
├── rms_norm_unit               # RMSNorm (eps=1e-6)
│
├── token_embed_lut             # Token Embedding LUT (Node 0 only)
├── lm_head_unit                # lm_head output projection (Node 3 only)
├── mtp_layer_unit              # MTP prediction layer (Node 3, after Layer 60)
│
└── debug_monitor               # on-chip ILA + performance counters
```

### 5.2 Key Sub-Module Detailed Design

#### 5.2.1 `fp4_multiplier_unit`

```
fp4 (E2M1) × fp8 (E4M3) multiplier:

  input:  w_fp4[3:0] = {sign, exp[0], mant[1:0]}
          a_fp8[7:0]  = {sign, exp[3:0], mant[2:0]}

  Implementation: LUT-based
    fp4 has only 16 possible values (15 valid)
    For each fp4 value, precompute 8 FP8 mantissa offsets
    → 1 BRAM + 8:1 MUX + exponent adder → 2 cycles completion

  Resource: ~200 LUTs + 1 BRAM (36Kb) per multiplier
  Each systolic_array_128x128 requires 16,384 multipliers
  → 16,384 × 200 LUTs ≈ 3.3M LUTs

  But can be reused: not all 16K multipliers operate simultaneously
  Actual 8 × 128×128 systolic arrays = 131,072 multiplier instances
  → This far exceeds Agilex 7 M LUT count

  Correction: implement with DSP
  Each DSP (with AI Tensor Block) does 2× fp4×fp8 MAC
  → 9,375 DSPs × 2 MAC/DSP = 18,750 MAC/cycle
  → At 450 MHz → 8.44 TMACs/s
```

#### 5.2.2 `mla_attention_pipeline` Pipeline Design

```
Attention stage pipeline (per layer, per token):

  Stage 0: Q compress      (7168×1536)  HBM:6μs + DSP:1.4μs  → ~6μs
  Stage 1: KV compress     (7168×576)   HBM:2.3μs + DSP:0.6μs → ~2.3μs
  Stage 2: Q·K^T           128-head parallel     DSP:3.8μs     → ~3.8μs
  Stage 3: Softmax         hardened              0.2μs         → ~0.2μs
  Stage 4: A·V             128-head parallel     DSP:3.7μs     → ~3.7μs
  Stage 5: O decompress    LoRA ×2               HBM:8.7μs + DSP:9.3μs → ~9.3μs
  ──────────────────────────────────────────────────────────
  Critical path:           Stage 0 (6μs) and Stage 5 (9.3μs) are HBM-bound
  Total latency:           ~25μs (serial) or ~15μs (partially overlapped)

MoE stage pipeline:
  Stage 6: Gating          Hash + top-6           ~0.5μs
  Stage 7: Dispatch        All-to-All send        ~3μs (cross-node RDMA)
  Stage 8: Expert FFN      66M MAC/expert         ~40μs (including HBM weight load)
  Stage 9: Combine         weighted result merge  ~2μs
  ──────────────────────────────────────────────
  Critical path:           Stage 8 (40μs) includes HBM weight loading
  Total per MoE layer:     ~65μs
```

### 5.3 Weight Conversion and Deployment Toolchain (Weight Layout Compiler)

Reviewer question: How does a PyTorch checkpoint become bitstreams + weight files on 30 FPGAs? How are weights rearranged when RTL changes? How is per-layer mixed precision determined? Does a model upgrade require re-synthesis? This section addresses the complete model-to-deployment toolchain.

**5.3.1 Toolchain Overview**

```
HuggingFace safetensors (fp8/bf16)
        │
        ▼
┌─────────────────────────────┐
│  fpgalpu-convert             │  ← Python, ~2000 lines
│  ┌─ Step 1: Parse + model    │
│  │          graph match      │
│  ├─ Step 2: fp4 quantization │
│  ├─ Step 3: Distribute to 30 │
│  │          chips            │
│  └─ Step 4: Generate HBM     │
│             bitstream        │
└─────────────┬───────────────┘
              │ 30 weight binary files
              ▼
┌─────────────────────────────┐
│  Weight Layout Compiler      │  ← Python, ~3000 lines
│  ┌─ Weight tiling (systolic) │
│  ├─ HBM bank interleaving    │
│  ├─ Address map generation   │
│  └─ Mixed-precision config   │
└─────────────┬───────────────┘
              │ weight binary + address map header
              ▼
┌─────────────────────────────┐
│  PCIe DMA Loader             │  ← C, ~500 lines
│  Load to each FPGA HBM at    │
│  initialization              │
└─────────────────────────────┘
```

**5.3.2 Steps 1-2: Model Parsing and fp4 Quantization**

```
Input: HuggingFace safetensors (standard format, DeepSeek official release)

Parsing:
  Extract named_parameters → {layer_id, param_type, shape}
  Cross-validate with config.json: n_layers=61, n_experts=384,
  n_heads=128, dim=7168, moe_intermediate=3072

fp4 Quantization (E2M1 + per-128-group scale):
  for each weight_matrix in [Attn_Q, Attn_KV, Attn_O, Expert_gate,
                              Expert_up, Expert_down, Shared_expert]:
    weight_2d = reshape(weight, (-1, 128))
    for each group of 128:
      scale = max(abs(group)) / 6.0           # E2M1 max = 6.0
      weight_fp4 = round(clamp(group/scale, -6, 6))
      scale_fp8 = float_to_e4m3(scale)

  After quantization: weight_fp4 (4-bit) + scale_fp8 (8-bit per 128)
  Effective bit-width: 4 + 8/128 = 4.0625 bits/weight (vs theoretical 4-bit)

  Router weights: skip quantization, keep FP8 (see §4.7.3)
```

**5.3.3 Step 3: Weight Distribution — Weight Layout Compiler (WLC)**

```
WLC Input:
  model_config:     layers, experts, heads, dim
  hardware_config:  systolic_K, systolic_N, HBM_bank_count,
                    HBM_bank_width, tp_size
  partition_config: per-FPGA layer_range, expert_range, head_range

WLC Core Logic:

  ① Systolic Tiling
     Slice weight matrices into tiles consumable by systolic array:
       e.g., systolic 128×128, Expert gate (7168×3072):
       → K direction 7168/128 = 56 tiles
       → N direction 3072/128 = 24 tiles
       → 56×24 = 1344 tiles, each 128×128×4b = 8 KB

  ② HBM Bank Interleaving
     Distribute tiles of the same row across different HBM banks (avoid bank conflict):
       24 N-tiles round-robin allocated to 32 HBM pseudo-channels
       → Up to 32 tiles readable in parallel per cycle

  ③ Address Map Table Generation
     Output per FPGA:
       ┌────────────────────────────────────┐
       │ Layer 00: Expert 003 gate @ 0x0000 │
       │ Layer 00: Expert 003 up   @ 0x0800 │
       │ Layer 00: Expert 003 down @ 0x1000 │
       │ ...                                │
       │ Layer 00: Shared Expert   @ 0x4000 │
       │ Layer 01: ...                      │
       └────────────────────────────────────┘
     This table is written into both the weight binary header and the FPGA RTL address lookup table
```

**5.3.4 Automated Per-Layer Mixed-Precision Determination**

```
Workflow (completed in Phase 1):

  ① 1-chip FPGA runs full-layer inference (1 chip per parallel group)
     → output per-layer activation (FP16)

  ② PyTorch fp8 reference runs the same layer
     → output reference activation (FP32)

  ③ Automatic comparison:
     np.dot(act_fpga, act_ref) /
     (norm(act_fpga) * norm(act_ref))  → cosine_similarity

     mean(|act_fpga - act_ref|²) /
     mean(|act_ref|²)                   → L2_relative_error

  ④ Decision rule:
     cosine_sim < 0.995  OR  L2_err > 1%  → mark "sensitive layer"
     → write to mixed_precision_config.yaml

  ⑤ WLC rerun:
     Sensitive layers → fp8 weights (1 byte/weight)
     Others           → fp4 weights (0.5 byte/weight + scale)
     HBM address layout unchanged (only weight data grows)

  Full 61-layer model profiling: ~2 minutes per chip, fully automated.
```

**5.3.5 Model Upgrade Adaptability — What Requires Re-Synthesis, What Does Not**

```
┌───────────────────────┬──────────────┬──────────┬────────────────┐
│ Model Change           │ Weight Update│ RTL Impact│ Deployment Cycle│
├───────────────────────┼──────────────┼──────────┼────────────────┤
│ Expert 384→512         │ WLC rerun    │ None     │ 1 hour          │
│ Layers 61→80           │ WLC rerun    │ None     │ 1 hour          │
│ dim 7168→8192          │ WLC rerun    │ None     │ 1 hour          │
│ hidden 3072→4096       │ WLC rerun    │ None     │ 1 hour          │
│ Heads 128→96 (TP fixed)│ WLC rerun    │ None     │ 1 hour          │
│ fp4→fp6 precision      │ WLC rerun    │ Change DSP│ 1-2 weeks       │
│                        │              │ MAC mode │                │
│ MLA→Standard MHA       │ WLC rerun    │ RTL rewrite│ 3-6 PM          │
│ MoE→Dense              │ N/A          │ Arch redo │ infeasible      │
└───────────────────────┴──────────────┴──────────┴────────────────┘

Key design philosophy:
  Expert count, layer count, dim, and heads in RTL are all Verilog parameters —
  changing header constants suffices; no re-synthesis triggered.
  Only structural changes to the attention algorithm (MLA) or MoE routing
  require RTL modification and Quartus rerun.

Quartus synthesis baseline:
  Full Agilex 7 M design (~300K LUT scale): 4-8 hours (64-core Linux)
  Parameter-only change without re-synthesis: no Quartus needed
  Incremental compilation (minor changes): 30-60 minutes
```

**5.3.6 Comparison with TensorRT-LLM — Honest Gaps**

```
┌──────────────────────┬──────────────────┬──────────────────┐
│                       │ NVIDIA TensorRT   │ Our FPGA Toolchain │
├──────────────────────┼──────────────────┼──────────────────┤
│ Model Import          │ 1 command from HF │ 1 command (Python)│
│ Quantization (PTQ/QAT)│ Built-in, auto    │ Built-in, auto    │
│ Per-layer precision   │ Automatic         │ Automatic (Ph 1)  │
│ profiling             │                   │                   │
│ Graph opt / op fusion │ Mature (100+ pass)│ N/A               │
│                       │                   │ (HW fusion native)│
│ Weight split (TP/PP)  │ Automatic         │ WLC automatic     │
│ Deploy                │ build → run       │ convert+load → run│
├──────────────────────┼──────────────────┼──────────────────┤
│ Model upgrade (params)│ Rerun build       │ Rerun WLC (1h)    │
│ Model upgrade (arch)  │ Wait for framework│ RTL rewrite+resyn │
├──────────────────────┼──────────────────┼──────────────────┤
│ Ecosystem maturity    │ ★★★★★             │ ★★★ (Phase 2)     │
│ Maintenance staffing  │ NVIDIA 100+ team  │ 1-2 people + WLC │
│ Community contribution│ Global developers │ In-house, closed  │
└──────────────────────┴──────────────────┴──────────────────┘

Honest gaps:
  TensorRT-LLM's maturity is not catchable, nor does it need to be caught.
  The FPGA toolchain does something simpler:
    → No need to handle 100+ CUDA kernel variants
    → No need for graph-level IR optimization (hardware data path is already fixed)
    → Only need to generate the correct weight layout for one fixed systolic array

  The core gap is rapid support for new model architectures — NVIDIA has an ecosystem,
  we have the Quartus synthesis cycle. But for enterprise private deployment:
    Set up once, run stably, no need to chase version upgrades.
```

---

## 6. Network Topology and Communication Scheme

### 6.1 Design Principles

**Within a single server — no external network needed.** All 8 cards are plugged into one 4U server. Four chips per card are interconnected via C2C SerDes; cards are interconnected via PCIe 5.0 backplane. No ToR switch, no RoCE IP purchase, no QSFP-DD cages or DAC cables.

```
Why PCIe backplane + C2C replaces Ethernet + ToR:

  Physical:
    8 cards all plugged into the same server backplane → PCIe backplane is a free switch
    4 chips per card on the same PCB → F-Tile SerDes is a free inter-chip bus

  Economic:
    Saved: 2× ToR Switch (¥200K) + RoCE v2 IP (¥1M) + QSFP-DD cages (¥32K) + DAC cables (¥20K)
    Total savings: ~¥1.25M

  Performance:
    PCIe 5.0 x16 P2P latency ~500 ns  vs  Ethernet RoCE ~2-5 μs  →  4-10× faster
    C2C SerDes latency ~50 ns/hop       vs  cross-card routing hop  →  negligible

  Simplification:
    Single protocol stack (PCIe TLP + C2C frame), no Ethernet/IP/UDP/RoCE five layers
    No congestion control (dedicated physical links, no contention)
    No ToR failure domain, no MLAG configuration
```

### 6.2 Intra-Card C2C Topology: Dual Ring

The AGM 039 R47A package contains 3× F-Tile. Inter-chip C2C uses F-Tile raw transceivers (NRZ 32 Gbps, eliminating PAM4 DSP overhead), each link 4-lane bonded.

```
Single-card 4-chip Dual Ring (redundant):

         ┌──────────────────────────────────┐
         │            Ring A                 │
         │   Chip0 ←──────────→ Chip1       │
         │     ↕                    ↕        │
         │   Chip2 ←──────────→ Chip3       │
         └──────────────────────────────────┘

         ┌──────────────────────────────────┐
         │            Ring B (redundant)     │
         │   Chip0 ←──────────→ Chip2       │
         │     ↕                    ↕        │
         │   Chip1 ←──────────→ Chip3       │
         └──────────────────────────────────┘

Per link: 4 lane × 32 Gbps NRZ = 128 Gbps unidirectional (256 Gbps bidirectional)
Per chip usage: 2 link × 4 lane = 8 lane (17% of F-Tile 48-lane total)
Single hop: ~50 ns (SerDes latency + < 200mm PCB trace)
Longest: 2 hops (Chip1 → Chip0 → Chip3) ≈ 100 ns

Why Ring over Mesh:
  - Ring: 2 link/chip, Mesh: 3 link/chip → saves 4 lane/chip
  - 2 hops 100 ns vs MoE single layer ~3 μs → 30× margin, two hops fully negligible
  - Ring B redundancy: Ring A link fails → auto switch to Ring B, zero frame loss
  - Saved F-Tile lanes reserved for debug ILA / future expansion
```

**Per-chip F-Tile usage:**

| Use | Lane Count | Notes |
|------|---------|------|
| Ring A link (C2C) | 4 lane TX + 4 lane RX | 32 Gbps NRZ |
| Ring B link (C2C) | 4 lane TX + 4 lane RX | Redundant link |
| Reserved (debug ILA) | 8 lane | Signal Tap remote capture |
| Unused | 28 lane | Future expansion |

### 6.3 C2C Protocol Layers

```
┌────────────────────────────────────────────┐
│ Transport Layer                             │
│  · Message type routing (MoE / Pipeline /   │
│    PCIe_Proxy)                              │
│  · 5-bit global chip addressing {CardID,    │
│    ChipID}                                  │
│  · Multi-VC multiplexing (Data/Credit/Mgmt) │
├────────────────────────────────────────────┤
│ Link Layer                                   │
│  · Frame delimiting & scrambling (64b/66b) │
│  · Credit-based flow control (per VC)       │
│  · CRC32 error detection + SeqNum + timeout │
│    retransmission                           │
│  · Lane alignment & Deskew (multi-lane)     │
├────────────────────────────────────────────┤
│ Physical Layer                               │
│  · F-Tile Transceiver (32 Gbps NRZ)         │
│  · 4 lane bonded × bidirectional            │
│  · AC coupled, on-chip termination           │
└────────────────────────────────────────────┘
```

**6.3.1 Frame Format**

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
┌───────────────────────────────┬───┬───┬───┬───┬───┬───────────────┐
│  SOP (8B: 0xFB_C2C_FRAME)     │Ver│Typ│Pri│ VC│HdrLen│           │
│                               │ 2b│ 4b│ 2b│ 2b│ 4b    │           │
├───────────────────────────────┴───┴───┴───┴───┼───┴───────────────┤
│  SrcChip[4:0]  │ DstChip[4:0]  │  SeqNum[7:0]   │  FrameLen[11:0]  │
├───────────────────────────────────────────────────────────────────┤
│  Header CRC16 (bytes 8-15)                                         │
├───────────────────────────────────────────────────────────────────┤
│  Payload (0-4088 Bytes, 8B aligned)                                │
├───────────────────────────────────────────────────────────────────┤
│  CRC32 (over SOP through payload end)                              │
├───────────────────────────────────────────────────────────────────┤
│  EOP (4B: 0xE0F_END)                                               │
└───────────────────────────────────────────────────────────────────┘

Type (4b):
  0x1 = MoE_Dispatch     [activation vector 7168B FP8]
  0x2 = MoE_Reduce       [expert output 7168B FP8]
  0x3 = Pipeline_Fwd     [hidden_state 7168B FP8, cross-layer forwarding]
  0x4 = PCIe_Proxy       [Host ↔ non-Chip0 MMIO/DMA forwarding]
  0x5 = Credit_Update    [flow control credit return, 0 payload]
  0x6 = Weight_Broadcast [weight loading]
  0x7 = Heartbeat        [link keep-alive probe, 0 payload]

Priority (2b): 0=Credit/Mgmt, 1=Pipeline, 2=MoE, 3=PCIe_Proxy
VC (2b):       0=Control, 1=Data_HP, 2=Data_Bulk, 3=Management

Single frame max payload: 4088 B
MoE Dispatch (7168B): split into 2 frames (4088 + 3080)
Frame overhead: 16B header + 4B CRC + 4B EOP = 24B
Efficiency: 7168 / (7168 + 2×24) = 99.3%
```

**6.3.2 Credit-Based Flow Control**

```
Initialization:
  RX power-up → send CREDIT_INIT frame:
    {VC0: 64, VC1: 256, VC2: 128, VC3: 32}
  Each credit = 1 frame (max 4096 B)

Transmit:
  TX decrements credit[VC] per frame sent
  credit[VC] == 0 → stall, wait for CREDIT_UPDATE

Return:
  RX consumes frame → send CREDIT_UPDATE {VC, returned_credits: N}
  TX credit[VC] += N

RX Buffer (per SerDes port, 2 ports per chip):
  VC0 Control:     8 KB   (32 frames × 256B)
  VC1 Data_HP:   128 KB   (MoE, low latency)
  VC2 Data_Bulk:  64 KB   (Pipeline, bulk)
  VC3 Mgmt:       16 KB   (Heartbeat, weight)
  Total: 216 KB/port, 432 KB/chip
  Fraction of M20K total: 432 KB / 46 MB ≈ 0.9%
```

**6.3.3 Routing Table**

```
Global chip address: {CardID[2:0], ChipID[1:0]} = 5-bit, 0-31

Each chip maintains 8-entry routing table:
  ┌──────────────┬──────────────────┬──────────┐
  │ CardID       │ NextHop           │ Egress   │
  ├──────────────┼──────────────────┼──────────┤
  │ Self         │ ChipID lookup     │ SerDes_A │
  │              │ (4 chips on card) │ SerDes_B │
  │ Card_0       │ PCIe_P2P_0       │ PCIe     │
  │ ...          │                  │          │
  │ Card_7       │ PCIe_P2P_7       │ PCIe     │
  └──────────────┴──────────────────┴──────────┘

Shortest path: Dijkstra over Ring A topology (static topology, fixed at compile time)
Ring A hop count: Chip0↔1=1, Chip0↔3=2 (via Chip1 or Chip2), Chip1↔2=2 (via Chip0)
```

**6.3.4 Error Handling**

```
Detection:
  CRC32 per frame (payload + header)
  Header CRC16 (independent check — know if header is corrupted before routing)
  64b/66b encoding provides DC balance + illegal codeword detection

Recovery:
  Single frame CRC error → NAK + retransmission (TX retains sent frames until ACK)
  3 consecutive frame CRC errors → link degraded, switch to Ring B
  Ring A + Ring B both down → interrupt Host, trigger card-level hot migration
  Timeout: 100 μs no credit return → send Heartbeat
           5 Heartbeats no response → link down

Link training (power-up):
  TX → RX: TS1/TS2 (PCIe-like training sequence)
  Bit lock → Word alignment → Lane deskew → Ready
  < 1 ms
```

**6.3.5 Latency Budget**

```
Same-card MoE Dispatch (Chip A → Chip B, 7168B, 2 frames):

  TX framer:                   ~20 ns
  SerDes TX (4 lane × 32G):    ~56 ns  (4088B + 24B) / 128 Gbps
  PCB trace (100mm):            ~0.2 ns
  SerDes RX + deframer:        ~50 ns  (deskew + CRC check)
  RX → MoE queue:              ~10 ns
  ────────────────────────────────────
  First frame arrival:         ~136 ns
  Second frame (3080B):        ~96 ns
  Total Dispatch (7168B):     ~232 ns → round to 250 ns

Cross-card MoE Dispatch (Chip A Card0 → Chip B Card3, via PCIe P2P):

  C2C → Chip0 proxy:          ~136 ns
  PCIe MWr (7168B, x16 64GB/s): ~112 ns
  Chip0 RX → C2C → target chip: ~136 ns
  ────────────────────────────────────
  Cross-card total latency:   ~384 ns → round to 400 ns

Comparison:
  Single MoE FFN layer (12,300 DSPs, fp4): ~3 μs
  Communication / computation ratio: 400 ns / 3,000 ns = 13%
  → All-to-all communication is not a bottleneck
```

### 6.4 Inter-Card PCIe 5.0 P2P

Only Chip0 per card connects to PCIe (R-Tile x16). Chip1/2/3 interact with the outside world via C2C → Chip0 → PCIe.

```
Chip0 BAR4 Layout (64 MB, uniform per card):

  ┌────────────────┬─────────┬──────────────────────────┐
  │ Offset          │ Size    │ Target                   │
  ├────────────────┼─────────┼──────────────────────────┤
  │ 0x0000_0000    │ 16 MB   │ Chip0 (local regs + DMA)  │
  │ 0x0100_0000    │ 16 MB   │ Chip1 (via C2C Proxy)    │
  │ 0x0200_0000    │ 16 MB   │ Chip2 (via C2C Proxy)    │
  │ 0x0300_0000    │ 16 MB   │ Chip3 (via C2C Proxy)    │
  └────────────────┴─────────┴──────────────────────────┘

Cross-card data flow (Chip0 CardA → Chip2 CardB):
  1. CardA Chip0 DMA Engine issues PCIe MWr
     Target address = CardB BAR4 base + 0x0200_0000 (Chip2 offset)
  2. PCIe fabric routing (CPU Root Complex or P2P direct)
  3. CardB Chip0 R-Tile receives MWr → write to Chip2 C2C TX queue
  4. CardB Chip0 C2C Proxy → Ring A → Chip2
  5. CardB Chip2 receives frame → MoE RX queue

P2P Bandwidth (unidirectional):
  PCIe 5.0 x16: ~64 GB/s (128b/130b)
  Cross-CPU socket (UPI 2.0): ~20 GB/s
  Bottleneck at UPI: 20 GB/s vs single-card 4 chips × ~7.2 Gbps = 28.8 Gbps < 20 GB/s → sufficient
  All 8 cards cross-socket: 8 × 28.8 Gbps = 230 Gbps ≈ 28.8 GB/s
  Worst case must traverse UPI: 28.8 GB/s > 20 GB/s → bottleneck exists

Optimization: prefer same-socket card allocation for MoE experts
  CPU0 (Card 0-3): experts #0-191
  CPU1 (Card 4-7): experts #192-383
  → Cross-socket MoE traffic halved → ~14.4 GB/s < 20 GB/s ✓
```

### 6.5 Communication Bandwidth Accounting

```
Per-Token, Per-MoE-Layer all-to-all (6 routed experts):
  Dispatch: 6 × 7168B FP8 = 42 KB
  Reduce:   6 × 7168B FP8 = 42 KB
  Total:                    ~84 KB / token / MoE layer

Expert distribution (after same-socket optimization, 48 experts/sock out of 192):
  P(same card)   = 12/192 = 6.25%    → via C2C SerDes
  P(same sock)   = (48-12)/192 = 18.75% → via PCIe P2P (same socket)
  P(cross sock)  = 144/192 = 75%     → via PCIe P2P (via UPI)

  Expected same-card experts:   6 × 0.0625 = 0.38
  Expected same-socket experts: 6 × 0.1875 = 1.13
  Expected cross-socket experts: 6 × 0.75   = 4.50

At 200 tps (tokens/sec) throughput:
  Intra-card C2C:     200 × 61 × 0.38 × 2 × 7KB = 65 MB/s   ← trivial
  Same-sock PCIe:     200 × 61 × 1.13 × 2 × 7KB = 193 MB/s ← < 1% of x16
  Cross-sock UPI:     200 × 61 × 4.50 × 2 × 7KB = 769 MB/s ← 3.8% of UPI 20GB/s

Layer forwarding (pipeline):
  31 chip-to-chip transitions/token × 7KB × 200 tps = 43 MB/s ← trivial

Conclusion: All communication paths utilization < 5%, bandwidth is ample.
            Latency is also not a bottleneck (C2C 250 ns, PCIe 400 ns vs compute 3 μs).
```

### 6.6 Fault Tolerance Design

**6.6.1 Fault Domains**

```
Four potential failure points:

  ① Single chip (on-chip SerDes/R-Tile/DSP/HBM failure)
  ② Single C2C link (F-Tile lane failure)
  ③ FPGA accelerator card (PCB/VRM failure → 4 chips down)
  ④ Host CPU / PCIe fabric failure

Not present:
  ✗ ToR Switch failure → no such hardware
  ✗ QSFP-DD optical module / DAC cable → no such hardware
  ✗ RoCE congestion / PFC deadlock → no such protocol
  ✗ Cross-machine network partition → single-machine only
```

**6.6.2 Chip-Level Fault Tolerance**

```
Single chip failure (highest occurrence rate):

  Mechanism: intra-card 4-chip weight mutual backup
    Each chip's HBM stores:
      Own: 12 experts + ~2 attention layers (~570 MB)
      Neighbor: another chip's weights on the same card (expert only, ~400 MB)
      Total: < 1 GB, HBM 32 GB far from full

  Detection: Heartbeat timeout 100 μs → failure confirmed

  Recovery:
    T+0:     C2C routing table update → failed chip's 12 experts shared by
             the other 3 chips on the same card
             Each chip +4 experts → 12→16 experts/chip
    T+50ms:  Neighbor chip activates backup weights (already in HBM)
    T+100ms: Full throughput restored
             Single chip failure: 0% throughput degradation (3 sibling chips take over)

  Degradation: None. All four chips failing simultaneously would require card-level hot spare.
```

**6.6.3 Link-Level Fault Tolerance**

```
Single C2C link failure:
  Detection: 3-frame CRC error/timeout → <1 μs
  Recovery: disable Ring A, all traffic via Ring B → zero frame loss
  Throughput: Ring B bandwidth 256 Gbps >> all-to-all ~65 MB/s → unchanged

Ring A + Ring B both down (extremely rare):
  → Card degraded to 4 independent chips (each communicates with Host directly via PCIe, brokered by Chip0)
  → Throughput unchanged, latency slightly increased (forwarded via Chip0)
```

**6.6.4 Card-Level Fault Tolerance**

```
Single card failure (PCB/VRM → 4 chips all down):
  8 cards → 7 cards, full cluster degraded

  Weight redundancy: same-socket other cards' Chip0 stores attention weights of this card's critical layers
  Recovery: same-socket 7 cards absorb the failed card's 12 layers + 48 experts
            Throughput drop ~12.5% (8→7 cards)
            Requires manual card replacement to restore full throughput

Dual-card simultaneous failure:
  Throughput drop ~25%, not auto-recoverable
  Probability: MTBF 50,000h, 8-card system MTBF ≈ 6,250h
               Dual failure within 4h window: (4/50,000)² × 28 ≈ 4.5 × 10⁻⁷ → negligible
```

**6.6.5 Availability Summary**

```
┌──────────────────┬────────────┬──────────────┬──────────────┐
│ Failure Type      │ Detection  │ Recovery     │ Throughput    │
│                   │ Latency    │ Time         │ After Recovery│
├──────────────────┼────────────┼──────────────┼──────────────┤
│ Single chip       │ <100 μs    │ <100 ms      │ 100%         │
│ Single C2C link   │ <1 μs      │ <1 μs        │ 100%         │
│ Single FPGA card  │ <100 μs    │ Manual ~4h   │ 87.5%        │
│ Host CPU / PCIe   │ <100 ms    │ Manual ~2h   │ Down (data)  │
│ Dual-card failure │ <100 μs    │ Manual ~4h   │ 75%          │
│ Dual C2C Ring down│ <1 μs      │ <1 ms        │ 100%         │
└──────────────────┴────────────┴──────────────┴──────────────┘

vs Old scheme (multi-machine Ethernet + dual ToR):
  Failure mode count: 4 → 4 (same)
  Missing: F-Tile port failure + ToR failure (no longer exist)
  New: C2C link failure + single chip failure (finer granularity)

  Key improvement: chip-level fault self-healing (old scheme required card replacement), higher availability
```

### 6.7 Comparison with Previous Scheme

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│                       │ Old (Ethernet + ToR) │ New (PCIe P2P + C2C) │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ Communication plane  │ Single plane (Eth)   │ Single plane (PCIe+C2C)│
│ Intra-card inter-chip│ — (single-chip/card) │ C2C SerDes Dual Ring │
│ Inter-card           │ 200GbE → ToR Switch  │ PCIe 5.0 x16 P2P     │
│ Cross-machine        │ ToR → ToR           │ — (single machine)   │
│ Latency (same-card)  │ —                    │ ~250 ns              │
│ Latency (cross-card) │ ~1.5 μs              │ ~400 ns              │
│ Protocol stack       │ RoCE v2 / UDP / IP   │ Bare PCIe TLP + C2C  │
│                      │                      │ frame                │
│ Hardware             │ FPGA + 2 ToR Switch  │ FPGA + 0 Switch      │
│ External IP purchase │ RoCE v2 IP (¥1M)     │ ¥0 (in-house DMA)    │
│ Cables               │ QSFP-DD DAC ×16      │ None                 │
│ Failure mode count   │ 4 types              │ 4 types              │
│ Hot spare            │ 2 hot-spare cards    │ Chip self-healing    │
│                      │                      │ (intra-card backup)  │
│ Software stack       │ RDMA verbs (complex) │ PCIe P2P DMA (simple)│
│ Per-cluster BOM delta│ +¥1.2M               │ +¥0                  │
│ Extra power/card     │ ~60W (F-Tile + cage) │ ~10W (C2C SerDes)    │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ Δ BOM (cluster)      │ Baseline             │ -¥1.25M              │
│ Δ Latency            │ Baseline             │ -1 μs/cross-card     │
│ Δ Complexity         │ Baseline             │ -1 protocol layer    │
│ Δ Hardware types     │ Baseline             │ -Switch, -Cable      │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

---
