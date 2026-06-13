# V2-Lite FFN Compute Architecture

> **Target**: DeepSeek V2-Lite MoE FFN Decode Accelerator  
> **Device**: Stratix 10 MX 1SM21BHU2F53E1VG  
> **Date**: 2026-06-13

## 1. Precision Model

| Parameter | Format | Bit Layout | Bias | DSP Mapping |
|-----------|--------|-----------|------|-------------|
| **Weight** | FP4 E2M1 | `[s][e1 e0][m]` | exp bias=1 | Lookup → int8 → DSP |
| **Activation** | FP8 E4M3 | `[s][e3 e2 e1 e0][m2 m1 m0]` | exp bias=7 | int8 to DSP A port |
| **Scale** | FP8 E4M3 | same as activation | bias=7 | per-group scale × product |
| **Accumulator** | Int32 Q12.20 | 2's complement | — | fabric adder tree |
| **SiLU Output** | FP8 E4M3 | same as activation | bias=7 | LUT + DSP interp |

**FP4 → int8 Decoding**:
```
FP4 weight (4-bit) → 16-entry LUT → int8 (signed, -8..+7)
FP8 activation → pass through as int8
DSP multiply: int8 × int8 → 16b product → accumulate to 32b
```

## 2. Architecture: GEMM = Σ Multi-GEMV

```
                        ┌─────────────────────────────────────────┐
                        │           GEMM Engine (Time-Mux)        │
                        │                                         │
  Activation            │  ┌───────┐  ┌───────┐  ┌───────┐      │  Output
  [H,1] (one token)    │  │ Gate  │  │  Up   │  │ Down  │      │  [H,1]
  ─────────────────────►│  │ Proj  │─►│ Proj  │─►│ Proj  │──────►
                        │  │ H×I   │  │ H×I   │  │ I×H   │      │
  Weights               │  └───┬───┘  └───┬───┘  └───┬───┘      │
  from HBM2 ────────────►│      │          │          │          │
                        │      ▼          ▼          ▼          │
                        │  ┌───────┐  ┌───────┐  ┌───────┐      │
                        │  │ SiLU  │  │ ×gate │  │Accum  │      │
                        │  │ LUT   │  │Merge  │  │Expert │      │
                        │  └───────┘  └───────┘  └───────┘      │
                        │                                         │
                        │  Expert 1 of TOP_K → loop 6 times       │
                        └─────────────────────────────────────────┘

   Dimensions:  H=2048, I=1408, TOP_K=6, NUM_EXPERTS=66
```

### 2.1 Systolic Array (GEMV Core)

```
                Weight[0:63]        Weight[64:127]      ...
      ┌────┐    ┌──┬──┬──┬──┐      ┌──┬──┬──┬──┐
Act[0]│MAC │───►│00│01│..│63│─────►│64│65│..│127│───► ... ──► out[0]
      │Row0│    └──┴──┴──┴──┘      └──┴──┴──┴──┘
      │    │    64 Lane × 8 Rows = 512 MAC/clk
      │....│
      │    │
Act[7]│MAC │───►│56│57│..│119│────►... ──► out[7]
      └────┘    └──┴──┴──┴──┘

      Pipeline: IDLE → PRELOAD → STREAM → DRAIN → REDUCE → STORE
      K_BEATS = HIDDEN / DSP_LANES = 2048 / 64 = 32 beats/row
```

### 2.2 Throughput Budget

| Projection | Dims | Cycles | Notes |
|-----------|------|--------|-------|
| Gate | 2048×1408 | 32×1408 = 45k | Weight read overlap |
| SiLU | 1408 | 1408/64 = 22 | Pipelined 64-wide |
| Up | 2048×1408 | 32×1408 = 45k | Reuse activation buffer |
| Merge | 1408 | 1408/64 = 22 | FP16 × gate |
| Down | 1408×2048 | 22×2048 = 45k | |
| **Per Expert** | | **~135k** | |
| **6 Experts** | | **~810k** | Without pipelining |
| **With pipelining** | | **~585k** | Overlap next expert |
| **Tokens/sec @100MHz** | | **~170** | 585k cycles / 100M |
| **Tokens/sec @250MHz** | | **~427** | Target |

## 3. Clock Architecture

| Domain | Frequency | Source | Purpose |
|--------|-----------|--------|---------|
| `core_clk` | **100MHz → 250MHz** | IOPLL (ed_synth) | FFN compute, AXI, control |
| `hbm_refclk` | 100MHz (fixed) | Board Si5341A | HBM2 controller |
| `pcie_clk` | 250MHz | PCIe HIP coreclkout | PCIe AXI domain |
| `dsp_clk` | 500MHz (future) | IOPLL C1 output | DSP overdrive |

## 4. Parallelism (Decode Only)

```
Single token decode:
  Token activation [2048×FP8] → FFN → accumulated output [2048×FP8]
  
  Batch parallelism: could process multiple tokens serially
  (one token × TOP_K experts = 810k cycles @ 100MHz = 8.1ms/token)
  
  Multi-token batching: requires activation buffer × batch_size
  Target: batch 1-4 tokens for improved HBM2 bandwidth utilization
```

## 5. Register Map

```
BAR0 Layout:
  0x0000–0x0FFF  SYS    System & Version
  0x1000–0x1FFF  WT     Weight Transfer (pcie_hbm_weight_writer)
  0x2000–0x2FFF  FFN    FFN Engine Control/Status/Counters
  0x3000–0x3FFF  ACT    Activation Buffer
  0x4000–0x4FFF  PERF   Performance Monitoring
  0x5000–0x5FFF  ERR    Error & Diagnostics
```

Full definition: `v2_lite/docs/v2_lite_pcie_regmap.atreg`

## 6. Gaps vs Design Target

| Gap | Current | Target | Priority |
|-----|---------|--------|----------|
| Precision | FP8×FP8 (RTL done) | FP4×FP8 (weight decode LUT) | P0 |
| Clock | 100MHz (IOPLL 1:1) | **250MHz** (N=15, C0=6) | P0 |
| DSP count | 128 @ 100MHz | **512-1000** @ 250MHz | P0 |
| Simulation | 0 | Behavioral model matching golden | P0 |
| TPS measurement | Not measured | ISP perf counters → real TPS | P1 |
| Multi-expert pipelining | Sequential | Overlap expert N+1 preload | P1 |
| Batch parallelism | Single token | 1-4 tokens | P2 |
| PCIe EP | Gen3 x8, no BAR connect | Gen3 x8, BAR0+2 active | P0 |
| HBM2 write path | AXI W tied 0 | PCIe → AXI Wr → HBM2 | P0 |

## 7. Verification Plan

| Stage | Tool | What | Time |
|-------|------|------|------|
| Lint | Verilator ARM server | All .sv files | 30s |
| Unit sim | Verilator C++ | systolic_array, fp8_mac, hbm2_weight_reader | 2min |
| Integration sim | Verilator C++ | FFN engine + AXI SRAM model | 5min |
| Precision check | Python vs RTL | FP4 decode, SiLU, accumulate golden | 10min |
| Synthesis | ic31 quartus_syn | DSP > 0, no errors | 3min |
| Full compile | ic31 quartus_sh | SOF generation | 50min |
| JTAG verify | ISP readback | PCIe PLL, HBM2 TG, FFN FSM | 5min |
