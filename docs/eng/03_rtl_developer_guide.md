# FPGA LPU RTL Developer Guide

**Intended audience:** Experienced FPGA RTL engineers joining the FPGA LLM inference cluster project.

**Hardware target:** Intel Agilex 7 M-Series (AGMF039R47A1E2VR0, DK-DEV-AGM039EA)
**Language:** SystemVerilog (IEEE 1800-2017, synthesis subset)
**Project scope:** 32-chip pipeline serving DeepSeek V4 Pro (61 layers, 384 experts, FP4/E2M1 + FP8/E4M3 inference)

---

## 1. Getting Started (5-Minute Setup)

### 1.1 Required Tools

| Tool | Version | Purpose | Installation |
|------|---------|---------|--------------|
| **Quartus Prime Pro** | 24.3 | Synthesis, P&R, timing closure, Signal Tap | [Intel Download Center](https://www.intel.com/content/www/us/en/software-kit/795188/intel-quartus-prime-pro-edition-design-software-version-24-3-for-windows.html) |
| **Icarus Verilog** | 12.0+ (`iverilog -g2012`) | Fast bring-up simulation | `apt install iverilog` (Linux) / `brew install icarus-verilog` (macOS) |
| **cocotb** (optional) | 1.9+ | Python-driven verification | `pip install cocotb` |
| **Git** | 2.40+ | Version control | Standard |
| **GTKWave** | 3.3+ | Waveform viewing | `apt install gtkwave` |

Quartus Pro requires a license (free Intel Quartus Prime Lite does **not** support Agilex 7). Contact Intel for the Agilex development kit license bundle.

### 1.2 Clone and Directory Structure

```
fpgalpu/
├── rtl/                        # All synthesizable RTL (single source of truth)
│   ├── include/                # Parameter & type packages
│   │   ├── lpu_config.svh      # Central config: bring-up vs production
│   │   ├── fp4_params.svh      # FP4 datapath shared parameters
│   │   └── fp4_types.svh       # FP4/FP8 type definitions + decode functions
│   ├── interfaces/             # Interface struct definitions
│   │   ├── avalon_stream.svh   # Token-level streaming (pipeline, MoE dispatch)
│   │   ├── c2c_packet.svh      # Chip-to-Chip SerDes frame format
│   │   └── pcie_dma.svh        # PCIe 5.0 DMA descriptor + BAR0 register map
│   ├── dsp/                    # FP4 compute datapath
│   ├── attention/              # MLA attention + KV cache
│   ├── moe/                    # MoE router + expert FFN
│   ├── activation/             # RMSNorm, SiLU, Q12<->FP8 conversion
│   ├── layer/                  # Full transformer layer integration
│   ├── chip/                   # Chip-level top + KV DMA engine
│   ├── engram/                 # Sparse memory lookup engine
│   ├── head/                   # Multi-Token Prediction (MTP) head
│   ├── debug/                  # UART debug, HBM bandwidth test, DSP stress test
│   ├── legacy/                 # Superseded v1 modules (DO NOT MODIFY)
│   └── sim/                    # Testbenches + Makefile
├── hw/
│   ├── src/                    # Board-level tops (top_master.sv, top_slave.sv)
│   ├── quartus/                # 9 Quartus projects
│   │   ├── common/             # common_modules.qsf — shared RTL file list
│   │   ├── master/             # Master FPGA (PCIe enabled, CHIP_ID=0)
│   │   ├── slave/              # Slave FPGA (no PCIe, CHIP_ID=1-31)
│   │   ├── bringup/            # Bring-up bitstream (fast iteration)
│   │   ├── dsp_char/           # DSP characterization project
│   │   ├── hbm_char/           # HBM characterization project
│   │   ├── pcie_test/          # PCIe standalone test
│   │   ├── c2c_test/           # C2C ring test
│   │   └── full_stack/         # Full production stack
│   └── constraints/            # SDC timing constraints
├── scripts/
│   ├── simulation/             # Python golden models + vector generation
│   ├── architecture/           # Architecture exploration (Python)
│   └── prefill/                # vLLM prefill coordinator
└── docs/                       # Documentation (you are here)
```

### 1.3 First Simulation (Icarus Verilog, 30 Seconds)

The fastest way to verify your environment works:

```bash
# 1. Open a terminal in the project root
cd fpgalpu

# 2. Run the fp4_mac unit test (the fundamental compute primitive)
cd rtl/sim
make SIM=iverilog tb_fp4_mac

# 3. If you see "ALL TESTS PASSED", your environment is ready.
# Typical runtime: ~30 seconds for bring-up parameters.

# 4. View waveforms (optional)
make SIM=iverilog tb_fp4_mac   # generates tb_fp4_mac.vcd
gtkwave tb_fp4_mac.vcd
```

**What this tests:** The 4-stage FP4(E2M1) x FP8(E4M3) multiply-accumulate pipeline with saturation. This is the fundamental DSP cell used in every compute array in the design.

### 1.4 Building Bring-Up vs Production Bitstream

| Aspect | Bring-Up (`FPGA_LPU_PRODUCTION` **NOT** defined) | Production (`FPGA_LPU_PRODUCTION` defined) |
|--------|--------------------------------------------------|--------------------------------------------|
| Hidden dim | 8 | 7168 |
| Intermediate dim | 4 | 3072 |
| Experts | 4 | 384 |
| Heads | 4 | 128 |
| Layers | 12 | 61 |
| KV cache slots | 64 | 4096 |
| DSP clock | 100 MHz | 450 MHz |
| Icarus sim time | ~30 seconds | **Impossible** (memory blowout) |
| Quartus compile | ~30 min | 4--6 hours (c6i.16xlarge EC2) |
| Use case | Algorithm verification, RTL debug | Bitstream for board testing/production |

**How to switch:**

```bash
# Bring-up (default — no define needed):
iverilog -g2012 -I../include -o tb.vvp ../dsp/fp4_mac.sv tb_fp4_mac.sv

# Production (Quartus):
# Add this line to your .qsf file:
#   set_global_assignment -name VERILOG_MACRO "FPGA_LPU_PRODUCTION"
```

See Section 4 for full details on the parameterization mechanism.

---

## 2. RTL Codebase Map

### 2.1 Complete Module Directory

Every synthesizable SystemVerilog file in the project. Parameters default from `lpu_config_pkg` and may be overridden per-instance.

#### `rtl/dsp/` — FP4 Compute Datapath

| File | Description | Pipeline | DSP Usage |
|------|-------------|----------|-----------|
| `fp4_mac.sv` | FP4xFP8 MAC unit with saturation | 4-stage | 2 DSPs (18x19 per multiply) |
| `fp4_scale_reader.sv` | BRAM-based per-group fp8 scale lookup | 2-stage | 0 (BRAM read) |
| `fp4_systolic_cell.sv` | Single systolic cell: fp4_mac + scale reader | 6-stage (combined) | 2 DSPs |
| `fp4_systolic_2d.sv` | 2D systolic array: `LPU_ARRAY_LANES x LPU_ARRAY_M_ROWS` | Pipelined rows | N*M*2 DSPs |
| `fp4_gemm_engine.sv` | Full GEMM engine: weight buffer + systolic array + accumulator | Deep pipeline | Array dependent |
| `fp4_prefill_engine.sv` | Superscalar prefill pipeline for dense GEMM (reserved for future FPGA prefill) | Multi-bank | Array dependent |

**Critical path:** `fp4_mac.sv` Stage 2 (base multiply) and Stage 3 (scale multiply) are the two DSP-bound operations. Each fits within a single Agilex DSP block (18x19 mode). The `(* altera_attribute = "-name ALLOW_RETIMING ON" *)` annotation on Stage 0/1 registers lets Quartus retime across DSP boundaries for 450 MHz closure.

#### `rtl/attention/` — MLA Attention + KV Cache

| File | Description |
|------|-------------|
| `mla_qkv_proj.sv` | Low-rank QKV projection: hidden -> Q, K, V, K_latent, V_latent |
| `mla_rope.sv` | Rotary Position Embedding: LUT-based sin/cos application |
| `mla_kv_cache.sv` | KV cache BRAM: store/retrieve K_latent and V_latent per token |
| `mla_attention_v2.sv` | Full MLA pipeline: QKV->RoPE->cache write->score->softmax->output |

**Key design decision (v2):** K and V are stored in their compressed latent forms (K_latent, V_latent of dimension 512) rather than full expanded forms (H=7168). This reduces KV cache memory by 14x. The decompression matrix W_UK/W_UV is applied on-the-fly during attention score computation.

#### `rtl/moe/` — Mixture of Experts

| File | Description |
|------|-------------|
| `router_topk.sv` | Top-K gating: activation x expert weights -> scores -> top-2 selection |
| `expert_ffn_engine_fp4_down.sv` | Expert FFN: gate_proj + up_proj + SiLU + down_proj in FP4 |

**Router pipeline:** 3-stage: latch activations -> pairwise products -> reduction + top-2 search. At production scale (384 experts, 7168 hidden), router uses ~275K DSP multiplies per token.

#### `rtl/activation/` — Activation Functions

| File | Description |
|------|-------------|
| `rms_norm.sv` | Root Mean Square normalization (sqrt-mean-square + scale) |
| `silu_q12_lut.sv` | SiLU activation via LUT (Q12 fixed-point in/out) |
| `q12_to_fp8_e4m3.sv` | Q12 signed -> FP8 E4M3 conversion (used at FFN ingress) |

**Data format transitions:**
- Hidden state internal: **Q12 signed** (32-bit, 12 fractional bits)
- FFN activation input: **FP8 E4M3** (converted from Q12 via `q12_to_fp8_e4m3`)
- Compute datapath: **FP4 E2M1** weights x **FP8 E4M3** activations, accumulated at Q12
- FFN output -> hidden state: **Q12 signed** (via `fp8_to_scaled12` at output)

#### `rtl/layer/` — Layer Integration

| File | Description |
|------|-------------|
| `mhc_mixer.sv` | Multi-Head Channel mixer (pre-MLA v2, superseded by integrated approach) |
| `full_transformer_layer.sv` | Complete transformer layer: RMS->Attn->RMS->Router->FFN->RMS |
| `layer_compute_engine.sv` | Multi-layer sequencing engine (chains N layers on one chip) |

**Layer FSM states** (`full_transformer_layer.sv`): S_IDLE -> S_R1 (RMS1) -> S_ATTN (MLA v2) -> S_R2 (RMS2) -> S_RTR (Router Top-K) -> S_FFN_LD1/S_FFN_LD2 (Q12->FP8 encoding) -> S_FFN (Expert FFN) -> S_R3 (RMS3) -> S_OUT -> loop back.

#### `rtl/chip/` — Chip-Level Top

| File | Description |
|------|-------------|
| `chip_top.sv` | Single chip wrapper: C2C rings, PCIe proxy, layer compute, config registers |
| `kv_dma_engine.sv` | Dedicated DMA engine for Host SSD -> HBM KV cache block transfers |
| `kv_dma_bridge.sv` | Bridge between KV DMA request stream and HBM write interface |

**chip_top.sv parameters:**
- `CHIP_ID` (0--31): Global chip identity
- `CARD_ID` (0--7): Which PCIe card
- `LAYER_START` / `LAYER_END`: Which transformer layers this chip computes (12 per chip)
- `IS_PCIE_MASTER`: 1 for Chip 0 of each card (R-Tile PCIe IP present), 0 for slaves

#### `rtl/engram/` — Sparse Memory Lookup

| File | Description |
|------|-------------|
| `hash_unit.sv` | Hash function for sparse key addressing |
| `sram_cache.sv` | BRAM-based SRAM cache with LRU eviction |
| `lookup_engine.sv` | Top-level engram lookup: hash -> cache -> miss handler |

Engram provides sparse key-value memory for context retrieval. Not on the critical decode path.

#### `rtl/head/` — Multi-Token Prediction

| File | Description |
|------|-------------|
| `mtp_head.sv` | MTP speculative head: predicts next-N tokens from final hidden state |
| `mtp_verify.sv` | MTP verification: compares predicted vs actual tokens |

MTP enables speculative decoding: the FPGA predicts 2--4 future tokens in parallel to reduce per-token latency.

#### `rtl/debug/` — Debug & Stress Test

| File | Description |
|------|-------------|
| `uart_debug.sv` | UART TX for printf-style debug output (115200 baud) |
| `hbm_bw_test.sv` | HBM2e bandwidth characterization (sequential + random access) |
| `dsp_stress_test.sv` | DSP array toggle coverage test at 450 MHz |

#### `rtl/legacy/` — DO NOT MODIFY

| File | Description | Why Legacy |
|------|-------------|------------|
| `fp4_systolic_array.sv` | v1 systolic array | Replaced by `fp4_systolic_2d.sv` |
| `fp4_linear_engine.sv` | v1 linear layer | Replaced by `fp4_gemm_engine.sv` |
| `fp4_scaled_tile.sv` | v1 tile with scale | Replaced by `fp4_systolic_cell.sv` |
| `fp4_systolic_tile.sv` | v1 systolic tile | Replaced by `fp4_systolic_2d.sv` |
| `mla_attention.sv` | v1 MLA attention | Replaced by `mla_attention_v2.sv` |
| `expert_ffn_engine.sv` | v1 expert FFN | Replaced by `expert_ffn_engine_fp4_down.sv` |
| `c2c_node.sv` | v1 C2C node | Replaced by integrated ring logic in `chip_top.sv` |

These modules are kept for reference only. They are excluded from production QSF files (commented out in `common_modules.qsf`). Any bug found in a legacy module should be fixed in its v2 replacement instead.

#### `hw/src/` — Board-Level Tops

| File | Description |
|------|-------------|
| `top_master.sv` | Master FPGA board wrapper: PCIe x16 + HBM2e + C2C ring origin |
| `top_slave.sv` | Slave FPGA board wrapper: HBM2e + C2C ring forwarder (no PCIe) |

Both use the same `chip_top.sv` RTL. The only difference is `IS_PCIE_MASTER` (1 vs 0), which gates PCIe IP at synthesis time.

#### `rtl/interfaces/` — Interface Definitions

| File | Contents |
|------|----------|
| `avalon_stream.svh` | `stream_beat_t`, `pipeline_fwd_beat_t`, `moe_dispatch_beat_t`, `moe_reduce_beat_t` |
| `c2c_packet.svh` | `c2c_header_t` (msg_type, ring, src/dst chip, seq_id, payload_len), `c2c_link_t`, `c2c_ctrl_t` |
| `pcie_dma.svh` | `pcie_dma_desc_t`, `pcie_bar0_regs_t`, `pcie_dma_stream_t`, `pcie_c2c_proxy_t` |

#### `rtl/include/` — Parameter & Type Headers

| File | Contents |
|------|----------|
| `lpu_config.svh` | `lpu_config_pkg` with all architectural parameters (two-mode: bring-up / production) |
| `fp4_params.svh` | `fp4_params_pkg`: group size, scale width, weight/activ/accum widths |
| `fp4_types.svh` | `fp4_decoded_t`, `fp4_mac_input_t`, `fp4_mac_output_t`, decode functions |

### 2.2 Testbench Directory (`rtl/sim/`)

| Testbench | DUT | Bring-Up Sim Time |
|-----------|-----|-------------------|
| `tb_fp4_mac.sv` | `fp4_mac` | ~5s |
| `tb_fp4_scale_reader.sv` | `fp4_scale_reader` | ~5s |
| `tb_cell_mini.sv` | `fp4_systolic_cell` (1 cell) | ~5s |
| `tb_2d_1x4.sv` | `fp4_systolic_2d` (1x4) | ~10s |
| `tb_2x2.sv` | Systolic 2x2 array | ~10s |
| `tb_2d_4x4.sv` | Systolic 4x4 array | ~15s |
| `tb_fp4_gemm_engine.sv` | `fp4_gemm_engine` | ~20s |
| `tb_fp4_systolic_2d.sv` | Full systolic 2D array | ~30s |
| `tb_silu_q12_lut.sv` | `silu_q12_lut` | ~5s |
| `tb_rms_norm.sv` | `rms_norm` | ~5s |
| `tb_router_topk.sv` | `router_topk` | ~10s |
| `tb_expert_ffn_engine_fp4_down.sv` | Expert FFN fp4 | ~15s |
| `tb_mla_qkv.sv` | `mla_qkv_proj` | ~10s |
| `tb_mla_attention_v2.sv` | `mla_attention_v2` | ~15s |
| `tb_mhc_mixer.sv` | `mhc_mixer` | ~10s |
| `tb_full_transformer_layer.sv` | `full_transformer_layer` | ~20s |
| `tb_chip_12layer.sv` | `chip_top` with 12 layers | ~45s |
| `tb_cluster_384.sv` | 32-chip cluster (384 layers) | ~5 min |
| `tb_c2c_ring.sv` | C2C ring (bring-up only) | ~20s |
| `tb_kv_dma.sv` | `kv_dma_engine` | ~10s |
| `tb_lookup_engine.sv` | `lookup_engine` | ~10s |
| `tb_mtp_head.sv` | `mtp_head` | ~10s |

**Golden vector testbenches** (compare RTL output vs Python reference):
| Testbench | Golden source |
|-----------|---------------|
| `tb_expert_ffn_engine_fp4_down_golden.sv` | `scripts/simulation/gen_ffn_tb_vectors.py` |
| `tb_layer_compute_engine_golden.sv` | `scripts/simulation/gen_layer_golden.py` |

---

## 3. Coding Conventions

### 3.1 File Naming

- **All files:** `snake_case.sv` for SystemVerilog, `snake_case.svh` for headers
- **File name == module name:** `fp4_mac.sv` contains `module fp4_mac`
- **Testbenches:** prefix `tb_` (e.g., `tb_fp4_mac.sv`)
- **Golden testbenches:** suffix `_golden` (e.g., `tb_layer_compute_engine_golden.sv`)
- **Packages:** suffix `_pkg` (e.g., `lpu_config_pkg`, `fp4_params_pkg`)

### 3.2 Signal Naming Conventions

This project uses a straightforward prefix convention (no Hungarian notation):

| Prefix | Meaning | Example |
|--------|---------|---------|
| (none) | Standard logic signal | `valid_in`, `weight`, `clk` |
| `sN_` | Pipeline stage N register | `s0_weight`, `s2_base_product` |
| `g_` | Generate block label | `g_pcie_master`, `g_c2c_slave` |
| `u_` | Module instance | `u_mac`, `u_attn`, `u_rtr` |
| `cfg_` | Configuration register | `cfg_layer_start`, `cfg_expert_bitmap` |

**Clock and reset:**
- `clk` — primary clock (the default in sub-modules)
- `rst_n` — active-low synchronous reset
- Multi-clock modules name clocks explicitly: `clk_sys`, `clk_dsp`, `clk_pcie`, `clk_hbm`

**Multi-bit signals:**
- Vectors use standard `[MSB:LSB]` ordering
- Wide hidden state uses flat concatenation: `hidden_flat[HIDDEN*DATA_W-1:0]`
- When individual elements are needed, use indexed ports: `a0, a1, a2, ... a7`

### 3.3 Module Port Ordering

Ports are declared in a consistent order throughout the codebase:

1. Clock and reset (`clk`, `rst_n`)
2. Control/configuration inputs (weight preload ports, enable signals)
3. Data inputs (streaming valid/data pairs)
4. Data outputs (streaming valid/data pairs)
5. Status outputs

In signal declarations, `input` comes before `output`.

### 3.4 Parameter vs `define` Usage

| Mechanism | When to Use | Example |
|-----------|-------------|---------|
| **`parameter`** (in `package` or module) | Architectural sizing that varies between bring-up and production | `HIDDEN`, `NUM_EXPERTS`, `ARRAY_LANES` |
| **`` `define ``** | Project-wide mode selection, include guards | `` `FPGA_LPU_PRODUCTION ``, `` `ifndef LPU_CONFIG_SVH `` |

**Rule:** Do NOT use `` `define `` for sizing parameters. Use package `parameter`s so they can be overridden per-instance. The only `` `define `` is `FPGA_LPU_PRODUCTION`, which controls which parameter set a package selects.

```systemverilog
// CORRECT: parameter from package, overridable at instantiation
full_transformer_layer #(.HIDDEN(256)) u_layer (...);

// WRONG: do not scatter model dimensions as `defines
// `define HIDDEN 7168  // <-- NO
```

### 3.5 State Machine Style

All FSMs in this project use the same pattern:

```systemverilog
typedef enum logic [N:0] { S_IDLE, S_STAGE1, S_STAGE2, ... } state_t;
state_t state;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S_IDLE;
        // reset all control outputs
    end else begin
        // default: pulse-zero control signals each cycle
        sig1 <= 0; sig2 <= 0; valid_out <= 0;
        case (state)
            S_IDLE: if (valid_in) begin
                // latch inputs
                state <= S_STAGE1;
            end
            S_STAGE1: if (done_condition) begin
                state <= S_STAGE2;
            end
            // ...
            default: state <= S_IDLE;
        endcase
    end
end
```

Key patterns:
- **Default zero-pulse:** Control signals (`ffn_start`, `r1_vi`, `attn_in_valid`) are set to 0 each clock cycle, then conditionally asserted. This prevents stale assertions.
- **Explicit next-state:** Every state has a clear transition condition. No implicit fall-through.
- **Single `case` block:** All control logic lives in one `always_ff` block for clarity.
- **Named states only:** No raw state constants. Use `typedef enum`.

### 3.6 DSP Inference Guidance for 450 MHz

Agilex 7 M-series DSP blocks (variable precision mode) are the critical resource. To achieve timing closure at 450 MHz:

1. **Minimum 4 pipeline stages inside fp4_mac:** Stage 0 (input reg), Stage 1 (decode), Stage 2 (base multiply, 1 DSP), Stage 3 (scale multiply, 1 DSP), Stage 4 (accumulate). Each stage maps to 1 DSP block.

2. **Use `ALLOW_RETIMING` attribute:** On the input registers (Stage 0/1), so Quartus can move them across DSP boundaries during fitting.
   ```systemverilog
   (* altera_attribute = "-name ALLOW_RETIMING ON" *)
   logic [3:0] s0_weight;
   ```

3. **Use `DSP_BLOCK_BALANCING AUTO`:** On the module, lets the fitter redistribute pipeline stages across DSP cascade chains.
   ```systemverilog
   (* altera_attribute = "-name DSP_BLOCK_BALANCING AUTO" *)
   module fp4_mac (...);
   ```

4. **Pre-decode scales at load time:** `fp8_to_scaled12()` is called in `fp4_scale_reader` (scale load time), not in the MAC critical path. This removes ~4 LUT levels from Stage 1.

5. **DSP and HBM share 450 MHz:** `clk_dsp = clk_hbm = 450 MHz` (same PLL). No CDC between compute and memory.

6. **Multicycle paths for accumulators:** SDC constraints relax setup on accumulator registers (see `fpga_lpu.sdc`):
   ```tcl
   set_multicycle_path -setup 2 -to [get_registers *u_mac*accumulator*]
   ```

### 3.7 Reset Strategy

- **Active-low reset:** `rst_n` throughout the design
- **3-FF synchronizer** on board-level reset input (see `top_master.sv`):
  ```systemverilog
  logic [2:0] rst_sr;
  always_ff @(posedge clk_sys or negedge cpu_reset_n) begin
      if (!cpu_reset_n) rst_sr <= '0;
      else              rst_sr <= {rst_sr[1:0], 1'b1};
  end
  assign rst_n_sys = rst_sr[2];
  ```
- Sub-modules reset ALL state elements. No partial resets.
- No reset for BRAM-based memories (initialized via weight preload, not reset).

---

## 4. Bring-Up vs Production Parameterization

### 4.1 How `FPGA_LPU_PRODUCTION` Works

The entire parameterization of the design is controlled by a single `` `define `` and one package file: `rtl/include/lpu_config.svh`.

```systemverilog
package lpu_config_pkg;

`ifdef FPGA_LPU_PRODUCTION
    parameter int LPU_HIDDEN     = 7168;   // production
    parameter int LPU_ARRAY_LANES = 128;   // production
    // ...
`else
    parameter int LPU_HIDDEN     = 8;      // bring-up
    parameter int LPU_ARRAY_LANES = 8;     // bring-up
    // ...
`endif

endpackage
```

Modules import defaults from the package:

```systemverilog
module full_transformer_layer #(
    parameter int HIDDEN = lpu_config_pkg::LPU_HIDDEN,  // 8 or 7168
    // ...
) (...);
```

**To use production in Quartus:**
```tcl
# In .qsf (e.g., master/fpga_lpu_master.qsf)
set_global_assignment -name VERILOG_MACRO "FPGA_LPU_PRODUCTION"
```

**To use bring-up (default):**
No `` `define `` needed. Icarus and default Quartus projects omit the production define.

### 4.2 What Changes

| Parameter | Bring-Up | Production | Impact |
|-----------|----------|------------|--------|
| `LPU_HIDDEN` | 8 | 7168 | Matrix dimensions, BRAM usage |
| `LPU_INTERMEDIATE` | 4 | 3072 | FFN width |
| `LPU_K_LATENT` | 4 | 512 | KV cache entry size |
| `LPU_V_LATENT` | 4 | 512 | KV cache entry size |
| `LPU_NUM_HEADS` | 4 | 128 | Attention parallelism |
| `LPU_QK_ROPE_DIM` | 4 | 64 | RoPE rotation dimension |
| `LPU_V_HEAD_DIM` | 8 | 128 | V projection dimension |
| `LPU_NUM_EXPERTS` | 4 | 384 | Router score vector size |
| `LPU_TOP_K` | 2 | 6 | Selected experts per token |
| `LPU_EXPERTS_PER_FPGA` | 4 | 12 | Local expert count per chip |
| `LPU_NUM_LAYERS` | 12 | 61 | Total transformer depth |
| `LPU_VOCAB_SIZE` | 16 | 129280 | LM head softmax dimension |
| `LPU_ARRAY_LANES` | 8 | 128 | K-direction systolic parallelism |
| `LPU_ARRAY_M_ROWS` | 4 | 32 | M-direction systolic parallelism |
| `LPU_KV_CACHE_SLOTS` | 64 | 4096 | Tokens storable per layer |
| `LPU_CLK_DSP_MHZ` | 100 | 450 | DSP clock frequency |

### 4.3 When to Use Each Mode

| Activity | Mode | Why |
|----------|------|-----|
| RTL development iteration | Bring-Up | Icarus simulation in 30s |
| Algorithm verification | Bring-Up | Full pipeline visible at small scale |
| Golden vector generation | Bring-Up | Python models match small-param RTL |
| Timing validation | Production | Must check 450 MHz closure on production sizes |
| Resource estimation | Production | BRAM/DSP/ALM count is production-accurate |
| Board testing | Production | Actual DeepSeek V4 Pro bitstream |
| Full bitstream generation | Production | 4-6 hour compile on c6i.16xlarge |

### 4.4 Cloud Build (c6i.16xlarge EC2)

For production bitstreams that require 4-6 hours and 64GB+ RAM:

```bash
# 1. Launch c6i.16xlarge (64 vCPU, 128 GB RAM, ~$2.90/hour spot)
# 2. Install Quartus Prime Pro 24.3 (license required)
# 3. Sync the repository
# 4. Build:
cd fpgalpu/hw/quartus/master
quartus_sh --flow compile fpga_lpu_master

# 5. Retrieve output files:
#    output_files/fpga_lpu_master.sof  (bitstream for JTAG)
#    output_files/fpga_lpu_master.sof.rpt  (timing report)
```

Spot instances are cost-effective: a 6-hour compile at $2.90/hr = ~$17.40 per bitstream.

---

## 5. Module Interface Standards

### 5.1 Avalon-ST Streaming Interface

Defined in `rtl/interfaces/avalon_stream.svh`. Four message types share a common handshake pattern.

**Base handshake:** `valid` (source) and `ready` (sink). Data transfers when both are asserted on the same clock edge. No backpressure if `ready` is tied to 1.

**Message types:**

| Type | Struct | Use Case |
|------|--------|----------|
| General stream | `stream_beat_t` | Any token-level data (dest chip, src chip, layer index, token ID, 32b data) |
| Pipeline forward | `pipeline_fwd_beat_t` | Hidden state passing to next chip in pipeline |
| MoE dispatch | `moe_dispatch_beat_t` | Send activation to remote chip for expert computation |
| MoE reduce | `moe_reduce_beat_t` | Remote expert result returned to requesting chip |

**Key fields:**
- `dest[7:0]` / `src[7:0]`: Global chip IDs (0--31)
- `layer[5:0]`: Layer index (0--60) or expert bitmap for dispatch
- `token_id[15:0]`: Unique token/session identifier for multi-request tracking
- `data[31:0]`: Payload (Q12 hidden state element or FP8 activation)

### 5.2 C2C (Chip-to-Chip) Packet Format

Defined in `rtl/interfaces/c2c_packet.svh`. F-Tile SerDes physical layer carries up to 4088-byte payloads per frame.

**Header format** (`c2c_header_t`, 64 bits):
```
[57:55] msg_type    : 0=PIPE_FWD, 1=MOE_DISP, 2=MOE_REDU, 3=CTRL
[54]    ring        : 0=Ring A (CW), 1=Ring B (CCW)
[53:46] src_chip    : origin chip global_id
[45:38] dst_chip    : final destination chip global_id
[37:22] seq_id      : sequence number for reorder/drop detection
[21:6]  payload_len : bytes in payload (0-4088)
[5:0]   reserved
```

**Link structure** (`c2c_link_t`):
```systemverilog
typedef struct packed {
    logic        tx_valid, tx_ready;
    c2c_header_t tx_header;
    logic [4087:0] tx_payload;
    logic        rx_valid, rx_ready;
    c2c_header_t rx_header;
    logic [4087:0] rx_payload;
} c2c_link_t;
```

**Dual ring topology:** Ring A is clockwise (0->1->2->3->0), Ring B is counter-clockwise. Each message is routed on the ring with fewer hops. A chip forwards messages where `dst_chip != local_chip_id`.

**Control messages** (`C2C_MSG_CTRL`):
Used for register read/write and expert bitmap configuration on slave chips (which lack PCIe). Slave chips receive their entire config via C2C CTRL from the master.

### 5.3 PCIe DMA Descriptor Format

Defined in `rtl/interfaces/pcie_dma.svh`. Chip 0 of each card has R-Tile PCIe 5.0 x16.

**DMA descriptor** (`pcie_dma_desc_t`):
```
src_addr[63:0]  : 64-bit host physical address (source)
dst_addr[63:0]  : 64-bit FPGA HBM address (destination)
length[31:0]    : bytes to transfer
flags[15:0]     : bit0=intr_on_done, bit1=chain
status[15:0]    : written by DMA engine on completion
```

**BAR0 register map** (`pcie_bar0_regs_t`):
```
ctrl[31:0]          : [0]=go, [1]=reset, [3:2]=mode
status[31:0]        : [0]=done, [1]=error, [15:8]=chip_id
desc_ring_base[63:0]: host physical address of descriptor ring
desc_ring_size[15:0]: number of descriptors
desc_head[15:0]     : written by host (next desc to submit)
desc_tail[15:0]     : written by FPGA (next desc completed)
irq_mask[31:0]      : MSI-X interrupt mask
perf_counters[8]    : DMA performance counters
```

**C2C proxy bridge** (`pcie_c2c_proxy_t`):
Chip 0 forwards host traffic for Chips 1--3 via this proxy. Non-master chips access the host exclusively through C2C -> Chip 0 -> PCIe proxying.

### 5.4 Handshake Protocol (Valid/Ready)

All streaming interfaces use the Avalon-ST valid/ready handshake:

```systemverilog
// Source
always_ff @(posedge clk) begin
    if (out_ready) out_valid <= 1'b0;  // transaction consumed
    if (new_data_available) begin
        out_data  <= data;
        out_valid <= 1'b1;
    end
end

// Sink
assign in_ready = !busy;  // or constant 1 if always accepting
always_ff @(posedge clk) begin
    if (in_valid && in_ready) begin
        latched_data <= in_data;  // only consume when both high
    end
end
```

**Important:** `valid` MUST NOT depend on `ready` (combinational loop). `ready` MAY depend on `valid` (e.g., `ready = valid && !full`).

In this codebase, many sink modules tie `ready = 1'b1` (always accepting) to minimize latency at the cost of requiring the source to track readiness. This is intentional for the decode path.

---

## 6. Simulation Flow

### 6.1 Icarus Verilog (Primary for Bring-Up Development)

Icarus is the default simulator for bring-up development. It is fast, free, and has sufficient SystemVerilog support for our bring-up parameter sizes.

**Setup (Linux/macOS):**
```bash
# Install
sudo apt install iverilog gtkwave   # Linux
brew install icarus-verilog gtkwave # macOS (x86)

# Verify
iverilog -V
# Expected: Icarus Verilog version 12.0 or later
```

**Running a single testbench:**
```bash
cd rtl/sim

# Use the Makefile with SIM=iverilog
make SIM=iverilog tb_fp4_mac
# Output: tb_fp4_mac.vvp and optionally tb_fp4_mac.vcd

# Or manual iverilog invocation:
iverilog -g2012 \
  -I../include \
  -o tb_fp4_mac.vvp \
  ../dsp/fp4_mac.sv \
  tb_fp4_mac.sv
vvp tb_fp4_mac.vvp

# View waveforms:
gtkwave tb_fp4_mac.vcd
```

**Running all bring-up tests:**
```bash
cd rtl/sim
make SIM=iverilog all     # all tests (may take ~5-10 min total)
```

**The `-g2012` flag is MANDATORY.** Without it, Icarus uses Verilog-2005, which lacks SystemVerilog `package`, `typedef struct packed`, `always_ff`, and `enum`. All our RTL uses `-g2012` features.

### 6.2 Questa / ModelSim (For Larger Simulations)

When you need production-scale simulation or advanced debugging, use Questa Intel Starter Edition:

```bash
cd rtl/sim

# Compile
make SIM=questa compile

# Run (command-line, no GUI)
make SIM=questa run

# Run with GUI (waveform viewer, signal tracing)
make SIM=questa gui

# Clean build artifacts
make SIM=questa clean
```

Questa handles production parameter sizes that Icarus cannot. However, production parameter simulation is very slow and only recommended for targeted debugging of specific modules (not full-chip or cluster-level).

### 6.3 Verilator (Linting)

Verilator provides fast lint checks without a simulator license:

```bash
cd rtl/sim
make SIM=verilator lint
```

The current `Makefile` supports Verilator lint-only mode (full C++ harness simulation is not yet implemented).

### 6.4 Waveform Viewing (GTKWave)

```bash
# After running a simulation that produces a .vcd:
gtkwave tb_fp4_mac.vcd

# Key signals to add when debugging fp4_mac:
#   clk, rst_n, accum_clr
#   mac_in.weight, mac_in.scale, mac_in.activ
#   s0_weight, s1_w_signed, s2_base_product, s3_product
#   accumulator, mac_out.result, mac_out.valid
#
# Tip: Save your signal set: File -> Write Save File -> fp4_mac.gtkw
```

### 6.5 Golden Vector Generation (Python -> RTL)

Test vector generation from Python reference models:

```bash
# Generate golden vectors for fp4_mac
cd scripts/simulation
python gen_tb_vectors.py

# Generate golden vectors for expert FFN
python gen_ffn_tb_vectors.py

# Generate golden vectors for full transformer layer
python gen_layer_golden.py

# Then run golden comparison testbench:
cd ../../rtl/sim
make SIM=iverilog tb_expert_ffn_engine_fp4_down_golden
make SIM=iverilog tb_layer_compute_engine_golden
```

**Flow:**
1. Python script generates random input vectors
2. Python computes expected (golden) output using FP32 reference math
3. Python writes both inputs and expected outputs to a SystemVerilog include file
4. Golden testbench feeds inputs to DUT, captures outputs, and asserts equality with tolerance

### 6.6 Test Vector Files

| Python Script | Generated Vectors | Golden Testbench |
|---------------|-------------------|------------------|
| `scripts/simulation/gen_tb_vectors.py` | `tb_golden_pkg.sv` | `tb_fp4_mac.sv` (built-in checks) |
| `scripts/simulation/gen_ffn_tb_vectors.py` | `tb_ffn_golden_pkg.sv` | `tb_expert_ffn_engine_fp4_down_golden.sv` |
| `scripts/simulation/gen_layer_golden.py` | `tb_layer_golden_pkg.sv` | `tb_layer_compute_engine_golden.sv` |

### 6.7 Running the Full Validation Suite

```bash
# Run all module smoke tests
cd fpgalpu
python scripts/run_module_smoke.py

# Run all validations (RTL sim + Python reference comparison)
python scripts/run_all_validations.py

# End-to-end validation (architecture model)
python scripts/run_e2e_validation.py
```

---

## 7. Quartus Build Flow

### 7.1 Project Structure (9 Quartus Projects)

| Project | Directory | Purpose | `FPGA_LPU_PRODUCTION` |
|---------|-----------|---------|----------------------|
| **master** | `hw/quartus/master/` | Master FPGA (PCIe + 12 layers) | Yes |
| **slave** | `hw/quartus/slave/` | Slave FPGA (12 layers, no PCIe) | Yes |
| **bringup** | `hw/quartus/bringup/` | Fast-iteration bring-up | No |
| **dsp_char** | `hw/quartus/dsp_char/` | DSP characterization (fp4_mac chain) | No |
| **hbm_char** | `hw/quartus/hbm_char/` | HBM bandwidth characterization | No |
| **pcie_test** | `hw/quartus/pcie_test/` | PCIe R-Tile standalone test | No |
| **c2c_test** | `hw/quartus/c2c_test/` | C2C ring standalone test | No |
| **full_stack** | `hw/quartus/full_stack/` | All IP integrated (full system) | Yes |
| **fpga_lpu** | `hw/quartus/` (legacy) | Original monolithic project | Yes |

**ALL projects include `common/common_modules.qsf`** for the shared RTL file list.

### 7.2 Building a Bring-Up Bitstream (Step by Step)

Bring-up is for fast iteration. DSP at 100 MHz, tiny model dimensions, quick compile.

```bash
# 1. Open Quartus Pro 24.3 GUI
quartus &
# File -> Open Project -> hw/quartus/bringup/fpga_lpu_bringup.qpf

# Or command line:
cd hw/quartus/bringup
quartus_sh --flow compile fpga_lpu_bringup
```

**Expect:** ~30 minute compile. Output in `output_files/fpga_lpu_bringup.sof`.

**Checklist before building:**
- All RTL files in `common_modules.qsf` exist and compile cleanly
- No legacy modules uncommented (check `common_modules.qsf` legacy section)
- Icarus simulations pass for the modules you changed

### 7.3 Building a Production Bitstream

Production: 450 MHz DSP, full DeepSeek V4 Pro dimensions. 4-6 hours on c6i.16xlarge.

```bash
# Master FPGA (Chip 0, PCIe enabled)
cd hw/quartus/master
quartus_sh --flow compile fpga_lpu_master

# Slave FPGA (Chip 1-31, no PCIe)
cd hw/quartus/slave
quartus_sh --flow compile fpga_lpu_slave
```

**Production compile stages:**
1. **Analysis & Synthesis** (~30 min): Elaborates full production parameters (H=7168, E=384)
2. **Fitter (Place & Route)** (~2-3 hours): Maps to Agilex 7 M die, 450 MHz timing closure
3. **Assembler** (~15 min): Generates .sof programming file
4. **Timing Analyzer** (~15 min): Reports setup/hold/clock domain crossing

### 7.4 Incremental Compilation

For targeted changes (modifying one module) in production builds:

```bash
# 1. Full initial compile (establishes partition database)
quartus_sh --flow compile fpga_lpu_master

# 2. After RTL change, mark partition for recompilation:
# In Quartus GUI: Assignments -> Design Partitions Window
# Set netlist type of your changed module to "Source File"
# Recompile (only changed partition + recomposition)
quartus_sh --flow compile fpga_lpu_master --incremental

# This can reduce recompile time from 6 hours to ~45 minutes.
```

**Incremental compile tips:**
- Keep partition boundaries at module interfaces (chip_top, full_transformer_layer)
- Do NOT modify interface structs (in `rtl/interfaces/`) — this invalidates all partitions
- Do NOT change `lpu_config.svh` — this invalidates all partitions

### 7.5 Pin Assignment

**Master FPGA** (chip 0 of each card):
- PCIe R-Tile: dedicated transceiver bank
- HBM2e UIB: dedicated HBM interface pins
- C2C: F-Tile transceivers (8 lanes: 4 for Ring A, 4 for Ring B)
- Board clock: `PIN_AA25` (100 MHz oscillator)
- Debug: LEDs (4), UART TX (1)

**Slave FPGA** (chips 1-3):
- HBM2e UIB: dedicated HBM interface pins
- C2C: F-Tile transceivers (8 lanes)
- Board clock: `PIN_AA25`
- Debug: LEDs (4), UART TX (1)
- No PCIe pins needed

**DIP switch:** Slaves can read a 5-bit DIP switch for CHIP_ID identification (pins TBD in board schematics). Configuration also arrives via C2C CTRL packets from the master.

### 7.6 Timing Closure Targets

| Clock Domain | Frequency | Constraint Type | Notes |
|-------------|-----------|-----------------|-------|
| `clk_dsp` | 450 MHz | `create_generated_clock` (PLL: 100x9/2) | DSP + HBM share this domain |
| `clk_hbm` | 450 MHz | `create_generated_clock` (HBM refclk) | Same PLL as clk_dsp |
| `clk_pcie` | 250 MHz | `create_generated_clock` (PLL: 100x5/2) | R-Tile reference |
| `clk_board_100m` | 100 MHz | `create_clock -period 10.000` | System/control |

**CDC boundaries** (from `fpga_lpu.sdc`):
- `clk_dsp` and `clk_hbm`: **Same domain** (same PLL, 450 MHz both). No CDC needed between compute and HBM.
- `clk_pcie` (250 MHz) <-> `clk_board_100m` (100 MHz): 2-FF synchronizers on control signals.
- `clk_dsp` (450 MHz) <-> `clk_board_100m` (100 MHz): Single-bit status signals, double-synchronized.

**Multicycle exceptions:**
- Accumulator registers in `fp4_mac`: setup relaxed to 2 cycles (accumulator toggles only on valid)
- Static config registers (`cfg_*`): false path (loaded once, stable during operation)

### 7.7 Signal Tap Setup for On-Board Debug

Signal Tap II can be added to any Quartus project for real-time logic analysis:

```bash
# 1. In Quartus GUI, open your project
# 2. File -> New -> Signal Tap Logic Analyzer File
# 3. Add signals of interest:
#    - clk_dsp for trigger clock
#    - fp4_mac accumulator values
#    - C2C valid/ready handshake signals
#    - FSM state registers
# 4. Set trigger conditions and sample depth
# 5. Recompile with Signal Tap embedded
# 6. Program FPGA and run: Tools -> Signal Tap Logic Analyzer
```

**Important:** Signal Tap uses FPGA logic and BRAM resources. On production builds, you may need to reduce sample depth or signal count to fit alongside the full design. For bring-up debug, sample depth of 4K-16K samples is typically fine.

False paths are set for Signal Tap debug logic in `fpga_lpu.sdc`:
```tcl
set_false_path -to [get_pins sld_signaltap:*]
```

---

## 8. RTL Module Development Workflow

This is the standard workflow for developing or modifying any RTL module. The goal is fast iteration (30-second sim cycles) with confidence that the code will work at production scale.

### Step 1: Write or Update RTL

- Edit files in `rtl/<category>/<module>.sv`
- Follow coding conventions (Section 3)
- Ensure module defaults from `lpu_config_pkg` for production compatibility
- Add/update corresponding testbench in `rtl/sim/tb_<module>.sv`

### Step 2: Run Icarus Simulation with Bring-Up Parameters

```bash
cd rtl/sim
make SIM=iverilog tb_<module>
# Expect: ~5-30 seconds for bring-up scale
# Output: "PASS" or assertion failure with line number
```

**Debug cycle:** Edit RTL -> `make SIM=iverilog tb_module` -> view waveforms. Target < 1 minute per cycle.

### Step 3: Generate Golden Vectors from Python Reference

If your module has a Python reference model:

```bash
cd scripts/simulation
# Update the relevant Python model if needed
# Regenerate golden vectors
python gen_<module>_vectors.py
```

For new modules without Python models, write one. The golden vector approach catches:
- Fixed-point precision mismatches (Q12 vs FP32 rounding)
- Edge cases (zero inputs, saturation, underflow)
- Pipeline timing issues (valid/data alignment)

### Step 4: Compare RTL Output vs Golden Reference

```bash
cd rtl/sim

# Run golden comparison testbench
make SIM=iverilog tb_<module>_golden

# Or add golden comparison to your existing testbench:
#   assert (rtl_result == golden_result) else $error(...);
```

**Tolerance:** When comparing fixed-point RTL output vs FP32 Python golden, allow +/- 1 LSB tolerance for rounding differences. At Q12 precision, this is +/- 1/4096 = +/- 0.00024.

### Step 5: Run Quartus Synthesis (Timing Check)

Once the RTL is functionally correct in simulation, verify it synthesizes and meets timing:

```bash
# Synthesis only (no fitter — faster, ~10 min)
cd hw/quartus/bringup
quartus_syn fpga_lpu_bringup

# Check synthesis warnings for:
# - Inferred latches (always unintended, fix immediately)
# - Width mismatches (truncation without explicit cast)
# - Unused ports/parameters
# - DSP inference failures (did synthesis map to DSP blocks?)
```

**Critical: Check the synthesis report for DSP inference:**

```
# In output_files/fpga_lpu_bringup.map.rpt or .fit.rpt:
# Look for "DSP Block Usage" section
# Verify fp4_mac instances use DSP blocks, not LUT-based multipliers
```

### Step 6: Full Quartus Compile (Overnight for Production)

When the module is verified in bring-up and passes synthesis:

```bash
# Option A: Bring-up compile (fast, ~30 min)
cd hw/quartus/bringup
quartus_sh --flow compile fpga_lpu_bringup

# Option B: Production compile (slow, 4-6 hours)
cd hw/quartus/master
quartus_sh --flow compile fpga_lpu_master
# Run overnight or on cloud instance
```

**Post-compile checklist:**
1. Check timing report: `report_timing -setup -npaths 100` -> all paths >= 0 slack at 450 MHz
2. Check resource utilization: DSP blocks, BRAM, ALMs within device limits
3. Check CDC report: `report_clock_transfer` -> no unsynchronized crossings
4. Check fitter warnings: no unconstrained I/O, no hold violations

### Summary Timeline

| Step | Duration | When |
|------|----------|------|
| 1. Write RTL | 1-4 hours | Once per module |
| 2. Icarus sim | 30 seconds | Every edit (dozens of times) |
| 3. Golden vectors | 2 minutes | After functional verification |
| 4. Golden comparison | 30 seconds | After step 3 |
| 5. Quartus synthesis | 10 minutes | Before commit |
| 6a. Bring-up compile | 30 minutes | Pre-merge validation |
| 6b. Production compile | 4-6 hours | Overnight / before board test |

---

## 9. Key Design Patterns

### 9.1 Systolic Array Parametrization

The 2D systolic array (`fp4_systolic_2d.sv`) is parametrized by `LPU_ARRAY_LANES` (K-direction, number of columns) and `LPU_ARRAY_M_ROWS` (M-direction, number of rows).

```
                    weight flow (left -> right)
                    ---->
    +---------+    +---------+    +---------+
    | cell[0,0]| -> | cell[0,1]| -> | cell[0,2]| -> ...  M=0
    +---------+    +---------+    +---------+
         |              |              |
    +---------+    +---------+    +---------+
    | cell[1,0]| -> | cell[1,1]| -> | cell[1,2]| -> ...  M=1
    +---------+    +---------+    +---------+
         |              |              |
activation flow (top -> bottom)
```

**Each cell** (`fp4_systolic_cell.sv`):
- Receives weight from left neighbor, activation from top neighbor
- Performs fp4_mac (4-stage pipeline internally)
- Passes weight right, passes activation down
- Passes partial sum down with accumulation

**Production scale:** 128 lanes x 32 rows = 4096 cells. Each cell uses 2 DSPs = 8192 DSPs. The AGMF039 has ~12,000 DSPs (est.), so this fits with margin.

### 9.2 Pipeline Stage Handshake

Modules with internal pipelines use a simple register-based handshake:

```systemverilog
// Pipeline advance condition: either downstream accepts, or stage is empty
logic pipe_advance;
assign pipe_advance = out_valid ? out_ready : 1'b1;

always_ff @(posedge clk) begin
    if (pipe_advance) begin
        s1_data <= s0_data;
        s1_valid <= s0_valid;
    end
end
```

This is a simplified variant of skid-buffer flow control. It works because most downstream modules always accept data (`out_ready = 1'b1`).

### 9.3 BRAM-Based Scale Factor Lookup

In FP4 GEMM, weights are grouped into blocks of 16. Each block of 16 FP4 weights shares one FP8 scale factor. The scale factor reader (`fp4_scale_reader.sv`) pre-fetches and pre-decodes the scale:

```systemverilog
// At scale load time (not on critical path):
// Read FP8 scale from BRAM
// Convert FP8 -> 12-bit signed scaled integer (x256)
// Store pre-decoded scale in register

// At compute time:
// MAC receives: fp4 weight + PRE-DECODED scale + fp8 activation
// No scale decode in MAC critical path
```

**Implementation:** `fp8_to_scaled12()` from `fp4_types.svh` is called at scale load time only. The BRAM stores the original FP8 scales. The pre-decoded 12-bit scales are in a register file alongside the MAC array.

### 9.4 Q12 Fixed-Point Accumulation and FP8 Conversion

**Internal data format: Q12** (32-bit signed, 12 fractional bits)
- Range: approximately [-524288, 524287.99975]
- Precision: 1/4096 = 0.000244
- Used for: hidden states, attention outputs, accumulator values

**Format conversion points:**

```
HOST (FP32/BF16) --[PCIe DMA + quantize]--> Q12 (internal)
                                               |
Q12 --[q12_to_fp8_e4m3.sv]--> FP8 E4M3 (FFN activation input)
                                |
FP8 activation x FP4 weight = Q12 product
                                |
Q12 --[accumulate]--> Q12 accumulator
                                |
FP8 output x FP4 weight --[dequantize]--> Q12
                                               |
Q12 --[PCIe DMA + dequantize]--> HOST (FP32/BF16)
```

**Key functions in `fp4_types.svh`:**

| Function | Input | Output | Use |
|----------|-------|--------|-----|
| `fp4_mag_to_scaled(mag)` | 3-bit magnitude | 6-bit x16 scaled | Weight decode in Stage 1 |
| `fp8_to_scaled12(fp8)` | 8-bit FP8 E4M3 | 12-bit signed x256 | Scale pre-decode at load time |
| `decode_fp8(fp8)` | 8-bit FP8 E4M3 | `fp8_decoded_t` | Activation decode in Stage 1 |

### 9.5 KV Cache Address Generation

The KV cache stores K_latent and V_latent (each 512-dim in production) for each token position. Address generation is linear:

```
KV_slot_addr = token_position % NUM_SLOTS
```

Each slot holds:
- `K_latent`: K_LATENT x DATA_W bits (512 x 32 = 16384 bits in production)
- `V_latent`: V_LATENT x DATA_W bits (512 x 32 = 16384 bits in production)

**Total KV cache per layer (production):**
4096 slots x (16384 + 16384) bits = 128 Mbits = 16 MB per layer
61 layers x 16 MB = 976 MB total KV cache (distributed across 32 chips)

At bring-up scale:
64 slots x (128 + 128) bits = 16 Kbits per layer
12 layers x 16 Kbits = 192 Kbits total (fits in on-chip BRAM entirely)

### 9.6 Saturation Arithmetic

The accumulator in `fp4_mac.sv` uses saturation (not wrap-around) to prevent overflow artifacts on deep accumulations:

```systemverilog
function automatic logic [ACCUM_WIDTH-1:0] sat_acc;
    input [ACCUM_WIDTH-1:0] old_acc, add_val;
    // old_sign=0, val_sign=0, sum_sign=1  => positive overflow => saturate to max positive
    // old_sign=1, val_sign=1, sum_sign=0  => negative overflow => saturate to max negative
    // else => normal sum
endfunction
```

This is inspired by TALOS-V2 saturation at 16 bits and prevents a single overflow from corrupting the entire accumulated dot product.

---

## 10. Common Pitfalls and FAQ

### 10.1 Common Pitfalls

#### P1: Forgetting `-g2012` with Icarus
```
Error: syntax error near 'package'
Error: syntax error near 'typedef'
```
**Fix:** Always use `iverilog -g2012`. The Makefile handles this automatically. If calling `iverilog` directly, add the flag.

#### P2: Including Production-Scale Parameters in Icarus Sim
```
ERROR: Memory overflow at array allocation (7168 x 7168 x 384)
```
**Fix:** Ensure `FPGA_LPU_PRODUCTION` is NOT defined when running Icarus. Check:
```bash
# If you see FPGA_LPU_PRODUCTION in your environment:
echo $IVERILOG_FLAGS
# Should NOT contain -DFPGA_LPU_PRODUCTION
```

#### P3: Modifying Interface Structs Without Updating All Consumers
Changing `stream_beat_t` in `avalon_stream.svh` affects every module that includes it. After any interface change:
1. Recompile ALL testbenches (not just the one you are working on)
2. Run `make SIM=iverilog all` to catch compile errors in dependent modules
3. Check Quartus synthesis for all projects (master, slave, bringup)

#### P4: DSP Inference Failures
If Quartus synthesizes `fp4_mac` multipliers as LUTs instead of DSP blocks:
- Check that you used the `(* altera_attribute *)` annotations on the module
- Verify the multiply operands match DSP block capability (18x19 or 27x27 mode)
- In production, check `clk_dsp = 450 MHz` — if timing fails, Quartus may fall back to LUTs

#### P5: Width Mismatch in Accumulator
```systemverilog
// WRONG: 20-bit product into 20-bit accumulator — overflow on 2nd MAC
logic [19:0] acc;
acc <= acc + s3_product;   // wraps on 2nd accumulation

// CORRECT: 32-bit accumulator (parameterizable)
logic [ACCUM_WIDTH-1:0] acc;
acc <= sat_acc(acc, s3_product);   // saturates on overflow
```

#### P6: Unintentional Latch Inference
```systemverilog
// WRONG: missing else branch, synthesis infers latch
always_comb begin
    if (state == S_ACTIVE) out_data = in_data;
    // else: out_data retains previous value -> LATCH
end

// CORRECT: always_ff for registered outputs, or complete always_comb
always_ff @(posedge clk) begin
    if (state == S_ACTIVE) out_data <= in_data;
end
```

#### P7: Clock Domain Crossing Without Synchronizer
The SDC sets `clk_dsp = clk_hbm` (same domain), but `clk_pcie` and `clk_board_100m` are asynchronous. Any signal crossing between these domains MUST use a 2-FF synchronizer:

```systemverilog
// 2-FF synchronizer for single-bit control signals
logic [1:0] sync_reg;
always_ff @(posedge clk_dst) begin
    sync_reg <= {sync_reg[0], signal_from_src_domain};
end
wire signal_synced = sync_reg[1];
```

For multi-bit data, use an async FIFO (not a synchronizer).

### 10.2 FAQ

**Q: Why are the parameter defaults in lpu_config_pkg and not in each module?**

To ensure every module in the design uses the same architectural dimensions. If every module independently defaulted HIDDEN=8 vs HIDDEN=7168, parameter mismatches would be nearly impossible to debug. The package is the single source of truth.

**Q: Can I override a parameter for just my simulation?**

Yes. All modules support per-instance parameter overrides:

```systemverilog
// Override to test at half-scale:
full_transformer_layer #(.HIDDEN(4), .NUM_SLOTS(32)) u_layer (...);
```

This overrides the package default ONLY for this instance.

**Q: Why do we use packed structs for interfaces instead of SystemVerilog interfaces?**

Portability. Icarus Verilog has limited support for `interface` / `modport`. Packed structs work with all three simulators (Icarus, Questa, Verilator) and synthesize identically. When Icarus support improves, we may migrate.

**Q: What happens if I modify a file in rtl/legacy/?**

Nothing — those files are excluded from production QSF files. They exist only for reference. If you find a bug in a legacy module, fix it in the v2 replacement module instead.

**Q: How do I add a new module to the project?**

1. Create `rtl/<category>/<module_name>.sv`
2. Add it to `hw/quartus/common/common_modules.qsf`:
   ```tcl
   set_global_assignment -name SYSTEMVERILOG_FILE ../../rtl/<category>/<module_name>.sv
   ```
3. Create `rtl/sim/tb_<module_name>.sv`
4. Add a make target or use the existing pattern to run it

**Q: How do I determine the number of pipeline stages needed?**

Rule of thumb: at 450 MHz (2.22 ns period), each pipeline stage should contain at most ~4-6 levels of LUT logic (Agilex 7 ALM has LUT6). DSP blocks add 1 cycle each. Count:
- Combinational decode: 2-3 LUT levels (1 stage)
- DSP multiply: 1 DSP cycle (1 stage)
- Wide addition: 3-4 LUT levels + carry chain (1 stage)

The `fp4_mac` 4-stage pipeline is: Input Reg -> Decode -> Base Multiply (DSP) -> Scale Multiply (DSP) -> Accumulate. Each stage meets timing at 450 MHz.

**Q: When should I use the cloud build vs local Quartus?**

| Scenario | Recommendation |
|----------|---------------|
| Synthesis-only check | Local (10 min) |
| Bring-up bitstream | Local (30 min) |
| Production bitstream | Cloud c6i.16xlarge (4-6 hours) |
| Repeated production builds | Cloud spot instance |

**Q: How do I check the FPGA resource utilization after a compile?**

Open the fitter report: `output_files/fpga_lpu_master.fit.rpt` and check:
- "Fitter Resource Usage Summary" section
- DSP blocks: should match expected count (N_cells x 2)
- M20K blocks: for BRAM weight buffers, KV cache, scale LUTs
- ALMs: total logic utilization

Or use:
```bash
quartus_fit --report_utilization fpga_lpu_master
```

**Q: What if I need to debug a production-scale timing failure?**

1. Identify the failing path from the timing report
2. Add one pipeline stage to that path
3. Re-run bring-up simulation to verify correctness
4. Run incremental Quartus synthesis to verify timing improvement
5. If fixed, run overnight production compile

Never add pipeline stages in production mode without verifying functional correctness in bring-up simulation first.

---

## Appendix A: Quick-Reference Command Cheat Sheet

```bash
# === SIMULATION ===
# Run single testbench (Icarus)
cd rtl/sim && make SIM=iverilog tb_fp4_mac

# Run all testbenches
make SIM=iverilog all

# Run with Questa
make SIM=questa run

# Run with GUI
make SIM=questa gui

# Verilator lint
make SIM=verilator lint

# Golden vector generation
cd scripts/simulation
python gen_tb_vectors.py
python gen_ffn_tb_vectors.py
python gen_layer_golden.py

# === QUARTUS ===
# Synthesis only (quick timing check)
cd hw/quartus/bringup && quartus_syn fpga_lpu_bringup

# Full bring-up flow
cd hw/quartus/bringup && quartus_sh --flow compile fpga_lpu_bringup

# Full production flow (master)
cd hw/quartus/master && quartus_sh --flow compile fpga_lpu_master

# Full production flow (slave)
cd hw/quartus/slave && quartus_sh --flow compile fpga_lpu_slave

# Incremental compile after RTL change
quartus_sh --flow compile fpga_lpu_master --incremental

# Timing report
quartus_sta -t timing_report.tcl

# === PYTHON VALIDATION ===
# Module smoke tests
python scripts/run_module_smoke.py

# Full validation suite
python scripts/run_all_validations.py

# End-to-end architecture validation
python scripts/run_e2e_validation.py
```

## Appendix B: Key File Reference

| Purpose | File Path |
|---------|-----------|
| Central config (bring-up vs production) | `rtl/include/lpu_config.svh` |
| FP4 type definitions + decode LUTs | `rtl/include/fp4_types.svh` |
| FP4 datapath parameters | `rtl/include/fp4_params.svh` |
| Streaming interface structs | `rtl/interfaces/avalon_stream.svh` |
| C2C packet format | `rtl/interfaces/c2c_packet.svh` |
| PCIe DMA descriptor format | `rtl/interfaces/pcie_dma.svh` |
| Fundamental MAC unit | `rtl/dsp/fp4_mac.sv` |
| 2D systolic array | `rtl/dsp/fp4_systolic_2d.sv` |
| GEMM engine | `rtl/dsp/fp4_gemm_engine.sv` |
| MLA attention v2 (production) | `rtl/attention/mla_attention_v2.sv` |
| MoE router | `rtl/moe/router_topk.sv` |
| Expert FFN | `rtl/moe/expert_ffn_engine_fp4_down.sv` |
| Full transformer layer | `rtl/layer/full_transformer_layer.sv` |
| Chip top | `rtl/chip/chip_top.sv` |
| KV DMA engine | `rtl/chip/kv_dma_engine.sv` |
| Master board top | `hw/src/top_master.sv` |
| Slave board top | `hw/src/top_slave.sv` |
| Shared RTL file list | `hw/quartus/common/common_modules.qsf` |
| Timing constraints | `hw/constraints/fpga_lpu.sdc` |
| Simulation Makefile | `rtl/sim/Makefile` |
| Quartus project README | `hw/quartus/README.md` |

---

**Document version:** 1.0 -- 2026-05-28

**Maintainer:** FPGA LPU RTL Team. Update this guide when conventions change or new modules are added.
