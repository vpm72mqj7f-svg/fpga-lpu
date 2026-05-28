# FPGA LPU Verification Engineer Guide

**Audience:** 3 overseas verification/test engineers  
**Project:** FPGA LLM Inference Cluster (DeepSeek V4 Pro)  
**Target FPGA:** Intel Agilex 7 M-Series (AGMF039R47A), 32-chip cluster

---

## Table of Contents

1. [Verification Philosophy](#1-verification-philosophy)
2. [Testbench Infrastructure](#2-testbench-infrastructure)
3. [Golden Model Methodology](#3-golden-model-methodology)
4. [Unit Test Strategy](#4-unit-test-strategy)
5. [Integration Test Strategy](#5-integration-test-strategy)
6. [Python Simulation Validation](#6-python-simulation-validation)
7. [On-Board Validation](#7-on-board-validation)
8. [Go/No-Go Criteria](#8-gono-go-criteria)
9. [Regression and CI Strategy](#9-regression-and-ci-strategy)
10. [Bug Tracking and Reporting](#10-bug-tracking-and-reporting)

---

## 1. Verification Philosophy

### 1.1 Core Principle: Three Parallel Tracks

Our verification strategy runs three tracks **simultaneously**, not sequentially:

| Track | What | Runs Where | Goal |
|-------|------|------------|------|
| **Module Simulation** | Unit + integration testbenches | Desktop (Icarus/Questa) | RTL correctness, bit-exact golden comparison |
| **Python Simulation** | Architecture + serving models | Desktop (Python/NumPy) | System-level throughput, scheduling, precision |
| **On-Board Validation** | Hardware tests | DK-DEV-AGM039EA FPGA | Electrical timing, HBM BW, DSP real behavior |

### 1.2 What We Do NOT Simulate

**We do NOT run full-system cycle-accurate simulation of 384 layers.**

A single token through 384 layers at production clock (450 MHz DSP = 2.22 ns) would require roughly 182,208 cycles per token (from the bring-up simulation baseline at 100 MHz). Running 100 tokens through a full simulation would require:

```
100 tokens x 182,208 cycles/token x 2 ns/cycle = 36.4 ms simulated
But sim runs at ~100 Hz real-time for a design this size
  -> 36.4 ms simulated / 100 Hz = 3.64e8 seconds = ~11.5 years
```

This is infeasible for any gate-level simulator. Instead, our strategy is:

1. **Unit-level cycle-accurate**: Every DSP module is simulated exhaustively with golden vectors (tens of microseconds of sim time)
2. **Integration-level functional**: Full layer pipeline runs with bring-up parameters (HIDDEN=8, NUM_SLOTS=64) in under 5 minutes
3. **Python-level system**: The entire 32-chip, 384-layer, 30-card cluster is modeled in Python/NumPy with high fidelity
4. **Hardware-level**: On-board bring-up validates what simulation cannot (DSP rounding, HBM controller behavior, timing closure)

### 1.3 Bring-Up vs. Production Parameters

Code in `rtl/sim/` testbenches use **bring-up parameters** for fast simulation turnaround:

| Parameter | Bring-Up Value | Production Value |
|-----------|---------------|-----------------|
| HIDDEN | 8 | 7168 |
| INTER (FFN intermediate) | 4 | 3072 |
| LANES (systolic array width) | 4 | 128 |
| K_BEATS (fp4 accumulation beats) | 2 | 512 |
| NUM_SLOTS (KV cache slots) | 64 | 4096 |
| MAX_POS (RoPE positions) | 64 | 163840 |
| WEIGHT_W | 16 | 16 |
| DATA_W | 32 | 32 |

**All testbenches** use these reduced dimensions. Production parameters are used only for bitstream synthesis. This is critical: simulation with production parameters would take weeks per run and is never done.

### 1.4 The Golden Model Cycle

The fundamental verification workflow is a closed loop:

```
  +-------------------+
  | Python Reference  |  (scripts/simulation/*.py)
  | (independent impl)|
  +--------+----------+
           |
           v  gen_tb_vectors.py / gen_ffn_tb_vectors.py / gen_layer_golden.py
  +--------+----------+
  | Golden SV Package |  (rtl/sim/tb_golden_pkg.sv, etc.)
  | (auto-generated)  |
  +--------+----------+
           |
           v  `include in testbench
  +--------+----------+
  | RTL Testbench     |  (rtl/sim/tb_*.sv)
  | (reads golden pkg)|
  +--------+----------+
           |
           v  runs DUT, compares output vs. expected
  +--------+----------+
  | PASS / FAIL       |
  +-------------------+
```

**Key design property**: The Python golden model and the RTL share **zero code**. The Python model is an independent re-implementation of the same arithmetic. This avoids shared-bug risk.

---

## 2. Testbench Infrastructure

### 2.1 Toolchain Selection

Three simulator options are supported through `rtl/sim/Makefile`:

| Simulator | SIM= Value | Licensing | SV Support | Recommended For |
|-----------|-----------|-----------|------------|-----------------|
| Intel Questa Starter | `questa` | Free (Intel FPGA) | Full | Primary workflow |
| Icarus Verilog | `iverilog` | Open source | Limited | Quick checks, CI |
| Verilator | `verilator` | Open source | Full (lint) | Lint checks |

**Default is Questa.** The guide assumes Questa unless stated otherwise.

### 2.2 Simulation Directory Structure

```
rtl/
  sim/                          <-- All testbenches live here
    Makefile                    <-- Build system
    tb_fp4_mac.sv              <-- Unit testbenches
    tb_fp4_scale_reader.sv
    tb_fp4_systolic_tile.sv
    tb_fp4_scaled_tile.sv
    tb_fp4_systolic_array.sv
    tb_fp4_linear_engine.sv
    tb_fp4_gemm_engine.sv
    tb_fp4_systolic_2d.sv
    tb_fp4_prefill_engine.sv
    tb_cell_mini.sv
    tb_2d_mini.sv
    tb_2d_1x4.sv
    tb_2x2.sv
    tb_2d_4x4.sv
    tb_rms_norm.sv
    tb_silu_q12_lut.sv
    tb_mla_qkv.sv               <-- Module-level testbenches
    tb_mla_attention.sv
    tb_mla_attention_v2.sv
    tb_router_topk.sv
    tb_expert_ffn_engine.sv
    tb_expert_ffn_engine_fp4_down.sv
    tb_expert_ffn_engine_fp4_down_golden.sv
    tb_mhc_mixer.sv
    tb_layer_compute_engine.sv  <-- Integration testbenches
    tb_layer_compute_engine_golden.sv
    tb_full_transformer_layer.sv
    tb_chip_12layer.sv          <-- System testbenches
    tb_cluster_384.sv
    tb_lookup_engine.sv
    tb_mtp_head.sv
    tb_c2c_ring.sv
    tb_kv_dma.sv
    tb_golden_pkg.sv            <-- Auto-generated golden data
    tb_ffn_golden_pkg.sv
    tb_layer_golden_pkg.sv
  include/
    fp4_types.svh               <-- Shared type definitions
  dsp/
    fp4_mac.sv                  <-- DUT source files
    ...
```

### 2.3 Basic Usage

#### Building and Running One Testbench

```bash
# Navigate to simulation directory
cd D:\workspace\fpgalpu\rtl\sim

# Using Questa (default):
make TOP=tb_fp4_mac compile   # Compiles DUT + testbench
make TOP=tb_fp4_mac run       # Compiles and runs in batch mode
make TOP=tb_fp4_mac gui       # Compiles and opens waveform GUI

# Using Icarus Verilog:
make SIM=iverilog TOP=tb_fp4_mac run

# Clean build artifacts:
make clean
```

#### Expected Output (Passing)

```
============================================================
 tb_fp4_mac -- Golden Vector Verification
============================================================
 Pipeline: 3-stage, 14 tests from tb_golden_pkg
============================================================

[ OK ] T1  single multiply         (0x00001000)
[ OK ] T2  4-term accumulation     (0x00001400)
[ OK ] T3  pos fp4 sweep (x8)      (0x0000a000)
...
[ OK ] T14 non-unity scale         (0x00008000)

============================================================
 ALL 15 TESTS PASSED
============================================================
```

#### Expected Output (Failing)

```
[FAIL] T5  mixed signs
       got:      0x00002000 (8192)
       expected: 0x00001000 (4096)
       diff:     0x00001000
```

### 2.4 Waveform Viewing with GTKWave

For Questa/ModelSim, use the GUI mode or add `$dumpfile`/`$dumpvars` to the testbench for VCD generation:

```systemverilog
// Add near top of testbench initial block:
initial begin
    $dumpfile("tb_fp4_mac.vcd");
    $dumpvars(0, tb_fp4_mac);
end
```

Then open with GTKWave:
```bash
gtkwave tb_fp4_mac.vcd
```

Key signals to inspect in `tb_fp4_mac`:
- `clk` -- clock
- `mac_in.valid`, `mac_in.weight`, `mac_in.activ`, `mac_in.scale` -- input handshake
- `mac_out.result` -- after pipeline drain
- `accum_clr` -- accumulator reset
- `pass_count`, `fail_count` -- real-time pass/fail counters

### 2.5 Golden Package Files (.svh / .sv)

The `tb_golden_pkg.sv`, `tb_ffn_golden_pkg.sv`, and `tb_layer_golden_pkg.sv` files are **auto-generated** SystemVerilog packages. They are `include`d directly into testbenches:

```systemverilog
`include "tb_golden_pkg.sv"
```

These packages contain `localparam` constants with packed test vectors and expected results. For example:

```systemverilog
package tb_golden_pkg;
    localparam int NUM_TESTS = 14;
    localparam int T0_LEN = 1;
    localparam logic [3:0] T0_W_PACK = {4'h4};
    localparam logic [7:0] T0_A_PACK = {8'h38};
    localparam logic [7:0] T0_S_PACK = {8'h38};
    localparam logic [31:0] T0_EXPECTED = 32'h00001000;
    ...
endpackage
```

**Important**: These files must be regenerated any time the RTL arithmetic changes. See Section 3.

### 2.6 Adding a New Testbench (Step by Step)

To add a new unit testbench for module `my_new_module.sv`:

**Step 1**: Create the testbench file `rtl/sim/tb_my_new_module.sv`:

```systemverilog
`timescale 1ns/1ps
`include "fp4_types.svh"

module tb_my_new_module;
    localparam int CLK_PERIOD = 10;

    logic clk, rst_n;
    // Declare DUT ports...

    my_new_module dut (.*);

    // Clock
    always #(CLK_PERIOD/2) clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    initial begin
        clk = 0; rst_n = 0;
        // Initialize inputs to 0...
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        $display("============================================");
        $display(" tb_my_new_module");
        $display("============================================");

        // --- Test Case 1 ---
        // Drive inputs...
        // Check outputs...

        // --- Results ---
        $display("");
        if (fail_count == 0)
            $display(" ALL %0d TESTS PASSED", pass_count);
        else
            $display(" %0d PASSED, %0d FAILED", pass_count, fail_count);
        $finish;
    end

    // Watchdog (prevents infinite sim hang)
    initial begin
        #500000;
        $error("TIMEOUT");
        $finish;
    end
endmodule
```

**Step 2**: Add a Makefile target (optional; use generic compile commands):

```makefile
# In rtl/sim/Makefile:
TB_MY_MODULE_SRC := $(SIM_DIR)/tb_my_new_module.sv
DUT_MY_MODULE_SRC := $(RTL_ROOT)/dsp/my_new_module.sv

tb_my_new_module: compile_my_new_module
    $(QUESTA_EXE) -c -do "run -all; quit" tb_my_new_module

compile_my_new_module:
    $(VLOG_EXE) +define+SIM +incdir+$(INC_DIR) $(DUT_MY_MODULE_SRC)
    $(VLOG_EXE) +define+SIM +incdir+$(INC_DIR) $(TB_MY_MODULE_SRC)
```

**Step 3**: If your module needs golden comparison vectors, create a Python generator:

```python
# scripts/simulation/gen_my_new_module_vectors.py
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# Implement RTL-matched arithmetic here...
# Generate expected outputs for test cases...

def write_sv_package(test_cases, filepath):
    with open(filepath, "w") as f:
        f.write("package tb_my_golden_pkg;\n")
        for i, tc in enumerate(test_cases):
            f.write(f"    localparam logic [31:0] C{i}_EXPECTED = 32'h{tc['expected']:08x};\n")
        f.write("endpackage\n")

if __name__ == "__main__":
    outpath = os.path.join(os.path.dirname(__file__), "..", "..", "rtl", "sim", "tb_my_golden_pkg.sv")
    outpath = os.path.abspath(outpath)
    write_sv_package(test_cases, outpath)
```

**Step 4**: Wire the golden package into your testbench:

```systemverilog
`include "tb_my_golden_pkg.sv"

// Use: tb_my_golden_pkg::C0_EXPECTED, etc.
```

---

## 3. Golden Model Methodology

This is the core of our verification approach. Every question about "how do we know the RTL is correct?" is answered by: **the RTL produces the same result as the independent Python reference model.**

### 3.1 Python Reference Models

Located in `scripts/simulation/`, these implement the exact same arithmetic as the RTL:

| Python Script | RTL Module(s) Verified | What It Models |
|--------------|----------------------|----------------|
| `gen_tb_vectors.py` | `fp4_mac.sv` | fp4 x fp8 multiply-accumulate |
| `gen_ffn_tb_vectors.py` | `expert_ffn_engine_fp4_down.sv` | FFN gate/up/down projections + SiLU + fp8 requant |
| `gen_layer_golden.py` | `layer_compute_engine.sv` | RMSNorm -> Router -> FFN -> RMSNorm full layer |
| `fp4_utils.py` | All fp4 modules | fp4 E2M1 encode/decode, quantize/dequant, GEMM |
| `mla_attention.py` | `mla_attention_v2.sv` | MLA attention: KV compression, Q loRA, attention |
| `moe_router.py` | `router_topk.sv` | MoE router: Top-K softmax/sqrt-softplus |
| `transformer_layer.py` | `full_transformer_layer.sv` | BF16 + fp4 full layer (PyTorch) |
| `verify_fp4_mac_stages.py` | `fp4_mac.sv` | Per-stage exhaustive verification |

### 3.2 How Golden Vectors Are Generated

The flow for `tb_fp4_mac.sv`:

```bash
cd D:\workspace\fpgalpu
python scripts/simulation/gen_tb_vectors.py
```

This produces output:
```
============================================================
 Golden Test Vector Summary
============================================================
  T1_SINGLE             n= 1  accum=0x00001000 (+4096)  float=+1.000000
  T2_ACCUM4             n= 4  accum=0x00001400 (+5120)  float=+1.250000
  T3_POS_SWEEP          n= 8  accum=0x0000a000 (+40960) float=+10.000000
  ...
Wrote D:\workspace\fpgalpu\rtl\sim\tb_golden_pkg.sv (14 test cases)

  Run: python scripts/simulation/gen_tb_vectors.py
```

What happened:
1. `gen_tb_vectors.py` defined 14 test cases (T0 through T13)
2. Each test case contains: fp4 weights, fp8 activations, optional fp8 scales
3. For each test case, `compute_accum()` looped through the weight/activation pairs, computing the **exact same arithmetic the RTL does**: fp4 LUT decode, fp8 E4M3 decode, product = (w_decoded * a_decoded * s_decoded) >> 8, accumulate
4. The expected accumulator value was packed into a SystemVerilog `localparam`
5. The output file `tb_golden_pkg.sv` was written to `rtl/sim/`

### 3.3 How the RTL Uses Golden Vectors

In `tb_fp4_mac.sv`:

```systemverilog
`include "tb_golden_pkg.sv"

module tb_fp4_mac;
    // ...
    initial begin
        // Run T0: Single multiply
        run_test(tb_golden_pkg::T0_LEN,      // number of terms
                 tb_golden_pkg::T0_W_PACK,    // packed fp4 weights
                 tb_golden_pkg::T0_A_PACK,    // packed fp8 activations
                 tb_golden_pkg::T0_S_PACK,    // packed fp8 scales
                 tb_golden_pkg::T0_EXPECTED,  // expected accumulator
                 "T1  single multiply        ");
        // ... continues for T1..T13
    end
endmodule
```

The `run_test` task:
1. Pulses `accum_clr`
2. Drives each `(weight, activation, scale)` triple with `valid=1`
3. Waits `PIPELINE_DEPTH` cycles for the pipeline to drain
4. Compares `mac_out.result` against `expected` using `!==`
5. Reports PASS or FAIL

### 3.4 Arithmetic Tolerances

**Bit-exact (no tolerance):**
- fp4_mac: All 14 golden tests must match exactly (0 ULP difference)
- fp4 LUT decode: 16 fp4 values must match exactly
- Accumulator: Integer arithmetic, must be exact
- Router top-K: must select exactly the same experts

**+/-1 ULP tolerance:**
- fp8 E4M3 decode: floor division (subnorm: m//2, e=1: (8+m)//2) introduces integer truncation that is identical in Python and RTL -- should be exact. If not, it is a bug.
- SiLU Q12 LUT: Piecewise linear interpolation with integer division; RTL and Python must agree exactly if both use the same knot table

**+/-4 tolerance (RMSNorm):**
- `layer_compute_engine` output: The RMSNorm uses integer square root approximation with 3 Newton iterations. Outputs allow +/-4 in Q12 integer units due to rounding differences in `isqrt` at the final iteration.

**Cosine similarity >= 0.995 (fp4 vs. BF16):**
- When comparing fp4-quantized GEMM output against a full BF16 PyTorch reference, per-token cosine similarity must be >= 0.995
- This is a statistical metric, not bit-exact; run `experiment_1_fp4_precision.py`

### 3.5 Workflow: RTL Change -> Regenerate -> Re-sim

When you modify RTL arithmetic, follow this exact sequence:

```
1. Edit RTL file (e.g., rtl/dsp/fp4_mac.sv)

2. Regenerate golden vectors:
   python scripts/simulation/gen_tb_vectors.py

3. Recompile and run simulation:
   make TOP=tb_fp4_mac run

4. If failures:
   a. Check that Python model and RTL agree on arithmetic
   b. If RTL changed intentionally, update Python model to match
   c. If Python was correct, fix RTL bug

5. Repeat until all tests pass
```

**Never skip step 2.** The golden vectors always reflect what the Python model expects. If the RTL changed, the golden vectors must be regenerated; otherwise you are comparing old expected values against new RTL behavior.

---

## 4. Unit Test Strategy

### 4.1 Rule: Every DSP Module Must Have a Golden-Comparison Testbench

| DSP Module | Testbench | Golden Source | Tests |
|-----------|-----------|---------------|-------|
| `fp4_mac` | `tb_fp4_mac` | `gen_tb_vectors.py` | 14 golden + 1 dynamic |
| `fp4_scale_reader` | `tb_fp4_scale_reader` | Self-checking | BRAM write/read + group addressing |
| `fp4_systolic_tile` | `tb_fp4_systolic_tile` | Self-checking | Weight/activation streaming |
| `fp4_scaled_tile` | `tb_fp4_scaled_tile` | Self-checking | Scale-aware tile ops |
| `fp4_systolic_array` | `tb_fp4_systolic_array` | Self-checking | Array-level data flow |
| `fp4_linear_engine` | `tb_fp4_linear_engine` | Self-checking | Linear layer integration |
| `fp4_gemm_engine` | `tb_fp4_gemm_engine` | Self-checking | Full GEMM engine |
| `fp4_systolic_2d` | `tb_fp4_systolic_2d` | Self-checking | 2D systolic array |
| `fp4_prefill_engine` | `tb_fp4_prefill_engine` | Self-checking | Prefill data path |
| `rms_norm` | `tb_rms_norm` | Self-checking | RMSNorm + isqrt |
| `silu_q12_lut` | `tb_silu_q12_lut` | Self-checking | SiLU LUT knots |
| `cell_mini` / `2d_*` / `2x2` | various | Self-checking | Minimal systolic cell tests |

### 4.2 What to Test: The Three Categories

Every unit testbench should cover:

**A. Normal cases (happy path):**
- Single operation: one weight x one activation, verify product
- Small accumulation: 4-term sum, verify correct accumulation
- Typical values: fp4 +1.0 x fp8 +1.0 (the most common case in inference)
- Streaming back-to-back: 16 consecutive inputs, no bubbles between them

**B. Corner cases (boundary values):**
- fp4 zero (+0 = 0x0, -0 = 0x8): both must produce zero output
- fp4 max magnitude (+3.0 = 0x7, -3.0 = 0xF)
- fp8 subnormal: e=0, m=1..7 (smallest representable values)
- fp8 e=1 boundary: right-shift quantization floor
- fp8 saturation: e >= 10 produces 12-bit clip at +/-2047
- fp8 zero: 0x00 and 0x80 (signed zero)
- Non-unity fp8 scale values: scale = 2.0, 0.5, etc.

**C. Random fuzz (statistical coverage):**
- Large accumulation: 32 terms of max-weight (+3.0 x +1.0) to check for overflow
- Alternating signs: +2.0/-2.0 pairs to verify cancellation to zero
- Mixed sign: positive and negative values intermixed
- If applicable, randomly generate 100+ test cases from Python and compare

### 4.3 Example: Testing fp4_mac

From `tb_fp4_mac.sv`, the 14 golden tests cover:

| Test | Name | What It Tests |
|------|------|---------------|
| T0 | Single multiply | fp4 +1.0 x fp8 +1.0 = +4096 |
| T1 | 4-term accumulation | Mixed fp4 values, same fp8 activation |
| T2 | Positive fp4 sweep | All 8 positive fp4 values x +1.0 |
| T3 | Negative fp4 sweep | All 7 negative fp4 values x +1.0 |
| T4 | Mixed signs | Pos + neg weight x pos + neg activ (cancellation) |
| T5 | Zeros | fp4 +0, fp4 -0, fp8 0 |
| T6 | fp8 subnorm | e=0 with various mantissa values |
| T7 | fp8 e=1 edge | Right-shift quantization boundaries |
| T8 | fp8 saturation | Near and at 12-bit saturation |
| T9 | Large accumulation | 32-term max-weight sum (overflow check) |
| T10 | Streaming | 16 back-to-back, no bubbles |
| T11 | Sign cancellation | Alternating +2.0/-2.0 pairs |
| T12 | fp8 exponent sweep | e=0 through e=7 |
| T13 | Non-unit scale | Scales = 2.0, 0.5 |

Plus one dynamic test: `accum_clr` mid-stream to verify accumulator reset works correctly.

### 4.4 fp4 Precision: Cosine Similarity >= 0.995

Run `experiment_1_fp4_precision.py` to verify:

```bash
python scripts/simulation/experiment_1_fp4_precision.py
```

This test:
1. Creates a realistic gate/up/down weight matrix with 5% outlier channels (simulating real LLM weight distributions)
2. Runs BF16 reference (PyTorch) to get golden outputs
3. Runs fp4 PTQ (direct quantization) and fp4 QAT (per-channel smoothed quantization)
4. Compares per-token cosine similarity

Expected output:
```
  指标                          PTQ (直接)    QAT (平滑)
  ------------------------------------------------------------
  Gate 离群通道比例                    5.0%         0.0%
  输出 余弦相似度 均值              0.987234      0.996512
  输出 余弦相似度 最差              0.951234      0.992345

  结论: [PASS] — fp4 精度达标 (cos >= 0.995)
```

If PTQ cosine < 0.995, the QAT approach must be enabled to suppress outlier channels. This is expected behavior.

### 4.5 SiLU LUT Verification

The SiLU activation uses a piecewise-linear LUT with 9 knots. From `gen_ffn_tb_vectors.py`:

```python
knots = [
    (-32768, -11), (-16384, -295), (-8192, -976), (-4096, -1102),
    (0, 0), (4096, 2994), (8192, 7215), (16384, 16089), (32768, 32768),
]
```

For any Q12 input x, the output y is computed by linear interpolation between knots. The tolerance is +/-1 from the Python reference (integer division rounding).

Test with `tb_silu_q12_lut.sv` using inputs at knot points and at midpoints between knots.

### 4.6 RMSNorm Verification

RMSNorm uses integer square root approximation with Newton's method (3 iterations):

```python
def rtl_isqrt(a):
    if a == 0: return 1
    lead = a.bit_length() - 1
    g = 1 << (lead >> 1)
    for _ in range(3):
        g = (g + a // g) >> 1
    return g
```

The Python model and RTL must use exactly 3 iterations. Output tolerance is +/-4 in Q12 due to integer division rounding at the final iteration.

---

## 5. Integration Test Strategy

### 5.1 Testbench Hierarchy

Tests are organized in four tiers, from smallest to largest scope:

```
TIER 4 (System):     tb_chip_12layer, tb_cluster_384
                         |
TIER 3 (Integration): tb_full_transformer_layer, tb_layer_compute_engine(_golden)
                         |
TIER 2 (Module):      tb_mla_qkv, tb_mla_attention(_v2), tb_router_topk,
                      tb_expert_ffn_engine*, tb_mhc_mixer
                         |
TIER 1 (Unit):        tb_fp4_mac, tb_fp4_scale_reader, tb_fp4_systolic_tile,
                      tb_fp4_scaled_tile, tb_fp4_systolic_array,
                      tb_fp4_linear_engine, tb_fp4_gemm_engine,
                      tb_fp4_systolic_2d, tb_fp4_prefill_engine,
                      tb_cell_mini, tb_2d_mini, tb_2d_1x4, tb_2x2, tb_2d_4x4,
                      tb_rms_norm, tb_silu_q12_lut
```

### 5.2 Layer-Level: tb_full_transformer_layer

This testbench instantiates the complete `full_transformer_layer` module (RMSNorm -> MLA Attention v2 -> RMSNorm -> Router -> FFN -> RMSNorm) and verifies that one token can flow through the entire pipeline end-to-end.

**Test flow:**
1. Preload all weights (scales, gamma, QKV, RoPE LUT, router, FFN gate/up/down) with identity values
2. Feed one token where all 8 hidden dimensions = 4096 (Q12 representation of 1.0)
3. Wait for `valid_out` (attention pipeline takes ~50+ cycles internally)
4. Verify outputs are non-zero (confirms the pipeline is alive)

**What this test catches:**
- Handshake deadlocks: any sub-module FSM hanging
- Connectivity errors: wrong port widths, missing connections
- Weight preload interface bugs: wrong address mapping
- Integration timing: the pipeline as a whole produces output

**Command:**
```bash
make TOP=tb_full_transformer_layer compile run
```

**Expected output:**
```
E2E TOKEN TEST RESULT
 Output:  5793 5793 5793 5793 0 0 0 0
 Router:  expert_0 selected = 1
 Latency: 77 cycles
 PASS (non-zero outputs -- token flowed through)
```

### 5.3 Layer-Level with Golden: tb_layer_compute_engine_golden

This testbench compares the `layer_compute_engine` output against the Python-generated golden values from `gen_layer_golden.py`. It uses the same test case structure that `tb_fp4_mac` uses but at the layer level.

**Test flow:**
1. Preload gate, up, down weights (packed fp4), gamma, router weights
2. Preload input activations from the golden package
3. Wait for `valid_out`
4. Compare outputs against expected:
   - y0..y3 (first 4 hidden dims): +/-4 tolerance (RMSNorm rounding)
   - y4..y7 (last 4 hidden dims): bit-exact (zero in identity tests)
5. Check `router_ok` signal is asserted

**Cases:**
- C0: All-ones input (4096 for all 8 dims) with identity weights
- C1: Mixed-sign input ([4096, 2048, 0, -2048, -4096, -2048, 0, 2048]) with identity weights

**Tolerance logic (from testbench):**
```systemverilog
if (y0 < expected_pack[0*32+:32]-4 || y0 > expected_pack[0*32+:32]+4 ||
    y1 < expected_pack[1*32+:32]-4 || y1 > expected_pack[1*32+:32]+4 ||
    ...
    y4 != expected_pack[4*32+:32] || y5 != expected_pack[5*32+:32] ||
    ...)
    $error("%s mismatch", name);
```

### 5.4 Chip-Level: tb_chip_12layer

Simulates 12 layers time-multiplexed on one FPGA chip (bring-up configuration: HIDDEN=8). The same `full_transformer_layer` instance is reused 12 times with weight reloads between layers.

**What it demonstrates:**
- Weight reload protocol works (router + FFN weights can be written between layers)
- The pipeline can process one token through all 12 layers
- Latency measurement per layer and per chip

**Extended mode:** Also simulates the full 32-chip cluster (384 layers total).

**Command:**
```bash
make TOP=tb_chip_12layer compile run
```

**Expected output:**
```
==================================================================
 FPGA LPU -- Full 32-Chip / 384-Layer Cluster Simulation
==================================================================
 Architecture: 32 chips x 12 layers = 384 total layers
 Data path:  Token -> [Chip 0: L0..L11] -> ... -> [Chip 31: L372..L383]
==================================================================

[PRELOAD] Configuring compute engine weights...
[PRELOAD] Done (840 weights loaded).

--- Chip 0 (Layers 0..11) ---
  L0: [5793,5793,5793,5793,0,0,0,0] (77 cyc)
  L11: [5793,5793,5793,5793,0,0,0,0] (77 cyc)
    ...

Total cluster latency: 27744 cycles (277.44 us = 0.28 ms)
Per-token throughput at 100 MHz: 3604.5 tokens/s
PASS: Token flowed through all 384 layers successfully.
```

### 5.5 Cluster-Level: tb_cluster_384

Same as chip-level but with **per-layer unique weights** generated by a deterministic hash function:

```systemverilog
function integer hash_weight(input integer layer, input integer idx,
                              input integer base, input integer range_val);
    begin
        hash_weight = base + ((layer * 257 + idx * 127) % range_val);
    end
endfunction
```

This produces genuinely different outputs per layer, so the token evolves as it passes through all 384 layers. The testbench captures output snapshots at layers 0, 1, 10, 100, 200, 300, and 383 to show the progression.

**Why this matters:** The homogeneous test (all identity weights) can mask bugs where per-layer weight reloads fail silently. The per-layer unique test exercises the full weight reload path for every single layer.

### 5.6 C2C Ring: tb_c2c_ring

Validates the chip-to-chip dual-ring interconnect:
- Packet format (header, payload, CRC)
- Per-hop routing (master -> slave -> slave -> master)
- Per-hop latency (< 50 ns target per the Go/No-Go criteria)
- Aggregate bandwidth

Pre-wired ILA probe points from `bringup_go_nogo.md`:
- `seq_state` (bring-up FSM state)
- Test result codes (GO/NO-GO/WARN/RUNNING)

### 5.7 KV DMA: tb_kv_dma

Validates the KV cache DMA engine:
- Address generation for KV block allocation
- Block alloc/free protocol
- Multi-chip KV cache distribution (each chip holds its layers' KV cache)

---

## 6. Python Simulation Validation

### 6.1 Module Smoke Tests: run_module_smoke.py

The fastest way to verify the entire Python simulation stack is working:

```bash
python scripts/run_module_smoke.py
```

This runs 10 tests covering both `fpga_arch` and `vllm_serve`:

| # | Test Module | Verifies |
|---|------------|----------|
| 1 | `chip_resources` | SRAM, HBM, DSP bank instantiation, weight placement |
| 2 | `interconnect` | C2C ring latency, PCIe fabric transfer time |
| 3 | `cluster_replication` | Hot expert replication, chip assignment counts |
| 4 | `expert_popularity` | Zipf distribution, top-K mass, replica plan |
| 5 | `pipeline_models` | Pipeline engine, decode TPS, prefill bottleneck |
| 6 | `weight_layout` | Weight layout compiler, HBM utilization |
| 7 | `kv_cache` | KV cache alloc/free, block counting |
| 8 | `scheduler` | Continuous batching, batch type selection |
| 9 | `api_server` | Request generation, arrival process |
| 10 | `serving_short` | 10-second end-to-end serving simulation |

Produces output files:
- `docs/module_smoke_results.json` (machine-readable)
- `docs/module_smoke_report.md` (human-readable markdown table)

**Expected output:**
```
| Module | Status | Key Output |
|---|---:|---|
| chip_resources | PASS | dsp_1G_mac_us=1.111, hbm_920MB_us=1024.0, chip_weight_gb=2.6 |
| interconnect | PASS | c2c_7KB_us=0.0006, pcie_7KB_us=0.3505 |
| ...
| serving_short | PASS | requests=28, finished=26, accept_rate=92.9, output_tps=105.7 |
Passed: 10/10
```

### 6.2 Full Functional Suite: run_all.py

```bash
python scripts/simulation/run_all.py
```

Runs 3 experiments:
1. **Exp 1 (fp4 precision):** fp4 x fp8 GEMM cosine similarity vs. BF16 reference
2. **Exp 2 (HBM bandwidth):** MoE expert loading effective bandwidth with Zipf access pattern
3. **Exp 3 (layer latency):** DSP + HBM end-to-end layer latency, weighted utilization

**Expected output:**
```
  ┌──────────────────────────────────────────────────────────────────┐
  │ 实验           │ 指标              │ 实测值        │ 目标          │ 判定    │
  ├──────────────────────────────────────────────────────────────────┤
  │ Exp 1 fp4 精度 │ 余弦相似度        │ 0.99651      │ ≥ 0.995       │ [PASS] │
  │ Exp 2 HBM 带宽 │ 有效带宽          │ 612 GB/s     │ ≥ 552 GB/s    │ [PASS] │
  │ Exp 3 层延迟   │ 加权层延迟        │ 12.5 μs      │ ≤ 15 μs       │ [PASS] │
  └──────────────────────────────────────────────────────────────────┘

  总体判定: [PASS] 全部 3 项实验通过
  -> 可以进入开发板 Phase 1 验证
```

### 6.3 Serving Simulation: run_serving.py

End-to-end serving simulation integrating FPGA hardware pipeline + vLLM-style continuous batching:

```bash
python scripts/run_serving.py --duration 60 --arrival-rate 5
python scripts/run_serving.py --duration 300 --arrival-rate 10 --verbose
```

Key metrics produced:
- Accept rate (%)
- Output TPS (tokens/second)
- TTFT P50/P95 (ms)
- TPOT P50 (ms -- time per output token)
- Average batch size
- Prefill admission rate (req/s)
- TTFT SLA compliance (%)

### 6.4 Bulk Validation: run_e2e_validation.py

Runs 18 configurations x 3 workloads (54 total runs):

```bash
python scripts/run_e2e_validation.py
```

Configurations are cumulative stacks:
- `baseline`, `+D` (large KV), `+D+C` (+microbatch), `+D+C+A` (+expert replication), `+all+PC2` (+pipeline clone 2), `+all+PC4` (+pipeline clone 4)

Workloads:
- `chat`: arrival=2, prompt=512, output=256
- `agent`: arrival=4, agent mode, 10 turns
- `burst`: arrival=20, prompt=1024, output=1024

Results are written to `docs/e2e_validation_results.json`.

### 6.5 Interpreting Results

| Verdict | Meaning | Action |
|---------|---------|--------|
| **PASS** | Metric meets or exceeds target | No action needed |
| **CHECK** | Metric is borderline (e.g., 0.9949 vs 0.995 target) | Review, flag for further analysis |
| **FAIL** | Metric clearly below target | File bug, investigate root cause |

For serving simulations, pay special attention to:
- **TTFT P95** exceeding 50 ms: bottleneck is prefill capacity
- **Accept rate** below 80%: bottleneck is decode capacity or KV cache
- **SLA compliance** below 95%: system is overloaded at current arrival rate

### 6.6 Per-Stage fp4 MAC Verification: verify_fp4_mac_stages.py

For debugging fp4 MAC issues:

```bash
python scripts/simulation/verify_fp4_mac_stages.py
```

This performs **exhaustive** verification of each pipeline stage:
- Stage 0a: 16 fp4 values, exhaustive decode
- Stage 0b: 256 fp8 values, exhaustive decode (including saturation analysis)
- Stage 1: Decoded operand width and range check
- Stage 2: All 4096 (16 x 256) products, exhaustive
- Stage 3: 100 random GEMMs (M=4, K=128, N=4), accumulation accuracy
- Saturation analysis: Quantifies probability of fp8 saturation for various activation distributions

**Expected output:**
```
================================================================
 SUMMARY
================================================================
  Stage 0a: fp4 decode          [PASS]
  Stage 0b: fp8 decode          [PASS]
  Stage 1:  operand check       [PASS]
  Stage 2:  multiply            [PASS]
  Stage 3:  accumulate          [PASS]
  Sat analysis                   [PASS]
  End-to-end                     [PASS]

  ALL STAGES PASSED
================================================================
```

---

## 7. On-Board Validation

### 7.1 Bring-Up Phases (from bringup_strategy.md)

The on-board validation follows a strict dependency chain over 8 weeks:

```
Week 1-2: Basic Link Bring-Up
  Quartus synthesis -> Golden Top bitstream -> LED/UART alive -> PCIe BAR0 MMIO

Week 2-3: Experiment 1 -- fp4 MAC Precision (Go/No-Go #1)
  Single MAC -> Scale Reader -> 15 golden vectors -> Signal Tap bit-level compare

Week 3-5: Experiment 2 -- HBM Bandwidth (Go/No-Go #2)
  Sequential read -> Zipf random read -> Bank conflict -> Double-buffer overlap

Week 5-7: Experiment 3 -- Single Layer End-to-End (Go/No-Go #3)
  RMSNorm -> Attention -> Router -> ExpertFFN -> RMSNorm
  Each sub-module valid handshake captured via Signal Tap

Week 8: Decision
  All 3 gates PASS -> Order 8x AGM039 production silicon + start Phase 2 PCB
  Any STOP -> Root cause analysis + architecture adjustment
```

### 7.2 HBM Bandwidth Test

The architecture depends on 920 GB/s HBM2e bandwidth. The test procedure:

1. QSYS generates HBM2e controller IP (4 stacks, 32 pseudo-channels)
2. `hbm_bw_test.sv` wires to one pseudo-channel AXI4 interface
3. Writes 256 MB pattern, reads back, measures throughput
4. Repeats for all 32 pseudo-channels in parallel

**Go/No-Go thresholds:**
- GO: Read >= 800 GB/s, Write >= 700 GB/s
- WARN: Read >= 500 GB/s, Write >= 400 GB/s
- NO-GO: Read < 500 GB/s or Write < 400 GB/s

### 7.3 DSP Stress Test

Verifies fp4 MAC array accuracy and timing closure:

1. Instantiate `dsp_stress_test.sv` with LANES=4
2. Mode 0 (sweep): exhaustively test all fp4 x fp8 value pairs
3. Mode 2 (max toggle): verify power delivery under worst-case switching
4. Check timing: `report_timing -setup -npaths 100`

**Go/No-Go thresholds:**
- GO: 0 errors in sweep mode, timing closed at 450 MHz
- WARN: 0 errors but timing at 350-450 MHz
- NO-GO: Any MAC errors OR timing < 350 MHz

### 7.4 PCIe DMA Test

Master chip (Chip 0) only:

1. QSYS generates R-Tile PCIe 5.0 x16 IP
2. Host loads driver, allocates 1 GB DMA buffer
3. Host -> FPGA: DMA write 1 GB, measure throughput
4. FPGA -> Host: DMA read 1 GB, measure throughput

**Go/No-Go thresholds:**
- GO: H2D >= 28 GB/s, D2H >= 28 GB/s
- WARN: H2D >= 16 GB/s, D2H >= 16 GB/s
- NO-GO: H2D < 16 GB/s or D2H < 16 GB/s

### 7.5 C2C Ring Test

All chips participate:

1. QSYS generates F-Tile SerDes IP (4 lanes per direction)
2. Master sends loopback packet, measures round-trip latency
3. Full ring: Master -> Slave 1 -> Slave 2 -> Slave 3 -> Master
4. BER (Bit Error Rate) measured over 1e12 bits

**Go/No-Go thresholds:**
- GO: BER < 1e-15, latency < 100 ns/hop
- WARN: BER < 1e-12, latency < 200 ns/hop
- NO-GO: BER >= 1e-12 or link fails to train

### 7.6 fp4 On-Board Precision Validation

After component tests pass, run inference on the full pipeline and compare token logprobs against a GPU reference:

```
1. Load the same model weights on FPGA and GPU (e.g., H100)
2. Run identical input prompts on both
3. Compare token-by-token logprobs
4. Expected: >95% top-1 token agreement, logprob correlation > 0.99
```

### 7.7 Signal Tap Usage for Debugging

Pre-wire these ILA probe points (from `bringup_go_nogo.md`):

| Probe | Width | Module | Purpose |
|-------|-------|--------|---------|
| `seq_state` | 4 | `top_bringup` | Bring-up FSM state |
| `test_result` | 2 | `top_bringup` | GO/NO-GO/WARN/RUNNING |
| `m_axi_awvalid/awready` | 2 | `hbm_bw_test` | HBM write handshake |
| `m_axi_rvalid/rready` | 2 | `hbm_bw_test` | HBM read handshake |
| `array_result_valid` | 1 | `dsp_stress_test` | DSP test output valid |
| `errors_detected` | 32 | `dsp_stress_test` | DSP error counter |
| `st` (FSM state) | 4 | `full_transformer_layer` | Layer pipeline state |
| `valid_in/valid_out` | 2 | `full_transformer_layer` | Layer handshake |
| `entry_count` | 7 | `mla_kv_cache` | KV cache fill level |

For fp4 MAC debug specifically, wire the per-stage signals from `bringup_strategy.md`:
```
u_mac|s0_weight[3:0]
u_mac|s0_scale[7:0]
u_mac|s0_activ[7:0]
u_mac|s1_w_signed[7:0]       <- fp4 decoded
u_mac|s1_a_scaled[11:0]      <- fp8 activation decoded
u_mac|s1_sc_scaled[11:0]     <- fp8 scale decoded
u_mac|s2_product[31:0]       <- (w x a x s) >>> 8
u_mac|accumulator[31:0]      <- running sum
u_mac|mac_out|valid          <- pipeline drain complete
```

Configure Signal Tap:
- Clock: `clk_dsp` (450 MHz, from PLL: 100 MHz x 9/2)
- Trigger: `mac_valid_in && weight == trigger_weight`
- Depth: 4K samples per node

**Interpretation workflow when a test fails on hardware:**
1. Identify which golden test failed (T0..T13)
2. Set Signal Tap trigger to that test's first weight value
3. Capture waveform for one complete multiply-accumulate
4. Compare each stage's output against the Python reference:
   ```
   s1_w_signed    vs. fp4_decode_signed(python)
   s1_a_scaled    vs. fp8_decode_signed(python)
   s1_sc_scaled   vs. fp8_decode_signed(python)
   s2_product     vs. product_rtl(python)
   accumulator    vs. compute_accum(python)
   ```
5. Identify the first stage where values diverge
6. That stage's RTL is the location of the bug

---

## 8. Go/No-Go Criteria

(Summary from `docs/bringup_go_nogo.md` -- the authoritative document)

### 8.1 Test Sequence and LED Codes

| Order | Test | LED Code | Module |
|-------|------|----------|--------|
| 1 | HBM2e Bandwidth | `0001` | `hbm_bw_test.sv` |
| 2 | DSP Array Accuracy | `0010` | `dsp_stress_test.sv` |
| 3 | PCIe DMA Throughput | `0011` | `pcie_dma_test.sv` |
| 4 | C2C Ring Link | `0100` | `c2c_node.sv` (loopback) |
| 5 | Full Layer Pipeline | `0101` | `full_transformer_layer.sv` |

LED Codes:
- `1111` = ALL TESTS PASSED
- `1010` = TEST FAILED (connect UART for fail_code and details)
- LED[0] always shows ~0.75 Hz heartbeat while FPGA is alive

### 8.2 Go/No-Go Summary Table

| Gate | Week | Test | Pass Criteria | Fail Response |
|------|------|------|--------------|---------------|
| **#1** | 3 | HBM BW | Read >= 800 GB/s, Write >= 700 GB/s | Check refclk, UIB, AXI config |
| **#2** | 5 | DSP Accuracy | 0 errors, timing at 450 MHz | Check DSP config, pipeline, power |
| **#3** | 5 | PCIe DMA | H2D >= 28 GB/s, D2H >= 28 GB/s | Check link training, lane count |
| **#4** | 7 | C2C Ring | BER < 1e-15, < 100 ns/hop | Check SerDes refclk, signal integrity |
| **#5** | 7 | Full Layer | Output matches C model, < 500 cycles | Check sub-module FSMs, KV cache |

### 8.3 fail_code Reference

| Code | Test | Common Causes |
|------|------|--------------|
| 1 | HBM BW | HBM refclk missing, UIB placement wrong, QSYS config error |
| 2 | DSP Acc | MAC pipeline depth wrong, timing violation, power noise |
| 3 | PCIe DMA | Link training failed, wrong lane count, BIOS MMIO config |
| 4 | C2C Link | SerDes PLL unlock, signal integrity, F-Tile placement |
| 5 | Layer Pipe | FSM hang, KV cache overflow, numerical divergence |

### 8.4 Power-Up Sequence

```
1. Apply 12V board power
2. Check power rails (via PMBus/I2C):
   - VCC 0.8V (core) -- within +/-3%
   - VCCP 1.0V (HBM) -- within +/-3%
   - VCCIO 1.2V -- within +/-5%
3. Assert cpu_reset_n (low -> high after clocks stable)
4. LED[0] should start blinking (~0.75 Hz)
5. Press start_button or send 'S' over UART
6. Monitor LED[3:1] for test progress
7. If LED = 1111: ALL PASS. If LED = 1010: read fail_code over UART
```

### 8.5 UART Console (115200 8N1)

Commands (send):
- `S` -- Start test sequence
- `A` -- Abort current test
- `R` -- Report last test result
- `H` -- Help / list commands

Receive: Test start/stop messages, measured metrics, fail_code on error.

---

## 9. Regression and CI Strategy

### 9.1 Three-Tier CI Pipeline

```
TIER 1: Pre-Commit (target < 5 minutes)
  - All Tier 1 unit testbenches (15 bench)
  - run_module_smoke.py (10 tests)
  - Block merge if any test fails

TIER 2: Nightly (target < 60 minutes)
  - All 24 testbenches (full suite)
  - run_all.py (3 experiments)
  - run_e2e_validation.py (54 config-workload combos)
  - run_serving.py --duration 60 (short serving run)
  - Generate HTML report with pass/fail matrix

TIER 3: Weekly (target < 8 hours, when hardware available)
  - All 5 on-board Go/No-Go tests
  - Compare hardware results against simulation baselines
  - Update timing/Power/Resource utilization reports
```

### 9.2 Pre-Commit Checklist

Before committing RTL changes, verify:

1. **Golden vectors regenerated:**
   ```bash
   python scripts/simulation/gen_tb_vectors.py
   python scripts/simulation/gen_ffn_tb_vectors.py
   python scripts/simulation/gen_layer_golden.py
   ```

2. **All unit testbenches pass:**
   ```bash
   # Run the affected testbench
   make TOP=tb_fp4_mac run
   # If modifying fp4 arithmetic, also run:
   python scripts/simulation/verify_fp4_mac_stages.py
   ```

3. **Module smoke tests pass:**
   ```bash
   python scripts/run_module_smoke.py
   ```

4. **No regression in serving simulation:**
   ```bash
   python scripts/run_serving.py --duration 30 --arrival-rate 2
   ```

### 9.3 Nightly Report

The nightly CI should generate a report containing:

1. **Pass/Fail matrix** for all 24 testbenches
2. **fp4 precision trends**: cosine similarity over time (detect slow degradation)
3. **Serving metrics trends**: TTFT P95, TPOT P50, accept rate
4. **Coverage metrics** (if using code coverage tools): statement/expression/branch/toggle coverage per module
5. **Resource estimation drift**: how DSP/BRAM/LUT estimates change with RTL modifications

### 9.4 CI Script Template

Create a CI script at `scripts/ci_nightly.sh`:

```bash
#!/bin/bash
set -e
REPORT_DIR="docs/ci_reports/$(date +%Y%m%d)"
mkdir -p $REPORT_DIR

# 1. RTL simulation suite
echo "=== RTL Testbenches ===" | tee $REPORT_DIR/rtl.log
for tb in tb_fp4_mac tb_fp4_systolic_tile tb_rms_norm tb_silu_q12_lut \
          tb_mla_attention_v2 tb_router_topk tb_expert_ffn_engine_fp4_down_golden \
          tb_layer_compute_engine_golden tb_full_transformer_layer; do
    make SIM=iverilog TOP=$tb run >> $REPORT_DIR/rtl.log 2>&1 || \
        echo "FAIL: $tb" >> $REPORT_DIR/rtl_failures.txt
done

# 2. Python simulation suite
echo "=== Python Simulations ===" | tee $REPORT_DIR/python.log
python scripts/run_module_smoke.py >> $REPORT_DIR/python.log 2>&1
python scripts/simulation/verify_fp4_mac_stages.py >> $REPORT_DIR/python.log 2>&1
python scripts/simulation/run_all.py >> $REPORT_DIR/python.log 2>&1

# 3. Serving simulation
echo "=== Serving Simulation ===" | tee $REPORT_DIR/serving.log
python scripts/run_serving.py --duration 60 --arrival-rate 5 >> $REPORT_DIR/serving.log 2>&1

# 4. Generate report
python scripts/gen_ci_report.py --date $(date +%Y%m%d) --output $REPORT_DIR/report.html
```

### 9.5 Coverage Goals

| Module Type | Statement | Branch | Toggle | FSM State |
|------------|-----------|--------|--------|-----------|
| DSP (fp4_mac, etc.) | 100% | 100% | 100% | 100% |
| Control (FSMs) | >95% | >90% | >90% | 100% |
| Integration (chip/cluster) | >90% | >80% | N/A | 100% |

Coverage is measured with Questa's coverage tool (`vcover`). This requires a Questa Prime license (not Starter Edition). If unavailable, manual code review is the fallback.

---

## 10. Bug Tracking and Reporting

### 10.1 Bug Report Template

When filing a bug found during verification, include ALL of the following:

```markdown
## Bug Report: [Short Descriptive Title]

### Severity
- [ ] BLOCKER: Prevents further verification (STOP-level Go/No-Go)
- [ ] CRITICAL: Produces wrong results but workaround exists
- [ ] MAJOR: Timing/schedule impact but functionally correct
- [ ] MINOR: Cosmetic, documentation, or non-functional

### Environment
- Simulator: Questa / Icarus / Verilator (version)
- RTL revision: [git commit hash]
- Python model revision: [git commit hash]
- OS: [Windows/Linux]

### Reproduction
1. Checkout commit: `[hash]`
2. Run command:
   ```
   [exact command line]
   ```
3. Observe failure

### Expected Behavior
[What the correct output should be]

### Actual Behavior
[What actually happened, with exact output]

### Golden vs. Actual Diff
```
Expected: 0x00001000 (4096)
Got:      0x00000800 (2048)
Diff:     0x00000800
```

### Root Cause Hypothesis
[Your analysis of which module/stage is likely at fault]

### Attachments
- [ ] Testbench or minimal reproduction case
- [ ] Waveform screenshot (GTKWave/Signal Tap)
- [ ] Python reference output
- [ ] RTL diff (if proposing fix)
```

### 10.2 How to Reproduce: The Minimal Testbench Rule

For every bug report, reduce the reproduction to the **smallest possible testbench** that still shows the failure:

1. Start from the failing integration-level testbench
2. Bypass sub-modules one at a time (wire inputs directly to the suspect module)
3. Reduce vector sizes (HIDDEN=2, INTER=2, LANES=1)
4. Use exactly the inputs that trigger the bug (from the golden package)
5. Create a minimal self-contained `tb_bug_NNN.sv` that:
   - Instantiates only the suspect module
   - Has the failing test vectors hardcoded
   - Self-checks against expected output
   - Prints clear PASS/FAIL

The goal: someone else should be able to check out your commit, run one command, and see the bug.

### 10.3 Attaching Waveform Screenshots

For Questa/ModelSim waveform captures:
1. Add the failing signals to the Wave window
2. Zoom to the region where values diverge from expected
3. Use `File -> Export -> Image` to save a PNG
4. Annotate the image: circle the first cycle where RTL and expected diverge
5. Label each signal with its corresponding Python reference value

For Signal Tap captures:
1. Configure trigger on the failing test case's first weight value
2. Capture with 4K sample depth
3. Export to CSV using `File -> Export`
4. Plot the per-stage values against Python expected values

### 10.4 On-Call Rotation Guide

When you are the on-call verification engineer:

**If a nightly CI run fails:**
1. Check `docs/ci_reports/<date>/rtl_failures.txt` for which testbenches failed
2. For each failure:
   a. Reproduce locally with the exact RTL revision
   b. Determine if it is a new failure or a pre-existing known issue
   c. If new: file a bug report, notify the RTL developer
   d. If known: check if it has worsened (more test cases failing vs. previous run)
3. Generate a summary for the team channel:
   ```
   Nightly CI 2026-05-28: 22/24 testbenches PASS
   FAIL: tb_fp4_gemm_engine (new), tb_expert_ffn_engine (known, regression)
   NEW: tb_fp4_gemm_engine T3 mismatch, golden=0x4800 actual=0x4000
   ```

**If an on-board test fails (Go/No-Go STOP):**
1. Read the fail_code over UART
2. Check power rails (PMBus/I2C) -- many hardware failures are power-related
3. Verify clock frequencies with oscilloscope at PLL outputs
4. If power and clocks are OK, enable Signal Tap for the failing module
5. Compare per-stage signals against Python model as described in Section 7.7
6. Escalate to the hardware team if it is a silicon-level issue (DSP behavior, SerDes training failure)

### 10.5 Common Failure Patterns

| Symptom | Likely Cause | Debug Approach |
|---------|-------------|----------------|
| All tests timeout | Clock not toggling, FSM deadlock | Check clock generation, check if rst_n is released |
| All tests zero output | Weight preload not working, accum_clr stuck | Check weight write enables, check that valid pulses properly |
| Single test fails, diff is a power of 2 | Bit-width truncation or sign extension error | Check signed/unsigned conversions, width matching |
| Cosine similarity degrades over time | Quantization error accumulation across layers | Run per-layer cosine check, look for outlier channels |
| Random intermittent failures | Race condition, metastability | Check CDC paths, add synchronizer stages |
| On-board fails but sim passes | Timing violation, DSP rounding difference | Signal Tap per-stage comparison, check timing reports |

### 10.6 Quick Reference: Key Files

**RTL Source (for debugging):**
- `rtl/dsp/fp4_mac.sv` -- fp4 multiply-accumulate engine
- `rtl/include/fp4_types.svh` -- shared type definitions, parameter constants

**Testbenches:**
- `rtl/sim/tb_fp4_mac.sv` -- pattern to follow for all unit testbenches
- `rtl/sim/tb_layer_compute_engine_golden.sv` -- pattern for golden comparison at module level
- `rtl/sim/tb_full_transformer_layer.sv` -- pattern for integration testbenches
- `rtl/sim/tb_chip_12layer.sv` -- pattern for time-multiplexed multi-layer tests
- `rtl/sim/tb_cluster_384.sv` -- pattern for multi-chip cluster simulation

**Golden Model (Python):**
- `scripts/simulation/gen_tb_vectors.py` -- generates golden vectors for fp4_mac
- `scripts/simulation/gen_ffn_tb_vectors.py` -- generates golden vectors for FFN engine
- `scripts/simulation/gen_layer_golden.py` -- generates golden vectors for layer compute engine
- `scripts/simulation/verify_fp4_mac_stages.py` -- per-stage exhaustive verification
- `scripts/simulation/fp4_utils.py` -- fp4 encode/decode/quantize reference
- `scripts/simulation/mla_attention.py` -- MLA attention independent reference
- `scripts/simulation/moe_router.py` -- MoE router independent reference
- `scripts/simulation/transformer_layer.py` -- BF16 + fp4 full layer reference

**CI and Automation:**
- `scripts/run_module_smoke.py` -- 10 fast smoke tests
- `scripts/run_all.py` -- 3-experiment functional suite
- `scripts/run_serving.py` -- end-to-end serving simulation
- `scripts/run_e2e_validation.py` -- bulk config-workload validation

**Documentation:**
- `docs/bringup_strategy.md` -- detailed on-board bring-up plan
- `docs/bringup_go_nogo.md` -- Go/No-Go criteria and test sequences
- `docs/eng/04_verification_guide.md` -- this document

---

## Appendix A: Icarus Verilog Quick Setup

Icarus Verilog is the recommended simulator for CI and quick checks because it requires no license:

```bash
# Installation (Ubuntu/Debian):
sudo apt install iverilog gtkwave

# Installation (Windows via MSYS2):
pacman -S mingw-w64-x86_64-iverilog mingw-w64-x86_64-gtkwave

# Run a testbench:
cd D:\workspace\fpgalpu\rtl\sim
iverilog -g2012 -I../include -o tb_fp4_mac.vvp ../dsp/fp4_mac.sv tb_fp4_mac.sv
vvp tb_fp4_mac.vvp

# With waveform:
iverilog -g2012 -I../include -o tb_fp4_mac.vvp ../dsp/fp4_mac.sv tb_fp4_mac.sv
# (Add $dumpfile/$dumpvars to testbench first)
vvp tb_fp4_mac.vvp
gtkwave tb_fp4_mac.vcd
```

**Known limitations of Icarus for this project:**
- Limited SystemVerilog support: some `localparam` string arrays may not compile
- Slower than Questa for large designs (e.g., `tb_cluster_384` may take 30+ seconds vs. 3 seconds in Questa)
- No code coverage

For day-to-day development, use Questa. For CI, Icarus is acceptable for all Tier 1 unit testbenches.

## Appendix B: Understanding fp4 E2M1 Format

The fp4 format used throughout this project:

```
Format: E2M1 (1 sign bit, 2 exponent bits, 1 mantissa bit)

Bit layout: [s][e1][e0][m]

Decoding:
  Normal (e != 0):  (-1)^s x 2^(e-1) x (1 + m/2)    e in {1,2,3}
  Subnorm (e = 0):  (-1)^s x 2^0    x m/2            e = 0
  Zero:             e=0, m=0, sign doesn't matter

All 16 values:
  +0.00 (0x0)    +0.25 (0x1)    +0.50 (0x2)    +0.75 (0x3)
  +1.00 (0x4)    +1.50 (0x5)    +2.00 (0x6)    +3.00 (0x7)
  -0.00 (0x8)    -0.25 (0x9)    -0.50 (0xA)    -0.75 (0xB)
  -1.00 (0xC)    -1.50 (0xD)    -2.00 (0xE)    -3.00 (0xF)

For fp4_mac: decoded value = FP4_LUT[mag] * (sign ? -1 : 1)
  FP4_LUT = [0, 4, 8, 12, 16, 24, 32, 48]  (x16 representation)
  So: 0.25 -> 4, 1.0 -> 16, 3.0 -> 48
```

## Appendix C: Understanding fp8 E4M3 Format

```
Format: E4M3 (1 sign, 4 exponent, 3 mantissa)

Bit layout: [s][e3][e2][e1][e0][m2][m1][m0]

Decoding:
  Normal (e != 0):  value = (1 + m/8) x 2^(e-7)
  Subnorm (e = 0):  value = m/8 x 2^(-6)
  Zero:             e=0, m=0

Key reference values:
  0x38 = 0_0111_000 = +1.0
  0x30 = 0_0110_000 = +0.5
  0x40 = 0_1000_000 = +2.0
  0xB0 = 1_0110_000 = -0.5
  0xB8 = 1_0111_000 = -1.0

For fp4_mac: decoded value scaled to x256 (12-bit signed)
  Saturation occurs at e >= 10 (value >= 8.0)
  Subnorm quantization: m//2 (not m/8 -- RTL uses integer floor)
```

## Appendix D: Quick Command Reference

```bash
# ── RTL Simulation ──────────────────────────────────────

# Run a single testbench (Questa, fastest)
cd D:\workspace\fpgalpu\rtl\sim
make TOP=tb_fp4_mac run

# Run with waveform GUI
make TOP=tb_fp4_mac gui

# Run all unit testbenches (Icarus, no license needed)
make SIM=iverilog TOP=tb_fp4_mac run
make SIM=iverilog TOP=tb_silu_q12_lut run
make SIM=iverilog TOP=tb_rms_norm run
make SIM=iverilog TOP=tb_layer_compute_engine_golden run

# ── Python Simulation ───────────────────────────────────

# Fast module smoke tests (10 tests, < 30 seconds)
python scripts/run_module_smoke.py

# Full functional suite (3 experiments, < 5 minutes)
python scripts/simulation/run_all.py

# Per-stage fp4 MAC exhaustive check
python scripts/simulation/verify_fp4_mac_stages.py

# fp4 precision experiment
python scripts/simulation/experiment_1_fp4_precision.py

# End-to-end serving (60 seconds)
python scripts/run_serving.py --duration 60 --arrival-rate 5

# Bulk validation (18 configs x 3 workloads)
python scripts/run_e2e_validation.py

# ── Golden Vector Regeneration ──────────────────────────

python scripts/simulation/gen_tb_vectors.py
python scripts/simulation/gen_ffn_tb_vectors.py
python scripts/simulation/gen_layer_golden.py

# ── Waveform ────────────────────────────────────────────

# Icarus: add these to testbench initial block:
#   $dumpfile("dump.vcd");
#   $dumpvars(0, tb_name);
# Then:
iverilog -g2012 -I../include -o tb.vvp dut.sv tb.sv
vvp tb.vvp
gtkwave dump.vcd
```
