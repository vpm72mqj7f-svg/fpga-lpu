# FPGA LPU -- Verification Engineer Onboarding Guide

**Role:** Verification Engineer (1 of 3 on the team)
**Project:** 32-chip Intel Agilex 7 M-Series FPGA inference cluster for DeepSeek V4 Pro
**Your manager:** Tech lead (see `.claude/roles/tech-lead.md`)
**RTL team:** 3 engineers (see `docs/eng/03_rtl_developer_guide.md`)
**Software team:** 2 engineers (see `docs/eng/05_software_guide.md`)

---

## Day 1: Environment Setup (2 hours)

### Step 1: Clone the repo and verify tools

Open a terminal and run these checks one by one. Every one must print a version number -- if one fails, stop and ask the tech lead for help.

```powershell
# Check Python (3.11 or newer required)
python --version
# Expected: Python 3.11.x or Python 3.12.x

# Check Icarus Verilog (open-source simulator)
iverilog -V
# Expected: Icarus Verilog version 11.0 or newer

# Check GTKWave (waveform viewer)
gtkwave --version
# Expected: GTKWave Analyzer v3.3.x or newer

# Check git
git --version
# Expected: git version 2.40.x or newer
```

If any tool is missing:

| Tool | Install on Windows (MSYS2) | Install on Ubuntu/Debian |
|------|--------------------------|--------------------------|
| Python 3.11+ | `winget install Python.Python.3.12` | `sudo apt install python3.12` |
| Icarus Verilog | `pacman -S mingw-w64-x86_64-iverilog` | `sudo apt install iverilog` |
| GTKWave | `pacman -S mingw-w64-x86_64-gtkwave` | `sudo apt install gtkwave` |

Verify your repo is on the latest commit:

```powershell
cd D:\workspace\fpgalpu
git log --oneline -1
# Expected: most recent commit message on branch master
```

---

### Step 2: Activate your AI assistant role

Your Claude AI assistant uses a role file that gives it project-specific context. Copy the verification engineer role into place:

**On PowerShell:**
```powershell
cd D:\workspace\fpgalpu
copy .\.claude\roles\verification-engineer.md .\CLAUDE.md
```

**On Git Bash / WSL:**
```bash
cd /d/workspace/fpgalpu
cp .claude/roles/verification-engineer.md CLAUDE.md
```

This gives your AI assistant full knowledge of the testbench inventory, golden model sources, accuracy tolerances, and debug workflows.

---

### Step 3: Run the Python module smoke tests

These 10 tests verify the entire Python simulation stack is healthy. They cover chip resources, interconnect, cluster replication, expert popularity, pipeline models, weight layout, KV cache, scheduler, API server, and a short serving run.

```powershell
cd D:\workspace\fpgalpu
python scripts\run_module_smoke.py
```

**Expected output (last 3 lines):**
```
| chip_resources | PASS | dsp_1G_mac_us=1.111, hbm_920MB_us=1024.0, chip_weight_gb=2.6 |
| interconnect | PASS | c2c_7KB_us=0.0006, pcie_7KB_us=0.3505 |
...
| serving_short | PASS | requests=28, finished=26, accept_rate=92.9, output_tps=105.7 |
Passed: 10/10
```

**If a test fails:** Run it individually to see the full traceback. For example:
```powershell
python -c "from fpga_arch.chip import FPGAChip; c = FPGAChip(0,0); print('OK')"
```

Two output files are written to `docs/`:
- `docs/module_smoke_results.json` -- machine-readable, for CI
- `docs/module_smoke_report.md` -- human-readable table

---

### Step 4: Run the functional experiments

The functional suite runs 3 experiments that validate key architectural decisions:

```powershell
cd D:\workspace\fpgalpu
python scripts\simulation\run_all.py
```

**Expected output:**
```
  ┌──────────────────────────────────────────────────────────────────┐
  │ 实验           │ 指标              │ 实测值        │ 目标          │ 判定    │
  ├──────────────────────────────────────────────────────────────────┤
  │ Exp 1 fp4 精度 │ 余弦相似度        │ 0.99651      │ >= 0.995      │ [PASS] │
  │ Exp 2 HBM 带宽 │ 有效带宽          │ 612 GB/s     │ >= 552 GB/s   │ [PASS] │
  │ Exp 3 层延迟   │ 加权层延迟        │ 12.5 us      │ <= 15 us      │ [PASS] │
  └──────────────────────────────────────────────────────────────────┘

  总体判定: [PASS] 全部 3 项实验通过
```

If all 3 pass, your Python environment is fully functional. If Exp 1 fails (cosine < 0.995), that is expected for the PTQ baseline -- QAT smoothing should recover it. Run the detailed precision experiment to investigate:

```powershell
python scripts\simulation\experiment_1_fp4_precision.py
```

---

### Step 5: Run your first RTL testbench

Navigate to the simulation directory and run the fp4 multiply-accumulate testbench using Icarus Verilog (no license needed):

```powershell
cd D:\workspace\fpgalpu\rtl\sim
make SIM=iverilog TOP=tb_fp4_mac run
```

**Expected output (abbreviated):**
```
============================================================
 tb_fp4_mac -- Golden Vector Verification
============================================================
 Pipeline: 3-stage, 14 tests from tb_golden_pkg
============================================================

[ OK ] T1  single multiply         (0x00001000)
[ OK ] T2  4-term accumulation     (0x00001400)
[ OK ] T3  pos fp4 sweep (x8)      (0x0000a000)
[ OK ] T4  neg fp4 sweep (x7)      (0xffff6000)
[ OK ] T5  mixed signs             (0x00002000)
[ OK ] T6  zeros                   (0x00000000)
[ OK ] T7  fp8 subnorm             (0x00000040)
[ OK ] T8  fp8 e=1 edge            (0x00001200)
[ OK ] T9  fp8 saturation edge     (0x0001ff80)
[ OK ] T10 32-term max accum       (0x0005a000)
[ OK ] T11 16-stream (no bubble)   (0x00012000)
[ OK ] T12 sign cancellation (x16) (0x00000000)
[ OK ] T13 fp8 exponent sweep      (0x00015040)
[ OK ] T14 non-unity scale         (0x00008000)

--- Dynamic: accum_clr mid-stream ---
  After 2 terms: 0x00002000 (expect 0x00002000)
  After clear + 2 terms: 0x00004000 (expect 0x00004000)
[ OK ] Dynamic accum_clr test

============================================================
 ALL 15 TESTS PASSED
============================================================
```

**What you just witnessed:**
1. The Makefile compiled `fp4_mac.sv` (the DUT) and `tb_fp4_mac.sv` (the testbench) with Icarus
2. The testbench ran 14 tests from `tb_golden_pkg.sv` -- an auto-generated file containing expected outputs
3. Plus 1 dynamic test verifying the accumulator reset works mid-stream
4. All 15 passed with bit-exact matches

**If the command fails:** Check that Icarus is on your PATH. Run `iverilog -V` to verify. If you see `make: *** No rule to make target`, confirm you are in the `rtl/sim` directory.

---

### Day 1 Wrap-Up Checklist

- [ ] Python 3.11+ prints version
- [ ] Icarus Verilog prints version
- [ ] GTKWave prints version
- [ ] CLAUDE.md contains verification engineer role
- [ ] `run_module_smoke.py` -- 10/10 PASS
- [ ] `run_all.py` -- 3/3 PASS
- [ ] `make SIM=iverilog TOP=tb_fp4_mac run` -- 15/15 PASS

---

## Day 2: Understand the Methodology (4 hours)

### Step 1: Read the verification guide

Read the first three sections of the master verification guide. These establish the philosophy and infrastructure you will use every day.

```powershell
# Open in VS Code
code D:\workspace\fpgalpu\docs\eng\04_verification_guide.md
```

**Read (30 minutes):**
- Section 1: Verification Philosophy -- the three parallel tracks, why we never run full-system cycle-accurate sim, bring-up vs. production parameters
- Section 2: Testbench Infrastructure -- testbench inventory, `make SIM=iverilog TOP=<name> run` syntax, GTKWave usage, golden package files
- Section 3: Golden Model Methodology -- the Python models, how golden vectors are generated, how RTL uses them, arithmetic tolerances

**Key insight you should internalize:** The Python golden model and the RTL share zero code. The Python model is an independent re-implementation of the same arithmetic. If they agree, both are correct. If they disagree, one is wrong -- and it is your job to figure out which.

---

### Step 2: Study the golden model flow

Open the golden vector generator in your editor and trace through it:

```powershell
code D:\workspace\fpgalpu\scripts\simulation\gen_tb_vectors.py
```

**Walk through the code and answer these questions (15 minutes):**
1. What is `FP4_LUT` and why is it a list of 8 integers?
2. How does `fp4_decode_signed()` handle the sign bit?
3. How does `fp8_decode_signed()` handle subnorm values (e=0)?
4. What does `compute_accum()` do -- how does it match the RTL pipeline?
5. What does `write_sv_package()` output, and where does the file go?

Then open the generated golden package it produces:

```powershell
code D:\workspace\fpgalpu\rtl\sim\tb_golden_pkg.sv
```

**Verify you understand:**
- Each `T<N>_LEN` controls how many weight/activation pairs are driven
- Each `T<N>_W_PACK` packs fp4 weights into a 256-bit vector (64 weights max per test)
- Each `T<N>_A_PACK` packs fp8 activations into a 512-bit vector (64 activations max)
- Each `T<N>_EXPECTED` is the 32-bit accumulator expected value
- The `NUM_TESTS` parameter tells the testbench how many test cases exist

---

### Step 3: Compare RTL vs Python side by side

Open both files and trace through one test case (T0: single multiply +1.0 x +1.0 = +4096):

```powershell
code D:\workspace\fpgalpu\rtl\dsp\fp4_mac.sv
code D:\workspace\fpgalpu\scripts\simulation\verify_fp4_mac_stages.py
```

**In `fp4_mac.sv`, identify the 4 pipeline stages:**
1. **Stage 0**: Input register -- captures `weight[3:0]`, `activ[7:0]`, `scale[11:0]` from `mac_in`
2. **Stage 1**: Decode -- fp4 LUT lookup (`FP4_LUT[mag]`) and fp8 E4M3 decode (sign, exponent, mantissa extraction)
3. **Stage 2**: Multiply -- signed 8b x signed 12b product, right-shift by 8, registered
4. **Stage 3**: Accumulate -- add product to accumulator, registered output

**In `verify_fp4_mac_stages.py`, find the corresponding Python code for each stage.**

The Python `compute_accum()` matches the RTL cycle-by-cycle: fp4 LUT decode, fp8 decode, product = (w_decoded * a_decoded * s_decoded) >> 8, accumulate.

---

### Step 4: Learn the tolerance table

This is the single most important reference for your daily work. Print it out or pin it:

| Operation | Tolerance | Why |
|-----------|-----------|-----|
| fp4 x fp8 MAC product | Bit-exact | Integer arithmetic, same LUT in both |
| Q12 accumulation (<=256 terms) | Bit-exact | Integer arithmetic, 32-bit accumulator |
| SiLU Q12 LUT | +/- 1 LSB | Piecewise linear interpolation with integer division |
| RMSNorm isqrt | +/- 4 LSB | 3 Newton iterations, rounding at final step |
| fp4 GEMM vs BF16 cosine | >= 0.995 | Statistical metric across many tokens |
| MLA attention vs PyTorch | Cosine >= 0.99 | Attention softmax quantization |
| Full layer token logprobs | Top-1 match >= 99.9% | Cross-layer error accumulation |

**Golden rule:** If a DSP unit test fails at bit-exact, it is a bug -- file it and block the PR. If a layer-level test fails by 1-2 LSB, it may be acceptable rounding -- check which operation introduced the difference before deciding.

---

### Step 5: Run the fp4 precision experiment and understand cosine similarity

```powershell
cd D:\workspace\fpgalpu
python scripts\simulation\experiment_1_fp4_precision.py
```

**Expected output:**
```
  指标                          PTQ (直接)    QAT (平滑)
  ------------------------------------------------------------
  Gate 离群通道比例                    5.0%         0.0%
  输出 余弦相似度 均值              0.987234      0.996512
  输出 余弦相似度 最差              0.951234      0.992345

  结论: [PASS] -- fp4 精度达标 (cos >= 0.995)
```

**What cosine similarity means for you:**
- cosine = 1.000 means fp4 and BF16 outputs point in exactly the same direction (perfect)
- cosine >= 0.995 means fp4 and BF16 token embeddings are nearly identical -- downstream layers won't diverge
- cosine = 0.95 means the model might start producing different tokens (degradation)

When you see cosine similarity in a test result, think: "Does the fp4 output vector point in the same semantic direction as the BF16 reference?" Cosine >= 0.995 means yes.

---

### Day 2 Wrap-Up Checklist

- [ ] Read verification guide sections 1-3
- [ ] Traced through `gen_tb_vectors.py` -- understand every function
- [ ] Opened `tb_golden_pkg.sv` -- understand the packed vector format
- [ ] Compared `fp4_mac.sv` and `verify_fp4_mac_stages.py` -- can map every pipeline stage
- [ ] Memorized the tolerance table (or bookmarked it)
- [ ] Ran `experiment_1_fp4_precision.py` and understand cosine >= 0.995

---

## Day 3: Your First Testbench Assignment

### Step 1: Read the testbench inventory

Open the verification guide and study the testbench hierarchy:

```powershell
code D:\workspace\fpgalpu\docs\eng\04_verification_guide.md
```

Read Section 5.1 (Testbench Hierarchy). Understand the four tiers:
- **Tier 1 (Unit):** Individual DSP primitives -- `tb_fp4_mac`, `tb_fp4_systolic_tile`, `tb_rms_norm`, etc. (~15 benches)
- **Tier 2 (Module):** Multi-DSP blocks -- `tb_mla_attention_v2`, `tb_router_topk`, `tb_expert_ffn_engine`, etc. (~8 benches)
- **Tier 3 (Integration):** Full pipeline sub-systems -- `tb_layer_compute_engine_golden`, `tb_full_transformer_layer` (~3 benches)
- **Tier 4 (System):** Multi-layer, multi-chip -- `tb_chip_12layer`, `tb_cluster_384` (~2 benches)

---

### Step 2: Pick one unit testbench and read it line by line

Start with the simplest one -- the SiLU LUT testbench:

```powershell
code D:\workspace\fpgalpu\rtl\sim\tb_silu_q12_lut.sv
```

**Read it from top to bottom. As you read, answer:**
1. What module is instantiated as the DUT (device under test)?
2. How is the clock generated?
3. What is the `initial` block doing -- how many test cases?
4. How does it check results -- what tolerance does it allow?
5. What is the watchdog timer protecting against?

Then read the MAC testbench you ran on Day 1:

```powershell
code D:\workspace\fpgalpu\rtl\sim\tb_fp4_mac.sv
```

**Compare with the SiLU testbench. Notice the differences:**
- `tb_fp4_mac` uses a golden package (` `include "tb_golden_pkg.sv" `)
- It has helper tasks: `drive()`, `drive_stream()`, `check_result()`, `run_test()`
- It tests the `accum_clr` signal dynamically
- It uses `!==` (case-inequality) for bit-exact comparison
- The SiLU testbench uses +/- 1 tolerance, the MAC testbench requires bit-exact

---

### Step 3: Understand what makes a good test case

Look at the 14 golden test cases in `tb_fp4_mac.sv`. They fall into three categories:

**A. Normal cases (T1-T3):**
- Single operation, small accumulation, sweeping through typical values
- Question: "Does the module work for the values we expect to see 99% of the time?"

**B. Corner cases (T4-T9, T12-T14):**
- Mixed signs, zeros (both +0 and -0), subnorm, exponent edge, saturation, non-unity scale
- Question: "Does the module handle every edge case the fp4/fp8 formats can produce?"

**C. Random fuzz (T10-T11):**
- Large accumulation (overflow check), streaming back-to-back (bubble-free)
- Question: "Does the module handle stress conditions without breaking?"

Every testbench you write should have at least one test from each category.

---

### Step 4: Run a testbench with waveform and explore the signals

Add VCD dumping to the MAC testbench by editing it:

```powershell
code D:\workspace\fpgalpu\rtl\sim\tb_fp4_mac.sv
```

Find the `initial begin` block (around line 115). Add these two lines right after `$finish;`:

```systemverilog
initial begin
    $dumpfile("tb_fp4_mac.vcd");
    $dumpvars(0, tb_fp4_mac);
end
```

Now rerun and open the waveform:

```powershell
cd D:\workspace\fpgalpu\rtl\sim
make SIM=iverilog TOP=tb_fp4_mac run
gtkwave tb_fp4_mac.vcd
```

**In GTKWave, add these signals to the wave window and find them:**

| Signal | What to Look For |
|--------|-----------------|
| `clk` | 100 MHz square wave (10 ns period) |
| `mac_in.valid` | Pulses high for 1 cycle per weight/activation pair, stays high for streaming tests |
| `mac_in.weight[3:0]` | fp4 weight values changing each valid cycle |
| `mac_in.activ[7:0]` | fp8 activation values |
| `mac_in.scale[11:0]` | Pre-decoded fp8 scale value |
| `accum_clr` | Pulses at start of each test case, then stays low |
| `mac_out.result[31:0]` | Updates after pipeline drain (PIPELINE_DEPTH=6 cycles after last input) |
| `pass_count[31:0]` | Increments after each PASS |
| `fail_count[31:0]` | Increments after each FAIL (should stay at 0) |

**Challenge:** Zoom in on T1 (single multiply). Find the waveform section where:
1. `accum_clr` pulses high
2. `mac_in.valid` pulses high with weight=0x4, activ=0x38 (both = +1.0)
3. After 6 cycles (PIPELINE_DEPTH), `mac_out.result` shows `0x00001000` (4096 in decimal)

Expected measurement: 4096 = (fp4 +1.0 decoded to 16) x (fp8 +1.0 decoded to 256) = 4096. This matches because the RTL uses x16 for fp4 and x256 for fp8.

---

### Step 5: Identify pipeline stages in the waveform

The `fp4_mac` has 4 pipeline stages. In the waveform, you should be able to see the delay between input and output:

1. `mac_in.valid` rising edge (input enters stage 0)
2. 1 cycle later: internal decode (stage 1)
3. 2 cycles later: multiply (stage 2)
4. 3 cycles later: accumulate (stage 3)
5. `mac_out.result` updates (output visible)

The testbench uses `PIPELINE_DEPTH = 6` (not 4) to add margin. The RTL reports "3-stage" because the accumulate stage's output register is not counted as a separate pipeline stage -- the MAC has 3 pipeline registers (decode, multiply, accumulate = 3 stages of registers, 4 stages of logic).

The internal pipeline signals (`s0_weight`, `s0_scale`, `s0_activ`, etc.) are named with the `s<N>_` prefix. In the RTL source, find the `always_ff @(posedge clk)` blocks that advance each stage.

---

### Day 3 Wrap-Up Checklist

- [ ] Read testbench hierarchy -- can name all four tiers and give an example of each
- [ ] Read `tb_silu_q12_lut.sv` and `tb_fp4_mac.sv` line by line
- [ ] Understand the three test case categories (normal, corner, fuzz)
- [ ] Generated and opened a VCD waveform in GTKWave
- [ ] Identified `mac_in` -> pipeline -> `mac_out` in the waveform with cycle-level annotation

---

## Day 4: First Real Task

Your first week ends with a hands-on assignment: modify an existing testbench and add a new one. This is exactly the kind of work you will do daily.

---

### Task A: Add a test case to an existing testbench

The SiLU LUT testbench only tests at the 9 knot points. It does not test values between knots (the piecewise linear interpolation path). Add one.

```powershell
code D:\workspace\fpgalpu\rtl\sim\tb_silu_q12_lut.sv
```

The SiLU knot table (from the verification guide, Section 4.5):

```
(-32768, -11), (-16384, -295), (-8192, -976), (-4096, -1102),
(0, 0), (4096, 2994), (8192, 7215), (16384, 16089), (32768, 32768)
```

**Your task:** Add a test case for `x = 6144` (midpoint between knots at 4096 and 8192).

**Step 1:** Compute the expected value using the interpolation formula yourself:

```
x = 6144
knot_lo = 4096, knot_hi = 8192
y_lo = 2994, y_hi = 7215
fraction = (x - knot_lo) / (knot_hi - knot_lo) = (6144 - 4096) / (8192 - 4096) = 2048/4096 = 0.5
y = y_lo + fraction * (y_hi - y_lo) = 2994 + 0.5 * (7215 - 2994) = 2994 + 0.5 * 4221 = 2994 + 2110.5
y = 5104.5, rounds to 5104 or 5105 (integer division -- check exact RTL rounding)
```

**Step 2:** Add the test case. Find the section where inputs are driven. Add:

```systemverilog
// Midpoint test: x = 6144 (between knot at 4096 and 8192)
inp = 6144;
@(posedge clk);
@(posedge clk);  // wait for LUT output
if (result >= 5104-1 && result <= 5105+1)  // +/-1 tolerance
    $display("[ OK ] Midpoint x=6144, got=%d", result);
else
    $error("[FAIL] Midpoint x=6144, got=%d (expected 5104-5105)", result);
```

**Step 3:** Run it:

```powershell
cd D:\workspace\fpgalpu\rtl\sim
make SIM=iverilog TOP=tb_silu_q12_lut run
```

**Step 4:** If PASS -- good. If the value is within +/-1 of 5104 or 5105, it passes. If not, debug: check whether the testbench uses signed or unsigned comparison.

---

### Task B: Find a module without a testbench and add one

Some modules in `rtl/dsp/` may not have an individual testbench yet. Check:

```powershell
# List DSP modules
ls D:\workspace\fpgalpu\rtl\dsp\

# List existing testbenches
ls D:\workspace\fpgalpu\rtl\sim\tb_*.sv
```

**Modules known to exist in `rtl/dsp/`:**
- `fp4_mac.sv` -- has `tb_fp4_mac.sv`
- `fp4_scale_reader.sv` -- has `tb_fp4_scale_reader.sv`
- `fp4_systolic_cell.sv` -- tested by `tb_cell_mini.sv`
- `fp4_systolic_2d.sv` -- has `tb_fp4_systolic_2d.sv`
- `fp4_gemm_engine.sv` -- has `tb_fp4_gemm_engine.sv`
- `fp4_prefill_engine.sv` -- has `tb_fp4_prefill_engine.sv`

If you find a DSP module without a testbench, or if none are missing, pick `fp4_scale_reader.sv` and add a new corner case to its existing testbench instead -- for example, testing the group addressing logic with boundary values.

The template for a new testbench is in the verification guide, Section 2.6. Follow it exactly. Key points:
1. ` `include "fp4_types.svh" ` for type definitions
2. Instantiate the DUT with bring-up parameters (NOT production parameters)
3. Use a clock period of 10 ns (100 MHz)
4. Include a watchdog timer to prevent infinite simulation hangs
5. Use `$display` for PASS, `$error` for FAIL

---

### Task C: Open a pull request

Once your new test case passes:

```powershell
cd D:\workspace\fpgalpu

# Create a feature branch
git checkout -b verif/add-silu-midpoint-test

# Stage your changes
git add rtl\sim\tb_silu_q12_lut.sv

# Commit
git commit -m "verif: add SiLU LUT midpoint interpolation test case"

# Push
git push -u origin verif/add-silu-midpoint-test

# Create PR (if gh CLI is set up)
gh pr create --title "verif: SiLU LUT midpoint test" --body "Add test case for x=6144 to verify piecewise linear interpolation between knots at 4096 and 8192."
```

**PR description should include:**
- What test case you added and why
- The expected value and how you computed it
- The actual test output (copy-paste the PASS line)
- A note that all existing tests still pass (no regression)

---

### Day 4 Wrap-Up Checklist

- [ ] Added a new test case to an existing testbench
- [ ] Test PASSES locally
- [ ] All existing tests still pass (no regressions)
- [ ] Verified the test covers a real corner case (not a duplicate)
- [ ] Branch created, committed, PR opened

---

## The Golden Model Cycle (your core workflow)

This is the loop you will run every day. Memorize it.

```
RTL engineer changes a module
          |
          v
You regenerate golden vectors:
    python scripts/simulation/gen_tb_vectors.py
    python scripts/simulation/gen_ffn_tb_vectors.py
    python scripts/simulation/gen_layer_golden.py
          |
          v
You re-run the testbench:
    cd rtl/sim && make SIM=iverilog TOP=tb_<module> run
          |
          v
    PASS -----> Approve the PR
    FAIL -----> File a bug with:
                  golden  = [expected hex value]
                  RTL     = [actual hex value]
                  screenshot of waveform at first-divergent stage
                  which stage of the pipeline diverged
```

**Which golden file to regenerate for which module:**

| RTL Module Changed | Golden File to Regenerate | Command |
|-------------------|--------------------------|---------|
| `fp4_mac.sv` | `tb_golden_pkg.sv` | `python scripts/simulation/gen_tb_vectors.py` |
| `expert_ffn_engine_fp4_down.sv` | `tb_ffn_golden_pkg.sv` | `python scripts/simulation/gen_ffn_tb_vectors.py` |
| `layer_compute_engine.sv` | `tb_layer_golden_pkg.sv` | `python scripts/simulation/gen_layer_golden.py` |

**Which testbench to run for which golden file:**

| Golden File | Testbench | Command |
|------------|-----------|---------|
| `tb_golden_pkg.sv` | `tb_fp4_mac` | `make SIM=iverilog TOP=tb_fp4_mac run` |
| `tb_ffn_golden_pkg.sv` | `tb_expert_ffn_engine_fp4_down_golden` | `make SIM=iverilog TOP=tb_expert_ffn_engine_fp4_down_golden run` |
| `tb_layer_golden_pkg.sv` | `tb_layer_compute_engine_golden` | `make SIM=iverilog TOP=tb_layer_compute_engine_golden run` |

---

## Quick Reference

### Run All Tests

```powershell
# Python module smoke (10 tests, < 30 sec)
cd D:\workspace\fpgalpu
python scripts\run_module_smoke.py

# Full validation suite (all Python suites, ~10 min)
python scripts\run_all_validations.py

# Functional experiments (3 experiments, < 5 min)
python scripts\simulation\run_all.py

# End-to-end serving (60 seconds)
python scripts\run_serving.py --duration 60 --arrival-rate 5

# Bulk validation (54 config x workload combos, ~20 min)
python scripts\run_e2e_validation.py
```

### fp4 Precision & Debug

```powershell
# fp4 precision experiment (cosine similarity vs BF16)
python scripts\simulation\experiment_1_fp4_precision.py

# fp4 strategy sweep (precision/performance tradeoffs)
python scripts\simulation\experiment_1b_fp4_strategies.py

# Per-stage fp4 MAC exhaustive verification
python scripts\simulation\verify_fp4_mac_stages.py
```

### Golden Vector Generation

```powershell
cd D:\workspace\fpgalpu

# fp4_mac golden vectors -> rtl/sim/tb_golden_pkg.sv
python scripts\simulation\gen_tb_vectors.py

# FFN engine golden vectors -> rtl/sim/tb_ffn_golden_pkg.sv
python scripts\simulation\gen_ffn_tb_vectors.py

# Layer compute engine golden vectors -> rtl/sim/tb_layer_golden_pkg.sv
python scripts\simulation\gen_layer_golden.py
```

### RTL Simulation

```powershell
cd D:\workspace\fpgalpu\rtl\sim

# Compile and run a single testbench (Icarus, no license needed)
make SIM=iverilog TOP=tb_fp4_mac run
make SIM=iverilog TOP=tb_silu_q12_lut run
make SIM=iverilog TOP=tb_rms_norm run
make SIM=iverilog TOP=tb_fp4_systolic_tile run
make SIM=iverilog TOP=tb_fp4_scaled_tile run
make SIM=iverilog TOP=tb_fp4_systolic_array run
make SIM=iverilog TOP=tb_fp4_linear_engine run
make SIM=iverilog TOP=tb_mla_attention_v2 run
make SIM=iverilog TOP=tb_router_topk run
make SIM=iverilog TOP=tb_expert_ffn_engine_fp4_down_golden run
make SIM=iverilog TOP=tb_layer_compute_engine_golden run
make SIM=iverilog TOP=tb_full_transformer_layer run
make SIM=iverilog TOP=tb_chip_12layer run
make SIM=iverilog TOP=tb_cluster_384 run

# Compile and run a single testbench (Questa, if license available)
make TOP=tb_fp4_mac run
make TOP=tb_full_transformer_layer run

# Run all unit testbenches (batch script)
make SIM=iverilog TOP=tb_fp4_mac compile
make SIM=iverilog TOP=tb_silu_q12_lut compile
make SIM=iverilog TOP=tb_rms_norm compile
make SIM=iverilog TOP=tb_router_topk compile

# Clean build artifacts
make SIM=iverilog clean
```

### Waveform Viewing

```powershell
cd D:\workspace\fpgalpu\rtl\sim

# Step 1: Add these lines to your testbench initial block:
#   initial begin
#       $dumpfile("tb_fp4_mac.vcd");
#       $dumpvars(0, tb_fp4_mac);
#   end

# Step 2: Compile and run (generates .vcd file)
make SIM=iverilog TOP=tb_fp4_mac run

# Step 3: Open waveform
gtkwave tb_fp4_mac.vcd
```

### Accuracy Tolerances at a Glance

| Operation | Tolerance |
|-----------|-----------|
| fp4 x fp8 MAC product | Bit-exact |
| Q12 accumulation (<=256 terms) | Bit-exact |
| SiLU Q12 LUT | +/- 1 LSB |
| RMSNorm isqrt approximation | +/- 4 LSB |
| fp4 GEMM vs BF16 cosine | >= 0.995 |
| MLA attention vs PyTorch reference | Cosine >= 0.99 |
| Full layer output token logprobs | Top-1 match >= 99.9% |

---

## Common First-Week Mistakes

### 1. Running production config in Icarus -- it will never finish

The Makefile testbenches use bring-up parameters (HIDDEN=8, LANES=4, etc.). Production parameters (HIDDEN=7168, LANES=128) are used only for Quartus bitstream synthesis. If you accidentally change the testbench parameters to production values, the simulation will run for hours (or days) and produce no useful result.

**Correct command:**
```powershell
make SIM=iverilog TOP=tb_fp4_mac run   # uses bring-up params, finishes in seconds
```

**Wrong:**
```
# Do NOT do this -- modify parameters in the testbench to production values
```

---

### 2. Comparing fp4 to BF16 and expecting bit-exact match

fp4 has only 3 bits of precision. When a GEMM output is compared to a BF16 reference, the outputs will differ in their least significant bits. This is expected. The metric is cosine similarity >= 0.995, not bit-exact match.

**When you see a cosine of 0.9949 vs 0.995:** This is a judgment call. Is the degradation from quantization strategy or an RTL bug? Run `experiment_1b_fp4_strategies.py` to see which strategy recovers the target. If all strategies are below 0.995, there may be a systematic issue.

---

### 3. Forgetting to regenerate golden vectors after RTL arithmetic changes

When the RTL team changes `fp4_mac.sv` arithmetic and you re-run the testbench, it will fail even if the RTL is correct -- because the golden package still contains the OLD expected values. The golden vectors always reflect what the Python model expects.

**Correct workflow:**
```powershell
# 1. RTL changed -> regenerate
python scripts\simulation\gen_tb_vectors.py

# 2. Re-run testbench
make SIM=iverilog TOP=tb_fp4_mac run

# 3. If still fails -> RTL and Python disagree -> one is wrong
```

**Wrong:**
```
# Run testbench first, see failure, THEN regenerate golden vectors
# This loses the information about which direction the mismatch was
```

---

### 4. Not checking which golden package file is being included

There are three golden packages, and they are not interchangeable:
- `tb_golden_pkg.sv` -- for `tb_fp4_mac` (MAC unit tests)
- `tb_ffn_golden_pkg.sv` -- for `tb_expert_ffn_engine_fp4_down_golden` (FFN engine tests)
- `tb_layer_golden_pkg.sv` -- for `tb_layer_compute_engine_golden` (layer compute engine tests)

Each testbench ` `include`s exactly one of them. If you regenerate `tb_golden_pkg.sv` but the testbench uses `tb_ffn_golden_pkg.sv`, the testbench will still use the old values. Check the ` `include` line at the top of the testbench to confirm.

---

### 5. Ignoring +/- 1 ULP differences -- they are acceptable for fp4, but NOT for fp4_mac

The tolerance table is strict for a reason:
- **fp4_mac**: Bit-exact. Any difference, even 1 ULP, is a BUG. File it and block the PR.
- **SiLU LUT**: +/- 1 OK. Integer division rounding differs between Python and RTL in predictable ways.
- **RMSNorm**: +/- 4 OK. The isqrt approximation uses 3 Newton iterations, and the final rounding depends on the intermediate integer division path.

When you see a 1-ULP difference on `fp4_mac`, do NOT dismiss it. It means the RTL and Python disagree on the fp4 or fp8 decode, the multiply, or the accumulate -- all of which are integer operations and must be identical. Trace the difference through the pipeline stages to find the root cause.

---

## Next Steps After Day 4

After completing the 4-day onboarding:

1. **Set up your CI watch.** Ask the tech lead for access to the nightly CI dashboard. You should know every morning which testbenches passed and which failed overnight.

2. **Run the full validation suite** to see the complete picture:
   ```powershell
   python scripts\run_all_validations.py
   ```
   This runs module smoke tests, functional experiments, architecture integration, and a short serving simulation. Outputs go to `docs/sim_*.log`.

3. **Pick up your first real bug** from the CI failure list or the GitHub issue tracker. The tech lead will assign you bugs that match your skill level.

4. **Pair with an RTL engineer** for one session. Watch them make a change to `fp4_mac.sv`, then run the golden model cycle yourself: regenerate vectors, re-run testbench, interpret the result.

5. **Read the bring-up strategy** to understand what you will eventually validate on hardware:
   ```powershell
   code D:\workspace\fpgalpu\docs\bringup_strategy.md
   code D:\workspace\fpgalpu\docs\bringup_go_nogo.md
   ```

---

## Key Files You Need to Know

| File | Purpose | When to Use |
|------|---------|------------|
| `docs/eng/04_verification_guide.md` | Master verification guide | Reference for methodology, tolerances, bring-up |
| `rtl/sim/Makefile` | Simulation build system | Every time you run a testbench |
| `rtl/sim/tb_fp4_mac.sv` | Template for all unit testbenches | When writing a new testbench |
| `rtl/sim/tb_golden_pkg.sv` | Auto-generated golden vectors | Regenerated after RTL changes |
| `rtl/include/fp4_types.svh` | Shared type definitions | Understanding port types |
| `scripts/simulation/gen_tb_vectors.py` | Golden vector generator | Regenerated after MAC arithmetic changes |
| `scripts/simulation/fp4_utils.py` | fp4 encode/decode reference | Understanding fp4 format |
| `scripts/simulation/verify_fp4_mac_stages.py` | Per-stage exhaustive verification | Debugging a specific pipeline stage |
| `scripts/run_module_smoke.py` | 10 fast smoke tests | Quick health check |
| `scripts/simulation/run_all.py` | 3 functional experiments | Validating architecture decisions |
| `.claude/roles/verification-engineer.md` | Your AI assistant role | Copied to CLAUDE.md on Day 1 |
