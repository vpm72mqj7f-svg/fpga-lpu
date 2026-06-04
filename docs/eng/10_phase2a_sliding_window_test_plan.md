# Phase 2A Sliding Window Attention Test Plan

> **Status:** Draft — awaiting review by VERIF-LEAD and RTL-DEV1
> **Module under test:** `mla_attention_v2.sv` (modified for sliding window)
> **New testbench:** `tb_mla_attention_v2_window.sv`
> **Date:** 2026-06-04

---

## 1. Overview

The current `mla_attention_v2` attends to all cached KV entries (0 to `cache_fill_count - 1`). The sliding window modification restricts attention to the most recent `WINDOW_SIZE` positions only. The change must:

- Produce **bit-identical** output to the current full-attention module when `cache_fill_count <= WINDOW_SIZE`.
- For `cache_fill_count > WINDOW_SIZE`, scores for positions outside the window are zero (positions are skipped entirely).
- Handle ring-buffer wrap-around correctly (KV cache is a circular buffer with `wr_ptr` wrapping at `NUM_SLOTS`).

The test plan covers 8 scenarios organized into three categories: backward compatibility, sliding semantics, and robustness.

---

## 2. Design Under Test Summary

### 2.1 Current Behavior (from `mla_attention_v2.sv`)

```
S_SCORE_INIT:  score_idx=0, score_max=-inf
    if cache_was_empty → S_OUTPUT (self-attn only)
    else → S_CACHE_RD

Loop (S_CACHE_RD → S_ATTN_SCORE):
    dot = Q_r · K_cache[score_idx]  (K_LATENT dims, Q12)
    scores[score_idx] = dot
    if score_idx == cache_fill_count - 1 → S_EXP_LOOP
    else score_idx++

S_EXP_LOOP:
    exp_sum += exp_lut(scores[exp_idx] - score_max)
    if exp_idx == cache_fill_count - 1 → S_INV
    else exp_idx++

S_INV: inv_scale = 4096*4096 / exp_sum

Loop (S_CACHE_RD2 → S_ACCUM_LOOP):
    weight = exp_lut(scores[accum_idx] - score_max) * inv_scale >>> 12
    V_acc += weight * V_cache[accum_idx] >>> 12
    if accum_idx == cache_fill_count - 1 → S_OUTPUT
    else accum_idx++
```

All three loops iterate `0 .. cache_fill_count - 1` (full attention).

### 2.2 Target Behavior (Sliding Window)

Define `WINDOW_SIZE` as a new parameter (default 128 in production). When sliding window is active:

```
eff_count    = min(cache_fill_count, WINDOW_SIZE)
window_start = (cache_fill_count <= WINDOW_SIZE) ? 0
             : cache_fill_count - WINDOW_SIZE
```

All three loops iterate over `window_start .. cache_fill_count - 1`, i.e., `eff_count` positions.

When `cache_fill_count <= WINDOW_SIZE`, `window_start = 0` and `eff_count = cache_fill_count`, producing **bit-identical** output to full attention. This is the backward compatibility guarantee.

### 2.3 Ring Buffer Addressing

The KV cache is a ring buffer. After `NUM_SLOTS` writes, `wr_ptr` wraps to 0 and old entries are overwritten. The logical index `i` (offset from `window_start`) must be translated to a physical cache address:

```
physical_addr = (wr_ptr - cache_fill_count + window_start + i) mod NUM_SLOTS
```

Alternatively, the cache module tracks `fill_count` and the attention module uses a modular subtraction to compute the read address for logical position `i`. This detail is the RTL designer's responsibility; the test plan verifies correctness by checking output against a golden model that independently computes the same addressing.

---

## 3. Test Configuration

### 3.1 Bring-Up Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HIDDEN | 8 | Bring-up fast sim |
| K_LATENT | 4 | Bring-up fast sim |
| V_LATENT | 4 | Bring-up fast sim |
| NUM_SLOTS | 256 | Large enough for real sliding window (128 < 256) |
| MAX_POS | 256 | Match NUM_SLOTS |
| WEIGHT_W | 16 | Q12 signed |
| DATA_W | 32 | Q12 signed |
| WINDOW_SIZE | 128 | Production window size |

256 slots with 128-size window ensures the test exercises:
- Full attention for `fill_count <= 128` (backward compat)
- True sliding for `fill_count > 128` (key decoupling)
- Ring-buffer wrap-around for `fill_count >= 256`

### 3.2 Production Parameters (Verilator only)

| Parameter | Value |
|-----------|-------|
| NUM_SLOTS | 4096 |
| WINDOW_SIZE | 128 |

Production-scale validation is deferred to Phase 2B verification runs. The bring-up testbench validates logical correctness; Verilator validates parameterization at scale.

---

## 4. Test Scenarios

### T1: Window Size Equals Cache Size (Backward Compatibility)

**Setup:** `WINDOW_SIZE = 128`, cache preloaded with `N = 64` entries.

**Stimulus:** Run one inference token after preload.

**Expected:** Output is bit-identical to the current `mla_attention_v2` (full attention) for the same input and cache state.

**Why:** This is the critical regression guard. If this test fails, the sliding window modification has broken the baseline full-attention path. `cache_fill_count (64) <= WINDOW_SIZE (128)` so the module must fall back to full attention.

**Golden:** Run the current (unmodified) `mla_attention_v2` with identical weights, inputs, and cache preload. Compare outputs element-by-element.

**Pass:** All `HIDDEN` output elements match the golden reference exactly.

### T2: Window Size Smaller Than Cache Size (True Sliding)

**Setup:** `WINDOW_SIZE = 128`, cache preloaded with `N = 200` entries.

**Stimulus:** Run one inference token after preload.

**Expected:** Output reflects attention only to positions 72..199 (the last 128 of 200). Positions 0..71 contribute zero to the output.

**Golden:** Python script computes:
1. Q vector (after RoPE) from the hidden input and Q weights.
2. K vectors for positions 72..199 from cache.
3. Dot-product scores: `score[i] = sum(Q_q12 * K_cache[72+i] >> 12)` for `i in 0..127`.
4. Softmax: `exp_score[i] = exp_lut(score[i] - max(score))`.
5. `inv_scale = 4096*4096 / sum(exp_score)`.
6. `weight[i] = exp_score[i] * inv_scale >> 12`.
7. `V_out = sum(weight[i] * V_cache[72+i] >> 12)` for `i in 0..127`.
8. Upper dims `V_LATENT..HIDDEN-1` = V_r (current token V, no attention blending).

**Pass:** All `HIDDEN` output elements match Python golden within Q12 rounding tolerance. Non-window positions (0..71) are verified to have zero contribution (checked by Python: compute what full-attention output would be and confirm it differs).

### T3: Empty Cache (Self-Attention Only)

**Setup:** Reset DUT. No preload. Cache is empty.

**Stimulus:** Run one inference token.

**Expected:** Output = V_r (current token's V vector), same as current behavior. Module detects `cache_was_empty` and skips directly from `S_SCORE_INIT` to `S_OUTPUT`.

**Pass:** Output matches V_r exactly. No X propagation. No hang waiting for cache reads that never come.

**Regression:** This must continue to work exactly as before -- the sliding window change should not touch the empty-cache fast path.

### T4: Single Entry Cache

**Setup:** Reset. Run token 1 (fills cache with 1 entry). Then run token 2.

**Stimulus:** Token 2 processes with `cache_fill_count = 1` (the one entry from token 1).

**Expected:** Attention blends between token 1's cached KV and token 2's Q. Since `cache_fill_count (1) <= WINDOW_SIZE (128)`, this is full attention -- output must be bit-identical to the current module for the same two-token sequence.

**Pass:** Token 2 output matches current `mla_attention_v2` output for the identical sequence.

### T5: Cache Wrap-Around with Window (Ring Buffer Boundary)

**Setup:** `NUM_SLOTS = 256`. Preload 256 entries (cache fully filled). Then run "overflow" writes: 30 additional tokens pushed through the pipeline, causing `fill_count` to saturate at 256 and `wr_ptr` to wrap. Effective logical positions are 30..285, but physical storage wraps at slot 255.

**Stimulus:** After the 30 overflow writes, run one inference token. `cache_fill_count = 256`, `WINDOW_SIZE = 128`. Window should cover logical positions 128..255 (the last 128 of the 256 cached entries, which physically span `wr_ptr` wraparound).

**Golden:** Python uses the known preload values and the known write pointer to reconstruct physical cache layout. Computes expected output for logical positions 128..255.

**Pass:** Output matches Python golden within Q12 tolerance. This is the most critical ring-buffer correctness test.

### T6: Deterministic Output (Same Seed, Same Result)

**Setup:** Reset. Preload cache with 200 entries using deterministic seed. Run the same inference twice.

**Stimulus:** Two identical inference tokens, back-to-back, with identical Q.

**Expected:** Both output vectors match each other bit-for-bit.

**Pass:** `y_flat(first_run) === y_flat(second_run)` for all bits.

**Note:** This test also verifies that the first run does not corrupt cache state (K/V entries used for attention are read-only during the attention pass), and that the second run's starting state (scores array cleared, accumulators reset) is identical.

### T7: Window Boundary Exactness (Edge Case)

**Setup:** Preload exactly `WINDOW_SIZE = 128` entries plus 1 extra (129 total).

**Stimulus:** Run one inference token.

**Expected:** Window covers positions 1..128 (128 positions). Position 0 is excluded.

**Golden:** Python computes attention for positions 1..128 and for position 0..128 (129 positions, full attention). The output must match the 128-position window result, NOT the 129-position full result.

**Pass:** Output matches Python window-128 golden (positions 1..128). Output does NOT match Python full-attention golden (positions 0..128). This confirms the boundary is correct and there is no off-by-one error.

### T8: Zero-Input Robustness (No X Propagation)

**Setup:** Preload cache with 200 entries of all-zero K/V.

**Stimulus:** Run inference with all-zero hidden input and identity weights.

**Expected:** All scores are 0. `score_max = 0`. `exp_lut(0) = 4096` (the first bin). `exp_sum = 128 * 4096`. `inv_scale = 4096*4096 / (128*4096) = 4096/128 = 32`. All weights = `4096 * 32 >> 12 = 32`. All V values are 0. Output = 0 for `V_LATENT` dims, V_r for upper dims.

**Pass:** No X propagation. Output is all defined values. Does not hang in the reciprocal computation (no div-by-zero since `exp_sum = 128*4096 > 0`).

---

## 4. Golden Reference Generation

### 4.1 Python Golden Script

A Python script (`scripts/simulation/sliding_window_golden.py`) generates per-test expected outputs. The script is **independent** of the RTL -- it does not read RTL source, only the algorithmic specification.

**Inputs (per test):**
- `W_qkv` weights: Q projection, K projection, V projection matrices
- `rope_lut`: sin/cos table for RoPE positions
- `cache_k`, `cache_v`: cached K_latent and V_latent entries (array of `cache_fill_count` vectors)
- `hidden`: input hidden state vector
- `position`: token position for RoPE
- `window_size`: sliding window size
- `cache_fill_count`: number of valid cache entries

**Computation flow:**

```python
# 1. QKV projection (Q12 arithmetic)
Q_raw = matmul(W_q, hidden) >> 12     # [HIDDEN]
K_latent = matmul(W_k, hidden) >> 12  # [K_LATENT]
V_r = matmul(W_v, hidden) >> 12       # [HIDDEN]

# 2. RoPE on Q
for dim_pair in range(HIDDEN // 2):
    cos_val = rope_lut[position][dim_pair]['cos']
    sin_val = rope_lut[position][dim_pair]['sin']
    q_a = Q_raw[2*dim_pair]
    q_b = Q_raw[2*dim_pair + 1]
    Q_rope[2*dim_pair]     = (q_a * cos_val - q_b * sin_val) >> 12
    Q_rope[2*dim_pair + 1] = (q_a * sin_val + q_b * cos_val) >> 12

# 3. Determine window range
if cache_fill_count <= window_size:
    win_start = 0
    win_count = cache_fill_count
else:
    win_start = cache_fill_count - window_size
    win_count = window_size

# 4. Score computation (dot product, Q12)
scores = []
for i in range(win_start, cache_fill_count):
    dot = 0
    for d in range(K_LATENT):
        dot += (Q_rope[d] * cache_k[i][d]) >> 12
    scores.append(dot)

# 5. Softmax (match hardware exp_lut exactly)
def exp_lut(adj):
    if adj > -256:      return 4096
    elif adj > -1024:   return 3545
    elif adj > -2048:   return 2588
    elif adj > -4096:   return 1507
    elif adj > -8192:   return 538
    else:               return 48

score_max = max(scores)
exp_sum = sum(exp_lut(s - score_max) for s in scores)

# 6. Weighted V sum
if exp_sum == 0:
    inv_scale = 4096
else:
    inv_scale = (4096 * 4096) // exp_sum

V_out = [0] * HIDDEN
for i, s in enumerate(scores):
    exp_val = exp_lut(s - score_max)
    weight = (exp_val * inv_scale) >> 12
    ci = win_start + i
    for d in range(V_LATENT):
        V_out[d] += (weight * cache_v[ci][d]) >> 12

# 7. Upper dims = V_r (no attention blending)
for d in range(V_LATENT, HIDDEN):
    V_out[d] = V_r[d]
```

### 4.2 Acceptable Error

For bring-up parameters (HIDDEN=8, small integer inputs < 10^4):

- **Bit-exact match** for all integer arithmetic paths (dot products, shifts, LUT lookups).
- The hardware uses `>>>` (arithmetic right shift) for all divisions by powers of 2. The Python golden must match this exactly using `>>` (Python's floor division on positive numbers is equivalent to arithmetic shift for signed Q12 values -- watch out for negative rounding on negative values).
- **Tolerance:** 0 (must be bit-exact) for bring-up. For production (HIDDEN=7168), tolerance of +/-1 LSB per dimension is acceptable due to accumulation order differences (the hardware accumulates one dot product at a time vs. Python's sum, which can differ in the last bit).

---

## 5. Testbench Structure

### 5.1 File: `rtl/sim/tb_mla_attention_v2_window.sv`

**Target:** Under 300 lines (complexity budget).

**Architecture:**

```
tb_mla_attention_v2_window
├── Clock/reset generation        (~15 lines)
├── DUT instantiation             (~25 lines)
│   └── mla_attention_v2 #(.NUM_SLOTS(256), ...)
├── Helper tasks                   (~60 lines)
│   ├── load_qkv_identity()       — from existing tb
│   ├── load_rope_identity()      — cos=4096, sin=0 for all positions
│   ├── preload_cache(task)       — fill cache via preload port
│   ├── run_inference(task)       — send hidden, wait for output
│   └── run_with_preload(task)    — preload + infer + return output
├── Checker tasks                  (~40 lines)
│   ├── check_output(name, expected[], actual[]) — per-element compare
│   ├── check_not_equal(a[], b[]) — verify two outputs differ
│   └── check_bit_identical(a[], b[]) — bitwise ===
├── Python golden arrays           (~40 lines)
│   └── Pre-computed expected values as SV literals
└── Main test sequence            (~100 lines)
    └── T1-T8 in order, with pass/fail counters
```

### 5.2 DUT Instantiation

```systemverilog
localparam int HIDDEN     = 8;
localparam int K_LATENT   = 4;
localparam int V_LATENT   = 4;
localparam int NUM_SLOTS  = 256;   // key: > WINDOW_SIZE for real sliding
localparam int WINDOW_SZ  = 128;   // sliding window size
localparam int MAX_POS    = 256;
localparam int WEIGHT_W   = 16;
localparam int DATA_W     = 32;

mla_attention_v2 #(
    .HIDDEN(HIDDEN), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
    .NUM_SLOTS(NUM_SLOTS), .MAX_POS(MAX_POS),
    .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W)
) dut (.*);
```

If sliding window adds a `WINDOW_SIZE` parameter to the module, it must be wired here. If the window size is derived from `lpu_config_pkg` (production default 128), the testbench overrides it locally via the parameter.

### 5.3 Cache Preload Strategy

To test true sliding window (T2, T5, T7), the cache must be filled with known, deterministic entries before inference begins. The `cache_preload_en` port on `mla_attention_v2` feeds directly into `mla_kv_cache.preload_en`, which writes K_latent/V_latent at the current `wr_ptr` and increments the pointer.

**Preload task:**

```systemverilog
task preload_cache(input int count, input int seed);
    for (int i = 0; i < count; i++) begin
        @(posedge clk);
        cache_preload_en <= 1;
        // Deterministic values: K[d] = seed + i*K_LATENT + d, V[d] = seed + 1000 + i*V_LATENT + d
        for (int d = 0; d < K_LATENT; d++)
            cache_preload_K_flat[d*DATA_W+:DATA_W] <= seed + i*K_LATENT + d;
        for (int d = 0; d < V_LATENT; d++)
            cache_preload_V_flat[d*DATA_W+:DATA_W] <= seed + 1000 + i*V_LATENT + d;
        @(posedge clk);
        cache_preload_en <= 0;
    end
endtask
```

### 5.4 Python Golden as SV Literals

For each test case, the golden output vector is pre-computed by the Python script and embedded as SystemVerilog array literals in the testbench. This eliminates runtime golden computation and keeps the testbench self-contained.

**Format:**

```systemverilog
// T2 golden: 200 entries, window=128, seed=42
localparam logic signed [DATA_W-1:0] T2_EXPECTED [0:HIDDEN-1] = '{
    32'sd1234, 32'sd5678, 32'sd9012, 32'sd3456,   // V_LATENT dims
    32'sd7890, 32'sd1111, 32'sd2222, 32'sd3333    // upper dims = V_r
};
```

The Python script that generates these values must be committed alongside the testbench (`scripts/simulation/sliding_window_golden.py`), so any engineer can regenerate the golden values if the test parameters change.

### 5.5 Self-Checking Logic

Every test case checks its result against the embedded golden and prints `[PASS]` or `[FAIL]` with the failing element and expected-vs-got values. The final summary reports `N PASSED, M FAILED` and calls `$fatal` on any failure.

---

## 6. Pass/Fail Criteria

### 6.1 Per-Test Criteria

| Check | Applied To | Method |
|-------|-----------|--------|
| Output matches Python golden | T1, T2, T4, T5, T7, T8 | Element-wise `===` comparison |
| Output differs from full-attn golden | T2, T7 | Verify difference proves window is active |
| Output is bit-identical across runs | T6 | Bitwise `===` between run1 and run2 |
| No X or Z in any output bit | All | `$isunknown()` check on y_flat |
| Pipeline completes within timeout | All | 500-cycle watchdog per inference |
| Empty-cache fast path works | T3 | Output = V_r exactly |

### 6.2 Go/No-Go

- **Go:** All 8 tests pass on Icarus (`iverilog -g2012`).
- **No-Go:** Any test fails or any X/Z propagation detected.
- **Soft blocker:** Production Verilator run (TBD Phase 2B). Bring-up correctness gates the RTL merge; production-scale validation can follow.

---

## 7. Regression Impact

### 7.1 Existing Tests That Must Continue to Pass

| Testbench | Tests | Impact of Sliding Window Change |
|-----------|-------|--------------------------------|
| `tb_mla_attention_v2.sv` | T1 Identity, T2 Non-identity, T3 Two-token, T4 RoPE | T1-T3 use cache_fill_count <= 4 (NUM_SLOTS=64). Window size = 128. No behavioral change expected. Must bit-identical. |
| `tb_full_transformer_layer.sv` | All 15 tests | Attention is a sub-module. No change expected since sequence lengths are short (< 64). |
| `tb_mla_qkv.sv` | QKV projection | Unaffected (only mla_qkv_proj, no attention). |
| `tb_mla_kv_cache.sv` | KV cache read/write | Unaffected (only mla_kv_cache, no attention). |
| `tb_chip_12layer.sv` | 12-layer integration | Attention sub-module within layers. Must continue passing. |

### 7.2 Regression Run Order

```
1. tb_mla_attention_v2.sv        ← must pass first (backward compat gate)
2. tb_mla_attention_v2_window.sv ← new tests (sliding window)
3. tb_full_transformer_layer.sv  ← integration (attention embedded)
4. tb_chip_12layer.sv            ← full chip (attention embedded)
5. All other testbenches         ← verify no collateral damage
```

The `tb_mla_attention_v2.sv` run is the **gating test** -- if it fails after the sliding window modification, the implementation is buggy and must not proceed to PR.

---

## 8. Implementation Notes for RTL Developer

### 8.1 Required Changes to `mla_attention_v2.sv`

The sliding window change touches three loop termination conditions and one address computation:

1. **New parameter:** `WINDOW_SIZE` (default from `lpu_config_pkg`, bring-up override = 128).
2. **Window start computation** (in `S_SCORE_INIT` or a combinational block):
   ```systemverilog
   logic [$clog2(NUM_SLOTS)-1:0] win_start;
   logic [$clog2(NUM_SLOTS)-1:0] eff_count;
   assign eff_count = (cache_fill_count <= WINDOW_SIZE) ? cache_fill_count : WINDOW_SIZE;
   assign win_start = (cache_fill_count <= WINDOW_SIZE) ? '0 : cache_fill_count - WINDOW_SIZE;
   ```
3. **Loop bounds:** Replace `cache_fill_count` with `eff_count` in termination checks (`score_idx == cache_fill_count - 1` becomes `score_idx == eff_count - 1`).
4. **Cache address:** Replace `score_idx` (which was the logical index) with `win_start + score_idx` (or the ring-buffer physical address derived from it).
5. **Same changes for** `exp_idx` loop and `accum_idx` loop.

### 8.2 Self-Attention Edge Case

When `cache_was_empty`, the module already skips to `S_OUTPUT`. This path must remain untouched -- no window computation should execute when there are no cached entries.

### 8.3 Ring Buffer Address Mapping

The current code uses `score_idx` directly as `cache_rd_addr`. With sliding window, the read address for logical position `i` (where `i` goes from 0 to `eff_count - 1`) is:

```
physical_addr = (wr_ptr - cache_fill_count + win_start + i) mod NUM_SLOTS
```

Since `wr_ptr = cache_fill_count mod NUM_SLOTS` (write pointer equals the count modulo slots in a ring buffer), this simplifies to:

```
physical_addr = (win_start + i) mod NUM_SLOTS
```

which is the same as the current `score_idx` when `win_start = 0` (full attention). For sliding window with wrap-around (when `win_start + eff_count > NUM_SLOTS`), the modular arithmetic correctly maps logical positions to physical addresses.

---

## 9. Deliverables Checklist

- [ ] `docs/eng/10_phase2a_sliding_window_test_plan.md` (this document) -- reviewed by VERIF-LEAD
- [ ] `scripts/simulation/sliding_window_golden.py` -- Python golden reference generator
- [ ] `rtl/sim/tb_mla_attention_v2_window.sv` -- new testbench (under 300 lines)
- [ ] Modified `mla_attention_v2.sv` -- sliding window support (by RTL-DEV1)
- [ ] `rtl/sim/Makefile` -- add `tb_mla_attention_v2_window` target
- [ ] Regression run: all existing testbenches pass with modified `mla_attention_v2`
- [ ] PR with all changes + passing CI results

---

## Appendix A: exp_lut Reference (from current RTL)

```
adj > -256      → 4096
adj > -1024     → 3545
adj > -2048     → 2588
adj > -4096     → 1507
adj > -8192     → 538
otherwise       → 48
```

The Python golden must replicate this exactly, not use `math.exp()`, to achieve bit-exact agreement with hardware.

## Appendix B: Q12 Arithmetic Conventions

- Multiplication of two Q12 values produces Q24; right-shift by 12 restores Q12.
- Arithmetic right shift (`>>>`) is used for signed values.
- The reciprocal `inv_scale` is scaled: `(4096 * 4096) / exp_sum`. In Q12, 4096 represents 1.0. So `4096*4096` represents 1.0 in Q24, and dividing by `exp_sum` (also Q12) yields `1/exp_sum` in Q12.
- Weight computation: `exp_val (Q12) * inv_scale (Q12) >> 12 = Q12 weight`.
- Output accumulation: `weight (Q12) * V (Q12) >> 12 = Q12`.
