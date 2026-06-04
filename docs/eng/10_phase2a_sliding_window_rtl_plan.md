# Phase 2A Sliding Window Attention -- RTL Implementation Plan

> **Owner:** RTL-ENG2 (MLA/Attention)
> **Scope:** `mla_attention_v2.sv` only (no changes to `mla_kv_cache.sv`)
> **Target:** Sliding window attention -- attend to last `SLIDING_WINDOW` tokens instead of all cached positions
> **Status:** PLAN -- not yet implemented

---

## 1. Port/Signal Changes

### 1.1 New Parameter: `SLIDING_WINDOW` (add to lpu_config.svh)

`lpu_config.svh` currently has no sliding window parameter. Add to both production and bring-up sections:

| Section | Value | Rationale |
|---------|-------|-----------|
| Production (`FPGA_LPU_PRODUCTION`) | 128 | Matches DeepSeek V4 architecture: local window = 128 |
| Bring-up (simulation) | 128 | Same value; graceful degradation when cache < window |

Add between `LPU_MAX_SEQ_LEN` and `LPU_NUM_LAYERS` in both sections:

```systemverilog
parameter int LPU_SLIDING_WINDOW  = 128;     // sliding window attention size
```

### 1.2 New Parameter on mla_attention_v2

Add after `MAX_POS` (current line 20):

```systemverilog
parameter int WINDOW_SIZE = lpu_config_pkg::LPU_SLIDING_WINDOW,
```

### 1.3 New Input Port

Add after `position` (current line 32):

```systemverilog
// Sliding window control (1 = window mode, 0 = full attention)
input  logic                         window_mode,
```

Stable per-token. Latched internally at `S_IDLE` accept to ensure it cannot change mid-computation.

### 1.4 New Internal Registers

| Signal | Width | Purpose |
|--------|-------|---------|
| `window_mode_r` | 1 | Latched copy of `window_mode` input (captured at S_IDLE) |
| `cache_wr_addr_latched` | `$clog2(NUM_SLOTS)` | Physical address written during S_CACHE_WR (the current token's KV slot) |
| `window_base` | `$clog2(NUM_SLOTS)` | Physical buffer address of first (oldest) entry in the window |
| `window_count` | `$clog2(NUM_SLOTS+1)` | Number of entries to iterate: `min(fill_count, WINDOW_SIZE)` |

### 1.5 Signal Change Summary

```
[NO CHANGE]  clk, rst_n, in_valid, hidden_flat, in_ready, position
[NO CHANGE]  qkv_wt_wr_en, qkv_wt_sel, qkv_wt_row, qkv_wt_col, qkv_wt_wr_data
[NO CHANGE]  rope_lut_wr_en, rope_lut_pos, rope_lut_pair, rope_lut_sin, rope_lut_cos
[NO CHANGE]  cache_preload_en, cache_preload_K_flat, cache_preload_V_flat
[NO CHANGE]  out_valid, out_ready, y_flat
[*** NEW ***] window_mode (line 33)
```

### 1.6 Scores Array Semantics (No Physical Change)

`scores` array remains `[NUM_SLOTS]` entries wide (line 141). In window mode, only indices `[0 : window_count-1]` are populated. Indices `[window_count : NUM_SLOTS-1]` contain stale data and are never read. No RAM increase.

---

## 2. FSM Changes

### 2.1 Existing FSM States (for reference)

```
S_IDLE          -- accept new token, latch cache_was_empty
S_QKV_PROJ      -- wait for QKV projection complete
S_ROPE          -- wait for RoPE rotation complete
S_CACHE_WR      -- write K_latent/V_latent to cache  [MODIFIED]
S_SCORE_INIT    -- initialize score loop              [MODIFIED]
S_CACHE_RD      -- read K_latent from cache           [MODIFIED: addr gen]
S_ATTN_SCORE    -- compute dot product, check end     [MODIFIED: end condition]
S_EXP_LOOP      -- softmax exp loop                   [MODIFIED: end condition]
S_INV           -- compute reciprocal                 [NO CHANGE]
S_ACCUM_INIT    -- initialize accum loop              [MODIFIED]
S_CACHE_RD2     -- read V_latent from cache           [MODIFIED: addr gen]
S_ACCUM_LOOP    -- compute weighted V sum, check end  [MODIFIED: end condition]
S_OUTPUT        -- output result                      [NO CHANGE]
```

**No new states.** All changes are modifications to existing states.

### 2.2 Modified State Behaviors

#### S_CACHE_WR (line 225)

Existing:
```systemverilog
S_CACHE_WR: begin
    state <= S_SCORE_INIT;
end
```

New:
```systemverilog
S_CACHE_WR: begin
    cache_wr_addr_latched <= cache_wr_addr;   // latch where we just wrote
    state <= S_SCORE_INIT;
end
```

Rationale: `cache_wr_addr` = `wr_ptr` from the KV cache module. During this cycle, it reflects the address being written. Latching it captures the physical slot of the current token, which is needed to compute the window base address in the next state.

#### S_SCORE_INIT (line 232)

Existing:
```systemverilog
S_SCORE_INIT: begin
    score_idx  <= '0;
    score_max  <= 32'sh80000000;
    if (cache_was_empty) begin
        state <= S_OUTPUT;
    end else begin
        state <= S_CACHE_RD;
    end
end
```

New:
```systemverilog
S_SCORE_INIT: begin
    score_idx  <= '0;
    score_max  <= 32'sh80000000;
    if (cache_was_empty) begin
        state <= S_OUTPUT;
    end else begin
        // Compute window parameters
        if (window_mode_r && (cache_fill_count > WINDOW_SIZE)) begin
            window_count <= WINDOW_SIZE[$clog2(NUM_SLOTS+1)-1:0];
            // window_base = cache_wr_addr_latched - WINDOW_SIZE + 1 (with wrap)
            // Equivalent to: (cache_wr_addr_latched + NUM_SLOTS - WINDOW_SIZE + 1) % NUM_SLOTS
            if (cache_wr_addr_latched >= (WINDOW_SIZE - 1))
                window_base <= cache_wr_addr_latched - (WINDOW_SIZE - 1);
            else
                window_base <= cache_wr_addr_latched + NUM_SLOTS - (WINDOW_SIZE - 1);
        end else begin
            // Full attention: window = entire cache
            window_count <= cache_fill_count;
            window_base <= '0;
        end
        state <= S_CACHE_RD;
    end
end
```

Key arithmetic: `window_base = (cache_wr_addr_latched - WINDOW_SIZE + 1) mod NUM_SLOTS`.
- When `cache_wr_addr_latched >= WINDOW_SIZE - 1`: no wrap, just subtract.
- When `cache_wr_addr_latched < WINDOW_SIZE - 1`: wrap around, add NUM_SLOTS before subtracting.

The `window_count` is computed as `WINDOW_SIZE` when window active, `cache_fill_count` otherwise.

Note: `WINDOW_SIZE - 1` is a compile-time constant (127 production, 127 bring-up). Strong synthesis can propagate it.

#### S_CACHE_RD (line 242) and S_CACHE_RD2 (line 298) -- Address Generation

Change the combinational `cache_rd_addr` logic (line 175):

Existing:
```systemverilog
cache_rd_addr = (state == S_CACHE_RD2) ? accum_idx : score_idx;
```

New:
```systemverilog
// Compute window-relative physical address
if (window_mode_r && (cache_fill_count > WINDOW_SIZE)) begin
    logic [$clog2(NUM_SLOTS)-1:0] rel_idx, candidate;
    rel_idx  = (state == S_CACHE_RD2) ? accum_idx : score_idx;
    candidate = window_base + rel_idx;
    // Single-wrap check: candidate can overflow by at most one wrap
    cache_rd_addr = (candidate >= NUM_SLOTS[$clog2(NUM_SLOTS)-1:0])
                    ? (candidate - NUM_SLOTS[$clog2(NUM_SLOTS)-1:0])
                    : candidate;
end else begin
    cache_rd_addr = (state == S_CACHE_RD2) ? accum_idx : score_idx;
end
```

This MUST remain combinational (`always_comb`) -- one cycle of address computation latency would break the read pipeline. The ternary chain adds ~1 LUT level; at 100 MHz bring-up and 450 MHz production, this is well within timing.

#### S_ATTN_SCORE (line 246) -- Loop End Condition

Change line 254 from:
```systemverilog
if (score_idx == cache_fill_count - 1) begin
```
to:
```systemverilog
if (score_idx == window_count - 1) begin
```

#### S_EXP_LOOP (line 267) -- Loop End Condition

Change line 270 from:
```systemverilog
if (exp_idx == cache_fill_count - 1) begin
```
to:
```systemverilog
if (exp_idx == window_count - 1) begin
```

Similarly, the exp computation at line 268 reads `scores[exp_idx]`. In window mode, `exp_idx` runs 0..`window_count-1`, and `scores[0..window_count-1]` are the window's dot products. Same behavior with different iteration count.

#### S_ACCUM_INIT (line 294) -- No change needed

The `accum_idx <= '0` and `V_acc[d] <= '0` reset is identical.

#### S_ACCUM_LOOP (line 302) -- Loop End Condition

Change line 314 from:
```systemverilog
if (accum_idx == cache_fill_count - 1) begin
```
to:
```systemverilog
if (accum_idx == window_count - 1) begin
```

### 2.3 Reset Initialization

Add to the reset block (line 182-195):
```systemverilog
window_mode_r  <= 1'b0;
cache_wr_addr_latched <= '0;
window_base    <= '0;
window_count   <= '0;
```

### 2.4 Latch window_mode at S_IDLE

Add in S_IDLE (line 200-205):
```systemverilog
S_IDLE: begin
    if (in_valid) begin
        window_mode_r <= window_mode;
        cache_was_empty <= cache_empty;
        state <= S_QKV_PROJ;
    end
end
```

This ensures `window_mode_r` is stable throughout the entire attention computation for a given token, even if the external `window_mode` input changes between tokens.

---

## 3. Cache Address Generation (Ring Buffer Wrap Handling)

### 3.1 Ring Buffer Recap

The KV cache (`mla_kv_cache.sv`) is a ring buffer:
- `wr_ptr` increments on each write, wrapping at `NUM_SLOTS - 1` (line 108)
- `entry_count` saturates at `NUM_SLOTS` (line 109)
- `fill_count` = `entry_count` (line 76)
- `wr_addr` = `wr_ptr` (line 75) -- reflects the address being written

### 3.2 Window Base Computation

```
Timeline (production, NUM_SLOTS=4096, WINDOW_SIZE=128):

Token 0: write @ addr 0. fill_count=1.   window_count=1 (fill<window). base=0.
Token 1: write @ addr 1. fill_count=2.   window_count=2. base=0.
...
Token 127: write @ addr 127. fill_count=128. window_count=128. base=0.
Token 128: write @ addr 128. fill_count=129. window_count=128. base=1.
Token 129: write @ addr 129. fill_count=130. window_count=128. base=2.
...

Token 4095: write @ addr 4095. fill_count=4096. window_count=128. base=3968.
Token 4096: write @ addr 0.    fill_count=4096. window_count=128. base=3969.
  -- Window should cover addr 3969..0 (wrapping). Check:
  -- cache_wr_addr_latched = 0, WINDOW_SIZE-1 = 127
  -- 0 >= 127? No. So window_base = 0 + 4096 - 127 = 3969. Correct!
Token 4097: write @ addr 1.    fill_count=4096. window_count=128. base=3970.
...
```

### 3.3 Address Generation During Iteration

During S_CACHE_RD and S_CACHE_RD2, the combinational logic computes:
```
candidate = window_base + rel_idx;   // rel_idx = score_idx or accum_idx
cache_rd_addr = (candidate >= NUM_SLOTS) ? (candidate - NUM_SLOTS) : candidate;
```

Since `window_base < NUM_SLOTS` and `rel_idx < window_count <= WINDOW_SIZE <= NUM_SLOTS`, the sum is at most `2*NUM_SLOTS - 2`, so a single subtraction handles the wrap.

Example (NUM_SLOTS=4096, base=3969, window_count=128):
- rel_idx=0  -> candidate=3969 -> addr=3969  (first window entry)
- rel_idx=126 -> candidate=4095 -> addr=4095  (last entry before wrap)
- rel_idx=127 -> candidate=4096 -> addr=0     (wraps, uses ">= NUM_SLOTS" path)

This correctly reads entries at physical addresses [3969, 3970, ..., 4095, 0] -- the 128 most recent tokens.

---

## 4. Score/Softmax/Accum Loop Modifications

### 4.1 Unified Change Pattern

Every loop end condition that references `cache_fill_count` changes to `window_count`:

| State | Line | Current | New |
|-------|------|---------|-----|
| S_ATTN_SCORE | 254 | `score_idx == cache_fill_count - 1` | `score_idx == window_count - 1` |
| S_EXP_LOOP | 270 | `exp_idx == cache_fill_count - 1` | `exp_idx == window_count - 1` |
| S_ACCUM_LOOP | 314 | `accum_idx == cache_fill_count - 1` | `accum_idx == window_count - 1` |

### 4.2 Graceful Degradation (Implicit)

The design handles three regimes automatically:

| Condition | window_count | Behavior |
|-----------|-------------|----------|
| `cache_was_empty` | -- | Skip all loops, output V_r directly (first token) |
| `fill_count <= WINDOW_SIZE` | `fill_count` | Full attention (identical to current behavior) |
| `fill_count > WINDOW_SIZE` (window_mode=1) | `WINDOW_SIZE` | Sliding window -- only last 128 entries |
| `fill_count > WINDOW_SIZE` (window_mode=0) | `fill_count` | Full attention (backward compatible) |

### 4.3 Test Scenario: Bring-Up (NUM_SLOTS=64, WINDOW=128)

In bring-up simulation:
- `NUM_SLOTS = 64`, `WINDOW_SIZE = 128`
- `WINDOW_SIZE > NUM_SLOTS`, so `fill_count <= WINDOW_SIZE` always
- The `if (window_mode_r && (cache_fill_count > WINDOW_SIZE))` condition in S_SCORE_INIT NEVER triggers in bring-up
- Result: full attention, identical to current behavior
- This is correct -- the window mode code path is not exercised, but the module continues to function

To test the window logic in bring-up, reduce `WINDOW_SIZE` via parameter override: `mla_attention_v2 #(.WINDOW_SIZE(4))` so that 4 < 64. See Section 6.

---

## 5. Estimated Resource Impact

### 5.1 Logic (ALMs/LUTs)

| Addition | Width | Gates/ALMs (est.) |
|----------|-------|-------------------|
| `cache_wr_addr_latched` register | 12 bits (production) / 6 bits (sim) | ~6 ALMs |
| `window_base` register | 12 bits | ~6 ALMs |
| `window_count` register | 13 bits | ~7 ALMs |
| `window_mode_r` register | 1 bit | ~1 ALM |
| Address generation combinational (cmp+mux+sub) | 12-bit paths | ~15 ALMs |
| S_SCORE_INIT window_base computation (cmp+sub) | 12-bit paths | ~12 ALMs |
| End-condition comparison (now uses window_count) | same width | ~0 (reconnect) |

**Total logic increase:** ~50 ALMs (negligible vs. ~12,300 DSPs and ~1M ALMs on Agilex 7).

### 5.2 RAM/DSP

**Zero additional RAM/DSP.** The `scores` array (NUM_SLOTS entries of DATA_W width) and `V_acc` array (HIDDEN entries) are unchanged. All new registers are distributed logic, not block RAM.

### 5.3 Timing

The critical path additions:
1. **S_SCORE_INIT combinational:** `(cache_wr_addr_latched >= (WINDOW_SIZE - 1)) ? subtract : subtract_with_wrap` -- single 12-bit subtract + mux, ~2-3 LUT levels.
2. **Address generation combinational:** `(window_base + rel_idx >= NUM_SLOTS) ? subtract_NUM_SLOTS : pass` -- single 12-bit add + 12-bit compare + mux, ~4-5 LUT levels.

Both are well under the 2.2 ns period at 450 MHz for typical Agilex 7 ALM delays (~300 ps per LUT level). No timing risk.

### 5.4 Operational Intensity (OI) Impact

Current (full attention, P=4096):
- KV reads per decode step: 4096 entries x 512 elements x 4 bytes = 8 MB
- MACs: 4096 x 512 dot + 4096 x 512 weighted-sum = 4.2M MACs
- OI = 4.2M / 8MB = 0.52 MACs/byte

After sliding window (window=128, P=4096):
- KV reads: 128 entries x 512 elements x 4 bytes = 256 KB
- MACs: 128 x 512 dot + 128 x 512 weighted-sum = 131K MACs
- OI = 131K / 256KB = 0.52 MACs/byte (same OI for attention sub-block)

The OI improvement comes from the SYSTEM level: with sliding window, the KV read portion of decode latency drops by 32x (4096/128), reducing the fraction of time spent waiting for KV reads to near zero. The roofline constraint (OI >= 13.1) is satisfied when you consider that attention computation is just one phase -- the dominant bottleneck is expert weight loading (see Phase 2 execution plan Section 0.4), and sliding window eliminates KV reads as a competing bandwidth consumer.

---

## 6. Test Strategy

### 6.1 Existing Tests (Must Continue to Pass)

All 4 existing tests in `tb_mla_attention_v2.sv` must pass with `window_mode=0`:
- T1: Identity passthrough (single-token)
- T2: Non-identity W_Q (single-token)
- T3: Two sequential tokens (attention blending)
- T4: RoPE rotation

### 6.2 New Test Scenarios

Add to `tb_mla_attention_v2.sv` (or create `tb_mla_attention_window.sv`):

#### T5: Window mode with window > fill_count (graceful degradation)

```
Setup: NUM_SLOTS=64, WINDOW_SIZE=128, window_mode=1
Steps:
  1. Load identity QKV weights + identity RoPE LUT
  2. Send 3 tokens sequentially
  3. Token 1: fill_count=0→1, window_count=1, full attention (self-only)
  4. Token 2: fill_count=1→2, window_count=2, full attention (2 entries)
  5. Token 3: fill_count=2→3, window_count=3, full attention (3 entries)
Check: All outputs match full-attention golden (identical to window_mode=0)
```

#### T6: Window mode with window < fill_count (window active)

```
Setup: NUM_SLOTS=64, WINDOW_SIZE=4, window_mode=1
Steps:
  1. Load identity QKV weights + identity RoPE LUT
  2. Send 8 tokens sequentially (h_vec = {100+token*200, 101+token*200, ...})
  3. For tokens 1-4: window_count = fill_count (graceful, full attention)
  4. For token 5: window_count=4, only attends to tokens 2,3,4,5 (last 4)
  5. For token 8: window_count=4, only attends to tokens 5,6,7,8
Check: Token 5 output differs from full-attention golden (proves window truncation)
        Token 5 output matches window-attention reference (4-entry attention)
```

#### T7: Ring buffer wrap-around window

```
Setup: NUM_SLOTS=64, WINDOW_SIZE=4, window_mode=1
Steps:
  1. Fill cache to capacity (64 tokens)
  2. Continue sending tokens 65, 66, 67, 68
  3. Token 65: wr_ptr wraps to 0, window covers addr 60,61,62,63 (physical)
     Actually: write @ 0, window_base = 0 - 3 + 64 = 61. Window = [61,62,63,0]
  4. Token 68: write @ 3, window_base = 3 - 3 = 0. Window = [0,1,2,3]
Check: Window correctly tracks ring buffer wrap. Outputs deterministic.
        Send tokens 64..68 twice with identical hidden values --
        token 68 output must be identical both times.
```

#### T8: window_mode toggle mid-sequence

```
Setup: NUM_SLOTS=64, WINDOW_SIZE=4
Steps:
  1. Send 5 tokens with window_mode=0 (full attention for all)
  2. Send 1 token with window_mode=1 (windowed attention, window_count=4)
  3. Send 1 token with window_mode=0 (full attention again, window_count=6)
Check: window_mode_r correctly latches per-token.
        Outputs deterministic and match expected mode.
```

#### T9: window_mode=0 regression

```
Setup: NUM_SLOTS=64, WINDOW_SIZE=4, window_mode=0
Steps: Repeat T6 scenario (8 tokens)
Check: All outputs match full-attention golden.
        window_mode=0 completely disables window logic.
```

### 6.3 Golden Model

A simple Python golden model for windowed attention exists in the testbench patterns. Since bring-up uses small integer values (Q12), the 6-bin `exp_lut` produces uniform weights (4096 for all entries when scores cluster near zero). This means:

- Full attention output = average of V_latent from all cached tokens
- Window attention output = average of V_latent from last WINDOW_SIZE tokens

For T6 verification: compute the expected V_latent average from the correct subset and compare.

### 6.4 Test File Location

Option A: Add T5-T9 to existing `rtl/sim/tb_mla_attention_v2.sv` (simpler, less file duplication).
Option B: Create `rtl/sim/tb_mla_attention_window.sv` (cleaner separation).

**Recommendation: Option A** -- add to existing testbench. Sliding window is a feature of the same module, not a different module. The testbench already has 4 tests; adding 5 more is manageable (~150 additional lines).

---

## 7. Risk Analysis

### 7.1 Empty Cache (First Token)

**Risk:** `cache_was_empty=1` triggers S_OUTPUT skip. `window_count` and `window_base` are not computed. If there's a code path that uses them before initialization...

**Mitigation:** The `if (cache_was_empty)` branch (S_SCORE_INIT, line 235-239) unconditionally jumps to S_OUTPUT. The `window_count`/`window_base` registers are initialized to 0 at reset. Even if accidentally read, they would produce index 0 (which is valid for NUM_SLOTS > 0). No functional hazard.

### 7.2 Cache < Window Size (Early Decode)

**Risk:** The `if (window_mode_r && (cache_fill_count > WINDOW_SIZE))` condition in S_SCORE_INIT correctly falls through to the `else` branch (full attention). Confirmed correct.
- `window_count = cache_fill_count` (not WINDOW_SIZE)
- `window_base = 0`
- Loop conditions use `window_count - 1 = cache_fill_count - 1` -- identical to current behavior.

**Mitigation:** T5 explicitly validates this.

### 7.3 Ring Buffer Wrap-Around

**Risk:** When `cache_wr_addr_latched < WINDOW_SIZE - 1`, the window_base computation wraps around: `base = addr + NUM_SLOTS - (WINDOW_SIZE - 1)`. This maps to addresses near the end of the physical buffer. The address generation during iteration then potentially wraps again, requiring the `candidate >= NUM_SLOTS` check.

**Double-check the arithmetic:**
- Case: NUM_SLOTS=4096, WINDOW=128, write @ addr 0 (token 4096)
- Window covers tokens [3969..4096], physical addrs [3969..4095,0]
- `cache_wr_addr_latched = 0`
- `0 >= 127`? No. So `window_base = 0 + 4096 - 127 = 3969`.
- Window indices and physical addrs:
  - rel_idx=0: candidate=3969, addr=3969 ✓ (token 3969)
  - rel_idx=126: candidate=4095, addr=4095 ✓ (token 4095)
  - rel_idx=127: candidate=4096 >= 4096, addr=0 ✓ (token 4096, just written)

**Mitigation:** T7 explicitly validates wrap-around behavior.

### 7.4 Entry Count Saturation

**Risk:** After `NUM_SLOTS` writes, `entry_count` saturates at `NUM_SLOTS` (mla_kv_cache.sv line 109). `fill_count` = `entry_count` = `NUM_SLOTS`. The `window_count` computation correctly uses `min(fill_count, WINDOW_SIZE)`. If `NUM_SLOTS=4096` and `WINDOW=128`, `window_count=128` (correct). No risk.

### 7.5 Comb-Loop Timing with Address Generation

**Risk:** Adding arithmetic to the combinational `cache_rd_addr` path (which feeds directly into `altera_syncram` read address) could increase the critical path. The signals `state`, `accum_idx`, `score_idx`, `window_base` are all registers. The only combinational path is:
```
window_base[11:0] + rel_idx[11:0] -> [11:0] candidate
-> compare with NUM_SLOTS[11:0]
-> mux: candidate or (candidate - NUM_SLOTS)
```

This is an 11-bit add + 12-bit compare + 12-bit mux. At 450 MHz (2.2 ns period), this is approximately 4-5 LUT levels. Agilex 7 ALM delay is ~150-200 ps per level. Total ~0.8 ns. Well under 2.2 ns.

**Mitigation:** If timing becomes an issue in production, pipeline the address generation by adding one cycle of latency and adjusting the FSM to pre-compute addresses. But this is unlikely to be needed.

### 7.6 window_mode Input Race

**Risk:** External `window_mode` changes while attention computation is in progress (between S_QKV_PROJ and S_OUTPUT).

**Mitigation:** `window_mode_r` is latched at S_IDLE and used throughout. Changes to `window_mode` between tokens are harmless -- they'll take effect on the next token.

### 7.7 Scores Array Index Out of Bounds

**Risk:** In window mode, `score_idx` runs 0..`window_count-1` where `window_count <= WINDOW_SIZE <= NUM_SLOTS`. The `scores` array is `[NUM_SLOTS]` entries wide. Since `window_count <= WINDOW_SIZE` and in production `WINDOW_SIZE=128 <= NUM_SLOTS=4096`, all accesses are in bounds.

In bring-up, `NUM_SLOTS=64` and `WINDOW_SIZE=128`. Window mode never activates because `cache_fill_count > WINDOW_SIZE` never holds (fill_count max = 64 < 128). So window-relative indexing never occurs. The `scores[score_idx]` access uses `score_idx` in 0..`cache_fill_count-1` (0..63), well within the 64-entry bounds.

**Mitigation:** You could add an assertion: `assert (score_idx < window_count)` during S_ATTN_SCORE, but this adds simulation-only logic. For production, the bounds are guaranteed by construction.

---

## 8. Implementation Sequence

### Step 1: Add SLIDING_WINDOW to lpu_config.svh

File: `rtl/include/lpu_config.svh`
- Add `LPU_SLIDING_WINDOW = 128` in both production and bring-up sections.

### Step 2: Add WINDOW_SIZE parameter + window_mode port to mla_attention_v2

File: `rtl/attention/mla_attention_v2.sv`
- Add `parameter int WINDOW_SIZE = lpu_config_pkg::LPU_SLIDING_WINDOW` after line 20
- Add `input logic window_mode` after line 32

### Step 3: Add internal registers and reset logic

- Declare `window_mode_r`, `cache_wr_addr_latched`, `window_base`, `window_count`
- Initialize in reset block

### Step 4: Modify S_IDLE -- latch window_mode_r

### Step 5: Modify S_CACHE_WR -- latch cache_wr_addr

### Step 6: Modify S_SCORE_INIT -- compute window_base and window_count

### Step 7: Modify combinational cache_rd_addr -- add window address generation

### Step 8: Modify S_ATTN_SCORE, S_EXP_LOOP, S_ACCUM_LOOP -- change end conditions

### Step 9: Add T5-T9 test scenarios to testbench

### Step 10: Regression test

- Run existing testbench with window_mode=0: T1-T4 must pass
- Run with window_mode=1: T5-T9 must pass
- Run with different WINDOW_SIZE overrides

---

## 9. Files Touched

| File | Change | Owner |
|------|--------|-------|
| `rtl/include/lpu_config.svh` | Add `LPU_SLIDING_WINDOW` parameter | RTL-ENG2 |
| `rtl/attention/mla_attention_v2.sv` | Add parameter, port, registers, FSM modifications | RTL-ENG2 |
| `rtl/sim/tb_mla_attention_v2.sv` | Add T5-T9 test scenarios | RTL-ENG2 |
| `rtl/attention/mla_kv_cache.sv` | **No changes** | -- |
| `rtl/attention/mla_qkv_proj.sv` | **No changes** | -- |
| `rtl/attention/mla_rope.sv` | **No changes** | -- |
| `rtl/layer/layer_compute_engine.sv` | **No changes** (window_mode passed through later) | RTL-ENG1 |
| `rtl/layer/full_transformer_layer.sv` | **No changes** (window_mode connection deferred) | RTL-ENG1 |
| `rtl/chip/chip_top.sv` | **No changes** (window_mode connection deferred) | RTL-ENG1 |

---

## 10. Deferred to Phase 2B

These items are explicitly out of scope for this plan:

1. **Sparse global attention (256 tokens).** New port `sparse_positions` and `sparse_count` described in `09_phase2_execution_plan.md` but deferred.
2. **Router-guided position selection.** Requires Router output as attention proxy -- currently addressed by `sparse_attn_topk.sv` (does not exist yet).
3. **Integration with layer_compute_engine.** The `window_mode` signal must eventually reach `mla_attention_v2` from `chip_top`. The wiring through `layer_compute_engine` and `full_transformer_layer` is deferred to Phase 2B integration.
4. **Batch accumulation scheduling.** The batch-accumulate FSM in `layer_compute_engine.sv` (Phase 2A.3c) is independent of sliding window and is RTL-ENG1's task.
