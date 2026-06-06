# RTL Engineer 4 — Activation / Head / Engram + Coverage Gaps

## Role
You own activation functions, multi-token prediction head, N-gram engram cache,
and the 6 modules currently missing testbenches. You're the cleanup + coverage engineer.

## Assigned Modules

| Module | Path | Lines | Status |
|--------|------|-------|--------|
| rms_norm | rtl/activation/rms_norm.sv | 192 | Has TB |
| silu_q12_lut | rtl/activation/silu_q12_lut.sv | 58 | Has TB |
| q12_to_fp8_e4m3 | rtl/activation/q12_to_fp8_e4m3.sv | 31 | **NO TESTBENCH** |
| mtp_head | rtl/head/mtp_head.sv | 133 | Has TB |
| mtp_verify | rtl/head/mtp_verify.sv | 55 | **NO TESTBENCH** |
| lookup_engine | rtl/engram/lookup_engine.sv | 148 | Has TB |
| hash_unit | rtl/engram/hash_unit.sv | 65 | **NO TESTBENCH** |
| sram_cache | rtl/engram/sram_cache.sv | 59 | **NO TESTBENCH** |
| uart_debug | rtl/debug/uart_debug.sv | 84 | Low priority — HW debug tool |
| dsp_stress_test | rtl/debug/dsp_stress_test.sv | 232 | Self-test harness (Go/No-Go #2) |
| hbm_bw_test | rtl/debug/hbm_bw_test.sv | 290 | Self-test harness (Go/No-Go #1) |

## Current Tasks (Phase 1)

1. **Write 4 testbenches** (in priority order):
   - `tb_mtp_verify.sv` — Speculative decode verification (HIGH, 55-line module)
   - `tb_hash_unit.sv` — 4-cycle pipelined hash for N-gram (MEDIUM)
   - `tb_sram_cache.sv` — Direct-mapped embedding cache (MEDIUM)
   - `tb_q12_to_fp8_e4m3.sv` — Trivial combinational encoder (LOW, ~15 lines)

2. **Run existing TBs** — tb_rms_norm, tb_silu_q12_lut, tb_mtp_head, tb_lookup_engine
3. **Validate mtp_head + mtp_verify** — Speculative decode acceptance/rejection logic
4. **Review debug harnesses** — Are dsp_stress_test and hbm_bw_test ready for on-board use?
5. **Production parameter test** — rms_norm at HIDDEN=7168

## Reference Files
- `scripts/simulation/gen_tb_vectors.py` — Golden vector generator
- `scripts/simulation/gen_ffn_tb_vectors.py` — FFN golden vectors
- `scripts/simulation/gen_layer_golden.py` — Layer golden outputs

## Coding Standards
- Activation functions: Q12 fixed-point internal, fp8 output
- RMS Norm: ±4 tolerance vs golden (not bit-exact)
- SiLU: ±1 LSB tolerance (LUT-based)

## Dependencies
- BLOCKS: nobody (leaf modules)
- BLOCKED BY: nobody
- CONSUMERS: verif-eng1 (activation), verif-eng2 (head/engram)
