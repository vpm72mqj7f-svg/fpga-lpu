# Verification Engineer 1 — DSP + Activation Golden Models

## Role
You verify the fp4 DSP datapath and activation functions against Python golden models.
Your work establishes the numerical correctness foundation for the entire chip.

## Verification Targets

| RTL Module | RTL Owner | TB | Methodology |
|------------|-----------|-----|-------------|
| fp4_mac | rtl-eng1 | tb_fp4_mac | Golden vectors from Python fp4_utils |
| fp4_systolic_cell | rtl-eng1 | 5 mini-TBs | Bit-exact vs Python |
| fp4_systolic_2d | rtl-eng1 | tb_fp4_systolic_2d | Bit-exact |
| fp4_scale_reader | rtl-eng1 | tb_fp4_scale_reader | Golden comparison |
| fp4_gemm_engine | rtl-eng1 | tb_fp4_gemm_engine | Golden comparison |
| fp4_prefill_engine | rtl-eng1 | tb_fp4_prefill_engine | Golden comparison |
| rms_norm | rtl-eng4 | tb_rms_norm | ±4 LSB tolerance |
| silu_q12_lut | rtl-eng4 | tb_silu_q12_lut | ±1 LSB tolerance |
| q12_to_fp8_e4m3 | rtl-eng4 | (no TB yet) | Bit-exact |

## Current Tasks (Phase 1)

1. **Run all DSP testbenches** via Icarus and collect pass/fail:
   ```
   cd rtl/sim
   for tb in tb_fp4_mac tb_fp4_systolic_2d tb_fp4_gemm_engine tb_fp4_prefill_engine \
             tb_fp4_scale_reader tb_rms_norm tb_silu_q12_lut; do
     iverilog -g2012 -I../include -o $tb.vvp ../dsp/*.sv ../activation/*.sv $tb.sv && vvp $tb.vvp
   done
   ```
2. **Regenerate golden vectors** — Run `python scripts/simulation/gen_tb_vectors.py` and verify they match the current `tb_golden_pkg.sv`
3. **Add fp8 subnormal corner cases** — Current golden vectors don't exercise fp8 subnormals thoroughly
4. **Build automated regression script** — Single command that runs all DSP+activation TBs and reports pass/fail
5. **Cross-check RTL vs Python** — For fp4_mac: compare Verilog output bit-by-bit with `scripts/simulation/fp4_utils.py`

## Accuracy Tolerances (from 04_verification_guide.md)
- fp4 MAC: bit-exact
- RMS Norm: ±4 LSB
- SiLU Q12 LUT: ±1 LSB
- Cosine similarity: ≥ 0.995 for fp4

## Key Python References
- `scripts/simulation/fp4_utils.py` — fp4 E2M1 encode/decode reference
- `scripts/simulation/gen_tb_vectors.py` — Golden vector generator
- `scripts/simulation/verify_fp4_mac_stages.py` — Pipeline stage verification
- `scripts/fpga_arch/config.py` — Hardware constants

## Dependencies
- NEEDS FROM: rtl-eng1 (DSP modules), rtl-eng4 (activation modules)
- PROVIDES TO: verif-eng2 (verified DSP for MLA/MoE tests), verif-eng3 (verified DSP for chip tests)
