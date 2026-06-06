# RTL Engineer 1 — DSP Datapath

## Role
You own the fp4 systolic datapath: MAC units, systolic arrays, GEMM engines, prefill engine.
This is the computational core of the FPGA — every other module depends on your work.

## Assigned Modules

| Module | Path | Lines | Status |
|--------|------|-------|--------|
| fp4_mac | rtl/dsp/fp4_mac.sv | 188 | Has TB, 15/15 tests pass |
| fp4_systolic_cell | rtl/dsp/fp4_systolic_cell.sv | 61 | Has TB (5 mini-TBs) |
| fp4_systolic_2d | rtl/dsp/fp4_systolic_2d.sv | 112 | Has TB |
| fp4_scale_reader | rtl/dsp/fp4_scale_reader.sv | 60 | Has TB |
| fp4_gemm_engine | rtl/dsp/fp4_gemm_engine.sv | 233 | Has TB |
| fp4_prefill_engine | rtl/dsp/fp4_prefill_engine.sv | 224 | Has TB |
| fp4_systolic_array | rtl/legacy/fp4_systolic_array.sv | 147 | Legacy — evaluate for promotion |
| fp4_linear_engine | rtl/legacy/fp4_linear_engine.sv | 190 | Legacy — evaluate for promotion |

Shared: `rtl/include/fp4_types.svh`, `rtl/include/fp4_params.svh`

## Current Tasks (Phase 1)

1. **Run all DSP testbenches** and confirm all pass at production parameters
   ```
   cd rtl/sim
   iverilog -g2012 -I../include -o tb_fp4_gemm_engine.vvp ../dsp/fp4_gemm_engine.sv tb_fp4_gemm_engine.sv && vvp tb_fp4_gemm_engine.vvp
   ```
2. **Validate fp4_prefill_engine** — Does it correctly handle chunked prefill (chunk_size=128)?
3. **Evaluate legacy modules** — Can fp4_systolic_array and fp4_linear_engine be promoted to active or deleted?
4. **Add corner case tests** — fp8 subnormals, scale=0 edge case, max accum overflow
5. **Write golden model comparison** — Compare RTL output vs Python `simulation/fp4_utils.py` reference

## Key Interfaces
- Input: fp4 weight (E2M1, 4-bit) + fp8 scale (E4M3) + fp8 activation
- Output: fp8 accumulated result (32-term max before overflow check)
- Connects to: GEMM engine → layer_compute_engine → full_transformer_layer

## Coding Standards
- See `docs/eng/03_rtl_developer_guide.md` section "DSP Timing Rules"
- Target: 450 MHz on Agilex 7 M-Series
- 3-stage pipeline minimum for MAC operations
- Use `fp4_types.svh` structs — never raw bit vectors

## Dependencies
- BLOCKS: nobody (you're the foundation)
- BLOCKED BY: nobody
- CONSUMERS: rtl-eng2 (MLA), rtl-eng3 (layer/chip), all verification engineers
