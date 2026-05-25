# FPGA LPU Simulation Validation Summary

Generated: 2026-05-23T09:58:12
Passed: 4/4

| Suite | Exit | Log |
|---|---:|---|
| module_smoke | 0 | `docs\sim_module_smoke.log` |
| functional_suite | 0 | `docs\sim_functional_suite.log` |
| architecture_integration | 0 | `docs\sim_architecture_integration.log` |
| serving_agent_short | 0 | `docs\sim_serving_agent_short.log` |

## Notes
- `module_smoke`: direct unit smoke for fpga_arch/vllm_serve modules including WLC.
- `functional_suite`: NumPy fp4/HBM/layer experiments.
- `architecture_integration`: legacy layered architecture demo.
- `serving_agent_short`: end-to-end serving simulation with D+C+A+Pipeline Cloning.

## fp4 Precision Result
- Exp 1 now PASS after switching fp4 quantization group size from 128 to 16 and fixing QAT smoothing math.
- Production-scale result: `group_size=16, alpha=1.0, fallback=0%` gives mean cosine `0.995543`, min cosine `0.995335`, relative error `0.0945`.
- No fp8 fallback is required for the simulated Expert FFN; group-wise scale overhead increases, but it is metadata-only and still far smaller than fp8 weights.
