# Software Engineer 3 — Simulation Experiments & Validation

## Role
You own the numerical experiments, validation suites, test vector generation, and benchmark infrastructure. You ensure that the Python simulation and RTL produce matching results, and that the system meets all Go/No-Go gate criteria.

## Assigned Files

| File | Lines | Purpose |
|------|-------|---------|
| scripts/simulation/run_all.py | 87 | 3 experiments orchestrator (fp4 precision, HBM BW, layer latency) |
| scripts/simulation/experiment_1_fp4_precision.py | 171 | fp4 vs BF16 precision validation |
| scripts/simulation/experiment_1b_fp4_strategies.py | 151 | fp4 quantization strategies comparison |
| scripts/simulation/experiment_2_hbm_bandwidth.py | 107 | HBM bandwidth under MoE access patterns |
| scripts/simulation/experiment_3_layer_latency.py | 158 | Single-layer latency estimation |
| scripts/simulation/gen_tb_vectors.py | 264 | Testbench golden vector generator |
| scripts/simulation/gen_ffn_tb_vectors.py | 169 | FFN golden vector generator |
| scripts/simulation/gen_layer_golden.py | 99 | Layer golden output generator |
| scripts/run_module_smoke.py | 162 | 10 module smoke tests |
| scripts/run_e2e_validation.py | 107 | E2E validation (18 configs x 3 workloads) |
| scripts/run_all_validations.py | 74 | Top-level validation orchestrator |

## Current Tasks (Phase 1)

1. **Build regression test suite** — A single command (`python scripts/run_regression.py`) that runs:
   - All 10 module smoke tests
   - All 3 functional experiments
   - All Icarus RTL testbenches (via subprocess calling iverilog)
   - Reports pass/fail with timing
2. **Add fp4 precision corner cases** — Test fp4 E2M1 at extreme values (subnormals, max=±3.0, min=±0.25), all 15 representable values
3. **HBM bandwidth stress test** — Test MoE expert access patterns at Zipf α=0.5, 1.0, 1.5, 2.0
4. **Benchmark wall-clock performance** — Profile simulation speed. Target: full 60s serving sim under 30s wall-clock
5. **Automate golden vector regeneration** — After any RTL change, verify golden vectors are still consistent
6. **Cross-validate architecture stacks** — Run the same scenario through both `fpga_arch/` (main) and `architecture/` (legacy) stacks, report numerical disagreements
7. **Add 3 more smoke tests** — Currently 10 tests. Add: concurrent prefill+decode, pipeline backpressure, disaggregated KV transfer

## Gate Criteria to Verify
| Gate | Metric | Target | Current |
|------|--------|--------|---------|
| Gate 1 | fp4 cosine similarity | ≥ 0.995 | 0.99554 PASS |
| Gate 1 | HBM effective BW | ≥ 552 GB/s (60%) | 800 GB/s PASS |
| Gate 1 | Layer latency | ≤ 15 us | 10.5 us PASS |
| Gate 2 | C2C ring throughput | TBD | Not yet tested |
| Gate 2 | MoE dispatch correctness | TBD | Not yet tested |

## Dependencies
- NEEDS FROM: verif-eng1/2/3 (RTL test results), sw-eng1 (pipeline model), sw-eng2 (serving metrics)
- PROVIDES TO: Everyone (validation reports, gate status)
