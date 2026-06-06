# Software Engineer 1 — FPGA Architecture Model & Pipeline Simulation

## Role
You own the hardware architecture simulation: `fpga_arch/` package, pipeline timing models,
cluster topology, chip resource tracking, and the `run_serving.py` entry point.
Your models are the software twin of the RTL — they must match cycle-accurately.

## Assigned Files

| File | Lines | Purpose |
|------|-------|---------|
| scripts/fpga_arch/config.py | 255 | Hardware constants (single source of truth) |
| scripts/fpga_arch/pipeline.py | 1,231 | 10-stage pipeline timing, dual-path prefill/decode |
| scripts/fpga_arch/cluster.py | 268 | 32-chip cluster assembly, layer/expert placement |
| scripts/fpga_arch/chip.py | 192 | Chip resource tracking, KV block management |
| scripts/fpga_arch/interconnect.py | 104 | C2C ring + PCIe fabric |
| scripts/fpga_arch/expert_popularity.py | 110 | Zipf MoE expert popularity |
| scripts/run_serving.py | 1,251 | Main entry point, event-driven simulation |

## Current Tasks (Phase 1)

1. **Validate pipeline model against RTL** — Run `scripts/simulation/experiment_3_layer_latency.py` and cross-check the 10.5us/layer result against actual RTL testbench cycle counts
2. **Fix microbatch session release** — `run_serving.py:829` has `if False and is_decode:` — restore or remove this dead code path
3. **Integrate prefill/ subdirectory** — `prefill/coordinator.py`, `prefill/scheduler.py`, `prefill/vllm_prefill.py` are standalone. Wire them into the main `run_serving.py` event loop
4. **Add MIXED batch support** — `vllm_serve/types.py:25` has `MIXED` enum value but it's not implemented. Add prefill+decode co-batching to the scheduler
5. **Consolidate dual architecture stacks** — `fpga_arch/` (main) and `architecture/` (legacy) have diverged. Merge useful code from legacy (especially `stages/interfaces.py` RTL bus definitions) and deprecate the rest
6. **Add ctypes binding for cpu_prefill.c** — The C file exists at `c_ref/prefill/cpu_prefill.c` but Python can't call it. Write the ctypes bridge

## Key Formulas
- Pipeline TPS: `TPS = 17445 * B / (B + 25.4)`
- Weighted layer latency: 10.5 us/layer (DSP=39.4% utilization)
- Decode TPS (B=1): ~724 tok/s; (B=8): ~4,490 tok/s

## Dependencies
- NEEDS FROM: verif-eng3 (chip-level RTL timing for model calibration)
- PROVIDES TO: sw-eng2 (pipeline model for scheduler), sw-eng3 (pipeline model for experiments)
- COORDINATES WITH: rtl-eng3 (chip_top timing)
