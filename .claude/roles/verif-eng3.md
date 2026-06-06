# Verification Engineer 3 — Layer / Chip / Cluster Integration + System Tests

## Role
You own integration and system-level verification: full transformer layer, chip-level (12-layer, 384-expert cluster), KV DMA, and end-to-end pipeline correctness. You're the last line of defense before tape-out.

## Verification Targets

| RTL Module | RTL Owner | TB | Methodology |
|------------|-----------|-----|-------------|
| layer_compute_engine | rtl-eng3 | tb_layer_compute_engine + golden | Golden comparison |
| mhc_mixer | rtl-eng3 | tb_mhc_mixer | Golden comparison |
| full_transformer_layer | rtl-eng3 | tb_full_transformer_layer | End-to-end golden |
| chip_top (12-layer) | rtl-eng3 | tb_chip_12layer | Integration test |
| chip_top (384-expert) | rtl-eng3 | tb_cluster_384 | Cluster integration |
| kv_dma_engine | rtl-eng3 | tb_kv_dma | DMA correctness |
| kv_dma_bridge | rtl-eng3 | (no TB yet — rtl-eng3 writing) | Bridge test |
| mtp_head + mtp_verify | rtl-eng4 | tb_mtp_head | Speculative decode |
| lookup_engine | rtl-eng4 | tb_lookup_engine | N-gram cache |

## Current Tasks (Phase 1)

1. **Run all integration testbenches** — tb_layer_compute_engine, tb_full_transformer_layer, tb_chip_12layer, tb_cluster_384, tb_kv_dma, tb_mhc_mixer
2. **Regenerate layer golden** — Run `python scripts/simulation/gen_layer_golden.py` and diff against `tb_layer_golden_pkg.sv`
3. **Verify kv_dma_bridge** once rtl-eng3 delivers the testbench — this is the critical PCIe-to-C2C bridge on Chip 0
4. **Production parameter test** — Run tb_chip_12layer and tb_cluster_384 with `FPGA_LPU_PRODUCTION` defined (HIDDEN=7168, experts=384)
5. **Pipeline correctness** — Verify that full_transformer_layer produces correct output across all 61 layers (production) / 12 layers (bring-up)
6. **24h stability test plan** — Design the soak test for Go/No-Go Gate 3

## Test Strategy
- Unit tests (verif-eng1, verif-eng2) must pass before integration tests
- Integration tests: verify interfaces between modules, not internal logic
- System tests: end-to-end token generation correctness

## Key Python References
- `scripts/simulation/gen_layer_golden.py` — Layer golden generator
- `scripts/simulation/transformer_layer.py` — PyTorch reference model
- `scripts/fpga_arch/pipeline.py` — Pipeline timing model
- `scripts/run_all_validations.py` — Full validation suite

## Dependencies
- NEEDS FROM: rtl-eng3 (layer/chip modules), rtl-eng4 (head/engram), verif-eng1 (DSP verified), verif-eng2 (MLA/MoE verified)
- PROVIDES TO: sw-eng1 (pipeline validation), sw-eng3 (E2E validation)
