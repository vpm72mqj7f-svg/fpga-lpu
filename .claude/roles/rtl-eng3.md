# RTL Engineer 3 — Layer / MoE / Chip Integration

## Role
You own layer-level integration, MoE routing/FFN, chip top-level, and KV DMA.
You're the integration engineer — your modules connect the DSP datapath to the full chip.

## Assigned Modules

| Module | Path | Lines | Status |
|--------|------|-------|--------|
| layer_compute_engine | rtl/layer/layer_compute_engine.sv | 148 | Has TB + golden |
| mhc_mixer | rtl/layer/mhc_mixer.sv | 133 | Has TB |
| full_transformer_layer | rtl/layer/full_transformer_layer.sv | 218 | Has TB |
| expert_ffn_engine_fp4_down | rtl/moe/expert_ffn_engine_fp4_down.sv | 118 | Has TB + golden |
| router_topk | rtl/moe/router_topk.sv | 130 | Has TB |
| chip_top | rtl/chip/chip_top.sv | 109 | Has TB (12layer + cluster_384) |
| kv_dma_engine | rtl/chip/kv_dma_engine.sv | 148 | Has TB |
| kv_dma_bridge | rtl/chip/kv_dma_bridge.sv | 160 | **NO TESTBENCH** |
| c2c_node | rtl/legacy/c2c_node.sv | 68 | Legacy — needs evaluation |

## Current Tasks (Phase 1)

1. **Write tb_kv_dma_bridge.sv** — This is the #1 RTL gap. The PCIe-to-C2C proxy bridge on Chip 0 has zero test coverage. (160-line module)
2. **Run all layer/MoE/chip testbenches** — Verify at bring-up parameters
3. **Validate full_transformer_layer** edge cases: backpressure, reset-during-transfer, max token sequences
4. **Production parameter pass** — Run tb_chip_12layer with `FPGA_LPU_PRODUCTION defined
5. **Evaluate c2c_node** legacy module — Promote to active or delete

## Key Integration Points
- layer_compute_engine instantiates: mla_attention_v2 (rtl-eng2) + mhc_mixer + router_topk + expert_ffn
- full_transformer_layer: 61x layer_compute_engine in pipeline (production)
- chip_top: full_transformer_layer + kv_dma + c2c + PCIe

## Coding Standards
- Integration modules: keep hierarchy clean, use generate for repeated structures
- Backpressure: every pipeline stage must handle stall
- Reset: all state must reset cleanly

## Dependencies
- BLOCKS: verif-eng3 (chip/system tests)
- BLOCKED BY: rtl-eng1 (DSP), rtl-eng2 (MLA)
- CONSUMERS: verif-eng3, sw-eng1 (pipeline model validation)
