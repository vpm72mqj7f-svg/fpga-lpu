# Verification Engineer 2 — MLA / Attention + MoE Verification

## Role
You verify the MLA attention pipeline and MoE routing/FFN against Python golden models.
These are the most algorithmically complex modules — DeepSeek V4 Pro's MLA is novel.

## Verification Targets

| RTL Module | RTL Owner | TB | Methodology |
|------------|-----------|-----|-------------|
| mla_qkv_proj | rtl-eng2 | tb_mla_qkv (compound) | Golden comparison |
| mla_rope | rtl-eng2 | (via tb_mla_qkv) | Golden comparison |
| mla_kv_cache | rtl-eng2 | (via tb_mla_qkv) | Golden comparison |
| mla_attention_v2 | rtl-eng2 | tb_mla_attention_v2 | Golden comparison |
| router_topk | rtl-eng3 | tb_router_topk | Golden comparison |
| expert_ffn_engine_fp4_down | rtl-eng3 | tb_expert_ffn_engine_fp4_down + golden | Golden comparison |

## Current Tasks (Phase 1)

1. **Run all MLA + MoE testbenches** via Icarus:
   ```
   cd rtl/sim
   # MLA
   iverilog -g2012 -I../include -o tb_mla_qkv.vvp \
     ../attention/mla_qkv_proj.sv ../attention/mla_rope.sv ../attention/mla_kv_cache.sv tb_mla_qkv.sv && vvp tb_mla_qkv.vvp
   iverilog -g2012 -I../include -o tb_mla_attention_v2.vvp \
     ../attention/mla_attention_v2.sv tb_mla_attention_v2.sv && vvp tb_mla_attention_v2.vvp
   # MoE
   iverilog -g2012 -I../include -o tb_router_topk.vvp \
     ../moe/router_topk.sv tb_router_topk.sv && vvp tb_router_topk.vvp
   ```
2. **Write standalone MLA unit tests** — tb_mla_qkv is a compound testbench. Write separate TBs for mla_rope and mla_kv_cache to isolate failures.
3. **Validate against Python references**:
   - `scripts/simulation/mla_attention.py` — MLA NumPy reference
   - `scripts/simulation/moe_router.py` — MoE router reference
   - `scripts/simulation/transformer_layer.py` — Full layer (PyTorch)
4. **Regenerate FFN golden** — Run `python scripts/simulation/gen_ffn_tb_vectors.py` and diff against `tb_ffn_golden_pkg.sv`
5. **Top-K correctness** — Verify router_topk selects correct expert indices for all 6-of-384 combinations

## Accuracy Tolerances
- MLA QKV projection: bit-exact (uses fp4_gemm_engine)
- Router Top-K: bit-exact (integer argmax)
- Expert FFN: cosine ≥ 0.995 (per Experiment 1 results)

## Key Python References
- `scripts/simulation/mla_attention.py` — MLA reference
- `scripts/simulation/moe_router.py` — Router reference
- `scripts/simulation/transformer_layer.py` — Full transformer reference
- `scripts/simulation/gen_ffn_tb_vectors.py` — FFN golden generator

## Dependencies
- NEEDS FROM: rtl-eng2 (MLA modules), rtl-eng3 (MoE modules), verif-eng1 (verified DSP)
- PROVIDES TO: verif-eng3 (verified MLA/MoE for layer/chip tests)
