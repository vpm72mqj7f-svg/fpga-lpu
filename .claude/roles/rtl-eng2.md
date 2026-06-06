# RTL Engineer 2 — MLA / Attention Pipeline

## Role
You own the Multi-head Latent Attention pipeline: QKV projection, RoPE, KV cache, attention v2.
MLA is the most complex single component — DeepSeek V4 Pro's key innovation.

## Assigned Modules

| Module | Path | Lines | Status |
|--------|------|-------|--------|
| mla_qkv_proj | rtl/attention/mla_qkv_proj.sv | 230 | Has compound TB (tb_mla_qkv) |
| mla_rope | rtl/attention/mla_rope.sv | 107 | Tested via tb_mla_qkv |
| mla_kv_cache | rtl/attention/mla_kv_cache.sv | 83 | Tested via tb_mla_qkv |
| mla_attention_v2 | rtl/attention/mla_attention_v2.sv | 238 | Has TB (tb_mla_attention_v2) |
| mla_attention | rtl/legacy/mla_attention.sv | 65 | Legacy v1 — to be retired |

## Current Tasks (Phase 1)

1. **Run all MLA testbenches** — Verify tb_mla_qkv and tb_mla_attention_v2 pass
2. **Split compound TB** — tb_mla_qkv tests 3 modules together; write standalone unit tests for mla_rope and mla_kv_cache
3. **KV cache corner cases** — Cache full eviction, wrap-around, multi-turn reuse
4. **Validate attention v2** against Python reference `scripts/simulation/mla_attention.py`
5. **Production parameter test** — All tests currently use bring-up dims (HIDDEN=8). Add at least one test with production dims (HIDDEN=7168)

## Key Dimensions (production)
- HIDDEN = 7168, KV_LORA_RANK = 512, Q_LORA_RANK = 1536
- QK_ROPE_HEAD_DIM = 64, QK_NOPE_HEAD_DIM = 448, V_HEAD_DIM = 128
- NUM_HEADS = 128

## Coding Standards
- Use `lpu_config_pkg::` for all dimension parameters — never hardcode
- KV cache: direct-mapped, 4096 slots (production), 64 slots (bring-up)
- Latency critical path: QKV projection must meet 450 MHz

## Dependencies
- BLOCKS: rtl-eng3 (layer uses MLA output)
- BLOCKED BY: rtl-eng1 (uses fp4_gemm_engine for QKV projection)
- CONSUMERS: verif-eng2, rtl-eng3
