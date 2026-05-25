# DeepSeek V4 FPGA 独有优势实现计划

> 基于 `docs/ds_v4_arch_gap_analysis.md` 的 6 个 gap 项
> 优先级: P0 (must-have, 上板前实现) → P1 (should-have, Phase 2) → P2 (nice-to-have, Phase 3)
> 总估算: 4.5 人月

---

## P0-1: DSA/CSA 稀疏注意力早停电路

**目标**: attention score 计算中，当 partial dot-product 低于动态阈值时，跳过该 KV token 的剩余累加。

**现有基础**: `fp4_systolic_array.sv` 已支持 K 维流式累加。插入早停逻辑即可。

**新增 RTL**:

```
rtl/attention/sparse_attn_ctl.sv  (~120 行)
  - 参数: SPARSE_THRESHOLD (Q12, 可从寄存器配置)
  - 输入: s2_product[31:0], beat_valid
  - 输出: skip_kv (停止当前 KV token 的后续累加)
  - 接口: 接入 fp4_systolic_array 的 accumulate 阶段

逻辑:
  always_ff @(posedge clk) begin
      if (accum_valid && partial_score < threshold && beat_count > MIN_BEATS) begin
          skip_current_kv <= 1'b1;  // 跳过该 KV token 的剩余 beat
      end
  end
```

**依赖**: `fp4_systolic_array.sv` (已开发并通过 Icarus 验证)

**集成点**: 在 `mla_attention.sv` 的 score computation 阶段，每个 KV token 的 dot-product 开始前复位 `skip_current_kv`。

**测试计划**:

```sv
// tb_sparse_attn_ctl.sv
// Case 1: 所有 score 远高于 threshold → 不跳过任何 KV
// Case 2: 前 2 beat 后 partial < threshold → 跳过后续 beat, score 保持 partial 值
// Case 3: threshold=0 → 与 baseline attention 完全一致
```

**人月估算**: **0.5 人月**（RTL 0.3 + testbench 0.1 + Icarus 验证 0.1）

---

## P0-2: Engram Hash Lookup Engine

**目标**: O(1) hash-based embedding lookup 替代部分 Transformer forward pass。

**背景**: DS 的 Engram 把 classic N-gram embedding 变成 O(1) hash lookup，用 LPDDR 存储完整 embedding table，用 SRAM 缓存热点 entry。GPU 无法高效利用 LPDDR，FPGA 可以。

**新增 RTL**:

```
rtl/engram/
  hash_unit.sv          (~80 行)   — MurmurHash3 硬件实现 (4-cycle pipeline)
  lookup_engine.sv      (~150 行)  — hash → SRAM cache → LPDDR miss handler
  sram_cache.sv         (~100 行)  — 热点 embedding SRAM 缓存 (LRU, 512 entries)

接口:
  module engram_lookup #(N_GRAMS=4, EMBED_DIM=64) (
      input  logic [31:0] token_ids [N_GRAMS],
      output logic [31:0] embedding  [EMBED_DIM],
      // LPDDR interface (for cache miss)
      output logic        lpddr_rd_req,
      input  logic [31:0] lpddr_rd_data,
      input  logic        lpddr_rd_valid
  );
```

**依赖**: 无——独立模块。

**集成点**: 在 `full_transformer_layer.sv` 的 RMSNorm1 之前或之后插入 engram 输出，作为 hidden state 的 additive component。

**测试计划**:

```sv
// tb_engram_lookup.sv
// Case 1: 4-gram hash → SRAM hit → 1-cycle output
// Case 2: SRAM miss → LPDDR read → 3-cycle output (pipeline stall)
// Case 3: 随机 1000 tokens → 统计 hit rate (应 >80% with LRU)
```

**人月估算**: **1.0 人月**（RTL 0.5 + testbench 0.2 + Icarus 0.2 + LPDDR 接口仿真 0.1）

---

## P1-1: mHC Layer Mixer

**目标**: 在层间插入 manifold-constrained hyper-connection 混合矩阵，嵌入 pipeline 零额外延迟。

**背景**: DS 的 mHC 每层只需一次小型矩阵乘法 + 逐元素乘加，计算量极小。GPU kernel launch 开销反而占主导。FPGA 可以直接嵌入流水线。

**新增 RTL**:

```
rtl/layer/
  mhc_mixer.sv  (~180 行)

参数: N_HIGHWAYS = 4, HIDDEN_DIM = 8
接口:
  module mhc_mixer #(HIDDEN=8, N_HW=4) (
      input  logic signed [31:0] layer_in   [HIDDEN],  // attention/FFN output
      input  logic signed [31:0] residual   [HIDDEN],  // residual stream
      output logic signed [31:0] highway_out[HIDDEN * N_HW]  // expanded highways
  );
内部:
  - 预加载的 learned mixing matrix (N_HW × 2, Q12)
  - 2 个 DSP 周期完成: matrix multiply + highway combine
  - 无状态，纯组合逻辑 + 1 register stage
```

**依赖**: 无——独立模块。

**集成点**: 在 `full_transformer_layer.sv` 的每个 sub-layer 输出后插入 mhc_mixer:

```
  Attention output → mhc_mixer(attn_out, residual_in) → highway expanded
  FFN output       → mhc_mixer(ffn_out, highway_in)    → mixed output
```

**测试计划**:

```sv
// tb_mhc_mixer.sv
// Case 1: identity mixing matrix → output matches residual (passthrough test)
// Case 2: learned matrix → output matches Python golden (gen from numpy)
// Case 3: Sinkhorn-Knopp projection → output has unit row/col sums
```

**人月估算**: **0.5 人月**（RTL 0.2 + testbench 0.15 + Icarus 0.1 + golden gen 0.05）

---

## P1-2: MTP Head (Multi-Token Prediction)

**目标**: 在最后一个芯片（chip 31）并行 2-4 个 lm_head projection，一次预测多个 token。

**背景**: DS 的 MTP 在训练时每 position 预测多个 future token。推理时用 speculative decoding：draft model 一次出 2-4 个候选 → target model 一次验证全部，加速 1.5-2×。

**新增 RTL**:

```
rtl/head/
  mtp_head.sv     (~200 行)   — 并行 2-4 个 lm_head projection
  mtp_verify.sv   (~100 行)   — speculative verification logic

接口:
  module mtp_head #(N_HEADS=2, VOCAB=129280) (
      input  logic signed [31:0] hidden_state [8],  // from last layer
      output logic [17:0] token_ids [N_HEADS],       // up to log2(VOCAB)
      output logic [31:0] logprobs  [N_HEADS],
      output logic         valid
  );
```

**依赖**: `fp4_linear_engine.sv` (reuse for projection)

**集成点**: 放在 `chip_top.sv` 的 chip 31 上，与 lm_head 并行。

**测试计划**:

```sv
// tb_mtp_head.sv
// Case 1: identity projection → all tokens equal → all 2 heads agree
// Case 2: known weights → verify head 0 != head 1 (speculative diversity)
```

**人月估算**: **0.8 人月**（RTL 0.4 + testbench 0.2 + Icarus 0.1 + golden gen 0.1）

---

## P2-1: MLA Full Pipeline (Q/K/V Low-Rank Projection)

**目标**: 把当前简化的 `mla_attention.sv` 升级为完整的 MLA，包含 Q/K/V low-rank projection + RoPE。

**新增 RTL**:

```
rtl/attention/
  mla_qkv_proj.sv    (~250 行)  — Q latent ↓, K latent ↓, V ↑ projection
  mla_rope.sv        (~100 行)  — decoupled RoPE (sin/cos LUT)
  mla_kv_cache.sv    (~200 行)  — hardware KV cache write/read (替换软件 allocator)

升级 mla_attention.sv: 集成以上三个子模块
```

**依赖**: `fp4_linear_engine.sv`, `fp4_scale_reader.sv`

**人月估算**: **1.2 人月**（RTL 0.6 + testbench 0.3 + Icarus 0.2 + golden gen 0.1）

---

## P2-2: KV Cache Offload DMA Engine

**目标**: Host SSD → FPGA HBM 的 KV 块级 DMA，支持 session 间快速切换。

**新增 RTL**:

```
rtl/chip/
  kv_dma_engine.sv   (~200 行)  — KV 块级 DMA 描述符引擎

接口:
  module kv_dma_engine (
      input  pcie_dma_stream_t  host_dma,
      output logic [31:0]       hbm_addr,
      output logic [31:0]       hbm_wr_data,
      output logic              hbm_wr_en,
      output logic              done,
      output logic [15:0]       session_id
  );
```

**依赖**: `pcie_dma.svh` (接口定义已完成), HBM controller (Intel IP, QSYS generate)

**人月估算**: **0.5 人月**（RTL 0.3 + integration 0.1 + test 0.1）

---

## 总估算

| 优先级 | 模块 | 人月 | 文件数 | 可并行 |
|--------|------|------|--------|--------|
| **P0** | DSA/CSA 早停 | 0.5 | 1 SV + 1 TB | 与 P0-2 并行 |
| **P0** | Engram lookup | 1.0 | 3 SV + 1 TB | 与 P0-1 并行 |
| **P1** | mHC mixer | 0.5 | 1 SV + 1 TB | 与 P1-2 并行 |
| **P1** | MTP head | 0.8 | 2 SV + 1 TB | 与 P1-1 并行 |
| **P2** | MLA full pipeline | 1.2 | 3 SV + 1 TB | - |
| **P2** | KV offload DMA | 0.5 | 1 SV | 与 P2-1 并行 |
| **总计** | | **4.5 人月** | **16 files** | 3 阶段 |

## 优先级逻辑

```
P0 (必须在上板前实现):
  — DSA/CSA 早停: 不增加新 DSP, 只在现有 array 加控制逻辑
  — Engram: DS 专门为"内存多算力少"设计, FPGA 是天选之子
  → 这两个是 FPGA vs GPU 的结构性差异点, BP 的核心论据

P1 (Phase 2 实现):
  — mHC: 嵌入 pipeline 零开销, GPU 做不到
  — MTP: 推理吞吐 1.5-2×, 硬件代价小
  → 这两个是 FPGA 的性能放大器

P2 (Phase 3 实现):
  — MLA full: 完整 attention, 代码量大但无架构风险
  — KV offload: 需要 Host driver 配合, 偏软件
```

## 与现有 RTL 的对接

```text
full_transformer_layer.sv
  ├── [P0-1] sparse_attn_ctl ← 插入到 mla_attention.sv 的 score 阶段
  ├── [P0-2] engram_lookup   ← 插入到 RMSNorm1 之后, Attention 之前
  ├── [P1-1] mhc_mixer       ← 插入到每个 sub-layer 输出后
  ├── [P1-2] mtp_head        ← 插入到 chip_top.sv 的 chip 31
  ├── [P2-1] mla_qkv_proj etc← 替换当前 mla_attention.sv 的内部实现
  └── [P2-2] kv_dma_engine   ← 插入到 chip_top.sv 的 PCIe 路径
```
