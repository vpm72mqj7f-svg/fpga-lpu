# DeepSeek V4 架构创新 vs FPGA 硬件设计对照分析

> 依据: `docs/X 上的 GDP：DeepSeek's 10 trillion USD grand strategy.docx`
> 对照: 当前 `rtl/` 下所有模块 + `scripts/fpga_arch/` + `docs/chip_decomposition.md`

---

## 一、已覆盖的 DS 创新 (✅ 继续深化)

| DS 创新 | 我们的 RTL 实现 | 覆盖度 | 差距 |
|----------|---------------|--------|------|
| **MoE** (top-6/384) | `router_topk.sv` + `expert_ffn_engine_fp4_down.sv` + Hot Replication | ✅ 80% | Router 目前是 4-expert 小规模原型；需要扩展到 384 并加入 load-balanced dispatch |
| **MLA** (KV 压缩 56×) | `mla_attention.sv` (简化 softmax + V 加权) | ⚠️ 30% | 现在的 attention 是预加载 score+V 的简化版，缺失 Q/K/V 的 low-rank 投影和 RoPE |
| **fp4 权重** (E2M1) | `fp4_mac.sv` (scale-aware) + `scale_reader` (group=16) | ✅ 95% | Python golden 精度验证已 PASS (cos 0.995543)；等待上板 DSP rounding 实证 |
| **Pipeline 流水线** | 32-chip forward via C2C ring | ✅ 70% | 协议已定义，c2c_node 原型已验证，缺 PCIe P2P DMA engine |

---

## 二、FPGA 独特优势但还未发挥的 DS 创新 (❌ → 🔥 高优先级)

### 2.1 DSA / CSA / HSA — 稀疏注意力 (最大空缺)

**DS 论文做了什么：**

```
DSA (V3.2): 动态稀疏注意力 — 计算量不随 context 长度增长
CSA (V4):   压缩稀疏注意力 — KV 再压缩 90%
HSA (V4):   混合稀疏注意力 — 组合多种稀疏模式

核心效果: context 从 128K → 1M tokens 时，处理时间保持平坦
```

**为什么 FPGA 应该硬做这个：**

- DSA/CSA 的稀疏 mask 是 **数据依赖** 的（根据 attention score 动态选择），GPU 上每次都要跑完整 softmax 再 mask
- FPGA 可以做 **在线稀疏**：在 DSP systolic array 的 partial sum 阶段就判断 score 是否低于阈值，低则直接跳过该 KV token 的后续累加
- 这本质上是一个 **early termination 电路**，GPU 无法高效实现（warp divergence）

**建议实现：**

```verilog
// 在 fp4_systolic_array 的 accumulate 阶段加入早停逻辑
// 伪代码:
always_ff @(posedge clk) begin
    if (partial_score < SPARSE_THRESHOLD && state == ATTENTION_SCORE) begin
        // 跳过当前 KV token 的后续累加
        skip_kv <= 1'b1;
    end
end
```

**优先级：🔥🔥🔥 最高。这是 FPGA 对 GPU 的结构性优势点。**

### 2.2 Engram — 用 LPDDR/SRAM 换 compute (FPGA 的天生优势)

**DS 论文做了什么：**

```
Engram: O(1) hash-based lookup 替代 Transformer forward pass
  - 用 classic N-gram embedding 做知识检索
  - 存储需求: 大量 embedding table (适合 LPDDR/SRAM)
  - 计算需求: 极小 (只需 hash + lookup)
  - 效果: 同参数量下, Engram 模型性能显著提升
```

**为什么 FPGA 比 GPU 更适合：**

- GPU HBM 是稀缺资源（H100 只有 80 GB），Engram 的大 embedding table 会占用 HBM
- FPGA 方案有 **32 GB HBM + 卡上 DDR5 16 GB + LPDDR5**，分层存储天然适合 embedding table 放 LPDDR、计算放 HBM
- SRAM (32 MB) 可以放热点 embedding 的 cache，做 **hash table accelerator**

**建议实现：**

```
rtl/engram/
  lookup_engine.sv    — O(1) hash-based embedding lookup
  hash_unit.sv        — 硬件 hash 函数 (MurmurHash / xxHash)
  sram_cache.sv       — 热点 embedding SRAM 缓存

内存布局:
  LPDDR5 (16 GB)     → full embedding table
  HBM (32 GB)        → expert weights + KV cache
  SRAM (32 MB)       → hotspot embedding cache (LRU)
```

**优先级：🔥🔥🔥。这是 DS 专门为 "内存多、算力少" 的硬件设计的 trade-off，FPGA 完美匹配。**

### 2.3 MTP — Multi-Token Prediction (推理加速)

**DS 论文做了什么：**

```
MTP: 一次预测多个 token（如 2-4 个）
  - 训练时: densified training signal（每个 position 预测多个 future token）
  - 推理时: speculative decoding 的基础（draft model 一次出多个候选）
```

**为什么 FPGA 可以做：**

- 当前 `chip_decomposition.md` 里只在芯片 31 标注了 `+ lm_head + MTP`，没有实际 RTL
- FPGA 的 MTP 可以硬化为 **并行 2-4 个 lm_head**，在最后一个芯片上同时输出多个 token
- 比 GPU 的 serial speculative decoding 更快（GPU 要分步生成候选 → 验证）

**建议实现：**

```
rtl/head/
  mtp_head.sv     — 2-4 parallel lm_head projections
  mtp_verify.sv   — speculative verification (check which candidates are correct)

放在芯片 31 (global id 31)，与 lm_head 共享输入 hidden state
```

**优先级：🔥🔥。推理加速直接提升吞吐，且硬件成本低（只需 2-4 个 lm_head 矩阵）。**

---

## 三、FPGA 应该做但 DS 论文没提的 (🆕 差异化)

### 3.1 mHC 硬件化 — Manifold Constrained Hyper-Connections

**DS 论文做了什么：**

```
mHC: 层间信息流重塑
  - 多条并行信息 highway，学习混合矩阵
  - Sinkhorn-Knopp 投影保证信号幅度不衰减
  - 仅 +6.7% 训练开销，+7.2 BBH 推理性能
```

**为什么 FPGA 应该硬化这个：**

- mHC 的计算量极小（只路由层间输出，不改变 FFN/Attention 内部 FLOPs）
- 但是它的混合矩阵需要在**每层**之后做**矩阵乘法 + Sinkhorn-Knopp 迭代**
- GPU 上这些操作是 kernel launch overhead 的重灾区（每层多一次 kernel launch）
- FPGA 上可以嵌入 layer pipeline 中，**零开销**

**建议实现：**

```verilog
// 在芯片的层输出路径上插入 mHC 混合模块
module mhc_mixer #(parameter int N_HIGHWAYS = 4) (
    input  logic [31:0] layer_outputs [8],  // per-highway hidden state
    output logic [31:0] mixed_outputs [8]    // mixed hidden state
);
    // 预加载的 Sinkhorn-Knopp 混合矩阵 (learned offline)
    // 在线只需一次矩阵乘法 — 纯 DSP 操作
endmodule
```

**优先级：🔥🔥。实现简单，性能收益明确。**

### 3.2 KV Cache Offload Engine — SSD → FPGA 直传

**DS 论文做了什么：**

```
Dual Path paper: KV cache 从 SSD 快速重载
  - V4 的 KV 压缩到 5.48 GB (1M context)
  - 从 SSD 重载 KV 比重新计算快 10-100×
```

**为什么 FPGA 应该做：**

- 当前设计中 KV cache 在 HBM 里（`chip_decomposition.md` 标注 ~22 GB KV 区）
- 但如果 KV 能 offload 到 SSD，HBM 全部释放给权重流式加载
- FPGA 卡连接 MCIO/PCIe，可以直接 DMA 从 Host SSD 读取 KV 到 HBM
- KV 的 5.48 GB 在 PCIe 5.0 ×16 (64 GB/s) 下只需 **<0.1 秒**

**建议实现：**

```
在 chip_top.sv 中增加 KV offload DMA engine:
  Host SSD → PCIe DMA → HBM KV region
  on session switch: load KV of new session, evict old session to SSD
  
rtl/chip/
  kv_dma_engine.sv    — KV 块级 DMA (与 Host 驱动协作)
```

**优先级：🔥。**

---

## 四、当前 RTL 覆盖度全景

```
DS V4 创新              当前 RTL 状态           FPGA 优势
─────────────────────────────────────────────────────────
MoE (top-6/384)          ✅ router + FFN          DSP 并行 expert 计算
MLA (KV 压缩 56×)        ⚠️ 简化版 attention       SRAM KV cache + 硬件寻址
DSA / CSA / HSA          ❌ 未实现 ← 🔥🔥🔥        Early termination 电路
Engram                   ❌ 未实现 ← 🔥🔥🔥        LPDDR + SRAM hash lookup
MTP                      ❌ 未实现 ← 🔥🔥          并行 lm_head + verification
mHC                      ❌ 未实现 ← 🔥🔥          嵌入 pipeline 零开销
KV Cache offload         ❌ 未实现 ← 🔥            PCIe DMA 直传
fp4 weights              ✅ scale-aware MAC        DSP 原生 fp4×fp8
Wide Expert Parallel     ⚠️ 协议已定义              C2C dispatch/reduce
Pipeline parallelism     ✅ C2C ring 验证通过      32-chip 流水线

覆盖度: 4/10 完整, 3/10 部分, 3/10 缺失
```

---

## 五、优先级排序 (建议实现顺序)

| 优先级 | 创新 | 硬件成本 | 收益 | 依赖已有什么 |
|--------|------|---------|------|-------------|
| **P0** | DSA/CSA 早停电路 | ~200 LUT + 1 DSP | context 从 128K→1M 时 compute 不增长 | `fp4_systolic_array.sv` |
| **P0** | Engram hash lookup | ~500 LUT + 2 BRAM | O(1) 替代 full transformer forward | `fp4_linear_engine.sv` |
| **P1** | mHC layer mixer | ~100 DSP (矩阵乘) | +7.2 BBH, 零额外延迟 | `full_transformer_layer.sv` |
| **P1** | MTP head | ~500 DSP (2-4 lm_head) | 推理吞吐 1.5-2× | `chip_top.sv` (chip 31) |
| **P2** | MLA full pipeline | ~2000 DSP (Q/K/V proj) | KV cache 完整压缩路径 | `mla_attention.sv` |
| **P2** | KV offload DMA | ~1000 LUT (FSM) | session 切换 <0.1s | `chip_top.sv` + `pcie_dma.svh` |

---

## 六、关键洞察

**DS 的架构选择和 FPGA 是天生一对：**

1. **fp4 权重 + MoE**: 权重小、expert 多 → HBM/算力比天然匹配 FPGA (83 vs GPU 3-5)
2. **MLA + DSA/CSA**: KV 压缩 + 稀疏注意力 → 减少 HBM 带宽需求 → 降低对高带宽 HBM 的依赖 → FPGA 的 920 GB/s 已够用
3. **Engram**: 用内存换计算 → FPGA 有 LPDDR + HBM + SRAM 三级存储, GPU 只有 HBM
4. **mHC**: 层间信号混合 → 嵌入 pipeline 零开销, GPU 需额外 kernel launch
5. **MTP**: 多 token 预测 → 硬件并行 head, 比 GPU serial speculative 快

**DS 的战略是 "降低对高端 GPU 的依赖，让中国硬件生态可行"——FPGA 是这个战略的最佳硬件载体。**

DeepSeek V4 的每一层架构创新都在减少 HBM 带宽压力、减少 FLOPs 需求、增加 memory-compute trade-off 的灵活性——而这些正是 FPGA 相对 GPU 的结构性优势。
