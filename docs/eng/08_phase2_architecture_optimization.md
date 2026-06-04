# Phase 2 架构优化分析：基于 HBM 瓶颈 + token/kWh 新认知

> **背景**: Phase 1 完成了 36 个 RTL 模块的功能验证和参数化修复。
> 但从 `HBM 容量是物理瓶颈` + `Prefill/Decode 物理分离` + `token/kWh 归一化`
> 这三个新认知出发重新审视，发现 Phase 1 的优化重点有偏差。

---

## 1. Phase 1 做了什么 vs 应该做什么

| Phase 1 做了什么 | 为什么做 | 新认知下的评价 |
|------|------|------|
| 验证 36 个 RTL 模块功能正确 | 基本功 | ✅ 必要但不够 |
| fp4_mac 精度验证 (Q12 vs float32) | 担心量化误差 | ⚠️ 过度关注。Decode 受限于 KV 读取而非 MAC 精度 |
| 参数化修复 (C1-C4) | 支持生产 HIDDEN=7168 | ✅ 必要 |
| Altera IP 替换 (同步RAM/DSP/FIFO) | 综合约束 | ✅ 必要 |
| HBM 带宽基准测试 (91.6% efficiency) | 验证内存墙 | ✅ 正确方向 |
| 24h 稳定性测试计划 | 可靠性 | ⚠️ 太早。先解决架构瓶颈再跑稳定性 |

**核心偏差**: Phase 1 围绕 "DSP 算力是否够、精度是否对" 展开，但新认知表明：
- DSP 算力 **严重过剩**（116× @ 1M context）
- HBM **容量**才是真正的瓶颈
- CPU Prefill TTFT 是用户体验的致命短板

---

## 2. 新认知框架下的架构差距

### 2.1 瓶颈转移：从算力到容量

```
Phase 1 假设的瓶颈:        实际瓶颈:
  DSP利用率                   HBM KV容量
  MAC精度                     TTFT (CPU Prefill)
  参数化正确性                KV offload 带宽
  同步逻辑安全性              滑动窗口实现
```

### 2.2 差距矩阵

| # | 差距 | 严重度 | Phase 1 状态 | 新认知下的优先级变化 |
|:---:|------|:---:|------|:---|
| **G1** | CPU Prefill TTFT = 6s @ P=512 | 🔴 致命 | 已知但未解决 | **必须**在 Phase 2 解决 |
| **G2** | 滑动窗口注意力未实现 | 🔴 致命 | SLIDING_WINDOW=128 已定义但未使用 | 1M context 下 KV 读取量减少 8-32× |
| **G3** | 无 KV Cache Host Offload 路径 | 🔴 致命 | PCIe DMA 存在但未集成到 KV 管理 | 容量扩展的唯一可行路径 |
| **G4** | KV Cache 无跨会话共享 | 🟡 高 | 未考虑 | Agent 多轮对话可复用 80%+ KV |
| **G5** | 固定专家分配无负载均衡 | 🟡 高 | 拓扑正确但路由未完整 | Zipf 热专家导致芯片间负载不均 |
| **G6** | 无可观测性基础设施 | 🟡 高 | 无性能计数器 | 无法验证 token/kWh 模型 |
| **G7** | Batch scaling 效率 K=23.1 | 🟢 中 | 已知常数 | 低 batch 时 HBM 利用率低 |
| **G8** | 单模型固化 | 🟢 中 | 已知限制 | 业务风险, 但非技术瓶颈 |

---

## 3. 优先级排序与行动方案

### P0 (Phase 2 必须完成)

#### G1: FPGA 侧 Prefill 引擎

**问题**: CPU Prefill P=512 → 6s TTFT，用户体验不可接受。

**根因**: O(P²) 注意力在 CPU 上跑，3 TFLOPS FP8 根本不够。

**方案**: 实现 fp4_prefill_engine (已存在 RTL 框架，但未完整实现 multi-pass weight reload)

```
当前: Token → CPU (6s @ P=512) → PCIe → KV Cache → FPGA Decode
目标: Token → FPGA fp4 Prefill (~150ms) → KV Cache → FPGA Decode
```

**关键设计决策**:
- Prefill 使用与 Decode 相同的 DSP 阵列（分时复用，非专用硬件）
- Weight reload 是核心挑战：prefill 需加载 6 个专家权重（非 Decode 的 1 个）
- Prefill 和 Decode 不可同时运行 → 引入 Prefill/Decode 调度仲裁

**预期收益**: TTFT 降低 40× (6s → 150ms @ P=512)

#### G2: 滑动窗口注意力

**问题**: 当前 attention 对所有 P 个历史 token 做全量 QK dot product。在 1M context 下，
每步 Decode 需要从 HBM 读取 1M 个 KV 条目，仅为了发现 99.99% 的注意力分数接近 0。

**方案**: 
```systemverilog
// 当前: 全量注意力
for (int pos = 0; pos < seq_len; pos++)  // O(P), P 可达 1M
    score[pos] = dot(Q, K[pos]);

// 目标: 滑动窗口 + 全局稀疏注意力
for (int pos = max(0, seq_len - 128); pos < seq_len; pos++)  // 128 局部窗口
    score[pos] = dot(Q, K[pos]);
// + 少量全局注意力 token (router-guided, 或 hash-based)
```

**RTL 改动**:
- `mla_attention_v2.sv`: 添加 `window_start` 信号，限制 KV 读取范围
- `mla_kv_cache.sv`: rd_addr 生成逻辑支持窗口裁剪
- 新增 `sparse_attn_topk.sv`: 基于 router score 选择全局关注 token (≤256 个)

**预期收益**:
- KV 读取量: 1M → 128 + 256 = 384 tokens/decode step (减少 2,600×)
- HBM 带宽需求: 从 1M×576B=576MB/step → 384×576B=221KB/step
- 等效并发: HBM 容量不再是瓶颈 → 回到吞吐瓶颈 (~1,744 会话)

### P1 (Phase 2 应完成)

#### G3: KV Cache Host Offload

**问题**: HBM 只有 32GB/芯片，无法支撑大规模 Agent 部署。

**方案**: PCIe DMA → Host DDR5 作为 KV 二级存储

```
HBM (32GB L1) ←→ PCIe DMA (64GB/s) ←→ Host DDR5 (512GB L2)
  热KV               冷热迁移              冷KV
```

**关键设计**:
- `kv_dma_bridge.sv` (已有, 5/5 test PASS) → 扩展为 KV 分层管理
- 新增 `kv_swap_controller.sv`: LRU 驱逐 + 预取
- Agent 多轮对话: 同一 session 的历史轮次 KV 自动 swap-in
- PCIe 延迟 ~5μs → swap 一个 256K 会话的 KV 需要 ~2.2ms (2.15GB / 64GB/s × 1000/0.85)

**预期收益**: 有效 KV 容量从 32GB → 544GB (L1+L2), 1M context 从 15 会话 → 250+ 会话

#### G4: 跨会话 KV 复用

**问题**: Agent 多轮对话中，每轮都重新存储完整 KV。轮次间 KV 有 >80% 重叠。

**方案**: KV Cache 引入 COW (Copy-on-Write) 块管理
```
Session A, Turn 1: Block[0..99]  → 100 blocks
Session A, Turn 2: Block[0..95] (COW from Turn1) + Block[100..104] (new) → 105 blocks
```

**RTL 改动**: `mla_kv_cache.sv` → 添加 block-level ref counting

**预期收益**: 多轮 Agent KV 节省 60-80%

#### G5: 动态专家负载均衡

**问题**: Zipf 分布下，前 20% 专家承载 55% 流量。固定 12 专家/芯片 → 部分芯片过热。

**方案**: 热专家复制到多芯片 + Router 全局调度
```
当前: Chip N 固定拥有 Expert[N*12 : N*12+11]
目标: 热专家 (top-8) 复制到 4 个芯片, 冷专家合并到 1 个芯片
      Router 在复制专家间做 round-robin
```

**预期收益**: 峰值芯片负载降低 40%, 等效 TPS 提升 1.5-2×

### P2 (Phase 3 考虑)

#### G6: 可观测性基础设施

```
新增性能计数器:
- hbm_rd_bytes, hbm_wr_bytes (per chip, per layer)
- kv_cache_hit, kv_cache_miss (per session)
- dsp_active_cycles, dsp_idle_cycles
- pcie_dma_bytes, pcie_dma_latency_us
- per_layer_latency_cycles
```

**用途**: 验证 token/kWh 模型，发现实际瓶颈

#### G7: Batch-Aware 流水线调度

**问题**: K=23.1, B=1 时流水线效率仅 4.1% (724/17445)。

**方案**: Token 累积缓冲区 + 动态 batch
```
不是每来一个 token 就发射 → 累积 min(N, 32) 个 token 再发射
引入可控延迟 (max 50ms) 换取 batch efficiency
```

#### G8: 增量综合工具链

**问题**: 修改一行 RTL → 重新综合 → 4-6 小时。阻碍快速迭代。

**方案**: Quartus 增量编译 + 预编译 IP 库 + 模块级综合验证

---

## 4. 什么不应该优化（反模式）

基于新认知，以下优化方向是**错误**的：

| 不要做 | 原因 |
|------|------|
| ❌ 堆更多 DSP | 算力已过剩 116×。HBM 才是瓶颈 |
| ❌ 提高 DSP 频率 (450→600MHz) | 时序收敛难，收益为 0（HBM 喂不饱） |
| ❌ 换更大的 FPGA (Agilex 9) | 更多 DSP, 同样 HBM → 浪费 |
| ❌ fp4→fp8 精度升级 | Decode 对精度不敏感。浪费 2× 带宽 |
| ❌ 增加芯片互联带宽 | C2C 通信仅占总延迟 <5% |
| ❌ 支持训练 / Fine-tuning | 背离 FPGA 核心优势 |
| ❌ 通用推理框架 (ONNX/TensorRT) | 破坏确定性流水线优势 |

---

## 5. 修正后的 Phase 2 Roadmap

```
Phase 1 (已完成):     RTL 功能验证 + 参数化修复
                      ↓
Phase 2A (P0, 4-6周): FPGA Prefill 引擎 + 滑动窗口注意力
                      目标: TTFT 6s→150ms, KV读取量 2600×↓
                      ↓
Phase 2B (P1, 4-6周): KV Host Offload + 跨会话复用 + 动态专家均衡
                      目标: 1M context 15→250+ 会话
                      ↓
Phase 3 (P2, 6-8周):  可观测性 + Batch调度 + 增量综合
                      目标: 生产可运维
                      ↓
Phase 4 (P3, 8-12周): 硬件实测 + 稳定性 + 多模型适配
```

---

## 6. 关键指标重定义

| 指标 | Phase 1 关注 | Phase 2 应关注 |
|------|------|------|
| **算力** | DSP TMACs, 利用率 | HBM 带宽利用率, KV 读取量/step |
| **延迟** | 每层 cycle 数 | TTFT (Prefill), TPOT (Decode) |
| **容量** | HBM 总 GB | 有效并发会话数 @ 1M context |
| **效率** | DSP 利用率 % | token/kWh (含 HBM 容量因子) |
| **正确性** | bit-exact vs Python golden | 端到端输出确定性 |
| **灵活性** | 参数化 (HIDDEN/K_LATENT) | 多模型权重热加载 |

---

> **核心结论**: Phase 1 证明了 "我们能正确实现这个架构"。
> Phase 2 需要回答 "这个架构在 HBM 瓶颈约束下是否真正有竞争力"。
> 答案是: 只有完成 G1(FPGA Prefill) + G2(滑动窗口) + G3(KV Offload) 三项，
> FPGA LPU 的 token/kWh 理论优势才能转化为生产环境的实际优势。
