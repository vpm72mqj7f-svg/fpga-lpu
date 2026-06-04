# Phase 2 执行计划：基于关键洞察的优化路线图

> **状态**: Phase 1 已完成（36 模块 RTL 验证，参数化修复）
> **驱动**: 四项关键洞察 + Roofline 带宽/算力比分析
> **方法**: 每步必须 "仿真模型 → RTL 实现 → 回归测试"，三 agent 独立协作

---

## 0. 关键洞察（不可遗忘的设计约束）

### 洞察 0.1: HBM 容量是真实物理瓶颈

```
算力过剩比 = 流水线吞吐限制 / HBM 容量限制
           = 1,744 会话 / 15 会话 (@ 1M context)
           = 116×

→ 堆 DSP 没有意义。优化 HBM 容量利用率才是正解。
```

**约束 C0.1**: 任何增加 DSP 利用率的优化，如果不改善 HBM 容量效率，实际收益为 0。
**约束 C0.2**: 新架构决策必须通过 "这个改动能支撑更多会话 @ 1M context 吗？" 的检验。

### 洞察 0.2: Prefill/Decode 物理分离是最优解

```
Prefill:  O(P²) 注意力, 大批量 → GPU Tensor Core 最优
Decode:   逐 token 串行, 延迟敏感 → FPGA 固定流水线最优

不存在一种硬件能同时最优处理两者。
CPU/GPU Prefill + FPGA Decode 不是妥协，是物理最优分解。
```

**约束 C0.3**: 不要尝试在 FPGA 上做通用 Prefill。用 CPU/GPU 做 Prefill，FPGA 专注 Decode。
**约束 C0.4**: FPGA Prefill 引擎（P0）仅针对 fp4 注意力加速，不是替代 CPU/GPU，是补充。

### 洞察 0.3: token/kWh 是唯一正确的统一度量

```
token/kWh = KV密度 × DSP利用率 × 1/功耗 × 架构效率

FPGA LPU: 16.5× H200 = 3.1 × 1.3 × 1.9 × 2.2 (四个因子全 >1.0)
B300:      1.6× H200 = 1.0 × 1.1 × 0.7 × 2.1 (HBM翻倍被功耗翻倍吃掉)
950PR:     1.2× H200 = 1.0 × 0.9 × 1.7 × 0.8 (CANN拖累)
```

**约束 C0.5**: 所有架构决策必须通过 token/kWh 四因子分解验证。
**约束 C0.6**: 不要优化次要因子（如 DSP 利用率 90%→95%）而忽视主要因子（KV 密度 3.1×）。

### 洞察 0.4: 专家权重加载是 Decode 的真正时间黑洞

```
每层 Decode 时间分解 (B=1, sliding window 128+256):

KV 读取:    432 KB →   0.5 μs ( 0%)  ← 不是瓶颈
注意力计算: 493M MACs → 45 μs (18%)  ← 也不是瓶颈
专家权重:   165 MB  →  205 μs (82%)  ← 真正的瓶颈

OI = 2.8 MACs/byte << 硬件 13.1 → 严重 BANDWIDTH-BOUND
DSP 78% 空闲, 在等 HBM 搬运专家权重
```

**约束 C0.7**: 优化重心 = 减少专家权重从 HBM 的加载量。不是优化 KV 读取，不是优化 MAC 精度。
**约束 C0.8**: Batch ≥ 6 可摊销权重加载，突破带宽墙。Pipeline 调度应优先保证 B≥6。
**约束 C0.9**: 最优 SRAM/TMAC = 18 MB/TMAC。当前 2.9 MB/TMAC。SRAM 扩展 > DSP 扩展。

---

## 1. 优化阶段总览

```
Phase 2A [已完成]: 滑动窗口注意力 + Batch 累积调度 + decode_pipeline wrapper
  → 滑动窗口: KV 读取 O(1), 消除 O(P) 注意力计算瓶颈
  → 批累积: B≥6 摊销专家权重加载, OI 2.8→14.9 MACs/byte
  → decode_pipeline: accumulator→transformer 串联, 权重自然复用

Phase 2B [P0, 3-4 周]: 热专家副本 (Expert Replication)
  → 目标: P(0 local) 从 82.7%→≈0%, DSP 利用率 22%→95%+
  → 手段: top-8 热专家全芯片复制, HBM 代价仅 8.2GB
  → token/kWh 从 16.5× → 30×+ H200

Phase 2C [P1, 4-6 周]: KV Host Offload + 跨会话复用
  → 目标: 1M context 从 15→250+ 会话
  → Agent 场景 token/kWh 再提升 3-5×

Phase 3  [P2, 6-8 周]: 可观测性 + 硬件实测 + 稳定性
  → 目标: 生产可运维

NOT IN PLAN (架构决策):
  → FPGA Prefill: 不放在 FPGA 上。CPU/GPU 处理 Prefill, FPGA 100% 打满 Decode
```

---

## 2. Phase 2A: 滑动窗口注意力 + Batch 调度

### 2A.1 目标

| 指标 | 当前 | 目标 | 手段 |
|------|:---:|:---:|------|
| 每层 Decode 时间 | 250 μs | 45 μs | 消除权重加载等待 |
| KV 读取量/step | 1M tokens (全量) | 384 tokens (128+256) | 滑动窗口 |
| Batch 调度 | 来一个跑一个 | 累积到 B≥6 | Token 缓冲 |
| DSP 利用率 | 22% | 95%+ | 权重预取 + B≥6 |

### 2A.2 仿真模型 (仿真实体: SW-ENG1)

**任务**: 在 Python pipeline 模型中验证滑动窗口 + batch 调度的端到端效果。

```
关键参数:
- SLIDING_WINDOW = 128 (局部窗口)
- SPARSE_TOPK = 256 (全局稀疏注意力)
- BATCH_ACCUMULATE_MAX = 32 (最大累积 batch)
- BATCH_ACCUMULATE_TIMEOUT_US = 50_000 (最多等 50ms)

输出:
- 不同 batch_size 下的 per-layer 时间 (验证 OI 转移)
- 端到端 TPS vs context length 曲线
- 与全量注意力的精度对比 (perplexity delta)
- token/kWh 变化
```

**文件**: `scripts/simulation/phase2a_window_batch_model.py`

### 2A.3 RTL 实现 (RTL 实体: RTL-ENG1 + RTL-ENG2)

#### 2A.3a: mla_kv_cache 滑动窗口支持

```systemverilog
// mla_kv_cache.sv 新增端口
input  logic [$clog2(MAX_SEQ_LEN)-1:0] window_start,  // 窗口起始位置
input  logic                             window_mode,   // 1=滑动窗口模式
```

**改动**: rd_addr 生成逻辑支持窗口裁剪。当 `window_mode=1` 时，rd_addr 仅遍历 `[window_start, window_start+128)` + 稀疏全局位置（从 router 获取）。

**约束**: 不影响现有 ring buffer 语义。`window_mode=0` 时行为不变（向后兼容）。

#### 2A.3b: mla_attention_v2 稀疏注意力

```systemverilog
// mla_attention_v2.sv 新增
input  logic [SPARSE_TOPK-1:0][$clog2(MAX_SEQ_LEN)-1:0] sparse_positions,
input  logic [$clog2(SPARSE_TOPK+1)-1:0]                 sparse_count,
```

**改动**: QK dot product 循环仅遍历 `window_positions ∪ sparse_positions`，而非全量 `[0, seq_len)`。

**约束**: 稀疏位置由 router score 离线生成（复用 router_topk 的 expert score 作为 attention relevance proxy）。

#### 2A.3c: layer_compute_engine Batch 累积

```systemverilog
// layer_compute_engine.sv 新增 FSM 状态
S_BATCH_ACCUMULATE  // 等待累积到 B>=6 或超时
```

**约束**: 
- 最大累积延迟 ≤ 50ms (TPOT SLA)
- 累积期间不阻塞新 token 到达
- 超时后即使 B<6 也发射（防止长尾延迟）

### 2A.4 回归测试 (测试实体: VERIF-ENG1 + VERIF-ENG2)

| 测试 | 内容 | 负责人 |
|------|------|------|
| `tb_mla_kv_cache_window` | 窗口模式读写正确性 | VERIF-ENG1 |
| `tb_mla_attention_sparse` | 稀疏注意力 vs 全量注意力精度对比 | VERIF-ENG2 |
| `tb_lce_batch_accum` | Batch 累积 FSM 正确性 | VERIF-ENG1 |
| `tb_full_transformer_window` | 端到端滑动窗口流水线, 确定性验证 | VERIF-ENG3 |

**精度标准**: 稀疏注意力输出与全量注意力的 Q12 误差 ≤ 1 LSB（基于 router score 的 top-256 全局注意力覆盖 >99% 注意力质量）。

---

## 3. Phase 2B: 热专家副本 (Expert Replication)

### 3B.0 为什么这是正确解法

当前专家权重加载是 Decode 的带宽瓶颈：
```
P(0 local expert) = 82.7% → 大多数 token 在某芯片上无本地专家
→ 需要从 HBM 加载 165MB 专家权重 → 205μs/layer (82% 总时间)
```

三种解决方案对比：

| 方案 | 效果 | 代价 | 可行性 |
|------|------|------|:---:|
| 堆 SRAM (198MB 缓存 6 专家) | P(0)→0% | 硬件不支持 (32.5MB max) | ❌ |
| Batch≥6 (摊销权重) | OI 2.8→14.9 | 延迟增加 | ⚠️ 部分解决 |
| **热专家全芯片副本** | **P(0)→0%** | **HBM 8.2GB (可接受)** | **✅** |

热专家副本直接消除权重加载需求——权重永远在本地, HBM 带宽完全释放给 KV cache。

### 3B.1 数学模型

```
Top-8 热专家覆盖 55% token mass (Zipf 分布)
热专家全芯片副本 (32x):
  → avg 本地专家 = 6 × 0.55 × 1.0 + 6 × 0.45 × 12/376 = 3.39 experts/token
  → P(0 local expert) ≈ 0%
  → 等同于 SRAM 无限大

HBM 代价:
  8 个热专家 × (32-1) 个额外副本 × 33MB/expert = 8.2 GB
  当前 HBM 空闲 (扣除 KV + 基础权重) = 31.3 GB
  → 8.2 GB << 31.3 GB ✓
```

### 3B.2 目标

| 指标 | 当前 | 目标 | 手段 |
|------|:---:|:---:|------|
| P(0 local expert) | 82.7% | ≈0% | Top-8 热专家全芯片副本 |
| avg local experts/token | 0.09 | 3.39 | 副本 × 权重本地化 |
| DSP 利用率 (Decode) | 22% | 95%+ | 消除 HBM 权重加载 |
| 每层 Decode 时间 | 250 μs | 45 μs | 无权重加载等待 |
| token/kWh vs H200 | 16.5× | 30×+ | 综合优化 |

### 3B.3 仿真模型 (仿真实体: SW-ENG1)

**任务**: 更新 expert_popularity.py，模拟热专家副本对芯片负载的影响。

```
关键参数:
- Zipf α sweep (0.5-2.0) → 确定 "热专家" 阈值
- 副本因子 sweep (1x-32x) → P(0 local) 曲线
- HBM 权重占用 vs 副本因子 → 容量约束
- 芯片间负载均衡 (热点芯片识别)

输出:
- 最优副本配置 (哪些专家, 几副本, 放哪些芯片)
- P(0 local) vs 副本因子
- 期望 DSP 利用率 vs 副本因子
- HBM 容量预算表
```

**文件**: `scripts/simulation/phase2b_expert_replication_model.py`

### 3B.4 RTL 实现 (RTL 实体: RTL-ENG3)

#### 3B.4a: chip_top 专家权重复制加载

```systemverilog
// chip_top.sv 修改: 支持加载复制专家的权重
// 当前: 每个芯片只加载自己的 12 个专家
// 目标: 额外加载 8 个热专家副本 (从全局权重表)
```

**约束**:
- 副本权重在初始化时从 Host 加载（一次加载, 常驻 HBM）
- 不改变 `expert_ffn_engine_fp4_down` 的 INSTANTIATED 专家数量
- 副本专家的 `expert_sel` 映射到本地专家 ID（软件可配置）

#### 3B.4b: Router 全局调度（可选, 延后到 Phase 2C）

```systemverilog
// 当前: Router 选择 top-6 专家, 由 C2C dispatch 到对应芯片
// 优化: Router 知道每个芯片的专家副本分布
//       → 优先选择本地有副本的专家
//       → 减少跨芯片 C2C 通信
```

**约束**: Router LUT 可动态更新（专家副本分布变化时重配置）

### 3B.5 回归测试 (测试实体: VERIF-ENG3)

| 测试 | 内容 | 负责人 |
|------|------|------|
| `tb_expert_replication_load` | 副本权重加载正确性 (gate/up/down) | VERIF-ENG3 |
| `tb_expert_replication_infer` | 副本专家 FFN 推理 vs 原始专家 (bit-exact) | VERIF-ENG3 |
| `tb_chip_12layer_replicated` | 全芯片流水线, 副本权重, 确定性 | VERIF-ENG3 |

---

## 4. Phase 2C: KV Host Offload + 跨会话复用

### 4C.1 目标

| 指标 | 当前 | 目标 | 手段 |
|------|:---:|:---:|------|
| 并发 @ 1M context | 15 | 250+ | Host DDR5 KV offload |
| 多轮 Agent KV 节省 | 0% | 60-80% | COW 块管理 |
| PCIe DMA 利用率 | 无 | 85%+ | kv_swap_controller |

### 4C.2 仿真模型 (仿真实体: SW-ENG3)

**任务**: 建立 KV 分层存储的性能模型（HBM L1 + DDR5 L2）。

```
关键参数:
- KV_SWAP_BLOCK_SIZE = 64 tokens (对齐 PCIe 传输粒度)
- PCIe effective BW = 54.4 GB/s (64 × 0.85)
- DDR5 capacity = 512 GB (单路 Xeon)
- COW block ref_count 管理开销

输出:
- KV swap-in 延迟 vs block 大小
- 多轮 Agent 场景的 swap 次数和 KV 节省率
- 不同 context 分布下的最优 L1/L2 划分
```

**文件**: `scripts/simulation/phase2c_kv_offload_model.py`

### 4C.3 RTL 实现 (RTL 实体: RTL-ENG2 + RTL-ENG3)

#### 4C.3a: kv_swap_controller

```systemverilog
// 新模块: kv_swap_controller.sv
// 管理 HBM (L1) ←→ Host DDR5 (L2) 的 KV 块迁移
// LRU 驱逐 + 预取 (基于当前活跃 session 的访问模式)
```

**约束**:
- Swap 操作优先级低于 Decode KV 读取（不阻塞推理）
- 最大 swap-in 延迟 ≤ 5ms（对 TPOT 影响 <10%）
- Block 粒度 64 tokens（平衡 PCIe 效率和延迟）

#### 4C.3b: mla_kv_cache COW 块管理

```systemverilog
// mla_kv_cache.sv 扩展
// 每个 block 增加 ref_count 字段 (在 valid SRAM 旁边)
// 新增: fork_session(session_id) → 复制 block 引用
// 新增: free_session(session_id) → 递减 ref_count, ref_count=0 时释放
```

**约束**: ref_count 不超过 255（8-bit 计数器）。溢出时回退到 copy-on-overflow。

### 4C.4 回归测试 (测试实体: VERIF-ENG2 + VERIF-ENG3)

| 测试 | 内容 | 负责人 |
|------|------|------|
| `tb_kv_swap_controller` | LRU 驱逐 + 预取正确性 | VERIF-ENG2 |
| `tb_mla_kv_cache_cow` | COW ref_count, fork/free 正确性 | VERIF-ENG2 |
| `tb_kv_offload_e2e` | Swap + 多轮对话 KV 一致性 | VERIF-ENG3 |

---

## 5. Agent 角色定义

### 5.1 仿真实体 (SW-ENG1 / SW-ENG2 / SW-ENG3)

```
角色文件: .claude/roles/sw-eng1.md (FPGA 架构仿真)
         .claude/roles/sw-eng2.md (服务栈仿真)
         .claude/roles/sw-eng3.md (验证与实验)

关键约束 (每次仿真必须检查):
□ HBM 容量是否构成瓶颈? (会话数 vs context 长度)
□ 专家权重加载占每层时间的比例? (B=1 时是否 >50%?)
□ OI 是否低于 HW ratio 13.1? (如是, bandwidth-bound)
□ Batch size 是否 ≥6? (如否, 无法突破权重墙)
□ token/kWh 四因子分解是否全部 ≥1.0?

交付物:
- Python 仿真脚本 (scripts/simulation/phase2x_*.py)
- 性能预测报告 (TPS, TTFT, token/kWh)
- 与 RTL 实现的偏差分析
```

### 5.2 RTL 实体 (RTL-ENG1 / RTL-ENG2 / RTL-ENG3)

```
角色文件: .claude/roles/rtl-eng1.md (DSP 数据通路)
         .claude/roles/rtl-eng2.md (MLA/Attention)
         .claude/roles/rtl-eng3.md (Layer/MoE/Chip)

设计约束 (每次 RTL 改动必须检查):
□ 全同步逻辑: 所有 always_ff 使用同一时钟沿
□ 参数化: 所有维度来自 lpu_config.svh, 无常量硬编码
□ flat ports: 无 unpacked array 端口 (Icarus 兼容)
□ Altera IP: 所有 RAM/DSP/FIFO 使用 altera_* wrapper
□ 确定性: 相同输入 + 相同权重 → 相同输出 (周期精确)
□ 向后兼容: window_mode=0 时行为不变

Roofline 约束:
□ 新模块的 OI 必须 ≥13.1 MACs/byte (避免引入新带宽瓶颈)
□ 权重访问必须是批处理友好的 (B≥6 时权重只需加载一次)
□ KV 读取必须支持滑动窗口 (不可退化为全量读取)

交付物:
- SystemVerilog 源文件 (rtl/*/*.sv)
- 模块级 testbench (rtl/sim/tb_*.sv)
- 仿真通过报告 (Icarus + Verilator)
```

### 5.3 测试实体 (VERIF-ENG1 / VERIF-ENG2 / VERIF-ENG3)

```
角色文件: .claude/roles/verif-eng1.md (DSP + 激活验证)
         .claude/roles/verif-eng2.md (MLA + MoE 验证)
         .claude/roles/verif-eng3.md (层/芯片/集群集成)

测试标准:
□ Icarus 编译零错误 (IVERILOG -g2012)
□ Verilator --lint-only 零警告 (生产参数)
□ 所有新增测试 PASS
□ 已有回归测试无退化 (make run_all_tests)
□ 精度验证: RTL vs Python golden (Q12 位精确)
□ 确定性验证: 相同种子跑两次, 输出 bit-exact

交付物:
- Testbench 源文件 (rtl/sim/tb_*.sv)
- Golden 参考数据 (Python 生成, tb_*_golden_pkg.sv)
- 测试报告 (PASS/FAIL, 覆盖率, 周期数)
```

---

## 6. 阶段门禁 (Gate Criteria)

### Gate 2A (滑动窗口 + Batch 调度通过)

```
□ 稀疏注意力精度: Q12 误差 ≤ 1 LSB vs 全量注意力 (1000 随机测试)
□ 滑动窗口 KV 读取量: ≤ 384 tokens/step (验证 HBM 带宽降低)
□ B≥6 批处理: DSP 利用率 ≥ 85% (从 22% 提升)
□ Per-layer 延迟: ≤ 50 μs (从 250 μs 降低)
□ 已有回归: 28/28 PASS (无退化)
□ token/kWh: ≥ 20× H200 (从 16.5× 提升)
```

### Gate 2B (热专家副本通过)

```
□ P(0 local expert): ≤ 5% (从 82.7% 降低 16×)
□ 专家权重 HBM 加载量: ≤ 33MB/layer (仅 1 冷专家, 从 165MB 降低 5×)
□ DSP Decode 利用率: ≥ 90% (从 22% 提升 4×)
□ 每层 Decode 时间: ≤ 50 μs (从 250 μs 降低 5×)
□ 副本权重 bit-exact vs 原始专家: 100% 验证通过
□ token/kWh: ≥ 30× H200 (从 16.5× 提升)
```

### Gate 2C (KV Offload + 会话复用通过)

```
□ KV swap-in 延迟: ≤ 5ms (对 TPOT 影响 <10%)
□ 多轮 Agent KV 复用率: ≥ 60% (COW 验证)
□ 并发 @ 1M context: ≥ 200 会话 (从 15 提升)
□ PCIe DMA + HBM 并发访问: 无死锁, 无数据竞争
□ 已有回归: 无退化
□ Agent 场景 token/kWh: ≥ 30× H200 (综合 HBM 容量 + 批处理 + 复用)
```

---

## 7. 不应该做的事 (反模式清单)

以下优化方向已被 Roofline 分析证明**无效**，在 Phase 2 中明确禁止：

| 禁止项 | 判死刑的原因 |
|------|------|
| ❌ 增加 DSP 数量 | 11.07 TMACs 已有 78% 空闲 |
| ❌ 提高 DSP 频率 (450→600MHz) | 时序收敛难, 收益 0 (HBM 喂不饱) |
| ❌ FP4→FP8 精度升级 | 浪费 2× BW, Decode 对精度不敏感 |
| ❌ 增加 C2C 互联带宽 | C2C 通信 <5% 总延迟 |
| ❌ 全量注意力优化 (FA/FlashDecode) | 滑动窗口直接跳过问题 |
| ❌ KV Cache 精度位宽扩展 | 576B FP8 已足够, 更大 = 更少会话 |
| ❌ 支持 ONNX/TensorRT 通用推理 | 破坏确定性流水线, 引入指令开销 |
| ❌ 扩大 systolic array (M_ROWS > 32) | 权重加载时间增长 > 并行收益 |

**一句话**: 任何不减少 HBM 流量或不增加有效并发会话数的优化，都是浪费。

---

## 8. 关键指标仪表盘

| 指标 | Phase 1 基线 | 2A 目标 | 2B 目标 | 2C 目标 |
|------|:---:|:---:|:---:|:---:|
| **Decode TPS** | 17,445 | 17,445 | 17,445 | 17,445 |
| **B=1 延迟** | 1.51 ms | 0.30 ms | 0.30 ms | 0.30 ms |
| **TTFT @ P=512** | 6,000 ms | 6,000 ms | **150 ms** | 150 ms |
| **并发 @ 1M ctx** | 15 | 15 | 15 | **250** |
| **DSP 利用率** | 22% | **85%** | **95%+** | 95%+ |
| **每层时间** | 250 μs | **45 μs** | **45 μs** | 45 μs |
| **token/kWh (vs H200)** | 16.5× | **20×** | **30×** | **35×** |
| **P(0 local expert)** | 82.7% | 82.7% | **≈0%** | ≈0% |
| **HBM 权重加载** | 165MB | 165MB | **≤33MB** | ≤33MB |
| **Agent KV 复用** | 0% | 0% | 0% | **60%** |

---

> **核心哲学**: Phase 1 证明了 "我们能正确实现"。Phase 2 要证明 "我们理解瓶颈在哪里，并系统地消除它"。
> 每一步都必须通过 仿真预测 → RTL实现 → 回归验证 的闭环，由三个独立 agent 角色协作完成。
