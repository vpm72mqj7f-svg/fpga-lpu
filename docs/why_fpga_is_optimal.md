# FPGA 方案为什么能成为 DeepSeek V4 Pro 的最优硬件适配

> 不是 "买不到 GPU 的备胎"，而是 "对于这类模型，FPGA 架构上就是比 GPU 更优"

---

## 一、先搞清楚 DeepSeek V4 Pro 推理到底在干什么

```
V4 Pro Decode (生成阶段, 推理时间的 95%):

  每生成 1 个 token, 需要:
  
    ① 从 HBM 加载权重:    ~6.1 GB  (全 61 层, 每层 ~100 MB)
    ② 执行矩阵乘:         ~37.4 GMACs
    ③ 跨层/跨卡通信:      可忽略 (<2%)

  核心特征:
    → HBM 带宽消耗巨大 (6.1 GB/token)
    → 计算量并不大 (37.4 GMACs << GPU 的 1,000+ TFLOPS)
    → 这是典型的 "内存受限" 工作负载

  为什么会 "内存受限"?
    → 生成了 1 个 token, 需要加载几乎全部权重 (6.1 GB)
    → 这些权重只被 1 个 token 用一次就换掉
    → 计算密度 = 37.4 GMACs / 6.1 GB ≈ 6 MACs/Byte
      任何一个 HBM 带宽 < 计算吞吐 × 6 的设备都会计算闲置
```

---

## 二、GPU 跑这个工作负载有什么问题

```
H100 SXM (80GB HBM3, 3.35 TB/s, 990 TFLOPS FP8):

  加载 6.1 GB 权重:  6.1 / 3,350 = 1.82 ms
  执行 37.4 GMACs:  37.4 / 990,000 = 0.04 ms
  
  Tensor Core 在这 1.86 ms 里干了 0.04 ms 的活。
  利用率 = 0.04 / 1.86 = 2.1%
  → 98% 的 Tensor Core 在空转等 HBM

  但 H100 的 3.35 TB/s HBM 是真金白银的硅片面积和功耗。
  Tensor Core 也是。
  你花了 $30K 买了 990 TFLOPS, 只用了 20 TFLOPS。

H200 (141GB HBM3e, 4.8 TB/s):
  加载: 6.1 / 4,800 = 1.27 ms
  计算: 0.04 ms
  利用率: 3.0% → 仍然 97% 空转

  更大 HBM 让权重可以多缓存, 但 decode 场景下每 token
  仍需遍历大量权重, HBM 带宽仍然是瓶颈。

B200 (192GB HBM3e, 8.0 TB/s):
  加载: 6.1 / 8,000 = 0.76 ms
  利用率: 5% → 仍然 95% 空转
```

**根本问题：GPU 的架构假设是高计算密度（训练），但 LLM decode 是低计算密度（推理）。** GPU 的设计哲学是"算力为王"，HBM 只负责喂数据给 Tensor Core。但在 decode 场景下，Tensor Core 饿死了，HBM 才是真正的瓶颈。

---

## 三、FPGA 为什么更适合

```
Agilex 7 M (32GB HBM2e, 920 GB/s, 8.44 TMACs fp4):

  加载 6.1 GB 权重:  6.1 / 920 = 6.63 ms
  执行 37.4 GMACs:  37.4 / 8,440 = 4.43 ms
  
  DSP 利用率 = 4.43 / 6.63 = 67%

  → 没有 98% 的空转
  → HBM 和 DSP 接近平衡
  → 你花的每一分钱都在干实事
```

**核心差异：HBM 带宽 / 算力 的比值。**

```
                ┌───────────┬──────────────┬──────────────────┐
                │ HBM 带宽   │ 算力           │ HBM/算力 比值      │
                │ (GB/s)    │ (TFLOPS/TOPS) │                  │
  ┌─────────────┼───────────┼──────────────┼──────────────────┤
  │ H100 SXM    │ 3,350     │ 990 (FP8)    │ 3.4 GB/T        │
  │ H200 SXM    │ 4,800     │ 990 (FP8)    │ 4.8 GB/T        │
  │ B200        │ 8,000     │ 2,250 (FP8)  │ 3.6 GB/T        │
  │ A100 SXM    │ 2,039     │ 312 (FP16)   │ 6.5 GB/T        │
  ├─────────────┼───────────┼──────────────┼──────────────────┤
  │ Agilex 7 M  │ 920       │ 8.4 (fp4)    │ 110 GB/T  ←     │
  └─────────────┴───────────┴──────────────┴──────────────────┘

  FPGA 的 HBM/算力比是 GPU 的 23-32×。
  这正是 LLM decode 需要的。
```

---

## 四、FPGA 的独特优势不靠供应管制，靠架构

三个 GPU 做不到的事：

### ① fp4 原生，零解压

```
DeepSeek V4 Pro 的专家权重是 fp4 (E2M1)。

GPU 路径:
  fp4 weights (HBM)
    → load → decompress → FP8 → Tensor Core FP8 MAC
    → 走了 3 步，解压步骤浪费 ALU，增加延迟

FPGA 路径:
  fp4 weights (HBM)
    → load → BRAM lookup → DSP fp4 MAC
    → 走了 2 步，查表在 BRAM 完成，不占用 DSP

国产 GPU (Ascend/寒武纪/海光/壁仞) 的 Tensor Core 只认 INT8/FP8/FP16，不认 fp4。
这不是驱动问题，是硅片级的硬化电路不支持。

NVIDIA B200/GB200 已支持 fp4, 但受出口管制且单卡 $30-40K。
在中国可获取的硬件中, 只有自研 FPGA 能做 fp4 原生推理。
```

### ② MLA 硬化，零 kernel launch

```
GPU 每层 Attention 需要:
  kernel_launch(Q_compress)  →  ~5μs launch overhead
  kernel_launch(KV_compress) →  ~5μs
  kernel_launch(QK_dot)      →  ~5μs
  kernel_launch(Softmax)     →  ~5μs
  kernel_launch(AV_dot)      →  ~5μs
  kernel_launch(O_decompress)→  ~5μs
  6 个 kernel × ~5μs = 30μs 纯 launch 开销

FPGA:
  6 级硬件流水线, 零 launch
  上一个 token 的结果流到下一级, 不间断
  对于 61 层 × 30μs = 1.83 ms 的 launch 开销,
  GPU 在吞吐敏感场景下可以通过 batch 来摊薄,
  但在 B=1~4 的 decode 场景下, 这个开销是真切的。
```

### ③ 硬件 KV Cache 管理，零软件干预

```
GPU (vLLM PagedAttention):
  KV Cache 存储在 GPU HBM
  → 软件 Block Table 管理
  → 每 token 每层分配/释放 block
  → CPU 发送管理指令到 GPU
  → 浪费 PCIe 带宽和 CPU 时间

FPGA:
  KV Cache 硬件地址生成器
  → {session_id, layer_id, seq_id} → HBM 物理地址
  → 滑动窗口 (128) 硬件自动淘汰
  → CPU 零参与
```

---

## 五、经济性对比

```
  方案 A: 8×H100 服务器 (假设能买到, $240K)
    实际有效算力 (decode 利用率 2.1%): ~14 TFLOPS (浪费的)
    HBM 总带宽: 26.8 TB/s (8 × 3.35)
    DeepSeek V4 Pro decode 吞吐 (估计): ~500-800 tok/s
    硬件成本: $240K

  方案 B: 30 FPGA 集群 (30 活跃 + 2 热备, $321K hardware + $150K R&D amort)
    DSP 利用率: ~50% (加权平均, 含 SRAM 缓存)
    HBM 总带宽: 27.6 TB/s (30 × 920 GB/s)
    F-Tile 内置 200GbE: 零外部 NIC
    DeepSeek V4 Pro decode 吞吐: ~800-1000 tok/s  
    硬件成本: $321K (含 R&D 摊薄: ~$471K)

  ┌────────────┬──────────┬──────────┬──────────────────────────┐
  │            │ 8×H100   │ 30 FPGA  │ 备注                      │
  ├────────────┼──────────┼──────────┼──────────────────────────┤
  │ 硬件成本    │ $240K    │ $321K    │ H100 假设未被管制          │
  │ 含 R&D TCO │ $240K    │ ~$471K   │ FPGA 方案有一次性 R&D       │
  │ 实际吞吐    │ ~600     │ ~980     │ tok/s, §4.4.1              │
  │ HBM 带宽    │ 26.8 TB/s│ 27.6 TB/s│ 30×920 GB/s               │
  │ $/百万token│ ~$12     │ ~$15-20  │ 单套集群 (70% 利用率)      │
  │ $/百万token│ ~$12     │ ~$7-9    │ 10 套集群 (R&D 摊薄)       │
  │ 网络芯片    │ NIC 额外 │ F-Tile内置│ FPGA 零外部 NIC 成本       │
  └────────────┴──────────┴──────────┴──────────────────────────┘

  10 套量产时, FPGA 的单位成本已经可以对标 H100。
  而 H100 在中国市场根本无法采购。
```

---

## 六、为什么说 FPGA 是 "最优适配" 而不是 "替代方案"

```
最优适配的三个层次:

层次 1: 技术适配 (架构级)
  → fp4 原生 ← DeepSeek 用 fp4
  → MLA 硬化 ← DeepSeek 用 MLA  
  → HBM/算力比 ← LLM decode 需要高 HBM/算力比
  → 这三个是 DeepSeek V4 Pro 和 FPGA 之间的 "天生的技术共振"

层次 2: 供应链适配 (地缘级)
  → GPU 受管制 ← 中国客户买不到
  → Ascend 产能受限 ← 华为自己也供应不够
  → FPGA 多源供应 ← Intel + 韩国 HBM + 东南亚封装
  → 这是地缘政治背景下的 "供应可靠性溢价"

层次 3: 部署适配 (商业级)  
  → 标准 PCIe 设备 ← 全球数据中心兼容
  → 不受 GPU 出口管制 ← 可部署到任何国家
  → 中国大模型出海 ← 唯一可用硬件底座
  → 这是商业模式上的 "部署自由度"

这三个层次, GPU 一个都做不到, Ascend 只能做到其中国内部分。
FPGA 是三个层次同时满足的唯一方案。
```

---

## 七、什么情况下 FPGA 不是最优

坦率承认局限：

### 并发短板分析

```
并发上限由两个因素决定: HBM KV Cache 容量 + 算力余量

┌──────────────────────┬────────────┬──────────────┐
│                      │ H200 (8卡) │ FPGA (30卡)  │
├──────────────────────┼────────────┼──────────────┤
│ 单卡 HBM              │ 141 GB     │ 32 GB        │
│ 总 HBM               │ 1,128 GB   │ 960 GB       │
│ 单卡管层数            │ 全 61 层   │ ~15 层(流水线)│
│ 单 session KV/卡      │ ~4.5 GB    │ ~1.18 GB     │
│ KV Cache 可用/卡      │ ~50 GB     │ ~8 GB        │
│ HBM 决定的并发上限     │ ~10-15     │ ~6-7         │
├──────────────────────┼────────────┼──────────────┤
│ Decode 算力利用率     │ ~2.1%      │ ~50% (B=1)   │
│ B=1 时算力余量        │ ~98%       │ ~50%         │
│ 算力决定的并发上限     │ ~40-50     │ ~1-2         │
├──────────────────────┼────────────┼──────────────┤
│ 实际并发 (取更紧者)    │ 10-15      │ 1-2          │
└──────────────────────┴────────────┴──────────────┘

FPGA 的并发上限由算力限定，不是 HBM:
  → B=1 时 DSP 加权利用率 ~50% (0-hit 层 100%, 1-hit 层 ~31%), 余量有限
  → B=4 微批次时, 每 session 吞吐降到 ~250 tok/s
  → 不适合公有云 API 的成百上千并发场景

H200 的并发上限由 HBM 容量限定:
  → Decode 98% 算力闲置, 可以摊给很多并发 session
  → 但 HBM 够大, 能存 10-15 个 session 的 KV Cache
  → 适合高并发公有云
```

### 但 MoE 架构本身就不适合大 Batch

```
以上是硬件层面的比较。还有一个更根本的论据:

DeepSeek V4 Pro 是 MoE 架构, MoE decode 天然排斥大 B:

  Dense 模型 (LLaMA):
    每 token 共享全部权重 → B=32 时 HBM 加载量 ≈ B=1
    GPU 可以靠大 B 摊薄延迟, 利用率高

  MoE 模型 (DeepSeek V4 Pro):
    每 token 激活 6/384 专家
    B=8 → 可能命中 B×6=48 个不同专家 → HBM 压力 ≈ B=1 的 8×
    
    更关键的是 All-to-All 通信:
      FB 增长: B=8 时跨节点 dispatch 数据量 = B=1 的 8×
      专家热点: B 越大, 某些热门专家被更多 token 集中命中
      负载不均: 热门专家过载, 冷门专家闲置, 全局延迟被最慢专家拖住

┌──────────────┬─────────────────┬─────────────────┐
│              │ Dense (LLaMA)   │ MoE (DeepSeek)  │
├──────────────┼─────────────────┼─────────────────┤
│ B=1          │ GPU 大量闲置    │ GPU 大量闲置    │
│ B=4          │ GPU 仍然闲置    │ GPU 利用率有限   │
│ B=8          │ GPU 利用率提升  │ All-to-All 开销显著 │
│ B=32         │ GPU 最佳区间   │ 专家不均衡严重   │
│ B=128        │ GPU 最优       │ 不可行          │
└──────────────┴─────────────────┴─────────────────┘

业界实际部署: MoE 推理通常 B ≤ 4-8。
DeepSeek 官方 API 也是小 B 或 B=1, 靠多实例水平扩展。

这意味着:
  在 dense 模型上, GPU 的大 B 优势是压倒性的 (B=32 vs B=1, 30× 吞吐差)
  在 MoE 模型上, 无论 GPU 还是 FPGA, 都只能在 B=1~8 区间运行
  → GPU 的算力闲置问题在 MoE decode 上无法通过堆 B 来根本解决
  → FPGA 的 B=1~4 区间恰好覆盖 MoE decode 的实际运行范围
```

### FPGA 的并发为什么低

```
根本原因: HBM/算力比。

GPU 的设计: 大量算力 + 相对少的 HBM → 训练时算力吃满, decode 时算力闲置
FPGA 的设计: 适中算力 + 匹配的 HBM → decode 时算力接近吃满, 没有闲置

同一件事的两种表述:
  优势面: FPGA 没有 98% 算力浪费, 每一分钱都在干实事
  劣势面: FPGA 的算力余量少, 不能靠堆并发 session 来摊薄延迟

这是同一把刀的两面 —— 不存在"既没有闲置算力, 又能无限提并发"的硬件。
```

### DeepSeek 的低价促销反过来证实了这一点

```
DeepSeek V3/R1 API 定价极低 (~$0.27/百万 token 输入, ~$1.10/输出),
2025-2026 年多次促销, 力度显著大于 OpenAI/阿里/百度。

这不是"烧钱补贴"——DeepSeek 没有那个资本。
更合理的解释:

  GPU 集群跑 MoE decode:
    Tensor Core 利用率 ~2-5%
    花了 $30K/H100 买了 990 TFLOPS → 只用了 ~20 TFLOPS
    但电力/机柜/运维是按 700W TDP 付的

  → 真实硬件成本被 98% 的闲置算力严重稀释
  → 闲置算力沉没成本为零 → 降价促销的边际成本极低
  → 只要收入覆盖电费+带宽就有利可图 (短期内)

反过来看 FPGA 的逻辑:

  FPGA 集群跑同样的 MoE decode:
    DSP 利用率 ~50% (加权平均, 含 SRAM)
    花了 $3K/芯片 买了 8.44 TMACs/s → 用了 ~4.2 TMACs/s
    电力按 ~75W TDP 付

  → 没有大笔闲置算力需要"消化"
  → $20/百万 token 是实打实的硬件成本, 没有 GPU 那种"虚胖"
  → 量产 10 套降到 $7-9/百万 token 后, 已经和 DeepSeek 输出定价的 $1.10 在同一量级

关键洞察:
  DeepSeek 的低价不是因为 GPU 效率高, 恰恰是因为 GPU 效率太低——
  闲置算力太多, 不打折也是浪费。
  
  FPGA 方案不需要靠"闲置算力红利"来压价,
  它的成本优势来自架构匹配本身: 每一颗 DSP 都在干活。
```

### 其他局限

```
✗ Prefill (大 batch): GPU 完胜 (但 CPU 已追近, 2026 更新)
  → Tensor Core 在 batch 下利用率高
  → 2026: Dual Xeon GNR / EPYC Turin P=128 chunk TTFT ~400ms
    (SPR 的 6x 提升), CPU prefill 可覆盖 80% 商业场景
  → 长 prompt: FPGA chunked prefill 兜底 (首 token ~125ms)
  → 极致低延迟: 加 GPU (可选, 管制风险)
  → 详见 fpga_inference_cluster_proposal.md 4.8.6

✗ Training: FPGA 根本不能做
  → 这是 GPU 的绝对领地

✗ 非 MLA/非 fp4 模型:
  → 如果没有 fp4 + MLA 这两个特殊需求
  → GPU 就是更好的方案

✗ 推理吞吐 > 10,000 tok/s (单套):
  → 这会需要 batch, GPU 在 batch 下更强

并发 1-2 的工程应对:
  → 部署多套集群 (每套 ¥1.9M 量产价)
  → 每套独立服务不同客户/地域
  → FPGA 的并发靠"套数"而不是"batch"来扩展
  → 这恰好匹配私有部署场景: 每个客户 1-2 套, 独立隔离
```

所以 FPGA 的最优区间非常明确:

```
  DeepSeek 系列 (fp4 + MLA) × Decode × 私有部署 × 中低并发 (1-2/session)
  
  在这个区间内, FPGA 是客观上的最优架构。
  
  出了这个区间:
    高并发公有云 → GPU 更优
    大 batch prefill → GPU 更优  
    训练 → GPU 唯一选择
    非 MLA/fp4 模型 → GPU 更简单
```

---

## 八、一句话总结

> DeepSeek V4 Pro 选择了 fp4 + MLA。fp4 是国产 GPU 的集体盲区（Ascend/寒武纪/海光均不支持）, MLA 的硬件加速是 GPU kernel launch 模型无法高效覆盖的。而 FPGA 恰好在这两个维度上天然占优。NVIDIA B200 虽已支持 fp4，但出口管制使其对中国市场不可用。DeepSeek 用算法创新绕开了 GPU 生态的锁定效应，而 FPGA 是中国市场可获取的、承接这种算法创新的最优硬件载体。
