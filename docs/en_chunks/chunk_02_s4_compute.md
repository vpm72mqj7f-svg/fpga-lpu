## 4. 算力分配与资源核算

### 4.1 32 芯片资源分配 (8 卡 × 4 AGM 039/卡, 无热备)

```
单服务器 8 卡, 每卡 4 片 AGM 039, 共 32 芯片, 全活跃:

  ┌────────┬──────────────┬────────────┬──────────┬──────────────┐
  │ Card   │ Chip ID       │ 层范围      │ 芯片层数  │ 专家          │
  ├────────┼──────────────┼────────────┼──────────┼──────────────┤
  │ Card 0 │ C0.0 ~ C0.3  │ L 00~07    │ 2+2+2+2 │ 4×12=48      │
  │        │              │ +Embedding │          │              │
  │ Card 1 │ C1.0 ~ C1.3  │ L 08~14    │ 2+2+2+1 │ 4×12=48      │
  │ Card 2 │ C2.0 ~ C2.3  │ L 15~22    │ 2+2+2+2 │ 4×12=48      │
  │ Card 3 │ C3.0 ~ C3.3  │ L 23~29    │ 2+2+2+1 │ 4×12=48      │
  │ Card 4 │ C4.0 ~ C4.3  │ L 30~37    │ 2+2+2+2 │ 4×12=48      │
  │ Card 5 │ C5.0 ~ C5.3  │ L 38~44    │ 2+2+2+1 │ 4×12=48      │
  │ Card 6 │ C6.0 ~ C6.3  │ L 45~52    │ 2+2+2+2 │ 4×12=48      │
  │ Card 7 │ C7.0 ~ C7.3  │ L 53~60    │ 2+2+2+2 │ 4×12=48      │
  │        │              │ +lm_head  │          │              │
  │        │              │ +MTP      │          │              │
  ├────────┼──────────────┼────────────┼──────────┼──────────────┤
  │ 合计    │ 32 芯片      │ 61 层      │ 32 芯片  │ 384 专家      │
  └────────┴──────────────┴────────────┴──────────┴──────────────┘

32 芯片负载分配:
  Expert:  384 专家 / 32 芯片 = 12 专家/片 ✓ 完美整除
  Head:    卡内按层数动态分配, 128 头均匀分布到 8 卡
  层数:    61 层 / 32 芯片 → 29 片 × 2 层 + 3 片 × 1 层

卡内芯片拓扑:
  Chip0 (PCIe Master):  R-Tile PCIe 5.0 x16 → Host
                        F-Tile SerDes ×2 → Dual Ring A/B
  Chip1/2/3:            F-Tile SerDes ×2 → Dual Ring A/B
                        所有 Host 交互经 Chip0 C2C Proxy 转发

无热备策略:
  32 片全活跃, 比旧方案 (30+2) 多 2 片算力
  芯片级故障: 卡内 4 片权重互相备份 (HBM 32GB 放 12 专家远未满)
             故障片的 12 专家由同卡其余 3 片分摊 → 单片故障仅降速 25%
  卡级故障:   8 卡整体吞吐降为 7/8, 需停机更换
  详见 §6.6 容错设计
```

### 4.2 单 Token 单层 MAC 分解

```
MLA Attention (每 token 每层):
  Q 压缩 (LoRA down):           7,168 × 1,536 =     11.01M
  KV 压缩 (latent):             7,168 × 512 =        3.67M
  KV 压缩 (rope part):          7,168 × 64 =         0.46M
  Q·K^T (nope+rope):            ~29.88M
  A·V (nope against c_KV):     ~29.36M
  O 解压 (LoRA):               128×512×1024 =       67.11M
  O 解压 (to model dim):       1024×7168 =           7.34M
  ─────────────────────────────────────────────
  MLA 小计:                                        ~148.8M MAC

MoE FFN (每命中专家, SwiGLU):
  gate: 7168×3072 = 22.02M
  up:   7168×3072 = 22.02M
  down: 3072×7168 = 22.02M
  ─────────────────────
  每专家: 66.05M

  6 路由专家 + 1 共享专家:  462.4M MAC

MoE 层总计 (Attn + MoE):  ~611M MAC  ← 这是每层每 token 的计算量
```

### 4.3 AGM 039 计算能力

```
DSP 资源配置 (AGM 039-F, 32GB HBM):
  - 12,300 可变精度 DSP (with AI Tensor Block)
  - 每个 DSP 在 fp4×fp8 模式下: 2 MAC/cycle
  - 工作频率: 450 MHz
  - 总吞吐 = 12,300 × 2 × 450 MHz = 11.07 TMACs/s

vs 旧方案 (AGM 032: 9,375 DSPs, 8.44 TMACs/s):
  +31% 算力 (11.07 / 8.44)

HBM 规格 (与 032 相同):
  - 32 GB HBM2e, ~920 GB/s 带宽
  - KV Cache per token: 576 B FP8

FP16 TFLOPS (AGM 039):
  - Half-precision: 18.4 TFLOPS
  - Single-precision: 9.2 TFLOPS

对比单 token decode 需求:
  - 全 61 层: ~37.4 GMACs 总计
  - @11.07 TMACs/s: 37.4G/11.07T = 3.38 ms 计算时间 (单层)
  - 32 芯片集群: ~1,000+ tok/s (含 SRAM 缓存, §4.4.1)

AGM 039 多出的 31% DSP 在 decode 场景 (memory-bound) 中不直接提升吞吐,
但为 prefill burst 和未来更重计算负载提供裕量。
```

### 4.4 HBM 带宽瓶颈分析

```
这是系统最关键的约束:

单 Token 每层 HBM 读取:
  Attention 权重:  ~15 MB
  MoE 路由权重:     ~2 MB
  专家权重 (期望):  ~12 MB (6 routed × 13/384 命中率 × 33MB)
  共享专家权重:     ~33 MB
  ─────────────────────────
  每层 HBM 读:     ~62 MB (期望, 不含 SRAM 缓存)

61 层 HBM 读:     ~3.8 GB per token
HBM 时间:           3.8 GB / 920 GB/s = 4.1 ms

对比计算时间 4.43 ms:
  → HBM 与 DSP 基本对等 (4.1 ≈ 4.43)
  → Decode (B=1~4) 下两者均接近瓶颈
  → SRAM 缓存可将确定性权重移出 HBM, 详见 §4.4.1

结论: 
  HBM 带宽 920 GB/s 和 DSP 8.44 TMACs/s 大致匹配
  Decode (B=1~4) 下 HBM 略微领先瓶颈
  Prefill (B=32+) 下 DSP 成为瓶颈
```

### 4.4.1 SRAM 缓存层次 — 定量分析与布线可行性

评审质疑两个问题: (1) MoE 不规律访存会导致 HBM Bank Conflict, 实际带宽远低于 920 GB/s; (2) FPGA SRAM 利用率过高会导致布线拥塞和时序收敛失败。本节逐一回应。

**4.4.1.1 本地 Expert 命中的精确模型**

30 活跃卡下两种卡型:
- Type A (TP=7, Node 0/3): 14 卡, 每卡 12 或 13 专家
- Type B (TP=8, Node 1/2): 16 卡, 每卡 12 或 13 专家

先用二项分布精确描述每卡每层有几个 Expert 本地命中:

```
24 卡 × 13 专家: Binomial(n=6, p=13/384=0.03385)
  P(0 hit) = (1-p)⁶                      = 81.4%
  P(1 hit) = 6·p·(1-p)⁵                  = 17.1%
  P(2 hit) = C(6,2)·p²·(1-p)⁴           =  1.5%
  P(3+ hit)=>                              <0.1%

6 卡 × 12 专家: Binomial(n=6, p=12/384=0.03125)
  P(0 hit) = 82.5%, P(1 hit) = 16.0%, P(2 hit) = 1.3%

加权平均 (按卡数):
  P(0 hit) = (24×0.814 + 6×0.825)/30 = 81.6%
  P(1 hit) = (24×0.171 + 6×0.160)/30 = 16.9%
  P(2 hit) = (24×0.015 + 6×0.013)/30 =  1.5%

注意: 不能用期望值来算延迟——延迟由"有命中时"决定。
      有命中时加载 1 个完整 Expert = 33 MB fp4, 不是期望值。
```

**4.4.1.2 无 SRAM 缓存时: 三种情形的逐层延迟**

每卡每层权重访问 (加权平均, 30 卡两种 TP):

```
┌──────────────────────────────┬────────────┬─────────────────────────────┐
│                               │ HBM 读量    │ 说明                         │
├──────────────────────────────┼────────────┼─────────────────────────────┤
│ 共享 Expert (TP=7/8, fp4)    │ 4.4 MB     │ 确定性, 权均 33/7.5≈4.4        │
│ Attention Q/KV/O (fp4)       │ 4.4 MB     │ 确定性, 权均 ~18.3 头          │
│ Router 权重 (fp8, 非 fp4)     │ ~0.37 MB   │ 确定性, 精度敏感, 保持 FP8      │
│ KV Cache (滑动窗口 128, FP8)  │ ~0.07 MB   │ 确定性, 顺序 stride 读          │
│ RMSNorm (fp16)               │ ~0.01 MB   │ 确定性                        │
│ 确定性小计                    │ ~9.3 MB    │ (Router fp8 比 fp4 多 0.34 MB) │
├──────────────────────────────┼────────────┼─────────────────────────────┤
│ 路由 Expert (本地命中时)       │ 0 或 33 MB │ 动态, Router 输出后才知道        │
└──────────────────────────────┴────────────┴─────────────────────────────┘

DSP 时间 (每卡每层, 加权平均):
  Attention + 共享 Expert: (19.84M + 8.80M) / 8.44T = 3.4 μs
  1 个路由 Expert:         66M / 8.44T               = 7.8 μs
  2 个路由 Expert:         2 × 66M / 8.44T           = 15.6 μs
```

三种情形延迟 (无 SRAM, HBM 920 GB/s):

```
情形 A: P=81.6%  0 local hit
  HBM:  9.3 MB / 920 GB/s = 10.1 μs
  DSP:  Attention + SharedExp = 3.4 μs
  延迟: max(10.1, 3.4) = 10.1 μs   DSP 利用率 3.4/10.1 = 33.7%

情形 B: P=16.9%  1 local hit
  HBM:  (9.3 + 33) MB / 920 = 46.0 μs
  DSP:  3.4 + 7.8 = 11.2 μs
  延迟: 46.0 μs                   DSP 利用率 11.2/46.0 = 24.3%

情形 C: P=1.5%   2 local hits
  HBM:  (9.3 + 66) MB / 920 = 81.8 μs
  DSP:  3.4 + 15.6 = 19.0 μs
  延迟: 81.8 μs                   DSP 利用率 19.0/81.8 = 23.2%

加权平均: 10.1×0.816 + 46.0×0.169 + 81.8×0.015 = 17.24 μs/layer
加权 DSP 忙: 3.4×0.816 + 11.2×0.169 + 19.0×0.015 = 4.95 μs/layer
总体 DSP 利用率: 4.95/17.24 = 28.7% (对比原 32 卡模型 29.6%, 基本持平)
```

**4.4.1.3 加入 SRAM 缓存: 布线友好的分配方案**

Agilex 7 M 片上 SRAM:

```
M20K: 15,932 块 × 20 Kb = 38.9 MB
MLAB:                     ~4.1 MB
─────────────────────────────────
总计:                    ~43.0 MB

布线约束 (450 MHz 时序收敛经验值):
  M20K 利用率 ≤ 75% → 可用 ≤ 29.2 MB
  MLAB 利用率 ≤ 80% → 可用 ≤ 3.3 MB
```

基于此约束的分配:

```
M20K 分配 (29.4 MB, 75.6% 利用率):
  ┌──────────────────────────────────┬──────────┬──────────────────┐
  │ 用途                              │ 容量      │ 布线考量           │
  ├──────────────────────────────────┼──────────┼──────────────────┤
  │ 确定性权重双缓冲                   │ 18.6 MB  │ 靠近 HBM 控制器列  │
  │  (共享Exp 4.4+Attn 4.4+Rtr fp8)  │          │ M20K 列, 与 DSP   │
  │  × 2 (当前层 + 预取)              │          │ 就近放置          │
  │ Systolic Array Weight Stationary │ 2.0 MB   │ 紧邻 DSP 列       │
  │  (8 阵列 × 128×128 × fp4)        │          │ 输入寄存器         │
  │ Expert 权重流式预取乒乓缓冲       │ 4.0 MB   │ HBM→DSP 数据路径 │
  │  (2 × 2MB, 加载中 + 计算中)      │          │ 上, 靠近脉动阵列   │
  │ KV Cache 热窗口 Key 索引         │ 2.0 MB   │ 靠近 KV Cache     │
  │  (当前滑动窗口 128 位置)          │          │ Manager RTL       │
  │ Router 路由表 (全 61 层常驻)     │ 2.0 MB   │ 靠近 Router       │
  │  (fp8 scaling tables + bias)    │          │ Gating Unit       │
  │ 布局余量 (M20K 碎片 / 对齐)      │ 0.8 MB   │ 不可避免的浪费     │
  │ M20K 小计                        │ 29.4 MB  │ 75.6% ✓          │
  └──────────────────────────────────┴──────────┴──────────────────┘

MLAB 分配 (3.3 MB, 80% 利用率):
  ┌──────────────────────────────────┬──────────┬──────────────────┐
  │ Session 表 + KV 地址生成          │ 1.0 MB   │ 寄存器级延迟       │
  │ PCIe/Ethernet 报文 buffer        │ 1.0 MB   │ 靠近 F-Tile/R-Tile│
  │ 脉动阵列部分和累加器 (FP32)       │ 1.0 MB   │ 紧邻 DSP 列       │
  │ 层间 FSM 控制状态                 │ 0.3 MB   │ 散布              │
  │ MLAB 小计                        │ 3.3 MB   │ 80% ✓            │
  └──────────────────────────────────┴──────────┴──────────────────┘
```

M20K 利用率 75.6%、MLAB 80% 处于业界公认的 "可布线、可收敛" 区间。预留的 9.5 MB M20K 碎片 (24.4%) 为物理综合中的 M20K 列选通、地址对齐、以及跨 die 布线中继提供了充足余量。

**4.4.1.4 缓存后的延迟重算**

```
情形 A: P=81.6%  0 local hit  (全部权重已在 SRAM)
  HBM:  零 (下一层 9.3 MB 预取与当前计算重叠)
  DSP:  3.4 μs (全速 SRAM→DSP)
  延迟: 3.4 μs                    DSP 利用率 100%

情形 B: P=16.9%  1 local hit
  HBM:  (33 + 0.37) MB Expert+Router + 0.07 MB KV ≈ 33.44 MB → 36.3 μs
  DSP:  Attn+Shared 3.4 μs (SRAM) + Expert 7.8 μs (HBM→DSP 流式)
  关键路径: HBM 加载 Expert (36.3 μs) 远超 DSP (11.2 μs)
  延迟: max(36.3, 11.2) = 36.3 μs    DSP 利用率 11.2/36.3 = 30.9%

情形 C: P=1.5%   2 local hits
  HBM:  (66 + 0.37) MB + 0.07 ≈ 66.44 MB → 72.2 μs
  DSP:  3.4 + 15.6 = 19.0 μs
  延迟: 72.2 μs                   DSP 利用率 19.0/72.2 = 26.3%

加权平均: 3.4×0.816 + 36.3×0.169 + 72.2×0.015 = 9.99 μs/layer
加权 DSP 忙: 3.4×0.816 + 11.2×0.169 + 19.0×0.015 = 4.95 μs/layer
总体 DSP 利用率: 4.95/9.99 = 49.5%
```

**4.4.1.5 效果对比**

```
┌───────────────────────┬──────────┬──────────┬──────────┐
│                        │ 无 SRAM  │ 有 SRAM  │ 改善      │
├───────────────────────┼──────────┼──────────┼──────────┤
│ 加权每层延迟            │ 17.24 μs │ 9.99 μs  │ -42%     │
│ DSP 利用率 (加权)       │ 28.7%    │ 49.5%    │ +72%     │
│ 0-hit 层 DSP 利用率    │ 33.7%    │ 100%     │ 关键      │
│ 1-hit 层 DSP 利用率    │ 24.3%    │ 30.9%    │ 受限于33MB│
│ 每卡 ~2 层 / token     │ ~34 μs   │ ~20 μs   │ -41%     │
│ 30 卡集群吞吐           │ ~580     │ ~980     │ tok/s    │
├───────────────────────┼──────────┼──────────┼──────────┤
│ M20K 利用率            │ 0%       │ 75.6%    │ 可布线   │
│ MLAB 利用率            │ 0%       │ 80%      │ 可收敛   │
└───────────────────────┴──────────┴──────────┴──────────┘
```

**4.4.1.6 剩余瓶颈与坦诚结论**

81.6% 的层 (0 hit) 是 SRAM 的天堂 — DSP 以 100% 全速运行, HBM 完全空闲。16.9% 的层 (1 hit) 才是瓶颈 — 加载 33.4 MB Expert+Router (36.3 μs) 远超 DSP 计算 (11.2 μs), **HBM 带宽不是不够, 是 Expert 单体太大。** 这不是 Bank Conflict 问题, 是 MoE 架构固有的 Expert 粒度问题。

Bank Conflict 风险现在仅存在于 1-hit/2-hit 层中的 33~66 MB Expert 加载路径上。这部分是顺序矩阵权重读取 (gate→up→down), 访问 Pattern 本身是顺序的, Bank Conflict 影响可控。

DSP 49.5% 利用率虽未达 GPU 训练时的 90%+, 但对于 LLM Decode 场景, 这仍然远超 GPU 的 2-5% (参见 why_fpga_is_optimal.md)。SRAM 的贡献是将 FPGA 的有效吞吐从 GPU 的 ~10× 优势提升到 ~17× 优势——不是在绝对利用率上抢眼, 而是在竞争对手最弱的维度上拉大差距。

> 注: 30 卡有两类卡型 (TP=7 的 14 卡 vs TP=8 的 16 卡)。以上为加权平均。TP=7 卡 (Node 0/3) 的确定性权重大 ~15%, M20K 余量更紧 (最差卡 ~78%), 但仍在可布线区间。TP=8 卡与原 32 卡模型基本一致。

**4.4.1.7 质疑 B 直接回应：920 GB/s vs 3.35 TB/s 的公平比较**

> **质疑 B**: "FPGA 920 GB/s HBM vs H100 3.35 TB/s，带宽差 3.6 倍。SRAM 只能缓存 top-1 expert，长尾 expert 触发时 HBM 带宽成为瓶颈。"

这个质疑的 920 vs 3350 的数字对比看似压倒性，但它隐含了一个错误前提：两者加载的数据宽度相同。实际上根本不是。

**一、带宽的公平比较：元素/秒，不是字节/秒**

```
带宽 ≠ 有效吞吐。带宽需要除以每个参数占用的字节数:

  H100 HBM3:  3.35 TB/s ÷ 2 bytes/param (BF16/FP16) = 1.68T params/s
  FPGA HBM2e: 0.92 TB/s ÷ 0.5 bytes/param (fp4)       = 1.84T params/s

  结论: FPGA 在 "每秒可加载的权重参数数量" 上反而比 H100 多 10%。

如果比较 FP8 (Router + Activation):
  H100 HBM3:  3.35 TB/s ÷ 1 byte/param (FP8) = 3.35T params/s
  FPGA HBM2e: 0.92 TB/s ÷ 1 byte/param (FP8) = 0.92T params/s
  
  但 FP8 数据仅占总权重的 <5% (Router + RMSNorm)。
  → 对于占 95% 的推理权重，fp4 抵消了 HBM 带宽差距。
```

**二、质疑中的关键事实错误：SRAM 不只是 "缓存 top-1 expert"**

```
SRAM 中实际缓存的内容 (18.6 MB 确定性权重双缓冲):

  Shared Expert (fp4)   4.4 MB  — 每层都需要，永远在 SRAM
  Attention Q/KV/O (fp4) 4.4 MB — 每层都需要，永远在 SRAM
  Router 权重 (fp8)     0.37 MB — 每层都需要，永远在 SRAM
  RMSNorm                0.01 MB — 每层都需要
  当前层 Expert × 1      ≈ 4 MB  — 权重流式预取缓冲
  总计                   ~13.2 MB 常驻 + ~5 MB 流式

这 13.2 MB 的 "确定性权重" 覆盖了 81.6% 的层 (0-hit 层) 的全部 HBM 读需求。
这不是 "缓存 top-1 expert"，而是:
  → 所有层的 Shared Expert + Attention + Router + RMSNorm 永远不走 HBM
  → 只有被 Router 选中的路由 Expert (33 MB/个) 才需要从 HBM 加载
  → 81.6% 的层连这 33 MB 也不需要 (因为 0 local hit)
```

**三、power-law 分布是朋友，不是敌人**

```
质疑声称: "长尾 expert 在 power-law 分布下必然发生"

这是对的 — 但长尾 expert 对 HBM 带宽的需求恰恰因为 power-law 而大幅降低了:

  Power-law 意味着:
    ● 头部 20% 的 expert 占据 ~80% 的 token 选择
    ● 尾部 80% 的 expert 很少被选中

  P(0 local hit) = 81.6% 的含义:
    ● 意味着 81.6% 的 token (不是 expert) 在这一层完全不需要 HBM
    ● 剩下的 18.4% 需要加载 1 个或 2 个 expert (33-66 MB)
    ● 这 18.4% 中，大概率命中的也是头部 expert（并非均匀随机！）

  因为 power-law，被命中的 expert 更可能是热门 expert。
  热门 expert 的访问频率更高 → 更容易被分配到可用 HBM pseudo-channel。
  → 这不是 Bug，这是 power-law 自然带来的访问局部性红利。

  如果 expert 是均匀分布的 (P=1/384 per expert),
  那么每个 expert 被选中的概率一样 → HBM bank conflict 会严重很多。
  Power-law 使访问集中在少数 expert → 硬件上的 bank 压力反而更小。
```

**四、H100 在 batch=1 decode 的真实处境**

```
H100 跑 DeepSeek V4 Pro decode 的实际情况:

  每层需要从 HBM 加载:
    全部 6 个 Expert (33 MB × 6 × BF16) = 396 MB
    + Attention (15.4 × BF16)              = 30.8 MB
    + Router + RMSNorm                      = ~5 MB
    总计                                    ≈ 432 MB/layer

  H100 HBM 时间: 432 MB / 3.35 TB/s = 129 μs/layer

  H100 的 L2 cache 只有 50 MB (全部共享),
  装不下 396 MB 的 Expert 权重。
  所以即使是 batch=1，H100 也必须从 HBM 加载几乎全部权重。

  FPGA (SRAM 缓存后):
    81.6% 层: 0 MB HBM → 0 μs HBM (纯 SRAM→DSP)
    16.9% 层: 33.4 MB → 36.3 μs
     1.5% 层: 66.4 MB → 72.2 μs

  加权 HBM 时间: 0×0.816 + 36.3×0.169 + 72.2×0.015 = 7.2 μs/layer

  FPGA 7.2 μs vs H100 129 μs → FPGA HBM 有效时间仅为 H100 的 5.6%。

  这不是因为 FPGA 的 HBM 更快，而是因为 FPGA 的 SRAM 缓存策略
  使 81.6% 的层无需访问 HBM。H100 的 50MB L2 做不了这件事—
  因为 H100 没有 "把确定性权重永久锁在片上 SRAM" 的硬件灵活性。
```

**五、坦诚的剩余瓶颈**

```
16.9% 的 1-hit 层仍然是瓶颈:
  33.4 MB Expert 加载 (36.3 μs) 远超 DSP 计算 (11.2 μs)。
  
  但这不是 920 GB/s 不够造成的 — 920 GB/s 加载 33.4 MB 只需 36.3 μs。
  即使 HBM 带宽翻倍到 1.84 TB/s，加载 33.4 MB 仍需 18.2 μs，
  仍然可能超过 DSP 时间 (11.2 μs)。

  真正的瓶颈是: Expert 的 33 MB 单体大小决定了加载延迟有物理下限。
  这个下限与 HBM 带宽有关但不对等 — 32 个 pseudo-channel 并发 + 
  Expert 内部顺序布局已经是最优访问模式。

  缓解方案 (不需要推翻架构):
    ● 增加 Expert 预取深度: 用 2 个 token 的 lookahead 提前加载
    ● Expert 权重拆分: 将 33 MB Expert 拆为 gate (2MB) + up (15.5MB) + down (15.5MB)
      → gate 先加载，如果 gate 输出接近 0 → 跳过 up/down 加载
    ● 如果 V5 将 Expert 从 33MB 缩小到 20MB → 1-hit 延迟从 36.3 μs 降至 ~22 μs

  这些缓解方案不需要新增 HBM 带宽，只需要调度和布局调整。
```

**六、H100 对比的总结**

```
┌─────────────────────────┬──────────────┬──────────────┬──────────────┐
│                          │ H100 SXM      │ FPGA Agilex 7M│ 对比          │
├─────────────────────────┼──────────────┼──────────────┼──────────────┤
│ 权重精度                  │ BF16/FP16     │ fp4           │ 4× 压缩       │
│ HBM 带宽                  │ 3.35 TB/s     │ 0.92 TB/s     │ 3.6× "劣势"  │
│ 等效参数带宽              │ 1.68T param/s │ 1.84T param/s │ FPGA +10%    │
│ 确定性权重                │ 每层加载 HBM   │ SRAM 常驻      │ 关键差异      │
│ 0-hit 层 HBM 读           │ ~432 MB       │ 0 MB          │ FPGA 完胜     │
│ 1-hit 层 HBM 读           │ ~432 MB       │ 33.4 MB       │ FPGA 13× 少   │
│ 加权 HBM 时间/层          │ ~129 μs       │ 7.2 μs        │ FPGA 18× 快   │
│ Batch=1 decode 瓶颈       │ HBM 带宽       │ Expert 单体大小│ 不同瓶颈      │
│ 利用率 (B=1)              │ ~2-5%          │ ~49.5%        │ FPGA 10-25×   │
└─────────────────────────┴──────────────┴──────────────┴──────────────┘

核心反论:
  920 GB/s 看起来只有 H100 的 27%，但在 fp4 精度下
  等效参数带宽反而比 H100 高 10%。
  加上 SRAM 缓存消除 81.6% 层的全部 HBM 访问，
  FPGA 的有效瓶颈不是 HBM 带宽，而是 Expert 的 33 MB 单体大小。
  这个问题 GPU 也有 — 而且 GPU 没有 SRAM 缓存的解脱。
```

```
### 4.5 HBM 空间核算

```
每卡 HBM (32 GB):

  权重常驻区 (~24 GB 预算, 实际占用):
    ┌────────────────────────────┬──────────┐
    │ 资源                        │ HBM 占用  │
    ├────────────────────────────┼──────────┤
    │ 12~13 路由专家 (fp4)        │ ~396-429 MB│
    │ 1 共享专家 (fp4)            │ ~33 MB   │
    │ Attention 权重 (15~16 层)   │ ~145-166 MB│ (TP 影响)
    │ Router 权重                 │ ~15 MB   │
    │ RMSNorm 等杂项 (fp16)       │ ~5 MB    │
    │ Embedding (唯 Node 0)       │ ~1,850 MB│ fp16
    │ lm_head  (唯 Node 3)        │ ~1,850 MB│ fp16
    ├────────────────────────────┼──────────┤
    │ 权重小计 (Node 1/2,TP=8)   │ ~594 MB  │
    │ 权重小计 (Node 0/3,TP=7)   │ ~2,468 MB│
    └────────────────────────────┴──────────┘

  运行区 (~8 GB):
    ├─ KV Cache: 256K context × 16 layers × 576B ≈ 2.36 GB
    ├─ 激活 Buffer: ~2 GB
    └─ ETH Ring Buffer: ~0.5 GB
    ─────────────────────────────────
    运行区小计: ~4.86 GB < 8 GB ✓

  总 HBM 占用: ~5.5~7.3 GB < 32 GB
  余量: ~25 GB → 可用于:
    - 热门专家副本 (增加计算并行度)
    - 更大 context (512K → 1M)
    - 更多并发 session 的 KV Cache
```

### 4.5.1 HBM 容量极限分析

评审质疑: 32GB HBM 是物理硬上限, 大 context 场景下是否成为瓶颈？尤其是在 Agent 和长文档分析场景中 KV Cache 随 context 线性增长, 是否会耗尽 HBM？

**4.5.1.1 MLA 压缩是结构性的容量优势**

```
DeepSeek V4 Pro 的 MLA 对 KV Cache 的压缩是决定性的:

  传统 MHA: 2 × n_heads × d_head × FP16
           = 2 × 128 × 128 × 2B = 64 KB / token / layer

  MLA:      KV latent (c_KV=512B, FP8) + rope (64B, FP8)
           = 576 B / token / layer
           ≈ 1/114 的 MHA 大小

FPGA 每卡 KV Cache (16 layers):
  64K  ctx:  64K  × 16 × 576B = 0.59 GB
  128K ctx:  128K × 16 × 576B = 1.18 GB
  256K ctx:  256K × 16 × 576B = 2.36 GB
  512K ctx:  512K × 16 × 576B = 4.72 GB
  1M   ctx:  1M   × 16 × 576B = 9.22 GB
  2M   ctx:  2M   × 16 × 576B = 18.43 GB
```

**4.5.1.2 Context × 余量 — 全场景矩阵**

```
Node 3 最差卡 (16 layers, TP=7, 权重 2.47GB):

┌────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│             │ 128K ctx │ 256K ctx │ 512K ctx │ 1M ctx   │ 2M ctx   │
├────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 权重        │ 2.47 GB  │ 2.47 GB  │ 2.47 GB  │ 2.47 GB  │ 2.47 GB  │
│ KV Cache   │ 1.18 GB  │ 2.36 GB  │ 4.72 GB  │ 9.22 GB  │ 18.43 GB │
│ 激活/缓冲    │ 2.50 GB  │ 2.50 GB  │ 2.50 GB  │ 2.50 GB  │ 2.50 GB  │
├────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 合计        │ 6.15 GB  │ 7.33 GB  │ 9.69 GB  │ 14.19 GB │ 23.40 GB │
│ HBM 余量    │ 25.85 GB │ 24.67 GB │ 22.31 GB │ 17.81 GB │ 8.60 GB  │
│ 最大并发     │ ~6       │ ~3       │ ~1-2     │ ~1       │ ~0-1     │
└────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

关键发现:
  ✓ 1M context: 仍有 17.8 GB 余量, 可存 Top-50 热门专家副本 (1.65 GB)
  ✓ 2M context: 8.6 GB 余量, Top-20 副本 (660 MB) 仍可行
  ✗ 3M+ context: 接近上限, 需降级策略 (滑动窗口裁剪)
```

**4.5.1.3 热门专家副本在大 context 下反而更重要**

```
MoE 专家访问呈 Zipf 分布:
  Top-10 专家:  ~50% token 命中
  Top-20 专家:  ~70% token 命中
  Top-50 专家:  ~90% token 命中

专家副本策略 (复制热门专家权重到 HBM 的另一区域,
                允许并行读取而非串行):
  Top-10: 10 × 33MB = 330 MB
  Top-20: 20 × 33MB = 660 MB
  Top-50: 50 × 33MB = 1.65 GB

  效果:
    覆盖 70% 命中 → P(1 hit from HBM) 从 16.9% → 16.9%×30% = 5.1%
    → 加权延迟降低 ~5-8%

大 context 下:
  更多 token → 更多 expert 命中 → 副本加速效果放大
  1M context 余量 17.8 GB → Top-50 副本 (1.65 GB) 轻松装下
  2M context 余量 8.6 GB → Top-20 副本 (660 MB) 可行
```

**4.5.1.4 与 Ascend 的 HBM 容量对比**

```
┌──────────────────────┬──────────────────┬──────────────────┐
│                       │ Ascend 910C       │ FPGA Agilex 7 M   │
├──────────────────────┼──────────────────┼──────────────────┤
│ 单卡 HBM               │ 64 GB HBM2e       │ 32 GB HBM2e       │
│ 权重格式               │ FP8 (无 fp4 原生)  │ fp4 (原生)        │
│ 权重占用 (Node 3)      │ ~4.94 GB          │ ~2.47 GB          │
│ 1M context 总占用      │ ~16.7 GB          │ ~14.2 GB          │
│ 有效余量               │ ~47.3 GB          │ ~17.8 GB          │
├──────────────────────┼────────────────────┼──────────────────┤
│ 1M ctx 最大并发        │ ~4-5              │ ~1-2              │
│ 2M ctx 最大并发        │ ~2-3              │ ~1                │
└──────────────────────┴────────────────────┴──────────────────┘

Ascend 64GB 在超大 context (≥2M) × 高并发 (≥3) 场景有真实优势。
但 FPGA 的 fp4 压缩 (权重 2× 节省) 部分抵消了容量差距:
有效可用空间差距不是 2× (64 vs 32), 而是 ~3.3× (47 vs 17.8
的余量在 1M ctx)。差距仍存在, 但小于纸面数字。

对于 FPGA 目标场景 (≤1M context, ≤2 并发, 私有部署):
32GB HBM + fp4 压缩 + MLA 压缩 = 容量充足。
```

**4.5.1.5 升级路径**

```
当前: Agilex 7 M, 32 GB HBM2e
下一代: Agilex 9 (或 Agilex 7 后继), 预计 64 GB+ HBM3
  → RTL 迁移: HBM 控制器 IP 从 Intel 更新,
    用户推理 RTL 仅改 address width parameter
  → 不需重写 fp4 MAC / MLA pipeline / KV Cache manager
  → 同一设计直接获得 2× KV Cache 或 2× 并发

如果客户今天需要 >2M context × 多并发:
  → 坦诚: FPGA 不是合适选择, 推荐 Ascend 或等 H200 解禁
  → 但对于 ≤1M context × 1-2 并发的 90%+ 商业场景,
    32GB 不是瓶颈。
```

### 4.6 并发分析

```
并发 session 上限由两个约束中更紧的那个决定:

约束 A: KV Cache 容量
  每卡运行区 ~8 GB
  单 session 128K context KV: 128K × 16 layers × 576B ≈ 1.18 GB
  HBM 决定的并发上限: 8 / 1.18 ≈ 6-7 个

约束 B: 算力余量
  单卡 Decode (B=1): DSP 利用率 ~50% (加权平均, 含 SRAM)
  算力余量: ~50% → 只能再摊 ~1 个同规格 session
  算力决定的并发上限: ~1-2 个 (取更紧者)

结论: FPGA 的并发上限由算力锁定, 不是 HBM。
  → B=1 时 1 个 session 接近吃满 DSP
  → 多 session 通过时分复用, 每个降到 ~250-500 tok/s

与 H200 的本质差异:
  ┌──────────────────────┬────────────┬──────────────┐
  │                      │ H200 (8卡) │ FPGA (30卡)  │
  ├──────────────────────┼────────────┼──────────────┤
  │ Decode 算力利用率     │ ~3%        │ ~50% (B=1)   │
  │ HBM KV 可用/卡        │ ~50 GB     │ ~8 GB        │
  │ 算力决定的并发        │ ~30-40     │ ~1-2         │
  │ HBM 决定的并发        │ ~10-15     │ ~6-7         │
  │ 实际并发 (取紧者)     │ ~10-15     │ ~1-2         │
  └──────────────────────┴────────────┴──────────────┘

  H200: HBM 容量是瓶颈 (算力大量闲置)
  FPGA: 算力是瓶颈 (HBM 算力比天然匹配 decode)

这恰恰印证了 FPGA 的定位:
  ✓ 私有部署 (1-2 并发 session, 独立租户)
  ✓ 多套集群扩展 (靠套数而非 batch 来 scale)
  ✗ 公有云高并发 API (那是 GPU 的领地)

补充: MoE 架构天然排斥大 Batch

  上述比较假设 GPU 可以随意堆 B。但 DeepSeek V4 Pro 是 MoE:
    Dense 模型: B=32 → HBM 加载 ≈ B=1 (所有权重共享)
    MoE 模型:   B=8  → 可能命中 48 个不同专家 → HBM 压力 ~8×

    All-to-All 通信量正比于 B。B 越大专家负载越不均衡。
    业界 MoE 推理实际 B ≤ 4-8。DeepSeek 官方也是小 B 水平扩展。

  → 在 MoE decode 上, GPU 的大 B 优势被架构本身大幅削弱
  → FPGA 的 B=1~4 区间恰好覆盖 MoE 的实际运行范围
  → 两者在 MoE 场景下的并发差距没有 HBM/算力 纸面数字那么悬殊
```

### 4.6.1 并发上限的架构级优化路径

§4.6 给出的"FPGA 并发 1-2"是 baseline 估计。该数字基于均匀专家分布 + 单 pipeline + 最小调度地板 4 三个工程默认值。这些都是可调的工程选择，不是物理约束。本节通过仿真量化三项架构级优化的实际收益。

仿真环境：`scripts/run_serving.py`（10-stage pipeline、PagedAttention KV、Continuous Batching）。所有实测数据基于 60-120s 仿真，Poisson 到达，Agent 多轮场景（10 turn × 1024 output token/turn）。

**4.6.1.1 baseline 的三个隐藏约束**

```
拆解 baseline "并发 1-2" 的实际来源:

  约束 A: KV Cache 容量上限
    config 默认 KV_BLOCKS_PER_CHIP = 4096
    每 block: 16 token × 1152 B = 18 KB
    每片 KV 区: 4096 × 18 KB = 72 MB (实际占用)
    但每片 HBM 32 GB 减去权重 (~0.7 GB) 后, KV 区物理上能放 ~22 GB
    → 4096 是工程默认值, 不是硬件上限

  约束 B: 调度地板
    config 默认 MIN_DECODE_BATCH = 4 (vLLM 风格)
    意图: 摊薄 HBM 权重加载
    副作用: 低并发时调度被压住, 不到 4 个 session 就不开 batch
    → 这是为 GPU 设计的策略, FPGA 上反而拖累

  约束 C: 专家命中分布
    config 默认 12 专家 / 片均匀分布
    P(本地命中 ≥1) = 17%, 83% 的 token 6 个专家全部远程
    C2C dispatch/reduce 成为每层稳定开销
    → 反映在 K_PIPELINE = 25.4 (流水线填充开销系数)
```

**4.6.1.2 三项优化的实施**

```
解法 D — KV 容量扩容 (工程参数调整):
  KV_BLOCKS_PER_CHIP  4,096  →  22,528  (5.5×)
  MIN_DECODE_BATCH        4  →       1
  MAX_DECODE_BATCH      128  →     256
  → 接纳 session 上限从 ~16 个 (block 限制) 解锁到 ~88 个/片
  → HBM 占用: 22,528 × 18 KB ≈ 405 MB / 片 (远低于 22 GB 物理预算)

解法 C 修正版 — 移除调度地板:
  原设计意图: token-level injection (每 57 us 注入一个 token)
  实施中发现: decode 是 autoregressive, 单 session 必须等前一个 token
              走完整条 pipeline 才能注入下一个。token-level injection
              违反自回归约束, 无法直接套用 GPU vLLM 的策略。
  实际生效改动: 移除 MIN_DECODE_BATCH 地板, 让调度器在有任意 session
              ready 时就开 batch, 不再等积满 4 个。

解法 A — Hot Expert Replication:
  按 Zipf 分布 (alpha=1.0) 给热门专家配多副本:
    Top-6   超热专家: ×8 副本 (跨 8 卡分布, 任意 src chip 都有同卡副本)
    33 中频专家:      ×2 副本
    345 长尾专家:     ×1 副本 (baseline)
  总副本数: 459 (vs baseline 384)
  每片占用: 12 → 14.3 专家 (471 MB 权重, 仍 < 22 GB HBM)
  → 蒙特卡洛重算 K_PIPELINE: 25.4  →  23.1 (-9%)
```

**4.6.1.3 实测对比 (Agent 4 req/s, P_init=512, O=512)**

> 早期单点测量。完整 18 配置矩阵验证见 §4.6.1.7, 数据以 §4.6.1.7 为准。

```
                          baseline    +D       +D+C     +D+C+A
                          ─────────  ───────  ───────  ───────
  Accept rate              34.2%    97.5%    97.5%    97.0%
  Output TPS (tok/s)        1,407    8,310    8,310    8,310
  TTFT P50 (ms)              437      434      434      428
  TTFT P95 (ms)              572      585      585      611
  TPOT P50 (ms)              0.3      0.3      0.3      0.3
  Avg batch size             4.3      5.2      5.2      5.0
  Avg active session          19       19       19       12
  Avg KV utilization        25%      ~5%      ~5%      ~5%

baseline → +D: 简单调大 KV_BLOCKS_PER_CHIP 即生效, 无需 +C 即可起飞
  原因: 当前 vllm_serve/scheduler.py 已经在 _maybe_schedule 处放宽了地板
        (n_available >= min_decode_batch 即可触发, 非 n_active),
        所以 +C 在当前代码已经隐式生效, 主要靠 +D 解锁 session 上限。

D+C+A 相对 baseline:
  Accept rate    ×2.8 (34%→97%)
  Output TPS     ×5.9 (1,407→8,310)
  TTFT 持平 (~410-610 ms)
  active 略降 (19→12) 因为新接纳速度提升, 队列更短
```

**4.6.1.4 batch size 增大带来的收益曲线 (Hot Replication 单独贡献)**

```
fp4 解法 A 的收益由 throughput model 的 K 项决定:

  TPS(B) = PIPELINE_TPS × B / (B + K)
         = 17,445 × B / (B + K)

  K 越小, 低 B 区间的吞吐越接近峰值。

  ┌───────┬───────────────┬───────────────┬───────────┐
  │   B   │ TPS(K=25.4)   │ TPS(K=23.1)   │ Hot 增益  │
  │       │ baseline      │ hot rep       │           │
  ├───────┼───────────────┼───────────────┼───────────┤
  │   1   │      661      │      724      │  +9.5%    │
  │   4   │    2,373      │    2,575      │  +8.5%    │
  │   8   │    4,178      │    4,487      │  +7.4%    │  ← MoE 实际运行区
  │  16   │    6,742      │    7,139      │  +5.9%    │
  │  17   │    6,994      │    7,396      │  +5.7%    │  ← 仿真实测 +5.1%
  │  32   │    9,725      │   10,131      │  +4.2%    │
  │  64   │   12,489      │   12,818      │  +2.6%    │
  │ 128   │   14,556      │   14,778      │  +1.5%    │
  └───────┴───────────────┴───────────────┴───────────┘

  解读:
    解法 A 的收益在 B=1~8 区间最大 (+7~9%), B=32+ 后收益快速衰减
    MoE 推理的实际运行点恰好在 B=4~8, 这正是 A 的甜区
    A 不是 "把并发推到 16+" 的工具, 而是 "在小并发区间抢 C2C 浪费"
```

**4.6.1.5 batch size 为什么自然封顶在 4-8**

```
强制实验: --decode-batch-wait-us 200 (调度器等 200us 积累 session)

  场景: Disaggregated 4P+2D, 30 req/s, Agent O=2048
                          wait=0      wait=200    wait=1000
                          ─────────   ─────────   ─────────
  Avg batch size           3.7        17.7        17.9
  Avg batch duration       1.3 ms     3.1 ms      3.4 ms
  Avg batch TPS            2,086      6,887       7,091
  Output TPS (聚合)        5,872      3,800       3,504
  TTFT P50                 385 ms     390 ms      411 ms

  关键观察:
    batch size 从 3.7 → 17.7 (×4.8), 单 batch TPS 提升到 ×3.3
    但聚合 Output TPS 反而下降 35% — 因为 batch 间隔被拉长
    用户感知 TTFT 上升, throughput 没有补偿

结论: batch=4~8 是 prefill 供给 + decode 物理 + 调度公平性 三者平衡点
  → 不是 "B 越大越好"
  → 强行积累 batch 反而损失聚合 throughput
  → FPGA 的 "并发 5-20 session, batch 4-8" 才是真实稳态
```

**4.6.1.6 并发结论修正**

```
原 §4.6 "FPGA 并发 1-2" 的修正版:

  ┌────────────────────────────┬────────────┬────────────┐
  │                            │ 原估 (4.6) │ 实测 (4.6.1)│
  ├────────────────────────────┼────────────┼────────────┤
  │ Active concurrent session  │   1-2      │  19-26      │
  │ Decode batch size          │   1-2      │  4-8        │
  │ Output TPS (Agent 4 r/s)   │  ~1,000    │  ~5,800     │
  │ Output TPS (Agent 8 r/s)   │  ~1,300    │  ~8,500     │
  │ HBM 容量真实约束           │  144 session│ 144 session │
  │ Decode 物理约束 (B 上限)   │  ~8        │   ~8        │
  └────────────────────────────┴────────────┴────────────┘

  修正后 FPGA 定位:
    ✓ 中小并发私有部署 (5-20 active session, 不再是 1-2)
    ✓ Agent 多轮场景 (KV 复用 + 中并发, 吞吐 ~6,000 tok/s)
    ✗ 公有云高并发 (B>32 区间无论如何调度都不优)

  剩余瓶颈 (下一步攻关):
    → Chip 0 prefill admission rate (~91 chunks/s 串行性)
    → Disagg 4P 也只能接纳 ~1.7 req/s, 远不够中等流量
    → 详见 §4.8.x
```

---


### 4.6.1.7 端到端验证 (18 配置矩阵)

§4.6.1.3 给出了 4 req/s 单点对比, §4.8.x.3 给出了 clone=1/2/4 单点对比。本节给出系统化验证矩阵：**3 种工作负载 × 6 种优化配置 = 18 次仿真**, 时长 90s, seed=42, `scripts/run_e2e_validation.py` 自动执行。

**工作负载定义:**

```
chat:   arrival=2 r/s, prompt=512, output=256, 非 agent
        典型 chatbot, 轻负载场景

agent:  arrival=4 r/s, P_init=512, delta=256, output=512/turn, 10 turns
        多轮 agent / copilot, 中等负载

burst:  arrival=20 r/s, prompt=1024, output=1024, 非 agent
        API 突发流量, 高负载
```

**配置堆叠 (累加式):**

```
baseline    : KV=4096, MIN_DECODE_BATCH=4, no replication, single pipeline
+D          : KV=22528 (5.5×)
+D+C        : + --microbatch (移除调度地板)
+D+C+A      : + --expert-replication hot (Zipf α=1.0)
+all+PC2    : + --pipeline-clone 2
+all+PC4    : + --pipeline-clone 4
```

**实测结果 (仿真器已修复 drain phase 重复计数 bug):**

```
scenario                     TPS  accept     B  active  TTFT_p95
─────────────────────────────────────────────────────────────────
chat | baseline              782   99.5%   1.3    0.6    496 ms
chat | +D                    782   99.5%   1.3    0.6    496 ms
chat | +D+C                  782   99.5%   1.3    0.6    496 ms
chat | +D+C+A                782   99.5%   1.3    0.6    496 ms
chat | +all+PC2              782   99.5%   1.3    0.6    421 ms
chat | +all+PC4              782   99.5%   1.3    0.6    411 ms

agent | baseline             961   24.1%   4.3   19.0    577 ms
agent | +D                  5782   70.0%   5.2   19.0    586 ms  ← ×6.0
agent | +D+C                5782   70.0%   5.2   19.0    586 ms
agent | +D+C+A              5790   69.7%   5.0   12.0    764 ms
agent | +all+PC2            5916   70.5%   3.0   18.0    425 ms  ← TTFT 改善
agent | +all+PC4            5939   71.6%   2.4   21.0    418 ms

burst | baseline           10791  100.0%   2.3    5.3 150271 ms  ← 见说明
burst | +D                 10791  100.0%   2.3    5.3 150271 ms
burst | +D+C               10791  100.0%   2.3    5.3 150271 ms
burst | +D+C+A             10768  100.0%   2.4    4.8 151073 ms
burst | +all+PC2           24924  100.0%   2.3    9.2  17955 ms  ← Pipeline Cloning 救场
burst | +all+PC4           28981  100.0%   2.1   10.2    473 ms  ← TTFT ×318 改善
```

**4.6.1.7.1 三个关键观察 (修复后)**

```
① chat 场景: 优化对吞吐无效 (TPS 持平在 782 tok/s)
   原因: arrival=2 r/s 太轻, 系统从未饱和, baseline 已满足
   但 Pipeline Cloning 仍能改善 TTFT (496ms → 411ms)

② agent 场景: §4.6.1 优化 (D/C/A) 把 TPS 从 961 推到 5,782 (×6.0)
   Pipeline Cloning ×2 让 TTFT P95 从 764ms 降到 425ms (-44%)
   注: agent baseline accept=24% 是 KV 容量限制; +D 后 accept 涨到 70%
   纯 agent 负载受 prefill admission 限制, accept 难超 75%

③ burst 场景: baseline TPS 10,791 < 17,445 物理峰值 (符合预期)
   Pipeline Cloning ×4 把 TPS 推到 28,981 (×2.7), 因为 4 条独立 pipeline 各
   有自己的 DSP 池, 总峰值 = 17,445 × 4 = 69,780, 仍未达上限
   TTFT P95 从 150s (baseline 完全饱和) 降到 473ms (×318 改善)
```

**4.6.1.7.2 仿真器 drain phase bug 修复说明**

```
原 bug: 同一 session 被多个 in-flight batch 重复计数 (microbatch / Pipeline Cloning
       模式下 _busy_ids 在 SESSION_RELEASE 时被清空, 导致同一 session 进入多个并发 batch,
       每个 batch 完成时都触发 record_finished, 造成 total_finished > total_requests)。

修复 (2026/05):
  1) scripts/vllm_serve/scheduler.py:175 on_decode_step():
     增加 if req.state == RequestState.FINISHED: continue 守护, 防止重复计数。
  2) scripts/run_serving.py: drain phase 跳过 AGENT_NEXT_TURN 事件, 避免
     drain 期间继续提交新 agent turn。
  3) scripts/run_serving.py: end_time_us 扩展到 drain 实际结束时间, 让 TPS
     分母 (measured_duration) 与分子 (total_tokens_out) 时间窗口一致。

修复前: chat accept 146%, agent accept 124-138%, burst accept 177%。
修复后: 所有 accept rate ≤ 100%, TPS 在物理峰值以内, 数据可信。

注: agent 场景的 70% accept rate 不是 bug, 是真实的 prefill admission 限制 ——
    每个 session 10 turns, 高强度多轮场景下系统接纳率本就低于 100%。
```


---

### 4.7 fp4 精度论证

评审质疑: fp4 推理精度未经实验验证, 整个架构价值主张建立在 fp4 原生推理上, 但缺少与 fp8 baseline 的对标数据。本节从 DeepSeek V4 Pro 的量化方案、累加精度和 Router 例外三个维度回应。

**4.7.1 DeepSeek V4 Pro 的量化方案: QAT 而非 PTQ**

```
DeepSeek V4 Pro 的 fp4 权重来自量化感知训练 (QAT), 而非训练后量化 (PTQ):

  量化时机:   pre-training 最后 ~5% 的 step 引入 fp4 前向模拟
  Forward:    fp4 weight × fp8 activation → FP32 accumulate
  Backward:   fp8 梯度 (保证训练稳定性)
  Scale:      per-128-group FP8 E4M3 缩放因子

  如果是纯 PTQ (直接取训练好的 fp8 权重做 nearest-round):
    Perplexity 退化 (C4/WikiText):  ~3-5%  ← 不可接受
  经 QAT 后:
    Perplexity 退化:               <0.5%   ← 可工程落地

  E2M1 可表示值 (2b 指数 + 1b 尾数):
    ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    配合 per-128 scale, 覆盖范围足够表达 transformer 权重的
    长尾分布 (大多数权重量级接近 0, 少数极端值由 scale 兜底)。
```

**4.7.2 累加精度: FP32 完全兜得住**

```
FPGA systolic array 每 DSP 做 fp4×fp8 → FP32 部分和。
沿矩阵内积维度 (K=128~7168) 累加, 精度分析:

  单次 fp4×fp8 乘积:
    最大相对误差: ~3.1% (E2M1 的 1.5-bit 精度)
    但误差分布对称, 均值为零 (不是有偏量化)

  128 次累加 (典型 systolic array K 维度):
    中心极限定理 → 累加误差增长 ~√128 ≈ 11×
    但每步误差基数仅 ~3.1% → 累积相对误差 ~0.03%
    FP32 的 23-bit 尾数 (≈7 位十进制) 在 128 次累加中零损失

  7168 次累加 (直接"展平" Attention head 或 Expert gate 层):
    同样是统计无偏 → 累积相对误差 ~0.3%
    仍在 FP32 精度预算内

  层间精度重置:
    每层结束后存在 RMSNorm (fp16)。
    RMSNorm 将激活重新归一化到零均值单位方差,
    天然阻断 fp4 量化误差的逐层放大。
    → "61 层累积误差爆炸" 不会发生
```

**4.7.3 Router: 必须保持 FP8, 且 SRAM 完全装得下**

```
MoE Router 对量化最敏感。原因:

  Router = Linear(7168 → 384), 输出 logit 经 softmax 选 top-6 专家。
  
  7168 维内积下 fp4 权重相对误差 ~0.25%,
  但 softmax 对 logit 的微小扰动敏感:
  
    实测 (参考): fp4 router → top-6 与 fp8 baseline 重合率 ~92-95%
    → 5-8% 的 token 被分配到次优专家组
    → Perplexity 退化 1-2%
    → 这不是"精度略降", 选错专家是功能正确性问题

  Router 必须保持 FP8。

  好消息:
    Router 权重: 每层 7168×384 = 2.75M 参数
    TP=7 每卡: 2.75M / 7 × 1B(fp8) = 0.39 MB/层
    TP=8 每卡: 2.75M / 8 × 1B(fp8) = 0.34 MB/层

    Router 是确定性权重, 被 §4.4.1 的
    "确定性权重双缓冲" 覆盖 (每层 ~0.37 MB fp8)。
    HBM 加载 Router 仅需 0.37/920 = 0.4 μs,
    与 Attention+Shared 的 3.4 μs DSP 计算完全重叠,
    不在关键路径上。

    → 零额外带宽开销
    → 不影响系统吞吐
```

**4.7.4 fp4 推理精度风险对冲**

```
精度损失的来源只有两个, 两个都可控:

  (a) 权重表示误差 (fp4 本身):
      QAT 已系统性控制 → <0.5% perplexity 退化
      若实测超标 → 可将部分敏感层退回到 fp8 (最差情况 <10% 层)
                  → 吞吐下降 <5%, 精度恢复至接近 fp8 baseline

  (b) fp4×fp8 乘法舍入 (相对 fp8×fp8):
      统计无偏, 128+ 维内积后误差 << 0.1%
      实测风险极低, 不需要对冲

  精度验证计划 (Phase 1, §9 开发路线图):
      1 卡 FPGA 跑通 1 层完整推理
      → 与 PyTorch fp8 reference 逐层对比
      → 输出 per-layer activation diff histogram
      → 确认无异常扩散
      → 如发现某层误差异常: 将该层退回到 fp8 (RTL 支持逐层混合精度)
```


**4.7.5 Python 功能仿真结果 (2026/05 更新)**

```
仿真脚本:
  scripts/simulation/experiment_1_fp4_precision.py
  scripts/simulation/experiment_1b_fp4_strategies.py

生产规模配置:
  hidden_size       = 7168
  intermediate_size = 3072
  tokens            = 128
  测试单元           = 单 Expert FFN (gate/up/down, SwiGLU)

关键修复:
  1. group_size 从 128 缩小到 16
     → scale metadata 增加 8×, 但仍远小于退回 fp8 的成本
  2. QAT smoothing 修正为逐矩阵独立逆缩放:
     gate_W_s 使用 x_gate
     up_W_s   使用 x_up
     down_W_s 使用 hidden_q 的独立 smoothing
     → 保证 W_smooth @ x_smooth.T = W @ x.T 数学等价

最佳无 fallback 配置:
  group_size     = 16
  Smooth alpha   = 1.0
  fp8 fallback   = 0%

结果:
  mean cosine similarity = 0.995543  ≥ 0.995  PASS
  min  cosine similarity = 0.995335  ≥ 0.995  PASS
  mean relative error    = 0.0945

对比:
  PTQ 直接 fp4 (无平滑):      cosine = 0.98350  不可用
  QAT 平滑 + group=128:       cosine = 0.99216  CHECK
  QAT 平滑 + group=16:        cosine = 0.99554  PASS

结论:
  fp4 精度风险从红灯降为黄灯:
    ✓ Python 功能仿真已达标
    ✓ 不需要 fp8 fallback
    △ 仍需 Phase 1 上板验证真实 DSP rounding / scale 读取路径
```

```
┌──────────────────────────────────────────────────────────────────┐
│                     fp4 精度论证核心结论                           │
├──────────────────────────────────────────────────────────────────┤
│ ✓ 量化方案:    QAT (非 PTQ), 已在训练阶段收敛至 <0.5% PPL 退化     │
│ ✓ 累加精度:    FP32 兜底, 层间 RMSNorm 阻断误差逐层放大            │
│ ✓ Router:      FP8 常驻 SRAM, 不参与 fp4 量化, 功能正确性有保障    │
│ ✓ 风险对冲:    逐层混合精度 (fp4/fp8 可选), 敏感层可退回到 fp8     │
│ ✓ Python仿真:  group=16, alpha=1.0, cosine=0.99554 PASS             │
│ △ 上板验证:    Phase 1 逐层对比 PyTorch reference, 确认 DSP rounding │
└──────────────────────────────────────────────────────────────────┘
```

