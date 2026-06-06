### 4.8 Prefill 性能分析与调度策略

方案书多处提及 "Prefill GPU 完胜"、"Prefill 只占推理时间的 ~5%"。评审的合理质疑: TTFT (Time-To-First-Token) 是用户感知延迟的唯一阶段, 800 tok/s decode 再快, TTFT 等 5 秒也不可接受。本节量化分析 FPGA 集群的 Prefill 性能边界。

**4.8.1 TTFT 估算**

```
Prefill 是 compute-bound 场景:
  每 token 每层 MAC: ~611M (见 §4.2)
  30 FPGA 聚合: 8.44 × 30 = 253 TMACs/s

按 4 节点流水线 (最慢节点决定吞吐):
  Node 0 (TP=7, 15 layers): P × 15 × 611M / 7  = P × 1.31T MACs
  Node 1 (TP=8, 15 layers): P × 15 × 611M / 8  = P × 1.15T MACs
  Node 2 (TP=8, 15 layers): P × 15 × 611M / 8  = P × 1.15T MACs
  Node 3 (TP=7, 16 layers): P × 16 × 611M / 7  = P × 1.40T MACs  ← 瓶颈

  P = prompt token 数
  瓶颈节点时间 = P × 1.40T / 8.44T = P × 166 ms
  流水线填充 ≈ 4 级 × 10 ms = 40 ms

  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ Prompt 长度   │ 计算时间      │ TTFT (含填充) │ 对比 H100*   │
  ├──────────────┼──────────────┼──────────────┼──────────────┤
  │ 200 tokens   │ 33 ms        │ ~70 ms       │ ~5 ms        │
  │ 512 tokens   │ 85 ms        │ ~125 ms      │ ~8 ms        │
  │ 1K tokens    │ 166 ms       │ ~210 ms      │ ~12 ms       │
  │ 4K tokens    │ 664 ms       │ ~800 ms      │ ~50 ms       │
  │ 8K tokens    │ 1.33 s       │ ~1.4 s       │ ~100 ms      │
  │ 16K tokens   │ 2.66 s       │ ~2.7 s       │ ~200 ms      │
  │ 32K tokens   │ 5.31 s       │ ~5.4 s       │ ~400 ms      │
  │ 128K tokens  │ 21.2 s       │ ~21.3 s      │ ~1.6 s       │
  └──────────────┴──────────────┴──────────────┴──────────────┘
  * H100 按 8×990 TFLOPs / 2 (MoE 稀疏 ≈ 50% MAC 利用率) 估算
```

**4.8.2 Chunked Prefill: 为什么 TTFT ≠ 完整 Prefill 延迟**

```
借鉴 vLLM/Sarathi 的 Chunked Prefill 策略:

  长 prompt → 切成 512 token 的 chunk
  decode_step → prefill_chunk → decode_step → prefill_chunk → ...

  关键效果:
    首个 chunk (512 tok) TTFT: ~125 ms  ← 用户感知延迟
    完整 prefill 在后台继续, 与 decode 交替执行

  128K prompt 的场景:
    完整 prefill 理论: 21.3s
    Chunked: 首 token 125ms 就可见
    256 个 chunk × 85ms = 21.8s 后台完成
    用户在前 125ms 就能看到第一个 token 开始生成

  适用条件:
    ✓ B=1 的 decode 场景 (agent, chatbot)
    ✗ 需要完整 prefill 结束才能 decode 的场景 (极少)
```

**4.8.3 "Prefill 占 5%" 的适用边界——按 workload 分级**

```
这个数字仅对短 prompt chatbot 成立。实际依赖 context 长度:

  ┌──────────────────┬─────────┬──────────┬────────┬──────────────┐
  │ Workload          │ Prompt  │ Response │ TTFT   │ Prefill 占比  │
  ├──────────────────┼─────────┼──────────┼────────┼──────────────┤
  │ 短问答 (ChatGPT)  │ 200     │ 2000     │ 70ms   │ ~0.2%        │
  │ Customer Service │ 1K      │ 500      │ 210ms  │ ~4%          │
  │ RAG (检索增强)    │ 8K      │ 500      │ 1.4s   │ ~14%         │
  │ 代码审查          │ 16K     │ 2000     │ 2.7s   │ ~7%          │
  │ 文档摘要          │ 32K     │ 1000     │ 5.4s   │ ~24%         │
  │ 长文写作 (Claude) │ 10K     │ 8000     │ 1.7s   │ ~1%          │
  │ Agent (多轮)      │ 5-20K   │ 300/轮   │ 0.8-3s │ ~15-30%     │
  └──────────────────┴─────────┴──────────┴────────┴──────────────┘

  "5%" 只在第一行成立。
  方案定位的私有部署客户 (金融/医疗/政府) 主要是:
    Customer Service、RAG、Agent — 中等 prompt, 需关注 TTFT
```

**4.8.4 GPU Prefill 优势: 坦白承认**

```
┌──────────────┬───────────────┬───────────────┬──────────────────┐
│ Prefill (4K)  │ 8×H100 (FP8) │ 8×B200 (FP4) │ 30 FPGA (fp4)    │
├──────────────┼───────────────┼───────────────┼──────────────────┤
│ TTFT          │ ~50ms         │ ~25ms         │ ~800ms           │
│ 成本           │ $240K (管制)  │ $320K (管制)  │ $321K (可得)      │
│ 倍率 vs FPGA   │ 16× 更快      │ 32× 更快      │ 1×               │
│ 中国可获取      │ ✗            │ ✗             │ ✓                │
└──────────────┴───────────────┴───────────────┴──────────────────┘

GPU 在 Prefill 的绝对优势来自物理: 大 batch 下 Tensor Core 利用率高,
算力密度是 FPGA 的 3-9×。这是不争的事实。

三个实事求是的对冲:
  (a) H100/B200 对中国市场不可获取。比"谁更快"不如比"谁能用"。
  (b) Chunked prefill 使首 token 延迟远小于完整 prefill。
      超过 80% 的商业 prompt < 4K token, TTFT < 800ms 可接受。
  (c) 国内可获取硬件中 (Ascend 910C 等), Prefill 也并不接近 H100。
      FPGA 在 Prefill 上的劣势是相对于"不可获取的 GPU",
      而非相对于竞品。
```

**4.8.7 CPU Prefill 质疑回应：内存带宽 & 存储**

常见质疑: "CPU 内存带宽吃不住" + "权重存储太大"。
以下用具体数字逐条回应。

```
质疑 1: "61 层权重太大, CPU 内存存不下"

  每层权重 (fp8 解压):  ~135 MB
  61 层总计 (fp8):       ~8.2 GB
  如果用 fp4 压缩:        ~4.1 GB
  典型服务器 RAM:         256 GB (Dual Xeon GNR)

  → 61 层权重只占 3.2% 的 RAM。
  → 即使同时缓存 KV (128K tokens × 61 层 = 8 GB),
    总计 16 GB, 只占 6%。
  → "存不下" 不成立。


质疑 2: "DDR5 内存带宽吃不住 GEMM"

  以 W_Q [P=128, 7168] × [7168, 7168] 为例:
    计算量:  128 × 7168² = 6.6 GMACs
    计算时间: 6.6G / 10.5T = 0.63 ms
    权重加载: 7168² × 1B = 51.4 MB
    内存时间: 51.4 MB / 307 GB/s = 0.17 ms

    → 计算/内存比 = 0.63/0.17 = 3.7x
    → 这是 compute-bound 操作, 不是 memory-bound

  完整的 P=128 chunk, 61 层:
    总权重:    8.2 GB
    内存时间:  8.2 GB / 307 GB/s = 27 ms
    计算时间:  395 ms (校准值)
    → 计算/内存比 = 395/27 = 14.7x
    → 计算比内存慢 15 倍!

  为什么内存不是瓶颈?
    Batch P=128: 每个权重字节被 128 个 token 复用。
    有效内存需求: 51.4 MB / 128 = 0.4 MB/token 权重带宽。
    DDR5 307 GB/s / 10.5 TFLOPS = 29 bytes/FLOP 可用。
    W_Q 需要: 51.4 MB / 6.6 GMACs = 0.008 bytes/FLOP。
    可用/需要 = 29 / 0.008 = 3625x 裕量。
    → "带宽吃不住" 不成立。


质疑 3: "128K 超长 context 的 KV cache 太大"

  KV per token:  K_latent(512) + V_latent(512) = 1024B fp8
  128K tokens:   128K × 1024 = 131 MB per layer
  61 层:         61 × 131 MB = 8.0 GB
  加上权重:      8.2 + 8.0 = 16.2 GB

  → 在 256 GB RAM 中只占 6%。
  → 对于 chunked prefill: 只需要存当前 chunk 的 KV (P=128 → 8 MB/层)


CPU Prefill 真正的瓶颈是算力, 不是内存:

  ┌──────────────────┬──────────┬──────────┬──────────────┐
  │ 约束              │ 需求     │ 可用     │ 裕量          │
  ├──────────────────┼──────────┼──────────┼──────────────┤
  │ 权重存储          │ 8.2 GB   │ 256 GB   │ 31x           │
  │ KV Cache (128K)   │ 8.0 GB   │ 256 GB   │ 32x           │
  │ DDR5 带宽         │ 27 ms    │ 395 ms   │ 14.7x         │
  │ CPU 算力 (TFLOPS) │ 395 ms   │ -        │ 瓶颈在此!     │
  └──────────────────┴──────────┴──────────┴──────────────┘
```



**4.8.6 2026 CPU Prefill 评估 — 硬件已追上来**

> 更新 (2026/05): CPU prefill 算力已从 SPR 的 1.7 TFLOPS 跃升至 GNR/Turin 的 10+ TFLOPS。
> 差距从 11× 缩小到 2×。CPU prefill 现在可以覆盖 80% 的商业场景。

```
2026 年可购买 CPU 的 Prefill 性能 (P=128 chunk, DeepSeek V4 Pro, 61 层):

┌──────────────────────────────┬──────────┬──────────┬──────────────┐
│ CPU                           │ 有效 TF   │ P=128 TTFT│ vs FPGA Decode│
├──────────────────────────────┼──────────┼──────────┼──────────────┤
│ Dual Xeon 6980P (GNR, 128c)  │ 10.5 TF   │  396 ms  │ 2.0x 慢       │
│ Dual EPYC 9755 (Turin, 128c) │ 10.5 TF   │  396 ms  │ 2.0x 慢       │
│ Dual EPYC 9965 (Turin, 192c) │  9.0 TF   │  462 ms  │ 2.4x 慢       │
│ Quad Xeon 6980P (4-socket)   │ 18.2 TF   │  228 ms  │ 1.2x 慢       │
│ Dual Xeon 8592+ (SPR, 2023)  │  1.7 TF   │ 2473 ms  │ 11x 慢 (参考) │
│ 1x A100 (GPU, fp16)           │ 187  TF   │   22 ms  │ 18x 快 (管制) │
└──────────────────────────────┴──────────┴──────────┴──────────────┘

按场景适用策略 (2026):

┌──────────────────────┬───────────┬──────────────┬────────────────┐
│ 场景                  │ Prompt    │ 推荐方案      │ TTFT (GNR)     │
├──────────────────────┼───────────┼──────────────┼────────────────┤
│ 短问答 / Chat         │ < 200     │ CPU 全量      │ 0.4-0.8s        │
│ Chat (短)             │ 200-500   │ CPU 全量      │ 0.8-1.6s        │
│ Agent warm (增量)     │ +500-2K   │ CPU 增量 ✅   │ 1.6-6.3s (增量) │
│ RAG / 客服            │ 1-2K      │ FPGA chunked  │ 首 chunk 85ms   │
│ 代码审查              │ 10-20K    │ FPGA chunked  │ 首 chunk 85ms   │
│ 长文档 / 128K context │ 32-128K   │ FPGA chunked  │ 首 chunk 85ms   │
│ 极致低延迟 TTFT       │ 任意      │ +GPU (A100等) │ < 50 ms         │
└──────────────────────┴───────────┴──────────────┴────────────────┘

> **注意**: 上表 TTFT 已根据 §4.8.8 审计结果修正 (v1.4)。
> 原 v1.3 版本对 CPU prefill TTFT 系统性低估 (混淆了 "首 chunk 完成时间" 与 "首 token 生成时间")。
> 修正后 CPU prefill 的实用范围从 <4K 缩小到 <500 tokens (或 Agent warm start 增量模式)。
> 中长 prompt 一律走 FPGA Tier 2 chunked prefill。

BOM 影响:

┌──────────────────────┬───────────┬──────────────┬────────────────┐
│ 方案                  │ 增加成本   │ Prefill 改善  │ 推荐            │
├──────────────────────┼───────────┼──────────────┼────────────────┤
│ 当前 SPR (已有)       │ 0          │ 基准         │ 2023 基线       │
│ -> 升级 GNR 6980P    │ +30K/CPU   │ x6 加速      │ 推荐           │
│ -> 升级 EPYC 9755    │ +25K/CPU   │ x6 加速      │ 推荐           │
│ -> 4-Socket GNR      │ +60K+主板  │ x11 加速     │ 性能极致        │
│ -> 加 1xA100 GPU     │ +80K       │ x30 加速     │ 最快, 管制风险  │
└──────────────────────┴───────────┴──────────────┴────────────────┘
```


### 4.8.8 CPU Prefill + FPGA Decode: 完整可行性审计

> 核心问题: CPU Prefill + FPGA Decode 混合架构是否真的可行?
> 短答案: 可行, 但有严格前提条件。当前文档 (§4.8.6/§4.8.7/§14.E) 的分析方向正确,
> 但存在 **TTFT 数字系统性低估**和**关键数据路径未完整描述**两个缺口。
> 以下逐一审计。

**A. CPU 算力可行性 — 已验证 ✅**

```
§4.8.7 的算力分析结论正确, 无需修正:

  Dual Xeon GNR 6980P 有效 fp8 算力: ~10.5 TFLOPS (AMX BF16→fp8 折算)
  P=128 单 chunk 61 层计算量:         ~4.1 GMACs × 61 = ~250 GMACs
  计算时间:                            ~395ms (含 AMX tile 配置 + 数据搬运开销)
  DDR5 带宽裕量:                       14.7× (307 GB/s vs 20.8 GB/s 需求)

CPU prefill 的瓶颈是算力, 不是内存带宽。对于 P≤512 的 chunk, 每个权重字节被
复用 ≥128 次 → compute-bound。这一点 §4.8.7 已经充分论证。

但是, 有一个隐含假设需要显式确认:

  权重预加载 (8.2 GB fp8) 需要 27ms (§4.8.7 line 1670)。
  这个加载只在 session 启动时做一次 (或者模型切换时)。
  稳态运行中权重驻留在 CPU pinned memory, 不重复加载。
  → 对稳态 TTFT 无影响, 但冷启动首请求 TTFT = 395 + 27 = 422ms。
  → 文档 396ms 是正确的稳态数字, 但应标注为 "稳态" 而非 "首请求"。
```

**B. KV Cache DMA + 32 芯片分发路径 — 可行但未完整描述 ⚠️**

```
这是当前文档最大的架构缺口: §14.E 只说 "CPU prefill → PCIe DMA → FPGA HBM 双缓冲",
但没有说明 KV cache 如何从 Chip 0 到达 Chips 1-31。

只有 Chip 0 有 PCIe 连接。Chips 1-31 必须通过 SERDES pipeline forwarding 获取
各自的 KV cache。以下补全这个路径:

  ┌─────────────────────────────────────────────────────────────────┐
  │ CPU Prefill 完成 → KV latent (576B/token/layer, fp8)            │
  │                                                                  │
  │ Step 1: CPU → Chip 0 (PCIe 5.0 x16, ~28 GB/s)                   │
  │   P=128 chunk: 128 × 576B × 61 = 4.5 MB → DMA ~0.16ms           │
  │   128K 全量:   128K × 576B × 61 = 4.5 GB → DMA ~161ms           │
  │                                                                  │
  │ Step 2: Chip 0 保留 layer 0-1 的 KV (2/61 ≈ 148 KB)             │
  │         转发剩余 59/61 ≈ 4.35 MB → Chip 1 (SERDES 56 GB/s)      │
  │                                                                  │
  │ Step 3: Chip k 保留 layer 2k-2k+1 的 KV, 转发剩余               │
  │         每跳数据量递减: 4.35→4.2→4.1→...→0 MB                    │
  │         每跳延迟: ~75ns (SERDES) + 数据/56GB/s                    │
  │                                                                  │
  │ Step 4: Chip 31 收到最后 2 层的 KV (~148 KB)                     │
  │                                                                  │
  │ 总 pipeline 分发延迟 (P=128 chunk):                               │
  │   DMA (CPU→Chip0):     0.16 ms                                   │
  │   31-hop forwarding:   ~1.24 ms (首跳数据最多, 尾跳数据最少)      │
  │   合计:                 ~1.4 ms ← 相对 395ms 计算可忽略 (0.35%)   │
  │                                                                  │
  │ 总 pipeline 分发延迟 (128K 全量, CPU 一次性 prefill 完后再分发):  │
  │   DMA (CPU→Chip0):     161 ms                                    │
  │   31-hop forwarding:   ~155 ms (4.5 GB / 56 GB/s × 平均系数)     │
  │   合计:                 ~316 ms ← 显著! 但在 decode 开始前必须完成 │
  └─────────────────────────────────────────────────────────────────┘

关键发现:
  1. Chunked prefill (P=128): KV 分发开销仅 1.4ms, 完全可忽略。
  2. 全量 prefill (128K): KV 分发开销 316ms, 不可忽略。
     但 128K 全量 prefill 本身就不现实 (计算需要 ~395s), 实际必然走 chunked。
  3. Pipeline forwarding 可以在 chunk N+1 的 CPU 计算期间并行进行,
     进一步隐藏延迟。但第一个 chunk 的 forwarding 在 TTFT 关键路径上。

结论: KV 分发路径可行, 开销可控。文档需补全这个路径描述。
```

**C. 端到端 TTFT 真实分解 — 文档数字需修正 🔴**

```
这是最严重的缺口。§4.8.6 的场景表列出了 "TTFT ~395ms"、"TTFT 0.8-1.5s" 等数字,
但没有区分 "首 chunk 完成时间" 和 "首 token 生成时间"。

Chunked prefill 的 TTFT = 所有 chunk 的 prefill 时间 + 首次 decode 时间。
不是首 chunk 完成时间!

真实分解 (Dual GNR 6980P, P_chunk=128):

  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
  │ 阶段                  │ 延迟     │ 占比     │ 累积     │ 备注     │
  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ Tokenize + Embed     │ 2-5 ms   │ -        │ 5 ms     │ CPU 单线程│
  │ 权重预加载 (冷启动)   │ 27 ms    │ -        │ 32 ms    │ 仅首请求 │
  │ AMX GEMM chunk 1     │ 395 ms   │ 98.5%    │ 427 ms   │ P=128    │
  │ KV DMA → Chip 0      │ 0.16 ms  │ 0.04%    │ 427 ms   │ 4.5 MB   │
  │ KV forwarding 31-hop │ 1.24 ms  │ 0.3%     │ 428 ms   │ SERDES   │
  │ FPGA decode step 1   │ 1.4 ms   │ 0.3%     │ 430 ms   │ B=1      │
  │ Token → Host         │ <1 ms    │ -        │ ~430 ms  │ PCIe     │
  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘
  → P≤128 短 prompt: TTFT ≈ 430ms (稳态), 与文档声称 ~396ms 接近 ✓

  但对于更长的 prompt, 需要多个 chunk:

  ┌──────────────┬──────────┬─────────────────────┬──────────────────┐
  │ Prompt 长度   │ Chunks   │ 真实 TTFT (计算)     │ 文档声称 TTFT     │
  ├──────────────┼──────────┼─────────────────────┼──────────────────┤
  │ 200 tokens   │ 2×P=128  │ ~0.8s               │ < 300 ms ✗       │
  │ 500 tokens   │ 4×P=128  │ ~1.6s               │ ~600 ms ✗        │
  │ 1,000 tokens │ 8×P=128  │ ~3.2s               │ -                │
  │ 2,000 tokens │ 16×P=128 │ ~6.3s               │ 0.8-1.5s ✗       │
  │ 4,000 tokens │ 32×P=128 │ ~12.6s              │ ~395ms ✗✗        │
  │ 128K tokens  │ 1000×P=128│ ~395s (6.6分钟)     │ 首 chunk 125ms*  │
  └──────────────┴──────────┴─────────────────────┴──────────────────┘

  *文档的 "首 chunk 125ms" 是 FPGA Tier 2 的数字, 用于 >4K 场景。
   CPU Tier 1 的 "首 chunk ~400ms" 是 395ms 计算 + 5ms 开销。

  修正后的场景 TTFT 表 (CPU prefill, Dual GNR 6980P, P_chunk=128):

  ┌──────────────────────┬───────────┬──────────────┬──────────────────┐
  │ 场景                  │ Prompt    │ 真实 TTFT     │ 推荐方案          │
  ├──────────────────────┼───────────┼──────────────┼──────────────────┤
  │ 短问答 / Chat         │ < 200     │ 0.4-0.8s     │ CPU ✅            │
  │ RAG / 客服            │ 200-500   │ 0.8-2.0s     │ CPU ✅ (边界)     │
  │ Agent 增量            │ 500-2K    │ 2.0-6.3s     │ CPU ⚠️ (勉强)    │
  │ 多轮 Agent (warm)     │ 2K-5K 增量│ 0.8-2.0s*    │ CPU ✅ (*仅增量)  │
  │ 代码审查              │ 5-20K     │ -            │ FPGA Tier 2 ✅    │
  │ 长文档 / 128K         │ >4K       │ -            │ FPGA Tier 2 ✅    │
  └──────────────────────┴───────────┴──────────────┴──────────────────┘

  *Agent warm start: 前缀 KV cache 复用, 只 prefill 新增 token (通常 500-2K)。

关键纠正:
  1. §4.8.6 的 "短问答 < 200: TTFT < 300 ms" → 应改为 "0.4-0.8s"
  2. §4.8.6 的 "RAG 1-2K: TTFT 0.8-1.5s" → 应改为 "3.1-6.3s"
     如此慢的 TTFT 意味着 RAG > 500 tokens 应该用 FPGA Tier 2, 不是 CPU
  3. §4.8.6 的 "Agent 2-5K: CPU chunked, TTFT 1.5-4s" → 应改为 "6.2-15.4s"
     对于 5K prompt, CPU prefill 需要 ~15.4s, 不可接受
     但 Agent warm start 只需 prefill 新增 token → 实际可接受
  4. Tier 1/Tier 2 的阈值应该从 4K 下调到 ~500 tokens
     (500 tokens → 4 chunks × 395ms = 1.6s, 已是用户感知边界)

结论: CPU prefill 仅适用于短 prompt (<500 tokens) 和 Agent warm start (增量 prefill)。
     中长 prompt 必须走 FPGA Tier 2。文档的场景表需要大幅修正。
```

**D. CPU/FPGA 并发调度正确性 — 可行但细节不足 ⚠️**

```
场景: Request A 正在 FPGA decode, Request B 的 prompt 需要 CPU prefill。
      CPU 必须同时处理 prefill (AMX GEMM) + decode 协调 (token 分发/收集) + NIC 流量。

资源分配分析:

  CPU 核心分配 (Dual GNR, 128C/256T):
    ┌─────────────────────┬──────────┬─────────────────────────────┐
    │ 任务                 │ 核心数   │ 说明                         │
    ├─────────────────────┼──────────┼─────────────────────────────┤
    │ AMX GEMM (prefill)   │ 64-96C   │ AMX 每个 core 一个 tile     │
    │ Decode 协调           │ 2-4C     │ Token dispatch, KV swap    │
    │ NIC 中断/轮询         │ 2-4C     │ 网络 I/O                    │
    │ OS + vLLM scheduler   │ 4-8C     │ 调度, 内存管理              │
    │ 剩余 (headroom)       │ 16-56C   │ 应对突发                    │
    └─────────────────────┴──────────┴─────────────────────────────┘

  DDR5 带宽分配 (307 GB/s total, 8-channel):
    ┌─────────────────────┬──────────┬─────────────────────────────┐
    │ 消费者               │ 带宽      │ 占比                         │
    ├─────────────────────┼──────────┼─────────────────────────────┤
    │ CPU prefill GEMM     │ ~21 GB/s │ 6.8% (权重流式读取)          │
    │ KV cache DMA (PCIe)  │ ~0.1 GB/s│ 0.03% (chunked, 平均)       │
    │ NIC 收发              │ ~5 GB/s  │ 1.6% (2×25GbE)              │
    │ OS + 其他             │ ~5 GB/s  │ 1.6%                        │
    │ 剩余                  │ ~276 GB/s│ 90% ← 充裕                  │
    └─────────────────────┴──────────┴─────────────────────────────┘

  正确性风险:
    1. AMX 寄存器状态: 在 prefill chunk 间隙需要保存/恢复 AMX tile 配置。
       XSAVE/XRSTOR 开销约 ~5-10μs → 如果每 chunk 切换 1 次, 可忽略。
       但如果 decode 协调需要频繁中断 prefill → 切换开销累积。

    2. KV cache 双缓冲 swap 时序 (§14.E):
       "原子 swap: B ready 且 A 耗尽时切换"
       未明确: swap 发生在 decode step 之间还是可以打断正在进行的 decode?
       如果不能在 decode 中间 swap:
         → swap 只能在 decode step 间隙 (~1.4ms 窗口) 执行
         → swap 本身耗时 < 10μs (PCIe 写一个 flag + FPGA 中断)
         → 可忽略
       如果 decode 正在读 buf A, CPU 写完了 buf B:
         → CPU 设置 "B ready" flag
         → FPGA 在下一个 decode step 开始时检查 flag
         → 原子切换到 buf B
         → 不需要立即打断 decode

    3. 并发 session 的 KV cache 隔离:
       多个 session 的 CPU prefill 产生各自的 KV cache。
       FPGA HBM 中需要分区管理 (per-session KV 区域)。
       vLLM 已有 PagedAttention 的 block table 管理, 这部分可复用。

  CPU prefill 完成 → 通知 FPGA 的信号路径:
    (a) CPU 写 "prefill_done" flag 到 Chip 0 的 PCIe BAR
    (b) Chip 0 收到后, 检查 KV buf B 是否完整
    (c) 下一个 decode step 开始时, swap buf A ↔ buf B
    (d) 开始用新 KV cache decode

  多 session 并发时序示例:

    t=0     : Session A 正在 FPGA decode (step 50)
    t=0     : Session B 请求到达, CPU 开始 prefill (AMX GEMM)
    t=395ms : Session B CPU prefill chunk 1 完成
    t=396ms : KV cache for B → PCIe DMA → Chip 0 → pipeline forward
    t=397ms : KV 分发完成, CPU 设 "B ready" flag
    t=397ms : Session A decode step 结束, FPGA 检查 flag, swap
    t=398ms : Session B 首次 decode step 开始
    → Session A 感知到的延迟增加: 0ms (prefill 在后台, decode 不受影响)
    → Session B TTFT: ~398ms ✓

  关键假设 (需要验证):
    - FPGA 能在 decode step 间隙接受 KV cache swap 中断
    - 多 session KV cache 在 HBM 中的分区不互相干扰
    - CPU 核心分配策略不影响 AMX 吞吐
```

**E. CPU→FPGA Prefill 移交阈值 — 需量化论证 🟡**

```
当前文档以 4K tokens 作为 CPU→FPGA 移交边界 (§14.E):
  "Prompt > 4K tok: FPGA chunked prefill"

但根据 §C 的真实 TTFT 分析, 这个阈值应该基于 TTFT 用户体验目标:

  用户体验 TTFT 容忍度 (行业经验值):
    < 500ms  : 实时对话, 用户无感知
    0.5-1.0s : 轻微延迟, 可接受
    1.0-2.0s : 明显延迟, 但 RAG/Agent 场景可接受
    > 2.0s   : 不可接受 (用户会刷新/重试)

  CPU prefill (GNR) P=128/chunk: ~3.1ms/token → TTFT ≈ prompts_tokens × 3.1ms
  FPGA prefill P=512/chunk:      ~0.66ms/token → TTFT ≈ prompts_tokens × 0.66ms

  移交阈值分析:

    ┌──────────────┬──────────────────┬──────────────────┬──────────┐
    │ TTFT 目标     │ CPU 最大 prompt   │ FPGA 最大 prompt  │ 推荐     │
    ├──────────────┼──────────────────┼──────────────────┼──────────┤
    │ < 500ms      │ ~160 tokens      │ ~750 tokens      │ FPGA     │
    │ < 1.0s       │ ~320 tokens      │ ~1,500 tokens    │ CPU/FPGA │
    │ < 2.0s       │ ~640 tokens      │ ~3,000 tokens    │ CPU 边界 │
    │ < 5.0s       │ ~1,600 tokens    │ ~7,500 tokens    │ CPU 差   │
    └──────────────┴──────────────────┴──────────────────┴──────────┘

  推荐移交策略 (修正版):

    ┌──────────────────────┬───────────┬──────────────┬────────────────┐
    │ Prompt 长度           │ Prefill 方式│ 典型 TTFT    │ 场景            │
    ├──────────────────────┼───────────┼──────────────┼────────────────┤
    │ < 500 tokens         │ CPU 全量   │ 0.4-1.6s     │ Chat, 短问答    │
    │ 500-2K tokens        │ CPU chunked│ 1.6-6.3s     │ 仅 Agent warm   │
    │ 2K-128K tokens       │ FPGA chunked│ 85ms 首 chunk│ 通用, RAG, 长文 │
    │ Agent warm (任意)     │ CPU 增量   │ 增量×3.1ms   │ 前缀复用 ✅     │
    └──────────────────────┴───────────┴──────────────┴────────────────┘

  与文档的差异:
    - §14.E Tier 1 "Prompt < 4K: chunked P=128, TTFT ~395ms" → 误导性数字
      应改为 "首 chunk 完成 395ms, 完整 TTFT = N_chunks × 395ms"
    - CPU prefill 的实用范围是 <500 tokens, 不是 <4K
    - Agent warm start 是 CPU prefill 真正的杀手场景 (增量 prefill 极轻)
    - 移交阈值应从 4K 下调到 ~500-2000 tokens
```

**F. 双缓冲 KV Cache 原子交换 — 机制描述不足 ⚠️**

```
§14.E 描述的 "原子 swap" 机制需要补全:

  当前 HBM 分配 (per session, 32 GB/chip):

    ┌──────────────────────────────────────────────────────────┐
    │ KV buf A (active):           session 当前 decode 使用     │
    │ KV buf B (shadow):           CPU/FPGA prefill 写入        │
    │ Weight cache (SRAM/HBM):     常驻, 不受 swap 影响          │
    │ Expert cache (HBM):          常驻, 不受 swap 影响          │
    └──────────────────────────────────────────────────────────┘

  Swap 时序 (FPGA 侧):

    每个 decode step 结束时:
      1. 检查 "B_ready" flag (来自 CPU via PCIe → Chip 0 → 广播)
      2. 如果 B_ready && decode_step_done:
          a. 硬件交换 A↔B 基地址寄存器 (单周期, 不拷贝数据!)
          b. 清除 B_ready flag
          c. 下一个 decode step 从新 A 读取 KV
      3. 否则继续用当前 A

    关键设计决策:
      - 交换的是地址指针, 不是数据 → 零拷贝, 单周期
      - 只在 decode step 边界 swap → 保证 KV 读取一致性
      - 最坏情况: decode step 耗时 ~1.4ms, swap 需等待当前 step 结束
        → 额外延迟 ≤1.4ms, 可忽略

  多 session 扩展:
    每个 session 独立 A/B 对, session 间 KV 区域不重叠。
    Swap 按 session 独立触发。复杂度 O(S) 在硬件地址生成器中管理。

  边界情况 (需在 RTL 中处理):
    1. CPU prefill 写入 buf B 期间, FPGA 开始读 buf B (swap 过早)
       → 硬件锁: swap 前检查 "B_write_done" flag (CPU 写入完成后置位)
    2. 多个 CPU prefill 同时完成, 多个 B 同时 ready
       → 硬件仲裁: 按 session_id 优先级排队 swap
    3. Session 结束时回收 buf A/B
       → 硬件 KV manager 标记区域为 free, 类似 PagedAttention block 回收

  这个机制在概念上是正确的, 但文档缺少:
    - 基地址寄存器交换的 RTL 实现描述 (kv_dma_bridge.sv 中应包含)
    - B_ready/B_write_done 的 flag 协议 (PCIe 地址映射)
    - 多 session 仲裁逻辑
```

**G. 综合评估与风险分级**

```
┌──────────────────────────────────────┬──────────┬──────────────────────┐
│ 维度                                  │ 结论     │ 关键前提              │
├──────────────────────────────────────┼──────────┼──────────────────────┤
│ A. CPU 算力 (10.5 TFLOPS AMX)        │ 🟢 可行   │ 已充分验证            │
│ B. KV 32 芯片分发                    │ 🟢 可行   │ SERDES 转发路径需补全 │
│ C. TTFT (短 prompt <500 tok)         │ 🟢 可接受 │ ~0.4-1.6s            │
│ C. TTFT (中 prompt 500-2K tok)       │ 🟡 勉强   │ 1.6-6.3s, 仅适合 Agent│
│ C. TTFT (长 prompt >2K tok)          │ 🔴 不可接受│ 必须走 FPGA Tier 2   │
│ D. 并发 CPU prefill + FPGA decode    │ 🟢 可行   │ 核心分区 + 中断策略   │
│ E. CPU→FPGA 移交阈值                 │ 🟡 需修正 │ 从 4K 下调到 ~500 tok │
│ F. 双缓冲 KV swap                    │ 🟢 可行   │ 地址交换, 零拷贝      │
│ G. Agent warm start (增量 prefill)   │ 🟢 最佳场景│ CPU prefill 的杀手应用│
└──────────────────────────────────────┴──────────┴──────────────────────┘

新增 CPU Prefill 专属风险 (补入 §11.A.3 风险矩阵):

  ┌────────────────────────────────────┬───────────┬──────────┬──────────┐
  │ 风险                                │ 概率      │ 影响     │ 等级     │
  ├────────────────────────────────────┼───────────┼──────────┼──────────┤
  │ CPU prefill TTFT 超预期 → 用户流失  │ 高 (60%)   │ 中       │ 🟡 中高  │
  │ (当前文档数字偏乐观, 需修正后重评)   │           │          │          │
  │ CPU AMX/DDR5 在 prefill+decode 并发 │ 低 (15%)   │ 中       │ 🟢 低    │
  │ 下出现非预期带宽争用                 │           │          │          │
  │ Intel AMX 指令集未来不兼容           │ 低 (10%)   │ 中       │ 🟢 低    │
  │ (AMX 是 x86 标准扩展, 不会消失)      │           │          │          │
  │ KV cache 分发路径 RTL bug           │ 中 (30%)   │ 中       │ 🟡 中    │
  │ (SERDES forwarding 逻辑错误)         │           │          │          │
  └────────────────────────────────────┴───────────┴──────────┴──────────┘

需要新增的实验闭合变量 (补入 §11.A.4):

  P0:
    7. CPU prefill + FPGA decode 并发场景的端到端 TTFT
       → 在真实 GNR 服务器 + 4-8 chip FPGA 原型上运行:
         Session A decode (steady) + Session B CPU prefill (variable length)
       → 测量: Session B TTFT, Session A per-step latency 是否受影响
       → 关闭标准: Session A decode 延迟增加 < 5%, Session B TTFT 与解析模型一致

  P1:
    8. KV cache SERDES pipeline forwarding 实际延迟 vs 解析
       → 在 4-8 chip 系统上测量 KV 数据从 Chip 0 广播到所有 chip 的时间
       → 关闭标准: 总分发延迟 ≤ 解析值 × 1.5

  P2:
    9. CPU prefill TTFT 实测 vs 解析
       → GNR 服务器上跑完整 61 层 AMX GEMM, P=128/256/512
       → 对比解析模型: 395ms/790ms/1580ms
       → 关闭标准: 实测 ≤ 解析值 × 1.2
```

**H. 最终裁决**

```
CPU Prefill + FPGA Decode 是否可行?

  可行 ✅, 但适用范围比当前文档声称的窄:

  最佳场景 (CPU prefill 价值最大):
    1. 短对话 (< 500 tokens prompt): TTFT 0.4-1.6s, 零额外成本
    2. Agent warm start: 只 prefill 增量 token, TTFT 极低
    3. 低流量私有部署: Intel SPR → GNR 升级即可获得 6× prefill 加速

  不适用场景 (必须 FPGA Tier 2 或 GPU Tier 3):
    1. 中长 prompt (> 2K tokens): CPU TTFT > 6s, 用户不可接受
    2. 高并发 API 服务: 多个并发 CPU prefill 争抢 AMX 单元
    3. 极致低延迟 TTFT (< 100ms): 即使 GPU 也需要 ≥1 个 A100

  文档需要修正的关键数字:
    1. §4.8.6 场景表: TTFT 从 "0.3-4s" 修正为 "0.4-15s"
    2. §14.E Tier 1 描述: "TTFT ~395ms" → "首 chunk 完成 395ms, 完整 TTFT = N_chunks × 395ms"
    3. CPU→FPGA 移交阈值: 4K → ~500 tokens
    4. 明确标注 Agent warm start 是 CPU prefill 的最佳场景 (增量模式)

  架构判断:
    这是一个正确的方向——CPU prefill 解决了 "短 prompt 不需要额外硬件" 的问题。
    但它不是银弹, 中长 prompt 仍然需要 FPGA 的 chunked prefill 能力。
    三级体系 (CPU Tier 1 → FPGA Tier 2 → GPU Tier 3) 的设计是正确的,
    只是各级的适用边界需要修正。
```


**4.8.5 Prefill 调度策略——与 Decode 共存**

```
推荐: Chunked Prefill (Phase 2 实现)
  512 token/chunk, max 1 chunk between decode steps

  多 session 调度:
    round_robin:
      session_A decode_step
      session_B prefill_chunk_1
      session_C decode_step
      session_A decode_step
      session_B prefill_chunk_2
      ...

  DSP 分配:
    prefill chunk (512 tok): DSP 全速, ~85ms
    decode step:            DSP 按 §4.4.1 加权利用率 ~50%
    → prefill chunk 期间 decode 被暂停 ~85ms
    → 对于 agent 场景 (B=1, 每轮输出 < 500 tok),
      每 85ms prefill 暂停不影响用户体验

  备选 (Phase 3+): DSP 分区调度
    → 70% DSP 给 decode (保证延迟)
    → 30% DSP 给 background prefill
    → RTL 需支持 DSP 阵列分区间隔, 额外工作量
```

---

### 4.8.x Chip 0 prefill 入口瓶颈分析

§4.8 给出了 chunked prefill 的基础模型。在 §4.6.1 的并发优化全部生效后，仿真和 disagg 模式下的实测都暴露出一个稳定的瓶颈：**Chip 0 的 admission rate 封顶了系统的请求接入速率**，而不是 decode 算力或 HBM 带宽。

本节量化这个瓶颈并评估两条架构级优化路径。

**4.8.x.1 Chip 0 为什么是 prefill 的串行点**

```
Chip 0 承载 layer 0-1 和 Embedding 查表。每个新请求必须依次完成：
  1. Host CPU tokenize
  2. PCIe DMA 把 prompt token 送到 Chip 0
  3. Embedding lookup（Chip 0 上单周期）
  4. 跑 layer 0-1 的第一个 chunk
  5. Pipeline forward 把 chunk 推到 Chip 1，Chip 0 才能接下一个 chunk

单 chunk 在 Chip 0 上的周转时间:
  per_layer_us @ chunked prefill (P=128, fp4+sparse) = 6,740 us
  Chip 0 承载层数                                       = 2
  per_chunk_us                                          = 13,480 us
  chunks/s                                              = 74.2

P=512 时（每个 request 4 个 chunk）:
  admission_rate                                        = 18.5 req/s
```

这正是 §4.6.1 实测的 disagg (4P+2D) 高负载下只接纳 ~1.7 req/s 的根本原因——即使部署 4 个 prefill 服务器，每个的 Chip 0 也只能合计接 18.5 req/s，Poisson 突发流量会迅速排队。

**4.8.x.2 两条架构级优化路径 — 解析模型**

通过新增的 `PipelineEngine.chip0_admission_rate()` 方法量化：

```
┌────┬──────────────────────────────────────┬───────────┬──────────┬────────┬────────┐
│ #  │ 配置                                  │ per_chunk │  req/s   │  tok/s │ gain   │
├────┼──────────────────────────────────────┼───────────┼──────────┼────────┼────────┤
│ A  │ Baseline（单 chip 0, embedding 在片） │  13.48 ms │    18.5  │   9495 │   1.0× │
│ B  │ Embedding 卸载到 host CPU             │  13.43 ms │    18.6  │   9531 │   1.0× │
│ C  │ Pipeline Cloning ×2 (16+16 chips)     │  13.48 ms │    37.1  │  18991 │   2.0× │
│ D  │ Pipeline Cloning ×2 + Embedding 卸载  │  13.43 ms │    37.2  │  19061 │   2.0× │
│ E  │ Pipeline Cloning ×4 (8+8+8+8 chips)   │  13.48 ms │    74.2  │  37981 │   4.0× │
│ F  │ Pipeline Cloning ×4 + Embedding 卸载  │  13.43 ms │    74.5  │  38123 │   4.0× │
└────┴──────────────────────────────────────┴───────────┴──────────┴────────┴────────┘
```

**4.8.x.3 端到端仿真验证（Agent 8 req/s, O=1024）**

把 Pipeline Cloning 接入 `ServingSimulation`（`--pipeline-clone N`），不同 clone 数对比：

```
                          clone=1    clone=2    clone=4
                          ────────   ────────   ────────
  Accept rate              52.7%      50.1%      54.0%
  Output TPS (tok/s)        8,526      7,752      8,389
  TTFT P50 (ms)               527        402        404    ← 关键改善
  TTFT P95 (ms)             1,150        543        418    ← ×2.7 改善
  Avg active session           23         35         36
  Avg batch size              7.1        4.8        2.9

高负载下（Agent 20 req/s, O=1024）:
                          clone=1    clone=2    clone=4
                          ────────   ────────   ────────
  Accept rate              25.4%      18.4%      19.1%
  Output TPS (tok/s)        8,515      5,930      6,066
  TTFT P50 (ms)               550        435        390
  TTFT P95 (ms)             2,108        615        429    ← ×4.9 改善
  Avg active session           17         43         66
```

**4.8.x.4 实测 vs 解析的不一致 — 解读**

```
解析模型：Pipeline Cloning ×2 应该让 admission 翻倍 → 接受率应该上升
实测：    Pipeline Cloning ×2 反而让 Accept rate 略下降 (52.7% → 50.1%)

原因：    Pipeline Cloning 把 32 chip 切成两条 pipeline,
          每条 pipeline 的 decode peak 算力减半（DSP 总数不变但分给两条流水线）。
          虽然 prefill 接纳率翻倍，但每条 pipeline 的 decode 处理能力也减半。

          在 Output TPS 上：
            clone=1: 一条 pipeline 跑 8526 tok/s (基本饱和 17,445 的 49%)
            clone=2: 两条各跑 ~3876 tok/s (合计 7752, 各饱和度 44%)
            clone=4: 四条各跑 ~2097 tok/s (合计 8389)

          所以 Pipeline Cloning 的真正价值是：
            ✓ 把 TTFT 从 P95 2.1s 降到 P95 0.4s (×5 改善)
            ✓ 把可服务并发 session 数提升 (17 → 66, ×4)
            ✗ 不会显著提升聚合 throughput (上限被 decode peak 锁定)
```

**4.8.x.5 各项优化的实际作用**

```
Embedding 卸载 (B vs A):
  Embedding 是 SRAM lookup, 每 chunk ~50 us。
  从 13,480 us 里省 50 us = 0.4%。
  结论：不是真瓶颈。Chip 0 的瓶颈是 2 层 MLA+MoE 的计算量,
       不是 embedding/tokenize 这步。
       Embedding 卸载不值得引入额外的 PCIe 往返 + host 协调复杂度。

Pipeline Cloning ×2 (C vs A):
  把 32 chip 切成两条独立 pipeline (各 16 chip, 4 layer/chip),
  prefill 接纳率翻倍, 同时 TTFT 大幅改善 (×3 P95 改善)。

  代价分析:
    HBM 每片占用: 0.7 GB 权重 + 22 GB KV = 22.7 GB
                  → 变成 ~1.2 GB 权重 + 21 GB KV (每条 pipeline 各持完整 384 专家,
                    每片权重翻倍)
                  仍在 32 GB HBM 物理上限内, OK。
    Decode 延迟成本: 单 token 延迟上升 ~10-20%, 因为每 chip 承载 4 层而不是 2 层
                   (单 chip stage 时间变长), 但 pipeline 深度减半 (32→16 chip)
                   抵消了一部分。
    无硬件改动: 纯部署/调度决策, RTL 无需变更。

Pipeline Cloning ×4 (E vs A):
  prefill 接纳率 ×4, 代价是单 token decode 延迟 ~30% 上升。
  每片承载 8 层, 计算密度变高。
  推荐场景: TTFT 预算宽松、流量峰值高 (API 型业务)。
```

**4.8.x.6 与 §4.6.1 优化的叠加效果**

```
单服务器端到端吞吐演进路径 (Agent 工作负载):

  Stage 1: baseline 配置
    accept_rate    28%
    output_tps    1,000 tok/s
    bottleneck     chip 0 prefill (18 req/s) + decode 调度地板

  Stage 2: §4.6.1 KV 扩容 + 移除调度地板 + Hot Expert Replication
    accept_rate    88%
    output_tps    5,800 tok/s
    bottleneck     chip 0 prefill (18 req/s)

  Stage 3: + Pipeline Cloning ×2 (§4.8.x)
    accept_rate    ~50% (高 arrival 下)
    output_tps    ~7,800 tok/s
    TTFT P95       从 1.15s 降到 0.54s (×2.1 改善)
    bottleneck     decode peak DSP

  Stage 4: + Pipeline Cloning ×4
    accept_rate    类似
    output_tps    封顶 ~8,400 tok/s
    TTFT P95       从 2.1s 降到 0.43s (×4.9 改善, 高负载场景)
    bottleneck     硬件物理上限
```

**4.8.x.7 部署建议**

针对 FPGA 集群作为推理服务平台的生产部署：

1. **总是启用 §4.6.1 优化组合**（KV 扩容 + 去调度地板 + Hot Expert Replication）。零硬件成本，吞吐提升 ×6。

2. **默认使用 Pipeline Cloning ×2**：对任何服务 5 个以上并发 agent session 的部署都建议开启。**主要收益是 TTFT 大幅下降（P95 改善 ×3-5）**，accept rate 略降但用户体验显著提升。纯部署期决策（无 RTL 改动）。

3. **Pipeline Cloning ×4 适用于高流量 API 场景**：TTFT 预算宽松、并发 session 数 50+ 时启用。代价是 30% 单 token 延迟增加。

4. **跳过 Embedding offload** — 不是真瓶颈，复杂度收益不成正比。

实施成本：Pipeline Cloning 需要权重布局编译器 (§5.3) 支持输出按 pipeline 切分的权重映射。这是一个低风险的软件任务，估算 ~1 人月。

---

### 4.9 Agent 场景适配分析

Agent 的工作负载与简单 chatbot 有本质差异, 这些差异恰好放大了 FPGA 的几个架构优势, 同时暴露了 Prefill 短板在特定子场景下的影响。

**4.9.1 Agent 工作负载特征**

```
典型 Agent loop:

  System prompt (5-20K) + Tool definitions (2-5K)  ← 首次 prefill
      ↓
  Turn 1: Full context → LLM → Tool call → Tool result (1-5K new tokens)
  Turn 2: Full history   → LLM → Tool call → Tool result
  ...
  Turn N: Full history   → LLM → Final answer

关键差异 vs 简单 chatbot:

  ① 上下文随轮次单调增长 (最终可达 32K-128K)
  ② 每轮输出很短 (<500 tokens, 通常是工具调用/简短推理)
  ③ 前缀不变 → KV Cache 绝大部分可复用 (第 2-N 轮)
  ④ 天然 B=1 (agent 串行推理, 不能并行多个方案)
  ⑤ 多轮交互 → 总 session 时间长 (分钟级)
```

**4.9.2 FPGA 在 Agent 场景的三个结构优势**

**优势 ①: KV Cache 前缀复用——增量 Prefill 极轻**

```
第 1 轮 (cold start):
  → 完整 prefill system prompt + user query (~10-30K tokens)
  → TTFT: 1.7-5.0s (完整) / 125ms (首 chunk)

第 2-N 轮 (warm, 占 agent 推理的 >90%):
  → 只 prefill 新增 token (上一轮 LLM 输出 200 + Tool output 2-5K)
  → 有效 prefill ≈ 2-5K tokens → TTFT < 1s
  → 其余前缀 KV Cache 硬件直接复用, 零计算

  GPU (vLLM prefix caching):
    同样可复用, 但需软件 block table 查找/验证/复制。

  FPGA:
    硬件 KV 地址生成器: {session_id, layer_id, seq_pos} → HBM 物理地址
    前缀匹配 = 地址偏移, 零延时, 零 CPU 参与。
    多轮 agent 的增量 prefill 路径比 GPU 更短。
```

**优势 ②: Agent Decode 天然 B=1, FPGA 最优点**

```
Agent 每次推理:
  → 生成 "是否调工具 + 工具名 + 参数" 或 "简短推理"
  → 通常 <500 token
  → 天然不需要 batch

GPU 在 B=1 下: Tensor Core ~3% 利用率, 几乎全部闲置
FPGA 在 B=1 下: DSP ~50% 加权利用率, 每一分钱在干活

Agent 场景的 decode 总量可能不大 (短输出去重),
但请求频率高 (多轮交互), 低成本 per-token 累积效应显著。
```

**优势 ③: 长 session KV Cache 硬件管理——GPU 的软件瓶颈**

```
128K context 的长 agent session:

  GPU (vLLM PagedAttention):
    → 128K / 16 blocks = 8K 个 KV block
    → 每次 decode 需查表寻址 8K 个 block
    → 软件 block table 管理随 session 数线性增长
    → CPU 分配/释放 block 的负载不可忽略

  FPGA:
    → 硬件地址生成器, 组合逻辑
    → 每 token KV 地址解析 < 10ns
    → 滑动窗口 (128 位置) 硬件自动淘汰
    → 零 CPU, 零软件, 随 context 增长零额外开销

  长 session 下 FPGA 的 KV 管理优势从 "可忽略" → "可测量"。
```

**4.9.3 Agent 场景的两个劣势**

**劣势 ①: 首次 Prefill 冷启动延迟**

```
首次 agent 请求 (system prompt + tools + history):
  短 (5K):  TTFT ~1s, 可接受
  中 (20K): TTFT ~3.3s (完整) / 125ms (首 chunk)
  长 (128K): TTFT ~21s (完整) / 125ms (首 chunk)

  Chunked prefill 对冲:
    → 首 token 仍 125ms 可见
    → agent 拿到首 token 即可开始推理/工具调用
    → 不需要等完整 prefill 完成

  Agent 通常从第 2 轮才进入高频交互,
  首次 prefill 的冷启动延迟对整体体验影响有限。
```

**劣势 ②: 大规模并发 Agent 受限**

```
1 FPGA = 1-2 并发 session × 30 FPGA = 30-60 并发 agent

  ┌────────────────────┬──────────────┬──────────────┐
  │ 场景                │ FPGA (30卡)  │ H200 (8卡)   │
  ├────────────────────┼──────────────┼──────────────┤
  │ 企业内 10 agent     │ 绰绰有余      │ 绰绰有余      │
  │ 部门 50 agent       │ 刚好够        │ 绰绰有余      │
  │ SaaS 1000 agent     │ ✗ 不够       │ 可            │
  │ 每 agent 独立部署    │ ✓ 天然隔离    │ △ 需 MIG 切分 │
  └────────────────────┴──────────────┴──────────────┘

  FPGA 的 30-60 并发覆盖企业私有部署的 agent 场景足够。
  多套集群扩展靠物理套数而非 batch——天然租户隔离。
  SaaS 平台的高并发多租户是 GPU/Groq 的领地。
```

**4.9.4 Agent 场景判定矩阵**

```
┌──────────────────────────────┬──────┬──────┬──────────────────────┐
│ Agent 维度                    │ GPU  │ FPGA │ 说明                  │
├──────────────────────────────┼──────┼──────┼──────────────────────┤
│ 首次长 context prefill       │ ★★★★ │ ★★   │ Chunked 首 token 可接受│
│ 增量 prefill (前缀复用)      │ ★★★  │ ★★★★ │ FPGA KV Cache 零软件   │
│ B=1 decode (agent 串行)     │ ★    │ ★★★★ │ FPGA 架构决定性优势     │
│ 长 session KV 管理           │ ★★   │ ★★★★ │ 硬件 > 软件, 长 context │
│ 多 agent 并发 (>100)         │ ★★★★ │ ★    │ GPU 绝对优势           │
│ 少 agent 并发 (<50)          │ ★★★  │ ★★★★ │ FPGA 最优区间           │
│ 每轮延迟 (短输出 <500 tok)   │ ★★★  │ ★★★  │ 相近                   │
│ 数据隔离 (金融/医疗 agent)   │ ★★   │ ★★★★ │ FPGA 物理隔离           │
│ 多模态 agent                 │ ★★★★ │ ★    │ 需额外 NPU             │
└──────────────────────────────┴──────┴──────┴──────────────────────┘

结论:
  纯文本 Agent × 企业私有部署 × 长 session × <50 并发 → FPGA 占优
  多模态 Agent × SaaS 多租户 × 短 session × >100 并发 → GPU 占优
```

**4.9.5 Agent 场景的商业故事**

```
中国企业的 agent 部署面临三重约束:

  ① 数据不外传: 金融交易记录、医疗病历、政府公文
     → 公共 API (DeepSeek/GPT) = 不可用
     → 必须是私有部署硬件

  ② 高端 GPU 不可获取: H100/B200 管制, Ascend 排队
     → 私有 GPU 集群 = 买不到
     → FPGA = 可获取

  ③ 长 session 多轮交互: agent 不是一问一答
     → KV Cache 管理成为计算之外的瓶颈
     → FPGA 硬件 KV 管理 = 免 CPU, 免软件

  Agent 是企业 AI 的终极形态。FPGA 在 agent 场景的
  结构优势 (B=1 decode, KV 前缀复用, 硬件缓存管理)
  比 chatbot 场景更突出——不只是"买不到 GPU 的备胎",
  而是"对于 agent 这个特定工作负载, FPGA 架构上就是更优"。
```


### 4.9.6 Coding Agent: FPGA 的杀手场景

**为什么 coding agent 和通用 agent 有本质区别？**

通用 agent 的工具调用稀疏——偶尔搜一下、查一下数据库。Coding agent
的工具调用极度密集——生成代码 → 编译 → 读错误 → 修复 → 再编译 → 读测试结果,
一回合可能 3-5 次 prefill/decode 交替。

```
通用 agent 的每轮模式:
  decode (决定做什么, ~100 tok) → 工具执行 (秒级等待) → prefill (结果, 1-5K tok)
  间隔长, decode 一次, prefill 一次, 用户对延迟不敏感

Coding agent 的每轮模式:
  decode (生成函数, ~200 tok) → 执行 (LSP/编译/test, 毫秒-秒)
  → prefill (错误信息, ~500B-2K tok)
  → decode (修复代码, ~100 tok)
  → 执行 → prefill (测试通过/失败, ~500B)
  → decode (继续写下一个函数)
  ...
  一回合 3-5 次 prefill/decode 交替, 每次间隔短
```

这个差异放大了 FPGA 的三个结构优势：

**优势 ①: 高频 prefill/decode 切换——GPU 的调度延迟被反复惩罚**

```
每次切换:
  FPGA: DSP 寄存器重配 → < 1μs (组合逻辑写一个配置字)
  GPU:  CPU scheduler → CUDA kernel launch → SM 上下文切换 → 毫秒级

Coding agent 一回合 3-5 次切换:
  FPGA 累计切换开销:  5 × 1μs = 5μs      (可忽略)
  GPU 累计切换开销:   5 × 1-5ms = 5-25ms  (累积到用户感知)

高频工具调用 (MCP、LSP、编译器) 会在 2026-2027 成为 agent 主流。
FPGA 的零切换开销不是"nice to have", 是 coding agent 的基础要求。
```

**优势 ②: 代码上下文的 KV cache 极其稳定——硬件管理 vs 软件管理的分水岭**

```
Coding agent 的上下文构成:
  ├── System prompt (角色 + 规则):       5-10K,  整个 session 不变
  ├── Project context (文件树、类型定义): 20-50K,  切换文件时部分更新
  ├── Conversation history:              增长中,  每次 prefill 追加
  └── Tool outputs (编译错误、LSP):       <2K,    每次丢弃旧结果

前缀稳定性: ~80-90% 的 KV cache 在整个 session 期间不变。

GPU (vLLM PagedAttention):
  → block table 仍要遍历全部 KV blocks (包括不变的前缀)
  → 30K 前缀 = ~2000 blocks, 每步查表 ~50μs
  → coding agent 每回合 3-5 次 decode → 累计查表 150-250μs

FPGA:
  → 硬件 KV 地址 = base + layer * stride + seq_pos * kv_bytes
  → 前缀匹配 = 地址偏移, 零软件, 零遍历
  → per-token KV 地址解析 < 10ns, 与 context 长度无关
```

**优势 ③: IDE 延迟预算极紧——确定性延迟 > 平均延迟**

```
IDE 场景的用户延迟预期:
  < 200ms:  "瞬间"——代码补全级别
  < 500ms:  "流畅"——agent 单步响应
  < 2s:     "等待"——agent 多步推理完成
  > 2s:     "卡顿"——用户开始怀疑出 bug 了

GPU 的非确定性来源:
  → KV block 碎片化触发 GC pause:        10-50ms, 随机
  → CUDA kernel 调度排队:                 1-5ms,  随 GPU 负载变化
  → vLLM continuous batching 重组:        5-20ms, 请求越多越频繁

FPGA 的确定性来源:
  → 硬件 KV 地址生成:      < 10ns (组合逻辑, 无排队)
  → 流式 pipeline:          1.4ms/token (确定性的, 无 stall)
  → 无 GC, 无 block table, 无 kernel launch
```

**Coding agent 帮 FPGA 避开了什么短板？**

```
FPGA 的三个主要短板在 coding agent 场景下自然规避:

  1. 多 agent 并发上限:
     AI IDE 场景: 一个 developer = 最多 1-2 个并行 agent session
     (一个在写后端, 一个在写前端, 极端情况)
     → FPGA 的 30-60 并发完全不被挑战

  2. 多模态需求:
     Coding agent 是纯文本交互 (代码 + 编译错误 + LSP + git diff)
     → 不需要 ViT/CLIP/视觉编码器

  3. 冷启动 prefill 延迟:
     IDE 打开项目时可以后台预热:
     → 项目打开时, FPGA 后台 prefilling system prompt + project context 的 KV cache
     → 用户开始写第一个 prompt 时, KV cache 已经就绪
     → 用户感知 TTFT ≈ 增量 prefill 时间 (< 1s)
     → Cold start 被 "warm start" 策略有效化解
```

**4.9.6.1 Coding Agent 商业模式："AI IDE 盒子"**

```
定位: 不是卖 FPGA 卡, 是卖 "coding agent 专用推理节点"。

主力硬件配置 (HBM-Only, 32 芯片拉满 decode 带宽):

  ┌─────────────────────────────────────────────────────────┐
  │ FPGA Coding Agent Node                                  │
  │                                                         │
  │  FPGA: 32 片 Agilex 7 M, 8 卡 × 4 芯片/卡              │
  │        HBM-Only 配置 (每片 32 GB HBM2e)                 │
  │        权重全在 HBM, pipeline-parallel 32 芯片分布       │
  │        带宽/层: 920 GB/s ÷ 2 层/芯片 = 460 GB/s/层      │
  │        芯片物料: 32 片 × ¥1.8万 = ¥57.6万                │
  │        卡级物料 + 服务器 BOM: ~¥133万                    │
  │                                                         │
  │  服务器: Dual Xeon GNR 6980P (含在 BOM 中)              │
  │          256 GB DDR5, CPU prefill 能力 (Tier 1)          │
  │                                                         │
  │  聚合吞吐: 5,800-8,500 tok/s (B≥4, §4.6.1 优化后)      │
  │  B=1 吞吐: ~720 tok/s (单 session decode)              │
  │                                                         │
  │  服务能力:                                               │
  │    并发: 30-60 个 coding agent session (含 Pipeline      │
  │          Cloning ×2 可扩展到 50+ session)               │
  │    延迟: per-token 1.4ms (确定性的, 流式 pipeline)      │
  │    上下文: 支持 128K context, KV cache 硬件管理          │
  │    安全: 代码永不出企业网络, 物理隔离                    │
  │                                                         │
  │  对比 950PR 8 卡 (¥200万):                               │
  │    BOM:     ¥133万 vs ¥200万                               │
  │    有效带宽: 29.4 TB/s vs ~11 TB/s → 2.7× at B=1           │
  │    吞吐:    5,800-8,500 vs 2,500-4,000 tok/s → 2.1-2.3×    │
  │    B=1:     ~720 vs ~200-300 tok/s                          │
  │    → 不靠"更便宜"竞争, 靠架构带宽效率取胜                    │
  └─────────────────────────────────────────────────────────┘

  降级选项 (HBM+DDR 经济配置, §二点七):
    对于小团队 (5-10 人), 可选 5-8 片 HBM+DDR:
      → 芯片物料 ¥17.5万, DDR 存权重, HBM 跑 KV cache
      → 吞吐 800-1,500 tok/s, 服务 10-20 个 agent session
      → DDR 降本是 FPGA 架构独有的灵活性, GPU/NPU 无此路径
    DDR 是降本路径, 不影响主力架构的带宽论证。

目标客户与决策链:
  ┌──────────────────────┬──────────────────┬──────────────────────┐
  │ 客户类型               │ 痛点              │ 决策者                │
  ├──────────────────────┼──────────────────┼──────────────────────┤
  │ 金融科技公司           │ 代码不能上公网    │ CTO + 安全部门         │
  │ 军工/政府 IT           │ 封闭网络, 国产化   │ IT 采购 + 安全审批      │
  │ 互联网公司 (中型)      │ GPU 排队/管制     │ 工程 VP               │
  │ 外包/软件服务商        │ 客户要求数据本地   │ 项目交付负责人         │
  │ 高校/研究所            │ 预算有限, 需私有   │ 实验室主任             │
  └──────────────────────┴──────────────────┴──────────────────────┘

中国市场对标:
  2025-2026 年国内 coding agent 已进入快速增长期:
    - 通义灵码 (阿里): 企业版私有部署方案
    - CodeBuddy (腾讯): 内部大规模使用
    - 商汤 Raccoon: 代码生成 + 审查
    - 各种基于 DeepSeek V3/V4 的私有 coding agent

  所有这些方案面临同一个后端问题:
    → 用公共 API: 代码安全不可接受
    → 用 H100/H200: 买不到
    → 用 Ascend: 排队, 且 fp4 原生不支持, 性价比差
    → FPGA coding agent node = 唯一同时满足 "可获取 + fp4 原生 + 私有部署" 的方案
```

**4.9.6.2 反驳预期质疑**

```
质疑 1: "Cursor/Copilot 用的都是云端 GPU, 用户不介意代码上传"

  回应:
    a) Cursor/Copilot 的个人用户和付费企业用户是两类群体。
       企业用户 (尤其是金融/军工/外包) 有明确的合规需求,
       "代码不出企业网络" 是硬性约束——这不是用户介不介意的问题,
       是合规过不过得了的问题。

    b) GitHub Copilot 2025 年推出 "Copilot Enterprise with data residency"
       正是因为企业客户要求数据本地化。这证明了 "代码不上传" 的市场需求是真实的。

    c) 中国市场的特殊性: 企业用 Cursor/Copilot 本身就有数据出境风险。
       同时国内 GPU 供应极度受限。FPGA coding agent node 同时解决
       两个在国内无法回避的问题: 数据安全和硬件获取。

质疑 2: "coding agent 的 TTFT 要求高, CPU prefill 太慢"

  回应:
    a) Warm start 是主打策略: IDE 打开项目时后台预热 system prompt +
       project context 的 KV cache。用户开始交互时, 前缀已就绪,
       只需增量 prefill (< 1s TTFT)。

    b) 即使冷启动, coding agent 的首次响应也有 "进度条" 心理模型——
       用户习惯了 "索引项目..." 这样的等待。不像 chatbot 那样即时。

    c) Tier 2 FPGA chunked prefill (P=512, ~85ms/chunk) 对 20K system prompt
       给出 ~3.4s 首 chunk 或 ~3.4s 全量 TTFT, 对标 "打开项目" 的预期可接受。

质疑 3: "写代码的模型能力比推理硬件更重要, DeepSeek V4 的 coding 能力不如
       Claude/GPT"

  回应:
    a) 模型差距在缩小: DeepSeek V4 的 coding benchmark 已接近 GPT-4o 水平。
       DeepSeek V5 预期 2025 年底-2026 年发布, coding 能力大概率进一步缩小差距。

    b) FPGA 的架构不绑定特定模型: 只要是 fp4 MoE + MLA 架构的模型都能部署。
       DeepSeek V5/V6, Qwen 3 MoE, 或任何未来开源 coding 模型都可以。

    c) "够用" 阈值: coding agent 不需要模型解决 IMO 级别的数学题。
       需要的能力是: 理解项目上下文 → 生成合理代码 → 理解编译错误 → 修复。
       这个任务对模型能力要求远低于 "赢得编程竞赛"。

质疑 4: "Per-token 1.4ms, 生成一个函数 200 tokens = 280ms, 太慢了"

  回应:
    a) Coding agent 的交互模式是流式的: 用户看到第一行代码就开始阅读,
       不需要等整个函数生成完。1.4ms 意味着 ~714 tok/s——比人的阅读速度快得多。

    b) 实际延迟感知: 200 tokens × 1.4ms = 280ms, 加上 prefill 增量 ~500ms,
       总共 < 800ms——在 IDE 的 "流畅" 感知范围内。

    c) 对比: 人的思考 + 打字时间通常是几秒到几十秒。
       Agent 的瓶颈在推理质量 (生成的代码对不对), 不在 per-token 延迟。
```

**4.9.6.3 Coding Agent 的终局判断**

```
Coding agent 是所有 agent 类别中, 对 FPGA 架构最有利的:

  ┌──────────────────────────────────────────────────────────┐
  │                          │ Chatbot │ 通用 Agent │ Coding Agent │
  ├──────────────────────────┼─────────┼───────────┼─────────────┤
  │ Prefill/decode 交替频率   │ 1:1     │ 1:1~1:2   │ 1:3~1:5 🔥  │
  │ KV cache 前缀稳定性       │ 低      │ 中         │ 极高 🔥      │
  │ 并发 session 数/用户      │ 1       │ 1-2       │ 1-2 🔥       │
  │ 延迟敏感度               │ 中      │ 低-中      │ 高 🔥        │
  │ 多模态需求               │ 低      │ 中-高      │ 低 🔥        │
  │ 数据隐私需求             │ 中      │ 高         │ 极高 🔥      │
  │ 模型能力门槛             │ 中      │ 高         │ 中-高        │
  │ ─────────────────────── │ ─────── │ ───────── │ ─────────── │
  │ FPGA 适配度              │ ★★★     │ ★★★★      │ ★★★★★       │
  └──────────────────────────────────────────────────────────┘

  终局判断:
    "私有部署 + Coding Agent" 不是 FPGA 在找不到 GPU 市场后的退路,
    而是 FPGA 推理架构从一开始就隐含设计的目标场景。

    在这个场景里, FPGA 不是 "够用且更便宜的替代品",
    而是在关键维度 (切换频率、KV 确定性、硬件隔离) 架构上占优的方案。

    如果能同时拿下 "金融/军工的代码安全合规" + "互联网公司 GPU 供应不足的增量",
    coding agent 推理硬件在中国市场的 TAM 估算为:
      - 10 万 developer × 30% AI agent 渗透率 = 3 万并发 agent
      - 每套 HBM-Only 32 芯片 (8 卡) 服务 30-60 session → 1,000 套
      - 1,000 套 × ¥133万/套 (芯片+卡级+服务器) = ¥13.3 亿 (仅 coding agent)
      - 小团队可选 HBM+DDR 降级 (5 片 ¥9万), 扩大可及市场
      - 扩展到通用 agent + 客服 + RAG → 叠加市场
```


