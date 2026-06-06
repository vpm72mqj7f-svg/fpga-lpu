### 4.10 Embedding / lm_head 串行瓶颈分析

评审质疑: §4.1 将 Embedding (1.85 GB) 放在 Node 0, lm_head (1.85 GB) 放在 Node 3。这两个最大的单 tensor 不在 DSP systolic array 的覆盖范围内——Embedding 是查表, lm_head 是 129K-vocab 稠密矩阵乘。它们会不会成为流水线瓶颈？

**4.10.1 Embedding: 可忽略的开销**

```
Embedding lookup (token_id → 7168-dim vector):

  单 token:
    HBM 地址 = base + token_id × 7168 × 2B (fp16)
    读取: 7168 × 2B = 14 KB, 一次顺序 burst
    HBM 延迟: ~100ns (tRC) + 14KB/920GB/s ≈ 15ns = 115ns

  Prefill 4K tokens:
    4K × 115ns ≈ 0.46 ms
    实际: 4K token 中大量重复, 重复 token 可缓存
    0.46ms / 800ms TTFT = 0.06% — 可忽略

  Decode 每步 1 token:
    115ns / 160μs per-token decode = 0.07% — 可忽略

结论: Embedding 不是矩阵乘, 但不需要是。
      纯 HBM burst 读, 开销在所有场景下可忽略。
      DSP 不参与, 也不需要参与。
```

**4.10.2 lm_head: 真正的瓶颈候选**

```
lm_head = Linear(7168 → 129,280), 926M params = 1.85 GB fp16

  Node 3 单卡 (TP=7):
    HBM:  1.85 GB / 7 / 920 GB/s = 287 μs
    DSP:  129,280 × 7,168 / 7 / 8.44T = 15.7 μs
    → lm_head 单 token: ~290 μs (HBM-bound)

放在流水线中:
  Node 3: [L45-60: ~150μs] + [lm_head: 290μs] = 440 μs/token
  Node 2: [L30-44: ~150μs]
  Node 1: [L15-29: ~150μs]
  Node 0: [L0-14:  ~150μs]

  → Node 3 比其他节点慢 290μs
  → 流水线吞吐由最慢节点决定: 1/440μs ≈ 2,270 tok/s
  → 这不是 "27 卡干等", 是流水线各段由最慢段定步调
```

**4.10.3 缓解: lm_head 与下一 token 流水线重叠**

```
  时刻 0:   Token 1 L45-60 @ Node 3 (150μs)
  时刻 150: Token 1 lm_head @ Node 3 (290μs)
             Token 2 L0-14 @ Node 0 (150μs)  ← 已开始, 不等待!
  时刻 300: Token 2 L15-29 @ Node 1
  时刻 440: Token 1 lm_head 完成, 首 token 可见
             Token 2 L30-44 @ Node 2
  时刻 590: Token 2 L45-60 @ Node 3
             Token 2 lm_head @ Node 3 (与 Token 3 L0-14 重叠)
  ...

流水线气泡:
  Node 3: 440μs/token, 利用率 150/440 = 34%
  Node 0-2: 等待 Node 3 完成, 利用率 ~34%

对比 GPU 8×H100 pipeline parallelism:
  同样有 bubble (各段不均匀), 这不是 FPGA 特有的问题。
```

**4.10.4 MTP (Multi-Token Prediction): lm_head 的意外救星**

```
V4 Pro MTP 一次预测 2-4 个后续 token。

MTP 下 lm_head 从 vector-matrix → matrix-matrix:

  B=4 (4 hidden states):
    DSP:  4 × 129,280 × 7,168 / 7 / 8.44T = 62.8 μs
    HBM:  1.85 GB / 7 / 920 GB/s = 287 μs  (权重读一次, 4 路共享)
    总延迟: ~350 μs (仅比 B=1 多 60μs)

  每 token 摊销: 350/4 = 87.5 μs (vs B=1 的 290 μs)
  → MTP 将 lm_head 的每 token 成本降到 ~30%
  → 流水线瓶颈从 440 → 238 μs/token
  → 吞吐从 ~2,270 → ~4,200 tok/s

MTP 不是加剧 lm_head 瓶颈, 而是缓解它。
大 batch 下 lm_head 恰好是 DSP 利用率可以提升的场景。
```

**4.10.5 更激进的方案: 分布式 lm_head (Phase 3+)**

```
将 lm_head 切分到全部 30 卡:

  vocab 129,280 切 30 份: 每卡 ~4,309 个 token (27 MB fp16)
  每卡跑本地 4,309 × 7168 矩阵乘 → top-k argmax
  → 30 卡 All-Reduce 归约 top-k 结果

  延迟: 4,309 × 7,168 / 8.44T = 3.7 μs (DSP)
        + HBM 27 MB / 920 GB/s = 29 μs
        + All-Reduce ~10 μs
        ≈ 43 μs

  → 比 Node 3 独占 (290μs) 快 6.7×
  → 需要 Phase 3 实现 (额外 RTL + 自定义 All-Reduce)

  代价: 每卡多存 27 MB lm_head 权重 (HBM 余量充足)
```

**4.10.6 坦率差距**

```
┌──────────────────────┬──────────────────┬──────────────────┐
│ lm_head (decode B=1) │ 30 FPGA           │ 8×H100            │
├──────────────────────┼──────────────────┼──────────────────┤
│ 瓶颈                  │ HBM (Node 3 独占) │ HBM (但 GPU 更大)  │
│ 单 token 延迟          │ ~290 μs          │ ~550 μs (HBM)     │
│ 缓解                   │ MTP batch → 88μs │ batch 更大 → 更低  │
│ 分布式 (全卡参与)       │ Phase 3, ~43μs   │ 天然 TP             │
└──────────────────────┴──────────────────┴──────────────────┘

GPU 在 lm_head 的优势来自更大的 HBM 带宽 + 天然 TP。
H100 8 卡总 HBM 带宽 26.8 TB/s vs FPGA 30 卡 27.6 TB/s —
但 GPU 权重只需存一份 (8 卡共享 HBM), FPGA 每卡存自己的。
lm_head 1.85 GB 在 GPU 上 TP=8 时每卡 0.23 GB,
HBM 读只需 0.23/3.35 = 69 μs, 快于 FPGA 的 287 μs。

坦率承认: lm_head 在非分布式模式下是 FPGA 的弱势。
但在 MTP (batch≥4) 场景下, 每 token 摊销成本已接近 GPU。
分布式 lm_head (Phase 3) 可完全消除这一劣势。
```

---

### 5.0 IP 复用策略与工作量核算

评审质疑 13 个模块仅分配 50 人月是否可行。关键答辩: **大部分基础设施来自 Intel 硬核 IP 或外部采购, 仅有推理特有的数据路径需自研。**

```
┌───────────────────────────────┬──────────────────┬──────────┬──────────┐
│ 模块                           │ 实现方式           │ 来源      │ 自研人月  │
├───────────────────────────────┼──────────────────┼──────────┼──────────┤
│ PCIe 5.0 x16 Endpoint         │ R-Tile Hard IP    │ Intel    │ 0        │
│ PCIe DMA (Scatter-Gather)     │ Intel DMA IP      │ Intel    │ 0.5      │
│ HBM2e 控制器 (2048-bit)        │ Avalon-MM HBM IP  │ Intel    │ 1.0      │
│ F-Tile 200GbE MAC/PCS/FEC     │ F-Tile Hard IP     │ Intel    │ 0        │
│ RoCE v2 RDMA + DCQCN + PFC    │ 外包采购            │ 外部IP   │ 0 (¥1M)  │
│ 推理载荷编解码 (on RDMA)        │ 自研 RTL           │ —        │ 1.0      │
│ fp4×fp8 Systolic Array (×8)  │ 自研 RTL (参数化)  │ —        │ 10.0     │
│ MLA Attention Pipeline        │ 自研 RTL           │ —        │ 12.0     │
│ Decoupled RoPE Unit           │ 自研 RTL           │ —        │ 1.0      │
│ MoE Router Gating + Dispatch  │ 自研 RTL           │ —        │ 4.0      │
│ Shared Expert Unit            │ 自研 RTL (复用SA)  │ —        │ 1.0      │
│ KV Cache Manager (硬件寻址)   │ 自研 RTL           │ —        │ 6.0      │
│ Chip2Chip Router (RoCE v2)     │ 自研 RTL           │ —        │ 3.0      │
│ RMSNorm Unit                  │ 自研 RTL           │ —        │ 0.5      │
│ Inference Control FSM         │ 自研 RTL           │ —        │ 2.0      │
│ ILA / 性能计数器 / Debug       │ Intel Debug IP     │ Intel    │ 1.0      │
│ Token Embedding LUT           │ 自研 RTL           │ —        │ 1.0      │
│ lm_head + MTP                 │ 自研 RTL           │ —        │ 2.0      │
├───────────────────────────────┼──────────────────┼──────────┼──────────┤
│ 合计                           │                    │          │ 46.0     │
│ 集成 + 系统联调 (20% 余量)      │                    │          │ 8.0      │
│ 总计                           │                    │          │ 54.0 人月 │
└───────────────────────────────┴──────────────────┴──────────┴──────────┘

关键复用:
  ● Intel Hard IP (零 RTL): PCIe EP, F-Tile MAC/PCS/FEC — 硅片硬化, 只配不用写
  ● Intel IP (少量定制): HBM 控制器, DMA Engine — Avalon-MM 标准接口, 适配工作量小
  ● 外部采购: RoCE v2 协议栈 — 成熟 FPGA IP 市场, ~¥1M, 免 36-54 人月自研
  ● 参数化复制: 8 个 Systolic Array 共享同一套 RTL, 只改顶层连线参数
  ● 共享 Expert 复用: 直接实例化 Systolic Array (只需 1 条 lane)
```

**与人力预算的对接:**

计划书人力预算为 5 FPGA RTL × 10 月 = 50 人月。自研模块核算为 55 人月 (含集成余量)。偏差在 10% 以内, 且有以下弹性:

- RoCE v2 外包释放了大量风险 (此项自研本会吃掉 36+ 人月)
- Systolic Array 和 MLA Pipeline 是仅有的两个 >10 人月模块, 其余均在 0.5-6 人月
- 如果团队 FPGA 经验充足 (≥5 年), 可压缩至 50 人月; 否则调整为 6 人 × 12 月 = 72 人月 (`+¥2.1M`)

### 5.1 顶层模块层次

```
ds_v4_fpga_top
│
├── pcie_cxl_ep_wrapper        # R-Tile PCIe 5.0 x16 Endpoint + CXL
│   ├── r_tile_hard_ip          # Intel PCIe 5.0 Hard IP (零 LUT)
│   ├── tlp_to_rdma_cmd           # TLP → 推理载荷 (RDMA payload)
│   └── rdma_cmd_to_tlp           # 推理完成 → TLP
│
├── f_tile_eth_wrapper          # F-Tile 200GbE 硬核 MAC + 定制RoCE
│   ├── f_tile_hard_mac         # Intel 200G/400G Ethernet Hard IP
│   ├── roce_v2_subsystem       # RoCE v2 RDMA (外购 IP ¥1M)
│   ├── rdma_payload_codec       # 推理载荷编解码 (on RDMA) 
│   └── credit_flow_ctrl        # 信用点反压流控
│
├── inference_ctrl_fsm          # 全局推理流水线状态机
│   ├── layer_counter           # 层计数 0~60
│   ├── pipeline_handshake      # 跨流水级握手
│   └── prefill_decode_mode     # Prefill/Decode 模式切换
│
├── mla_attention_pipeline      # MLA Attention 完整流水线
│   ├── q_compress_unit         # Q: 7168 → 1536 (LoRA)
│   ├── kv_compress_unit        # KV: 7168 → 576 (512+64)
│   ├── qk_dot_product_unit     # Q·K^T (128头, nope×c_KV + rope×k_R)
│   ├── online_softmax_unit     # Online Safe Softmax (FP32 acc)
│   ├── av_dot_product_unit    # A·V (nope against c_KV latent)
│   └── o_decompress_unit       # O: 128×512 → 1024 → 7168 (LoRA)
│
├── rope_hardware_unit          # Decoupled RoPE (仅 64-dim rope part)
│
├── moe_expert_core             # fp4×fp8 混合精度 MoE 推理核心
│   ├── systolic_array_128x128  # 8 个并行脉动阵列
│   ├── fp4_multiplier_unit     # fp4 × fp8 乘法器 (200 LUTs)
│   ├── swiglu_hard_unit        # SiLU + element-wise multiply
│   ├── router_gating_unit      # Hash routing + top-6 selection
│   └── expert_dispatch_unit    # Expert dispatch 到目标 FPGA
│
├── shared_expert_unit          # 共享专家 FFN
│
├── kv_cache_manager            # KV Cache 硬件管理
│   ├── kv_addr_generator       # {session, layer, seq} → HBM addr
│   ├── sliding_window_ctrl     # sliding_window=128 窗口管理
│   └── kv_fp8_compress_unit    # FP8 量化/反量化
│
├── hbm_memory_controller       # HBM2e 控制器 (2048-bit 接口)
│
├── chip2chip_router            # 片间通信引擎 (RoCE v2)
│   ├── all2all_scheduler       # MoE All-to-All 调度
│   └── roce_qp_ctrl            # RoCE QP 管理 / 流控
│
├── rms_norm_unit               # RMSNorm (eps=1e-6)
│
├── token_embed_lut             # Token Embedding LUT (仅 Node 0)
├── lm_head_unit                # lm_head 输出投影 (仅 Node 3)
├── mtp_layer_unit              # MTP 预测层 (Node 3, Layer 60 之后)
│
└── debug_monitor               # 片上 ILA + 性能计数器
```

### 5.2 关键子模块详细设计

#### 5.2.1 `fp4_multiplier_unit`

```
fp4 (E2M1) × fp8 (E4M3) 乘法器:

  input:  w_fp4[3:0] = {sign, exp[0], mant[1:0]}
          a_fp8[7:0]  = {sign, exp[3:0], mant[2:0]}

  实现: 查表法 (LUT-based)
    fp4 只有 16 种可能值 (15 有效)
    对每个 fp4 值, 预计算 8 个 FP8 尾数偏移
    → 1 个 BRAM+ 8:1 MUX + 指数加器 → 2 cycle 完成

  resource: ~200 LUTs + 1 BRAM (36Kb) per multiplier
  每个 systolic_array_128x128 需要 16,384 个乘法器
  → 16,384 × 200 LUTs ≈ 3.3M LUTs

  但可以复用: 不是 16K 个乘法器同时工作
  实际 8 个 128×128 脉动阵列 = 131,072 个乘法器实例
  → 这远超 Agilex 7 M 的 LUT 数量

  修正: 用 DSP 实现
  每个 DSP (with AI Tensor Block) 做 2× fp4×fp8 MAC
  → 9,375 DSPs × 2 MAC/DSP = 18,750 MAC/cycle
  → 同步在 450 MHz → 8.44 TMACs/s
```

#### 5.2.2 `mla_attention_pipeline` 流水线设计

```
Attention 阶段流水线 (每层每 token):

  Stage 0: Q 压缩     (7168×1536)  HBM:6μs + DSP:1.4μs  → ~6μs
  Stage 1: KV 压缩    (7168×576)   HBM:2.3μs + DSP:0.6μs → ~2.3μs
  Stage 2: Q·K^T      128头并行      DSP:3.8μs            → ~3.8μs
  Stage 3: Softmax    硬化           0.2μs               → ~0.2μs
  Stage 4: A·V        128头并行      DSP:3.7μs            → ~3.7μs
  Stage 5: O 解压     LoRA ×2       HBM:8.7μs + DSP:9.3μs → ~9.3μs
  ──────────────────────────────────────────────────────────
  关键路径:              Stage 0 (6μs) 和 Stage 5 (9.3μs) 受 HBM 限制
  总延迟:              ~25μs (串行) 或 ~15μs (部分重叠)

MoE 阶段流水线:
  Stage 6: Gating      Hash + top-6   ~0.5μs
  Stage 7: Dispatch    All-to-All 发送 ~3μs (跨节点 RDMA)
  Stage 8: Expert FFN  66M MAC/expert ~40μs (含 HBM 权重加载)
  Stage 9: Combine     结果加权合并    ~2μs
  ──────────────────────────────────────────────
  关键路径:              Stage 8 (40μs) 包含 HBM 权重加载
  每 MoE 层总延迟:     ~65μs
```

### 5.3 权重转换与部署工具链 (Weight Layout Compiler)

评审质疑: PyTorch checkpoint 怎么变成 30 片 FPGA 上的比特流 + 权重文件？RTL 变更时权重如何重排？逐层混合精度如何判定？模型升级后是否需要重综合？本节回应从模型到部署的完整工具链。

**5.3.1 工具链总览**

```
HuggingFace safetensors (fp8/bf16)
        │
        ▼
┌─────────────────────────────┐
│  fpgalpu-convert             │  ← Python, ~2000 行
│  ┌─ Step 1: 解析 + 模型图匹配  │
│  ├─ Step 2: fp4 量化          │
│  ├─ Step 3: 分配到 30 卡      │
│  └─ Step 4: 生成 HBM 比特流   │
└─────────────┬───────────────┘
              │ 30 个 weight binary files
              ▼
┌─────────────────────────────┐
│  Weight Layout Compiler      │  ← Python, ~3000 行
│  ┌─ Weight tiling (systolic) │
│  ├─ HBM bank interleaving    │
│  ├─ Address map generation   │
│  └─ Mixed-precision config   │
└─────────────┬───────────────┘
              │ weight binary + address map header
              ▼
┌─────────────────────────────┐
│  PCIe DMA Loader             │  ← C, ~500 行
│  初始化时加载到每片 FPGA HBM  │
└─────────────────────────────┘
```

**5.3.2 Step 1-2: 模型解析与 fp4 量化**

```
输入: HuggingFace safetensors (标准格式, DeepSeek 官方发布)

解析:
  提取 named_parameters → {layer_id, param_type, shape}
  与 config.json 交叉验证: n_layers=61, n_experts=384,
  n_heads=128, dim=7168, moe_intermediate=3072

fp4 量化 (E2M1 + per-128-group scale):
  for each weight_matrix in [Attn_Q, Attn_KV, Attn_O, Expert_gate,
                              Expert_up, Expert_down, Shared_expert]:
    weight_2d = reshape(weight, (-1, 128))
    for each group of 128:
      scale = max(abs(group)) / 6.0           # E2M1 max = 6.0
      weight_fp4 = round(clamp(group/scale, -6, 6))
      scale_fp8 = float_to_e4m3(scale)

  量化完成后: weight_fp4 (4-bit) + scale_fp8 (8-bit per 128)
  有效位宽: 4 + 8/128 = 4.0625 bits/weight (vs 理论 4-bit)

  Router 权重: 跳过量化, 保持 FP8 (见 §4.7.3)
```

**5.3.3 Step 3: 权重分配——Weight Layout Compiler (WLC)**

```
WLC 输入:
  model_config:     layers, experts, heads, dim
  hardware_config:  systolic_K, systolic_N, HBM_bank_count,
                    HBM_bank_width, tp_size
  partition_config: 每 FPGA 的 layer_range, expert_range, head_range

WLC 核心逻辑:

  ① Systolic Tiling
     将权重矩阵切成 systolic array 能消费的 tile:
       例如 systolic 128×128, Expert gate (7168×3072):
       → K 方向 7168/128 = 56 个 tile
       → N 方向 3072/128 = 24 个 tile
       → 56×24 = 1344 个 tile, 每个 128×128×4b = 8 KB

  ② HBM Bank Interleaving
     同一行的 tile 分配到不同 HBM bank (避免 bank conflict):
       24 个 N-tile 轮转分配到 HBM 的 32 个 pseudo-channel
       → 同一 cycle 最多 32 个 tile 可并行读取

  ③ 地址映射表生成
     每个 FPGA 输出:
       ┌────────────────────────────────────┐
       │ Layer 00: Expert 003 gate @ 0x0000 │
       │ Layer 00: Expert 003 up   @ 0x0800 │
       │ Layer 00: Expert 003 down @ 0x1000 │
       │ ...                                │
       │ Layer 00: Shared Expert   @ 0x4000 │
       │ Layer 01: ...                      │
       └────────────────────────────────────┘
     此表同时写入 weight binary header 和 FPGA RTL 地址查找表
```

**5.3.4 逐层混合精度的自动化判定**

```
流程 (Phase 1 完成):

  ① 1 卡 FPGA 运行全层推理 (每个并行组 1 卡)
     → 输出 per-layer activation (FP16)

  ② PyTorch fp8 reference 跑同一层
     → 输出 reference activation (FP32)

  ③ 自动对比:
     np.dot(act_fpga, act_ref) /
     (norm(act_fpga) * norm(act_ref))  → cosine_similarity

     mean(|act_fpga - act_ref|²) /
     mean(|act_ref|²)                   → L2_relative_error

  ④ 判定规则:
     cosine_sim < 0.995  OR  L2_err > 1%  → 标记 "敏感层"
     → 写入 mixed_precision_config.yaml

  ⑤ WLC 重跑:
     敏感层 → fp8 weights (1 byte/weight)
     其余   → fp4 weights (0.5 byte/weight + scale)
     HBM 地址布局不变 (仅 weight data 变大)

  全模型 61 层 profiling: 单卡 ~2 分钟, 全自动。
```

**5.3.5 模型升级适应性——什么需要重综合, 什么不需要**

```
┌───────────────────────┬──────────────┬──────────┬────────────────┐
│ 模型变化               │ Weight 更新  │ RTL 影响  │ 部署周期        │
├───────────────────────┼──────────────┼──────────┼────────────────┤
│ Expert 384→512        │ WLC 重跑     │ 无        │ 1 小时          │
│ Layers 61→80          │ WLC 重跑     │ 无        │ 1 小时          │
│ dim 7168→8192         │ WLC 重跑     │ 无        │ 1 小时          │
│ hidden 3072→4096      │ WLC 重跑     │ 无        │ 1 小时          │
│ Heads 128→96 (TP 不变)│ WLC 重跑     │ 无        │ 1 小时          │
│ fp4→fp6 精度           │ WLC 重跑     │ 改 DSP MAC│ 1-2 周          │
│                       │              │ mode      │                │
│ MLA→标准 MHA          │ WLC 重跑     │ RTL 重写  │ 3-6 人月        │
│ MoE→Dense             │ 不适用       │ 架构重做  │ 不可行          │
└───────────────────────┴──────────────┴──────────┴────────────────┘

关键设计理念:
  RTL 中的专家数、层数、dim、heads 都是 Verilog parameter —
  改头文件常量即可, 不触发重新综合。
  仅 attention 算法 (MLA) 或 MoE 路由的结构性变化
  才需要改 RTL 并重跑 Quartus。

Quartus 综合基准:
  完整 Agilex 7 M 设计 (~30万 LUT 级): 4-8 小时 (64核 Linux)
  仅改 parameter 不重综合: 无需 Quartus
  增量编译 (小改动): 30-60 分钟
```

**5.3.6 与 TensorRT-LLM 的对标——诚实差距**

```
┌──────────────────────┬──────────────────┬──────────────────┐
│                       │ NVIDIA TensorRT   │ 本方案 FPGA 工具链 │
├──────────────────────┼──────────────────┼──────────────────┤
│ 模型导入              │ 1 命令 from HF    │ 1 命令 (Python)   │
│ 量化 (PTQ/QAT)       │ 内置, 自动        │ 内置, 自动        │
│ 逐层精度 profiling    │ 自动              │ 自动 (Phase 1)    │
│ 图优化 / 算子融合     │ 成熟 (100+ pass)  │ 不适用            │
│                       │                   │ (已有硬件融合)    │
│ 权重切分 (TP/PP/EP)  │ 自动              │ WLC 自动          │
│ 部署                  │ build → run       │ convert+load → run│
├──────────────────────┼──────────────────┼──────────────────┤
│ 模型升级 (参数变)     │ 重跑 build        │ 重跑 WLC (1h)     │
│ 模型升级 (架构变)     │ 等框架更新        │ RTL 重写+重综合    │
├──────────────────────┼──────────────────┼──────────────────┤
│ 生态成熟度            │ ★★★★★            │ ★★★ (Phase 2)    │
│ 维护人力              │ NVIDIA 百人团队   │ 1-2 人 + WLC     │
│ 社区贡献              │ 全球开发者        │ 自研, 闭源        │
└──────────────────────┴──────────────────┴──────────────────┘

坦率差距:
  TensorRT-LLM 的成熟度追不上, 也不需要追。
  FPGA 工具链做的事更简单:
    → 不需要处理 100+ CUDA kernel 变体
    → 不需要图级 IR 优化 (硬件数据路径已固定)
    → 只需要为一个固定的脉动阵列生成正确的 weight layout

  核心差距在对新模型架构的快速支持 — NVIDIA 有生态,
  我们有 Quartus 综合周期。但对于企业私有部署:
    一次配好, 稳定运行, 不需要追版本升级。
```

---

## 6. 组网拓扑与通信方案

### 6.1 设计原则

**单机内，不再需要外部网络。** 8 张卡全插在一台 4U 服务器里，卡内 4 片经 C2C SerDes 互联，卡间经 PCIe 5.0 背板直达。不设 ToR 交换机，不买 RoCE IP，不用 QSFP-DD 笼子和 DAC 线缆。

```
为什么用 PCIe 背板 + C2C 替代 Ethernet + ToR:

  物理:
    8 张卡都插在同一个服务器背板上 → PCIe 背板是免费的交换机
    卡内 4 片在同一张 PCB 上 → F-Tile SerDes 是免费的片间总线

  经济:
    省: 2× ToR Switch (¥200K) + RoCE v2 IP (¥1M) + QSFP-DD cages (¥32K) + DAC cables (¥20K)
    总额节省: ~¥1.25M

  性能:
    PCIe 5.0 x16 P2P 延迟 ~500 ns  vs  Ethernet RoCE ~2-5 μs  →  快 4-10×
    C2C SerDes 延迟 ~50 ns/跳       vs  跨卡 routing hop        →  可忽略

  简化:
    单协议栈 (PCIe TLP + C2C frame), 无 Ethernet/IP/UDP/RoCE 五层
    无 congestion control (物理链路专线, 无争用)
    无 ToR 故障域, 无 MLAG 配置
```

### 6.2 卡内 C2C 拓扑: Dual Ring

AGM 039 R47A 封装含 F-Tile ×3。片间 C2C 走 F-Tile raw transceiver (NRZ 32 Gbps, 省去 PAM4 DSP 开销)，每 link 4 lane bonded。

```
单卡 4 片 Dual Ring (冗余):

         ┌──────────────────────────────────┐
         │            Ring A                 │
         │   Chip0 ←──────────→ Chip1       │
         │     ↕                    ↕        │
         │   Chip2 ←──────────→ Chip3       │
         └──────────────────────────────────┘

         ┌──────────────────────────────────┐
         │            Ring B (冗余)          │
         │   Chip0 ←──────────→ Chip2       │
         │     ↕                    ↕        │
         │   Chip1 ←──────────→ Chip3       │
         └──────────────────────────────────┘

每 link: 4 lane × 32 Gbps NRZ = 128 Gbps 单向 (双向 256 Gbps)
每片用量: 2 link × 4 lane = 8 lane (占 F-Tile 48 lane 总量的 17%)
单跳: ~50 ns (SerDes latency + < 200mm PCB trace)
最长: 2 跳 (Chip1 → Chip0 → Chip3) ≈ 100 ns

为什么 Ring 而非 Mesh:
  - Ring: 2 link/片, Mesh: 3 link/片 → 省 4 lane/片
  - 2 跳 100 ns vs MoE 单层 ~3 μs → 差距 30×, 两跳完全可忽略
  - Ring B 冗余: Ring A 链路断 → 自动走 Ring B, 零丢帧
  - 省下的 F-Tile lane 留做 debug ILA / 未来扩展
```

**每片 F-Tile 用量：**

| 用途 | Lane 数 | 备注 |
|------|---------|------|
| Ring A link (C2C) | 4 lane TX + 4 lane RX | 32 Gbps NRZ |
| Ring B link (C2C) | 4 lane TX + 4 lane RX | 冗余链路 |
| 预留 (debug ILA) | 8 lane | Signal Tap 远程抓取 |
| 未使用 | 28 lane | 未来扩展 |

### 6.3 C2C 协议分层

```
┌────────────────────────────────────────────┐
│ Transport Layer                             │
│  · 消息类型路由 (MoE / Pipeline / PCIe_Proxy)│
│  · 5-bit 全局芯片寻址 {CardID, ChipID}       │
│  · 多虚拟通道复用 (Data/Credit/Mgmt)        │
├────────────────────────────────────────────┤
│ Link Layer                                   │
│  · 帧定界 & 扰码 (64b/66b)                  │
│  · Credit-based 流控 (per VC)               │
│  · CRC32 检错 + 序列号 + 超时重传            │
│  · Lane 对齐 & Deskew (multi-lane)          │
├────────────────────────────────────────────┤
│ Physical Layer                               │
│  · F-Tile Transceiver (32 Gbps NRZ)         │
│  · 4 lane bonded × 双向                    │
│  · AC 耦合, 片内终端                        │
└────────────────────────────────────────────┘
```

**6.3.1 帧格式**

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
┌───────────────────────────────┬───┬───┬───┬───┬───┬───────────────┐
│  SOP (8B: 0xFB_C2C_FRAME)     │Ver│Typ│Pri│ VC│HdrLen│           │
│                               │ 2b│ 4b│ 2b│ 2b│ 4b    │           │
├───────────────────────────────┴───┴───┴───┴───┼───┴───────────────┤
│  SrcChip[4:0]  │ DstChip[4:0]  │  SeqNum[7:0]   │  FrameLen[11:0]  │
├───────────────────────────────────────────────────────────────────┤
│  Header CRC16 (bytes 8-15)                                         │
├───────────────────────────────────────────────────────────────────┤
│  Payload (0-4088 Bytes, 8B aligned)                                │
├───────────────────────────────────────────────────────────────────┤
│  CRC32 (over SOP through payload end)                              │
├───────────────────────────────────────────────────────────────────┤
│  EOP (4B: 0xE0F_END)                                               │
└───────────────────────────────────────────────────────────────────┘

Type (4b):
  0x1 = MoE_Dispatch     [激活向量 7168B FP8]
  0x2 = MoE_Reduce       [专家输出 7168B FP8]
  0x3 = Pipeline_Fwd     [hidden_state 7168B FP8, 跨层转发]
  0x4 = PCIe_Proxy       [Host ↔ 非 Chip0 的 MMIO/DMA 转发]
  0x5 = Credit_Update    [流控信用归还, 0 payload]
  0x6 = Weight_Broadcast [权重加载]
  0x7 = Heartbeat        [链路存活探测, 0 payload]

Priority (2b): 0=Credit/Mgmt, 1=Pipeline, 2=MoE, 3=PCIe_Proxy
VC (2b):       0=Control, 1=Data_HP, 2=Data_Bulk, 3=Management

单帧 max payload: 4088 B
MoE Dispatch (7168B): 拆成 2 帧 (4088 + 3080)
帧 overhead: 16B header + 4B CRC + 4B EOP = 24B
效率: 7168 / (7168 + 2×24) = 99.3%
```

**6.3.2 Credit-based 流控**

```
初始化:
  RX 上电 → 发 CREDIT_INIT 帧:
    {VC0: 64, VC1: 256, VC2: 128, VC3: 32}
  每 credit = 1 帧 (最大 4096 B)

发送:
  TX 每发一帧, credit[VC]--
  credit[VC] == 0 → 停发, 等 CREDIT_UPDATE

归还:
  RX 消费完帧 → 发 CREDIT_UPDATE {VC, returned_credits: N}
  TX credit[VC] += N

RX Buffer (每 SerDes 端口, 每片 2 端口):
  VC0 Control:     8 KB   (32 帧 × 256B)
  VC1 Data_HP:   128 KB   (MoE, 低延迟)
  VC2 Data_Bulk:  64 KB   (Pipeline, bulk)
  VC3 Mgmt:       16 KB   (Heartbeat, weight)
  合计: 216 KB/port, 432 KB/片
  占 M20K 总量: 432 KB / 46 MB ≈ 0.9%
```

**6.3.3 路由表**

```
全局芯片地址: {CardID[2:0], ChipID[1:0]} = 5-bit, 0-31

每片维护 8-entry 路由表:
  ┌──────────────┬──────────────────┬──────────┐
  │ CardID       │ NextHop           │ Egress    │
  ├──────────────┼──────────────────┼──────────┤
  │ Self         │ ChipID lookup     │ SerDes_A │
  │              │ (本卡 4 片)        │ SerDes_B │
  │ Card_0       │ PCIe_P2P_0       │ PCIe     │
  │ ...          │                  │          │
  │ Card_7       │ PCIe_P2P_7       │ PCIe     │
  └──────────────┴──────────────────┴──────────┘

最短路径: Dijkstra over Ring A topology (静态 topology, 编译时固化)
Ring A 跳数: Chip0↔1=1, Chip0↔3=2 (经 Chip1 或 Chip2), Chip1↔2=2 (经 Chip0)
```

**6.3.4 错误处理**

```
检错:
  CRC32 per frame (payload + header)
  Header CRC16 (独立校验, 路由前就知道头部是否损坏)
  64b/66b 编码提供 DC balance + 非法码字检测

恢复:
  单帧 CRC 错 → NAK + 重传 (TX 保留已发帧直到 ACK)
  连续 3 帧 CRC 错 → 链路降级, 切到 Ring B
  Ring A + Ring B 同时断 → 中断 Host, 触发卡级热迁移
  超时: 100 μs 无 credit return → 发 Heartbeat
        5 次 Heartbeat 无响应 → 链路 down

链路训练 (上电):
  TX → RX: TS1/TS2 (仿 PCIe training sequence)
  Bit lock → Word alignment → Lane deskew → Ready
  < 1 ms
```

**6.3.5 延迟预算**

```
同卡 MoE Dispatch (Chip A → Chip B, 7168B, 2 帧):

  TX framer:                   ~20 ns
  SerDes TX (4 lane × 32G):    ~56 ns  (4088B + 24B) / 128 Gbps
  PCB trace (100mm):            ~0.2 ns
  SerDes RX + deframer:        ~50 ns  (deskew + CRC check)
  RX → MoE queue:              ~10 ns
  ────────────────────────────────────
  第一帧到达:                  ~136 ns
  第二帧 (3080B):              ~96 ns
  总 Dispatch (7168B):        ~232 ns → 取 250 ns

跨卡 MoE Dispatch (Chip A Card0 → Chip B Card3, via PCIe P2P):

  C2C → Chip0 proxy:          ~136 ns
  PCIe MWr (7168B, x16 64GB/s): ~112 ns
  Chip0 RX → C2C → 目标 chip:  ~136 ns
  ────────────────────────────────────
  跨卡总延迟:                  ~384 ns → 取 400 ns

对照:
  单层 MoE FFN (12,300 DSPs, fp4): ~3 μs
  通信 / 计算比: 400 ns / 3,000 ns = 13%
  → All-to-all 通信完全不是瓶颈
```

### 6.4 卡间 PCIe 5.0 P2P

每卡仅 Chip0 连 PCIe (R-Tile x16)。Chip1/2/3 经 C2C → Chip0 → PCIe 与外部交互。

```
Chip0 BAR4 布局 (64 MB, 每卡统一):

  ┌────────────────┬─────────┬──────────────────────────┐
  │ Offset          │ Size    │ 目标                      │
  ├────────────────┼─────────┼──────────────────────────┤
  │ 0x0000_0000    │ 16 MB   │ Chip0 (本地寄存器 + DMA)   │
  │ 0x0100_0000    │ 16 MB   │ Chip1 (经 C2C Proxy 转发) │
  │ 0x0200_0000    │ 16 MB   │ Chip2 (经 C2C Proxy 转发) │
  │ 0x0300_0000    │ 16 MB   │ Chip3 (经 C2C Proxy 转发) │
  └────────────────┴─────────┴──────────────────────────┘

跨卡数据流 (Chip0 CardA → Chip2 CardB):
  1. CardA Chip0 DMA Engine 发 PCIe MWr
     目标地址 = CardB BAR4 base + 0x0200_0000 (Chip2 offset)
  2. PCIe fabric 路由 (CPU Root Complex 或 P2P direct)
  3. CardB Chip0 R-Tile 收到 MWr → 写 Chip2 C2C TX 队列
  4. CardB Chip0 C2C Proxy → Ring A → Chip2
  5. CardB Chip2 收到帧 → MoE RX queue

P2P 带宽 (单向):
  PCIe 5.0 x16: ~64 GB/s (128b/130b)
  跨 CPU socket (UPI 2.0): ~20 GB/s
  瓶颈在 UPI: 20 GB/s 对单卡 4 片 × ~7.2 Gbps = 28.8 Gbps < 20 GB/s → 够用
  全部 8 卡跨 socket: 8 × 28.8 Gbps = 230 Gbps ≈ 28.8 GB/s
  最坏情况下需经过 UPI: 28.8 GB/s > 20 GB/s → 有瓶颈

优化: 同 socket 卡优先分配 MoE 专家
  CPU0 (Card 0-3): experts #0-191
  CPU1 (Card 4-7): experts #192-383
  → 跨 socket MoE 流量减半 → ~14.4 GB/s < 20 GB/s ✓
```

### 6.5 通信带宽核算

```
单 Token 单 MoE 层 all-to-all (6 routed experts):
  Dispatch: 6 × 7168B FP8 = 42 KB
  Reduce:   6 × 7168B FP8 = 42 KB
  合计:                    ~84 KB / token / MoE 层

专家分布 (同 socket 优化后, 同 sock 48 专家 / 192):
  P(同卡)     = 12/192 = 6.25%    → 走 C2C SerDes
  P(同 sock)  = (48-12)/192 = 18.75% → 走 PCIe P2P (同 socket)
  P(跨 sock)  = 144/192 = 75%    → 走 PCIe P2P (经 UPI)

  期望同卡专家:    6 × 0.0625 = 0.38
  期望同 sock 专家: 6 × 0.1875 = 1.13
  期望跨 sock 专家: 6 × 0.75   = 4.50

200 tps (tokens/sec) 吞吐下:
  卡内 C2C:    200 × 61 × 0.38 × 2 × 7KB = 65 MB/s   ← trivial
  同 sock PCIe: 200 × 61 × 1.13 × 2 × 7KB = 193 MB/s ← < 1% of x16
  跨 sock UPI:  200 × 61 × 4.50 × 2 × 7KB = 769 MB/s ← 3.8% of UPI 20GB/s

Layer forwarding (pipeline):
  31 chip-to-chip transitions/token × 7KB × 200 tps = 43 MB/s ← trivial

结论: 所有通信路径利用率 < 5%, 带宽充裕。
      延迟也非瓶颈 (C2C 250 ns, PCIe 400 ns vs 计算 3 μs)。
```

### 6.6 容错设计

**6.6.1 故障域**

```
四个潜在故障点:

  ① 单芯片 (片内 SerDes/R-Tile/DSP/HBM 故障)
  ② 单 C2C 链路 (F-Tile lane 故障)
  ③ FPGA 加速卡 (PCB/VRM 故障 → 4 片全下)
  ④ Host CPU / PCIe fabric 故障

不存在:
  ✗ ToR Switch 故障 → 无此硬件
  ✗ QSFP-DD 光模块 / DAC 线缆 → 无此硬件
  ✗ RoCE congestion / PFC 死锁 → 无此协议
  ✗ 跨机网络分区 → 单机内
```

**6.6.2 芯片级容错**

```
单芯片故障 (发生率最高):

  机制: 卡内 4 片权重相互备份
    每片 HBM 存:
      自身: 12 专家 + ~2 层 attention (~570 MB)
      邻居: 同卡另一片的权重 (仅 expert, ~400 MB)
      总计: < 1 GB, HBM 32 GB 远未满

  检测: Heartbeat 超时 100 μs → 故障确认
  
  恢复:
    T+0:     C2C 路由表更新 → 故障片的 12 专家由同卡 3 片分摊
             每片 +4 专家 → 12→16 专家/片
    T+50ms:  邻片激活备份权重 (HBM 中已有)
    T+100ms: 恢复全吞吐
             单片故障仅降速 0% (同卡 3 片接管)

  降级: 无。4 芯片全同时故障才需要卡级热备。
```

**6.6.3 链路级容错**

```
单 C2C 链路故障:
  检测: 3 帧 CRC 错/超时 → <1 μs
  恢复: 禁用 Ring A, 全流量走 Ring B → 零丢帧
  吞吐: Ring B 带宽 256 Gbps >> all-to-all ~65 MB/s → 不变

Ring A + Ring B 同时断 (极罕见):
  → 该卡退化为 4 片独立 (由 PCIe 经 Chip0 协商每片直连 Host)
  → 吞吐不变, 延迟略增 (经 Chip0 转发)
```

**6.6.4 卡级容错**

```
单卡故障 (PCB/VRM → 4 片全下):
  8 卡 → 7 卡, 全集群降级
  
  权重冗余: 同 sock 其他卡 Chip0 存该卡关键层的 attention 权重
  恢复: 同 sock 7 卡分摊故障卡的 12 层 + 48 专家
        降速 ~12.5% (8→7 卡)
        需人工更换卡后恢复全吞吐

双卡同时故障:
  降速 ~25%, 不可自动恢复
  概率: MTBF 50,000h, 8 卡系统 MTBF ≈ 6,250h
        窗口 4h 内双故障概率: (4/50,000)² × 28 ≈ 4.5 × 10⁻⁷ → 可忽略
```

**6.6.5 可用性汇总**

```
┌──────────────────┬────────────┬──────────────┬──────────────┐
│ 故障类型           │ 检测延迟    │ 恢复时间      │ 恢复后吞吐    │
├──────────────────┼────────────┼──────────────┼──────────────┤
│ 单芯片            │ <100 μs    │ <100 ms      │ 100%         │
│ 单 C2C 链路       │ <1 μs      │ <1 μs        │ 100%         │
│ 单 FPGA 卡        │ <100 μs    │ 人工 ~4h     │ 87.5%        │
│ Host CPU / PCIe   │ <100 ms    │ 人工 ~2h     │ 停摆(数据面)  │
│ 双卡同时故障       │ <100 μs    │ 人工 ~4h     │ 75%          │
│ 双 C2C Ring 同断  │ <1 μs      │ <1 ms        │ 100%         │
└──────────────────┴────────────┴──────────────┴──────────────┘

vs 旧方案 (多机 Ethernet + 双 ToR):
  故障模式数: 4 种 → 4 种 (持平)
  缺少:F-Tile 端口故障 + ToR 故障 (因不存在)
  新增: C2C 链路故障 + 单芯片故障 (粒度更细)
  
  关键改进: 芯片级故障自愈 (旧方案需换卡), 可用性更高
```

### 6.7 与旧方案对比

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│                       │ 旧 (Ethernet + ToR)    │ 新 (PCIe P2P + C2C)  │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ 通信平面              │ 单平面 (Ethernet)     │ 单平面 (PCIe + C2C)  │
│ 卡内片间              │ — (单芯片/卡)          │ C2C SerDes Dual Ring │
│ 卡间                  │ 200GbE → ToR Switch  │ PCIe 5.0 x16 P2P     │
│ 跨机                  │ ToR → ToR            │ — (单机内)            │
│ 延迟 (卡内同卡)        │ —                    │ ~250 ns              │
│ 延迟 (跨卡)           │ ~1.5 μs              │ ~400 ns              │
│ 协议栈               │ RoCE v2 / UDP / IP   │ 裸 PCIe TLP + C2C帧  │
│ 硬件                 │ FPGA + 2 ToR Switch  │ FPGA + 0 Switch      │
│ 外部 IP 外购          │ RoCE v2 IP (¥1M)     │ ¥0 (自研 DMA)        │
│ 线缆                  │ QSFP-DD DAC ×16      │ 无                   │
│ 故障模式数            │ 4 种                  │ 4 种                  │
│ 热备                  │ 2 张卡热备            │ 片级自愈 (卡内备份)   │
│ 软件栈               │ RDMA verbs (复杂)     │ PCIe P2P DMA (简单)  │
│ 每集群 BOM 增量       │ +¥1.2M               │ +¥0                  │
│ 额外功耗/卡           │ ~60W (F-Tile + cage)  │ ~10W (C2C SerDes)    │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ Δ BOM (cluster)      │ 基线                  │ -¥1.25M             │
│ Δ 延迟               │ 基线                  │ -1 μs/cross-card    │
│ Δ 复杂度             │ 基线                  │ -1 protocol layer   │
│ Δ 硬件种类            │ 基线                  │ -Switch, -Cable     │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

---

