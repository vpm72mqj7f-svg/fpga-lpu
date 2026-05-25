# DeepSeek V4 Pro — FPGA 算力推理集群

## 可行性论证 & 工程设计方案

> 2025.05 | 内部评审

---

# 目录

1. 背景与定位
2. 架构总览
3. 关键技术参数
4. 算力分配设计
5. RTL 核心模块
6. 组网通信方案
7. 平台与物理形态
8. 软件生态
9. 开发路线与里程碑
10. 成本与财务
11. 竞争分析
12. 风险评估

---

## CH1 | 背景与定位

---

## 问题：中国大模型的部署瓶颈

```
      模型能力                     硬件底座
    ════════════              ═══════════════
                                    
  DeepSeek V4 Pro           NVIDIA H100/B200
  1.6T MoE 顶尖              → 出口管制, 买不到
      │                         
      │                     华为 Ascend 910C  
  全球市场需求               → SMIC 产能受限
  东南亚/中东/拉美           → 先进封装制裁
      │                     → 无法全球出口
      │                         
      └──────────────→     国产 GPU (寒/海/壁)
                             → 软件栈不成熟
                             → 供应同样不稳定

  能做出最好的模型           但不知道怎么部署出去
```

---

## 答案：FPGA 是全球可部署的唯一解

```
  Intel Agilex 7 M + HBM
  ════════════════════════

  供应链分布:
    FPGA 芯片  → Intel 美国/爱尔兰/以色列  Fab
    HBM 堆叠   → SK Hynix / Samsung      韩国
    先进封装   → Intel 马来西亚/越南      东南亚
    PCB 制造   → 中国大陆                 本土

  ✓ 不受 GPU 算力出口管制  (TPP 远低于 4800 阈值)
  ✓ 不依赖 SMIC 产能
  ✓ 标准 PCIe 设备, 全球兼容
  ✓ 部署地点不受限

  → 唯一可将中国大模型推理部署到全球的硬件路径
```

---

## 战略定位

```
               ┌──────────────────────┐
  NVIDIA GPU   │  中国: ✗ 管制        │
               │  全球: ✓ 可部署      │
               │  但中国买不到         │
               └──────────────────────┘

               ┌──────────────────────┐
  华为 Ascend  │  中国: ✓ 可售        │
               │  全球: ✗ 难以出口    │
               │  且产能受限           │
               └──────────────────────┘

               ┌──────────────────────┐
  本方案 FPGA  │  中国: ✓ 可获取      │
               │  全球: ✓ 可部署      │
               │  唯一双向覆盖方案     │
               └──────────────────────┘
```

---

## 不是 "FPGA 替代 GPU"，是 "中国模型出海唯一底座"

```
       中国大模型 → 全球部署的商业闭环

  模型训练 (中国)
      │  充分竞争的 GPU 池 / 自有集群
      ▼
  模型权重 (fp4 量化)
      │
      ▼
  FPGA 算力集群 ← 全球可获取
      │
      ▼
  海外推理服务 API
      │  东南亚、中东、欧洲、拉美
      ▼
  全球终端用户
```

---

## CH2 | 架构总览

---

## 整体三层架构

```
┌─────────────────────────────────────────────┐
│ [用户接入层]  OpenAI REST API               │
│   /v1/chat/completions                      │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│ [推理服务层]  x86 主控服务器                  │
│  · Token 编码 / 采样 / 会话管理              │
│  · 推理命令编排 / 结果拼接                   │
└──────────────────────┬──────────────────────┘
                       │ 400GbE RoCE v2 RDMA
┌──────────────────────▼──────────────────────┐
│ [FPGA 算力集群]  4 节点 × 8 卡 = 32 卡       │
│                                              │
│  Node 0       Node 1      Node 2      Node 3│
│  8 FPGA       8 FPGA      8 FPGA      8 FPGA│
│  Layer 0-14   L15-29      L30-44      L45-60│
│  +Embedding                        +lm_head │
└──────────────────────────────────────────────┘
```

---

## 单节点内部拓扑

```
        标准 4U GPU 服务器 (Supermicro 821GE)
   ┌──────────────────────────────────────────┐
   │  Dual Intel Xeon (Sapphire Rapids)       │
   │  ┌───────────┐      ┌───────────┐       │
   │  │  CPU 0    │◄UPI──┤  CPU 1    │       │
   │  └──┬──┬──┬──┘      └──┬──┬──┬──┘       │
   │     x16                x16               │
   │  ┌──┴──┴──┴──┐    ┌───┴───┴───┴──┐      │
   │  │  FPGA ×4  │    │   FPGA ×4    │      │
   │  └───────────┘    └──────────────┘      │
   │         │                 │              │
   │         └────PCIe P2P─────┘              │
   │                  │                       │
   │     ┌────────────▼──────────┐            │
   │     │  F-Tile 200GbE    │            │
   │     └───────────────────────┘            │
   └──────────────────────────────────────────┘
```

---

## 单 FPGA 内部 RTL 模块

```
    PCIe 5.0 x8 CEM 金手指
           │
    ┌──────▼──────────────────────────┐
    │  PCIe 5.0 EP Hard IP + 报文控制  │
    └──────┬──────────────────────────┘
           │
    ┌──────▼──┬──────┬──────┬──────┬──────┐
    │fp4 脉动 │ MLA  │ RoPE │ RMS  │ MoE  │
    │阵列 ×8  │Attn  │ Hard │ Norm │Router│
    │(9,375   │Pipe  │ Unit │ Hard │ +    │
    │ DSPs)   │line  │      │      │Disp  │
    ├─────────┴──────┴──────┴──────┴──────┤
    │  KV Cache Mgr │ Chip2Chip Router   │
    └──────┬────────┴────────────────────┘
           │
    ┌──────▼──────────────────────────────┐
    │  HBM2e 控制器 (2,048-bit @ 920GB/s)  │
    │  ┌────────────┬───────────────────┐ │
    │  │ 权重 ≤24GB │ 运行区 ≤8GB       │ │
    │  │ 13 专家fp4 │ KV Cache + Buffer │ │
    │  │ Attn权重   │ ETH Ring Buffer   │ │
    │  └────────────┴───────────────────┘ │
    │          32 GB HBM2e                │
    └─────────────────────────────────────┘
```

---

## CH3 | 关键技术参数

---

## DeepSeek V4 Pro 架构

> 来源: 开源 config.json (已验证)

| 参数 | 值 | 关键影响 |
|------|-----|---------|
| `hidden_size` | **7,168** | 计算粒度 |
| `num_hidden_layers` | **61** | 流水线分段 |
| `n_routed_experts` | **384** | 32 卡均分 = 12/卡 |
| `n_shared_experts` | **1** | 每卡冗余存储 |
| `num_experts_per_tok` | **6** | all-to-all 通信量 |
| `moe_intermediate_size` | **3,072** | fp4 权重 33MB/专家 |
| `num_attention_heads` | **128** | 8 卡均分 = 16 头/卡 |
| `num_key_value_heads` | **1** | **MLA 架构!** |
| `expert_dtype` | **fp4** | 4bit 浮点, E2M1 |
| `max_position_embeddings` | **1,048,576** | 1M context |

---

## MLA: 为什么 KV Cache 只有 576 字节

```
  标准 MHA (LLaMA):
    Q/K/V: 128 heads × 128 dim
    KV Cache: 2 × 128 × 128 = 32 KB/token/layer (FP16)

  DeepSeek MLA:
    KV 压缩为 1 个 latent vector: 512 dim (nope) + 64 dim (rope)
    KV Cache: 576 Bytes/token/layer (FP8)
    
  56× 压缩!
  → 这是 DeepSeek 1M context 能力的核心
  → 也是 FPGA 硬件加速的关键优势
```

---

## fp4: 为什么国产 GPU 都不支持

```
  fp4 (E2M1) 格式:
    {sign, 1b exponent, 2b mantissa}
    有效值: {±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}

  推理优势:
    ● 4× HBM 空间节省 vs FP16
    ● 2× HBM 空间节省 vs FP8
    ● DeepSeek 用 QAT 保证精度

  国产 GPU 支持情况:
    华为 Ascend:  INT4 ✓   FP8 ✓   fp4 ✗
    寒武纪:        INT4 ✓   FP8 ✗   fp4 ✗
    海光 DCU:      INT8 ✓   FP8 ✗   fp4 ✗
    
  FPGA: 自研 fp4×fp8 乘法器, 原生支持, 零解压开销
```

---

## 单 Token 计算量分解

```
              每 token 每层            每 token 全 61 层
  ─────────────────────────────────────────────────────
  MLA Attention      149M  MAC            ~9.1B
  MoE Routing          1M  MAC            ~0.06B
  MoE Expert (×6)    396M  MAC           ~24.2B
  Shared Expert       66M  MAC            ~4.0B
  ─────────────────────────────────────────────────────
  合计               611M  MAC           ~37.4B  MAC

  单卡 8.44 TMACs/s 计算耗时:
    37.4B / 8.44T = 4.4 ms  (纯计算, 无 HBM/通信开销)
    
  HBM 权重加载耗时:
    ~6.1 GB × 920GB/s = 6.6 ms
    
  → 计算与 HBM 近乎平衡 (1.35×)
```

---

## CH4 | 算力分配设计

---

## 32 卡资源切分 (完美整除)

```
  ┌────────┬──────────┬──────────┬──────────┐
  │ Node   │ FPGA ID   │ 层范围    │ 独家专家   │
  ├────────┼──────────┼──────────┼──────────┤
  │ Node 0 │ 00 ~ 07  │ L 00~14  │ Exp 000~095│
  │        │          │ 含Emb     │ 12/卡      │
  │ Node 1 │ 08 ~ 15  │ L 15~29  │ Exp 096~191│
  │ Node 2 │ 16 ~ 23  │ L 30~44  │ Exp 192~287│
  │ Node 3 │ 24 ~ 31  │ L 45~60  │ Exp 288~383│
  │        │          │ 含MTP     │ 含lm_head  │
  └────────┴──────────┴──────────┴──────────┘

  关键:
  ● 128 头 / 8 卡 = 16 头/卡  (TP 切分, 完美整除)
  ● 384 专家 / 32 卡 = 12 专家/卡  (专家并行, 完美整除)
  ● 共享专家每卡冗余 (33MB, 可忽略)
```

---

## 每卡 HBM 布局

```
  32 GB HBM2e 空间分配:

  ╔══════════════════════════════════════╗
  ║  权重区 (~24GB 预算, 实际占用 ~0.6GB)║
  ╠══════════════════════════════════════╣
  ║  12 路由专家 (fp4)      396 MB      ║
  ║  1 共享专家 (fp4)        33 MB      ║
  ║  Attention 权重         ~145 MB     ║
  ║  Router 权重            ~15 MB      ║
  ║  Embedding (Node 0 独有) 1.85 GB    ║
  ║  lm_head   (Node 3 独有) 1.85 GB    ║
  ╠══════════════════════════════════════╣
  ║  运行区 (~8GB)                       ║
  ║  KV Cache (256K×16层)    ~2.4 GB    ║
  ║  激活 Buffer             ~2.0 GB    ║
  ║  ETH Ring Buffer         ~0.5 GB    ║
  ╠══════════════════════════════════════╣
  ║  余量: >25 GB ← 大量空间可扩展      ║
  ╚══════════════════════════════════════╝
```

---

## 性能推演

```
  单 Token Decode (B=1):
    每层延迟:      ~65 μs (Attn 25 + MoE 40)
    15 层节点延迟:  ~975 μs
    全 61 层 (流水线):  ~4 ms/token
    单节点稳态吞吐:   ~960 tok/s
    4 节点集群吞吐:   ~1,000 tok/s (受最慢节点限制)

  B=4 微批次:
    每层延迟:      ~150 μs
    单节点吞吐:     ~435 tok/s
    集群总吞吐:     ~450 tok/s
    (吞吐下降, 但单 session 延迟改善)

  Prefill (128K context, B=32):
    预填充延迟:     ~2-3 秒 (全部 61 层)
    
  支持: 10-20 并发 session @ 50-100 tok/s each
```

---

## CH5 | RTL 核心模块

---

## 顶层模块划分 (13 个主模块)

```
  ds_v4_fpga_top
  ├── pcie_cxl_ep_wrapper      PCIe 5.0 Endpoint
  ├── inference_ctrl_fsm        全局推理状态机
  ├── mla_attention_pipeline    MLA Attention 流水线
  │   ├── q_compress_unit      Q: 7168→1536
  │   ├── kv_compress_unit     KV: 7168→576
  │   ├── qk_dot_product_unit  Q·K^T (128头)
  │   ├── online_softmax       Safe Softmax
  │   ├── av_dot_product_unit  A·V
  │   └── o_decompress_unit    O: LoRA→7168
  ├── rope_hardware_unit       Decoupled RoPE
  ├── moe_expert_core          fp4 脉动阵列
  ├── shared_expert_unit       共享专家
  ├── kv_cache_manager         KV Cache 硬件寻址
  ├── hbm_memory_controller    HBM2e 2048-bit
  ├── chip2chip_router         片间通信引擎
  ├── rms_norm_unit            RMSNorm (eps=1e-6)
  └── debug_monitor            ILA + 性能计数器
```

---

## 关键模块 1: fp4 脉动阵列

```
  fp4 (E2M1) × fp8 (E4M3) 乘法器设计:

    方案: 查表预计算 + DSP INT 模式

    ● fp4 只有 16 种值 (15 有效)
    ● 用 BRAM 存储 fp4→INT8 预计算的缩放因子
    ● DSP58 在 18×19 模式下跑 2× INT8 MAC/cycle

    阵列配置:
      8 个 128×128 脉动阵列
      每阵列: 16,384 个 MAC 单元
      合计: 8 × 128 × 128 = 131,072 MAC/cycle
      @ 450 MHz: 59 GMACs/s per array
      8 阵列总计: 8.44 TMACs/s

    HBM 喂入:
      加载 33MB 专家权重 (fp4): 36 μs
      喂入 32 tokens × 7168 激活 (FP8): 230 μs
      → 权重加载是瓶颈, 不是计算
```

---

## 关键模块 2: MLA Attention 流水线

```
  每层 Attention 阶段流水线:

  Stage 0: Q 压缩    (7168×1536)  [HBM:6μs]  → 6.0 μs
  Stage 1: KV 压缩   (7168×576)   [HBM:2.3μs] → 2.3 μs
  Stage 2: Q·K^T     128头并行    [DSP:3.8μs]  → 3.8 μs
  Stage 3: Softmax   硬化         [0.2μs]      → 0.2 μs
  Stage 4: A·V       128头并行    [DSP:3.7μs]  → 3.7 μs
  Stage 5: O 解压    LoRA ×2级    [HBM:8.7μs]  → 9.3 μs
  ────────────────────────────────────────────
  总延迟:              ~25 μs (部分可重叠)

  关键:
  ● 没有 V 投影矩阵 — MLA 的 V 直接复用 c_KV latent
  ● Q 和 O 走 LoRA 压缩 — 降低 Attention 权重 70%
  ● 128 头全部并行计算, 无顺序依赖
```

---

## 关键模块 3: KV Cache 硬件管理器

```
  HBM 寻址: {session_id, layer_id, seq_id} → 物理地址
  (硬件哈希, 无需软件 page table)

  容量 (每卡):
    256K context × 16 layers × 576B = 2.36 GB
    512K context × 16 layers × 576B = 4.72 GB
    1M   context × 16 layers × 288B = 4.61 GB (FP4 量化)

  特性:
    ● Sliding Window (128) 硬件实现
    ● 新 token 到来 → 自动写入, 自动淘汰最旧 token
    ● 与 vLLM PagedAttention 的本质区别:
      软件 Block Table → 硬件地址生成器 (零 CPU 开销)
```

---

## CH6 | 组网通信方案

---

## 集群拓扑

```
                   400GbE Spectrum Switch
                  (RoCE v2, DCQCN, PFC+ECN)
                 ┌─────────────────────────┐
                 │  400GbE × 4 端口         │
                 └──┬─────┬─────┬─────┬────┘
                    │     │     │     │
            400GbE QSFP-DD (或 200GbE ×2)
                    │     │     │     │
              ┌─────┘     │     │     └─────┐
              ▼           ▼     ▼           ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
         │ Node 0  │ │ Node 1  │ │ Node 2  │ │ Node 3  │
         │ 8 FPGA  │ │ 8 FPGA  │ │ 8 FPGA  │ │ 8 FPGA  │
         │ F-Tile 200GbE   │ │ F-Tile 200GbE   │ │ F-Tile 200GbE   │ │ F-Tile 200GbE   │
         └─────────┘ └─────────┘ └─────────┘ └─────────┘
              │           │           │           │
         L 0-14       L15-29      L30-44      L45-60
         +Emb                                 +lm_head+MTP
```

---

## 通信模式与带宽

```
  ┌────────────────────┬──────────┬──────────────┐
  │ 通信模式            │ 每层数据量│ 链路利用率    │
  ├────────────────────┼──────────┼──────────────┤
  │ 组内 TP All-Reduce  │  ~224KB │  ~0.3%       │
  │ 组内 MoE Dispatch   │  ~640KB │  ~0.5%       │
  │ 跨节点 MoE Dispatch │  ~640KB │  ~3%         │
  │ Pipeline 边界       │  ~7KB   │  ~0.01%      │
  │ 结果回传            │  ~0.3KB │  忽略         │
  └────────────────────┴──────────┴──────────────┘

  FPGA → Switch 链路: x8 PCIe 5.0 = 28 GB/s 有效
  需求: ~2.7 GB/s → 9.6% 利用率

  跨节点: 200GbE = 25 GB/s
  需求: ~2.5 GB/s per node → 10% 利用率

  → 带宽不是瓶颈, 延迟才是关键
```

---

## 延迟分析

```
  单次通信延迟:
    同 CPU 下 FPGA P2P:      ~260 ns
    跨 CPU (UPI) FPGA P2P:   ~500 ns
    跨节点 RDMA (同 Switch):  ~3 μs
    
  单 MoE 层 All-to-All 总延迟:
    6 个 expert 并行 dispatch + compute + combine
    ≈ 40 μs (含远端 HBM 权重加载)

  总推理延迟占比:
    TP All-Reduce  (61 层):  ~55 μs  ←  < 1.4%
    MoE All-to-All (58 层):  ~87 μs  ←  < 2.2%
    计算 + HBM:            ~3,850 μs ← 96.4%
    
  → 通信延迟在总延迟中可忽略
```

---

## F-Tile 内置 Ethernet (无需外部 NIC)

```
  Agilex 7 M F-Tile 硬核 Ethernet MAC/PCS/FEC:
  
    ● 400GbE / 4×100GbE / 2×200GbE 硬核 MAC
    ● RS-FEC (IEEE 802.3 Clause 134) 硬核
    ● PCS (100G/200G/400G) 硬核
    ● G8 收发器 up to 116 Gbps
    ● 板载 QSFP-DD Cage → 直连交换机
    
    零外部 NIC 的优势:
    ├─ 零 PCIe 中转: FPGA 数据面直出 Ethernet, 零 CPU 参与
    ├─ 延迟降低: 省掉 FPGA→NIC PCIe DMA (~1μs)
    ├─ 成本节省: 省掉 Intel E830 ¥12K/节点 + 1 PCIe 槽
    ├─ 功耗节省: F-Tile MAC ~5W vs NIC ~25W
    └─ 供应简化: 不依赖任何 NIC 供应商
    
    备选: 兼容模式下仍可插标准 NIC (如华为 SP680)
```

---

## CH7 | 平台与物理形态

---

## FPGA 算力卡: 标准 PCIe 双宽卡

```
  ┌──────────────────────────────────────┐
  │                                      │
  │  尺寸: FHFL 111.15mm × 312mm        │
  │        PCIe 双槽宽 (×2 bracket)     │
  │                                      │
  │  ├─ Agilex 7 M AGFB027             │
  │  ├─ 32 GB HBM2e (2 Stack)          │
  │  ├─ VRM (Core + HBM + IO)          │
  │  ├─ PCIe 5.0 x16 金手指            │
  │  ├─ 2× PCIe 8-pin AUX 供电         │
  │  ├─ 被动散热片 (依赖服务器风道)     │
  │  ├─ SMBus → BMC (温度/功耗上报)     │
  │  ├─ JTAG Header (调试用)           │
  │  └─ QSFP-DD Cage (可选, 预留直出)  │
  │                                      │
  │  TDP:  ~75W (card) + 外部          │
  └──────────────────────────────────────┘
```

---

## 服务器: Supermicro SYS-821GE-TNHR

```
  ┌─────────────────────────────────────────┐
  │  [风扇墙]                                │
  │  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐
  │  │ 0│ │ 1│ │ 2│ │ 3│ │ 4│ │ 5│ │ 6│ │ 7│  ×8 FPGA
  │  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘
  │  [→风道] [→风道] [→风道] [→风道] ...    │
  │                                          │
  │  10× PCIe 5.0 x16 FHFL 插槽             │
  │  双路 Xeon SPR (160 Lane total)          │
  │  4× 3000W PSU (2+2 冗余)                │
  └──────────────────────────────────────────┘

  为什么选它:
    ● X13 (SPR) → X14 (GNR) → X15 (DMR) 同机箱持续迭代
    ● 不关心插的是 GPU 还是 FPGA
    ● 全球采购 (不受管制)
```

---

## 跨代兼容性保证

```
  2025    X13 + Agilex 7 M   卡, Gen5 x16     ← 当前
  2027    X14 + 同一张卡        Gen5 仍可用
  2029    X15 + Agilex 10 M  卡, Gen6 x16     ← 升级
          └─ 旧卡插新机: Gen5 降速, 正常工作
          └─ 新卡插旧机: Gen5 降速, 正常工作

  硬约束:
    ✓ PCIe CEM 5.0 标准金手指 (不自定义连接器)
    ✓ 标准 8-pin PCIe AUX 供电
    ✓ FHFL 尺寸 (所有 GPU 服务器支持)
    ✓ SMBus + IPMI 标准 BMC 管理
    ✓ Linux 标准 VFIO 驱动

  → 算力卡和平台完全解耦, 各自独立进化
```

---

## CH8 | 软件生态

---

## 用户调用方式

```
  ┌────────────────────────────────────────┐
  │  应用开发者                              │
  │                                        │
  │  import openai                         │
  │  client = OpenAI(                      │
  │    base_url="http://fpga-cluster:8080/v1")│
  │  client.chat.completions.create(       │
  │    model="deepseek-v4",                │
  │    messages=[...])                     │
  │                                        │
  └────────────────┬───────────────────────┘
                   │ OpenAI REST API
  ┌────────────────▼───────────────────────┐
  │  推理服务层 (x86 主控)                  │
  │  · Tokenizer + 采样器                   │
  │  · 多 session 并发调度                  │
  │  · 流式输出 (SSE)                       │
  │  · FastAPI HTTP Server                │
  └────────────────┬───────────────────────┘
                   │ VFIO + libfpga.so
  ┌────────────────▼───────────────────────┐
  │  FPGA 算力卡 (硬件)                     │
  └────────────────────────────────────────┘
```

---

## 生态兼容矩阵

```
  ┌─────────────────────┬────────────┬──────────┐
  │ 框架/工具            │ 兼容方式    │ 工作量    │
  ├─────────────────────┼────────────┼──────────┤
  │ OpenAI Python SDK   │ HTTP 100%  │ 零       │
  │ LangChain/LlamaIdx  │ HTTP API   │ 零       │
  │ Dify / FastGPT      │ HTTP API   │ 零       │
  │ Open WebUI          │ HTTP API   │ 零       │
  │ Continue.dev        │ HTTP API   │ 零       │
  │ vLLM                │ Fork(可选) │ 3-6 人月 │
  │ HuggingFace TF      │ 不兼容     │ N/A      │
  │ PyTorch Runtime     │ 不兼容     │ N/A      │
  └─────────────────────┴────────────┴──────────┘

  关键:
  ● 不跑 PyTorch, 不跑 CUDA
  ● 模型训练在 GPU 上完成 → 导出 fp4 checkpoint
  ● 推理在 FPGA 裸金属上运行
  ● 用户只看到 OpenAI API, 看不到 FPGA
```

---

## 驱动模型: Linux 标准 VFIO

```
  FPGA 设备对 Linux 的呈现:

    $ lspci | grep FPGA
    04:00.0 Class 1200: Intel Corporation Device XXXX

    BAR0: 256 MB (MMIO 控制寄存器)
    BAR2:  32 GB (HBM 直通, 64-bit prefetchable)

  用户态驱动 (无内核模块):

    fpga = vfio_open("0000:04:00.0");
    hbm  = vfio_mmap_bar2(fpga);
    vfio_send_cmd(fpga, &inference_cmd);
    vfio_wait_completion(fpga, MSI-X interrupt);

  ✓ 不写内核代码
  ✓ 不依赖 Intel oneAPI / FPGA AI Suite
  ✓ 不依赖任何闭源 SDK
  ✓ 标准 Linux 部署
```

---

## CH9 | 开发路线与里程碑

---

## 五阶段递进开发

```
  Phase 1  单卡验证    Month 1-2    ── 1 FPGA
  Phase 2  8 卡组内    Month 3-4    ── 1 节点
  Phase 3  双节点互联  Month 5-6    ── 2 节点
  Phase 4  四节点集群  Month 7-8    ── 4 节点, 32 卡
  Phase 5  生产调优    Month 9-10   ── Benchmark + 优化

  ████████░░░░░░░░░░░░░░░░░░  Phase 1  单卡
  ████████████████░░░░░░░░░░  Phase 2  8 卡
  ████████████████████████░░  Phase 3  双节点
  ███████████████████████████  Phase 4  全集群
```

---

## Phase 1: 单卡验证 (最关键的阶段)

```
  目标: 验证核心假设, 决定项目去留

    □ PCIe 5.0 EP 链路调通
    □ HBM2e 读写带宽 ≥ 80% 理论值 (736 GB/s)
    □ fp4×fp8 脉动阵列 bit-accurate vs PyTorch reference
    □ 单层推理 Micro-benchmark (延迟/功耗/精度)
    □ fp4 精度验证:
        - Per-layer output diff < 1e-3
        - 61 层累积 diff < 2%
        - 超过任一阈值 → 启动 fp8 备选方案

  Go/No-Go 标准:
    ✗ HBM 带宽 < 50% 理论值 → 停
    ✗ fp4 累积精度差 > 2% → 启动备选 (fp8 权重)
    ✗ 单层延迟 > 200μs → 分析瓶颈 → 决定
```

---

## Phase 2-5: 递进集成

```
  Phase 2: 单节点 8 卡 (Month 3-4)
    ● PCIe P2P 验证 (同 CPU + 跨 CPU UPI)
    ● 8 卡 TP All-Reduce
    ● 组内 MoE Dispatch
    ● 8 卡跑通 15 层完整推理
    ● 吞吐 target: >200 tok/s

  Phase 3: 双节点互联 (Month 5-6)
    ● F-Tile 200GbE + Switch 部署
    ● RoCE v2 RDMA 跨节点通信
    ● 跨节点 MoE Dispatch + Combine
    ● 双节点 30 层流水线

  Phase 4: 四节点全集群 (Month 7-8)
    ● 32 卡完整 61 层 + MTP
    ● 128K context 长序列
    ● 多 session 并发

  Phase 5: 生产优化 (Month 9-10)
    ● 512K/1M context 极限测试
    ● 热门专家 Multi-replica
    ● 故障注入 + Failover
    ● OpenAI API 兼容认证
```

---

## CH10 | 成本与财务

---

## 原型开发预算

```
  ┌──────────────────────────────┬──────────┐
  │ 项目                         │ 金额 (¥)  │
  ├──────────────────────────────┼──────────┤
  │ 硬件 (芯片+物料 1:1)          │          │
  │   32 × FPGA 芯片 (¥21.6K)    │  0.69M   │
  │   32 × 卡级物料 (PCB/散热/组装)│  0.69M   │
  │   4 × 服务器机头 (¥170K)     │  0.68M   │
  │   1 × 400GbE Switch          │  0.10M   │
  │   线缆/电源/机柜              │  0.06M   │
  │   备件 (2 FPGA 卡)           │  0.09M   │
  │   硬件小计                    │  2.31M   │
  ├──────────────────────────────┼──────────┤
  │ 人力                         │          │
  │   5 FPGA RTL × 10 月          │  3.33M   │
  │   3 软件 × 10 月              │  1.50M   │
  │   1 PCB × 5 月               │  0.25M   │
  │   2 测试验证 × 8 月           │  0.67M   │
  │   人力小计                    │  5.75M   │
  ├──────────────────────────────┼──────────┤
  │ 其他 (工具/IP/测试/不可预见)  │  2.00M   │
  ├──────────────────────────────┼──────────┤
  │ 总计 (~20% 余量)              │ ~12.00M  │
  └──────────────────────────────┴──────────┘

  物料:人工 = 1:2.5 (原型特征: 硬件不贵人贵)
```

---

## 量产成本路径

```
  原型 (1 套 32 卡):      ¥12M  (含全部 R&D)
  小批量 (5 套):          ¥5M/套 (R&D 摊薄)
  量产 (10+ 套):           ¥2M/套 (硬件物料)

  成本下降驱动:
    ● FPGA 芯片: ¥21.6K → ¥18K (小批量折扣)
    ● 外围物料: ¥21.6K → ¥15K (批量 PCB/组装)
    ● 服务器机头: ¥170K → ¥150K (框架采购)
    ● 交换机/线缆量: 几乎不变
    ● RTL NRE 已摊完

  单卡物料成本: ¥43K (原型) → ¥33K (量产)
  对比:
    Ascend 910B 单卡: ~¥180K (但供货不稳定, 出不去)
    FPGA 卡: ¥33K + 全球可部署 = 战略溢价
  
  人工:物料 = 2.5:1 (原型) → 0.4:1 (10套摊薄)
```

---

## CH11 | 竞争分析

---

## 与替代方案的对标矩阵

```
  ┌──────────────┬──────┬──────┬──────┬──────┐
  │              │NVIDIA│Ascend│国产  │本方案 │
  │              │ B200 │910C  │GPU   │FPGA  │
  ├──────────────┼──────┼──────┼──────┼──────┤
  │ 中国可获取    │  ✗   │  △   │  △   │  ✓   │
  │ 全球可部署    │  ✓   │  ✗   │  ✗   │  ✓   │
  │ 供应稳定性    │  ✗   │  △   │  △   │  ✓   │
  │ fp4 原生      │  ✗   │  ✗   │  ✗   │  ✓   │
  │ MLA 硬件加速  │  ✗   │  ✗   │  ✗   │  ✓   │
  │ 软件生态      │★★★★★│ ★★★★ │★★~★★★│ ★★   │
  │ 部署灵活性    │  ★★  │  ★★  │  ★★  │★★★★★│
  │ 运维成熟度    │★★★★★│ ★★★  │  ★★  │ ★★   │
  └──────────────┴──────┴──────┴──────┴──────┘

  唯一性: 本方案在 "可获取性" + "全球部署" 上双满分
  这是结构性优势, 其他方案无法复制
```

---

## FPGA 护城河

```
  护城河 ①: fp4 原生推理

    国产 GPU (Ascend/寒武纪/海光/壁仞) → 全都不支持 fp4
NVIDIA B200/GB200 支持 fp4, 但受出口管制不可获取
    DeepSeek V4 Pro 权重 = fp4
    GPU 方案: fp4→解压→FP8→Tensor Core → 浪费 HBM
    FPGA 方案: fp4 → fp4 MAC → 零解压 → 全链路 fp4

  护城河 ②: MLA 硬件加速

    MLA = DeepSeek 独家 Attention 架构
    GPU 方案: 需要定制 CUDA/CANN kernel 实现 Q/KV/O 压缩/解压
    FPGA 方案: 硬连线 MLA pipeline, 零软件开销

  护城河 ③: KV Cache 硬件管理

    GPU 方案: vLLM PagedAttention → Block Table 软件管理
    FPGA 方案: 硬件哈希寻址 → {session, layer, seq} → HBM addr
              零 CPU 参与 KV Cache 管理
```

---

## 供应链对比

```
  NVIDIA B200:
    TSMC 4nm → CoWoS-L → HBM3e (SK Hynix)
    全链条受美国管制, 中国 zero allocation

  华为 Ascend 910C:
    SMIC 7nm → CoWoS → HBM2e (Samsung, 受限)
    SMIC 产能有限, 华为内部优先
    先进封装设备受荷兰/日本管制

  FPGA 方案:
    Intel 7 (Intel 自有 Fab) → EMIB (Intel 自有)
    → HBM2e (SK Hynix / Samsung)
    供应链分散在 4+ 国家/地区
    不受任何单一司法管辖区完全控制
    不受 GPU 算力出口管制条款约束
```

---

## CH12 | 风险评估

---

## 风险矩阵

```
  ┌───┬──────────────────────┬──────┬──────┬──────────────┐
  │ # │ 风险                  │ 概率 │ 影响 │ 对策          │
  ├───┼──────────────────────┼──────┼──────┼──────────────┤
  │ 1 │ DeepSeek V5 变架构    │  中  │  高  │ 跟踪V5动态    │
  │ 2 │ fp4 精度 61 层超标    │  中  │  高  │ Phase1验证    │
  │ 3 │ FPGA 遭新管制         │ 中低 │ 极高 │ 保持库存      │
  │ 4 │ Agilex 7 M 供应波动   │  低  │  高  │ 签供货协议    │
  │ 5 │ PCIe P2P 跨 CPU 兼容  │  中  │  中  │ 备选Switch卡  │
  │ 6 │ 人才获取              │  高  │  中  │ 高校合作      │
  │ 7 │ Ascend 突然支持 fp4   │  中  │  高  │ 定位不改变    │
  │ 8 │ 运维复杂度            │  中  │  中  │ BIST + runbook│
  └───┴──────────────────────┴──────┴──────┴──────────────┘
```

---

## 风险 1: DeepSeek V5 改变架构 (P=中, I=高)

```
  场景: DeepSeek V5 放弃 MLA, 改用新 Attention 机制

  影响:
    ● MLA pipeline RTL → 全部报废
    ● fp4 脉动阵列 → 仍然可用 (不依赖 Attention 机制)
    ● 需重新设计 Attention 模块: 6-12 个月

  对策:
    ● 跟踪 DeepSeek 研发动态 (公开论文、技术报告)
    ● MLA 从 V2→V3→V4 持续三代, 是 DeepSeek 核心壁垒
      大概率 V5 仍然保留 MLA
    ● 关键矩阵维度 (d_model=7168) 三代未变
    ● 将维度参数化: n_layers, n_heads, d_model 可配置

  判断: MLA 放弃概率低 (DeepSeek 已投资 3 代)
       d_model 变更概率中低 (三代稳定在 7168)
       moe_intermediate 变更概率中 (V4 已从 2048 扩到 3072)
```

---

## 风险 2: fp4 精度 61 层累积超标 (P=中, I=高)

```
  问题: fp4 只有 15 个有效值, 每层舍入误差累积

  验证方法 (Phase 1):
    ① 单层: FPGA 脉动阵列输出 vs PyTorch fp4 reference
       → diff < 1e-3 (per element)
    ② 61 层累积: 用真实 fp4 checkpoint 跑完 61 层
       → 最终 hidden state diff < 2%
    ③ 端到端: 用标准 benchmark (MMLU/HumanEval) 跑分
       → score 差异 < 1%

  Go/No-Go: 任一项不达标 → 启动备选方案

  备选方案:
    方案 B: 专家权重 fp8, Attention fp8
            → HBM 权重占用 2× (589MB → 1.2GB, 仍远 < 24GB)
            → 计算精度保证, 但 HBM 带宽压力 2×
    方案 C: 混合精度 — 前几层 fp8, 后面 fp4
            → 平衡精度和带宽
```

---

## 风险 3: FPGA 遭新一轮管制 (P=中低, I=极高)

```
  情景: 美国将高密度 FPGA (含 HBM) 加入出口管制清单

  影响:
    ● Agilex 7 M 对中国断供 (类似 GPU 管制)
    ● 但已购芯片不受追溯管制
    ● 海外部署不受影响 (Intel 全球供应)

  当前状态:
    ● Agilex 7 M 的 TPP 远低于 4800 阈值
    ● FPGA 传统上不被视为 AI 加速器
    ● 但管制有扩大趋势 (从 GPU 扩散到 AI 芯片)

  对策:
    ① 一次性采购足够的原型 + 初批量产芯片 (100-200 片)
    ② 评估 Agilex 7 F-Series (无 HBM) + 外部 DDR4/DDR5 方案
    ③ 长期跟踪国产 FPGA 的 HBM 集成进展
       (当前无, 但 3-5 年内可能出现)

  和 Ascend 的区别:
    Ascend 被制裁 = 华为停摆 (依赖单一供应商)
    FPGA 被管制 ≠ Intel 停摆 (只是对中国禁运, 全球仍可用)
    → 海外部署能力不受影响
```

---

## Go/No-Go 决策门

```
  Phase 1 结束后 (Month 2):

  □ 1. fp4 精度差异 > 2%  →  停 / 启用 fp8 备选
  □ 2. HBM 带宽实测 < 50% 理论值  →  停
  □ 3. Intel 供货确认 > 26 周周期  →  重新评估

  任何一条触发, 项目必须暂停重评。

  Phase 4 结束后 (Month 8):

  □ 4. 32 卡集群吞吐 < 300 tok/s  →  评估经济性
  □ 5. 单卡故障率 > 1/月  →  重新设计散热/供电

  前两条是硬停止条件, 后两条是经济性评估条件。
```

---

## 路线图总览

```
  2025 H2    Phase 1-2    单卡 + 8 卡节点
  2026 H1    Phase 3-4    双节点 + 全集群
  2026 H2    Phase 5      优化 + 生产化
  2027       量产         首批客户部署

  里程碑:
    Month 2  单卡 fp4 精度验证 ← Go/No-Go #1
    Month 4  8 卡组内推理       ← 首个系统级演示
    Month 6  双节点跨网络推理   ← 验证 RDMA 架构
    Month 8  32 卡集群 Benchmark ← 性能定标
    Month 10 生产化完成         ← 向客户交付
```

---

## 总结

```
  战略层: 中国大模型出海 → 唯一可全球部署的推理硬件
  ─────────────────────────────────────────────────
  架构层: 32 FPGA, 4×8 标准节点, 400GbE RDMA
  ─────────────────────────────────────────────────
  技术层: fp4 原生 + MLA 硬化 + HBM 常驻权重
  ─────────────────────────────────────────────────
  生态层: OpenAI API 兼容, Linux VFIO 标准驱动
  ─────────────────────────────────────────────────
  财务层: ¥12M 原型, 10 个月到全系统, ¥2M 量产
  ─────────────────────────────────────────────────

  下一步:
    → 启动 Phase 1 单卡验证
    → 采购 2 张 Agilex 7 M 开发板
    → 验证 fp4 精度 + HBM 带宽
```

---

## Q&A

```
  联系方式: [待填]
  文档版本: v1.0
  日期: 2025-05
```
