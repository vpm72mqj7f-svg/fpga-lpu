## 7. 服务器平台与物理形态

### 7.1 FPGA 算力卡物理规范

```
4× AGM 039 加速卡:
  ┌──────────────────────────────────────────────┐
  │  形态:  FHFL Extended (全高全长加长)          │
  │         111.15mm × 340mm, 双槽宽             │
  │         (标准 FHFL 312mm + 28mm 延伸,          │
  │          4U 服务器均可支持)                    │
  │                                              │
  │  芯片:  4 × AGM 039-F (32GB HBM)             │
  │         56mm × 66mm R47A 封装                │
  │         单行排列, 片间距 12mm                  │
  │                                              │
  │  接口:  PCIe 5.0 x16 CEM 金手指               │
  │         (仅 Chip0 R-Tile 对外)                │
  │         无 QSFP-DD Cage                       │
  │         (片间走板内 C2C SerDes)               │
  │                                              │
  │  供电:  75W PCIe slot                        │
  │         + 12VHPWR 600W (单连接器)             │
  │         = 675W 总额定 → 550W 满载             │
  │         余量 23%                              │
  │                                              │
  │  散热:  4 片分离, 独立散热片 + 均温板          │
  │         4U 服务器 120mm 风扇阵列              │
  │         前→后强制风冷                          │
  │         风道长度 ~500mm                       │
  │         芯片结温 < 85°C (Extended temp range) │
  │                                              │
  │  管理:  SMBus (I2C) ×4 (每片独立)             │
  │         Chip0 汇总 → BMC (IPMI 标准)          │
  │         温度/功耗/链路状态/心跳                │
  └──────────────────────────────────────────────┘

卡片级布局 (~340mm × 111mm):

  ← Bracket                    12VHPWR →
  ┌──────┬────────┬────────┬────────┬────────┬──────┐
  │ PCIe │ AGM039 │ gap    │ AGM039 │ gap    │ VRM  │
  │ x16  │ Chip0  │ 12mm   │ Chip1  │ 12mm   │      │
  │ conn │ 56×66  │        │ 56×66  │        │      │
  ├──────┴────────┴────────┴────────┴────────┴──────┤
  │  gap 12mm  │ AGM039 │ gap    │ AGM039 │ aux    │
  │            │ Chip2  │ 12mm   │ Chip3  │ conn   │
  │            │ 56×66  │        │ 56×66  │        │
  └────────────────────────────────────────────────┘

Chip0 位置靠近 PCIe 金手指 (R-Tile 直连), 减少 PCIe trace 长度。
Chip1/2/3 在板上均匀分布, 片间 C2C trace < 200mm。
```

### 7.2 服务器平台

| 平台 | 推荐 | PCIe 槽 | PSU | 备注 |
|------|------|---------|-----|------|
| **Inspur NF5688M7** | ★★★★★ | 8× x16, 4U | 2×3000W | 国产首选, 12VHPWR 就绪 |
| **Lenovo SR670 V3** | ★★★★★ | 8× x16, 4U | 2×2600W | 支持 extended FHFL |
| **Supermicro SYS-841GE-TNHR** | ★★★★ | 8× x16, 4U | 2×2600W | X13→X14→X15 |
| H3C R5500 G6 4U | ★★★★ | 8× x16 | 2×3000W | 国产次选 |

```
单台 4U 服务器供电核算:

  8 FPGA 卡:  8 × 550W = 4,400W
  CPU + I/O:             500W
  风扇 (120mm ×8):       200W
  ─────────────────────────
  单台总计:             ~5,100W

PSU 2×3000W (1+1 冗余):
  5100W / 3000W = 170% → 单 PSU 不能承载全部
  → 需负载均衡模式 2×2550W < 3000W, 余量 18%
  → 或 2+0 模式 (双 PSU 同时供电), 可接受 (GPU 服务器常用)

FPGA 余量 vs GPU:
  H100 8卡: 8×700W + 500W = 6,100W
  FPGA 8卡:             = 5,100W
  → FPGA 负载更轻, GPU 方案的 PSU 即可覆盖
```

### 7.3 跨代兼容保证

```
  2025:  4U 服务器 (Xeon SPR) + Agilex 7 M 卡,   Gen5 x16
  2027:  4U 服务器 (Xeon GNR) + 同一张卡,          Gen5 仍可用
  2029:  4U 服务器 (Xeon NVL) + Agilex 10 M 卡,   Gen6 x16
         └─ 旧卡插新机: 降速 Gen5, 正常工作
         └─ 新卡插旧机: 降速 Gen5, 正常工作

  关键约束:
    ✓ 标准 PCIe CEM 金手指 (不自定义连接器)
    ✓ 标准 12VHPWR 供电 (不依赖主板自定义)
    ✓ FHFL Extended 尺寸 (4U GPU 服务器已支持)
    ✓ SMBus IPMI 标准管理 (不用私有 BMC 协议)
    ✓ Linux 标准 VFIO 驱动 (不依赖闭源 SDK)
```

### 7.4 驱动模型

```
Linux 内核:
  ├── PCIe Subsystem (VFIO)
  ├── /dev/vfio/N   ← 用户态直接控制 FPGA (每卡 1 设备)
  ├── MSI-X 中断    ← 推理完成 / 错误 / 心跳 (per chip)
  ├── IOMMU         ← DMA 地址隔离
  ├── PCIe P2P      ← drivers/pci/p2p.c (内核原生支持)
  └── 无内核模块    ← 不碰内核 API, 零维护

用户态:
  ├── libfpga.so    ← C 库, VFIO mmap + P2P DMA
  ├── fpga_infer()  ← 推理 API
  ├── p2p_setup()    ← PCIe P2P BAR mapping (一次初始化)
  └── 对接推理服务层

P2P 配置 (一次性):
  echo 1 > /sys/bus/pci/devices/0000:01:00.0/p2pmem/enable   # Card A
  echo 1 > /sys/bus/pci/devices/0000:02:00.0/p2pmem/enable   # Card B
  → 之后 PCIe MWr 直接在卡间转发, 不经 CPU memory
  → v5.4+ 内核原生支持, 无需打补丁
```

### 7.5 功耗分析与散热方案

**7.5.1 单卡功耗拆解**

```
AGM 039 ×4 板级功耗估算 (满载推理):

  单片 AGM 039:
    DSP core (12,300 blocks, 450MHz, 50% util):  ~52W
    HBM2e (32GB, 持续读写):                       ~18W
    PCIe 5.0 (R-Tile, Chip0 only):                ~8W
    C2C SerDes (F-Tile, 8 lane NRZ):              ~6W
    M20K/MLAB (75% util):                        ~10W
    静态功耗 (10nm SuperFin, 039 更大 die):        ~14W
    ─────────────────────────────────────────
    单芯片:                                        ~108W → 取 110W

  4 芯片合计: 4 × 110W = 440W

  PCB 辅助:
    VRM 损耗 (4 路独立 SmartVID, ~12%):            ~53W
    时钟/复位/JTAG/调试:                            ~5W
    12VHPWR 连接器损耗:                             ~5W
    SMBus/I²C/BMC:                                ~3W
  ─────────────────────────────────────────
  单卡板级满载:                                    ~506W → 取 510W
  含 10% 余量:                                    ~560W → 取 550W (标称)

比较: H100 SXM TDP 700W — 4×FPGA 单卡 550W, 功耗更低。
      每 TOPS 功耗: FPGA 550W / 74 TFLOPS = 7.4 W/TFLOPS
                    H100 700W / 990 TFLOPS = 0.7 W/TFLOPS (FP8)
      但 FPGA 是 fp4 native, GPU 需量化 → 性能/W 实际差距更小
```

**7.5.2 整机 Wall Power**

```
┌──────────────────────────────┬──────────┬────────────────────┐
│ 组件                          │ 数量      │ 功耗                │
├──────────────────────────────┼──────────┼────────────────────┤
│ 4×AGM 039 加速卡 (满载)       │ 8        │ 8×550W = 4,400W    │
│ 4U 服务器机头 (双 Xeon+外设)  │ 1        │ ~500W              │
│ 风扇 (120mm ×8, full speed)  │ 8        │ ~200W              │
│ PSU 损耗 (80+ Titanium, ~6%) │ —        │ ~200W              │
├──────────────────────────────┼──────────┼────────────────────┤
│ 整套 Wall Power               │          │ ~5,300W ≈ 5.3kW    │
└──────────────────────────────┴──────────┴────────────────────┘

vs 旧方案: 6.2kW (4 节点+2 Switch)
节省: 15%, 同时算力 +31%

vs H100 8卡: 5,600W (GPU) + 500W (CPU) = 6,100W
FPGA 单套: 5,300W → 比 GPU 集群还低 13%

单机架 (42U) 密度:
  5.3kW /套, 每套占 5U (4U 服务器 + 1U 线缆管理)
  可放: 42U / 5U = 8 套
  功率: 8 × 5.3kW = 42.4kW → 超出风冷上限 (15kW)
  实际: 2-3 套/机架 (混合部署), 或走液冷 (量产)
```

**7.5.3 供电验证**

```
单台 4U 服务器 (Inspur NF5688M7):
  PSU: 2× 3,000W (1+1 冗余, 80+ Titanium)
  负载: 5,100W
  
  1+1 模式: 5,100W / 3,000W = 170% → 单 PSU 不够
  2+0 模式: 5,100W / 6,000W = 85% → 双 PSU 同时供电, OK
    单 PSU 故障 → 另一台超载 → 需降频 (GPU 服务器同类方案)
  
  或选配 2× 3,500W PSU → 1+1 冗余满足

FPGA 负载特性 (vs GPU):
  GPU:  瞬态 spike 1.5-2× TDP (数百 μs)
  FPGA: 几乎无瞬态 spike (DSP 稳定负载)
  → PSU 稳定性更好, 电网设计更简单
```

**7.5.4 散热方案**

```
4U 服务器风冷:

  风道: 前 120mm 风扇 ×6-8 → 卡阵列 → 后排气
  风量: 120mm ×8 @ 3,000 RPM ≈ 250 CFM
  温升: ΔT = 5,100W / (0.316 × 250 CFM) ≈ 64°C
        (入口 25°C → 出口 89°C, 偏高)
  需验证: 实际 deployment 可能需降低入口温度或提高风量

  卡级散热:
    4 片 56×66mm 封装分离布局 → 热源分散
    每片 ~110W / (56×66mm²) ≈ 30 W/cm²
    vs H100: 700W / ~800mm² = 87 W/cm²
    → FPGA 热密度仅 H100 的 1/3, 风冷可行性更好

  均温板 (Vapor Chamber):
    每片独立 VC + 铝鳍片
    热阻: 芯片→air < 0.15°C/W → 结温升高 < 110W × 0.15 = 16.5°C
    入口 35°C → 结温 ~52°C, 远低于 85°C 规范

  对比 GPU 散热:
    H100 8卡 6,100W → 多需液冷 (DGX H100 标配液冷)
    FPGA 8卡 5,100W → 高风量风冷即可
    节省: 液冷基建 ¥50-100K → ¥0
```

**7.5.5 功耗优化空间**

```
降功耗手段 (Phase 3+):

  ① R-Tile 降宽: x16→x8 (P2P 带宽够用)
     → 省 ~4W/片, 8 片省 32W (仅 Chip0 有 R-Tile)

  ② C2C SerDes 降频: 32G→16G NRZ (带宽远未饱和)
     → 省 ~3W/片 × 32 片 = 96W

  ③ DSP 动态调频: HBM-bound 时降 DSP 频率
     → 省 ~15W/片 × 32 片 = 480W

  ④ HBM 低功耗模式: 无 token 层 standby
     → 省 ~8W/片 × 32 片 = 256W

  四项合计: 可降至 ~85W/片, 整机 ~3.7kW

  实现前提: RTL 支持动态电源管理, Phase 3 完成
```

---

## 8. 软件生态与推理服务层

> **质疑 C**: "软件栈从零开始，谁来用？没有 vLLM 级别的 serving framework、没有 profiler、没有 debugger。ML 工程师不会碰 Verilog。OpenAI API 兼容不是贴个 endpoint 就完事了——continuous batching、prefix caching、speculative decoding、P/D 分离这些 serving 系统功能，FPGA 方案里有几个？"

**直接回答：软件栈不是"从零开始"，是"绕开了 CUDA，上面全是现成的。"** 以下说清楚哪部分复用、哪部分自研、自研的有多少。

### 8.0 软件栈"从零开始"？—— 一张图说清楚

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     整个推理软件栈的构成                                   │
│                                                                         │
│  ┌─────────────────────────────────────────┐   ┌─────────────────────┐ │
│  │          复用开源 (零开发)                │   │   自研 (~14 人月)    │ │
│  ├─────────────────────────────────────────┤   ├─────────────────────┤ │
│  │                                         │   │                     │ │
│  │  Tokenizer  ─── HuggingFace tokenizers  │   │  libfpga.so 驱动    │ │
│  │  HTTP Server ─ FastAPI + uvicorn        │   │  (VFIO/mmap/MMIO)   │ │
│  │  Sampling ─── PyTorch/numpy (CPU)       │   │                     │ │
│  │  JSON Mode ── LM Format Enforcer        │   │  FPGA 推理调度器     │ │
│  │  SSE ──────── sse-starlette             │   │  (session/priority/ │ │
│  │  Logprobs ─── scipy softmax             │   │   round-robin/prefill)│
│  │  Monitoring ── Prometheus + Grafana     │   │                     │ │
│  │  Logging ──── ELK / Loki                │   │  OpenAI API 适配层  │ │
│  │  Auth ─────── API Key / JWT             │   │  (protocol mapping) │ │
│  │  Rate Limit ── Redis token bucket       │   │                     │ │
│  │  Load Balance─ Nginx / Envoy            │   │  KV Cache 管理器    │ │
│  │  CI/CD ────── GitHub Actions / Argo     │   │  (地址映射/prefix)  │ │
│  │  LangChain ── 原生 HTTP client          │   │                     │ │
│  │  Dify ─────── 原生 HTTP client          │   │  权重加载器         │ │
│  │  Open WebUI ─ 原生 HTTP client          │   │  (PCIe DMA)        │ │
│  │                                         │   │                     │ │
│  │  → 这些都是成熟的、有文档的开源组件       │   │  → 这是唯一需要写的  │ │
│  │    一行代码不用改                         │   │    ~14,000 行 C/Python│ │
│  └─────────────────────────────────────────┘   └─────────────────────┘ │
│                                                                         │
│  自研比例: ~15% (14 人月 / 总计 ~90 人月等效开源代码)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**为什么自研部分这么少？**

```
GPU 软件栈为什么庞大:
  CUDA Driver → CUDA Runtime → cuBLAS/cuDNN → PyTorch → vLLM → API
  每一层都要处理: kernel launch, stream sync, device memory alloc,
  graph capture, NCCL comm, CUDA graph replay, ...
  
  vLLM 的核心复杂性来自:
    ① GPU 显存管理 (PagedAttention: 虚拟内存 → 物理 Block Table)
    ② CUDA Stream 调度 (多个 kernel 的异步执行与同步)
    ③ Continuous Batching (在 GPU kernel 执行中动态插拔请求)
    ④ 与 NCCL 的集成 (TP/PP 通信)

FPGA 软件栈为什么薄:
  FPGA Driver (VFIO) → libfpga.so → 调度器 → FastAPI → OpenAI API

  因为计算在硬件里:
    ① 没有 kernel launch → RTL 通过 FSM 自主调度
    ② 没有 stream sync → 流水线在 FPGA 内部闭环
    ③ 没有 device memory alloc → HBM 分区在编译时确定
    ④ 没有 NCCL → Ethernet RoCE v2 是标准协议，FPGA 硬核处理
    
  FPGA 的 "kernel" 是 RTL bitstream，加载一次后自动运行。
  FPGA 的 "CUDA graph" 是固化在硅片上的流水线 FSM。
  FPGA 的 "tensor core" 是 fp4 脉动阵列，通过 valid/ready 握手自动流动。
  
  → 软件只需做一件事: 把 token 写进寄存器，从寄存器读出结果。
```

### 8.0.1 质疑中的功能逐条回应

```
质疑声称 FPGA 方案缺少以下功能。逐条回应:

1. Continuous Batching:
   → 不需要。GPU 因为 B=1 利用率 2% 才必须做。
   → FPGA B=1 利用率为 50% — 不需要"抢救"算力。
   → 多用户用 Token Round-Robin (2.2ns 切换)，详见 §8.4.1。
   → 但诚实承认: 百并发公有云场景 FPGA 确实做不了。
     目标客户是私有部署 (1-20 并发)，这个量级不需要 CB。

2. Prefix Caching:
   → FPGA 方案有，而且是硬件级实现。
   → GPU 需要 hash + Block Table + CPU 参与
   → FPGA: prefix_hash 直接编码到 HBM 物理地址高位, 零 CPU
   → 详见 §8.4.1 第 2 项

3. Speculative Decoding:
   → 可行但不紧迫。SD 在 GPU 上是为了抢救 B=1 时的闲置算力。
   → FPGA B=1 已有 ~50% DSP 利用率，SD 的边际收益小。
   → 列为 v2 特性。

4. P/D 分离 (Prefill/Decode Disaggregation) — **已实现 (2026/05)**:
   -> FPGA 方案结构上就支持。Prefill 在 §4.8 已分析 —
   -> **2026 更新**: CPU prefill 已可用。Dual GNR/Turin 有效 10.5 TFLOPS,
      是 SPR 的 6x。P=128 chunk TTFT ~400ms, 覆盖 80% 商业场景。
   -> **三级 Prefill 架构 (已编码)**:
      Tier 1 — CPU (Intel AMX / AMD AVX-512): 短/中 prompt, TTFT 395-618ms
      Tier 2 — FPGA chunked prefill: 长 prompt, TTFT 85ms 首 chunk
      Tier 3 — GPU (可选): 极致低延迟, TTFT < 50ms
   -> **代码就绪**: c_ref/prefill/cpu_prefill.c (AMX GEMM),
      scripts/prefill/{coordinator,scheduler,vllm_prefill}.py,
      rtl/dsp/fp4_{prefill,gemm}_engine.sv,
      rtl/chip/kv_dma_bridge.sv
   -> 详见 §4.8.6 "2026 CPU Prefill 评估" 和 §14.E "Prefill 架构速查"

5. Profiler / Debugger:
   → FPGA 的可观测性远超 GPU profiler:
     ● Signal Tap: 抓取任意 RTL 内部信号 (GPU Tensor Core 内部不可见)
     ● 硬件性能计数器: 零开销 per-layer delay / DSP util / HBM BW
     ● Per-layer CRC32: 硬件级精度校验
   → 详见 §8.3.2 "可观测性"
   → ML 工程师不需要碰 Verilog — 性能计数器和 Signal Tap
     通过 Python API 暴露 (P0 交付项)
```

### 8.0.2 谁会用？使用者的三种角色

```
角色 A: ML 应用开发者 (90% 的使用者)
  → 对接 OpenAI Python SDK → 零学习成本
  → 用 LangChain/Dify/Open WebUI → 零迁移成本
  → 不用知道下面是 FPGA 还是 GPU
  → 和调用任何 OpenAI 兼容 API 完全一样

角色 B: 运维工程师 (10% 的使用者)
  → Prometheus + Grafana 看板 → 标准运维工具
  → libfpga.so 作为 systemd service 部署 → 标准 Linux 运维
  → 硬件更换: 拔旧卡、插新卡、加载 bitstream → <5 分钟
  → 不碰 Verilog、不碰 Quartus

角色 C: FPGA 开发者 (我们自己团队, 5 人)
  → 写 RTL、编译、上板 → 这就是我们做的事
  → 客户完全不需要这个角色
  → 类比: AWS 用户不需要知道 Nitro Hypervisor 的 RTL

软件栈的产出是一个 pip install 的 Python 包 + 一个 systemd service。
不是 "ML 工程师需要学 FPGA 开发" — 这是把开发者和使用者混淆了。
```

### 8.1 分层架构

```
┌────────────────────────────────────────┐
│  应用层: OpenAI REST API                │
│  /v1/chat/completions                  │
│  /v1/completions                       │
│  /v1/models                            │
│  → 任何 OpenAI client 零成本接入        │
├────────────────────────────────────────┤
│  推理服务层: 自研调度器                  │
│  ├── Tokenizer (HuggingFace tokenizer) │
│  ├── 采样器 (top-p, top-k, temperature)│
│  ├── 会话管理 (多 session 并发)         │
│  ├── KV Cache 分配器 (硬件寻址)         │
│  ├── 流式输出 (SSE)                     │
│  └── FastAPI HTTP Server               │
├────────────────────────────────────────┤
│  驱动层: libfpga.so (C 用户态库)        │
│  ├── FPGA 设备枚举 (VFIO)               │
│  ├── HBM 地址空间映射 (mmap)            │
│  ├── 推理命令下发 (MMIO write)          │
│  ├── 完成中断处理 (MSI-X)               │
│  └── DMA Buffer 管理                    │
├────────────────────────────────────────┤
│  硬件层: FPGA 算力卡                    │
│  ├── 固化 RTL: fp4 脉动阵列 + MLA +    │
│  │   KV Cache + MoE Router             │
│  └── 32 GB HBM2e                       │
└────────────────────────────────────────┘
```

### 8.2 兼容性矩阵

| 框架/工具 | 兼容方式 | 成本 |
|-----------|---------|------|
| OpenAI Python SDK | HTTP API 100% 兼容 | 零 |
| LangChain | HTTP API | 零 |
| LlamaIndex | HTTP API | 零 |
| Dify / FastGPT | HTTP API | 零 |
| Open WebUI | HTTP API | 零 |
| Continue.dev | HTTP API | 零 |
| vLLM | Fork (可选, 非必须) | ~3-6 人月 |
| HuggingFace Transformers | 不兼容 (无需兼容) | N/A |

### 8.3 部署运维特性

FPGA 集群在部署运维上具有三个 GPU 不具备的结构性优势：

#### 8.3.1 冷启动：毫秒级就绪

```
GPU 冷启动 (典型流程):
  服务器上电
    → GPU 初始化 (NVRM load, 10-15s)
    → CUDA Context 创建 (1-3s)
    → 模型权重 HBM→HBM 加载 (5-20s, 取决于磁盘/网络)
    → Kernel 预热 (JIT compile, 3-10s, 首次推理)
    → KV Cache Block 预分配 (1-2s)
  ────────────────────────────
  总计: ~20-50s (首次) / ~5-10s (热重启)

FPGA 冷启动 (本方案):
  服务器上电
    → FPGA 从 QSPI Flash 加载 bitstream (自配置, 无需 Host CPU)
    → 30 卡并行加载, 最后一块就绪即集群就绪
  ────────────────────────────
  Bitstream 加载: ~200ms (QSPI x4 @ 100MHz, Agilex 典型值)
  
  权重加载 (PCIe DMA):
    → 30 FPGA × 32 GB / (30 × 28 GB/s PCIe 有效带宽)
    → 32 GB / 28 GB/s ≈ 1.1s (并行, 无需顺序加载)
  
  总就绪时间 (Power-on to Ready):
    → Bitstream + Weight Load + 寄存器初始化
    → <500ms (bitstream 自配置期间权重可并行传输)
    
  对比:
  ┌──────────────┬──────────────┬──────────────┐
  │              │ GPU          │ FPGA         │
  ├──────────────┼──────────────┼──────────────┤
  │ 首次冷启动    │ 20-50s       │ <500ms       │
  │ 权重热切换    │ 5-20s        │ <500ms       │
  │ 节点重启      │ 10-30s       │ <500ms       │
  └──────────────┴──────────────┴──────────────┘
  
  工程意义:
    → 故障恢复: 备卡接管后 <500ms 恢复服务 (vs GPU 30s+)
    → 弹性伸缩: 快速上下线, 匹配波动负载
    → 频繁升级: 模型迭代时可滚动重启, 对用户几乎无感
```

#### 8.3.2 可观测性：硬件级信号级可视

```
GPU 可观测性:
  → nsys/ncu Profiler (采样模式, 有性能开销)
  → DCGM (GPU 级指标: 功耗/温度/利用率, 粗粒度)
  → CUPTI (API Trace, 软件层级)
  → Tensor Core 内部信号: 不可见
  → 流水线 stall 根因: 只能从外部推测

FPGA 可观测性 (本方案):
  
  ① Signal Tap 在线逻辑分析 (PCIe 通道):
    → 选择任意 RTL 内部信号, 通过 PCIe 实时抓取
    → 无需 JTAG/物理探针, 在线远程操作
    → 触发条件: 指定地址/数据模式/层号/异常事件
    → 抓取深度: 每个信号 128K samples (消耗少量 M20K)
    → 典型用例:
      · MoE Router 某层专家选择分布异常 → 抓取 Router logits
      · DSP 阵列输出某位置持续 NaN → 抓取 MAC 流水线中间值
      · KV Cache HBM 读延迟超预期 → 抓取 HBM 控制器状态机
    
  ② RTL 性能计数器 (Performance Monitor):
    → 固化在 RTL 中的硬件计数器 (零性能开销):
      · per-layer decode 延迟 (cycle 精度)
      · DSP 活跃周期 / 空闲周期 → 精确利用率
      · HBM 读/写有效带宽 (GB/s, 实时)
      · KV Cache hit/miss 计数 (per session)
      · PCIe TLP 发送/接收计数
      · Pipeline stall cycle (按 cause 分类: HBM wait / DSP busy / network wait)
    → 所有计数器通过 BAR0 MMIO 可读, 无需停止推理
    
  ③ Per-layer Activation 校验:
    → 每层输出可配置计算 CRC32 摘要
    → 与 GPU reference 逐层对比
    → 定位精度异常到特定层/特定专家
    → 用于调试和回归测试

  GPU vs FPGA 可观测性:
  ┌──────────────────────┬──────────────┬──────────────┐
  │                      │ GPU          │ FPGA         │
  ├──────────────────────┼──────────────┼──────────────┤
  │ 内部信号可见          │ ✗ (black box)│ ✓ (Signal Tap)│
  │ 性能开销              │ Profiler 有  │ 零 (硬件计数) │
  │ 时间精度              │ μs (CUPTI)   │ cycle (ns)    │
  │ 远程抓取              │ 有限         │ PCIe 在线     │
  │ Per-layer 校验        │ 需改代码     │ CRC32 硬件    │
  │ 根因定位速度          │ 小时级       │ 分钟级         │
  └──────────────────────┴──────────────┴──────────────┘
  
  工程意义:
    → 开发阶段: cycle 级 precision debug, 加速 RTL 验证
    → 线上运维: 异常检测 + 根因定位, 无需重启/停机
    → 持续优化: 精确性能瓶颈数据驱动迭代
```

#### 8.3.3 模型切换：秒级重配置

```
FPGA 的独特优势: 硬件可重编程 ≠ 每次都要重新编译

① HBM 分区双权重 (最快, 零重编程):
  → 32 GB HBM 划分为两区:
    Region A (24 GB): 当前模型权重
    Region B ( 8 GB): 预加载备用模型 (如 qwen3-235B fp4)
  → 切换方式: 修改 FPGA 寄存器中的 "Weight Base Pointer"
  → 切换延迟: 1 个时钟周期 = 2.2ns (450MHz)
  → 适合: 同架构不同权重 (DeepSeek V4 → V4.1 fine-tune)
  → 限制: 备模型需 ≤ 8 GB (压缩后可容纳 235B 级模型 fp4 权重)

② 权重热重载 (次快, 需 PCIe DMA):
  → HBM 保留 KV Cache 区域不动
  → 只覆盖 Weight 区域 (24 GB)
  → 30 卡并行: 24 GB / 28 GB/s ≈ 0.86s
  → 总切换延迟: <1s (含寄存器重新初始化 ~50ms)
  → 适合: 切换到不同架构模型 (如 DeepSeek → Qwen-MoE)

③ 部分重配置 (PR, 较慢但灵活):
  → 只修改特定 Pipeline Stage 的 RTL 逻辑
  → 其他 Stage 继续运行或保持权重
  → 切换时间: 数十 ms (取决于 reconfig region 大小)
  → 适合: 更新特定层的算法 (如升级 MLA 变体)

④ 完整 Bitstream 重载 (最慢, 极少使用):
  → 完全重写 FPGA 逻辑
  → 时间: ~200ms (QSPI) 或 ~100ms (PCIe x8 并行配置)
  → 适合: 切换到完全不同的模型架构 (如 Dense → MoE)

GPU vs FPGA 模型切换:
┌──────────────────────┬──────────────────┬──────────────────┐
│                      │ GPU              │ FPGA             │
├──────────────────────┼──────────────────┼──────────────────┤
│ 同架构换权重          │ 5-20s (HBM copy) │ <1s (PCIe DMA)  │
│ 不同架构模型          │ 10-30s (重载)    │ <1s (热重载)     │
│ 算法更新 (MLA 优化)   │ 需重新部署镜像   │ PR 数十ms        │
│ KV Cache 保留         │ 需显式保存/恢复  │ 不同 HBM 分区    │
│ 滚动升级 (多套)       │ 逐套重启 30s+    │ 逐套 <1s         │
└──────────────────────┴──────────────────┴──────────────────┘

工程意义:
  → A/B 测试: 两个模型版本间秒级切换, 对比效果
  → 灰度发布: 新模型上线可快速回滚
  → 多租户: 不同客户用不同模型版本, 时段切换
  → 滚动升级: 对用户几乎无感知的在线更新
```

### 8.4 推理服务功能矩阵

#### 8.4.1 核心推理功能覆盖

```
┌──────────────────────────┬──────────────────────┬──────────────────────┬──────────┐
│ 功能                       │ GPU (vLLM)           │ FPGA (本方案)         │ 评估      │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Continuous Batching       │ CUDA Stream 动态插入  │ Token Round-Robin    │ 场景不同  │
│                           │ B=1→8, 吞吐 7× 提升   │ 交替推理, 切换 2.2ns  │          │
│                           │ 百并发公有云场景      │ B=1~2 私有部署场景    │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Speculative Decoding      │ Draft + Target 并行   │ 可行但边际收益小      │ v2 考虑   │
│                           │ B=1 时加速 1.5-2×     │ B=1 DSP 已 ~50% util │          │
│                           │ 抢救 GPU 闲置算力     │ 不紧迫               │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Structured Output/JSON    │ 软件层 logit mask     │ 同 GPU, CPU 端处理    │ ✓ 纯软件  │
│                           │ LM Format Enforcer    │ 复用开源库            │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Function Calling/Tool Use│ 流式 JSON 解析        │ 同 GPU, CPU 端处理    │ ✓ 纯软件  │
│                           │ + 增量返回            │ + SSE chunk 解析      │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Request Prioritization    │ Per-request priority   │ Token 时隙按比例分配  │ ✓ 硬件级  │
│                           │ CUDA Stream 级调度    │ 高优:低优 = 2:1 等    │ 更简洁    │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Prompt Prefix Caching     │ 软件 hash + Block     │ 硬件地址编码 prefix   │ ★ FPGA优  │
│                           │ Table 管理, CPU 参与   │ 跨 session 零拷贝共享 │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Multi-LoRA 同时服务       │ LoRA 增量权重并行      │ 不支持同时, 但可快速   │ 架构限制  │
│                           │ 一张卡服务多 adapter   │ 切换 (<1s), 分时复用   │ 场景接受  │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Graceful Degradation      │ 软件 OOM 检测         │ 硬件 HBM 地址越界检测  │ ★ FPGA优  │
│                           │ try/catch 拒绝新请求  │ MSI-X 中断 + 调度器    │ 更安全    │
└──────────────────────────┴──────────────────────┴──────────────────────┴──────────┘
```

**关键差异说明：**

1. **Continuous Batching → Token Round-Robin**

```
GPU 为什么必须做 Continuous Batching:
  B=1 时 H100 Tensor Core 利用率 2%, 为抢救算力必须叠 B
  → 用户请求异步到达 → 需要动态加入/退出 batch
  → 这才是 vLLM 最复杂的调度逻辑

FPGA 为什么不需要:
  B=1 时 DSP 利用率 ~50% — 没有"必须叠 B 才能用的闲置算力"
  → 多 session 用 token 级交替: session A 1 token → session B 1 token → ...
  → 切换仅需修改 KV Cache base pointer (2.2ns)
  → 两个 session 各得约一半吞吐, 单 session 延迟翻倍但 token 到达翻倍
  
  这是架构差异, 不是功能缺失。FPGA 不需要靠 Continuous Batching 来"抢救"利用率。
  但需要承认: 百并发公有云场景, FPGA 确实做不到。
```

2. **Prefix Caching — FPGA 硬件级优势**

```
GPU:
  hash(prefix_tokens) → Lookup Block Table → 匹配 KV blocks → 共享引用
  软件管理, CPU 参与, 有内存碎片

FPGA:
  {prefix_hash, session_id, layer_id, seq_id} → HBM 物理地址
  → 硬件地址生成器直接编码 prefix_hash 到地址高位
  → 多个 session 共享同一个 prefix: 设置相同的 prefix_hash 寄存器
  → 零 CPU 参与, 零拷贝, 零内存碎片
```

3. **Multi-LoRA — 诚实承认架构限制**

```
GPU 可以一张卡同时服务 LoRA-A + LoRA-B + LoRA-C:
  → 基础权重共享, LoRA 增量独立
  → 推理: y = Wx + A₁B₁x (请求 1), y = Wx + A₂B₂x (请求 2)
  → 适合 SaaS 公有云

FPGA 不能"同时"服务多个 LoRA:
  → 权重固化在 HBM, 换 LoRA = 重载权重
  → 但可以快速切换 (<1s 热重载)
  → 私有部署场景: 每个客户有自己的集群, 只跑一个模型
  → 不需要同时服务多个 LoRA

如果真需要多 LoRA:
  → 部署 N 套集群, 各跑不同 LoRA (硬件隔离, 更安全)
  → 或分时复用: T₁ 跑 LoRA-A, T₂ 跑 LoRA-B (<1s 切换)
```

#### 8.4.2 API 参数完整度

```
P0 (MVP 必须):
  ✓ /v1/chat/completions        — 基础对话
  ✓ /v1/models                   — 模型列表
  ✓ stream: true                 — SSE 流式输出
  ✓ stop: [...]                  — 停止词列表
  ✓ temperature / top_p / top_k  — 采样参数 (CPU 端, 纯软件)
  ✓ max_tokens                   — 截断 (CPU 端)
  ✓ seed                         — 可复现推理 (CPU 端 set random seed)
  ✓ messages[].role              — system/user/assistant 角色

P1 (v1.0 应该实现):
  ○ logprobs                     — CPU 端 softmax → log, 纯软件
  ○ response_format              — JSON mode, 接 LM Format Enforcer
  ○ tool_choice / tools          — Function Calling, 流式 JSON chunk
  ○ presence_penalty             — 修改 logits, CPU 端
  ○ frequency_penalty            — 修改 logits, CPU 端
  ○ n: 2+                        — 多个候选, 可软件串行 (非并行)

P2 (v1.1+ 按需):
  ○ logit_bias                   — 特定 token 加权, CPU 端
  ○ user                         — 用户标识 (多租户跟踪)
  ○ response_format: json_schema — 复杂 JSON Schema 约束
```

#### 8.4.3 软件工作量拆分

```
原方案书 "软件系统开发 3人×10月 = 30人月" 已隐含覆盖, 此处明确拆分:

  推理引擎核心 (3 人月):
    → libfpga.so 驱动 (VFIO, mmap, MMIO, MSI-X)
    → 推理命令协议 (61 层流水线控制)
    → DMA Buffer 管理 + 权重加载器

  调度器 (4 人月):
    → Session Manager (创建/销毁/超时)
    → KV Cache 分配器 (硬件地址映射表管理)
    → Continuous Batching 调度器 (Token Round-Robin → 多 session 微批)
    → Prefix Cache 管理 (跨 session 共享)
    → Priority / SLA 分级

  API 服务层 (3 人月):
    → OpenAI REST API 完整兼容
    → SSE 流式输出
    → 所有 P0 + P1 参数处理
    → Tokenizer 集成 (HuggingFace tokenizer)

  生态适配 (2 人月):
    → Structured Output (集成 LM Format Enforcer / Outlines)
    → Function Calling (增量 JSON 解析 + tool call 协议)
    → LangChain / LlamaIndex / Dify 对接实测 + bug fix

  测试与稳定性 (2 人月):
    → 72h+ 连续运行稳定性测试
    → 多 session 并发压力测试
    → 异常注入 + 恢复测试
    → 与 GPU reference 端到端精度对比

  权重布局编译器扩展 (§5.3 + §4.6.1 + §4.8.x, 1.5 人月) ← 新增:
    → Hot Expert Replication 副本放置策略 (Zipf-based, 见 §4.6.1)
    → Pipeline Cloning 切分输出 (32 chip → N pipeline 权重映射, 见 §4.8.x)
    → 多副本路由表生成 (closest replica selection)

  小计: ~15.5 人月 (在原 30 人月预算内)

实施完成度 (截至 2026/05):
  ✓ 推理引擎核心 (scripts/vllm_serve/model_runner.py)
  ✓ 调度器 (scripts/vllm_serve/scheduler.py + run_serving.py)
    ✓ Continuous Batching
    ✓ KV Cache 扩容 (4096→22528 blocks/chip, §4.6.1 解法 D)
    ✓ 微批调度地板移除 (MIN_DECODE_BATCH 4→1, §4.6.1 解法 C)
    ✓ Pipeline Cloning 仿真 (--pipeline-clone N, §4.8.x)
  ✓ Hot Expert Replication (scripts/fpga_arch/expert_popularity.py + cluster.py)
  ✓ Chip 0 admission rate 解析模型 (scripts/fpga_arch/pipeline.py:chip0_admission_rate)
  ○ API 服务层 (P0 接口, 待实施)
  ○ 生态适配 (待实施)
  ○ 测试与稳定性 (待 RTL 上板)
  ○ 权重布局编译器 (待权重格式确定后实施)

仿真验证结果 (端到端):
  ✓ 6 倍吞吐提升验证 (1,000 → 5,800 tok/s, Agent 4 req/s, §4.6.1.3 实测)
  ✓ TTFT P95 改善验证 (1.15s → 0.54s, --pipeline-clone 2, §4.8.x.3 实测)
  ✓ 解法 A 收益曲线验证 (K_pipeline 25.4 → 23.1, 蒙特卡洛 + 解析模型吻合)
```

---

## 9. 开发路线图

```
Phase 1: 单卡验证 (Month 1-2)
  ├─ PCIe 5.0 x8 链路调通
  ├─ HBM2e 读写测试 (验证 >80% 理论带宽)
  ├─ fp4×fp8 矩阵乘 core RTL 仿真验证
  ├─ fp4 精度对比 (vs PyTorch reference)
  └─ 单层推理 Micro-benchmark

Phase 2: 单节点 8 卡 (Month 3-4)
  ├─ F-Tile 200GbE + 双 ToR 搭建
  ├─ RoCE v2 RDMA 片间通信 (FPGA F-Tile → ToR)
  ├─ 8 卡 TP All-Reduce + MoE Dispatch (全走 Ethernet)
  ├─ 8 卡跑通 15 层完整推理
  └─ 吞吐 benchmark (目标 >200 tok/s)

Phase 3: 双节点互联 (Month 5-6)
  ├─ 双 ToR MLAG + RoCE multipath 验证
  ├─ 跨节点 RoCE RDMA 通信 (FPGA F-Tile 直连)
  ├─ ToR 故障切换测试 (单台 ToR 下电 → 自动恢复)
  ├─ 跨节点 MoE Dispatch + Combine
  └─ 双节点跑通 30 层流水线

Phase 4: 四节点全集群 (Month 7-8)
  ├─ 4 节点 32 卡完整部署
  ├─ 全 61 层 + MTP 流水线
  ├─ 128K context 长序列测试
  ├─ 多 session 并发 (5 → 20)
  └─ 系统级 benchmark (目标 >500 tok/s)

Phase 5: 优化与生产化 (Month 9-10)
  ├─ 512K → 1M context 极限测试
  ├─ 热门专家 Multi-replica 优化
  ├─ 故障注入 + Failover 测试
  ├─ 功耗优化 + 散热验证
  └─ 推理服务层完 + OpenAI API 兼容认证
```

### 9.2 FPGA 验证策略与开发节奏

FPGA 开发不是 ASIC 的"仿真验证完 → 一次流片成功"流程，而是"模块仿真 + 快速上板迭代 + Signal Tap 在线抓信号"。本方案的设计规模决定了这个流程是高效的。

**9.2.1 设计规模：生产 vs 仿真 (2026/05 更新)**

```
Bring-Up (仿真, `ifndef FPGA_LPU_PRODUCTION`):
  fp4 脉动阵列 (1D):   ~8 DSP    (LANES=4)
  2D 脉动阵列:         ~16 DSP   (LANES=4, M_ROWS=2, test only)
  MLA 数据路径:         ~5 DSP
  MoE Router:           ~2 DSP
  单卡 DSP 用量:        ~30 DSP   (占 9,375 的 0.3%)
  Icarus 仿真:          ~30s 全编译
  → 用于快速功能验证

Production (`define FPGA_LPU_PRODUCTION`):
  2D 脉动阵列:          ~8192 DSP (LANES=128, M_ROWS=32, 87% 利用率)
  MLA 数据路径:          ~200 DSP (QKV 投影并行化)
  MoE Router:            ~50 DSP  (EXPERTS=384)
  RMSNorm:               ~30 DSP
  单卡 DSP 用量:         ~8500 DSP (占 9,375 的 91%)
  Quartus 全编译:        4-6h (cloud: c6i.16xlarge)
  增量编译:              1-2h
  → 生产 bitstream

通过 `ifdef FPGA_LPU_PRODUCTION 控制两套参数。
Bring-up 编译 30s 快速迭代, Production 编译 4-6h 生成 bitstream。
```

**9.2.2 模块级验证：并行推进，不跑全系统仿真**

```
评审推导: 100 token × 10ms / 2ns = 5×10^8 cycle → 仿真 16 年
这个推导假设 "必须跑完整 61 层 × 30 卡 × 100 token 的全系统仿真"。

实际策略: 6 个模块独立仿真，并行推进:

┌────────────────────┬──────────────────┬──────────────────┬──────────┐
│ 模块                │ 仿真规模           │ 仿真速度          │ 验证方法  │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ① fp4 脉动阵列      │ ~1000 cycle/次   │ Verilator ~100KHz │ 逐 bit   │
│                    │ (1 次 MAC)        │ → 10ms wall time  │ 对比 Python│
│                    │                  │                  │ reference │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ② MLA Pipeline     │ ~5000 cycle/次   │ Verilator ~50KHz  │ pattern   │
│                    │ (1 Attn 层)       │ → 100ms/pattern  │ 遍历      │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ③ MoE Router       │ ~200 cycle/次    │ 即时              │ BRAM 查表 │
│                    │ (纯组合逻辑)      │                  │ 功能验证  │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ④ KV Cache 地址生成│ ~50 cycle/次     │ 形式化验证        │ SVA +     │
│                    │ (纯组合逻辑)      │ (Jasper/SymbiYosys)│ 数学证明  │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ⑤ RoCE 协议栈      │ ~10,000 cycle/次 │ ~10KHz           │ loopback  │
│                    │ (1 RDMA 事务)     │                  │ 模式      │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ⑥ 流水线控制 (FSM) │ N/A (形式化)     │ TLA+ / SVA       │ 死锁证明  │
│                    │                  │ 不变量验证        │          │
└────────────────────┴──────────────────┴──────────────────┴──────────┘

仿真策略总结:
  → 不是 "仿真全系统 100 token" → 是 "6 个模块各仿真各自的 corner case"
  → 每个模块仿真规模 < 10,000 cycle → 秒级 wall time → 即时反馈
  → 模块间接口用标准 valid/ready 握手 → 集成时不会出现协议不匹配
```

**9.2.3 上板迭代：FPGA 的真正开发节奏**

```
ASIC 流程 (H100/B200):
  写 RTL → 仿真 → 发现 bug → 改 RTL → 等 tapeout (6-18 月)
  → 迭代周期 = 年

GPU CUDA 流程:
  写 kernel → 编译 (秒) → 跑 → nsys profile → 发现瓶颈
  → 迭代周期 = 分钟

FPGA RTL 流程 (本方案):
  写 RTL → Quartus 编译 → 上板 → Signal Tap 抓信号 → 发现 bug
  → 增量编译 20-40min → 上板 → 验证 fix
  → 迭代周期 = 30min-2h

关键工具: Signal Tap 在线逻辑分析仪
  → 选择任意 RTL 内部信号
  → 设置触发条件 (如 "Layer 35 Expert 73 被选中 且 DSP 输出 bit[15] != expected")
  → 硬件自动抓取 128K samples → PCIe 实时回传到 Host
  → 在 Quartus GUI 看波形 (如同仿真波形)
  → 无需外接逻辑分析仪, 无需 JTAG 探针

上板迭代 vs 仿真的本质区别:
  仿真 1 秒的系统行为需要 N 小时 wall time
  上板跑 1 秒只需要 1 秒 → 可以跑百万 token 的压测
  → 验证吞吐/延迟/功耗/72h 稳定性 → 只能上板, 不能仿真
```

**9.2.4 开发节奏与 10 个月计划的匹配**

```
5 人 × 10 月 = 50 人月:

  写 RTL:          15 人月 (30%)
  仿真验证:         8 人月 (16%)  ← 模块级, 秒级反馈
  上板调试:        12 人月 (24%)  ← 200+ 迭代
  系统集成+性能:    10 人月 (20%)
  余量:             5 人月 (10%)  ← 风险缓冲

每个 RTL 工程师平均每天 1-2 个 "改代码 → 增量编译 → 上板验证" 循环
  → 10 个月 ≈ 200 工作日 → 200-400 个迭代
  → 加 Signal Tap 减少 "猜 bug" 的时间
  → 与软件开发相比迭代慢 10-100×, 但对 FPGA 是现实和高效的

Phase 1 (Month 1-2): 单卡, 最重要的阶段
  → 所有 6 个模块在单卡上验证通过
  → 迭代最快 (无跨卡协调, 1 人 1 卡独占)
  → 如果 fp4 DSP 精度/HBM 带宽不达标 → 立即止损 (Go/No-Go #1)

Phase 2-4: 扩展卡数
  → 每张卡的 RTL 完全相同 (仅寄存器参数不同)
  → 2 卡验证通过 = 30 卡验证通过 (数据路径层面)
  → 30 卡验证重点: 通信稳定性 (72h 连续运行) + 性能调优
```

**9.2.5 分布式同步的形式化保证**

```
死锁风险: 61 层流水线 + 30 卡 All-to-All + Ring All-Reduce

保证方法 (三层):

  ① 架构层面:
     → Pipeline 单向 (Layer 0→1→...→60), 无反向依赖
     → MoE Dispatch 单向 (发送方→接收方), 无循环
     → All-Reduce Ring 有明确的 step 序号, 不会循环等待

  ② 协议层面 (Credit-based 流控):
     → 每个 RoCE QP 独立 credit 计数
     → 发送方只有收到 credit 才发送
     → 接收方 HBM 写端口通过仲裁器保证不冲突
     → SVA 不变量:
       "credit_count >= 0" (永不为负)
       "hbm_write_port_owner 是独热的" (不会同时 2 个请求)
       "每个 RDMA Send 最终收到 ACK OR 超时"

  ③ 硬件超时兜底:
     → 每个 RoCE 事务有硬件超时计数器 (可配置, 默认 1ms)
     → 超时 → MSI-X 中断 → 调度器重试
     → 不会出现 "永久等待"

  死锁验证在 Phase 2 (2-4 卡) 就已经全覆盖, 不需要等到 30 卡。
  因为通信协议和卡数无关 — 4 卡 Ring 和 30 卡 Ring 是同一套协议。
```

**9.2.6 ASIC vs FPGA vs GPU 验证对比**

```
┌────────────────────┬──────────────┬──────────────┬──────────────┐
│                    │ GPU ASIC      │ FPGA RTL      │ GPU CUDA     │
│                    │ (H100/B200)  │ (本方案)       │ (kernel)     │
├────────────────────┼──────────────┼──────────────┼──────────────┤
│ 开发周期            │ 2-3 年        │ 10 月         │ 周-月        │
│ 团队规模            │ 300-500 人    │ 5 人           │ 1-3 人       │
│ Tapeout 次数        │ 2-3 次        │ 0 次           │ 0 次         │
│ Bug 修复周期        │ 6-18 月       │ 30min-2h       │ 分钟         │
│ 上板迭代次数        │ <10 (pre-sil) │ 200+ 次         │ 数千次       │
│ 仿真覆盖            │ 全芯片        │ 模块级          │ N/A (无硬件) │
│ 在线抓内部信号      │ 极少 (metal fix)│ ✓ Signal Tap  │ N/A          │
│ 量产修改成本        │ $10M+ (metal) │ ¥0 (重配)      │ ¥0 (重编译) │
└────────────────────┴──────────────┴──────────────┴──────────────┘

关键结论:
  FPGA 开发的 "慢" 是相对于软件的, 不是相对于 ASIC 的。
  相对于 GPU ASIC 的 2-3 年周期 + 不可修复的硬件 bug,
  FPGA 的 200+ 上板迭代 + Signal Tap 在线调试已经是 "快" 的了。
```

### 9.3 质疑 A 回应：开发板实证验证计划

> **质疑 A**: "整个方案是纯纸面分析。fp4 精度没有实验数据，HBM 实际带宽利用率没有实测，单卡 end-to-end latency 没有。100 页文档建立在假设链上。"

**回答：对。这就是为什么 Phase 1 的首要任务不是写更多文档，而是买开发板、上板跑实验。以下是从质疑 A 直接推导出的实证计划——每一个实验都对应一个具体假设。**

#### 9.3.1 开发板选型

```
目标芯片: Intel Agilex 7 M AGFB027 (2×HBM2e, 32 GB, 9,375 DSP)

可选开发板:

┌─────────────────────┬──────────────────────┬──────────────────────┐
│                      │ Intel Agilex 7 M      │ BittWare IA-840F     │
│                      │ Dev Kit (DK-SI-AGM027)│                      │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ 芯片                  │ AGFB027 (目标芯片)    │ AGFB027 (同一芯片)    │
│ HBM2e                │ 32 GB                  │ 32 GB                │
│ 接口                  │ PCIe 5.0 x16, QSFP-DD │ PCIe 5.0 x16, QSFP  │
│ 内存                  │ DDR4 (控制用)          │ DDR4 + 可选 HBM 扩展 │
│ F-Tile 200GbE        │ ✓ (QSFP-DD)             │ ✓ (QSFP28 可能)      │
│ 价格 (预估)           │ ~$8,000-12,000         │ ~$10,000-15,000      │
│ 交期                   │ ~4-8 周 (正常库存)     │ ~6-12 周             │
│ 配套软件               │ Quartus Prime Pro      │ Quartus + BittWare   │
│                       │ + 参考设计 + BSP       │ BSP + Board Mgmt     │
│ 采购渠道               │ Intel/Altera 官方      │ BittWare 直接或代理  │
│ 推荐                    │ ★★★ (官方, 参考设计全) │ ★★ (可能需要额外 BSP) │
└─────────────────────┴──────────────────────┴──────────────────────┘

推荐: Intel DK-DEV-AGM039EA × 1 (AGM 039-F 直接验证, 12,300 DSP, HBM2e 32 GB)
      总预算: ~$8-12K 硬件 + Quartus Pro License ~$4K/年

> **2026/05 更新**: 原方案推荐 DK-SI-AGM027 (AGFB027, 9,375 DSP), 现已确认实际可采购
> DK-DEV-AGM039EA (AGMF039R47A, 12,300 DSP, HBM2e 32 GB)。芯片与量产方案一致,
> 不需要降级验证。详见 docs/bringup_strategy.md 和 docs/bringup_checklist.md。
```

#### 9.3.2 三大关键实验

**实验 1: fp4 DSP MAC 精度验证 (最高优先级)**

```
假设: fp4×fp8 乘法 + FP32 累加 → 整层输出与 PyTorch BF16 reference
     的 per-token 差异 < 2%

实验设计:
  目标: 验证 "fp4 精度足够" 这个最关键假设

  步骤 1 — Python 建模 (1 周):
    ● 从 DeepSeek V4 Pro 公开权重中提取 1 层完整参数
    ● 用 PyTorch 实现 fp4 量化模拟 (权重 fp4→fp8 解压 + fp8×fp8 MAC + FP32 累加)
    ● 与 BF16 baseline 逐 token 对比，输出 per-token cosine similarity
    ● 这是 "纯软件" 验证，不需要 FPGA，先确认数值上没有原则性问题

  步骤 2 — FPGA RTL 实现 (3 周):
    ● 实现最小化 fp4 脉动阵列: 1 个 128×128 systolic array
    ● DSP 配置为 fp4×fp8 → FP32 累加模式
    ● 权重从 HBM 或片上 SRAM 加载
    ● 跑 1 个 MLP 层的 GEMM (非完整 Transformer，先做最小的可验证单元)

  步骤 3 — 逐 bit 对比 (1 周):
    ● Python fp4 simulation → 生成 golden output (逐元素)
    ● FPGA 上板跑相同 GEMM → Signal Tap 抓内部 DSP 输出链
    ● 逐 bit 对比: DSP output[31:0] vs Python FP32 output
    ● 差异应 ≤ 1 ULP (unit in last place) — 因为两者都是 fp4×fp8 + FP32 acc

  步骤 4 — 完整 1 层对比 (1 周):
    ● 扩展到完整 1 层 Transformer block (Q/K/V/O + Expert FFN + Router)
    ● 上板跑 1000 token，每个 token 记录最终 activation
    ● 与 PyTorch reference 对比 per-layer output
    ● 统计: max diff, mean diff, diff histogram

  判定标准:
    ✓ 成功: per-token cosine similarity ≥ 0.995 (等价于 PPL 退化 <1%)
    △ 可接受: cosine similarity 0.98-0.995，需分析差异来源
    ✗ 止损: cosine similarity < 0.98 或某层差异系统性 >5%
          → 触发 Go/No-Go #2，启动 fp8 备选方案

  实验风险:
    ● 如果 fp4 精度验证失败 → 整个方案的 "fp4 原生" 基础不成立
    ● 但这仍然是有价值的发现: 知道 fp4 在 MoE 推理中不可行，避免了在海市蜃楼上建楼
    ● 备选: 全 fp8 方案 (HBM 带宽翻倍，但仍可在 32GB 内装下 15 层/layer)
```

**实验 2: HBM 有效带宽实测 (第二优先级)**

```
假设: 在 MoE expert 随机加载 pattern 下，HBM 有效带宽 ≥ 550 GB/s
      (理论 920 GB/s 的 60%)

实验设计:
  目标: 验证 "HBM 带宽不会成为瓶颈" 这个假设

  步骤 1 — 理论上限测试 (3 天):
    ● 顺序读取 1GB 连续块 → 测量纯顺序带宽
    ● 预期: ≥ 800 GB/s (接近理论值 920)
    ● 目的: 确认 HBM controller 和 PHY 工作正常

  步骤 2 — MoE expert 模拟 (1 周):
    ● 在 HBM 中放置 12 个 "expert" 块，每块 33 MB
    ● 按 power-law 分布随机选择 expert 块 (α=1.2, 模拟真实 Router 分布)
    ● 测量有效带宽: total_data_read / total_time
    ● 同时监测: HBM controller 的 bank conflict 计数
    ● 变体: 1 expert, 2 experts, 6 experts (模拟 0-hit, 1-hit, 2-hit)

  步骤 3 — 双缓冲流水线测试 (1 周):
    ● 实现权重双缓冲: buffer_A 在计算中, buffer_B 从 HBM 预取
    ● 测量: HBM 加载时间 vs DSP 计算时间的重叠率
    ● 目标: 重叠率 ≥ 80% (理想 100%，即 HBM 加载完全被 DSP 计算隐藏)

  判定标准:
    ✓ 成功: MoE random access 有效带宽 ≥ 550 GB/s (方案中的 "保守" 假设)
            双缓冲重叠率 ≥ 80%
    △ 可接受: 有效带宽 400-550 GB/s (吞吐降 20-30%，但仍可接受)
    ✗ 止损: 有效带宽 < 400 GB/s 或 bank conflict 导致带宽 < 40%
            → 触发 Go/No-Go #3，需要重新评估 32 卡集群的 HBM 约束
            → 如果 1-hit 层成为绝对瓶颈，考虑减少每卡层数 or 增加卡数

  Bank Conflict 缓解 (如果实测不达标):
    ● HBM2e 有 32 个 pseudo-channel，每 pseudo-channel 有独立的 bank
    ● Expert 权重布局: 按 pseudo-channel 交错存储
    ● 每个 expert 33MB → 每个 pseudo-channel 约 1MB
    ● 加载 1 个 expert: 32 pseudo-channels 并发读 → 理论 920 GB/s
    ● 问题只会在访问 pattern 引起 bank conflict 时出现
    ● 如果出现: 调整 expert 在 HBM 中的物理布局 (关键优势: 这是权重文件生成工具的事情，不需要改 RTL)
```

**实验 3: 单层端到端延迟实测 (第三优先级)**

```
假设: 单层加权平均延迟 ≈ 10 μs (方案 §4.4.1.4 的计算)

实验设计:
  目标: 验证单 layer 推理延迟的纸面估算

  方法:
    ● 跑完整 1 层 Transformer block 推理 (基于实验 1 的 RTL)
    ● 测量 3 种情形的实际延迟:
      - 0 expert hit (权重在 SRAM): 目标 ≤ 5 μs
      - 1 expert hit (加载 1 个 33MB expert + Router): 目标 ≤ 40 μs
      - 2 expert hits (加载 2 个 33MB expert): 目标 ≤ 75 μs
    ● 用 Signal Tap 测量 HBM 忙 vs DSP 忙的时间占比
    ● 跑 10,000 token 统计延迟分布 (长尾 expert 的影响)

  判定标准:
    ✓ 成功: 加权平均延迟 ≤ 15 μs (方案估算 10 μs 的 1.5× 容差)
            HBM stall 占比 < 50%
    △ 可接受: 加权平均延迟 15-25 μs (吞吐降 < 50%)
    ✗ 止损: 加权平均延迟 > 25 μs (意味着 30 卡集群吞吐 < 400 tok/s，
            对标云 GPU 的竞争劣势过大)
```

#### 9.3.3 实验边界：哪些问题可以上板验证，哪些不能

```
单卡可以验证的 (这 3 个实验覆盖):
  ✓ fp4 精度 (完整流水线，不需要多卡)
  ✓ HBM 有效带宽 (单卡存储系统独立)
  ✓ 单层延迟 (完整 layer pipeline)

单卡不能验证的 (需要多卡，不属于 Phase 1):
  ✗ 跨卡 MoE dispatch 延迟 (需要 ≥2 卡 + 交换机)
  ✗ 全 61 层流水线端到端延迟 (需要 4 节点 32 卡)
  ✗ 72h 连续运行稳定性 (需要完整集群)
  → 这些在 Phase 2 (2-4 卡) 验证

为什么 3 个实验的先后顺序重要:
  fp4 精度 (实验 1) 是整个方案的价值基石 → 如果失败，后面不用做
  HBM 带宽 (实验 2) 决定系统可行性 → 如果失败，架构需要推倒重来
  单层延迟 (实验 3) 是性能预测的验证 → 如果失败，TCO 需要重算
```

#### 9.3.4 实验完成后的决策路径

```
实验 1 (fp4 精度):
  ├─ 通过 → 继续实验 2
  └─ 失败 → 评估 fp8 备选 (权重 ×2 大小，仍可能可行)
            → 如果 fp8 也不行 → 项目止损

实验 2 (HBM 带宽):
  ├─ 通过 → 继续实验 3
  └─ 失败 → 评估 Bank Conflict 缓解方案
            → 如果缓解后仍不达标 → 重新设计 weight layout or 增加卡数

实验 3 (单层延迟):
  ├─ 通过 → Phase 1 完成，进入 Phase 2 (2 卡验证)
  └─ 失败 → 重新评估 TCO 和 $/百万token
            → 如果超标 >2× → 项目重评

全部 3 个实验通过:
  → 核心假设被实验验证
  → 方案从 "纸面分析" 升级为 "有实证支撑的工程方案"
  → 拿着 benchmark 数据去谈种子客户
  → 启动 Phase 2
```

---

## 10. 成本分析

### 10.1 硬件 BOM（8 卡 × 4 AGM 039，单台 4U）

> 核算原则: FPGA 加速卡自研自造；服务器机头外购市场价。无 ToR 交换机、无 RoCE IP、无 QSFP-DD 笼。

| 项目 | 规格 | 数量 | 单价 (¥) | 小计 (¥) |
|------|------|------|---------|---------|
| FPGA 芯片 | AGM 039-F 32GB HBM | 32 | 18,000 | 576,000 |
| 卡级物料 | PCB 14+层 / 4路VRM / 散热片 / 组装 | 8 | 48,000 | 384,000 |
| 服务器机头 | Inspur NF5688M7 / Lenovo SR670 V3 4U | 1 | 220,000 | 220,000 |
| 线缆/电源/机柜 | PDU + 42U 机架 | — | — | 30,000 |
| 备件 | 整卡备件 1 张 | 1 | 120,000 | 120,000 |
| **硬件合计** | | | | **~1,330,000** |

vs 旧方案 (¥2,415K, 32 卡分散方案): **节省 ¥1,085K (-45%)**

修正说明 (2026/05): 按实际询价 $2,500 ≈ ¥18,000/芯片 ($1=¥7.3)

```
芯片 AGM 039 ¥18,000/ea (~$2,500) 为实际询价 (与 scripts/fpga_arch/config.py 一致)。
  vs AGM 032 ¥21,600/ea (~$3,000): 价格相近, 但 DSP +31%, LE +19%, 性价比更优.
  且仅 Chip0 需要 R-Tile (Chip1/2/3 可省 R-Tile 成本 → 考虑 R31B 无 R-Tile 封装)

卡级物料 ¥48K/卡 — 多芯片卡 PCB 复杂度更高, 但无 QSFP-DD cage + 
  4 芯片共享散热片/组装, 单位芯片开销低于单芯片卡。

无 ToR Switch — 8 卡同在 4U 服务器背板, PCIe 5.0 P2P 直达。
无 RoCE v2 IP — PCIe P2P 自研 DMA 引擎替代 (RTL ~1.5 人月)。
无 QSFP-DD Cage / DAC 线缆 — 卡间不经过任何外部网络设备。

单台 4U 服务器 = 一套完整集群, 部署仅需一根电源线 + 一根网线 (BMC).
```

### 10.2 人力成本

| 角色 | 人数 | 周期 | 年薪 (¥) | 小计 (¥) |
|------|------|------|---------|---------|
| FPGA RTL 工程师 | 5 | 10 月 | 800,000 | 3,333,000 |
| 软件/系统工程师 | 3 | 10 月 | 600,000 | 1,500,000 |
| PCB 硬件工程师 | 1 | 5 月 | 600,000 | 250,000 |
| 测试/验证工程师 | 2 | 8 月 | 500,000 | 667,000 |
| **人力合计** | | | | **~5,750,000** |

### 10.3 硬件定价与毛利

> 定价原则: 硬件 BOM + 制造/测试/组装费用 + 毛利 = 客户售价。研发成本不进入硬件定价, 按 IP 资产单独摊薄。

```
                         单套 (原型)    10 套 (种子)   100 套 (批量)   10K 套 (规模)
                         ──────────    ──────────    ───────────    ───────────
硬件 BOM (§10.1 水平):
  AGM 039-F (×32)          ¥18,000       ¥18,000       ¥14,000        ¥10,000
  卡级物料 (×8)             ¥48,000       ¥45,000       ¥35,000        ¥22,000
  4U 服务器 (×1)            ¥220,000      ¥200,000      ¥170,000       ¥130,000
  线缆/PDU/备件             ¥218,000      ¥220,000      ¥180,000       ¥130,000
  ───────────────────────────────────────────────────────────────────────────
  硬件 BOM 小计             ~¥1.40M       ~¥1.36M       ~¥1.08M        ~¥0.76M

制造成本 (组装/测试/老化):
  整机组装测试               ¥80,000       ¥60,000       ¥40,000        ¥20,000
  48h 老化 + QC             ¥50,000       ¥40,000       ¥30,000        ¥15,000
  ───────────────────────────────────────────────────────────────────────────
  全成本 (BOM + 制造)        ~¥1.53M       ~¥1.46M       ~¥1.15M        ~¥0.79M

硬件毛利:
  毛利率                     35%           40%           45%            50%
  毛利金额                   ¥823K         ¥971K         ¥939K          ¥791K
  ───────────────────────────────────────────────────────────────────────────
  硬件售价 (不含 IP)         ~¥2.35M       ~¥2.43M       ~¥2.09M        ~¥1.58M
                            ≈ $326K       ≈ $337K       ≈ $290K        ≈ $220K
```

```
毛利率随规模变化逻辑:
  10 套:   35% — 小批量制造成本高, 客户议价空间大 (种子客户折扣)
  100 套:  45% — 制造效率提升, 品牌溢价开始体现
  10K 套:  50% — 规模效应全面释放, 但保持 IT 硬件行业标准毛利
  
  对照: NVIDIA H100 8卡服务器毛利率 ~65-70% (垄断溢价)
        Huawei Ascend 整机毛利率 ~40-50% (国产替代溢价)
        FPGA 方案 35-50% — 低于 GPU 垄断溢价, 但高于通用服务器 (15-20%)
```

### 10.4 研发投入 (IP 资产)

> 研发成本不作为硬件成本的一部分，而是形成 IP 资产，通过 License 费或 NRE 摊薄回收。

| 项目 | 金额 (¥) | 摊薄方式 |
|------|---------|---------|
| FPGA RTL IP (5人×10月) | 3,333,000 | 5 年直线摊薄, 按出货套数计 |
| 软件/驱动 IP (3人×10月) | 1,500,000 | |
| PCB 参考设计 (1人×5月) | 250,000 | |
| 测试验证 (2人×8月) | 667,000 | |
| 工具 / IP License / 设备 | 1,000,000 | (Quartus License, 仿真验证) |
| **研发 IP 合计** | **~6,750,000** | |

```
IP 摊薄模型 (5 年直线, 残值 = 0):

                      10 套 (种子)    100 套 (批量)    10K 套 (规模)
                      ───────────    ────────────    ─────────────
5 年总出货假设           15 套          150 套          15,000 套
  (含追加订单)

IP 摊薄/套              ¥450K          ¥45K            ¥0.45K
                       ≈ $62K         ≈ $6.2K         ≈ $62

IP 占硬件售价比例        16%            2.0%            0.03%
```

```
关键逻辑:
  ● 10 套原型: IP 摊薄很重 (¥450K/套), 但种子客户看重的是独家能力而非单价
    → 可以把部分 IP 费转为 NRE 一次性收取 (客户支付 ¥2-5M 获取定制权)
  ● 100 套: IP 摊薄 ¥45K/套, 占售价 2%, 几乎可忽略
  ● 10K 套: IP 摊薄 <¥500/套, 完全淹没在硬件毛利中
    → 此时 IP 已是纯利润引擎: ¥6.75M 投入 → 每年产生持续的 License 收入

  与 GPU 模式的本质差异:
    NVIDIA 的 R&D 投入 (数十亿美元) 已摊入每颗芯片的售价中 (H100 die cost ~$300, 售价 ~$30K)
    FPGA 的 RTL IP 是自有资产, 不需要支付给第三方 (无 RoCE IP, 无 PCIe Switch SDK)
    → 规模越大, IP 摊薄越薄, 硬件毛利越厚
```

### 10.5 三档客户交付价

```
┌──────────────────────┬──────────────┬──────────────┬──────────────┐
│                       │ 10 套 (原型)  │ 100 套 (批量) │ 10K 套 (规模) │
├──────────────────────┼──────────────┼──────────────┼──────────────┤
│ 硬件售价 (含毛利)       │ ¥2.80M       │ ¥2.20M       │ ¥1.47M       │
│ IP License (摊薄/套)   │ ¥450K        │ ¥45K         │ ¥0.5K        │
│ 年运维 (可选)          │ ¥300K        │ ¥250K        │ ¥200K        │
├──────────────────────┼──────────────┼──────────────┼──────────────┤
│ 客户首年 TCO (含 IP)   │ ~¥3.55M      │ ~¥2.50M      │ ~¥1.67M      │
│ 客户首年 TCO (IP 趸交) │ ~¥3.10M      │ ~¥2.45M      │ ~¥1.67M      │
└──────────────────────┴──────────────┴──────────────┴──────────────┘

  IP 趸交: 客户可选择一次性支付 IP License (¥2-5M) 替代按套摊薄,
           适合买断式部署 (如金融/政府私有化场景)。
```

### 10.6 对照与结论

```
对照 (单套, 含 3 年运维):

  本方案 FPGA (10K 套):  ~¥1.67M + 运维 ~¥0.6M = ~¥2.27M (≈ $312K)
  本方案 FPGA (100 套):  ~¥2.50M + 运维 ~¥0.75M = ~¥3.25M (≈ $447K)
  NVIDIA H100 8卡服务器:  ~¥1.5M (买不到, 受管制)
  Huawei Ascend 950PR:    ~¥1.2M (产能受限, 仅中国)

核心优势不变:
  ① 可买到 → 价格有意义的绝对前提
  ② 可全球部署 → 不受出口管制
  ③ 单台 4U = 一套集群 → 部署成本最低
  ④ 供应链多条来源 → 不被单一供应商卡脖子
  ⑤ IP 自有 → 无第三方 IP 税 (vs GPU 的 CUDA 生态锁定)
  ⑥ 硬件毛利健康 → 可持续商业模型 (35-50% vs 通用服务器 15-20%)
  ⑦ 架构优势: B=1 有效带宽 ~83× vs GPU (§11.A.2) → 不是"更便宜", 是架构不同
```

---


### 10.7 修正口径成本 (§4.6.1 + §4.8.x 优化后)

§10.1-§10.6 的成本数据基于 baseline 配置 (single-session 800 tok/s)，与 §11 表格一致。本节给出 §4.6.1/§4.8.x 软件优化全开后的修正口径，供客户实际部署测算时使用。

```
关键变化 (基于 §4.6.1.7 端到端验证, 18 配置矩阵):
  分子: 单套年度 TCO (修正 $2,500/芯片基准): ¥643-768K (10 套-10K 套)
        Pipeline Cloning ×2 后 HBM 权重区从 0.7→1.2 GB,
        仍在 32 GB 预算内, 不增加硬件成本。
  分母: 有效年产出取决于工作负载形态:

  ┌────────────┬──────────────────┬──────────────────┬──────────┐
  │ 工作负载    │ baseline TPS    │ 优化后 TPS       │ 倍数      │
  ├────────────┼──────────────────┼──────────────────┼──────────┤
  │ chat       │   792 tok/s      │   803 tok/s      │ ×1.01    │
  │ agent      │   961 tok/s      │ 5,782 tok/s      │ ×6.0     │
  │ burst      │ ~17,445 上限     │ ~17,445 上限     │ ×1.0 TPS │
  │            │ (TTFT 142s)      │ (TTFT 469ms)     │ TTFT ×304│
  └────────────┴──────────────────┴──────────────────┴──────────┘

  → chat 负载下优化无效, baseline 已满足
  → agent 负载是主要受益场景 (×5.9 TPS)
  → burst 负载受 DSP 物理峰值限制 (TPS 持平), 但 Pipeline Cloning 救场 TTFT

按 70% 年利用率换算 (假设 50% 时间是 agent + 50% 时间是 chat):
  baseline 有效 TPS = 0.5 × 961 + 0.5 × 782 = 871 tok/s
                    → 年产出 ~19B token
  优化后 有效 TPS  = 0.5 × 5,782 + 0.5 × 782 = 3,282 tok/s
                    → 年产出 ~72B token
  提升 ×3.0

┌──────────────────────────┬──────────┬──────────┬──────────┐
│                          │ 10 套     │ 100 套    │ 10K 套    │
├──────────────────────────┼──────────┼──────────┼──────────┤
│ baseline $/百万 token     │ $6.0     │ $5.0     │ $3.8     │
│ 修正口径 $/百万 token     │ $1.73    │ $1.30    │ $1.03    │
│ 改善幅度                  │ -71%     │ -74%     │ -73%     │
└──────────────────────────┴──────────┴──────────┴──────────┘

注: 修正口径数字基于混合负载假设 (50% agent + 50% chat)。
    纯 agent 负载下成本更低 (×5.9 全摊薄): ~$0.70 (10 套), ~$0.55 (100 套)
    纯 chat 负载下成本接近 baseline: ~$6 (优化无收益, 系统未饱和)

对标:
  Ascend 910C        ~$12-18/M  (单 session, 受 CANN 调度限制)
  NVIDIA H100 云租赁  ~$12-20/M  (但中国客户买不到)
  DeepSeek V4 Pro API $1.46/M    (混合负载: ¥0.1/¥12/¥24 缓存命中/未命中/输出)
  FPGA 10 套修正口径  $1.73/M    ← 略高于 API, 但供应链/数据主权胜出
  FPGA 100 套修正口径 $1.30/M   ← 已优于 API (架构带宽效率的直接结果)
  FPGA 10K 套修正口径 $1.03/M   ← 大幅优于 API (量产后规模效应叠加)
  ASIC 阶段 (§13)     $0.4-0.6/M (架构效率固化 + 制程成本崩塌)

关键论点:
  1. FPGA $/token 优势的根因不是"硬件更便宜", 而是有效带宽利用率 83× (见 §11.A.2)
     — 同样 $1 硬件, 更多有效带宽 → 更多 token → $/M 自然更低
  2. 数据主权 + 隐私合规场景 (金融/医疗/政府): 修正口径下 FPGA 已有竞争力
  3. 海外部署 (一带一路 / 出海): GPU 不可获取, 价格对比无意义

修正口径的前提:
  ✓ 客户启用 §4.6.1 优化组合 (默认推荐, 零硬件成本)
  ✓ 服务负载形态属于多 session (agent/copilot/API)
  ✗ 不适用于纯 batch=1 的单用户极致延迟场景
```

详细推导见 `docs/tco_per_million_tokens.md` §5.2。

---

