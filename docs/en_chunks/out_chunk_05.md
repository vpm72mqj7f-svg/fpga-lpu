## 7. Server Platform and Physical Form Factor

### 7.1 FPGA Accelerator Card Physical Specifications

```
4× AGM 039 Accelerator Card:
  ┌──────────────────────────────────────────────┐
  │  Form Factor:  FHFL Extended (Full-Height    │
  │         Full-Length Extended)                │
  │         111.15mm × 340mm, dual-slot width    │
  │         (standard FHFL 312mm + 28mm extension,│
  │          compatible with all 4U servers)      │
  │                                              │
  │  Chip:  4 × AGM 039-F (32GB HBM)             │
  │         56mm × 66mm R47A package             │
  │         single-row layout, 12mm chip spacing │
  │                                              │
  │  Interface:  PCIe 5.0 x16 CEM edge connector │
  │         (only Chip0 R-Tile exposed externally)│
  │         No QSFP-DD Cage                      │
  │         (inter-chip via on-board C2C SerDes) │
  │                                              │
  │  Power:  75W PCIe slot                       │
  │         + 12VHPWR 600W (single connector)    │
  │         = 675W total rated → 550W full load  │
  │         23% headroom                         │
  │                                              │
  │  Cooling:  4 discrete chips, independent     │
  │         heatsinks + vapor chambers           │
  │         4U server 120mm fan array            │
  │         front→rear forced air cooling        │
  │         airflow path length ~500mm           │
  │         junction temp < 85°C (Extended temp  │
  │         range)                               │
  │                                              │
  │  Management:  SMBus (I2C) ×4 (per-chip)      │
  │         Chip0 aggregation → BMC (IPMI std)   │
  │         temperature/power/link status/heartbeat│
  └──────────────────────────────────────────────┘

Card-level layout (~340mm × 111mm):

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

Chip0 positioned near PCIe edge connector (R-Tile direct), minimizing PCIe trace length.
Chip1/2/3 distributed evenly on board, inter-chip C2C trace < 200mm.
```

### 7.2 Server Platform

| Platform | Rating | PCIe Slots | PSU | Notes |
|------|------|---------|-----|------|
| **Inspur NF5688M7** | ★★★★★ | 8× x16, 4U | 2×3000W | Top domestic pick, 12VHPWR ready |
| **Lenovo SR670 V3** | ★★★★★ | 8× x16, 4U | 2×2600W | Supports extended FHFL |
| **Supermicro SYS-841GE-TNHR** | ★★★★ | 8× x16, 4U | 2×2600W | X13→X14→X15 |
| H3C R5500 G6 4U | ★★★★ | 8× x16 | 2×3000W | Domestic secondary option |

```
Single 4U server power budget:

  8 FPGA cards:  8 × 550W = 4,400W
  CPU + I/O:                 500W
  Fans (120mm ×8):           200W
  ─────────────────────────
  Total per node:         ~5,100W

PSU 2×3000W (1+1 redundant):
  5100W / 3000W = 170% → single PSU cannot carry full load
  → Requires load-balancing mode 2×2550W < 3000W, 18% headroom
  → Or 2+0 mode (dual PSU active), acceptable (common for GPU servers)

FPGA headroom vs GPU:
  H100 8-card: 8×700W + 500W = 6,100W
  FPGA 8-card:               = 5,100W
  → FPGA load is lighter, GPU-class PSU is sufficient
```

### 7.3 Cross-Generation Compatibility Guarantee

```
  2025:  4U server (Xeon SPR) + Agilex 7 M card,    Gen5 x16
  2027:  4U server (Xeon GNR) + same card,           Gen5 still usable
  2029:  4U server (Xeon NVL) + Agilex 10 M card,    Gen6 x16
         └─ old card in new server: downgrades to Gen5, works normally
         └─ new card in old server: downgrades to Gen5, works normally

  Key constraints:
    ✓ Standard PCIe CEM edge connector (no custom connector)
    ✓ Standard 12VHPWR power (no motherboard-custom power)
    ✓ FHFL Extended dimensions (already supported by 4U GPU servers)
    ✓ SMBus IPMI standard management (no proprietary BMC protocol)
    ✓ Linux standard VFIO driver (no closed-source SDK dependency)
```

### 7.4 Driver Model

```
Linux Kernel:
  ├── PCIe Subsystem (VFIO)
  ├── /dev/vfio/N   ← userspace direct FPGA control (1 device per card)
  ├── MSI-X interrupts ← inference done / error / heartbeat (per chip)
  ├── IOMMU         ← DMA address isolation
  ├── PCIe P2P      ← drivers/pci/p2p.c (native kernel support)
  └── No kernel module ← zero kernel API dependency, zero maintenance

Userspace:
  ├── libfpga.so    ← C library, VFIO mmap + P2P DMA
  ├── fpga_infer()  ← inference API
  ├── p2p_setup()    ← PCIe P2P BAR mapping (one-time init)
  └── Interfaces with inference service layer

P2P Configuration (one-time):
  echo 1 > /sys/bus/pci/devices/0000:01:00.0/p2pmem/enable   # Card A
  echo 1 > /sys/bus/pci/devices/0000:02:00.0/p2pmem/enable   # Card B
  → After this, PCIe MWr forwards directly between cards, bypassing CPU memory
  → v5.4+ kernel native support, no patches required
```

### 7.5 Power Analysis and Cooling Solution

**7.5.1 Per-Card Power Breakdown**

```
AGM 039 ×4 board-level power estimate (full-load inference):

  Per AGM 039 chip:
    DSP core (12,300 blocks, 450MHz, 50% util):   ~52W
    HBM2e (32GB, sustained read/write):            ~18W
    PCIe 5.0 (R-Tile, Chip0 only):                 ~8W
    C2C SerDes (F-Tile, 8 lane NRZ):               ~6W
    M20K/MLAB (75% util):                         ~10W
    Static power (10nm SuperFin, 039 larger die):  ~14W
    ─────────────────────────────────────────
    Per chip:                                      ~108W → round to 110W

  4-chip total: 4 × 110W = 440W

  PCB auxiliaries:
    VRM loss (4 independent SmartVID rails, ~12%): ~53W
    Clock/reset/JTAG/debug:                         ~5W
    12VHPWR connector loss:                         ~5W
    SMBus/I²C/BMC:                                 ~3W
  ─────────────────────────────────────────
  Per-card board-level full load:                  ~506W → round to 510W
  With 10% margin:                                 ~560W → round to 550W (nominal)

Comparison: H100 SXM TDP 700W — 4×FPGA single card 550W, lower power.
            Per TOPS power: FPGA 550W / 74 TFLOPS = 7.4 W/TFLOPS
                            H100 700W / 990 TFLOPS = 0.7 W/TFLOPS (FP8)
            But FPGA is fp4 native, GPU requires quantization → real perf/W gap is smaller
```

**7.5.2 System Wall Power**

```
┌──────────────────────────────┬──────────┬────────────────────┐
│ Component                     │ Qty      │ Power              │
├──────────────────────────────┼──────────┼────────────────────┤
│ 4×AGM 039 accelerator (full) │ 8        │ 8×550W = 4,400W    │
│ 4U server node (dual Xeon+)  │ 1        │ ~500W              │
│ Fans (120mm ×8, full speed)  │ 8        │ ~200W              │
│ PSU loss (80+ Titanium, ~6%) │ —        │ ~200W              │
├──────────────────────────────┼──────────┼────────────────────┤
│ Total Wall Power              │          │ ~5,300W ≈ 5.3kW    │
└──────────────────────────────┴──────────┴────────────────────┘

vs old design: 6.2kW (4 nodes + 2 Switches)
Savings: 15%, while compute +31%

vs H100 8-card: 5,600W (GPU) + 500W (CPU) = 6,100W
FPGA single system: 5,300W → 13% lower than GPU cluster

Per-rack (42U) density:
  5.3kW/system, each occupies 5U (4U server + 1U cable management)
  Capacity: 42U / 5U = 8 systems
  Power: 8 × 5.3kW = 42.4kW → exceeds air-cooling limit (15kW)
  Practical: 2-3 systems/rack (mixed deployment), or liquid cooling (volume production)
```

**7.5.3 Power Supply Verification**

```
Single 4U server (Inspur NF5688M7):
  PSU: 2× 3,000W (1+1 redundant, 80+ Titanium)
  Load: 5,100W

  1+1 mode: 5,100W / 3,000W = 170% → single PSU insufficient
  2+0 mode: 5,100W / 6,000W = 85% → dual PSU active, OK
    Single PSU failure → remaining overloaded → requires throttling (GPU servers use same approach)

  Or select 2× 3,500W PSU → 1+1 redundancy satisfied

FPGA load characteristics (vs GPU):
  GPU:  transient spike 1.5-2× TDP (hundreds of μs)
  FPGA: virtually no transient spike (DSP steady load)
  → Better PSU stability, simpler power grid design
```

**7.5.4 Cooling Solution**

```
4U server air cooling:

  Airflow: front 120mm fans ×6-8 → card array → rear exhaust
  Air volume: 120mm ×8 @ 3,000 RPM ≈ 250 CFM
  Temp rise: ΔT = 5,100W / (0.316 × 250 CFM) ≈ 64°C
            (inlet 25°C → outlet 89°C, on the high side)
  Verification needed: actual deployment may require lower inlet temp or higher airflow

  Card-level cooling:
    4 chips 56×66mm package discrete layout → heat sources spread out
    Per chip ~110W / (56×66mm²) ≈ 30 W/cm²
    vs H100: 700W / ~800mm² = 87 W/cm²
    → FPGA heat density only 1/3 of H100, better air-cooling feasibility

  Vapor Chamber:
    Per-chip independent VC + aluminum fins
    Thermal resistance: chip→air < 0.15°C/W → junction rise < 110W × 0.15 = 16.5°C
    Inlet 35°C → junction ~52°C, well below 85°C spec

  Comparison with GPU cooling:
    H100 8-card 6,100W → mostly requires liquid cooling (DGX H100 standard liquid cooling)
    FPGA 8-card 5,100W → high-volume air cooling is sufficient
    Savings: liquid cooling infrastructure ¥50-100K → ¥0
```

**7.5.5 Power Optimization Headroom**

```
Power reduction measures (Phase 3+):

  ① R-Tile width reduction: x16→x8 (P2P bandwidth sufficient)
     → saves ~4W/chip, 8 chips save 32W (only Chip0 has R-Tile)

  ② C2C SerDes downclock: 32G→16G NRZ (bandwidth far from saturated)
     → saves ~3W/chip × 32 chips = 96W

  ③ DSP dynamic frequency scaling: reduce DSP freq when HBM-bound
     → saves ~15W/chip × 32 chips = 480W

  ④ HBM low-power mode: standby during no-token layers
     → saves ~8W/chip × 32 chips = 256W

  Four measures combined: can drop to ~85W/chip, system ~3.7kW

  Prerequisite: RTL dynamic power management support, completed in Phase 3
```

---

## 8. Software Ecosystem and Inference Service Layer

> **Challenge C**: "The software stack is built from scratch — who will use it? There's no vLLM-grade serving framework, no profiler, no debugger. ML engineers won't touch Verilog. OpenAI API compatibility isn't just slapping on an endpoint — continuous batching, prefix caching, speculative decoding, P/D disaggregation — how many of these serving system features does the FPGA solution actually have?"

**Direct answer: The software stack is not "from scratch." It is "bypassing CUDA, and everything above is off-the-shelf."** Below we clarify what is reused, what is built in-house, and how much in-house work there actually is.

### 8.0 "Software Stack From Scratch"? — A Single Diagram Explains It

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Structure of the Entire Inference Software Stack           │
│                                                                         │
│  ┌─────────────────────────────────────────┐   ┌─────────────────────┐ │
│  │      Reused Open Source (zero dev)       │   │  In-House (~14 PM)  │ │
│  ├─────────────────────────────────────────┤   ├─────────────────────┤ │
│  │                                         │   │                     │ │
│  │  Tokenizer  ─── HuggingFace tokenizers  │   │  libfpga.so driver  │ │
│  │  HTTP Server ─ FastAPI + uvicorn        │   │  (VFIO/mmap/MMIO)   │ │
│  │  Sampling ─── PyTorch/numpy (CPU)       │   │                     │ │
│  │  JSON Mode ── LM Format Enforcer        │   │  FPGA inference     │ │
│  │  SSE ──────── sse-starlette             │   │  scheduler          │ │
│  │  Logprobs ─── scipy softmax             │   │  (session/priority/ │ │
│  │  Monitoring ── Prometheus + Grafana     │   │   round-robin/      │ │
│  │  Logging ──── ELK / Loki                │   │   prefill)          │ │
│  │  Auth ─────── API Key / JWT             │   │                     │ │
│  │  Rate Limit ── Redis token bucket       │   │  OpenAI API adapter │ │
│  │  Load Balance─ Nginx / Envoy            │   │  (protocol mapping) │ │
│  │  CI/CD ────── GitHub Actions / Argo     │   │                     │ │
│  │  LangChain ── native HTTP client        │   │  KV Cache manager   │ │
│  │  Dify ─────── native HTTP client        │   │  (address mapping/  │ │
│  │  Open WebUI ─ native HTTP client        │   │   prefix)           │ │
│  │                                         │   │                     │ │
│  │  → These are mature, well-documented    │   │  Weight loader      │ │
│  │    open-source components               │   │  (PCIe DMA)         │ │
│  │    Zero lines of code changed           │   │                     │ │
│  │                                         │   │  → This is the only │ │
│  │                                         │   │    code we write    │ │
│  │                                         │   │    ~14,000 lines    │ │
│  │                                         │   │    C/Python         │ │
│  └─────────────────────────────────────────┘   └─────────────────────┘ │
│                                                                         │
│  In-house ratio: ~15% (14 PM / ~90 PM equivalent of reused open source) │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why is the in-house portion so small?**

```
Why GPU software stacks are enormous:
  CUDA Driver → CUDA Runtime → cuBLAS/cuDNN → PyTorch → vLLM → API
  Every layer handles: kernel launch, stream sync, device memory alloc,
  graph capture, NCCL comm, CUDA graph replay, ...

  vLLM's core complexity comes from:
    ① GPU memory management (PagedAttention: virtual memory → physical Block Table)
    ② CUDA Stream scheduling (async execution and synchronization of multiple kernels)
    ③ Continuous Batching (dynamically inserting/removing requests mid GPU kernel execution)
    ④ Integration with NCCL (TP/PP communication)

Why FPGA software stacks are thin:
  FPGA Driver (VFIO) → libfpga.so → Scheduler → FastAPI → OpenAI API

  Because computation lives in hardware:
    ① No kernel launch → RTL self-schedules via FSM
    ② No stream sync → pipeline closed-loop inside FPGA
    ③ No device memory alloc → HBM partitioning determined at compile time
    ④ No NCCL → Ethernet RoCE v2 is a standard protocol, FPGA hard IP handles it

  The FPGA "kernel" is an RTL bitstream — load once, runs autonomously.
  The FPGA "CUDA graph" is a pipeline FSM burned into silicon.
  The FPGA "tensor core" is an fp4 systolic array, data flows automatically via valid/ready handshake.

  → Software only needs to do one thing: write tokens into registers, read results out of registers.
```

### 8.0.1 Point-by-Point Response to the Challenge's Feature Claims

```
The challenge claims the FPGA solution lacks the following features. Point-by-point response:

1. Continuous Batching:
   → Not needed. GPUs must do it because B=1 utilization is 2%.
   → FPGA B=1 utilization is 50% — there is no idle compute to "rescue."
   → Multi-user handled by Token Round-Robin (2.2ns switch), see §8.4.1.
   → But honest admission: hundreds-of-concurrency public cloud scenarios, FPGA genuinely cannot do.
     Target customers are private deployments (1-20 concurrent), this scale does not need CB.

2. Prefix Caching:
   → The FPGA solution has it, and it is hardware-level.
   → GPU requires hash + Block Table + CPU involvement
   → FPGA: prefix_hash encoded directly into HBM physical address high bits, zero CPU
   → See §8.4.1 item 2

3. Speculative Decoding:
   → Feasible but not urgent. SD exists on GPU to rescue idle compute at B=1.
   → FPGA B=1 already has ~50% DSP utilization, SD's marginal benefit is small.
   → Listed as a v2 feature.

4. P/D Disaggregation (Prefill/Decode Disaggregation) — **Implemented (2026/05)**:
   -> The FPGA solution is structurally supportive. Prefill analyzed in §4.8 —
   -> **2026 Update**: CPU prefill is now available. Dual GNR/Turin effective 10.5 TFLOPS,
      which is 6× SPR. P=128 chunk TTFT ~400ms, covering 80% of commercial scenarios.
   -> **Three-Tier Prefill Architecture (implemented in code)**:
      Tier 1 — CPU (Intel AMX / AMD AVX-512): short/medium prompts, TTFT 395-618ms
      Tier 2 — FPGA chunked prefill: long prompts, TTFT 85ms first chunk
      Tier 3 — GPU (optional): extreme low latency, TTFT < 50ms
   -> **Code ready**: c_ref/prefill/cpu_prefill.c (AMX GEMM),
      scripts/prefill/{coordinator,scheduler,vllm_prefill}.py,
      rtl/dsp/fp4_{prefill,gemm}_engine.sv,
      rtl/chip/kv_dma_bridge.sv
   -> See §4.8.6 "2026 CPU Prefill Evaluation" and §14.E "Prefill Architecture Quick Reference"

5. Profiler / Debugger:
   → FPGA observability far exceeds GPU profilers:
     ● Signal Tap: capture any internal RTL signal (GPU Tensor Core internals are invisible)
     ● Hardware performance counters: zero-overhead per-layer delay / DSP util / HBM BW
     ● Per-layer CRC32: hardware-grade accuracy verification
   → See §8.3.2 "Observability"
   → ML engineers do not need to touch Verilog — performance counters and Signal Tap
     are exposed through a Python API (P0 deliverable)
```

### 8.0.2 Who Will Use It? Three User Roles

```
Role A: ML Application Developer (90% of users)
  → Connects via OpenAI Python SDK → zero learning curve
  → Uses LangChain/Dify/Open WebUI → zero migration cost
  → No need to know whether it's FPGA or GPU underneath
  → Exactly the same as calling any OpenAI-compatible API

Role B: Operations Engineer (10% of users)
  → Prometheus + Grafana dashboards → standard ops tooling
  → libfpga.so deployed as systemd service → standard Linux ops
  → Hardware replacement: remove old card, insert new card, load bitstream → <5 minutes
  → No Verilog, no Quartus

Role C: FPGA Developer (our own team, 5 people)
  → Writes RTL, compiles, runs on board → this is what we do
  → Customers absolutely do not need this role
  → Analogy: AWS users don't need to know the Nitro Hypervisor's RTL

The software stack output is a pip-installable Python package + a systemd service.
It is not "ML engineers need to learn FPGA development" — that conflates developers with users.
```

### 8.1 Layered Architecture

```
┌────────────────────────────────────────┐
│  Application Layer: OpenAI REST API     │
│  /v1/chat/completions                  │
│  /v1/completions                       │
│  /v1/models                            │
│  → Any OpenAI client zero-cost access  │
├────────────────────────────────────────┤
│  Inference Service Layer: custom       │
│  scheduler                             │
│  ├── Tokenizer (HuggingFace tokenizer) │
│  ├── Sampler (top-p, top-k, temperature)│
│  ├── Session management (multi-session)│
│  ├── KV Cache allocator (HW addressing)│
│  ├── Streaming output (SSE)            │
│  └── FastAPI HTTP Server               │
├────────────────────────────────────────┤
│  Driver Layer: libfpga.so (C userspace)│
│  ├── FPGA device enumeration (VFIO)    │
│  ├── HBM address space mapping (mmap)  │
│  ├── Inference command dispatch (MMIO) │
│  ├── Completion interrupt handling     │
│  │   (MSI-X)                           │
│  └── DMA Buffer management             │
├────────────────────────────────────────┤
│  Hardware Layer: FPGA accelerator card │
│  ├── Hardened RTL: fp4 systolic array  │
│  │   + MLA + KV Cache + MoE Router     │
│  └── 32 GB HBM2e                       │
└────────────────────────────────────────┘
```

### 8.2 Compatibility Matrix

| Framework/Tool | Compatibility Approach | Cost |
|-----------|---------|------|
| OpenAI Python SDK | 100% HTTP API compatible | Zero |
| LangChain | HTTP API | Zero |
| LlamaIndex | HTTP API | Zero |
| Dify / FastGPT | HTTP API | Zero |
| Open WebUI | HTTP API | Zero |
| Continue.dev | HTTP API | Zero |
| vLLM | Fork (optional, not required) | ~3-6 PM |
| HuggingFace Transformers | Not compatible (no need) | N/A |

### 8.3 Deployment and Operations Characteristics

FPGA clusters possess three structural advantages over GPUs in deployment and operations:

#### 8.3.1 Cold Start: Millisecond Readiness

```
GPU cold start (typical flow):
  Server power-on
    → GPU initialization (NVRM load, 10-15s)
    → CUDA Context creation (1-3s)
    → Model weight HBM→HBM loading (5-20s, depending on disk/network)
    → Kernel warm-up (JIT compile, 3-10s, first inference)
    → KV Cache Block pre-allocation (1-2s)
  ────────────────────────────
  Total: ~20-50s (first) / ~5-10s (warm restart)

FPGA cold start (this design):
  Server power-on
    → FPGA loads bitstream from QSPI Flash (self-configuring, no Host CPU needed)
    → 30 cards load in parallel, last card ready = cluster ready
  ────────────────────────────
  Bitstream load: ~200ms (QSPI x4 @ 100MHz, Agilex typical)

  Weight loading (PCIe DMA):
    → 30 FPGA × 32 GB / (30 × 28 GB/s PCIe effective bandwidth)
    → 32 GB / 28 GB/s ≈ 1.1s (parallel, no sequential loading)

  Total readiness (Power-on to Ready):
    → Bitstream + Weight Load + register init
    → <500ms (weight transfer can overlap with bitstream self-config)

  Comparison:
  ┌──────────────┬──────────────┬──────────────┐
  │              │ GPU          │ FPGA         │
  ├──────────────┼──────────────┼──────────────┤
  │ First cold start │ 20-50s    │ <500ms       │
  │ Weight hot swap  │ 5-20s     │ <500ms       │
  │ Node restart     │ 10-30s    │ <500ms       │
  └──────────────┴──────────────┴──────────────┘

  Engineering significance:
    → Failure recovery: spare card takeover restores service in <500ms (vs GPU 30s+)
    → Elastic scaling: fast scale-up/down, matching fluctuating loads
    → Frequent upgrades: rolling restarts during model iterations, nearly invisible to users
```

#### 8.3.2 Observability: Hardware-Level Signal Visibility

```
GPU observability:
  → nsys/ncu Profiler (sampling mode, has performance overhead)
  → DCGM (GPU-level metrics: power/temp/utilization, coarse granularity)
  → CUPTI (API Trace, software level)
  → Tensor Core internal signals: invisible
  → Pipeline stall root cause: can only infer from external symptoms

FPGA observability (this design):

  ① Signal Tap online logic analyzer (PCIe channel):
    → Select any internal RTL signal, capture in real time via PCIe
    → No JTAG/physical probe needed, online remote operation
    → Trigger conditions: specified address/data pattern/layer number/exception event
    → Capture depth: 128K samples per signal (consumes small M20K)
    → Typical use cases:
      · MoE Router expert selection distribution anomaly at a given layer → capture Router logits
      · DSP array output persistently NaN at a position → capture MAC pipeline intermediates
      · KV Cache HBM read latency exceeding expectation → capture HBM controller state machine

  ② RTL Performance Counters (Performance Monitor):
    → Hardened hardware counters in RTL (zero performance overhead):
      · per-layer decode latency (cycle-accurate)
      · DSP active cycles / idle cycles → exact utilization
      · HBM read/write effective bandwidth (GB/s, real-time)
      · KV Cache hit/miss count (per session)
      · PCIe TLP transmit/receive count
      · Pipeline stall cycles (by cause: HBM wait / DSP busy / network wait)
    → All counters readable via BAR0 MMIO, no need to stop inference

  ③ Per-layer Activation Verification:
    → Configurable CRC32 digest computation on each layer's output
    → Compare layer-by-layer with GPU reference
    → Pinpoint accuracy anomalies to specific layer / specific expert
    → Used for debugging and regression testing

  GPU vs FPGA observability:
  ┌──────────────────────┬──────────────┬──────────────┐
  │                      │ GPU          │ FPGA         │
  ├──────────────────────┼──────────────┼──────────────┤
  │ Internal signal vis  │ ✗ (black box)│ ✓ (Signal Tap)│
  │ Performance overhead │ Profiler has │ Zero (HW ctrs)│
  │ Time precision       │ μs (CUPTI)   │ cycle (ns)    │
  │ Remote capture       │ Limited      │ PCIe online   │
  │ Per-layer verify     │ Code change  │ CRC32 hardware│
  │ Root cause speed     │ Hours        │ Minutes       │
  └──────────────────────┴──────────────┴──────────────┘

  Engineering significance:
    → Development phase: cycle-level precision debug, accelerates RTL verification
    → Production ops: anomaly detection + root cause location, no restart/downtime
    → Continuous optimization: precise performance bottleneck data drives iteration
```

#### 8.3.3 Model Switching: Second-Level Reconfiguration

```
FPGA's unique advantage: hardware reconfigurability ≠ recompilation every time

① HBM dual-weight partition (fastest, zero reprogramming):
  → 32 GB HBM partitioned into two regions:
    Region A (24 GB): current model weights
    Region B ( 8 GB): preloaded standby model (e.g., qwen3-235B fp4)
  → Switch method: modify FPGA register "Weight Base Pointer"
  → Switch latency: 1 clock cycle = 2.2ns (450MHz)
  → Suitable for: same architecture different weights (DeepSeek V4 → V4.1 fine-tune)
  → Limitation: standby model must fit ≤ 8 GB (compressed, fits 235B-class model fp4 weights)

② Weight hot reload (second fastest, requires PCIe DMA):
  → HBM retains KV Cache region untouched
  → Overwrite only Weight region (24 GB)
  → 30 cards parallel: 24 GB / 28 GB/s ≈ 0.86s
  → Total switch latency: <1s (including register re-init ~50ms)
  → Suitable for: switching to different-architecture models (e.g., DeepSeek → Qwen-MoE)

③ Partial Reconfiguration (PR, slower but flexible):
  → Modify only specific Pipeline Stage RTL logic
  → Other Stages continue running or retain weights
  → Switch time: tens of ms (depends on reconfig region size)
  → Suitable for: updating specific layer algorithms (e.g., upgrading MLA variant)

④ Full Bitstream Reload (slowest, rarely used):
  → Completely rewrite FPGA logic
  → Time: ~200ms (QSPI) or ~100ms (PCIe x8 parallel configuration)
  → Suitable for: switching to entirely different model architectures (e.g., Dense → MoE)

GPU vs FPGA model switching:
┌──────────────────────┬──────────────────┬──────────────────┐
│                      │ GPU              │ FPGA             │
├──────────────────────┼──────────────────┼──────────────────┤
│ Same arch, diff weights│ 5-20s (HBM copy)│ <1s (PCIe DMA)  │
│ Different arch model │ 10-30s (reload)   │ <1s (hot reload)│
│ Algorithm update (MLA)│ Redeploy image   │ PR tens of ms   │
│ KV Cache retention   │ Explicit save/rest│ Separate HBM part│
│ Rolling upgrade (multi)│ Per-node 30s+   │ Per-node <1s    │
└──────────────────────┴──────────────────┴──────────────────┘

Engineering significance:
  → A/B testing: second-level switch between two model versions for comparison
  → Canary releases: fast rollback of new model deployments
  → Multi-tenancy: different customers use different model versions, time-sliced switching
  → Rolling upgrades: nearly imperceptible online updates to users
```

### 8.4 Inference Service Feature Matrix

#### 8.4.1 Core Inference Feature Coverage

```
┌──────────────────────────┬──────────────────────┬──────────────────────┬──────────┐
│ Feature                   │ GPU (vLLM)           │ FPGA (this design)   │ Assessment│
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Continuous Batching       │ CUDA Stream dynamic  │ Token Round-Robin    │ Different│
│                           │ insert, B=1→8,       │ alternating inference│ scenarios│
│                           │ throughput 7× gain   │ switch 2.2ns         │          │
│                           │ 100-concurrency cloud│ B=1~2 private deploy │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Speculative Decoding      │ Draft + Target in    │ Feasible, marginal   │ v2 target│
│                           │ parallel, B=1 1.5-   │ benefit small, B=1   │          │
│                           │ 2× speedup, rescues  │ DSP already ~50% util│          │
│                           │ idle GPU compute     │ not urgent           │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Structured Output/JSON    │ Software logit mask   │ Same as GPU, CPU-side│ ✓ pure SW│
│                           │ LM Format Enforcer    │ reuse open-source lib│          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Function Calling/Tool Use│ Streaming JSON parse  │ Same as GPU, CPU-side│ ✓ pure SW│
│                           │ + incremental return  │ + SSE chunk parse    │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Request Prioritization    │ Per-request priority  │ Token time-slot      │ ✓ HW-level│
│                           │ CUDA Stream scheduling│ proportional alloc   │ simpler  │
│                           │                      │ high:low = 2:1 etc.  │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Prompt Prefix Caching     │ Software hash + Block │ Hardware address     │ ★ FPGA + │
│                           │ Table mgmt, CPU in    │ encoding prefix, zero│          │
│                           │ the loop              │ copy cross-session   │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Multi-LoRA concurrent     │ LoRA delta weights    │ No concurrent, but   │ Arch      │
│                           │ parallel, one card    │ fast switch (<1s),   │ limit OK │
│                           │ serves multi-adapters │ time-share            │          │
├──────────────────────────┼──────────────────────┼──────────────────────┼──────────┤
│ Graceful Degradation      │ Software OOM detection│ Hardware HBM address  │ ★ FPGA + │
│                           │ try/catch reject new  │ bounds check, MSI-X  │ safer    │
│                           │ requests              │ interrupt + scheduler │          │
└──────────────────────────┴──────────────────────┴──────────────────────┴──────────┘
```

**Key Differences Explained:**

1. **Continuous Batching → Token Round-Robin**

```
Why GPU must do Continuous Batching:
  At B=1, H100 Tensor Core utilization is 2% — must stack batch to rescue compute
  → User requests arrive asynchronously → need dynamic join/leave batch
  → This is vLLM's most complex scheduling logic

Why FPGA does not need it:
  At B=1, DSP utilization is ~50% — no "must stack batch to use idle compute"
  → Multiple sessions use token-level interleaving: session A 1 token → session B 1 token → ...
  → Switch only requires modifying KV Cache base pointer (2.2ns)
  → Two sessions each get ~half throughput, single-session latency doubles but token arrival doubles

  This is an architectural difference, not a missing feature. FPGA doesn't need Continuous Batching to "rescue" utilization.
  But must acknowledge: 100-concurrency public cloud scenarios, FPGA genuinely cannot do.
```

2. **Prefix Caching — FPGA Hardware-Level Advantage**

```
GPU:
  hash(prefix_tokens) → Lookup Block Table → match KV blocks → share references
  Software managed, CPU in the loop, memory fragmentation

FPGA:
  {prefix_hash, session_id, layer_id, seq_id} → HBM physical address
  → Hardware address generator directly encodes prefix_hash into address high bits
  → Multiple sessions sharing the same prefix: set the same prefix_hash register
  → Zero CPU involvement, zero copy, zero memory fragmentation
```

3. **Multi-LoRA — Honest Admission of Architectural Limitation**

```
GPU can serve LoRA-A + LoRA-B + LoRA-C concurrently on one card:
  → Base weights shared, LoRA deltas independent
  → Inference: y = Wx + A₁B₁x (request 1), y = Wx + A₂B₂x (request 2)
  → Suitable for SaaS public cloud

FPGA cannot serve multiple LoRAs "concurrently":
  → Weights burned into HBM, changing LoRA = reloading weights
  → But can switch quickly (<1s hot reload)
  → Private deployment scenario: each customer has their own cluster, runs only one model
  → No need to serve multiple LoRAs concurrently

If truly need multi-LoRA:
  → Deploy N sets of clusters, each running a different LoRA (hardware isolation, more secure)
  → Or time-share: T₁ runs LoRA-A, T₂ runs LoRA-B (<1s switch)
```

#### 8.4.2 API Parameter Completeness

```
P0 (MVP required):
  ✓ /v1/chat/completions        — basic chat
  ✓ /v1/models                   — model list
  ✓ stream: true                 — SSE streaming output
  ✓ stop: [...]                  — stop token list
  ✓ temperature / top_p / top_k  — sampling params (CPU-side, pure software)
  ✓ max_tokens                   — truncation (CPU-side)
  ✓ seed                         — reproducible inference (CPU-side set random seed)
  ✓ messages[].role              — system/user/assistant roles

P1 (v1.0 should implement):
  ○ logprobs                     — CPU-side softmax → log, pure software
  ○ response_format              — JSON mode, connects to LM Format Enforcer
  ○ tool_choice / tools          — Function Calling, streaming JSON chunks
  ○ presence_penalty             — modify logits, CPU-side
  ○ frequency_penalty            — modify logits, CPU-side
  ○ n: 2+                        — multiple candidates, software sequential (not parallel)

P2 (v1.1+ on demand):
  ○ logit_bias                   — specific token weighting, CPU-side
  ○ user                         — user identifier (multi-tenant tracking)
  ○ response_format: json_schema — complex JSON Schema constraints
```

#### 8.4.3 Software Work Breakdown

```
The original proposal's "software system development 3 people × 10 months = 30 PM" already implicitly covers this; here is the explicit breakdown:

  Inference Engine Core (3 PM):
    → libfpga.so driver (VFIO, mmap, MMIO, MSI-X)
    → Inference command protocol (61-layer pipeline control)
    → DMA Buffer management + Weight loader

  Scheduler (4 PM):
    → Session Manager (create/destroy/timeout)
    → KV Cache allocator (hardware address mapping table management)
    → Continuous Batching scheduler (Token Round-Robin → multi-session micro-batch)
    → Prefix Cache management (cross-session sharing)
    → Priority / SLA classification

  API Service Layer (3 PM):
    → Full OpenAI REST API compatibility
    → SSE streaming output
    → All P0 + P1 parameter handling
    → Tokenizer integration (HuggingFace tokenizer)

  Ecosystem Adaptation (2 PM):
    → Structured Output (integrate LM Format Enforcer / Outlines)
    → Function Calling (incremental JSON parse + tool call protocol)
    → LangChain / LlamaIndex / Dify integration testing + bug fixes

  Testing and Stability (2 PM):
    → 72h+ continuous run stability testing
    → Multi-session concurrency stress testing
    → Fault injection + recovery testing
    → End-to-end accuracy comparison against GPU reference

  Weight Layout Compiler Extension (§5.3 + §4.6.1 + §4.8.x, 1.5 PM) ← newly added:
    → Hot Expert Replication replica placement strategy (Zipf-based, see §4.6.1)
    → Pipeline Cloning split output (32 chip → N pipeline weight mapping, see §4.8.x)
    → Multi-replica routing table generation (closest replica selection)

  Subtotal: ~15.5 PM (within original 30 PM budget)

Implementation completeness (as of 2026/05):
  ✓ Inference engine core (scripts/vllm_serve/model_runner.py)
  ✓ Scheduler (scripts/vllm_serve/scheduler.py + run_serving.py)
    ✓ Continuous Batching
    ✓ KV Cache expansion (4096→22528 blocks/chip, §4.6.1 Solution D)
    ✓ Micro-batch floor removed (MIN_DECODE_BATCH 4→1, §4.6.1 Solution C)
    ✓ Pipeline Cloning simulation (--pipeline-clone N, §4.8.x)
  ✓ Hot Expert Replication (scripts/fpga_arch/expert_popularity.py + cluster.py)
  ✓ Chip 0 admission rate analytical model (scripts/fpga_arch/pipeline.py:chip0_admission_rate)
  ○ API service layer (P0 endpoints, pending implementation)
  ○ Ecosystem adaptation (pending implementation)
  ○ Testing and stability (pending RTL on-board)
  ○ Weight layout compiler (pending weight format finalization)

Simulation validation results (end-to-end):
  ✓ 6× throughput improvement verified (1,000 → 5,800 tok/s, Agent 4 req/s, §4.6.1.3 measured)
  ✓ TTFT P95 improvement verified (1.15s → 0.54s, --pipeline-clone 2, §4.8.x.3 measured)
  ✓ Solution A benefit curve verified (K_pipeline 25.4 → 23.1, Monte Carlo and analytical model agree)
```

---

## 9. Development Roadmap

```
Phase 1: Single-Card Verification (Month 1-2)
  ├─ PCIe 5.0 x8 link bring-up
  ├─ HBM2e read/write test (verify >80% theoretical bandwidth)
  ├─ fp4×fp8 matrix multiply core RTL simulation verification
  ├─ fp4 precision comparison (vs PyTorch reference)
  └─ Single-layer inference micro-benchmark

Phase 2: Single-Node 8-Card (Month 3-4)
  ├─ F-Tile 200GbE + dual ToR setup
  ├─ RoCE v2 RDMA inter-chip communication (FPGA F-Tile → ToR)
  ├─ 8-card TP All-Reduce + MoE Dispatch (all via Ethernet)
  ├─ 8-card full 15-layer inference
  └─ Throughput benchmark (target >200 tok/s)

Phase 3: Dual-Node Interconnect (Month 5-6)
  ├─ Dual ToR MLAG + RoCE multipath verification
  ├─ Cross-node RoCE RDMA communication (FPGA F-Tile direct)
  ├─ ToR failover test (single ToR power down → auto recovery)
  ├─ Cross-node MoE Dispatch + Combine
  └─ Dual-node full 30-layer pipeline

Phase 4: Four-Node Full Cluster (Month 7-8)
  ├─ 4-node 32-card full deployment
  ├─ Full 61-layer + MTP pipeline
  ├─ 128K context long-sequence test
  ├─ Multi-session concurrency (5 → 20)
  └─ System-level benchmark (target >500 tok/s)

Phase 5: Optimization and Production (Month 9-10)
  ├─ 512K → 1M context extreme test
  ├─ Hot Expert Multi-replica optimization
  ├─ Fault injection + Failover testing
  ├─ Power optimization + thermal verification
  └─ Inference service layer complete + OpenAI API compatibility certification
```

### 9.2 FPGA Verification Strategy and Development Cadence

FPGA development is not the ASIC flow of "simulate to completion → one tapeout success." It is "module simulation + fast on-board iteration + Signal Tap online signal capture." The design scale of this proposal makes this flow efficient.

**9.2.1 Design Scale: Production vs Simulation (2026/05 Update)**

```
Bring-Up (simulation, `ifndef FPGA_LPU_PRODUCTION`):
  fp4 systolic array (1D):   ~8 DSP    (LANES=4)
  2D systolic array:         ~16 DSP   (LANES=4, M_ROWS=2, test only)
  MLA datapath:               ~5 DSP
  MoE Router:                 ~2 DSP
  Single-card DSP usage:      ~30 DSP   (0.3% of 9,375)
  Icarus simulation:          ~30s full compile
  → Used for fast functional verification

Production (`define FPGA_LPU_PRODUCTION`):
  2D systolic array:          ~8192 DSP (LANES=128, M_ROWS=32, 87% utilization)
  MLA datapath:               ~200 DSP (QKV projection parallelization)
  MoE Router:                 ~50 DSP  (EXPERTS=384)
  RMSNorm:                    ~30 DSP
  Single-card DSP usage:      ~8500 DSP (91% of 9,375)
  Quartus full compile:       4-6h (cloud: c6i.16xlarge)
  Incremental compile:        1-2h
  → Production bitstream

Two parameter sets controlled via `ifdef FPGA_LPU_PRODUCTION.
Bring-up compiles in 30s for fast iteration, Production compiles in 4-6h for bitstream generation.
```

**9.2.2 Module-Level Verification: Parallel Progression, No Full-System Simulation**

```
Reviewer's deduction: 100 token × 10ms / 2ns = 5×10^8 cycles → 16 years of simulation
This assumes "must run full 61-layer × 30-card × 100-token system-level simulation."

Actual strategy: 6 modules independently simulated, in parallel:

┌────────────────────┬──────────────────┬──────────────────┬──────────┐
│ Module              │ Simulation Scale │ Simulation Speed │ Verification│
│                     │                  │                  │ Method     │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ① fp4 systolic     │ ~1000 cycles/op  │ Verilator ~100KHz│ bit-exact │
│                    │ (1 MAC)          │ → 10ms wall time │ vs Python  │
│                    │                  │                  │ reference  │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ② MLA Pipeline     │ ~5000 cycles/op  │ Verilator ~50KHz │ pattern   │
│                    │ (1 Attn layer)   │ → 100ms/pattern  │ sweep      │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ③ MoE Router       │ ~200 cycles/op   │ Instant           │ BRAM LUT  │
│                    │ (pure combinational)│                │ functional │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ④ KV Cache addr gen│ ~50 cycles/op    │ Formal verification│ SVA +     │
│                    │ (pure combinational)│ (Jasper/        │ math proof │
│                    │                  │ SymbiYosys)       │            │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ⑤ RoCE protocol    │ ~10,000 cycles/op│ ~10KHz           │ loopback   │
│                    │ (1 RDMA transaction)│                │ mode       │
├────────────────────┼──────────────────┼──────────────────┼──────────┤
│ ⑥ Pipeline ctrl    │ N/A (formal)     │ TLA+ / SVA        │ deadlock   │
│    (FSM)            │                  │ invariant check   │ proof      │
└────────────────────┴──────────────────┴──────────────────┴──────────┘

Simulation strategy summary:
  → Not "simulate full system 100 tokens" → but "6 modules each simulate their own corner cases"
  → Each module simulation scale < 10,000 cycles → seconds wall time → instant feedback
  → Inter-module interfaces use standard valid/ready handshake → no protocol mismatch during integration
```

**9.2.3 On-Board Iteration: The True FPGA Development Cadence**

```
ASIC flow (H100/B200):
  Write RTL → simulate → find bug → fix RTL → wait for tapeout (6-18 months)
  → Iteration cycle = years

GPU CUDA flow:
  Write kernel → compile (seconds) → run → nsys profile → find bottleneck
  → Iteration cycle = minutes

FPGA RTL flow (this design):
  Write RTL → Quartus compile → on-board → Signal Tap capture signals → find bug
  → Incremental compile 20-40min → on-board → verify fix
  → Iteration cycle = 30min-2h

Key tool: Signal Tap online logic analyzer
  → Select any internal RTL signal
  → Set trigger condition (e.g., "Layer 35 Expert 73 selected AND DSP output bit[15] != expected")
  → Hardware auto-captures 128K samples → PCIe real-time stream to Host
  → View waveforms in Quartus GUI (just like simulation waveforms)
  → No external logic analyzer needed, no JTAG probe

The essential difference between on-board iteration and simulation:
  Simulating 1 second of system behavior takes N hours wall time
  Running 1 second on board takes 1 second → can run million-token stress tests
  → Verify throughput/latency/power/72h stability → can only be done on board, not in simulation
```

**9.2.4 Development Cadence Matching the 10-Month Plan**

```
5 people × 10 months = 50 PM:

  RTL coding:      15 PM (30%)
  Simulation:       8 PM (16%)  ← module-level, seconds feedback
  On-board debug:  12 PM (24%)  ← 200+ iterations
  System integration+perf: 10 PM (20%)
  Buffer:             5 PM (10%)  ← risk buffer

Each RTL engineer averages 1-2 "modify code → incremental compile → on-board verify" cycles per day
  → 10 months ≈ 200 working days → 200-400 iterations
  → Plus Signal Tap reduces "guessing bug" time
  → 10-100× slower iteration than software development, but realistic and efficient for FPGA

Phase 1 (Month 1-2): Single card, the most critical phase
  → All 6 modules verified on a single card
  → Fastest iteration (no cross-card coordination, 1 person 1 card exclusive)
  → If fp4 DSP precision / HBM bandwidth does not meet spec → immediate stop-loss (Go/No-Go #1)

Phase 2-4: Scale up card count
  → Every card's RTL is identical (only register parameters differ)
  → 2-card verification pass = 30-card verification pass (data path level)
  → 30-card verification focus: communication stability (72h continuous run) + performance tuning
```

**9.2.5 Formal Guarantees for Distributed Synchronization**

```
Deadlock risk: 61-layer pipeline + 30-card All-to-All + Ring All-Reduce

Guarantee method (three layers):

  ① Architectural level:
     → Pipeline unidirectional (Layer 0→1→...→60), no reverse dependencies
     → MoE Dispatch unidirectional (sender→receiver), no cycles
     → All-Reduce Ring has explicit step number, no circular wait

  ② Protocol level (Credit-based flow control):
     → Each RoCE QP has independent credit count
     → Sender only sends upon receiving credit
     → Receiver HBM write port guaranteed conflict-free via arbiter
     → SVA invariants:
       "credit_count >= 0" (never negative)
       "hbm_write_port_owner is one-hot" (no simultaneous 2 requests)
       "Every RDMA Send eventually receives ACK OR timeout"

  ③ Hardware timeout fallback:
     → Every RoCE transaction has a hardware timeout counter (configurable, default 1ms)
     → Timeout → MSI-X interrupt → scheduler retry
     → No "permanent wait" scenario

  Deadlock verification fully covered at Phase 2 (2-4 cards), no need to wait for 30 cards.
  Because the communication protocol is independent of card count — 4-card Ring and 30-card Ring use the same protocol.
```

**9.2.6 ASIC vs FPGA vs GPU Verification Comparison**

```
┌────────────────────┬──────────────┬──────────────┬──────────────┐
│                    │ GPU ASIC      │ FPGA RTL      │ GPU CUDA     │
│                    │ (H100/B200)  │ (this design) │ (kernel)     │
├────────────────────┼──────────────┼──────────────┼──────────────┤
│ Development cycle   │ 2-3 years    │ 10 months     │ weeks-months │
│ Team size           │ 300-500      │ 5             │ 1-3          │
│ Tapeout count       │ 2-3          │ 0             │ 0            │
│ Bug fix cycle       │ 6-18 months  │ 30min-2h      │ minutes      │
│ On-board iterations │ <10 (pre-sil)│ 200+          │ thousands    │
│ Simulation coverage │ Full-chip    │ Module-level  │ N/A (no HW)  │
│ Online internal sig │ Rare (metal) │ ✓ Signal Tap  │ N/A          │
│ Production fix cost │ $10M+ (metal)│ ¥0 (reconfig) │ ¥0 (recompile)│
└────────────────────┴──────────────┴──────────────┴──────────────┘

Key conclusion:
  FPGA development is "slow" relative to software, not relative to ASIC.
  Compared to GPU ASIC's 2-3 year cycle + unfixable hardware bugs,
  FPGA's 200+ on-board iterations + Signal Tap online debug is already "fast."
```

### 9.3 Challenge A Response: Development Board Empirical Validation Plan

> **Challenge A**: "The entire proposal is pure paper analysis. There is no experimental data for fp4 precision, no measurement of actual HBM bandwidth utilization, no single-card end-to-end latency. The 100-page document rests on a chain of assumptions."

**Answer: Correct. That is why Phase 1's top priority is not writing more documents, but buying a dev board and running experiments on it. The following is an empirical plan directly derived from Challenge A — every experiment maps to a specific assumption.**

#### 9.3.1 Development Board Selection

```
Target chip: Intel Agilex 7 M AGFB027 (2×HBM2e, 32 GB, 9,375 DSP)

Available development boards:

┌─────────────────────┬──────────────────────┬──────────────────────┐
│                      │ Intel Agilex 7 M      │ BittWare IA-840F     │
│                      │ Dev Kit (DK-SI-AGM027)│                      │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Chip                  │ AGFB027 (target chip) │ AGFB027 (same chip)  │
│ HBM2e                │ 32 GB                  │ 32 GB                │
│ Interface             │ PCIe 5.0 x16, QSFP-DD │ PCIe 5.0 x16, QSFP  │
│ Memory                │ DDR4 (control)         │ DDR4 + opt. HBM ext  │
│ F-Tile 200GbE        │ ✓ (QSFP-DD)            │ ✓ (QSFP28 possible)  │
│ Price (est.)          │ ~$8,000-12,000         │ ~$10,000-15,000      │
│ Lead time             │ ~4-8 weeks (stock)     │ ~6-12 weeks          │
│ Software              │ Quartus Prime Pro      │ Quartus + BittWare   │
│                       │ + ref design + BSP     │ BSP + Board Mgmt     │
│ Procurement           │ Intel/Altera official  │ BittWare direct/agent│
│ Recommended           │ ★★★ (official, ref full)│ ★★ (extra BSP needed)│
└─────────────────────┴──────────────────────┴──────────────────────┘

Recommendation: Intel DK-DEV-AGM039EA × 1 (AGM 039-F direct validation, 12,300 DSP, HBM2e 32 GB)
              Total budget: ~$8-12K hardware + Quartus Pro License ~$4K/year

> **2026/05 Update**: The original proposal recommended DK-SI-AGM027 (AGFB027, 9,375 DSP).
> It is now confirmed that DK-DEV-AGM039EA (AGMF039R47A, 12,300 DSP, HBM2e 32 GB) is
> actually available for purchase. The chip matches the production design — no downgrade
> validation needed. See docs/bringup_strategy.md and docs/bringup_checklist.md.
```

#### 9.3.2 Three Critical Experiments

**Experiment 1: fp4 DSP MAC Precision Validation (Highest Priority)**

```
Assumption: fp4×fp8 multiplication + FP32 accumulation → per-token difference
            between full layer output and PyTorch BF16 reference < 2%

Experiment design:
  Goal: Validate the most critical assumption — "fp4 precision is sufficient"

  Step 1 — Python Modeling (1 week):
    ● Extract 1 full layer's complete parameters from public DeepSeek V4 Pro weights
    ● Implement fp4 quantization simulation in PyTorch (weight fp4→fp8 decompress + fp8×fp8 MAC + FP32 accumulation)
    ● Compare against BF16 baseline token-by-token, output per-token cosine similarity
    ● This is "pure software" validation, no FPGA needed — first confirm no fundamental numerical issues

  Step 2 — FPGA RTL Implementation (3 weeks):
    ● Implement minimal fp4 systolic array: 1 128×128 systolic array
    ● DSP configured for fp4×fp8 → FP32 accumulation mode
    ● Weights loaded from HBM or on-chip SRAM
    ● Run 1 MLP layer's GEMM (not full Transformer, start with the smallest verifiable unit)

  Step 3 — Bit-Exact Comparison (1 week):
    ● Python fp4 simulation → generate golden output (element-wise)
    ● FPGA on-board runs same GEMM → Signal Tap capture internal DSP output chain
    ● Bit-exact comparison: DSP output[31:0] vs Python FP32 output
    ● Difference should be ≤ 1 ULP (unit in last place) — both use fp4×fp8 + FP32 accumulation

  Step 4 — Full 1-Layer Comparison (1 week):
    ● Extend to full 1-layer Transformer block (Q/K/V/O + Expert FFN + Router)
    ● On-board run 1000 tokens, record final activation for each token
    ● Compare per-layer output against PyTorch reference
    ● Statistics: max diff, mean diff, diff histogram

  Pass/fail criteria:
    ✓ Success: per-token cosine similarity ≥ 0.995 (equivalent to PPL degradation <1%)
    △ Acceptable: cosine similarity 0.98-0.995, need to analyze difference sources
    ✗ Stop-loss: cosine similarity < 0.98 or some layer's systemic difference >5%
              → Trigger Go/No-Go #2, activate fp8 fallback plan

  Experiment risk:
    ● If fp4 precision validation fails → the "fp4 native" foundation of the entire proposal collapses
    ● But this is still a valuable finding: knowing fp4 is infeasible for MoE inference prevents building on a mirage
    ● Fallback: full fp8 scheme (HBM bandwidth doubles, but still fits 15 layers/chip within 32GB)
```

**Experiment 2: HBM Effective Bandwidth Measurement (Second Priority)**

```
Assumption: Under MoE expert random-access load pattern, HBM effective bandwidth ≥ 550 GB/s
            (60% of theoretical 920 GB/s)

Experiment design:
  Goal: Validate the assumption that "HBM bandwidth will not become a bottleneck"

  Step 1 — Theoretical Upper-Bound Test (3 days):
    ● Sequential read 1GB contiguous block → measure pure sequential bandwidth
    ● Expectation: ≥ 800 GB/s (near theoretical 920)
    ● Purpose: confirm HBM controller and PHY are working correctly

  Step 2 — MoE Expert Simulation (1 week):
    ● Place 12 "expert" blocks in HBM, each 33 MB
    ● Randomly select expert blocks following power-law distribution (α=1.2, simulating real Router distribution)
    ● Measure effective bandwidth: total_data_read / total_time
    ● Simultaneously monitor: HBM controller bank conflict count
    ● Variants: 1 expert, 2 experts, 6 experts (simulating 0-hit, 1-hit, 2-hit)

  Step 3 — Double-Buffer Pipeline Test (1 week):
    ● Implement weight double-buffering: buffer_A in compute, buffer_B prefetching from HBM
    ● Measure: HBM load time vs DSP compute time overlap ratio
    ● Target: overlap ratio ≥ 80% (ideal 100%, i.e., HBM load completely hidden by DSP compute)

  Pass/fail criteria:
    ✓ Success: MoE random access effective bandwidth ≥ 550 GB/s (the "conservative" assumption in the proposal)
               Double-buffer overlap ratio ≥ 80%
    △ Acceptable: effective bandwidth 400-550 GB/s (throughput drops 20-30%, still acceptable)
    ✗ Stop-loss: effective bandwidth < 400 GB/s or bank conflict causes bandwidth < 40%
                → Trigger Go/No-Go #3, need to reassess 32-card cluster HBM constraints
                → If 1-hit layers become absolute bottleneck, consider reducing layers per card or increasing card count

  Bank Conflict Mitigation (if measurements fall short):
    ● HBM2e has 32 pseudo-channels, each pseudo-channel has independent banks
    ● Expert weight layout: interleaved storage by pseudo-channel
    ● Each expert 33MB → each pseudo-channel ~1MB
    ● Loading 1 expert: 32 pseudo-channels concurrent read → theoretical 920 GB/s
    ● Problems only arise when access patterns cause bank conflicts
    ● If they occur: adjust expert physical layout in HBM (key advantage: this is a weight file generation tool matter, no RTL changes needed)
```

**Experiment 3: Single-Layer End-to-End Latency Measurement (Third Priority)**

```
Assumption: Single-layer weighted-average latency ≈ 10 μs (calculation in proposal §4.4.1.4)

Experiment design:
  Goal: Validate the paper estimate of single-layer inference latency

  Method:
    ● Run full 1-layer Transformer block inference (based on Experiment 1's RTL)
    ● Measure actual latency for 3 scenarios:
      - 0 expert hit (weights in SRAM): target ≤ 5 μs
      - 1 expert hit (load 1 × 33MB expert + Router): target ≤ 40 μs
      - 2 expert hits (load 2 × 33MB experts): target ≤ 75 μs
    ● Use Signal Tap to measure HBM busy vs DSP busy time ratio
    ● Run 10,000 tokens to capture latency distribution (impact of long-tail experts)

  Pass/fail criteria:
    ✓ Success: weighted-average latency ≤ 15 μs (1.5× tolerance over proposal's 10 μs estimate)
               HBM stall ratio < 50%
    △ Acceptable: weighted-average latency 15-25 μs (throughput drop < 50%)
    ✗ Stop-loss: weighted-average latency > 25 μs (means 30-card cluster throughput < 400 tok/s,
                too large a competitive disadvantage against cloud GPUs)
```

#### 9.3.3 Experiment Boundaries: What Can and Cannot Be Validated on Board

```
Validatable on a single card (these 3 experiments cover):
  ✓ fp4 precision (full pipeline, no multi-card needed)
  ✓ HBM effective bandwidth (single-card storage system is independent)
  ✓ Single-layer latency (full layer pipeline)

Not validatable on a single card (needs multi-card, not part of Phase 1):
  ✗ Cross-card MoE dispatch latency (needs ≥2 cards + switch)
  ✗ Full 61-layer pipeline end-to-end latency (needs 4-node 32-card)
  ✗ 72h continuous run stability (needs full cluster)
  → These are verified in Phase 2 (2-4 cards)

Why the ordering of the 3 experiments matters:
  fp4 precision (Experiment 1) is the value foundation of the entire proposal → if it fails, no need to do the rest
  HBM bandwidth (Experiment 2) determines system feasibility → if it fails, architecture needs a complete redesign
  Single-layer latency (Experiment 3) is performance prediction validation → if it fails, TCO needs recalculation
```

#### 9.3.4 Decision Path After Experiment Completion

```
Experiment 1 (fp4 precision):
  ├─ Pass → continue to Experiment 2
  └─ Fail → evaluate fp8 fallback (2× weight size, may still be feasible)
            → If fp8 also fails → project stop-loss

Experiment 2 (HBM bandwidth):
  ├─ Pass → continue to Experiment 3
  └─ Fail → evaluate Bank Conflict mitigation
            → If still below threshold after mitigation → redesign weight layout or increase card count

Experiment 3 (single-layer latency):
  ├─ Pass → Phase 1 complete, enter Phase 2 (2-card verification)
  └─ Fail → reassess TCO and $/million-tokens
            → If >2× over threshold → project reassessment

All 3 experiments pass:
  → Core assumptions validated by experiments
  → Proposal upgraded from "paper analysis" to "empirically-supported engineering plan"
  → Take benchmark data to pitch seed customers
  → Launch Phase 2
```

---

## 10. Cost Analysis

### 10.1 Hardware BOM (8 Cards × 4 AGM 039, Single 4U System)

> Accounting principle: FPGA accelerator cards are self-designed and self-manufactured; server node hardware is purchased at market price. No ToR switch, no RoCE IP, no QSFP-DD cage.

| Item | Spec | Qty | Unit Price (¥) | Subtotal (¥) |
|------|------|------|---------|---------|
| FPGA chip | AGM 039-F 32GB HBM | 32 | 18,000 | 576,000 |
| Card-level materials | PCB 14+ layer / 4 VRM rails / heatsink / assembly | 8 | 48,000 | 384,000 |
| Server node | Inspur NF5688M7 / Lenovo SR670 V3 4U | 1 | 220,000 | 220,000 |
| Cables/Power/Rack | PDU + 42U rack | — | — | 30,000 |
| Spares | Full card spare × 1 | 1 | 120,000 | 120,000 |
| **Hardware Total** | | | | **~1,330,000** |

vs old design (¥2,415K, 32-card distributed): **Saves ¥1,085K (-45%)**

Correction note (2026/05): Based on actual quote $2,500 ≈ ¥18,000/chip ($1=¥7.3)

```
Chip AGM 039 ¥18,000/ea (~$2,500) is an actual quote (consistent with scripts/fpga_arch/config.py).
  vs AGM 032 ¥21,600/ea (~$3,000): similar price, but DSP +31%, LE +19%, better price/performance.
  Also, only Chip0 needs R-Tile (Chip1/2/3 can omit R-Tile cost → consider R31B package without R-Tile)

Card-level materials ¥48K/card — multi-chip card PCB is more complex, but no QSFP-DD cage +
  4 chips share heatsink/assembly, per-chip overhead lower than single-chip cards.

No ToR Switch — 8 cards all on 4U server backplane, PCIe 5.0 P2P direct.
No RoCE v2 IP — PCIe P2P self-developed DMA engine replaces it (RTL ~1.5 PM).
No QSFP-DD Cage / DAC cables — no external network devices between cards.

Single 4U server = one complete cluster, deployment requires only one power cable + one network cable (BMC).
```

### 10.2 Labor Cost

| Role | Headcount | Duration | Annual Salary (¥) | Subtotal (¥) |
|------|------|------|---------|---------|
| FPGA RTL Engineer | 5 | 10 months | 800,000 | 3,333,000 |
| Software/System Engineer | 3 | 10 months | 600,000 | 1,500,000 |
| PCB Hardware Engineer | 1 | 5 months | 600,000 | 250,000 |
| Test/Verification Engineer | 2 | 8 months | 500,000 | 667,000 |
| **Labor Total** | | | | **~5,750,000** |

### 10.3 Hardware Pricing and Gross Margin

> Pricing principle: Hardware BOM + manufacturing/testing/assembly cost + gross margin = customer selling price. R&D costs are not included in hardware pricing; they are amortized separately as IP assets.

```
                         Single (proto)  10 units (seed)  100 units (vol.)  10K units (scale)
                         ──────────    ──────────    ───────────    ───────────
Hardware BOM (§10.1 level):
  AGM 039-F (×32)          ¥18,000       ¥18,000       ¥14,000        ¥10,000
  Card-level mats (×8)     ¥48,000       ¥45,000       ¥35,000        ¥22,000
  4U server (×1)            ¥220,000      ¥200,000      ¥170,000       ¥130,000
  Cables/PDU/spares         ¥218,000      ¥220,000      ¥180,000       ¥130,000
  ───────────────────────────────────────────────────────────────────────────
  Hardware BOM subtotal     ~¥1.40M       ~¥1.36M       ~¥1.08M        ~¥0.76M

Manufacturing cost (assembly/test/burn-in):
  System assembly+test      ¥80,000       ¥60,000       ¥40,000        ¥20,000
  48h burn-in + QC          ¥50,000       ¥40,000       ¥30,000        ¥15,000
  ───────────────────────────────────────────────────────────────────────────
  Full cost (BOM + mfg)     ~¥1.53M       ~¥1.46M       ~¥1.15M        ~¥0.79M

Hardware gross margin:
  Margin %                   35%           40%           45%            50%
  Margin amount              ¥823K         ¥971K         ¥939K          ¥791K
  ───────────────────────────────────────────────────────────────────────────
  Hardware price (excl. IP)  ~¥2.35M       ~¥2.43M       ~¥2.09M        ~¥1.58M
                             ≈ $326K       ≈ $337K       ≈ $290K        ≈ $220K
```

```
Gross margin scaling logic:
  10 units:  35% — small-batch manufacturing cost high, customer discount space large (seed customer discount)
  100 units: 45% — manufacturing efficiency improves, brand premium begins to show
  10K units: 50% — scale effect fully realized, but maintained at IT hardware industry standard margin

  Benchmarks: NVIDIA H100 8-card server gross margin ~65-70% (monopoly premium)
              Huawei Ascend system gross margin ~40-50% (domestic substitution premium)
              FPGA solution 35-50% — below GPU monopoly premium, but above commodity server (15-20%)
```

### 10.4 R&D Investment (IP Assets)

> R&D costs are not included as part of hardware cost but form IP assets, recovered through License fees or NRE amortization.

| Item | Amount (¥) | Amortization Method |
|------|---------|---------|
| FPGA RTL IP (5×10 months) | 3,333,000 | 5-year straight-line, per unit shipped |
| Software/Driver IP (3×10 months) | 1,500,000 | |
| PCB Reference Design (1×5 months) | 250,000 | |
| Test Verification (2×8 months) | 667,000 | |
| Tools / IP License / Equipment | 1,000,000 | (Quartus License, simulation verification) |
| **R&D IP Total** | **~6,750,000** | |

```
IP Amortization Model (5-year straight-line, residual = 0):

                      10 units (seed)  100 units (vol.)  10K units (scale)
                      ───────────    ────────────    ─────────────
5-year total shipment    15 units       150 units       15,000 units
  (incl. follow-on orders)

IP amortization/unit     ¥450K          ¥45K            ¥0.45K
                         ≈ $62K         ≈ $6.2K         ≈ $62

IP as % of HW price      16%            2.0%            0.03%
```

```
Key logic:
  ● 10 units proto: IP amortization is heavy (¥450K/unit), but seed customers value exclusive capability over unit price
    → Can convert part of IP fee to one-time NRE (customer pays ¥2-5M for customization rights)
  ● 100 units: IP amortization ¥45K/unit, 2% of selling price, nearly negligible
  ● 10K units: IP amortization <¥500/unit, completely submerged in hardware margin
    → At this point, IP is a pure profit engine: ¥6.75M investment → generating continuous annual License revenue

  Essential difference from GPU model:
    NVIDIA's R&D investment (billions of dollars) is already amortized into each chip's selling price (H100 die cost ~$300, selling price ~$30K)
    FPGA RTL IP is a self-owned asset, no third-party payments required (no RoCE IP, no PCIe Switch SDK)
    → The larger the scale, the thinner the IP amortization, the thicker the hardware margin
```

### 10.5 Three-Tier Customer Delivery Pricing

```
┌──────────────────────┬──────────────┬──────────────┬──────────────┐
│                       │ 10 (proto)   │ 100 (volume) │ 10K (scale)  │
├──────────────────────┼──────────────┼──────────────┼──────────────┤
│ HW price (incl. margin)│ ¥2.80M      │ ¥2.20M       │ ¥1.47M       │
│ IP License (amort/unit)│ ¥450K       │ ¥45K         │ ¥0.5K        │
│ Annual ops (optional)  │ ¥300K       │ ¥250K        │ ¥200K        │
├──────────────────────┼──────────────┼──────────────┼──────────────┤
│ Customer Year-1 TCO   │                              │              │
│   (incl. IP)          │ ~¥3.55M      │ ~¥2.50M      │ ~¥1.67M      │
│ Customer Year-1 TCO   │                              │              │
│   (IP lump-sum)       │ ~¥3.10M      │ ~¥2.45M      │ ~¥1.67M      │
└──────────────────────┴──────────────┴──────────────┴──────────────┘

  IP lump-sum: Customer can choose one-time IP License payment (¥2-5M) instead of per-unit
           amortization, suitable for buyout deployments (e.g., finance/government on-premise scenarios).
```

### 10.6 Comparison and Conclusions

```
Comparison (single system, incl. 3-year ops):

  This design FPGA (10K units):  ~¥1.67M + ops ~¥0.6M = ~¥2.27M (≈ $312K)
  This design FPGA (100 units):  ~¥2.50M + ops ~¥0.75M = ~¥3.25M (≈ $447K)
  NVIDIA H100 8-card server:     ~¥1.5M (unavailable, under export controls)
  Huawei Ascend 950PR:           ~¥1.2M (capacity-constrained, China only)

Core advantages unchanged:
  ① Purchasable → absolute prerequisite for meaningful pricing
  ② Globally deployable → not subject to export controls
  ③ Single 4U = one cluster → lowest deployment cost
  ④ Multiple supply chain sources → no single-vendor lock-in
  ⑤ Self-owned IP → no third-party IP tax (vs GPU's CUDA ecosystem lock-in)
  ⑥ Healthy hardware margin → sustainable business model (35-50% vs commodity server 15-20%)
  ⑦ Architectural advantage: B=1 effective bandwidth ~83× vs GPU (§11.A.2) → not "cheaper," it's architecturally different
```

---

### 10.7 Revised Unit Economics (§4.6.1 + §4.8.x Post-Optimization)

§10.1-§10.6 cost data is based on the baseline configuration (single-session 800 tok/s), consistent with §11 tables. This section provides revised figures after §4.6.1/§4.8.x software optimizations are all enabled, for use in customer deployment cost estimation.

```
Key changes (based on §4.6.1.7 end-to-end validation, 18-configuration matrix):
  Numerator: annual TCO per system (revised $2,500/chip base): ¥643-768K (10-10K units)
            After Pipeline Cloning ×2, HBM weight region goes from 0.7→1.2 GB,
            still within 32 GB budget, no hardware cost increase.
  Denominator: effective annual output depends on workload type:

  ┌────────────┬──────────────────┬──────────────────┬──────────┐
  │ Workload   │ baseline TPS    │ optimized TPS    │ Multiplier│
  ├────────────┼──────────────────┼──────────────────┼──────────┤
  │ chat       │   792 tok/s      │   803 tok/s      │ ×1.01    │
  │ agent      │   961 tok/s      │ 5,782 tok/s      │ ×6.0     │
  │ burst      │ ~17,445 upper    │ ~17,445 upper    │ ×1.0 TPS │
  │            │ (TTFT 142s)      │ (TTFT 469ms)     │ TTFT ×304│
  └────────────┴──────────────────┴──────────────────┴──────────┘

  → chat workload: optimization ineffective, baseline already sufficient
  → agent workload: primary beneficiary scenario (×5.9 TPS)
  → burst workload: limited by DSP physical peak (TPS flat), but Pipeline Cloning rescues TTFT

At 70% annual utilization (assuming 50% time agent + 50% time chat):
  baseline effective TPS = 0.5 × 961 + 0.5 × 782 = 871 tok/s
                          → annual output ~19B tokens
  optimized effective TPS = 0.5 × 5,782 + 0.5 × 782 = 3,282 tok/s
                          → annual output ~72B tokens
  Improvement ×3.0

┌──────────────────────────┬──────────┬──────────┬──────────┐
│                          │ 10 units │ 100 units│ 10K units│
├──────────────────────────┼──────────┼──────────┼──────────┤
│ baseline $/million token  │ $6.0     │ $5.0     │ $3.8     │
│ revised $/million token   │ $1.73    │ $1.30    │ $1.03    │
│ Improvement               │ -71%     │ -74%     │ -73%     │
└──────────────────────────┴──────────┴──────────┴──────────┘

Note: Revised figures based on mixed workload assumption (50% agent + 50% chat).
      Pure agent workload yields even lower cost (×5.9 fully amortized): ~$0.70 (10 units), ~$0.55 (100 units)
      Pure chat workload cost near baseline: ~$6 (optimization yields no benefit, system unsaturated)

Benchmarks:
  Ascend 910C          ~$12-18/M  (single session, limited by CANN scheduling)
  NVIDIA H100 cloud    ~$12-20/M  (but Chinese customers cannot purchase)
  DeepSeek V4 Pro API  $1.46/M    (mixed workload: ¥0.1/¥12/¥24 cache hit/miss/output)
  FPGA 10-unit revised $1.73/M    ← slightly above API, but supply chain/data sovereignty wins
  FPGA 100-unit revised $1.30/M   ← already better than API (direct result of architecture bandwidth efficiency)
  FPGA 10K-unit revised $1.03/M   ← significantly better than API (scale effect at volume)
  ASIC phase (§13)     $0.4-0.6/M (architecture efficiency hardened + process cost collapse)

Key arguments:
  1. The root cause of FPGA $/token advantage is not "hardware is cheaper," but effective bandwidth
     utilization of 83× (see §11.A.2) — same $1 hardware, more effective bandwidth → more tokens → $/M naturally lower
  2. Data sovereignty + privacy compliance scenarios (finance/healthcare/government): FPGA is competitive at revised figures
  3. Overseas deployment (Belt and Road / going global): GPUs unobtainable, price comparison meaningless

Prerequisites for revised figures:
  ✓ Customer enables §4.6.1 optimization set (default recommended, zero hardware cost)
  ✓ Service workload type is multi-session (agent/copilot/API)
  ✗ Not applicable to pure batch=1 single-user extreme-low-latency scenarios
```

Detailed derivation in `docs/tco_per_million_tokens.md` §5.2.

---

