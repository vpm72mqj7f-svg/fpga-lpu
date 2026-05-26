# FPGA LPU Phase 1 开发板验证详细方案

> 开发板: DK-DEV-AGM039EA (AGMF039R47A, 12,300 DSP, 32 GB HBM2e)
> 核心原则: 逐级扩大验证范围，每级通过后才进入下一级
>            MAC → Scale Reader → Systolic Tile → Linear Engine → Expert FFN → Full Layer
> 总周期: 8 周 (可并行部分标注)

---

## 一、验证策略总览

```
Week 1-2: 基础链路打通
  Quartus 综合 → Golden Top 跑通 → LED/UART alive → PCIe BAR0 MMIO 读写

Week 2-3: 实验 1 — fp4 MAC 精度 (Go/No-Go #1)
  单 MAC → Scale Reader → 15 golden vectors → Signal Tap 逐 bit 对比

Week 3-5: 实验 2 — HBM 带宽 (Go/No-Go #2)
  顺序读写 → Zipf 随机读 → Bank conflict 测量 → 双缓冲重叠率

Week 5-7: 实验 3 — 单层端到端 (Go/No-Go #3)
  RMSNorm → Attention → Router → ExpertFFN → RMSNorm
  每个 sub-module 的 valid handshake 用 Signal Tap 抓时序

Week 8: 决策
  三 gate 均 PASS → 下单 8×AGM039 量产芯片 + 启动 Phase 2 PCB 设计
  任一 STOP → 根因分析 + 架构调整
```

---

## 二、实验 1：fp4 MAC 精度

### 2.1 为什么这是最高优先级

整个方案的价值主张建立在 "fp4 原生推理" 上。如果 Agilex DSP 在 fp4×fp8 模式下的实际 rounding 行为和 Python 模型偏差超过 2%，方案必须切换到 fp8 路径（HBM 带宽需求翻倍，单芯片能承载的 expert 数减半）。

**仿真不能回答的问题：**
- DSP block 的 18×19 signed multiplier 在 infer fp4×fp8 时的低 4-bit rounding 行为
- DSP cascade chain 中 scale 乘法的延迟（仿真假设同一拍完成，实际 DSP 有 register stage）
- Scale reader 的 BRAM read 与 MAC valid 的时序对齐（Icarus 没有 setup/hold 概念）

### 2.2 硬件连线

```text
┌─────────────── BAR0 MMIO (PCIe) ───────────────┐
│                                                  │
│  ┌──────────────────────┐   ┌─────────────────┐  │
│  │   Scale Load FSM      │   │  Golden Checker │  │
│  │   (writes 512 groups) │   │  (readback +     │  │
│  │   scale_mem[0..511]  │   │   compare)       │  │
│  └─────────┬────────────┘   └────────┬────────┘  │
│            │ scale_wr_en/addr/data    │           │
│            ▼                          │           │
│  ┌──────────────────────┐             │           │
│  │  fp4_scale_reader    │             │           │
│  │  (GROUP_SIZE=16)     │             │           │
│  └──────────┬───────────┘             │           │
│             │ r_scale (8b)            │           │
│             ▼                          │           │
│  ┌──────────────────────┐             │           │
│  │  fp4_mac             │─────────────┼──► compare │
│  │  (scale-aware)       │ mac_result   │           │
│  │  weight(4b)          │              │           │
│  │  activ(8b)           │              │           │
│  └──────────────────────┘              │           │
│                                        │           │
│  Signal Tap nodes:                     │           │
│    s0_weight, s1_w_signed,             │           │
│    s1_a_scaled, s1_sc_scaled,          │           │
│    s2_product, accumulator             │           │
└────────────────────────────────────────┘
```

### 2.3 上板步骤

**Step 1 — Scale memory 写入 (Day 1-2)**

从 PCIe BAR0 写入 512 个 group 的 scale 值。对于 15 golden tests，所有 scale 都是 `0x38`（fp8 +1.0），所以这一步几乎不需要操作。但我们仍然验证 scale reader 的 write/read 路径：

```
Python host script:
  for addr in range(512):
      fpga.write_scale(addr, 0x38)       # all groups = 1.0
  for addr in [0, 16, 32, 511]:
      assert fpga.read_scale(addr) == 0x38
```

**Step 2 — 单 MAC golden vector 验证 (Day 2-3)**

从 Python 加载 T1（single multiply：fp4 +1.0 × fp8 +1.0）：

```
Python → BAR0 write (0x1004):
  accum_clr=1, go=1, weight=0x4, scale=0x38, activ=0x38
→ wait test_done bit
→ BAR0 read (0x1008): expect 0x00001000
```

如果 T1 通过，再跑 T2（4-term accumulation）、T3（positive sweep）、直到 T14（non-unity scale）。

**Step 3 — Signal Tap 介入 (Day 3-5，仅在任一测试失败时)**

```text
Signal Tap 配置:
  Clock:      clk_dsp (450 MHz, from PLL: 100 MHz × 9/2)
  Trigger:    mac_valid_in && weight == trigger_weight
  Depth:      4K samples per node
  Nodes:
    u_mac|s0_weight[3:0]
    u_mac|s0_scale[7:0]
    u_mac|s0_activ[7:0]
    u_mac|s1_w_signed[7:0]      ← fp4 decoded, compare to Python fp4_decode_signed()
    u_mac|s1_a_scaled[11:0]     ← fp8 activation decoded, compare to Python fp8_decode_signed()
    u_mac|s1_sc_scaled[11:0]    ← fp8 scale decoded, compare to Python fp8_decode_signed()
    u_mac|s2_product[31:0]      ← (w × a × s) >>> 8, compare to Python product_rtl()
    u_mac|accumulator[31:0]     ← running sum, compare to Python compute_accum()
    u_mac|mac_out|valid         ← pipeline drain complete
```

**Step 4 — 扩展到 128×128 systolic tile (Day 5-7)**

把 `fp4_systolic_tile.sv`（4-lane）扩展到 32 个 tile（128 lanes），用 Python 生成的 128×128 GEMM golden 对比：

```
Python: gen_tb_vectors.py --mode tile --lanes 128
  → 生成 128 个 weight × 128 个 activation 的 GEMM expected output
  → 通过 PCIe DMA 加载权重和激活到 HBM
  → FSM 从 HBM 流式读取，送入 tile array
  → 累加器输出与 Python golden 逐元素对比
```

### 2.4 判定标准

```
✓ PASS:  15/15 golden tests 逐 bit 匹配
         Per-token cosine similarity ≥ 0.995 (128×128 tile test)
         → 实验 1 通过，fp4 精度假设在硬件上验证成立

△ WARN:  13-14/15 通过，1-2 个测试有 ≤ 2 ULP rounding 差异
         分析差异是否来自 DSP rounding 而非逻辑 bug
         如果是 DSP 行为 → 记录为 known behavior，继续
         如果是逻辑 bug → 修复后重跑全 suite

✗ STOP:  < 13/15 通过，或任一测试差异 > 2 ULP
         → Signal Tap 逐 stage 对比，定位差异来源
         → 如果确认为 DSP 硬件限制 → 评估 fp8 备选方案
         → 如果是 RTL bug → 修复后重新综合 (30min-2h iteration)
```

---

## 三、实验 2：HBM 有效带宽

### 3.1 为什么需要实测

Python 模型假设 HBM 是 "零延迟顺序读"。实际硬件有 bank conflict、row buffer miss、command bus contention 三个降速因素。MoE expert 随机访问下，有效带宽可能从 920 GB/s 降到 300-500 GB/s，直接影响 decode per-token 延迟。

### 3.2 硬件连线

```text
┌─────────────── BAR2 HBM Aperture (PCIe) ─────────────┐
│                                                        │
│  Host → DMA write 1 GB pattern to HBM                  │
│  Host → DMA write expert blocks (12 × 33 MB) to HBM   │
│                                                        │
│  ┌──────────────────────────────┐                      │
│  │  HBM Read Engine (FSM)       │                      │
│  │  - trace[0..N] in BRAM       │                      │
│  │  - issues Avalon-MM reads     │                      │
│  │  - measures total cycles      │                      │
│  │  - writes result to BAR0      │                      │
│  └──────────┬───────────────────┘                      │
│             │ Avalon-MM (2048-bit)                     │
│             ▼                                          │
│  ┌──────────────────────────────┐                      │
│  │  HBM2e Controller (Hard IP)  │                      │
│  │  32 GB, 32 pseudo-channels   │                      │
│  └──────────────────────────────┘                      │
│                                                        │
│  Signal Tap nodes:                                     │
│    hbm_rd_req_valid, hbm_rd_data_valid                 │
│    hbm_rd_addr, cycle_counter                          │
└────────────────────────────────────────────────────────┘
```

### 3.3 实验矩阵

```
测试 A: 顺序 1 GB 读 (pure sequential BW upper bound)
  方法:  从 base_addr 连续读 1 GB
  预期:  ≥ 800 GB/s (>87% theoretical)
  Gate:  ≥ 700 GB/s
  不通过: HBM controller 或 PHY 配置问题 → 排查参考时钟/PLL/时序约束

测试 B: 12 expert blocks, Zipf random access
  方法:
    Python gen_hbm_trace.py → 100K addresses, α=1.0 Zipf
    Load trace into FPGA BRAM (100K × 32b = 400 KB, fits in M20K)
    FSM reads from BRAM, issues Avalon-MM read per address
    Measure: effective_BW = 100K × 64B / total_cycles × f_clk
  变体:
    B1: contiguous 33 MB blocks (naive layout)
    B2: 32 KB interleaved across pseudo-channels
    B3: address XOR with token_id for hash-based spreading
  预期:  B1: 400-600 GB/s, B2: 500-700 GB/s, B3: 550-750 GB/s
  Gate:  ≥ 550 GB/s (any variant)

测试 C: 双缓冲预取 + DSP 并行
  方法:
    buffer_A: DSP 当前计算的 expert weights
    buffer_B: HBM 预取的下一个 expert weights
    两个 buffer 在 BRAM 中 ping-pong
  测量:
    Signal Tap 同时抓 hbm_rd_req_valid 和 dsp_start
    重叠率 = (cycles where both active) / total_cycles
  预期:  ≥ 80%
```

### 3.4 判定标准

```
✓ PASS:  MoE random BW ≥ 550 GB/s, overlap ≥ 80%
        → HBM 带宽假设成立，decode 延迟模型可信

△ WARN:  400-550 GB/s, overlap 60-80%
         → 吞吐降 20-30%，需重新计算 TCO 和 $/M token
         → 仍然可接受；量产时可优化 expert placement 提升 BW

✗ STOP:  < 400 GB/s or overlap < 60%
         → 每 token HBM 延迟翻倍
         → 重新评估: 减少 per-chip expert 数、增加芯片数、或接受更低吞吐
```

---

## 四、实验 3：单层端到端

### 4.1 目的

验证 `full_transformer_layer.sv` 在真实硬件上的：
- 控制流正确性（RMS→Attn→RMS→Rtr→FFN→RMS FSM 不死锁）
- 数据流正确性（每级输入/输出数值匹配 Python golden）
- Per-stage 延迟（与 Icarus 仿真对比）
- 资源利用率 vs 预估

### 4.2 硬件连线

```text
┌─────────────────── BAR0 MMIO ───────────────────────┐
│                                                      │
│  Weight Load FSM (preloads all sub-module weights)   │
│    - Gate/Up/Down fp4 weights (to FFN)               │
│    - Router weights (diagonal)                       │
│    - Attention scores + V matrix                     │
│    - RMSNorm gamma (4096 ×24)                        │
│    - Scale memory (512 groups)                       │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │         full_transformer_layer                │    │
│  │                                              │    │
│  │  in ─► RMS1 ─► Attn ─► RMS2 ─► Rtr ─► FFN   │    │
│  │                                              │    │
│  │         RMS3 ◄────────────────────────────── │    │
│  │           │                                   │    │
│  │           ▼ out                               │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  Signal Tap nodes (per sub-module valid handshake):  │
│    r1_vi → r1_vo      (RMSNorm1)                     │
│    attn_vi → attn_vo  (Attention)                    │
│    r2_vi → r2_vo      (RMSNorm2)                     │
│    rtr_vi → rtr_vo    (Router)                       │
│    ffn_start → ffn_done (ExpertFFN)                  │
│    r3_vi → r3_vo      (RMSNorm3)                     │
│    valid_out + y0..y7                                 │
│                                                      │
│  Python golden compare:                               │
│    read y0..y7 from BAR0 result registers            │
│    compare to gen_layer_golden.py expected            │
└──────────────────────────────────────────────────────┘
```

### 4.3 测试 case

**C0: 全均匀输入 (identity test)**

```
输入:   [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
权重:   全部 identity (FFN gate/up 1.0, down identity,
        Router 对角线, Attention 全 uniform score + V identity)
期望:   [5793, 5793, 5793, 5793, 0, 0, 0, 0]
        router_ok=1 (expert 0 selected)
```

**C1: 混合符号输入**

```
输入:   [4096, 2048, 0, -2048, -4096, -2048, 0, 2048]
权重:   同 C0
期望:   gen_layer_golden.py 生成的 expected values
        允许 ±4 tolerance (RMSNorm isqrt rounding)
```

### 4.4 延迟分解

用 Signal Tap 抓每个 sub-module 的 valid 握手时序：

```text
Trigger: valid_in rising edge

Timeline (示例, 实际周期数取决于综合后时序):
  T=0:     valid_in ↑
  T+1:     r1_vi ↑
  T+6:     r1_vo ↑  (RMSNorm1: 5-cycle latency)
  T+7:     attn_vi ↑
  T+8:     attn_vo ↑  (Attention: 1-cycle)
  T+9:     r2_vi ↑
  T+14:    r2_vo ↑  (RMSNorm2: 5-cycle)
  T+15:    rtr_vi ↑
  T+17:    rtr_vo ↑  (Router: 2-cycle)
  T+18:    ffn_start ↑
  T+70:    ffn_done ↑  (ExpertFFN: ~50-60 cycles)
  T+71:    r3_vi ↑
  T+76:    r3_vo ↑  (RMSNorm3: 5-cycle)
  T+77:    valid_out ↑
  
Total: ~77 cycles @ 450 MHz ≈ 171 ns → ~5,800,000 tok/s theoretical
```

与 Icarus 仿真对比：仿真值 ~50-70 cycles。硬件如果慢于 2×（>140 cycles），需要排查：
- 哪个 sub-module 的延迟偏离最大
- 是否有时序违例导致额外的 register stage
- FSM 状态机是否有死等

### 4.5 判定标准

```
✓ PASS:  C0 所有 8 个输出逐 bit 匹配 Python golden
         C1 在 ±4 tolerance 内
         Total latency ≤ 2× Icarus simulation
         → Phase 1 验证成功，方案技术可行性在硬件上确认

△ WARN:  C0 匹配，C1 部分超出 tolerance
         或 latency 在 2-3× 仿真值之间
         → 调试具体差异来源
         → 如果确定是已知行为（FP8 encode/decode rounding chain），标记为 known

✗ STOP:  C0 不匹配（>2 elements 不一致）
         或 latency > 3× simulation
         或 router_ok=0（expert 选择错误）
         → Signal Tap 逐级抓数据，定位错误级
         → 如果是 RTL bug → 修复后重新综合 (30min-2h)
         → 如果是硬件限制 → 评估降低 f_clk 或增加 pipeline stage
```

---

## 五、Python Host 端脚本

### 5.1 `hw/scripts/run_golden_tests.py`（实验 1）

```python
# 解析 rtl/sim/tb_golden_pkg.sv → 15 组测试向量
# 逐组: BAR0 write(scale mem) → BAR0 write(MAC inputs) → BAR0 read(result)
# 对比: FPGA result vs Python golden expected
# 报告: PASS/FAIL per test, mismatch delta

$ python hw/scripts/run_golden_tests.py --device 0 --golden rtl/sim/tb_golden_pkg.sv
[ OK ] T1  single multiply          (0x00001000)
[ OK ] T2  4-term accumulation      (0x00001400)
...
[ OK ] T14 non-unity scale          (0x00007800)
Passed: 14/14
```

### 5.2 `hw/scripts/hbm_bench.py`（实验 2）

```python
# Sequential: DMA write 1GB pattern → DMA read back → measure BW
# Random:    Python gen trace → load to FPGA BRAM → trigger FSM → read counters
# Overlap:   Ping-pong weight load + DSP trigger → capture overlap from Signal Tap

$ python hw/scripts/hbm_bench.py --device 0 --mode sequential --size 1G
Sequential BW: 843 GB/s

$ python hw/scripts/hbm_bench.py --device 0 --mode random --trace trace_100k.bin --layout interleaved
Random BW: 612 GB/s (layout=interleaved)
```

### 5.3 `hw/scripts/run_layer_test.py`（实验 3）

```python
# Loads full_transformer_layer weights via BAR0
# Pulses valid_in → waits valid_out → reads y0..y7
# Compares to gen_layer_golden.py expected

$ python hw/scripts/run_layer_test.py --device 0 --case C0
Layer output: 5793 5793 5793 5793 0 0 0 0
PASS (exact match, router_ok=1)

$ python hw/scripts/run_layer_test.py --device 0 --case C1
Layer output: 5792 5792 5792 5792 0 0 0 0
PASS (within ±4 tolerance, router_ok=1)
```

---

## 六、三个 Go/No-Go 闸门总览

| Gate | 时间 | 实验 | 成功标准 | 失败应对 |
|------|------|------|----------|----------|
| **#1** | Week 3 | fp4 MAC 精度 | 15/15 golden tests pass, cosine ≥ 0.995 | fp8 fallback 或停止 |
| **#2** | Week 5 | HBM 带宽 | MoE random ≥ 550 GB/s, overlap ≥ 80% | 重算 TCO |
| **#3** | Week 7 | 单层端到端 | C0 exact match, latency ≤ 2× sim | 延期 1-2 周调试 |

---

## 七、开发板到量产芯片的路径

```
DK-DEV-AGM039EA (ES silicon)
  芯片: AGMF039R47A1E2VR0 (12,300 DSP, HBM2e 32 GB)
  用途: 单芯片功能验证 + 精度/带宽/延迟实证
  
  验证通过 ↓
  
量产芯片采购
  芯片: AGMF039R47A1E1VC (production silicon, 同规格)
  数量: 8 颗 (Phase 2 单卡 4 芯片 × 2 张卡)
  
  验证通过 ↓
  
Phase 2: 4-chip 加速卡 PCB + C2C 双环
  验证: 多 chip 流水线 + MoE dispatch/reduce
  
  验证通过 ↓
  
Phase 3: 8 卡 × 4 芯片 = 32 芯片完整集群
```

关键: 开发板上的 AGM039 和生产硅片是**同一个 die**，只是封装和测试等级不同。所以 Phase 1 验证的 DSP 精度、HBM 带宽数据直接适用于量产芯片。
