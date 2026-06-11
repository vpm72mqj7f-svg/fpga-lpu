# V2-Lite FFN Decode Accelerator — Engineering Target Specification

> **Document Type**: Performance Target & Architecture Spec  
> **Date**: 2026-06-11  
> **Status**: DRAFT — guides next synthesis iteration  

---

## 1. Mission Statement

**FPGA 加速 DeepSeek V4 Pro MoE FFN decode 阶段，最大化单卡 TPS (Tokens Per Second)。**

不做训练、不做 prefill（CPU 负责），只管 decode 的 FFN 前向推理。

---

## 2. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Decode TPS** | **> 500 TPS** per FPGA | 单 token 延迟 < 2ms |
| **时钟频率** | **250 MHz** (S10), **450 MHz** (V4 Agilex) | 当前 100MHz 差 2.5x |
| **HBM2 BW 利用率** | **> 70%** of theoretical 256 GB/s | ~180 GB/s effective |
| **PCIe BW 利用率** | **> 80%** of Gen3 x16 (128 Gbps) | 权重下载 + KV cache 通信 |

### 为什么是 500 TPS？

| 因素 | 计算 |
|------|------|
| 单个 expert FFN 权重 (FP4) | 7168 × 28672 × 0.5 bytes ≈ **102 MB** |
| HBM2 有效 BW | ~180 GB/s |
| 权重加载时间 / token | 102 MB / 180 GB/s ≈ **0.57 ms** |
| 计算时间 (102M MAC / expert) | ~0.3 ms @ 250MHz with DSP |
| 总延迟 / token | ~0.87 ms → **~1150 TPS** 理论上限 |
| 保守估计 (含 MoE gating overhead) | **> 500 TPS** |

---

## 3. Clock Architecture

| Clock Domain | Current | Target | Purpose |
|-------------|---------|--------|---------|
| `core_clk` | **100 MHz** | **250 MHz** | FFN compute, AXI interconnect, control |
| `hbm_refclk` | 100 MHz | 100 MHz | HBM2 controller ref (fixed) |
| `pcie_clk` | 50 MHz | 100-250 MHz | PCIe HIP user clock |
| `dsp_clk` | N/A | **400 MHz** (V4) | DSP systolic array (2x overdrive) |

**Action**: Replace 100MHz oscillator / PLL config → 250MHz. Update SDC `create_clock -period 4.0`.

---

## 4. Compute Architecture

### Current (V2-Lite bring-up)
```
FP4 weight × FP8 activation → LUT-based multiply → 32-bit accumulate
No DSP blocks used (0 / 3960)
Clock: 100 MHz
Throughput: ~1 MAC/clk/lane (serial)
```

### Target (V2-Lite production)
```
FP4 weight × FP8 activation → DSP-block MAC (1 DSP = 2× int9 multiply)
2D systolic array, weight-stationary
Clock: 250 MHz
DSP: 1000-2000 blocks
Throughput: ~2K MAC/clk → at 250MHz = 500 GMAC/s
```

### V4 Target (Agilex 7 M-Series)
```
FP4/FP8 mixed precision, DSP with AI Tensor blocks
Variable precision: FP4→FP8 at runtime
Clock: 450 MHz core, DSP tile 2x overdrive
AI Tensor blocks for matrix multiply
Throughput: > 10 TMAC/s
```

---

## 5. Precision Plan

| Phase | Weight | Activation | Accumulation | DSP Usage |
|-------|--------|------------|-------------|-----------|
| **V2 bring-up** | FP4 E2M1 | FP8 E4M3 | 32-bit int | LUT only |
| **V2 production** | FP4 E2M1 | FP8 E4M3 | 32-bit int | DSP (int9×int9) |
| **V4 target** | FP4/FP8 | FP8/BF16 | FP32 | AI Tensor + DSP |

FP4 权重 = 2× 压缩比 vs FP8，HBM2 BW 瓶颈下这是关键优势。

---

## 6. HBM2 Bandwidth Budget

| Consumer | BW (GB/s) | % of 256 GB/s |
|----------|-----------|---------------|
| FFN weight read | 180 | 70% |
| KV cache read/write | 30 | 12% |
| Attention weight read | 20 | 8% |
| Misc / overhead | 10 | 4% |
| **Usable** | **240** | **94%** |

Current: 1 UIB / 2 used. For V4, both UIBs should be enabled → 512 GB/s.

---

## 7. Data Path Plan (Weight Loading)

```
ARM Host (172.16.95.198)
  │  PCIe Gen3 x16 (128 Gbps)
  ▼
PCIe EP (BAR0 → AXI-MM master)
  │
  ▼
HBM2 Write Port (256-bit AXI)
  │
  ▼
HBM2 DRAM (16 GB, 8 channels)
  │  Read Port (256-bit AXI)
  ▼
FFN Engine (DSP systolic array)
```

- ARM pre-loads all expert weights into HBM2 via PCIe BAR
- Inference: FFN reads weights from HBM2, streams through systolic array
- KV cache: stored in remaining HBM2 space, accessed by attention engine

---

## 8. ISP Register Map Requirements (Next Build)

Already designed in `v2_lite_full.atreg`. Add for next build:

| Register | Width | Description |
|----------|-------|-------------|
| `FFN_PERF_CLK_FREQ` | 32b | Actual core clock frequency measurement |
| `FFN_THROUGHPUT` | 32b | Real-time TPS measurement |
| `HBM2_BW_READ` | 32b | HBM2 read BW measurement (MB/s) |
| `HBM2_BW_WRITE` | 32b | HBM2 write BW measurement (MB/s) |
| `PCIE_LTSSM` | 32b | Actual LTSSM state from HIP (not TODO) |

---

## 9. Next Steps

1. ~~ISP CLI 验证~~ (this build)
2. **时钟升频**: 100MHz → 250MHz (改 PLL + SDC)
3. **DSP 集成**: 替换 LUT-based MAC → DSP `altera_mult_add`
4. **PCIe 真实 endpoint**: 换掉 XCVR loopback → 真正 PCIe HIP EP
5. **权重下载通路**: PCIe BAR → HBM2 write port
6. **性能测量**: ISP 计数器验证 TPS / BW

---

## 10. Constraints

- S10 MX 1SM21BHU2F53E1VG: 702K ALMs, 3960 DSPs, 2 UIBs
- HBM2: 16 GB total, 8 channels × 2 pseudo-channels
- PCIe Gen3 x16: 128 Gbps (16 GB/s) per direction
- Power: < 75W (PCIe slot limit without aux power)
