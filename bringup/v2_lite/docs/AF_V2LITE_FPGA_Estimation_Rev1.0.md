# V2-Lite FPGA Resource Estimation — Rev 1.0

> **Reference**: AF5CExxx_FPGA_Estimation_130607_Rev1.0_Eng_LUT.xls
> **Date**: 2026-06-11
> **Device**: Stratix 10 MX — 1SM21BHU2F53E1VG
> **Available**: 702,720 ALM / 3,960 DSP / 6,847 BRAM / 2 UIB

---

## 1. Resource Budget Summary

| Block | ALM | DSP | BRAM | HSSI | UIB | Notes |
|-------|-----|-----|------|------|-----|-------|
| **PCIe HIP** | — | — | — | 16 | — | Hard IP (no ALM) |
| **HBM2 Controller** | — | — | — | — | 1 | Hard IP |
| **Register Map** | 2,000 | 0 | 4 | — | — | BAR0 decode + CSR |
| **FFN Engine** | 60,000 | **2,000** | 100 | — | — | DSP systolic array |
| **AXI Interconnect** | 15,000 | 0 | 20 | — | — | Crossbar + FIFOs |
| **ISP Debug** | 5,000 | 0 | 10 | — | — | 4 ISP + SLD hub |
| **Misc (LED, PLL, etc)** | 2,000 | 0 | 5 | — | — | |
| **Margin (20%)** | 16,800 | — | — | — | — | |
| **TOTAL (est)** | **~100,800** | **2,000** | **139** | **16** | **1** | |
| **Available** | 702,720 | 3,960 | 6,847 | 96 | 2 | |
| **Utilization** | **14%** | **51%** | **2%** | **17%** | **50%** | |

## 2. Current Build (2026-06-11) vs Target

| Resource | Current Build | Target | Gap |
|----------|--------------|--------|-----|
| ALM | 93,785 (13%) | ~100,800 (14%) | LUT→DSP conversion saves ALM |
| **DSP** | **0 (0%)** | **2,000 (51%)** | 🔴 **MUST FIX** |
| BRAM | 149 (2%) | ~139 (2%) | OK |
| HSSI | 16 (17%) | 16 (17%) | OK |
| UIB | 1 (50%) | 1 (50%) | OK |
| PLL | 19 (11%) | ~15 (9%) | OK |

## 3. FFN DSP Budget

### 3.1 Per-Systolic-Cell DSP
```
1 FP4×FP8 MAC = 1 int9 × int9 multiply (DSP)
  + scale × product multiply (DSP pipe stage or separate)
  
Each systolic_cell: 2 DSP (FP4×FP8 + scale multiply)
```

### 3.2 Array Dimensions
```
LANES    = 16 (parallel activations, match 256-bit AXI / 16b)
M_ROWS   = 8  (parallel output rows)
K_BEATS  = 32 (time-multiplexed input dimension)
```

### 3.3 DSP Count
| Component | DSP per instance | Instances | Total DSP |
|-----------|-----------------|-----------|-----------|
| systolic_cell (MAC) | 2 | 16 × 8 = 128 | 256 |
| 3 linear engines (gate/up/down) | — | 3 arrays | 256×3 = 768 |
| scale_reader (pre-decode multiply) | 1 | 16 | 16 |
| silu_activation (interp multiply) | 1 | 1 | 1 |
| dot_product (output reduction) | 1 | 8 | 8 |
| **Subtotal FFN** | | | **~793** |
| **Round up (with spares)** | | | **~1,000** |
| **Target (2× expansion headroom)** | | | **~2,000** |

### 3.4 LUT→DSP ALM Savings
```
Current: 93,785 ALM (all MAC in LUT)
Target:  60,000 ALM (MAC in DSP) + 2,000 DSP

ALM saved: ~33,785 ALM → available for other logic
```

## 4. BRAM Budget

| Component | BRAM | Depth × Width | Notes |
|-----------|------|---------------|-------|
| HBM2 TG FIFOs | 50 | — | from ed_synth Qsys |
| AXI Crossbar FIFOs | 20 | 512 × 256b | per-channel buffering |
| Weight Buffer (FFN) | 32 | 4096 × 256b | double-buffered |
| Activation Buffer | 8 | 1024 × 256b | input staging |
| Scale LUT | 4 | 448 × 16b | pre-decoded scales |
| SiLU LUT | 1 | 256 × 32b | piecewise-linear |
| ISP SLD RAM | 10 | — | JTAG buffer |
| Register Map | 4 | — | CSR registers |
| Margin | 10 | — | |
| **Total** | **~139** | | |
| Available | 6,847 | | |
| **Utilization** | **2%** | | |

## 5. PCIe Bandwidth Budget

| Direction | Peak | Effective (85%) | Consumer |
|-----------|------|-----------------|----------|
| **Downstream** (Host→FPGA) | 128 Gbps (16 GB/s) | 13.6 GB/s | Weight download, control |
| **Upstream** (FPGA→Host) | 128 Gbps (16 GB/s) | 13.6 GB/s | Results, status |

```
Weight preload time: 8.2 GB / 13.6 GB/s ≈ 0.6 seconds (cold start)
Per-token result: < 1 KB → negligible PCIe BW
```

## 6. Power Budget

| Rail | Power (W) | Notes |
|------|-----------|-------|
| Core (VCC) | ~25W | 100K ALM @ 100MHz |
| HBM2 (VCC_HBM) | ~15W | 1 UIB active |
| PCIe (VCCR_GXB) | ~8W | 16 lanes Gen3 |
| I/O | ~5W | GPIO, JTAG |
| **Total (est)** | **~53W** | |
| PCIe slot limit | 75W | ✅ Within limit |

## 7. Timing Budget

| Clock | Period | Target Slack | Notes |
|-------|--------|-------------|-------|
| core_clk | 10.0 ns (100 MHz) | > 0.2 ns | **-0.5ns violation in current build** |
| core_clk (target) | 4.0 ns (250 MHz) | > 0.2 ns | Needs DSP pipeline stages |
| hbm_refclk | 10.0 ns (100 MHz) | > 0.5 ns | OK |
| pcie_user_clk | 4.0 ns (250 MHz) | TBD | After EP config |

**Current timing issue**: core_clk at 100 MHz has -0.5 ns setup slack.
Fix: Add pipeline registers between ISP debug and HBM2 UIB → user logic boundary.
