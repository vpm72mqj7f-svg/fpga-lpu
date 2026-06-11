# V2-Lite FPGA Design Specification — Rev 1.0

> **Reference**: AF5ACC_Design_Spec_TM_CE24_v1.0.odt (Arrive Technologies)
> **Date**: 2026-06-11
> **Device**: Stratix 10 MX — 1SM21BHU2F53E1VG
> **Top Module**: `v2_lite_full`

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ARM Host (172.16.95.198)                  │
│                        PCIe Root Complex                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ PCIe Gen3 x16 (128 Gbps)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  v2_lite_full (Stratix 10 MX — 1SM21BHU2F53E1VG)               │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ PCIe HIP │   │ Register │   │  HBM2    │   │   FFN    │     │
│  │ Gen3 x16 │◄─►│   Map    │◄─►│Controller│◄─►│  Engine  │     │
│  │ Endpoint │   │ (BAR0)   │   │ (1 UIB)  │   │  (DSP)   │     │
│  └────┬─────┘   └──────────┘   └────┬─────┘   └────┬─────┘     │
│       │                             │               │            │
│       │    ┌──────────────────────┐ │               │            │
│       └───►│  SLD Hub (JTAG)      │◄┘               │            │
│            │  ┌────┐┌────┐┌────┐┌────┐              │            │
│            │  │PCIE││HBM2││FFN ││SYS │              │            │
│            │  │ISP ││ISP ││ISP ││ISP │              │            │
│            │  └────┘└────┘└────┘└────┘              │            │
│            └──────────────────────┘                 │            │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Clock Architecture

| Domain | Frequency | Source | Usage |
|--------|----------|--------|-------|
| `core_clk` | **100 MHz** (→250 MHz target) | IOPLL refclk | FFN, AXI interconnect, control |
| `hbm_refclk` | 100 MHz | Dedicated refclk | HBM2 controller (fixed) |
| `pcie_refclk` | 100 MHz | PCIe refclk pair | PCIe HIP (PLL inside HIP) |
| `pcie_user_clk` | 250 MHz (TBD) | PCIe HIP output | User logic in PCIe domain |

**SDC constraint:**
```tcl
create_clock -period 10.0 -name core_clk [get_ports core_clk_iopll_ref_clk_clk]
create_clock -period 10.0 -name hbm_refclk [get_ports hbm_0_example_design_pll_ref_clk_clk]
```

## 3. Module Hierarchy

```
v2_lite_full (top)
├── u_pcie: pcie_xcvr_system (Qsys)
│   └── Stratix 10 PCIe HIP (Gen3 x16, EP mode)
├── u_hbm: ed_synth (Qsys)
│   ├── HBM2 Controller (altera_hbm) — UIB0
│   ├── Traffic Generator × 8 channels × 2 pseudo-channels
│   └── AXI4 User Interface (256-bit)
├── u_ffn: v2_lite_ffn_engine (RTL)
│   ├── fp8_mac — **DSP altera_mult_add** (per requirement)
│   ├── systolic_array — 2D weight-stationary
│   ├── silu_activation — SiLU LUT
│   └── hbm2_weight_reader — AXI4 master
├── u_isp: v2_lite_isp_debug (RTL)
│   ├── PCIE ISP (96-bit probe, altsource_probe)
│   ├── HBM2 ISP (96-bit probe, altsource_probe)
│   ├── FFN  ISP (128-bit probe, altsource_probe)
│   └── SYS  ISP (32-bit probe + 32-bit source)
└── u_reg: (planned) Register Map / CSR block
```

## 4. PCIe Subsystem Design

### 4.1 Configuration
- **Mode**: Endpoint (EP)
- **Generation**: Gen3 (8 GT/s)
- **Lanes**: x16
- **HIP**: Stratix 10 PCIe Hard IP (pcie_xcvr_system Qsys)
- **Refclk**: 100 MHz (refclk_pcie_ep_p/n)

### 4.2 BAR Assignment

| BAR | Size | Type | Purpose |
|-----|------|------|---------|
| BAR0 | 4 KB | Memory (32-bit) | Register Map (CSR) |
| BAR2 | 4 GB | Memory (64-bit, prefetchable) | HBM2 Window |

### 4.3 AXI-MM Interfaces
- **BAR0 → AXI4-Lite Master**: Register read/write
- **BAR2 → AXI4 Master (256-bit)**: Direct HBM2 access

### 4.4 Interrupts
- MSI-X: 32 vectors
- Interrupt sources: FFN done, HBM ECC error, DMA complete

## 5. HBM2 Subsystem Design

### 5.1 Configuration
- **UIB**: UIB0 (bottom), UIB1 reserved
- **Capacity**: 16 GB (1 UIB × 8 channels × 2 GB)
- **Channels**: 8 channels × 2 pseudo-channels = 16 logical
- **Bandwidth**: 256 GB/s peak, ~180 GB/s effective

### 5.2 AXI4 Interface
- **Data width**: 256-bit (32 bytes)
- **Address width**: 28-bit (256 MWords = 8 GB addressable per UIB)
- **ID width**: 9-bit
- **Outstanding transactions**: 64 (read), 64 (write)

### 5.3 Weight Storage Layout
```
HBM2 Address Space (per channel):
  Channel 0: Expert 0-47 weights, aligned to 256-bit
  Channel 1: Expert 48-95 weights
  ...
  Channel 7: Expert 336-383 weights
  
Per Expert: 7168 × (3072+7168+28672) × 0.5B ≈ 140 MB (FP4 packed)
```

## 6. FFN Engine Design

### 6.1 Precision
| Parameter | Format | Width |
|-----------|--------|-------|
| Weight | FP4 E2M1 | 4-bit |
| Activation | FP8 E4M3 | 8-bit |
| Accumulator | Integer Q12 | 32-bit |
| Scale | FP8 E4M3 | 8-bit |

### 6.2 Compute Pipeline
```
Weight[HBM2] ──→ [AXI Reader] ──→ [Scale Decode] ──→ [2D Systolic Array]
Activ[Input] ──→ [FP8 Decode] ──→                      │
                                                        ▼
                                              [32-bit Accumulate]
                                                        │
                                                        ▼
                                              [SiLU Activation]
                                                        │
                                                        ▼
                                              [FP4 Re-quantize]
                                                        │
                                                        ▼
                                              [AXI Writer] ──→ Output[HBM2]
```

### 6.3 DSP Usage (MANDATORY)
```
Rule R-DSP-01: ALL multiply operations use altera_mult_add IP

Instance counts:
  - systolic_cell (FP4×FP8 MAC): 1 DSP per cell
  - 2D array: LANES × M_ROWS DSP instances
  - Scale decode: per-lane DSP
  - SiLU interpolation: 1 DSP (shared)
  
Gate: synthesize → DSP count > 0 (must not be zero)
```

### 6.4 Performance Estimate
| Parameter | Current(100MHz, LUT) | Target(250MHz, DSP) |
|-----------|---------------------|---------------------|
| MAC/lane/clk | 1 | 1 (DSP pipelined) |
| Total MAC/clk | LANES | LANES × M_ROWS |
| Peak TPS | — | > 500 |

## 7. ISP Debug Architecture

### 7.1 ISP Instance Map
```
JTAG Chain → SLD Hub
  ├── Node 00486E00: ISP "PCIE" — 96-bit probe
  │   └── {pcie_probe2[31:0], pcie_probe1[31:0], pcie_probe0[31:0]}
  ├── Node 00486E01: ISP "HBM2" — 96-bit probe
  │   └── {hbm2_probe2[31:0], hbm2_probe1[31:0], hbm2_probe0[31:0]}
  ├── Node 00486E02: ISP "FFN"  — 128-bit probe
  │   └── {ffn_probe3[31:0], ffn_probe2[31:0], ffn_probe1[31:0], ffn_probe0[31:0]}
  └── Node 00486E03: ISP "SYS"  — 32-bit probe + 32-bit source
      └── {sys_probe0[31:0]}, {sys_source0[31:0]}
```

### 7.2 Version Register (REQUIRED — each ISP)
Per Arrive .atreg convention: every function block MUST have version.
```
Format: {day[7:0], month[7:0], year[7:0], number[7:0]}
  day:    build day (e.g. 0x0B = 11)
  month:  build month (e.g. 0x06 = June)
  year:   build year - 2000 (e.g. 0x1A = 2026)  
  number: build number that day (0x01, 0x02, ...)
```

## 8. Register Map (PCIe BAR0)

### 8.1 Address Layout
```
0x0000 — 0x0FFF: Global / SYS registers
0x1000 — 0x1FFF: PCIe registers
0x2000 — 0x2FFF: HBM2 registers
0x3000 — 0x3FFF: FFN registers
0x4000 — 0xFFFF: Reserved
```

### 8.2 Register Types (per Arrive convention)
| Type | Meaning |
|------|---------|
| R/W | Read / Write |
| R/W/C | Read / Write-1-to-Clear (sticky) |
| R_O | Read Only |
| W_O | Write Only |

### 8.3 Key Registers (see .atreg for full list)
| Address | Name | Width | Type | Description |
|---------|------|-------|------|-------------|
| 0x0000 | SYS_VERSION | 32 | R_O | Global version (day/month/year/num) |
| 0x0004 | SYS_SCRATCH | 32 | R/W | Scratchpad / CPU bus test |
| 0x0008 | SYS_CONTROL | 32 | R/W | FFN_START, FFN_RESET, LED override |
| 0x1000 | PCIE_VERSION | 32 | R_O | PCIe block version |
| 0x1004 | PCIE_LTSSM_STATE | 32 | R_O | Current LTSSM state |
| 0x1008 | PCIE_LINK_STATUS | 32 | R_O | Link speed/width/up |
| 0x2000 | HBM2_VERSION | 32 | R_O | HBM2 block version |
| 0x2004 | HBM2_TG_STATUS | 32 | R_O | TG pass/fail per channel |
| 0x2008 | HBM2_BW_READ | 32 | R_O | Read BW (MB/s) |
| 0x3000 | FFN_VERSION | 32 | R_O | FFN block version |
| 0x3004 | FFN_CONTROL | 32 | R/W | Start/Stop/Reset |
| 0x3008 | FFN_STATUS | 32 | R_O | State/Busy/Done/Pass |
| 0x300C | FFN_TOKEN_COUNT | 32 | R_O | TPS counter |
| 0x3010 | FFN_CYCLE_COUNT | 32 | R_O | Cycle counter |
