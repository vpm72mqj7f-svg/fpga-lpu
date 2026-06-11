# AF_V2LITE FPGA Design Specification — Rev 1.1

> **Reference**: AF5ACC_Design_Spec_TM_CE24_v1.0.odt (Arrive Technologies)
> **Date**: 2026-06-11
> **Device**: Stratix 10 MX — 1SM21BHU2F53E1VG
> **Top Module**: `v2_lite_full`
> **Document Type**: RTL Design Specification

---

## Document History

| Rev  | Date       | Author            | Description                                |
|------|------------|-------------------|--------------------------------------------|
| 1.0  | 2026-06-11 | V2-Lite Team      | Initial draft — bring-up baseline           |
| 1.1  | 2026-06-11 | V2-Lite Team      | Full rewrite: Clock Arch, PCIe EP, DSP IP, CDC |

## Document Review

| Date       | Reviewer | Comment |
|------------|----------|---------|
| 2026-06-11 | —        | Pending |

## Document Approval

| Date       | Approver | Signature |
|------------|----------|-----------|
| 2026-06-11 | —        | Pending   |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Design Goals & Technology Overview](#2-design-goals--technology-overview)
3. [Design Requirements](#3-design-requirements)
4. [Clock Architecture](#4-clock-architecture)
5. [Top-Level Block Design](#5-top-level-block-design)
6. [Compute Architecture — FFN Engine](#6-compute-architecture--ffn-engine)
7. [PCIe Subsystem Design](#7-pcie-subsystem-design)
8. [HBM2 Subsystem Design](#8-hbm2-subsystem-design)
9. [ISP Debug Architecture](#9-isp-debug-architecture)
10. [Register Map — PCIe BAR0](#10-register-map--pcie-bar0)
11. [Configuration Sequence](#11-configuration-sequence)
12. [Functional Timing](#12-functional-timing)
13. [FPGA Resource Estimation](#13-fpga-resource-estimation)
14. [Power Consumption](#14-power-consumption)
15. [Appendix](#15-appendix)

---

## 1. Introduction

### 1.1 Purpose

This document defines the complete RTL architecture for the V2-Lite FFN Decode Accelerator FPGA design. It serves as the single source of truth for hardware implementation, verification, and software driver development.

### 1.2 Scope

| Item | Coverage |
|------|----------|
| **Device** | Stratix 10 MX — 1SM21BHU2F53E1VG |
| **Function** | DeepSeek V4 Pro MoE FFN decode-only inference (no training, no prefill) |
| **Interfaces** | PCIe Gen3 x16 Endpoint, HBM2 UIB0 (16 GB), JTAG ISP |
| **Compute** | DSP-based systolic array — FP4 weight × FP8 activation |
| **Out of scope** | Training, attention mechanism, KV cache (future), multi-FPGA clustering (V4 target) |

### 1.3 Acronyms

| Acronym | Expansion |
|---------|-----------|
| FFN | Feed-Forward Network |
| MoE | Mixture of Experts |
| FP4 / FP8 / BF16 | 4-bit / 8-bit / 16-bit Brain Floating Point |
| HBM2 | High Bandwidth Memory (2nd gen) |
| HIP | Hard IP (Intel silicon-hardened block) |
| AXI4-MM | ARM AMBA AXI4 Memory-Mapped |
| ISP | In-System Probe (Intel `altsource_probe` JTAG IP) |
| SLD | System-Level Debug (JTAG hub) |
| CDC | Clock Domain Crossing |
| SiLU | Sigmoid Linear Unit activation |
| LTSSM | Link Training & Status State Machine |
| MSI-X | Message Signaled Interrupts — Extended |
| BAR | Base Address Register (PCIe) |
| UIB | Universal Interface Bus (HBM2 PHY) |
| TG | Traffic Generator |
| TPS | Tokens Per Second |
| ALM | Adaptive Logic Module |

### 1.4 References

| Ref # | Document | Description |
|-------|----------|-------------|
| [1] | AF5ACC_Design_Spec_TM_CE24_v1.0.odt | Arrive canonical RTL design spec template |
| [2] | `AF_V2LITE_Feature_List_Rev1.0.md` | Feature requirements with traceability IDs |
| [3] | `AF_V2LITE_FPGA_Estimation_Rev1.0.md` | Resource budget and synthesis reports |
| [4] | `AF_V2LITE_Register_Map_Rev1.0.atreg` | Full register map in Arrive format |
| [5] | `AF_V2LITE_ISP_DEBUG_REGISTER_MAP.md` | ISP probe bitfield definitions |
| [6] | Intel Stratix 10 MX Device Overview | Device datasheet |
| [7] | Intel UG-20151 | DK-DEV-1SMX-H-A Development Kit User Guide |
| [8] | `V2_LITE_ENGINEERING_SPEC.md` | Superseded engineering target spec (archived) |

---

## 2. Design Goals & Technology Overview

### 2.1 Mission Statement

**FPGA 加速 DeepSeek V4 Pro MoE FFN decode 阶段，最大化单卡 TPS (Tokens Per Second)。**

The FPGA serves as a PCIe-attached inference accelerator. The ARM host handles orchestration and prefill; the FPGA streams expert weights from HBM2 through a DSP systolic array for decode-only FFN computation.

### 2.2 Performance Targets

| Metric | Phase 1 (Bring-Up) | Phase 2 (Production) | V4 Target (Agilex 7) |
|--------|-------------------|----------------------|----------------------|
| **Decode TPS** | LUT validation only | **> 500 TPS** | > 2,000 TPS |
| **Core Clock** | 100 MHz | **250 MHz** | 450 MHz |
| **DSP Usage** | 0 / 3,960 | 1,000–2,000 | AI Tensor + DSP |
| **HBM2 BW Utilization** | TG test (pass) | **> 70%** of 256 GB/s | > 80% of 512 GB/s |
| **PCIe BW Utilization** | XCVR loopback | **> 80%** of Gen3 x16 | Gen4 x16 |
| **Single-Token Latency** | N/A | < 2 ms | < 0.5 ms |

### 2.3 Why 500 TPS?

| Factor | Calculation |
|--------|-------------|
| Single expert FFN weights (FP4) | 7168 × (3072+7168+28672) × 0.5 bytes ≈ **140 MB** |
| HBM2 effective BW | ~180 GB/s (70% of 256 GB/s peak) |
| Weight load time / token | 140 MB / 180 GB/s ≈ **0.78 ms** |
| Compute time (102M MAC / expert) | ~0.5 ms @ 250 MHz with DSP |
| Total latency / token | ~1.3 ms → **~770 TPS** theoretical |
| Conservative (MoE gating + overhead) | **> 500 TPS** |

### 2.4 Technology Platform

**Stratix 10 MX (1SM21BHU2F53E1VG) key resources:**

| Resource | Available | Used (Phase 1) | Notes |
|----------|-----------|----------------|-------|
| ALM | 702,720 | 93,785 (13%) | LUT-only, no DSP |
| DSP | 3,960 | **0 (0%)** | Variable-precision, 2× int9 per block |
| BRAM (M20K) | 6,847 | 149 (2%) | |
| HBM2 UIB | 2 | 1 (50%) | 16 GB per UIB, 8 channels |
| HSSI (XCVR) | 96 | 16 (17%) | H-Tile, up to 28.3 Gbps |
| GPIO PLL | 168 | 19 (11%) | IOPLL + ATX PLL used |

---

## 3. Design Requirements

### 3.1 Functional Requirements

| REQ ID | Feature ID | Requirement | Priority | Phase |
|--------|-----------|-------------|----------|-------|
| REQ-PCIE-01 | F-PCIE-01 | PCIe Gen3 x16 Endpoint mode via Stratix 10 HIP | P0 | 2 |
| REQ-PCIE-02 | F-PCIE-02 | BAR0: 4 KB memory-mapped register space | P0 | 2 |
| REQ-PCIE-03 | F-PCIE-03 | BAR2: 4 GB prefetchable HBM2 window | P1 | 2 |
| REQ-PCIE-04 | F-PCIE-04 | MSI-X interrupt support (32 vectors) | P1 | 2 |
| REQ-HBM-01 | F-HBM-01 | HBM2 UIB0 operational (TG self-test passing) | P0 | 1 |
| REQ-HBM-02 | F-HBM-02 | AXI4 read master to FFN (256-bit, 64 O/S) | P0 | 2 |
| REQ-HBM-03 | F-HBM-03 | AXI4 write master from PCIe (weight download) | P0 | 2 |
| REQ-FFN-01 | F-FFN-01 | **DSP-based** FP4×FP8 multiply-accumulate | P0 | 2 |
| REQ-FFN-02 | F-FFN-02 | 2D weight-stationary systolic array | P0 | 2 |
| REQ-FFN-03 | F-FFN-03 | Gate / Up / Down projections (3 linear engines) | P0 | 2 |
| REQ-FFN-04 | F-FFN-04 | SiLU activation (LUT + DSP interpolation) | P0 | 2 |
| REQ-ISP-01 | F-ISP-01..04 | 4-instance ISP debug infrastructure | P0 | 1 |
| REQ-ISP-02 | F-ISP-05 | Version register per function block (Arrive .atreg) | P0 | 1 |
| REQ-REG-01 | F-REG-01 | Block version register at offset 0x00 per block | P0 | 1 |

### 3.2 DSP IP Mandate (GATE CHECK)

```
Rule R-DSP-01: ALL multiply operations MUST use altera_mult_add IP.
Rule R-DSP-02: LUT-based multiply is FORBIDDEN in production RTL.
Rule R-DSP-03: DSP PIPE_STAGES ≥ 1 for timing closure at 250 MHz.
Rule R-DSP-04: Synthesis gate check: DSP count > 0 (must not be zero).
```

**Violation of R-DSP-01 blocks synthesis sign-off.**

### 3.3 Version Register Mandate

Per Arrive .atreg convention: every function block MUST expose a 32-bit version register at offset 0x00 of its address space, formatted as `{day[7:0], month[7:0], year[7:0], number[7:0]}` where `year = actual_year - 2000`.

### 3.4 Constraints

| Constraint | Value | Source |
|------------|-------|--------|
| FPGA device | 1SM21BHU2F53E1VG | Board fixed |
| HBM2 capacity | 16 GB (UIB0 only) | 1 UIB populated |
| PCIe lanes | x16 (H-Tile banks 1C/1D/1E) | Board routing |
| Slot power limit | 75 W | PCIe CEM spec |
| JTAG | USB-Blaster II, SLD hub | Quartus 26.1 Pro |
| Build system | Quartus Prime Pro 26.1, Windows | Project standard |

---

## 4. Clock Architecture

### 4.1 Clock Sources

| Signal Name | Source Chip | Source Pin | Frequency | FPGA Pin | I/O Standard | Destination |
|-------------|------------|------------|-----------|----------|-------------|-------------|
| `core_clk_iopll_ref_clk_clk` | Si5341A U16 (CLK_SYS_100M) | — | 100 MHz | AU17 / AU16 | LVDS | IOPLL refclk → core_clk generation |
| `hbm_0_example_design_pll_ref_clk_clk` | Si5341A U16 (CLK_UIB1) | — | 100 MHz | P27 / R27 | LVDS | HBM2 UIB0 PLL (fixed refclk) |
| `clk_50m` | Si5338A U18 (CLK_SYS_50M) | — | 50 MHz | BE17 | LVDS | PCIe management domain |
| `refclk_pcie_ep_p/n` | Si5341A U16 (REFCLK_PCIE_EP) | — | 100 MHz | AW43 / AW42 | HCSL, AC-coupled | PCIe ATX PLL refclk |
| `refclk_pcie_ep_edge_p/n` | Si5341A U16 | — | 100 MHz | AR43 / AR42 | HCSL, AC-coupled | PCIe edge refclk (reserved) |
| `refclk_pcie_ep1_p/n` | Si5338A U18 | — | 100 MHz | BA43 / BA42 | HCSL, AC-coupled | PCIe secondary refclk (reserved) |

### 4.2 PLL Configuration

#### 4.2.1 IOPLL — Core Clock Generator

| Parameter | Phase 1 (Current) | Phase 2 (Target) | Notes |
|-----------|-------------------|-------------------|-------|
| **IP Instance** | `ed_synth_core_clk_iopll` | `ed_synth_core_clk_iopll` (reconfig) | Intel IOPLL IP |
| **Reference Clock** | 100 MHz LVDS | 100 MHz LVDS | Same physical refclk |
| **Feedback Mode** | Normal | Normal | |
| **VCO Frequency** | 600 MHz | 1,500 MHz | N = 6 → 15 |
| **M Counter** | 1 | 1 | Pre-divider |
| **N Counter** | 6 | 15 | Feedback divider |
| **C0 — core_clk** | 100 MHz (C=6) | 250 MHz (C=6) | Main compute clock |
| **C1 — dsp_clk** | Not used | 500 MHz (C=3) | DSP 2× overdrive (future) |
| **C2 — axi_clk** | 100 MHz (C=6) | 250 MHz (C=6) | AXI interconnect clock |
| **Bandwidth** | Low | High | Higher BW for 250 MHz stability |
| **Charge Pump Current** | 4 (default) | 6 | |
| **Loop Filter Resistance** | 7 (default) | 8 | |
| **Lock Time** | ~1 ms | ~1 ms | |
| **PLL Location** | BANK 3C (IOPLL) | BANK 3C (IOPLL) | Same physical IOPLL |

**Phase 2 IOPLL equations:**

```
f_ref = 100 MHz
f_vco = f_ref × N / M = 100 × 15 / 1 = 1,500 MHz
f_core = f_vco / C0 = 1,500 / 6  = 250 MHz     ← Target
f_dsp  = f_vco / C1 = 1,500 / 3  = 500 MHz     ← Future
f_axi  = f_vco / C2 = 1,500 / 6  = 250 MHz
```

#### 4.2.2 HBM2 UIB PLL

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Reference Clock** | 100 MHz LVDS | Dedicated UIB refclk |
| **PLL Type** | HBM2 hard IP internal | Managed by `altera_hbm` IP |
| **Configuration** | Fixed by Intel IP | Not user-configurable |
| **Outputs** | HBM2 UIB internal clocks | Separate from core logic |

#### 4.2.3 PCIe ATX PLL

| Parameter | Value | Notes |
|-----------|-------|-------|
| **IP Instance** | `pcie_xcvr_system_xcvr_atx_pll_s10_htile_0` | Intel ATX PLL IP |
| **Reference Clock** | 100 MHz HCSL, AC-coupled | `refclk_pcie_ep_p/n` |
| **Output** | XCVR serial clock for 8 GT/s (Gen3) | Drives all 16 lanes |
| **Lock Time** | ~10 ms | Longer than IOPLL |
| **Lock Indicator** | `pcie_atx_pll_locked` | Routed to LED[1] and ISP |

### 4.3 Clock Tree

```
                        Si5341A U16                        Si5338A U18
                       ┌──────────────┐                  ┌──────────────┐
                       │ CLK_SYS_100M │                  │ CLK_SYS_50M  │
                       │   100 MHz    │                  │   50 MHz     │
                       └──────┬───────┘                  └──────┬───────┘
                              │ LVDS                            │ LVDS
          ┌───────────────────┼───────────────────┐              │
          ▼                   ▼                   ▼              ▼
    ┌───────────┐      ┌───────────┐       ┌───────────┐  ┌───────────┐
    │  IOPLL    │      │ HBM2 PLL  │       │ ATX PLL   │  │ clk_50m   │
    │ core_clk  │      │  (fixed)  │       │ (PCIe)    │  │ domain    │
    │ generator │      │           │       │           │  │           │
    └─────┬─────┘      └─────┬─────┘       └─────┬─────┘  └─────┬─────┘
          │                  │                   │               │
    ┌─────┼─────┐            │                   │               │
    ▼     ▼     ▼            ▼                   ▼               ▼
┌──────┐┌────┐┌──────┐  ┌─────────┐       ┌───────────┐  ┌───────────┐
│core  ││dsp ││axi   │  │hbm_ref  │       │pcie_user  │  │pcie_mgmt  │
│_clk  ││_clk││_clk  │  │_clk     │       │_clk       │  │           │
│250MHz││500 ││250MHz│  │100 MHz  │       │250 MHz    │  │50 MHz     │
│      ││MHz ││      │  │         │       │(from HIP) │  │           │
└──┬───┘└──┬─┘└──┬───┘  └────┬────┘       └─────┬─────┘  └─────┬─────┘
   │       │     │            │                  │               │
   ▼       ▼     ▼            ▼                  ▼               ▼
┌──────┐┌────┐┌──────┐  ┌──────────┐      ┌───────────┐  ┌───────────┐
│ FFN  ││DSP ││ AXI  │  │  HBM2    │      │  PCIe HIP │  │  PLL lock │
│ Ctrl ││Arr ││ XBar │  │  Ctrl    │      │  + BAR    │  │  status   │
│      ││    ││      │  │  + TG    │      │  Decoder  │  │  + reset  │
└──────┘└────┘└──────┘  └──────────┘      └───────────┘  └───────────┘

CDC Boundaries:
  ═══  core_clk ↔ hbm_refclk  : AXI4 FIFO (HBM2 IP internal)
  ═══  core_clk ↔ pcie_user   : Dual-clock async FIFO
  ═══  core_clk ↔ dsp_clk     : Intel DCFIFO IP
  ═══  any ↔ JTAG (SLD)       : altsource_probe IP
  ═══  clk_50m ↔ core_clk     : 2-stage synchronizer (slow signals)
```

### 4.4 Clock Domain Crossing (CDC) Strategy

| # | Source Domain | Dest Domain | Mechanism | Sync Depth | IP / Custom | Notes |
|---|--------------|-------------|-----------|------------|-------------|-------|
| 1 | `core_clk` (250 MHz) | `hbm_refclk` (100 MHz) | AXI4 Clock Converter | IP-managed | HBM2 IP internal | Inherent in UIB bridge; async ratio 2.5:1 |
| 2 | `pcie_user_clk` (250 MHz) | `core_clk` (250 MHz) | Dual-Clock Async FIFO | 3-stage Gray-coded | Intel DCFIFO | PCIe HIP BAR → core AXI domain |
| 3 | `core_clk` (250 MHz) | `dsp_clk` (500 MHz) | DCFIFO | IP-managed | Intel DCFIFO | Data crossing for systolic array feed |
| 4 | `core_clk` (250 MHz) | SLD JTAG clock | `altsource_probe` | IP-managed | Intel ISP IP | No custom CDC needed |
| 5 | `clk_50m` (50 MHz) | `core_clk` (250 MHz) | 2-stage FF synchronizer | 2 | Custom RTL | PLL lock status, slow-changing signals |
| 6 | `pcie_user_clk` (250 MHz) | `hbm_refclk` (100 MHz) | AXI4 Clock Converter | IP-managed | Custom/Qsys | PCIe DMA write → HBM2 AXI write |
| 7 | `cpu_resetn` (async) | All domains | Reset synchronizer per domain | 2-stage | Custom RTL | Async assert, sync deassert |

**CDC Design Rules:**
- No multi-bit bus crossings without Gray-code or handshake
- No combinational logic between synchronizer stages
- All CDC paths routed through dedicated synchronizer modules (no inline `2'bff` chains)
- Each CDC instance tagged with parameters for SDC false-path generation

### 4.5 Reset Architecture

#### 4.5.1 Reset Sources

| Reset Signal | Source | Polarity | FPGA Pin | Synchronization |
|-------------|--------|----------|----------|-----------------|
| `cpu_resetn` | Board push-button / power-good | Active-low | PIN_BL14 (1.8V) | Async input, sync deassert per domain |
| `pcie_perstn0` | PCIe slot PERST# | Active-low | PIN_AH39 | Sync to pcie_user_clk |
| `hbm_only_reset_in_reset` | Derived from cpu_resetn | Active-high | Internal | Sync to hbm_refclk |
| `int_reset_n` | Random-start delay IP | Active-low | Internal | Auto-generated at config |

#### 4.5.2 Reset FSM

```
                          ┌──────────┐
                   ──────►│  PWR_ON  │ (power rails ramp)
                          └────┬─────┘
                               │ cpu_resetn = 0 (button pressed)
                               ▼
                          ┌──────────┐
                          │ ASSERT   │ All domain resets asserted
                          │ ALL_RST  │ ─ core, dsp, hbm2, pcie
                          └────┬─────┘
                               │ cpu_resetn = 1 (released)
                               ▼
                          ┌──────────┐
                          │ WAIT_PLL │ Counter ≈ 1 ms
                          │  _LOCK   │ Poll: pll_locked[all] == 1
                          └────┬─────┘
                               │ all PLLs locked
                               ▼
                          ┌──────────┐
                          │ RELEASE  │ Deassert core_clk domain reset
                          │  _CORE   │ Wait 16 core_clk cycles
                          └────┬─────┘
                               │
                               ▼
                          ┌──────────┐
                          │ RELEASE  │ Deassert dsp_clk domain reset
                          │  _DSP    │ Wait 16 dsp_clk cycles
                          └────┬─────┘
                               │
                               ▼
                          ┌──────────┐
                          │ RELEASE  │ Deassert hbm_refclk domain reset
                          │  _HBM2   │ Trigger HBM2 calibration via MMR
                          └────┬─────┘
                               │ HBM2 cal_done
                               ▼
                          ┌──────────┐
                          │ WAIT     │ Wait for PCIe PERST# deassert
                          │ _PERST   │ Wait for LTSSM → L0
                          └────┬─────┘
                               │ PCIe link up
                               ▼
                          ┌──────────┐
                     ┌───►│ NORMAL   │ (operational)
                     │    │  _OP     │
                     │    └──────────┘
                     │         │ cpu_resetn = 0
                     └─────────┘
```

#### 4.5.3 Reset Domain Map

| Domain Reset | Clock Domain | Asserted By | Deassert Sequence | Active Cycles |
|-------------|-------------|-------------|-------------------|---------------|
| `rst_n_core` | `core_clk` (250 MHz) | `cpu_resetn` or SW reset | After IOPLL lock + 16 cycles | ≥ 16 core_clk |
| `rst_n_dsp` | `dsp_clk` (500 MHz) | `cpu_resetn` or SW reset | After rst_n_core deassert + 16 cycles | ≥ 16 dsp_clk |
| `rst_n_hbm` | `hbm_refclk` (100 MHz) | `cpu_resetn` or SW reset | After rst_n_dsp deassert + HBM2 cal_done | ≥ HBM2 cal time (~1 ms) |
| `rst_n_pcie` | `pcie_user_clk` (250 MHz) | `pcie_perstn0` or SW reset | After ATX PLL lock + PERST# deassert | ≥ 100 ms (PCIe spec) |
| `rst_n_axi` | `axi_clk` (250 MHz) | `cpu_resetn` | After rst_n_core deassert | ≥ 16 axi_clk |

### 4.6 Timing Budget

| Clock Domain | Frequency | Period | Target Setup Slack | Clock Uncertainty | Phase 1 Slack | Notes |
|-------------|-----------|--------|-------------------|-------------------|---------------|-------|
| `core_clk` | 100 MHz → **250 MHz** | 10.0 → **4.0 ns** | > 0.20 ns | 0.10 ns | **−0.5 ns** (VIOLATION) | Fix: pipeline regs + DSP retiming |
| `dsp_clk` | — → **500 MHz** | — → **2.0 ns** | > 0.15 ns | 0.05 ns | N/A | Phase 2: needs ≥ 2 pipe stages per DSP |
| `hbm_refclk` | **100 MHz** (fixed) | **10.0 ns** | > 0.50 ns | 0.10 ns | OK | Hard IP domain |
| `pcie_user_clk` | **250 MHz** (from HIP) | **4.0 ns** | > 0.20 ns | 0.10 ns | TBD | Depends on EP config |
| `clk_50m` | **50 MHz** | **20.0 ns** | > 1.00 ns | 0.10 ns | OK | Non-critical |

**Current timing violation (Phase 1):** `core_clk` at 100 MHz shows −0.5 ns setup slack at the HBM2 UIB → user logic boundary. Root cause: long combinational path through ISP signal tap wiring. Mitigation: add pipeline registers at the Qsys user-interface boundary.

### 4.7 SDC Clock Constraints

```tcl
# Refclk definitions (Phase 1 — 100 MHz)
create_clock -name hbm_refclk  -period 10.000 -waveform {0.000 5.000} \
    [get_ports {hbm_0_example_design_pll_ref_clk_clk}]
create_clock -name core_refclk  -period 10.000 -waveform {0.000 5.000} \
    [get_ports {core_clk_iopll_ref_clk_clk}]

# Phase 2 — IOPLL generated clocks (replace after IOPLL reconfig)
# create_generated_clock -name core_clk   -source [get_ports core_clk_iopll_ref_clk_clk] \
#     -divide_by 6 -multiply_by 15 [get_pins {u_hbm|iopll|*clk[0]}]
# create_generated_clock -name dsp_clk    -source [get_ports core_clk_iopll_ref_clk_clk] \
#     -divide_by 3 -multiply_by 15 [get_pins {u_hbm|iopll|*clk[1]}]

# PCIe refclk
create_clock -name pcie_refclk -period 10.000 \
    [get_ports {refclk_pcie_ep_p}]

# clk_50m
create_clock -name clk_50m -period 20.000 \
    [get_ports {clk_50m}]

# CDC — asynchronous clock groups (Phase 2, after generated clocks exist)
# set_clock_groups -asynchronous \
#     -group {core_clk dsp_clk axi_clk} \
#     -group {hbm_refclk} \
#     -group {pcie_user_clk} \
#     -group {clk_50m}

# False paths for synchronizer chains
# set_false_path -through [get_pins -hier *cdc*sync*reg*]
```

> **Phase 1 Status (2026-06-11):** IOPLL configured for 100 MHz output (1:1, no multiplication). No generated clock SDC entries. Timing shows −0.5 ns slack at HBM2 UIB boundary. HBM2 PLL and ATX PLL locking confirmed via ISP readback.
>
> **Phase 2 Target:** IOPLL reconfigured for 250 MHz output (N=15, C0=6). Full generated clock SDC. All PLL outputs documented and constrained. Pipeline registers added to resolve timing violation.
>
> **Gate Check:** Quartus Timing Analyzer reports positive slack on all 5 clock domains at target frequencies.

---

## 5. Top-Level Block Design

### 5.1 Problem Statement

Design an FPGA-based accelerator that:
1. Receives token requests from an ARM host via PCIe Gen3 x16
2. Streams expert FFN weights from on-package HBM2 memory
3. Computes FP4 weight × FP8 activation matrix multiplies using DSP blocks
4. Returns computed activations to the host via PCIe

The critical path is **HBM2 weight bandwidth** → **DSP compute throughput** → **PCIe result return**. The design must saturate HBM2 bandwidth (>180 GB/s effective) and use DSP blocks efficiently (≥1,000 DSP, no LUT-based multiply).

### 5.2 Top-Level Block Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                      ARM Host (172.16.95.198)                          │
│                      PCIe Root Complex                                  │
│                      Linux + V2-Lite Driver                             │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ PCIe Gen3 x16 (128 Gbps / direction)
            ┌──────────────────┼──────────────────┐
            │  refclk           │  PERST#          │  WAKE#
            ▼                   ▼                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│  v2_lite_full — Stratix 10 MX 1SM21BHU2F53E1VG                         │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    PCIe Subsystem (pcie_xcvr_system)              │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐   │  │
│  │  │ ATX PLL  │──▶│ PCIe HIP │──▶│   BAR    │──▶│ AXI4 Master  │   │  │
│  │  │ (100MHz) │   │ Gen3 x16 │   │ Decoder  │   │  (256-bit)   │───┼──┤──► HBM2 Write
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────────┘   │  │
│  │                                       │                            │  │
│  │                       ┌──────────┐    │  AXI4-Lite (32-bit)        │  │
│  │                       │  MSI-X   │    ▼                            │  │
│  │                       │  (32vec) │ ┌──────────┐                    │  │
│  │                       └──────────┘ │ Register │                    │  │
│  │                                    │   Map    │                    │  │
│  │                                    │ (BAR0)   │                    │  │
│  │                                    └──────────┘                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌────────────────────────── HBM2 Subsystem ────────────────────────┐  │
│  │  ┌──────────┐   ┌──────────────────────────────┐                 │  │
│  │  │ HBM2 PLL │──▶│  HBM2 Controller (altera_hbm) │                 │  │
│  │  │ (100MHz) │   │  UIB0 — 8 ch × 2 pseudo      │                 │  │
│  │  └──────────┘   │  16 GB, 256 GB/s peak         │                 │  │
│  │                  │  ┌──────────────────────┐    │                 │  │
│  │                  │  │  AXI4 User Interface │    │                 │  │
│  │                  │  │  256-bit R/W          │◄───┼── PCIe AXI Wr  │  │
│  │                  │  └──────────┬───────────┘    │                 │  │
│  │                  └─────────────┼────────────────┘                 │  │
│  └────────────────────────────────┼──────────────────────────────────┘  │
│                                   │ AXI4 Read (256-bit)                  │
│                                   ▼                                      │
│  ┌──────────────────────── FFN Compute Engine ───────────────────────┐  │
│  │  ┌─────────────────┐   ┌──────────────────────┐                   │  │
│  │  │ HBM2 Weight     │──▶│ 2D Systolic Array    │                   │  │
│  │  │ Reader (AXI4)   │   │ 64 lanes × 8 rows    │                   │  │
│  │  │ Double-buffered  │   │ Weight-stationary     │                   │  │
│  │  │ M20K prefetch    │   │ ┌──────────────────┐ │                   │  │
│  │  └─────────────────┘   │ │ fp8_mac (×512)   │ │                   │  │
│  │                         │ │ DSP altera_mult  │ │                   │  │
│  │                         │ │ _add per cell    │ │                   │  │
│  │                         │ └────────┬─────────┘ │                   │  │
│  │                         └──────────┼───────────┘                   │  │
│  │                                    │                                │  │
│  │                                    ▼                                │  │
│  │                         ┌──────────────────┐                       │  │
│  │                         │ SiLU Activation  │                       │  │
│  │                         │ LUT + DSP interp │                       │  │
│  │                         └────────┬─────────┘                       │  │
│  │                                  │                                  │  │
│  │                                  ▼                                  │  │
│  │                         ┌──────────────────┐                       │  │
│  │                         │ FP4 Re-quantize  │                       │  │
│  │                         │ + AXI Writer     │──► HBM2 Output Buffer │  │
│  │                         └──────────────────┘                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────── JTAG Debug ──────────────────────────────┐   │
│  │  SLD Hub ◄── USB-Blaster II                                      │   │
│  │  ├── ISP "PCIE" (96b probe)  ├── ISP "HBM2" (96b probe)        │   │
│  │  ├── ISP "FFN"  (128b probe) └── ISP "SYS"  (32b + 32b src)   │   │
│  │  └── SignalTap (optional)                                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Clock Domains:  [core_clk 250]  [dsp_clk 500]  [hbm_refclk 100]      │
│                  [pcie_user_clk 250]  [clk_50m 50]  [jtag_clk var]    │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Top-Level Pin Description

| Port Name | Direction | Width | I/O Standard | Pin Location | Description |
|-----------|-----------|-------|-------------|-------------|-------------|
| `core_clk_iopll_ref_clk_clk` | Input | 1 | LVDS | AU17 | Core IOPLL reference clock (100 MHz) |
| `core_clk_iopll_ref_clk_clk(n)` | Input | 1 | LVDS | AU16 | Core IOPLL reference clock N |
| `hbm_0_example_design_pll_ref_clk_clk` | Input | 1 | LVDS | P27 | HBM2 UIB PLL reference clock (100 MHz) |
| `hbm_0_example_design_pll_ref_clk_clk(n)` | Input | 1 | LVDS | R27 | HBM2 UIB PLL reference clock N |
| `clk_50m` | Input | 1 | LVDS | BE17 | 50 MHz system clock (PCIe management) |
| `cpu_resetn` | Input | 1 | 1.8V | BL14 | Global reset (active-low) |
| `refclk_pcie_ep_p` | Input | 1 | LVDS | AW43 | PCIe EP reference clock P |
| `refclk_pcie_ep_n` | Input | 1 | LVDS | AW42 | PCIe EP reference clock N |
| `refclk_pcie_ep_edge_p` | Input | 1 | LVDS | AR43 | PCIe edge reference clock P |
| `refclk_pcie_ep_edge_n` | Input | 1 | LVDS | AR42 | PCIe edge reference clock N |
| `refclk_pcie_ep1_p` | Input | 1 | LVDS | BA43 | PCIe secondary reference clock P |
| `refclk_pcie_ep1_n` | Input | 1 | LVDS | BA42 | PCIe secondary reference clock N |
| `pcie_ep_rx_p` | Output | 16 | HSSI | Banks 1C/1D/1E | PCIe RX serial lanes |
| `pcie_ep_tx_p` | Input | 16 | HSSI | Banks 1C/1D/1E | PCIe TX serial lanes |
| `s10_pcie_perstn0` | Input | 1 | 1.8V | AH39 | PCIe fundamental reset 0 |
| `s10_pcie_perstn1` | Input | 1 | 1.8V | — | PCIe fundamental reset 1 |
| `pcie_ep_waken` | Input | 1 | 1.8V | — | PCIe WAKE# signal |
| `pcie_ep_i2c_scl` | Input | 1 | 1.8V | — | PCIe SMBus clock |
| `pcie_ep_i2c_sda` | Inout | 1 | 1.8V | — | PCIe SMBus data |
| `m2u_bridge_cattrip` | Input | 1 | 1.2V HBM | — | HBM2 catastrophic trip |
| `m2u_bridge_temp` | Input | 3 | 1.2V HBM | — | HBM2 temperature code |
| `m2u_bridge_wso` | Input | 8 | 1.2V HBM | — | HBM2 WSO (DFx) |
| `m2u_bridge_reset_n` | Output | 1 | 1.2V HBM | — | HBM2 M2U bridge reset |
| `m2u_bridge_wrst_n` | Output | 1 | 1.2V HBM | — | HBM2 WRST |
| `m2u_bridge_wrck` | Output | 1 | 1.2V HBM | — | HBM2 WRCK |
| `m2u_bridge_shiftwr` | Output | 1 | 1.2V HBM | — | HBM2 SHIFTWR |
| `m2u_bridge_capturewr` | Output | 1 | 1.2V HBM | — | HBM2 CAPTUREWR |
| `m2u_bridge_updatewr` | Output | 1 | 1.2V HBM | — | HBM2 UPDATEWR |
| `m2u_bridge_selectwir` | Output | 1 | 1.2V HBM | — | HBM2 SELECTWIR |
| `m2u_bridge_wsi` | Output | 1 | 1.2V HBM | — | HBM2 WSI |
| `led` | Output | 4 | 1.8V | BG12/BF12/BG11/BH11 | Status LEDs |

### 5.4 Module Hierarchy

```
v2_lite_full (top)                              [Phase 1: active | Phase 2: retains]
│
├── u_pcie: pcie_xcvr_system (Qsys)             [Phase 1: XCVR loopback | Phase 2: Real EP]
│   ├── PCIe Hard IP — Stratix 10 Gen3 x16      [Phase 1: loopback test | Phase 2: EP mode]
│   ├── ATX PLL — `xcvr_atx_pll_s10_htile`     [DONE: locks on all 16 lanes]
│   ├── BAR Decoder — BAR0 (4KB) + BAR2 (4GB)   [Phase 2: implement]
│   ├── AXI4-Lite Master (32-bit) — CSR access  [Phase 2: implement]
│   ├── AXI4 Master (256-bit) — HBM2 DMA write  [Phase 2: implement]
│   └── MSI-X Controller (32 vectors)           [Phase 2: implement]
│
├── u_hbm: ed_synth (Qsys)                      [Phase 1: TG only | Phase 2: AXI active]
│   ├── HBM2 Controller — `altera_hbm` (UIB0)   [DONE: calibrated]
│   ├── IOPLL — `core_clk_iopll`                [DONE: 100 MHz | Phase 2: 250 MHz]
│   ├── Traffic Generator ×16 (8ch × 2pseudo)   [DONE: 16/16 PASS]
│   ├── AXI4 User Interface — 256b R/W          [Phase 1: TG only | Phase 2: active]
│   └── AXI4 Clock Crossing FIFOs               [DONE: IP-managed]
│
├── u_ffn: v2_lite_ffn_engine (RTL)             [Phase 1: LUT .v | Phase 2: DSP .sv]
│   ├── hbm2_weight_reader.sv — AXI4 Master     [Phase 1: N/A | Phase 2: active]
│   │   └── M20K Double-Buffer (64-bank)         [Phase 2: 4096×256b per bank]
│   ├── systolic_array.sv — 2D Wt-Stationary    [Phase 1: N/A | Phase 2: active]
│   │   ├── fp8_mac.sv — DSP `altera_mult_add`  [Phase 1: N/A | Phase 2: active]
│   │   │   └── 1 DSP per MAC cell (int9×int9)   [Phase 2: 512 DSP for 64×8 array]
│   │   └── Array Controller FSM                 [Phase 2: 7-state pipeline]
│   ├── silu_activation.sv — SiLU LUT+DSP       [Phase 1: N/A | Phase 2: active]
│   │   └── 256-entry PWL LUT + 1 DSP interp    [Phase 2: 1 DSP]
│   └── (Phase 1: v2_lite_ffn_engine.v — LUT)   [Phase 1: ACTIVE | Phase 2: replaced]
│       └── Serial MAC, no DSP, 1 MAC/clk        [Phase 1: functional validation]
│
├── u_isp: v2_lite_isp_debug (RTL)              [DONE: 4 ISPs operational]
│   ├── ISP "PCIE" — 96-bit probe (3×32b)       [DONE]
│   ├── ISP "HBM2" — 96-bit probe (3×32b)       [DONE]
│   ├── ISP "FFN"  — 128-bit probe (4×32b)      [DONE]
│   └── ISP "SYS"  — 32-bit probe + 32-bit src  [DONE]
│
└── u_reg: Register Map / CSR Block             [Phase 2: implement]
    ├── PCIe BAR0 Address Decoder                [Phase 2]
    ├── Block-Level Register Interfaces          [Phase 2]
    ├── Interrupt Aggregator (MSI-X)             [Phase 2]
    └── (Phase 1: ISP readback only)             [Phase 1: JTAG-based access]
```

---

## 6. Compute Architecture — FFN Engine

### 6.1 Precision Plan

| Parameter | Format | Bit Layout | Bias | DSP Mapping |
|-----------|--------|-----------|------|-------------|
| **Weight** | FP4 E2M1 (IEEE-like) | `[s][e1 e0][m]` | exp bias=1 | Lookup → int9 for DSP |
| **Activation** | FP8 E4M3 | `[s][e3..e0][m2..m0]` | exp bias=7 | 8-bit DSP input A |
| **Scale** | FP8 E4M3 | `[s][e3..e0][m2..m0]` | exp bias=7 | × product (DSP pipe 2) |
| **Accumulator** | Int32 Q12.20 | 2's complement | — | Fabric adder tree |
| **SiLU Output** | FP8 E4M3 | `[s][e3..e0][m2..m0]` | bias=7 | FP16 DSP intermediate |

### 6.2 DSP MAC Cell — `fp8_mac.sv`

```
                    ┌─────────────────────────────────┐
   weight[3:0] ────►│ FP4 → int9 Lookup               │
   (E2M1)           │ (16-entry, 9-bit signed)        │
                    └───────────────┬─────────────────┘
                                    │ int9
                                    ▼
                    ┌─────────────────────────────────┐
   activ[7:0] ─────►│  altera_mult_add (1 DSP)        │
   (E4M3)           │  A=8b, B=9b, Result=18b         │
                    │  PIPE_STAGES=2                   │
                    └───────────────┬─────────────────┘
                                    │ product[17:0]
                                    ▼
                    ┌─────────────────────────────────┐
   scale[7:0] ─────►│  Scale × Product → FP16         │
   (E4M3)           │  (DSP pipe 2, or fabric)        │
                    └───────────────┬─────────────────┘
                                    │ fp16_result[15:0]
                                    ▼
                    ┌─────────────────────────────────┐
   accum[31:0] ◄───►│  32-bit Accumulator (fabric)    │
                    │  acc += fp16_to_int32(result)    │
                    └─────────────────────────────────┘
```

**DSP Configuration:**
```systemverilog
altera_mult_add #(
    .A_WIDTH         (8),          // activation (int8)
    .B_WIDTH         (9),          // FP4 weight decoded to int9
    .RESULT_WIDTH    (18),         // 8+9+1 = 18 bits
    .PIPE_STAGES     (2),          // 2 stages for 250+ MHz
    .INPUT_REGISTER_A("CLOCK0"),
    .INPUT_REGISTER_B("CLOCK0"),
    .OUTPUT_REGISTER  ("CLOCK0")
) u_dsp_mult (
    .clock  (clk),
    .aclr0  (1'b0),
    .dataa  (s0_a),                // int8 activation
    .datab  (s0_b),                // int9 weight
    .result (product_full)
);
```

### 6.3 Systolic Array Architecture

**Parameters:**
```systemverilog
parameter LANES    = 64;   // activation parallelism (8B per lane)
parameter M_ROWS   = 8;    // output row parallelism
parameter K_BEATS  = 32;   // input dimension time-multiplexed
parameter DATA_W   = 8;    // element width
```

**Array Organization:**
```
      activ[0]──►┌─────┐   ┌─────┐        ┌─────┐
                 │MAC  │──►│MAC  │──► ... ─►│MAC  │──► out_row[0]
      wt_row[0]─►│00   │   │01   │        │063  │
                 └─────┘   └─────┘        └─────┘
      activ[1]──►┌─────┐   ┌─────┐        ┌─────┐
                 │MAC  │──►│MAC  │──► ... ─►│MAC  │──► out_row[1]
      wt_row[1]─►│10   │   │11   │        │163  │
                 └─────┘   └─────┘        └─────┘
                    ...        ...           ...
      activ[7]──►┌─────┐   ┌─────┐        ┌─────┐
                 │MAC  │──►│MAC  │──► ... ─►│MAC  │──► out_row[7]
      wt_row[7]─►│70   │   │71   │        │763  │
                 └─────┘   └─────┘        └─────┘
                 ◄──────── 64 columns ────────►
                 (activation dimension: 7168 / 64 = 112 columns tiled)
```

**FSM States:**
```
IDLE ──► WEIGHT_PRELOAD ──► STREAM ──► DRAIN ──► REDUCE ──► STORE ──► NEXT_ROW ──► DONE
  ▲                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
```

| State | Description | Duration |
|-------|-------------|----------|
| IDLE | Wait for start signal | Until FFN_START |
| WEIGHT_PRELOAD | Load weight tiles from HBM2 → M20K buffer | ~4K cycles per tile |
| STREAM | Feed activations, accumulate dot-products | K_BEATS cycles (32) |
| DRAIN | Pipeline flush | 2 cycles (DSP latency) |
| REDUCE | Row-wise reduction via adder tree | log2(M_ROWS)=3 cycles |
| STORE | Write result row to output buffer | 1 cycle |
| NEXT_ROW | Loop to next output row, or advance to next tile | 0 cycles (combinatorial) |
| DONE | Assert done flag, signal ISP counter | Until FFN_START deassert |

### 6.4 SiLU Activation — `silu_activation.sv`

**Method:** Piecewise-Linear Approximation (256-entry LUT + 1 DSP for interpolation)

```
   x[15:0] ─────►┌──────────────┐
   (FP16)        │ 256-entry LUT │──► base[15:0]
                 │  (M20K ROM)   │──► slope[15:0]
                 └──────────────┘
                                 │
                                 ▼
                 ┌──────────────────────────┐
                 │ DSP: y = base + slope×dx │  (1 DSP, fp16)
                 └──────────┬───────────────┘
                            │
                            ▼
                      silu(x)[15:0]
```

### 6.5 HBM2 Weight Reader — `hbm2_weight_reader.sv`

**Configuration:**
```systemverilog
parameter MAX_BURST     = 256;      // AXI4 max burst length
parameter DATA_WIDTH    = 256;      // AXI4 data width (bits)
parameter ADDR_WIDTH    = 28;       // AXI4 address width
parameter NUM_BUFFERS   = 2;        // Double-buffered
parameter BUF_DEPTH     = 4096;     // Words per buffer (256-bit words)
parameter NUM_BANKS     = 64;       // Bank-interleaved M20K
```

**Operation:**
- Reads 256-bit words from HBM2 via AXI4 read channel
- Double-buffered: prefetch next tile while current tile is computing
- Bank-interleaved: 64 M20K banks, sequential addresses rotate through banks
- Throughput: 256b × 100 MHz = 25.6 Gbps per pseudo-channel → ~200 Gbps aggregate (16 pseudo-channels)

### 6.6 Performance Analysis

**Throughput Equation:**
```
MAC_per_clk  = LANES × M_ROWS = 64 × 8 = 512 MAC/clk
GMAC_per_sec = 512 × 250 MHz = 128 GMAC/s (DSP multiply rate)

Effective TPS:
  MAC_per_token = 7168 × (3072 + 7168 + 28672) ≈ 278M MAC (all 3 projections)
  Compute time  = 278M / 128G ≈ 2.2 ms
  Weight load   = 140 MB / 180 GB/s ≈ 0.78 ms
  Total latency ≈ 2.2 + 0.78 + 0.5 (overhead) ≈ 3.5 ms
  Theoretical   = ~285 TPS (single expert, single tile)

  With multi-expert pipelining and weight prefetch:
  Conservative   > 500 TPS
```

> **Phase 1 Status (2026-06-11):** LUT-only FFN engine (`v2_lite_ffn_engine.v`) with serial MAC (1 MAC/clk). DSP count = 0. Self-test FSM drives AXI reads from HBM2. No weight preloader, no systolic array, no SiLU in hardware. Production SystemVerilog modules (`systolic_array.sv`, `fp8_mac.sv`, `silu_activation.sv`, `hbm2_weight_reader.sv`) are written and simulation-verified but NOT in QSF synthesis flow.
>
> **Phase 2 Target:** All `.sv` production modules integrated into QSF. `altera_mult_add` DSP IP instantiated for 512 MAC cells (64 lanes × 8 rows). 3 projection engines (gate/up/down). SiLU LUT + DSP interpolation. Double-buffered HBM2 weight reader. 128 GMAC/s at 250 MHz.
>
> **Gate Check:** Quartus synthesis reports DSP > 0. Quartus fitter reports positive slack at 250 MHz.

---

## 7. PCIe Subsystem Design

### 7.1 Configuration

| Parameter | Phase 1 (Current) | Phase 2 (Target) |
|-----------|-------------------|-------------------|
| **Mode** | XCVR Loopback Test | **Endpoint (EP)** |
| **HIP** | None (raw XCVR) | Stratix 10 PCIe Hard IP |
| **Generation** | — | Gen3 (8 GT/s) |
| **Lanes** | 16 (banks 1C/1D/1E) | x16 |
| **Refclk** | 100 MHz (AC-coupled) | 100 MHz (ATX PLL) |
| **User Clock** | — | 250 MHz (from HIP) |
| **QSys System** | `pcie_xcvr_system` (loopback) | `pcie_ep_system` (real EP) |

### 7.2 BAR Assignment

| BAR | Size | Type | Prefetchable | Purpose |
|-----|------|------|-------------|---------|
| **BAR0** | 4 KB | Memory (32-bit) | No | Register Map (CSR) — all control/status |
| **BAR2** | 4 GB | Memory (64-bit) | Yes | HBM2 Window — direct weight/activation access |

### 7.3 AXI-MM Bridge Architecture

```
PCIe HIP (Hard IP)
  │
  ├── RX Master (BAR0) ──► AXI4-Lite Master ──► Register Map (CSR)
  │   32-bit, non-prefetchable
  │
  ├── RX Master (BAR2) ──► AXI4 Master (256-bit) ──► HBM2 Write Port
  │   64-bit, prefetchable         │
  │                                └── Address translation: BAR2 offset → HBM2 phys addr
  │
  └── TX Slave ──► MSI-X Controller ──► Interrupt Generation
       (completions from FPGA → Host)
```

**Address translation (BAR2 → HBM2):**
```
HBM2_phys_addr = {channel[2:0], pseudo[0], row[12:0], bank[2:0], col[9:0]}
BAR2_offset     = HBM2_phys_addr[31:0]  // linear window, 4 GB
```

### 7.4 Interrupt Architecture

| Vector | Source | Description |
|--------|--------|-------------|
| 0 | FFN Engine | Token computation complete |
| 1 | HBM2 Controller | ECC correctable error |
| 2 | HBM2 Controller | ECC uncorrectable error |
| 3 | PCIe HIP | Link status change |
| 4 | DMA Engine | Weight download complete |
| 5–31 | Reserved | Future expansion |

### 7.5 LTSSM State Machine (EP View)

```
         ┌──────┐
    ────►│Detect│ (receiver detection)
         └──┬───┘
            │
            ▼
         ┌──────┐
         │Polling│ (bit lock, symbol lock)
         └──┬───┘
            │
            ▼
         ┌──────────┐
         │Configurat.│ (link width/speed negotiation)
         └──┬───────┘
            │
            ▼
    ┌──────────────┐
    │ L0 (normal)  │◄──── Data transfer active
    └──┬───────┬───┘
       │       │
       ▼       ▼
   ┌──────┐ ┌──────┐
   │ L0s  │ │ L1   │  (power saving)
   └──┬───┘ └──┬───┘
      │        │
      └───┬────┘
          ▼
    ┌──────────┐
    │ Recovery │ (retrain)
    └──────────┘
```

**Key LTSSM registers (ISP accessible):**

| Register | Address | Description |
|----------|---------|-------------|
| PCIE_LTSSM_STATE | 0x1004 | Current LTSSM state encoding (0=Detect, ..., 6=L0) |
| PCIE_LINK_STATUS | 0x1008 | `{speed[3:0], width[5:0], link_up}` |
| PCIE_LTSSM_HISTORY | 0x100C | Sticky history of state transitions |

### 7.6 Phase 1 → Phase 2 Migration

```
Phase 1 (Current):
  XCVR Native PHY (16 lanes) → Loopback Controller → Pattern Gen/Check
  ATX PLL locks, per-lane PLL locks verified via ISP
  No PCIe HIP, no BAR, no AXI bridge, no driver

Phase 2 (Target):
  XCVR Native PHY → PCIe HIP (Gen3 x16 EP) → BAR Decoder → AXI Bridges
  BAR0 (4KB CSR) + BAR2 (4GB HBM2 Window)
  MSI-X (32 vectors)
  ARM host driver: weight download via BAR2, control via BAR0
```

> **Phase 1 Status (2026-06-11):** XCVR loopback only. PCIe HIP instantiated as loopback test mode. ATX PLL locks confirmed on all 16 lanes via ISP bitfields. Per-lane PLL lock status readable at ISP "PCIE" probe bits. No PCIe link training, no BAR, no AXI bridge, no DMA.
>
> **Phase 2 Target:** Full Gen3 x16 Endpoint. BAR0 register read/write functional from ARM host. BAR2 HBM2 write functional — host can download weights via memory-mapped writes. MSI-X interrupts tested.
>
> **Gate Check:** `lspci -vv` on ARM host shows V2-Lite device at Gen3 x16. BAR0 register read returns correct SYS_VERSION (0x0B061A01).

---

## 8. HBM2 Subsystem Design

### 8.1 Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **UIB** | UIB0 (bottom) | UIB1 reserved for future |
| **Capacity** | 16 GB | 8 channels × 2 GB per channel |
| **Channels** | 8 (A–H) | Each: 2 pseudo-channels |
| **Logical Channels** | 16 | 8 × 2 pseudo-channels |
| **Peak BW** | 256 GB/s | 2 Gbps/pin × 1024 DQ |
| **Effective BW** | ~180 GB/s | ~30% overhead for refresh, bank conflict |
| **AXI4 Data Width** | 256 bits (32 bytes) | Per pseudo-channel |
| **AXI4 Address Width** | 28 bits | 256 MWords addressable |
| **AXI4 ID Width** | 9 bits | |
| **Outstanding Transactions** | 64 Read + 64 Write | Per pseudo-channel |
| **AXI Clock** | 100 MHz (Phase 1) → 250 MHz (Phase 2) | Same as core_clk |

### 8.2 Weight Storage Layout

```
HBM2 Physical Address Space (per UIB, 8 GB):

Channel 0 (pseudo 0):  Expert Group 0   — Experts 0..47
Channel 0 (pseudo 1):  Expert Group 1   — Experts 48..95
Channel 1 (pseudo 0):  Expert Group 2   — Experts 96..143
Channel 1 (pseudo 1):  Expert Group 3   — Experts 144..191
...
Channel 7 (pseudo 1):  Expert Group 15  — Experts 336..383

Per Expert Layout:
  Offset 0x00000000: gate_proj   weight (7168 × 3072 × 0.5B =  11 MB)  — FP4 packed
  Offset 0x00B00000: up_proj     weight (7168 × 7168 × 0.5B =  26 MB)  — FP4 packed
  Offset 0x02500000: down_proj   weight (7168 × 28672 × 0.5B = 103 MB) — FP4 packed
  Total per expert: ~140 MB

Per Channel: 48 experts × 140 MB = 6.7 GB (fits within 8 GB channel capacity)
```

### 8.3 AXI4 Read Interface (FFN Weight Loading)

```
AR Channel (FFN → HBM2):
  araddr[27:0]  — Byte address within UIB address space
  arlen[7:0]    — Burst length − 1 (0–255, max 256 beats)
  arsize[2:0]   — 0x5 (32-byte transfer = 256-bit AXI data width)
  arburst[1:0]  — 2'b01 (INCR — incrementing burst)
  arvalid       — Read address valid
  arready       — HBM2 ready to accept address

R Channel (HBM2 → FFN):
  rdata[255:0]  — Read data (32 bytes per beat)
  rresp[1:0]    — Response (00=OKAY, 10=SLVERR, 11=DECERR)
  rlast         — Last beat of burst
  rvalid        — Read data valid
  rready        — FFN ready to accept data
```

### 8.4 Traffic Generator Self-Test

The TG subsystem validates HBM2 calibration by writing/reading pseudo-random patterns through all 16 logical channels. Status is aggregated by the GPIO subsystem and displayed on LED[0].

| TG Channel | Pseudo-Channel | Pass/Fail Status |
|-----------|---------------|------------------|
| TG0..7_0 | Pseudo 0 of each channel | 8/8 PASS (Phase 1) |
| TG0..7_1 | Pseudo 1 of each channel | 8/8 PASS (Phase 1) |

> **Phase 1 Status (2026-06-11):** HBM2 calibrated. All 16 traffic generators passing (16/16 PASS). AXI4 read channel connected to FFN engine, write channel tied inactive (FFN is read-only in Phase 1). No PCIe → HBM2 write path yet.
>
> **Phase 2 Target:** PCIe BAR2 → AXI4 write master active for weight download. FFN weight reader uses AXI4 read with MAX_BURST=256 for maximum throughput. BW counters operational.
>
> **Gate Check:** Host writes 1 GB of data to BAR2, reads back via FFN AXI reader, data matches.

---

## 9. ISP Debug Architecture

### 9.1 ISP Instance Map

```
JTAG Chain (USB-Blaster II)
  │
  ▼
SLD Hub (auto-generated)
  ├── Node 00486E00: altsource_probe "PCIE" — 96-bit probe
  │   └── {pcie_probe2[31:0], pcie_probe1[31:0], pcie_probe0[31:0]}
  │
  ├── Node 00486E01: altsource_probe "HBM2" — 96-bit probe
  │   └── {hbm2_probe2[31:0], hbm2_probe1[31:0], hbm2_probe0[31:0]}
  │
  ├── Node 00486E02: altsource_probe "FFN"  — 128-bit probe
  │   └── {ffn_probe3[31:0], ffn_probe2[31:0], ffn_probe1[31:0], ffn_probe0[31:0]}
  │
  └── Node 00486E03: altsource_probe "SYS"  — 32-bit probe + 32-bit source
      └── probe: {sys_probe0[31:0]}, source: {sys_source0[31:0]}
```

### 9.2 ISP Probe Bitfields

#### PCIE ISP (96-bit)
| Word | Bits | Signal | Description |
|------|------|--------|-------------|
| probe0 | [15:0] | `pcie_pll_locked_bank[15:0]` | Per-lane PLL lock (1 per lane) |
| probe0 | [16] | `pcie_atx_pll_locked` | ATX PLL lock aggregate |
| probe0 | [31:17] | Reserved | |
| probe1 | [31:0] | `pcie_ltssm_state` | LTSSM state (Phase 2) |
| probe2 | [31:0] | `pcie_link_status` | Speed / width / up (Phase 2) |

#### HBM2 ISP (96-bit)
| Word | Bits | Signal | Description |
|------|------|--------|-------------|
| probe0 | [15:0] | `tg_pass[15:0]` | Per-pseudo-channel TG pass |
| probe0 | [31:16] | `tg_fail[15:0]` | Per-pseudo-channel TG fail |
| probe1 | [15:0] | `tg_timeout[15:0]` | Per-pseudo-channel TG timeout |
| probe1 | [31:16] | Reserved | |
| probe2 | [31:0] | `hbm_bw_read` | HBM2 read BW counter (Phase 2) |

#### FFN ISP (128-bit)
| Word | Bits | Signal | Description |
|------|------|--------|-------------|
| probe0 | [3:0] | `ffn_state` | FSM state encoding |
| probe0 | [4] | `ffn_busy` | Engine busy |
| probe0 | [5] | `ffn_done` | Computation done |
| probe0 | [6] | `ffn_pass` | Self-test pass |
| probe0 | [7] | `ffn_rx_valid` | PCIe RX valid |
| probe0 | [8] | `ffn_rx_ready` | PCIe RX ready |
| probe0 | [9] | `ffn_tx_valid` | PCIe TX valid |
| probe0 | [10] | `ffn_tx_ready` | PCIe TX ready |
| probe0 | [11] | `ffn_arvalid` | AXI read address valid |
| probe0 | [12] | `ffn_arready` | AXI read address ready |
| probe0 | [31:13] | Reserved | |
| probe1 | [7:0] | `ffn_tdata_lo` | Output data byte 0 |
| probe2 | [7:0] | `ffn_tdata_hi` | Output data byte 1 |
| probe3 | [31:0] | Reserved | |

#### SYS ISP (32-bit probe + 32-bit source)
| Word | Bits | Signal | Description |
|------|------|--------|-------------|
| probe0 | [3:0] | `led_state` | Current LED output state |
| probe0 | [16] | `config_done` | FPGA configuration done |
| probe0 | [17] | `pcie_link_up` | PCIe link up |
| probe0 | [18] | `atx_pll_lck` | ATX PLL locked |
| probe0 | [19] | `hbm_pll_lck` | HBM PLL locked |
| source0 | [0] | `ffn_start` | FFN engine start (from JTAG) |
| source0 | [1] | `ffn_reset` | FFN soft reset (from JTAG) |
| source0 | [2] | `counter_rst` | Reset performance counters |

### 9.3 Version Register Format

Per Arrive .atreg convention:

```
Register: <BLOCK>_VERSION
Address:  <block_base> + 0x00
Width:    32 bits
Type:     R_O (Read Only)

Field:
  [31:24]  day    — Build day (1–31)
  [23:16]  month  — Build month (1–12)
  [15:08]  year   — Build year − 2000
  [07:00]  number — Build number that day (1-indexed)

Example: 0x0B061A01 = June 11, 2026, Build #1
```

> **Phase 1 Status (2026-06-11):** All 4 ISP instances operational. Per-lane PLL lock, TG pass/fail, FFN FSM state, and LED state verified via `quartus_issp` TCL scripts. SYS ISP source writes functional — FFN_START can be driven from JTAG. Version registers NOT yet implemented (F-ISP-05, REQ-ISP-02).
>
> **Phase 2 Target:** Add version registers to each ISP probe. Add PCIe LTSSM state monitoring. Add HBM2 BW counters.
>
> **Gate Check:** `quartus_issp --source=tcl/read_isp.tcl` returns all 4 ISP values including version registers.

---

## 10. Register Map — PCIe BAR0

### 10.1 Address Layout

```
0x0000 — 0x0FFF:  SYS / Global registers
0x1000 — 0x1FFF:  PCIe subsystem registers
0x2000 — 0x2FFF:  HBM2 subsystem registers
0x3000 — 0x3FFF:  FFN engine registers
0x4000 — 0xFFFF:  Reserved (future expansion)
```

### 10.2 Register Types (Arrive .atreg Convention)

| Type | Meaning | Usage |
|------|---------|-------|
| R/W | Read / Write | Configuration registers |
| R/W/C | Read / Write-1-to-Clear | Sticky status bits (write 1 to clear) |
| R_O | Read Only | Status, counters |
| W_O | Write Only | Trigger-only registers |

### 10.3 Register Table

#### SYS Block (0x0000–0x0FFF)

| Address | Name | Width | Type | Reset | Description |
|---------|------|-------|------|-------|-------------|
| 0x0000 | SYS_VERSION | 32 | R_O | 0x0B061A01 | `{day, month, year, number}` |
| 0x0004 | SYS_SCRATCH | 32 | R/W | 0x4154564E | Scratchpad — CPU bus test |
| 0x0008 | SYS_CONTROL | 32 | R/W | 0x00000000 | `[7:4]` LED override, `[3]` counter_rst, `[2]` ffn_reset, `[1]` ffn_start |
| 0x000C | SYS_STATUS | 32 | R_O | 0x00010000 | `[19]` hbm_pll_lck, `[18]` atx_pll_lck, `[17]` pcie_link_up, `[16]` config_done, `[3:0]` led_state |
| 0x0010 | SYS_CLK_FREQ | 32 | R_O | — | Measured core_clk frequency (Hz) |
| 0x0014 | SYS_CLK_FREQ_DSP | 32 | R_O | — | Measured dsp_clk frequency (Hz) |
| 0x0018 | SYS_RESET_STATUS | 32 | R/W/C | — | Per-domain reset status / sticky |

#### PCIe Block (0x1000–0x1FFF)

| Address | Name | Width | Type | Reset | Description |
|---------|------|-------|------|-------|-------------|
| 0x1000 | PCIE_VERSION | 32 | R_O | 0x0B061A01 | PCIe block version |
| 0x1004 | PCIE_LTSSM_STATE | 32 | R_O | 0x00000000 | Current LTSSM state (0=Detect ... 6=L0) |
| 0x1008 | PCIE_LINK_STATUS | 32 | R_O | 0x00000000 | `[9:4]` negotiated width, `[3:0]` speed (1=Gen1, 2=Gen2, 3=Gen3), `[0]` link_up |
| 0x100C | PCIE_LTSSM_HISTORY | 32 | R/W/C | 0x00000000 | Sticky history of LTSSM state transitions |
| 0x1010 | PCIE_PLL_STATUS | 32 | R_O | — | `[15:0]` per-lane PLL lock, `[16]` ATX PLL lock |
| 0x1014 | PCIE_INTERRUPT_STATUS | 32 | R/W/C | 0x00000000 | Pending interrupt flags |
| 0x1018 | PCIE_INTERRUPT_MASK | 32 | R/W | 0x00000000 | Interrupt enable mask |
| 0x101C | PCIE_DMA_CONTROL | 32 | R/W | 0x00000000 | DMA start/stop/status |
| 0x1020 | PCIE_DMA_SRC_ADDR | 32 | R/W | 0x00000000 | DMA source address (BAR2 offset) |
| 0x1024 | PCIE_DMA_DST_ADDR | 32 | R/W | 0x00000000 | DMA destination address (HBM2 phys) |
| 0x1028 | PCIE_DMA_LENGTH | 32 | R/W | 0x00000000 | DMA transfer length (bytes) |

#### HBM2 Block (0x2000–0x2FFF)

| Address | Name | Width | Type | Reset | Description |
|---------|------|-------|------|-------|-------------|
| 0x2000 | HBM2_VERSION | 32 | R_O | 0x0B061A01 | HBM2 block version |
| 0x2004 | HBM2_TG_STATUS | 32 | R_O | — | `[15:0]` TG pass, `[31:16]` TG fail |
| 0x2008 | HBM2_TG_TIMEOUT | 32 | R_O | — | `[15:0]` TG timeout |
| 0x200C | HBM2_BW_READ | 32 | R_O | — | HBM2 read BW (MB/s) |
| 0x2010 | HBM2_BW_WRITE | 32 | R_O | — | HBM2 write BW (MB/s) |
| 0x2014 | HBM2_ECC_CORR | 32 | R/W/C | 0x00000000 | Correctable ECC error count |
| 0x2018 | HBM2_ECC_UNCORR | 32 | R/W/C | 0x00000000 | Uncorrectable ECC error count |
| 0x201C | HBM2_TEMP | 32 | R_O | — | `[2:0]` HBM2 temperature code |

#### FFN Block (0x3000–0x3FFF)

| Address | Name | Width | Type | Reset | Description |
|---------|------|-------|------|-------|-------------|
| 0x3000 | FFN_VERSION | 32 | R_O | 0x0B061A01 | FFN block version |
| 0x3004 | FFN_CONTROL | 32 | R/W | 0x00000000 | `[1]` ffn_start, `[2]` ffn_reset, `[7:4]` expert_id |
| 0x3008 | FFN_STATUS | 32 | R_O | 0x00000000 | `[3:0]` state, `[4]` busy, `[5]` done, `[6]` pass |
| 0x300C | FFN_TOKEN_COUNT | 32 | R/W/C | 0x00000000 | Tokens processed (sticky) |
| 0x3010 | FFN_CYCLE_COUNT | 32 | R/W/C | 0x00000000 | Clock cycles elapsed (sticky) |
| 0x3014 | FFN_ACC_SATURATION | 32 | R/W/C | 0x00000000 | Accumulator saturation events |
| 0x3018 | FFN_EXPERT_ID | 32 | R/W | 0x00000000 | Current expert selection |
| 0x301C | FFN_THROUGHPUT | 32 | R_O | — | Real-time TPS × 1000 |

> **Phase 1 Status (2026-06-11):** Register map defined but accessible only via ISP (JTAG), not PCIe BAR0. SYS_VERSION and SYS_SCRATCH implemented in ISP. SYS_CONTROL.BIT[1] (FFN_START) functional via SYS ISP source. Version registers NOT implemented per block. Full register map in `.atreg` format at `AF_V2LITE_Register_Map_Rev1.0.atreg`.
>
> **Phase 2 Target:** PCIe BAR0 AXI4-Lite bridge active. All registers accessible from ARM host. Version registers implemented per block per Arrive convention. Performance counters operational.
>
> **Gate Check:** Host reads 0x0000 → returns 0x0B061A01. Host writes 0x0004 → host reads 0x0004 → returns written value.

---

## 11. Configuration Sequence

### 11.1 Power-On Sequence

```
1. Board power rails ramp (12V PCIe slot → PMBus regulators)
2. SmartVID PMBus negotiation (VCC core voltage auto-calibration)
3. FPGA configuration from QSPI flash (ASx4 Fast, ~2 seconds)
4. CONFIG_DONE asserts
5. INIT_DONE asserts (device enters user mode)
```

### 11.2 PLL Lock Sequence

```
1. IOPLL (core_clk):
   - Refclk stable (~100 µs after power)
   - IOPLL locks within ~1 ms
   - core_clk stable at 100 MHz (Phase 1) or 250 MHz (Phase 2)

2. HBM2 PLL:
   - Dedicated refclk stable
   - PLL locked by HBM2 hard IP during calibration
   - hbm_pll_lck indicator → ISP readable

3. ATX PLL (PCIe):
   - refclk_pcie_ep_p/n stable
   - ATX PLL locks within ~10 ms
   - atx_pll_locked indicator → LED[1] (off = locked) + ISP
   - All 16 per-lane PLLs lock within ~1 ms of ATX lock
```

### 11.3 HBM2 Initialization

```
1. HBM2 controller reset deasserted
2. UIB calibration sequence (auto, ~1 ms)
   - DQ calibration, write leveling, read DQS gate training
3. MMR programming (auto by altera_hbm IP)
4. Traffic generators start (Phase 1) or AXI channels enabled (Phase 2)
5. TG pass signals → 16/16 → LED[0] on
```

### 11.4 PCIe Link Training (Phase 2)

```
1. PERST# deasserted (PCIe spec: 100 ms after power stable)
2. LTSSM: Detect → Polling → Configuration → L0
3. Link width negotiation: x16
4. Link speed negotiation: Gen3 (8 GT/s)
5. PCIe HIP generates pcie_user_clk (250 MHz)
6. BAR0 and BAR2 enabled
7. MSI-X configured by host driver
```

### 11.5 FFN Engine Start (Phase 2)

```
Host procedure:
1. Load weights: write expert weights to BAR2 → HBM2
2. Configure: write expert_id to FFN_EXPERT_ID (0x3018)
3. Start: write FFN_START=1 to FFN_CONTROL (0x3004)
4. Poll: read FFN_STATUS (0x3008) until FFN_STATUS.DONE=1
5. Read results: read FFN_TOKEN_COUNT, FFN_THROUGHPUT
6. Clear counters: write COUNTER_RST=1 to SYS_CONTROL
```

### 11.6 Runtime PLL Reconfiguration (Phase 2)

```
For frequency scaling without full FPGA reconfiguration:
1. Gate FFN clock (write FFN_RESET=1)
2. Reconfigure IOPLL via Avalon-MM (M/N/C counter update)
3. Wait for IOPLL relock (~1 ms)
4. Release FFN reset
5. Verify: read SYS_CLK_FREQ (0x0010)

Note: HBM2 PLL and ATX PLL are fixed — not reconfigurable at runtime.
```

---

## 12. Functional Timing

### 12.1 AXI4 Read Burst Timing (HBM2 → FFN)

```
         T0   T1   T2   T3   T4   T5   T6   T7   T8   T9
clk      ─┐   ┌┐   ┌┐   ┌┐   ┌┐   ┌┐   ┌┐   ┌┐   ┌┐   ┌┐
                                      

araddr   ──X<ADDR>──────────────────────────────────────X──
arvalid  ────┐     ┌───────────────────────────────────────
arlen    ──X<LEN=X>──────────────────────────────────────X──
arready  ──────┐   ┌───────────────────────────────────────

rdata    ──────────X<D0>X<D1>X<D2>X<...>X<DN>X────────────
rvalid   ────────────┐                       ┌─────────────
rlast    ───────────────────────────────────┐ ┌────────────
rready   ──────────────────────────────────────────────── (always ready)

Burst length: len+1 beats. Read latency: ~3 cycles (HBM2 UIB).
At 250 MHz: 256b × (len+1) / 4ns = 64 × (len+1) Gbps per channel.
```

### 12.2 Systolic Array Pipeline Timing

```
         T0    T1    T2    ...  T31   T32   T33   T34
clk      ─┐    ┌┐    ┌┐        ┌┐    ┌┐    ┌┐    ┌┐

State    <IDLE><── WEIGHT_PRELOAD ──><── STREAM (K_BEATS=32) ──><DR><RED><STO>

activ[0] ───────X<a0>──X── ... ──X<a31>─────────────────────────────
wt[0]    ───────X<w0>──X── ... ──X<w31>─────────────────────────────
mac[0]   ───────────────X<p0>─X── ... ──X<p31>───────────────────────
accum    ─────────────────────── ... ──────────────────X<sum>────────
done     ────────────────────────────────────────────────────┐ ┌─────

Pipeline depth: 2 cycles (DSP pipe stages) + 1 cycle (DRAIN).
Total: 3 + K_BEATS cycles per output row.
```

### 12.3 PCIe TLP → AXI Write Translation

```
PCIe MWr TLP (from Host)          AXI4 Write (to HBM2)
┌─────────────────────┐           ┌─────────────────────┐
│ TLP Header (3-4 DW) │           │ AW Channel          │
│  Fmt=3, Type=0x00   │           │  awaddr[27:0]       │
│  Length=10 DW       │  ──►      │  awlen=9            │
│  Address[31:2]      │           │  awsize=0x2 (4B)    │
│  (BAR2 offset)      │           │  awburst=INCR       │
├─────────────────────┤           ├─────────────────────┤
│ Data Payload        │           │ W Channel           │
│  DW0..DW9           │  ──►      │  wdata[255:0]       │
│  (320 bytes)        │           │  wstrb[31:0]        │
└─────────────────────┘           │  wlast at beat 9   │
                                  └─────────────────────┘
Translation latency: 2-3 pcie_user_clk cycles.
```

---

## 13. FPGA Resource Estimation

### 13.1 Resource Budget Summary

| Block | ALM | DSP | BRAM (M20K) | HSSI | UIB | PLL |
|-------|-----|-----|-------------|------|-----|-----|
| PCIe HIP + ATX PLL | — | — | — | 16 | — | 1 |
| HBM2 Controller (UIB0) | — | — | — | — | 1 | 1 |
| AXI Interconnect | 15,000 | — | 20 | — | — | — |
| Register Map / CSR | 2,000 | — | 4 | — | — | — |
| **FFN Engine** | **60,000** | **1,000** | **100** | — | — | — |
| ISP Debug (4 instances) | 5,000 | — | 10 | — | — | — |
| Misc (LED, PLL, etc.) | 2,000 | — | 5 | — | — | — |
| Margin (20%) | 16,800 | — | — | — | — | — |
| **TOTAL (estimated)** | **~100,800** | **1,000** | **~139** | **16** | **1** | **2** |
| **Available** | 702,720 | 3,960 | 6,847 | 96 | 2 | 168 |
| **Utilization** | **14%** | **25%** | **2%** | **17%** | **50%** | **1%** |

### 13.2 Phase 1 vs Phase 2 Comparison

| Resource | Phase 1 (Current) | Phase 2 (Target) | Notes |
|----------|-------------------|-------------------|-------|
| ALM | 93,785 (13%) | ~100,800 (14%) | LUT→DSP saves ~33K ALM |
| **DSP** | **0 (0%)** 🔴 | **1,000 (25%)** | Gate check item |
| BRAM | 149 (2%) | ~139 (2%) | Similar |
| HSSI | 16 (17%) | 16 (17%) | x16 Gen3 |
| UIB | 1 (50%) | 1 (50%) | UIB0 only |
| **Timing Slack** | **−0.5 ns** 🔴 | **> 0.2 ns** | Gate check item |

### 13.3 FFN DSP Budget Breakdown

| Component | DSP/Instance | Instances | DSP Total |
|-----------|-------------|-----------|-----------|
| systolic_cell (FP4×FP8 MAC) | 1 | 64 × 8 = 512 | 512 |
| 3 Projection Engines (gate/up/down) | — | 3 arrays | 512 × 1 = 512 (time-muxed) |
| Scale Pre-Decode | 1 | 16 | 16 |
| SiLU Interpolation | 1 | 1 | 1 |
| Dot-Product Reduction | 1 | 8 | 8 |
| **Subtotal** | | | **~537** |
| **With Spares + Headroom** | | | **~1,000** |

**Note:** With time-multiplexed projection engines, the 512-DSP array is reused across gate/up/down projections, reducing total DSP to ~537. The 1,000 DSP target (25% utilization) provides comfortable headroom.

### 13.4 Timing Closure Plan

The Phase 1 −0.5 ns slack is caused by long combinational paths at the HBM2 UIB → user logic boundary. Fixes for Phase 2:

1. **Pipeline registers** at Qsys user-interface boundary (HBM2 AXI → FFN)
2. **DSP retiming** — `altera_mult_add` with PIPE_STAGES=2 adds pipeline stages inside DSP blocks
3. **Physical synthesis** — `FLOW_ENABLE_HYPER_RETIMER_FAST_FORWARD ON` (already in QSF)
4. **Logic Lock regions** — constrain FFN DSP array to specific DSP columns for predictable routing

> **Phase 1 Status (2026-06-11):** 93,785 ALM, 0 DSP, 149 BRAM. Timing −0.5 ns slack at 100 MHz. HBM2 calibration passing. All 16 TGs passing.
>
> **Phase 2 Target:** Replace LUT-only FFN with DSP-based systolic array. 1,000 DSP. Positive slack at 250 MHz on all domains.
>
> **Gate Check:** Quartus Fitter reports `; DSP Block Usage ; 1000 / 3960 (25%)` in fitter report. No negative slack in clock domains.

---

## 14. Power Consumption

### 14.1 Power Rail Budget

| Power Rail | Voltage | Estimated Power | Source |
|-----------|---------|----------------|--------|
| VCC (Core) | 0.85V (SmartVID) | ~25 W | PMBus regulator |
| VCC_HBM | 1.2V | ~15 W | Dedicated HBM2 rail |
| VCCR_GXB (XCVR) | 1.0V | ~8 W | H-Tile transceiver rail |
| VCCT_GXB (XCVR TX) | 1.0V | ~3 W | H-Tile TX buffer |
| VCCIO | 1.8V | ~5 W | GPIO, JTAG, LED |
| **Total** | | **~56 W** | |
| **PCIe Slot Limit** | | **75 W** | ✅ Compliant |

### 14.2 Phase 2 Power Projection

At 250 MHz with 1,000 DSP:

| Rail | Phase 1 (100 MHz, 0 DSP) | Phase 2 (250 MHz, 1K DSP) | Delta |
|------|--------------------------|---------------------------|-------|
| VCC (Core) | ~25 W | ~35 W | +10 W (higher freq + DSP) |
| VCC_HBM | ~15 W | ~15 W | 0 (same UIB activity) |
| VCCR_GXB | ~8 W | ~8 W | 0 (same Gen3 x16) |
| Other | ~5 W | ~5 W | 0 |
| **Total** | **~53 W** | **~63 W** | **+10 W** |
| Margin to 75W | 22 W | 12 W | Acceptable |

---

## 15. Appendix

### 15.1 SDC Constraint File

Refer to: `v2_lite/v2_lite.sdc` (Phase 1), `v2_lite/v2_lite_250m.sdc` (Phase 2).

Key constraints:
```tcl
# Phase 1 — 100 MHz (current)
create_clock -name hbm_refclk -period 10.000 -waveform {0.000 5.000} \
    [get_ports {hbm_0_example_design_pll_ref_clk_clk}]
create_clock -name core_refclk -period 10.000 -waveform {0.000 5.000} \
    [get_ports {core_clk_iopll_ref_clk_clk}]

# Phase 2 — 250 MHz (target)
# create_generated_clock -name core_clk -source [get_ports core_clk_iopll_ref_clk_clk] \
#     -divide_by 6 -multiply_by 15 [get_pins {u_hbm|iopll|*clk[0]}]
# create_generated_clock -name dsp_clk  -source [get_ports core_clk_iopll_ref_clk_clk] \
#     -divide_by 3 -multiply_by 15 [get_pins {u_hbm|iopll|*clk[1]}]

# I/O constraints
set_input_delay -clock core_refclk -max 1.0 [get_ports {cpu_resetn}]
set_input_delay -clock core_refclk -min 0.5 [get_ports {cpu_resetn}]
set_output_delay -clock core_refclk -max 1.0 [get_ports {led[*]}]
set_output_delay -clock core_refclk -min 0.5 [get_ports {led[*]}]

# False paths
set_false_path -from [get_ports {cpu_resetn}] -to [get_clocks {core_refclk}]
```

### 15.2 QSF Pin Assignment Summary

Refer to: `v2_lite/v2_lite_full.qsf` for complete pin assignments (~350 lines).

Key assignments covered in Section 5.3 (Pin Description) and Section 4.1 (Clock Sources).

### 15.3 IOPLL Phase 2 Configuration

To be generated by Quartus IOPLL IP wizard and exported as `ip/ed_synth/ed_synth_core_clk_iopll.ip`:

```
Reference Clock:     100.000 MHz
Operation Mode:      Normal (not source-synchronous)
Feedback Mode:       Normal
PLL Mode:            Fractional-N (if needed)

M Counter:           1 (pre-divider)
N Counter:           15 (feedback divider)
VCO Frequency:       1,500 MHz

C0: core_clk         250 MHz  (divide by 6)   50% duty cycle
C1: dsp_clk          500 MHz  (divide by 3)   50% duty cycle
C2: axi_clk          250 MHz  (divide by 6)   50% duty cycle

Bandwidth Setting:   High (for 250 MHz stability)
Charge Pump Current: 6
Loop Filter R:       8
Loop Filter C:       2

Lock Time:           ~1 ms
```

### 15.4 Build Commands

```
# Quartus synthesis + fit + assembly
cd v2_lite
quartus_sh --flow compile v2_lite_full

# Program FPGA
quartus_pgm -c "USB-BlasterII" -m JTAG -o "p;output_files/v2_lite_full.sof"

# ISP probe readback
quartus_stp -t read_isp_local.tcl

# SignalTap capture (optional)
quartus_stp -t capture.tcl
```

### 15.5 Known Gaps (Phase 1 → Phase 2)

| # | Gap | Impact | Planned Resolution |
|---|-----|--------|-------------------|
| 1 | DSP count = 0 | All MAC in LUT fabric | Integrate `systolic_array.sv` + `fp8_mac.sv` into QSF |
| 2 | PCIe is XCVR loopback | Cannot load weights from host | Replace Qsys with real PCIe HIP EP |
| 3 | −0.5 ns timing slack | Might fail at 250 MHz | Pipeline registers + DSP retiming |
| 4 | No BAR → AXI bridge | No host-to-HBM2 path | Implement AXI4-Lite (BAR0) + AXI4 (BAR2) bridges |
| 5 | Version registers missing | Violates Arrive convention | Add to each ISP / CSR block |
| 6 | IOPLL at 100 MHz (1:1) | Core clock below target | Reconfigure IOPLL for N=15, C0=6 |
| 7 | No Linux driver | No host-side control | Write PCIe kernel driver or UIO userspace driver |

### 15.6 Revision History (This Document)

| Rev | Date | Author | Description |
|-----|------|--------|-------------|
| 1.0 | 2026-06-11 | V2-Lite Team | Initial bring-up spec (superseded) |
| 1.1 | 2026-06-11 | V2-Lite Team | Full rewrite: 15 chapters, Clock Arch, PCIe EP design, DSP IP mandate, CDC strategy, reset FSM, timing budget, Arrive .atreg register map |
