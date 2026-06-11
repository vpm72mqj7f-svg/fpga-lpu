# V2-Lite FPGA Feature List — Rev 1.0

> **Reference**: Arrive_CE24_CE32_Internal_Feature_List_Rev1.0.xls
> **Date**: 2026-06-11
> **Status**: DRAFT — maps to AF_V2LITE_Design_Spec

---

## 1. Top-Level Features

| Feature ID | Feature | Priority | Status | Dependency |
|-----------|---------|----------|--------|------------|
| F-TOP-01 | PCIe Gen3 x16 Endpoint (HIP) | **P0** | In Progress | — |
| F-TOP-02 | HBM2 Controller (1 UIB, 16 GB) | **P0** | Done (TG pass) | — |
| F-TOP-03 | FFN Decode Engine (FP4×FP8) | **P0** | In Progress | DSP IP |
| F-TOP-04 | ISP Debug Infrastructure (JTAG) | **P0** | **Done** | — |
| F-TOP-05 | Register Map (PCIe BAR0) | **P0** | Planned | PCIe EP |
| F-TOP-06 | SignalTap Logic Analyzer | P1 | Done | — |
| F-TOP-07 | ARM Host Driver (Linux) | P1 | Planned | PCIe EP |
| F-TOP-08 | 250 MHz Core Clock | P1 | Planned | PLL reconfig |

---

## 2. PCIe Subsystem

| Feature ID | Feature | Priority | Status |
|-----------|---------|----------|--------|
| F-PCIE-01 | Gen3 x16 Endpoint mode | P0 | **CONFIG NEEDED** |
| F-PCIE-02 | BAR0 — Register Map (4 KB) | P0 | Planned |
| F-PCIE-03 | BAR2 — HBM2 Window (4 GB) | P1 | Planned |
| F-PCIE-04 | MSI-X Interrupts (32 vectors) | P1 | Planned |
| F-PCIE-05 | LTSSM State Monitor (ISP) | P0 | Planned |
| F-PCIE-06 | Per-Lane PLL Lock Monitor (ISP) | P0 | **Done** |

---

## 3. HBM2 Subsystem

| Feature ID | Feature | Priority | Status |
|-----------|---------|----------|--------|
| F-HBM-01 | 8-Channel TG Self-Test | P0 | **Done** (16/16 PASS) |
| F-HBM-02 | AXI4 Read Master (256-bit) | P0 | Planned |
| F-HBM-03 | AXI4 Write Master (256-bit) | P0 | Planned |
| F-HBM-04 | Weight Storage Layout (8 ch interleave) | P1 | Planned |
| F-HBM-05 | BW Monitor Counters (ISP) | P1 | Planned |
| F-HBM-06 | ECC Error Counter (ISP) | P2 | Planned |

---

## 4. FFN Compute Engine

| Feature ID | Feature | Priority | Status |
|-----------|---------|----------|--------|
| F-FFN-01 | **DSP-based** FP4×FP8 MAC | **P0** | **NOT YET — DSP=0** |
| F-FFN-02 | 2D Weight-Stationary Systolic Array | P0 | RTL exists (LUT only) |
| F-FFN-03 | Gate/Up/Down Projections (3 linear engines) | P0 | RTL exists |
| F-FFN-04 | SiLU Activation (LUT) | P0 | RTL exists |
| F-FFN-05 | Token Counter (ISP) | P0 | **Done** |
| F-FFN-06 | Cycle Counter (ISP) | P0 | **Done** |
| F-FFN-07 | AXI Read Transaction Counter (ISP) | P0 | **Done** |
| F-FFN-08 | Accumulator Saturation Monitor | P1 | Planned |

---

## 5. ISP Debug Infrastructure

| Feature ID | Feature | Priority | Status |
|-----------|---------|----------|--------|
| F-ISP-01 | PCIe ISP (96-bit probe) | P0 | **Done** ✅ |
| F-ISP-02 | HBM2 ISP (96-bit probe) | P0 | **Done** ✅ |
| F-ISP-03 | FFN ISP (128-bit probe) | P0 | **Done** ✅ |
| F-ISP-04 | SYS ISP (32-bit probe + 32-bit source) | P0 | **Done** ✅ |
| F-ISP-05 | **Version Register (each ISP)** | **P0** | **TODO** |
| F-ISP-06 | SYS Source → FFN_START control | P1 | Planned |
| F-ISP-07 | JTAG → Register Map bridge | P2 | Planned |

---

## 6. DSP IP Usage Mandate

| Rule ID | Rule | Applies To |
|---------|------|------------|
| **R-DSP-01** | **All multiply operations MUST use `altera_mult_add` IP** | FP4×FP8 MAC, systolic cell, dot-product |
| R-DSP-02 | LUT-based multiply is **FORBIDDEN** in production RTL | All modules |
| R-DSP-03 | DSP config: PIPE_STAGES=1 (min) for timing closure | altera_mult_add |
| R-DSP-04 | FPGA resource report: DSP > 0 is a **GATE CHECK** | Every synthesis run |

---

## 7. Register Map Features

| Feature ID | Feature | Priority | Status |
|-----------|---------|----------|--------|
| F-REG-01 | Block Version Register (each function block) | P0 | **TODO** |
| F-REG-02 | Global Chip ID / Version | P0 | **Done** (SYS v1.0) |
| F-REG-03 | Scratchpad Register (CPU bus test) | P1 | Planned |
| F-REG-04 | SW Force Reset (per block) | P1 | Planned |
| F-REG-05 | Interrupt Status / Mask | P2 | Planned |
