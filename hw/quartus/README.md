# FPGA LPU — Quartus Projects

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        PCIe 5.0 Backplane                        │
│   Master0    Master1    Master2    Master3  ...  Master7         │
│   (Chip 0)   (Chip 4)   (Chip 8)   (Chip12)      (Chip28)      │
│      │          │          │          │               │          │
│   ┌──┴──┐   ┌──┴──┐   ┌──┴──┐   ┌──┴──┐         ┌──┴──┐       │
│   │C2C  │   │C2C  │   │C2C  │   │C2C  │         │C2C  │       │
│   │Ring │   │Ring │   │Ring │   │Ring │         │Ring │       │
│   ├─────┤   ├─────┤   ├─────┤   ├─────┤         ├─────┤       │
│   │Slave│   │Slave│   │Slave│   │Slave│         │Slave│       │
│   │1 2 3│   │5 6 7│   │9AB │   │D E F│         │1D1E1F│      │
│   └─────┘   └─────┘   └─────┘   └─────┘         └─────┘       │
│   Card 0    Card 1    Card 2    Card 3    ...    Card 7        │
└──────────────────────────────────────────────────────────────────┘
```

**Scale:** 8 cards × 4 chips = 32 chips. 384 layers / 32 chips = 12 layers/chip.

**Per card:**
- 1 Master (Chip 0): PCIe 5.0 x16 host link + C2C ring origin + 12 layers
- 3 Slaves (Chips 1-3): C2C ring forwarding + 12 layers each
- Internal C2C dual-ring (A clockwise, B counter-clockwise)

**8 Masters** connected via PCIe 5.0 backplane to dual-socket host.

## Code Uniformity

**Both Master and Slave use the same `chip_top.sv` RTL.** The only difference
is the `IS_PCIE_MASTER` parameter:

| Parameter | Master (Chip 0,4,8...) | Slave (Chips 1-3,5-7...) |
|-----------|------------------------|---------------------------|
| `IS_PCIE_MASTER` | 1 | 0 |
| `LAYERS_PER_CHIP` | 12 | 12 |
| PCIe R-Tile IP | Included | Not synthesized (gated by parameter) |
| KV DMA engine | Included | Included (gated by parameter) |
| C2C role | Ring origin | Ring forwarder |

The synthesis tool automatically optimizes away PCIe logic when `IS_PCIE_MASTER=0`,
so a **single RTL codebase** produces both images. This ensures behavioral
consistency across all 32 chips.

## Project Structure

```
hw/quartus/
├── README.md                        This file
├── common/common_modules.qsf        Shared RTL list (all 21 modules)
├── master/
│   ├── fpga_lpu_master.qpf          Master project
│   └── fpga_lpu_master.qsf          CHIP_ID=0, PCIe enabled
├── slave/
│   ├── fpga_lpu_slave.qpf           Slave project
│   └── fpga_lpu_slave.qsf           CHIP_ID=1-31, PCIe disabled
└── fpga_lpu.qsf                     Legacy reference

hw/src/
├── top_master.sv                    Board wrapper: PCIe + HBM + C2C + chip_top
└── top_slave.sv                     Board wrapper: HBM + C2C + chip_top (no PCIe)
```

## Master FPGA Image

- **Top entity:** `top_master` (`hw/src/top_master.sv`)
- **Instantiates:** `chip_top` with `IS_PCIE_MASTER=1`
- **Used on:** 8 chips (Chip 0 of each card; global IDs 0, 4, 8, 12, 16, 20, 24, 28)
- **PCIe:** R-Tile ×16 to host root complex
- **Functions:** Host MMIO, DMA, KV cache offload, 12 transformer layers, C2C ring origin

## Slave FPGA Image

- **Top entity:** `top_slave` (`hw/src/top_slave.sv`)
- **Instantiates:** `chip_top` with `IS_PCIE_MASTER=0`
- **Used on:** 24 chips (Chips 1-3 of each card; all other global IDs)
- **No PCIe:** Saves ~15-20% area vs Master
- **Functions:** 12 transformer layers, C2C ring forwarding

## Building

1. Open Quartus Prime Pro 24.3
2. File → Open Project → `master/fpga_lpu_master.qpf` or `slave/fpga_lpu_slave.qpf`
3. Platform Designer (QSYS):
   - Master: PCIe R-Tile + HBM2e
   - Slave: HBM2e only
4. Analysis & Synthesis → Fitter → Timing Analysis

## Dev Board Testing

Single DK-DEV-AGM039EA board:

| Test | Image | Connections | Verify |
|------|-------|-------------|--------|
| 1 | Master | PCIe to host | BAR0 access, DMA functional |
| 2 | Slave | C2C to Master board | Passthrough, config via C2C |

Multi-board (1 card = 4 boards):

| Board | Image | Role |
|-------|-------|------|
| 0 | Master | PCIe host, ring origin, layers 0-11 |
| 1 | Slave | Ring forward, layers 12-23 |
| 2 | Slave | Ring forward, layers 24-35 |
| 3 | Slave | Ring forward, layers 36-47 |
