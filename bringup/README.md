# FPGA LPU Bring-Up Workspace

> S10 MX 开发板调试工程 — CPU Attention + FPGA FFN 混合推理

## 硬件平台

| 项目 | 规格 |
|------|------|
| **开发板** | Intel Stratix 10 MX FPGA Development Kit |
| **SKU** | DK-DEV-1SMX-H-A (Rev B) |
| **FPGA** | 1SM21BHU2F53E1VG (Production, Speed Grade -1) |
| **工艺** | 14nm, 2597-pin BGA, 1.0mm pitch |
| **逻辑资源** | 702,720 ALM, 2,073K LE, 2,810,880 Registers |
| **M20K** | 6,847 blocks (~17 MB on-chip) |
| **DSP** | 7,920 (18×19 multipliers) |
| **HBM2** | 8 GB (4GB×2 stacks), 8 channels×128-bit per stack, 2 Gbps/pin, ~512 GB/s aggregate |
| **DDR4 On-Board** | 8 GB (5× MT40A1G16KNR-075E, x72 ECC), 1333 MHz (DDR4-2666) |
| **DDR4 DIMM** | 288-pin socket, x72, 1333 MHz (DDR4/DDR-T) |
| **HiLo** | x72 connector, DDR4/QDR-IV support |
| **PCIe** | Gen3 ×16 Endpoint (gold fingers, banks 1C/1D/1E) + Gen3 ×16 Root Port (banks 4C/4D/4E) |
| **QSFP28** | 2× cages (banks 1F + 4F), 100G each |
| **收发器** | 96× H-Tile, 28.3 Gbps NRZ / 57.8 Gbps PAM4 |
| **配置** | AS×4 Fast (MSEL=001), 2 Gb QSPI Flash (MT25QU02G) |
| **USB-Blaster** | On-board USB-Blaster II (Micro-USB J15, MAX10 U24) |
| **JTAG** | 外部 JTAG 头 (J9), MAX10 系统控制器 + MAX10 电源管理 |

## 时钟架构

| 时钟源 | 信号名 | 频率 | FPGA Pin | 用途 |
|--------|--------|------|----------|------|
| Si5341A U16 | CLK_SYS_100M | 100 MHz LVDS | AU17/AU16 | 系统主时钟 |
| Si5341A U16 | REFCLK_PCIE_EP | 100 MHz LVDS | AW43/AW42 | PCIe EP 参考时钟 |
| Si5341A U16 | REFCLK_PCIE_RT | 100 MHz LVDS | AW9/AW10 | PCIe RP 参考时钟 |
| Si5341A U16 | CLK_UIB0 | 100 MHz LVDS | AR26/AP26 | HBM2 UIB0 时钟 |
| Si5341A U16 | CLK_UIB1 | 100 MHz LVDS | P27/R27 | HBM2 UIB1 时钟 |
| Si5341A U16 | CLK_ESRAM0 | 100 MHz LVDS | AU31/AU32 | HBM2 ESRAM0 时钟 |
| Si5341A U16 | CLK_ESRAM1 | 100 MHz LVDS | V31/U31 | HBM2 ESRAM1 时钟 |
| Si5341A U16 | REFCLK_ZQSFP0 | 644.53125 MHz LVDS | AJ43/AJ42 | QSFP0 参考时钟 |
| Si5341A U16 | REFCLK_ZQSFP1 | 644.53125 MHz LVDS | AJ9/AJ10 | QSFP1 参考时钟 |
| Si5338A U18 | CLK_SYS_50M | 50 MHz LVDS | BE17/BD17 | FPGA 时钟 |
| Si5338A U18 | CLK_CORE_BAK | 100 MHz LVDS | AT13/AU13 | FPGA Core 备用时钟 |
| Si5338A U18 | REFCLK_PCIE_EP1 | 100 MHz LVDS | BA43/BA42 | PCIe 收发器时钟 |
| Si5338B U19 | CLK_DDR4_COMP | 133.333 MHz LVDS | A42/B41 | 板载 DDR4 时钟 |
| Si5338B U19 | CLK_DDR4_DIMM | 133.333 MHz LVDS | B18/C18 | DIMM DDR4 时钟 |
| Si5338B U19 | CLK_HILO_MEM | 133.333 MHz LVDS | AW31/AY31 | HiLo 存储器时钟 |
| Si510 U17 | S10_OSC_CLK_1 | 125 MHz LVCMOS | AR35 | 配置时钟 |

## 用户 I/O

| 外设 | 信号名 | FPGA Pin | 说明 |
|------|--------|----------|------|
| LED0 (D7) | S10_LED0 | BG12 | 用户 LED, 低有效 |
| LED1 (D8) | S10_LED1 | BF12 | 用户 LED, 低有效 |
| LED2 (D9) | S10_LED2 | BG11 | 用户 LED, 低有效 |
| LED3 (D10) | S10_LED3 | BH11 | 用户 LED, 低有效 |
| CPU Reset (S10) | CPU_RESETn | BL14 | 系统复位, 低有效 |
| PCIe PERST0 (S1) | S10_PCIe_PERST_0 | AH39 | PCIe EP 复位 |
| PCIe PERST1 (S11) | S10_PCIe_PERST_1 | BL10 | PCIe RP 复位 |
| CONFIG_DONE (D14) | S10_CONF_DONE | AY39 | 配置完成指示灯 |
| CvP_DONE (D16) | S10_CVP_CONFDONE | BC42 | CvP 完成指示灯 |

## 电源

| 电源轨 | 电压 | 最大电流 | 说明 |
|--------|------|----------|------|
| VCC/VCCP | 0.85V | 132A | FPGA 核心 + 外设 |
| VCCERAM | 0.9V | 4.6A | 嵌入式存储器 |
| VCCIO_UIB | 1.2V | 12A | HBM2 UIB |
| VCCM | 2.5V | 2.6A | HBM2 存储 |
| VCCT_GXB | 1.12V | 2.1A | 收发器发送 |
| VCCR_GXB | 1.12V | 4.0A×2 | 收发器接收 |
| VCCIO | 1.8V | 11A | I/O + PLL + SDM |
| VCCIO_DDR4 | 1.2V | — | DDR4 I/O |
| 总功率 | — | ≤192W | 强制风冷 (22 CFM) |

上电时序 (MAX10 控制):
- Group 1: VCC, VCCP, VCCERAM, VCCPLLDIG_SDM, VCCR_GXB, VCCT_GXB
- Group 2: VCCPT, VCCBAT, VCCIO_SDM, VCCIO_1.8V, VCCH_GXB, VCCA_PLL
- Group 3: VCCIO_1.2_DDR4, VCCM, VCCIO_UIB, VCCIO_SDM
- 下电顺序: Group 3 → 2 → 1 (反向)

## 项目结构

```
bringup/
├── bringup.qpf              Quartus Prime Pro 项目文件
├── bringup.qsf              Quartus 设置 (器件/SmartVID/引脚/时钟)
├── bringup.sdc              SDC 时序约束
├── rtl/
│   ├── bringup_top.sv       顶层模块 (时钟/复位/FFN/调试)
│   ├── s10_ffn_engine.sv    FFN 引擎 (CPU→FFN→CPU)
│   ├── pll_controller.sv    PLL 控制器 (100M→500M/250M)
│   ├── reset_controller.sv  多域复位同步器
│   └── led_controller.sv    LED 调试状态显示
├── sim/                     仿真目录 (待添加)
├── scripts/
│   ├── build_quartus.tcl    Quartus 编译脚本
│   └── program_board.tcl    JTAG 烧录脚本
├── logs/                    实验日志
└── doc/                     板卡文档
    └── stratix10MX_1sm21bhu2f53_fpga_revB_v18.1.1b263_v1.0/
        ├── board_design_files/   原理图/PCB/BOM
        ├── documents/            用户手册 (UG-20151)
        └── examples/             BTS/工厂恢复
```

## 三步演进

```
Phase 1: V2 Lite (16B, FP8)  → S10 MX 验证 FFN-only 架构
Phase 2: V4 Flash (284B, FP8) → 扩展到更大模型
Phase 3: V4 Pro (1.6T, FP4)  → 全规模目标
```

## 快速开始

### 1. 环境准备

- Quartus Prime Pro 18.1.1+ (Stratix 10 MX 仅支持 Pro Edition)
- Intel FPGA Download Cable II 驱动
- Micro-USB 线连接 J15
- 12V ATX 辅助电源连接 J11

### 2. 编译

```tcl
# 在 Quartus shell 中运行:
quartus_sh -t scripts/build_quartus.tcl
```

### 3. 烧录

```tcl
quartus_pgm -t scripts/program_board.tcl
```

### 4. 验证

烧录完成后观察:
- **D4 (蓝色)** — 电源指示, 常亮
- **D14 (绿色)** — CONFIG_DONE, 配置成功常亮
- **D7 (LED0)** — PLL 锁定心跳 (2 Hz 闪烁)
- **D8 (LED1)** — FFN 忙指示
- **D9 (LED2)** — FFN 完成脉冲
- **D10 (LED3)** — 通过/失败 指示 (灭=通过, 常亮=失败)

## SmartVID 注意事项

生产设备 (1SM21BHU2F53**E1**VG) 必须启用 SmartVID, QSF 中已配置:
```
VID_OPERATION_MODE = "PMBUS MASTER"
PWRMGT_SLAVE_DEVICE0_ADDRESS = 47
PWRMGT_BUS_SPEED_MODE = "100 KHZ"
```
缺少这些设置将导致编程错误:
`Error(19192): File .sof is incomplete - Power management settings`

## 参考文档

- UG-20151: Intel Stratix 10 MX FPGA Development Kit User Guide (2020.06.15)
- 板卡原理图: `doc/.../schematic/s10mx_pcie_devkit.pdf`
- Intel Stratix 10 MX Device Overview
- AN 692: Power Sequencing Considerations for Stratix 10
