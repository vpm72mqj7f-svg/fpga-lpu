# Phase 1 Stage 2 — PCIe EP + Partition Flow PR

> **基线**: Stage 1 SOF (12MB, DSP=128, PCIe ATX lock, HBM2 TG pass)  
> **目标**: PCIe 枚举 → 权重下载 → FFN PR 分区 → 增量迭代

## Stage 2 目标

```
Stage 1 (Done)                              Stage 2 (This)
┌──────────┐                                ┌──────────┐
│ PCIe EP  │ XCVR loopback   ──────────────►│ PCIe EP  │ ARM lspci 可见
│ HBM2     │ calibration ✓   ──────────────►│ HBM2     │ 权重可写
│ FFN      │ DSP=128 (3%)    ──────────────►│ FFN PR   │ 独立分区,部分重配
│ SOF      │ 12MB            ──────────────►│ SOF+PRBF │ .sof + .rbf
│ Build    │ 60min flat      ──────────────►│ Build    │ 10min FFN-only
└──────────┘                                └──────────┘
```

## 里程碑

### M1: PCIe 枚举 (P0)
```
ARM $ lspci -vv → Intel Stratix 10 Memory Controller [1172:E001]
```
- [ ] pcie_ep.v 补 BAR0 port（done）
- [ ] 顶层连接 pcie_ep → v2_lite_full_top.sv（done）  
- [ ] check_pins.sh pass
- [ ] 全编 → SOF → JTAG 烧写
- [ ] ARM 验证 `lspci` + `lspci -vv`

### M2: 权重下载 (P0)

```
ARM mmap BAR0 → 写 WT_DATA_PORT → AXI4 → HBM2 → 读回验证
```
- [ ] pcie_hbm_weight_writer.sv 全同步版集成到顶层
- [ ] 顶层: pcie_ep BAR0 → writer AVMM → AXI4 → ed_synth write port
- [ ] ARM 侧 UIO driver 写 1KB → 读回对比
- [ ] ISP 读 HBM2 TG 状态确认数据到达

### M3: FFN PR 分区 (P1)

```
Root Partition (POST_FIT, 不重配):
  ├── PCIe HIP + HSSI    ← 不掉线
  ├── HBM2 UIB           ← 校准保持
  └── ed_synth + ISP     ← 调试保留

FFN Partition (SOURCE, 可部分重配):
  └── u_ffn: v2_lite_ffn_engine
      ├── systolic_array ×2
      ├── silu_activation
      └── hbm2_weight_reader
```
- [ ] QSF 分区配置（只 u_ffn 为 SOURCE）
- [ ] FFN 规模放大：DSP=128 → ~2000 (50%)
- [ ] 首次全编 → 导出 QDB
- [ ] 修改 FFN → 增量编译 → 生成 .rbf
- [ ] PCIe `pcie_reconfig()` 下载 .rbf

### M4: 验证链 (P1)

```
[Lint 30s] → [Sim 2min] → [check_pins 30s] → [quartus_syn 3min] → [full 10-50min]
  ✅            ❌ (stage 2)       ✅                    ✅                  ✅
```
- [ ] FFN engine Verilator testbench
- [ ] 黄金模型 Python 对比
- [ ] pre-build checklist 自动化

## 文件变更清单

| File | M1 | M2 | M3 | Description |
|------|:--:|:--:|:--:|-------------|
| pcie_ep/synth/pcie_ep.v | ● | | | BAR0 port patch |
| v2_lite_full_top.sv | ● | ● | ● | 顶层连接 |
| pcie_hbm_weight_writer.sv | | ● | | 权重下载引擎 |
| systolic_array.sv | | | ● | DSP 规模放大 |
| v2_lite_ffn_engine.sv | | | ● | 参数对齐 |
| v2_lite_full.qsf | | | ● | 分区配置 |
| check_pins.sh | ● | | | 预检脚本 |
| tb_ffn_engine.cpp | | | ● | Verilator sim |

## 时间估计

| 阶段 | 首次全编 | 增量编译 |
|------|---------|---------|
| M1 (PCIe EP 集成) | 60min | — |
| M2 (权重写入) | 60min | — |
| M3 (FFN PR) | 60min | **<10min** |
| M4 (仿真) | — | 2min |
