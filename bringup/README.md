# FPGA LPU Bring-Up Workspace

> 工程迭代目录：FPGA 板子调试 + ARM 服务器跑分

## 硬件

| 平台 | 用途 | 状态 |
|------|------|:---:|
| **ARM Server** (172.16.95.198) | CPU Attention + GPU FFN 基准 | ✅ 就绪 |
| **S10 MX Dev Kit** (DK-DEV-1SMx-H-A) | FPGA FFN 验证 | ⏳ 待板子到 |
| **Quartus Server** | 综合 + 布局布线 | ⏳ Day 2 |

## 三步演进

```
Phase 1: V2 Lite (16B, FP8)  → S10 MX 验证 FFN-only 架构
Phase 2: V4 Flash (284B, FP8) → 扩展到更大模型
Phase 3: V4 Pro (1.6T, FP4)  → 全规模目标
```

## 目录结构

```
bringup/
  rtl/          S10 MX FFN-only RTL (针对板子定制)
  sim/          仿真 testbench
  scripts/      ARM 服务器跑分脚本
  logs/         实验结果日志
```

## S10 MX 关键参数

```
芯片:     1SM21BHU2F53E1VG (H-Tile)
ALM:      702,720
M20K:     6,847 blocks
HBM2:     8 GB / 256 GB/s
PCIe:     Gen3 x16 (~16 GB/s)
DSP:      ~5,760
工艺:     14nm
```
