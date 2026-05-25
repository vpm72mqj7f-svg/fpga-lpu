"""
fpga_layer — FPGA 硬件侧架构.

目录结构:
  phys/       物理资源层 (PCIe, SRAM, HBM, DSP, Ethernet)
  stages/     流水线节拍层 (9个 Stage, token 逐拍推进)
  pipeline.py 流水线控制器 (串 stages, 管理 overlap)
  tp_group.py TP 组协调器 (以太网 AllReduce + 跨组 expert fetch)
"""
