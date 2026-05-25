"""
物理资源层 — FPGA 卡内/卡间硬件资源.

每种资源一个模块, 被流水线阶段调用:
  pcie.py       PCIe Gen5 x16 DMA (Host ↔ FPGA)
  sram.py       片上 SRAM 43MB (确定性权重常驻)
  hbm.py        HBM2e 控制器 32GB (expert 权重存储)
  dsp_array.py  DSP 计算阵列 fp4×FP8 / FP8×FP8
  ethernet.py   100GbE RDMA (卡间唯一通信信道)
"""
