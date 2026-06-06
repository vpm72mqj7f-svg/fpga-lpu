"""
DeepSeek V4 Pro FPGA 推理 — 软硬架构原型 (Python).

vLLM Layer (软件栈):  API Server → Scheduler → KV Cache → Model Runner
         ↕ PCIe DMA
FPGA Layer (硬件栈):  Runtime → DSP Kernels + HBM Manager + SRAM Cache

所有模块只定义接口和数据流, RTL 实现留空。
"""

import warnings
warnings.warn("architecture/ is deprecated, use fpga_arch/", DeprecationWarning, stacklevel=2)

from . import config
