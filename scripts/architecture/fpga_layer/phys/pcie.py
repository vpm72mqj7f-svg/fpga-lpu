"""
PCIe Gen5 x16 DMA — Host ↔ FPGA 唯一数据入口.

Agilex 7 F-tile 提供 PCIe Gen5 x16 endpoint.
所有 Host→FPGA / FPGA→Host 数据流经此通道:
  - Activation 传入 (每层输入)
  - Activation 传出 (每层输出)
  - KV cache swap (CPU↔FPGA HBM, 调度触发)
  - 初始权重加载 (启动时, 不计入推理延迟)

带宽: 32 GB/s 单向有效 (128 GT/s × 16 lane / 8 × 0.8 协议效率)
固定开销: 2 μs DMA 启动延迟

RTL 接口:
  - AXI-Stream TX/RX → PCIe IP core → DMA engine
  - 每笔传输: 启动(2μs) + ceil(size / MTU) × MTU / BW
"""

from dataclasses import dataclass, field
from ... import config


@dataclass
class PCIeOp:
    """一次 PCIe DMA 记录."""
    dir: str         # 'h2d' (host→device) or 'd2h' (device→host)
    size_bytes: int
    tag: str
    latency_us: float


class PCIeDMA:
    """PCIe DMA 引擎 — Host ↔ FPGA 的数据管道."""

    def __init__(self):
        self.bw_gbps = config.HW_PCIE_BW_GBPS        # 32 GB/s
        self._bw_bytes_per_us = self.bw_gbps * 1e3    # GB/s → bytes/μs
        self.startup_us = config.HW_PCIE_LATENCY_US   # 2 μs
        self.mtu_bytes = config.HW_PCIE_MTU_KB * 1024  # 256 KB

        self.ops: list[PCIeOp] = []
        self.total_h2d = 0
        self.total_d2h = 0
        self.total_latency_us = 0.0

    def latency(self, size_bytes: int) -> float:
        """单次传输延迟 = 启动 + 传输."""
        if size_bytes <= 0:
            return 0.0
        return self.startup_us + size_bytes / self._bw_bytes_per_us

    def send(self, size_bytes: int, tag: str = "") -> float:
        """Host → FPGA (e.g. activation 下发)."""
        lat = self.latency(size_bytes)
        self.ops.append(PCIeOp(dir='h2d', size_bytes=size_bytes, tag=tag, latency_us=lat))
        self.total_h2d += size_bytes
        self.total_latency_us += lat
        return lat

    def recv(self, size_bytes: int, tag: str = "") -> float:
        """FPGA → Host (e.g. hidden state 返回)."""
        lat = self.latency(size_bytes)
        self.ops.append(PCIeOp(dir='d2h', size_bytes=size_bytes, tag=tag, latency_us=lat))
        self.total_d2h += size_bytes
        self.total_latency_us += lat
        return lat

    @property
    def stats(self) -> dict:
        return {
            'bw_gbps': self.bw_gbps,
            'total_h2d_bytes': self.total_h2d,
            'total_d2h_bytes': self.total_d2h,
            'total_latency_us': self.total_latency_us,
            'num_ops': len(self.ops),
        }

    def reset(self):
        self.ops.clear()
        self.total_h2d = 0
        self.total_d2h = 0
        self.total_latency_us = 0.0
