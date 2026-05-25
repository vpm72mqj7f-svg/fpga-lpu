"""
PCIe DMA 仿真 — host ↔ FPGA 数据传输。

模拟: PCIe Gen5 x16, ~32 GB/s 单向有效带宽, ~2 μs 启动延迟。
数据以 "packet" 为单位传输, 每个 packet 有固定开销 + 按带宽计时的传输延迟。
"""

from dataclasses import dataclass, field
from .. import config


@dataclass
class PCIeTransfer:
    """一次 DMA 传输的记录."""
    direction: str   # 'h2d' (host→device) or 'd2h' (device→host)
    size_bytes: int
    latency_us: float


class PCIeDriver:
    """FPGA 侧 PCIe DMA 引擎.

    vLLM (host) ←→ FPGA (device) 之间的唯一数据通道.
    每次传输 = 固定启动延迟 + size_bytes / 有效带宽.
    """

    def __init__(self):
        self._bw_bytes_per_us = config.HW_PCIE_BW_GBPS * 1e3  # GB/s → bytes/μs
        self._latency_us = config.HW_PCIE_LATENCY_US
        self.transfers: list[PCIeTransfer] = []
        self.total_bytes_h2d = 0
        self.total_bytes_d2h = 0
        self.total_latency_us = 0.0

    def transfer_latency(self, size_bytes: int) -> float:
        """计算传输 size_bytes 所需的延迟 (μs)."""
        if size_bytes <= 0:
            return 0.0
        return self._latency_us + size_bytes / self._bw_bytes_per_us

    def send_to_fpga(self, data_size: int, tag: str = "") -> float:
        """Host → FPGA: 发送数据 (模拟). 返回延迟 μs."""
        lat = self.transfer_latency(data_size)
        self.transfers.append(PCIeTransfer('h2d', data_size, lat))
        self.total_bytes_h2d += data_size
        self.total_latency_us += lat
        return lat

    def recv_from_fpga(self, data_size: int, tag: str = "") -> float:
        """FPGA → Host: 接收数据 (模拟). 返回延迟 μs."""
        lat = self.transfer_latency(data_size)
        self.transfers.append(PCIeTransfer('d2h', data_size, lat))
        self.total_bytes_d2h += data_size
        self.total_latency_us += lat
        return lat

    @property
    def stats(self) -> dict:
        return {
            'total_h2d_bytes': self.total_bytes_h2d,
            'total_d2h_bytes': self.total_bytes_d2h,
            'total_latency_us': self.total_latency_us,
            'num_transfers': len(self.transfers),
        }

    def reset(self):
        self.transfers.clear()
        self.total_bytes_h2d = 0
        self.total_bytes_d2h = 0
        self.total_latency_us = 0.0
