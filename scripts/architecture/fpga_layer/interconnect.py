"""
卡间互联 — FPGA 片间 SerDes 通信 + AllReduce 延迟模型.

Agilex 7 F-tile: 最高 58 Gbps/lane, 卡间通常用 4-8 lane 绑定.
有效带宽: ~200 Gbps/lane-pair (双向) ≈ 25 GB/s.

通信模式:
  1. AllReduce (ring): 每层 TP 结果汇总, 数据量小 (KB 级)
  2. Expert Result Share: 非本地专家结果点对点传输
  3. Weight Transfer: Expert cache miss 时全量传输 (MB 级, 罕见)
"""

from dataclasses import dataclass, field
from .. import config


@dataclass
class TransferRecord:
    src: int
    dst: int
    size_bytes: int
    latency_us: float
    op: str  # 'allreduce', 'p2p', 'broadcast'


class Interconnect:
    """卡间通信网络.

    拓扑: ring (默认), 也可配置为 tree.
    每张卡通过 SerDes 连接相邻卡, 形成环.
    AllReduce 用 ring-allreduce 算法: N-1 步, 每步传 data/N 数据.
    """

    def __init__(self, num_cards: int = None):
        self.num_cards = num_cards or config.HW_FPGA_CARD_COUNT
        # 每卡到相邻卡的有效带宽 (GB/s)
        self.link_bw_gbps = 25.0  # ~200 Gbps 有效
        self._link_bytes_per_us = self.link_bw_gbps * 1e3  # GB/s → bytes/μs

        # 每次传输的固定开销 (链路建立 + SerDes 对齐)
        self._hop_latency_us = 0.15  # ~150 ns per hop

        # 统计
        self.transfers: list[TransferRecord] = []
        self.total_bytes = 0
        self.total_latency_us = 0.0

    def _transfer_time(self, size_bytes: int, hops: int = 1) -> float:
        """单次传输延迟 = 跳数 × (固定开销 + 数据传输)."""
        if size_bytes <= 0:
            return 0.0
        data_time = size_bytes / self._link_bytes_per_us
        return hops * (self._hop_latency_us + data_time)

    def allreduce(self, data_bytes: int, group_size: int) -> float:
        """Ring AllReduce within a TP group.

        算法: ring-allreduce, 每个 rank 传 (group_size-1)/group_size 数据.
        总步数: 2 × (group_size - 1) (reduce-scatter + all-gather).
        每步数据: data_bytes / group_size.
        """
        if group_size <= 1:
            return 0.0
        steps = 2 * (group_size - 1)
        per_step_bytes = data_bytes // group_size
        lat = steps * self._transfer_time(per_step_bytes, hops=1)

        self.transfers.append(TransferRecord(
            src=-1, dst=-1, size_bytes=data_bytes * steps // group_size,
            latency_us=lat, op='allreduce'))
        self.total_bytes += data_bytes * steps // group_size
        self.total_latency_us += lat
        return lat

    def p2p_send(self, src: int, dst: int, data_bytes: int) -> float:
        """点对点传输 (卡间 expert result sharing).

        环形拓扑: 跳数 = min(|dst-src|, num_cards - |dst-src|).
        """
        hops = min(abs(dst - src), self.num_cards - abs(dst - src))
        lat = self._transfer_time(data_bytes, hops)

        self.transfers.append(TransferRecord(
            src=src, dst=dst, size_bytes=data_bytes, latency_us=lat, op='p2p'))
        self.total_bytes += data_bytes
        self.total_latency_us += lat
        return lat

    def broadcast(self, src: int, data_bytes: int, group_size: int) -> float:
        """广播 (权重分发). Ring broadcast: group_size-1 hops."""
        if group_size <= 1:
            return 0.0
        lat = self._transfer_time(data_bytes, hops=group_size - 1)
        self.total_bytes += data_bytes * (group_size - 1)
        self.total_latency_us += lat
        return lat

    @property
    def stats(self) -> dict:
        return {
            'total_transfers': len(self.transfers),
            'total_bytes': self.total_bytes,
            'total_latency_us': self.total_latency_us,
        }

    def reset(self):
        self.transfers.clear()
        self.total_bytes = 0
        self.total_latency_us = 0.0
