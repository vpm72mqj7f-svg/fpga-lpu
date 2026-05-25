"""
100GbE RDMA 链路 — FPGA 卡间唯一通信信道.

Agilex 7 F-tile 集成 100GbE MAC, 通过外部交换机互联.
所有卡间流量统一走此链路:
  1. AllReduce (TP 组内 reduce-scatter + all-gather)
  2. Expert fetch (跨卡拉取 expert 结果)
  3. 权重分发 (启动时 broadcast)

拓扑: 所有 FPGA 通过交换机全互联, 任意两卡 1 跳 (SW延迟 ~1μs)
有效带宽: 100 Gbps / 8 × 0.85 (编码+协议开销) ≈ 10.6 GB/s

RTL 接口:
  - TX: 权重/activation → 分片 → RDMA write → MAC TX
  - RX: MAC RX → RDMA read → 重组 → 权重 buffer / 激活 buffer
"""

from dataclasses import dataclass, field
from ... import config


@dataclass
class RDMAOp:
    """一次 RDMA 操作记录."""
    op: str           # 'allreduce' | 'p2p' | 'broadcast'
    src: str          # 'card_N' or 'all'
    dst: str
    size_bytes: int
    hops: int         # 跳数: 同机箱=1 (经交换机), 跨服务器=2
    latency_us: float


class EthernetMAC:
    """100GbE RDMA 引擎 — FPGA 卡间的唯一互联信道.

    延迟模型: latency = hop_base + size_bytes / (bw_gbps * 1e9) * 1e6
                           = hop_base_us + size_bytes / bw_bytes_per_us
    """

    def __init__(self, num_cards: int = None):
        self.num_cards = num_cards or config.HW_FPGA_CHIP_COUNT

        # 100GbE 有效带宽
        self.bw_gbps = 100.0           # 线速
        self.encoding_overhead = 0.85  # 8b/10b + 以太网帧 + RDMA 协议
        self.eff_bw_gbps = self.bw_gbps * self.encoding_overhead  # 85 Gbps
        self._bw_bytes_per_us = self.eff_bw_gbps / 8 * 1e3  # Gbps → bytes/μs

        # 每跳延迟 (交换机转发)
        self.switch_latency_us = 1.0   # 数据中心交换机典型 <1μs
        self.mac_latency_us = 0.5      # MAC 层处理

        # 同机箱: FPGA → SW → FPGA = 1 跳 (经交换机)
        # 跨服务器: FPGA → SW1 → SW2 → FPGA = 2 跳 (经两层交换机)
        self._hops_same_server = 1
        self._hops_cross_server = 2
        # 默认跨服务器 (保守), TP 组内可用同机箱优化
        self.same_server = False

        self.buffer: list[RDMAOp] = []
        self.total_bytes = 0
        self.total_latency_us = 0.0

    def _hop_latency(self, hops: int) -> float:
        return hops * (self.mac_latency_us + self.switch_latency_us)

    def _data_latency(self, size_bytes: int) -> float:
        if size_bytes <= 0:
            return 0.0
        return size_bytes / self._bw_bytes_per_us

    def _transfer(self, size_bytes: int, hops: int) -> float:
        return self._hop_latency(hops) + self._data_latency(size_bytes)

    # ── AllReduce (Ring over Ethernet) ──

    def allreduce(self, data_bytes: int, group_size: int, same_server: bool = True) -> float:
        """Ring AllReduce 通过以太网 (流水线模型).

        reduce-scatter + all-gather 两阶段.
        流水线化: 总延迟 = 环遍历延迟 + 数据传输延迟.
          - 环遍历: (N-1) × hop × 2 (两阶段)
          - 数据:    2 × data_bytes / BW (每阶段传 data_bytes)
        """
        if group_size <= 1:
            return 0.0

        hops = self._hops_same_server if same_server else self._hops_cross_server
        # 流水线环遍历: 最后一跳到达 + 数据尾
        circuit = 2 * (group_size - 1) * self._hop_latency(hops)
        data_time = 2 * self._data_latency(data_bytes)
        lat = circuit + data_time

        self.buffer.append(RDMAOp(
            op='allreduce', src='all', dst='all',
            size_bytes=data_bytes, hops=hops, latency_us=lat))
        self.total_bytes += data_bytes * 2  # 两阶段各传 data_bytes
        self.total_latency_us += lat
        return lat

    # ── P2P (点对点 expert fetch) ──

    def p2p_fetch(self, src: int, dst: int, data_bytes: int,
                  same_server: bool = False) -> float:
        """点对点拉取 expert 结果.

        src: 持有 expert 权重的卡
        dst: 需要 expert 结果的卡
        """
        hops = self._hops_same_server if same_server else self._hops_cross_server
        lat = self._transfer(data_bytes, hops)

        self.buffer.append(RDMAOp(
            op='p2p', src=f'card_{src}', dst=f'card_{dst}',
            size_bytes=data_bytes, hops=hops, latency_us=lat))
        self.total_bytes += data_bytes
        self.total_latency_us += lat
        return lat

    # ── Broadcast (权重分发) ──

    def broadcast(self, src: int, data_bytes: int, group_size: int) -> float:
        """广播权重到 TP 组内所有卡. Ring broadcast: N-1 跳."""
        if group_size <= 1:
            return 0.0
        hops = self._hops_same_server  # 权重分发一般在同机箱
        lat = self._transfer(data_bytes, hops) * (group_size - 1)

        self.buffer.append(RDMAOp(
            op='broadcast', src=f'card_{src}', dst='all',
            size_bytes=data_bytes, hops=hops, latency_us=lat))
        self.total_bytes += data_bytes * (group_size - 1)
        self.total_latency_us += lat
        return lat

    @property
    def stats(self) -> dict:
        return {
            'effective_bw_gbps': self.eff_bw_gbps,
            'bw_bytes_per_us': self._bw_bytes_per_us,
            'total_ops': len(self.buffer),
            'total_bytes': self.total_bytes,
            'total_latency_us': self.total_latency_us,
            'allreduce_ops': sum(1 for o in self.buffer if o.op == 'allreduce'),
            'p2p_ops': sum(1 for o in self.buffer if o.op == 'p2p'),
            'broadcast_ops': sum(1 for o in self.buffer if o.op == 'broadcast'),
        }

    def reset(self):
        self.buffer.clear()
        self.total_bytes = 0
        self.total_latency_us = 0.0
