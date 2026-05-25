"""
fpga_arch/interconnect.py — C2C Dual Ring + PCIe P2P fabric.

Extracted from fpga_4chip_pipeline.py:172-180 (C2CMessageType),
295-353 (C2CDualRing), 359-376 (PCIeFabric).
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
from enum import Enum, auto
from collections import defaultdict
import math

from .config import (
    CHIPS_PER_CARD, C2C_LINK_BW_GBPS, C2C_HOP_LATENCY_NS,
    C2C_FRAME_OVERHEAD_B, C2C_MAX_PAYLOAD_B,
    C2C_DISPATCH_LATENCY_NS, C2C_REDUCE_LATENCY_NS, C2C_FWD_LATENCY_NS,
    C2C_MSG_DISPATCH_B, PCIE_P2P_BW_GBPS, PCIE_P2P_LATENCY_NS,
)
from .chip import FPGAChip


class C2CMessageType(Enum):
    MOE_DISPATCH     = 0x1
    MOE_REDUCE       = 0x2
    PIPELINE_FWD     = 0x3
    PCIE_PROXY       = 0x4
    CREDIT_UPDATE    = 0x5
    WEIGHT_BROADCAST = 0x6
    HEARTBEAT        = 0x7


@dataclass
class C2CMessage:
    """A single C2C message routed through the ring."""
    msg_type: C2CMessageType
    src_chip: int
    dst_chip: int
    payload_bytes: int
    hops: int = 1
    cross_card: bool = False


class C2CDualRing:
    """C2C Dual Ring interconnect within a 4-chip card.

    Ring A topology: C0-C1-C2-C3-C0 (bidirectional per hop)
    Ring B topology: C0-C2, C1-C3 (redundant cross-links)

    Dijkstra shortest-path routing on Ring A (static, compile-time).
    """

    RING_A_LINKS = [(0, 1), (1, 2), (2, 3), (3, 0)]
    RING_B_LINKS = [(0, 2), (1, 3)]

    def __init__(self, card_id: int, chips: List[FPGAChip]):
        self.card_id = card_id
        self.chips: Dict[int, FPGAChip] = {c.chip_id: c for c in chips}
        self._build_routing_table()
        self.link_usage_gbps: Dict[Tuple[int, int], float] = defaultdict(float)
        self.messages_sent: int = 0
        self.total_bytes_sent: int = 0

    def _build_routing_table(self):
        """Dijkstra shortest path on Ring A."""
        self.routes: Dict[Tuple[int, int], Tuple[int, int]] = {}

        for src in range(4):
            for dst in range(4):
                if src == dst:
                    continue
                dist_clockwise = (dst - src) % 4
                dist_counter = (src - dst) % 4
                if dist_clockwise <= dist_counter:
                    hops = dist_clockwise
                    next_hop = (src + 1) % 4 if hops > 0 else dst
                else:
                    hops = dist_counter
                    next_hop = (src - 1) % 4 if hops > 0 else dst
                self.routes[(src, dst)] = (next_hop, hops)

    def route(self, src_chip: int, dst_chip: int) -> Tuple[int, int]:
        """Return (next_hop_chip_id, num_hops) for src→dst on Ring A."""
        return self.routes.get((src_chip, dst_chip), (dst_chip, 1))

    def transfer_time_us(self, src_chip: int, dst_chip: int, payload_bytes: int) -> float:
        """C2C transfer time within the same card (microseconds)."""
        _, hops = self.route(src_chip, dst_chip)
        frames = max(1, math.ceil(payload_bytes / C2C_MAX_PAYLOAD_B))
        avg_frame_bytes = payload_bytes / frames + C2C_FRAME_OVERHEAD_B
        serdes_ns = frames * avg_frame_bytes * 8 / C2C_LINK_BW_GBPS
        hop_ns = C2C_HOP_LATENCY_NS * hops
        return (serdes_ns + hop_ns) / 1000.0

    def record_transfer(self, src_chip: int, dst_chip: int, payload_bytes: int):
        """Record a C2C transfer for bandwidth accounting."""
        link = (min(src_chip, dst_chip), max(src_chip, dst_chip))
        bw_gbps = payload_bytes * 8 / (C2C_DISPATCH_LATENCY_NS / 1e9) / 1e9
        self.link_usage_gbps[link] += bw_gbps
        self.messages_sent += 1
        self.total_bytes_sent += payload_bytes

    @property
    def peak_link_gbps(self) -> float:
        return max(self.link_usage_gbps.values()) if self.link_usage_gbps else 0.0

    @property
    def avg_link_usage_gbps(self) -> float:
        return sum(self.link_usage_gbps.values()) / max(1, len(self.link_usage_gbps))


class PCIeFabric:
    """PCIe 5.0 P2P fabric connecting 8 cards via host backplane."""

    def __init__(self):
        self.transfer_count: int = 0
        self.total_bytes: int = 0

    def transfer_time_us(self, src_card: int, dst_card: int, payload_bytes: int) -> float:
        """
        Cross-card transfer time (microseconds).

        Same CPU socket: PCIe P2P direct (~400 ns)
        Cross CPU socket: PCIe P2P via UPI (~400 ns + UPI overhead)
        """
        self.transfer_count += 1
        self.total_bytes += payload_bytes
        serdes_ns = payload_bytes * 8 / PCIE_P2P_BW_GBPS
        return max(serdes_ns, PCIE_P2P_LATENCY_NS) / 1000.0
