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


# ============================================================================
# Disaggregated KV Transfer Model
# ============================================================================

# KV bytes per token: K (kv_latent 512 + rope 64 = 576) + V (kv_latent 512)
# Total = 1088 bytes FP8.  Rounds to 1152 for conservative headroom.
_KV_RAW_BYTES_PER_TOKEN = 1088
_KV_CONSERVATIVE_BYTES_PER_TOKEN = 1152

# Effective per-link bandwidth with efficiency factor.
# C2C intra-card: 128 GB/s raw, 85% efficiency.
# PCIe P2P cross-server: 64 GB/s raw, 85% efficiency (PCIe protocol overhead).
C2C_EFF_GBPS = 128 * 0.85   # 108.8 GB/s
P2P_EFF_GBPS = 64 * 0.85    #  54.4 GB/s


def kv_disaggregated_transfer_time_us(num_tokens: int,
                                       kv_bytes_per_token: int = _KV_CONSERVATIVE_BYTES_PER_TOKEN,
                                       num_chips: int = 32) -> float:
    """Estimate KV transfer latency in a disaggregated (P+D) deployment.

    Transfer path: Prefill chip -> C2C -> PCIe bridge -> PCIe P2P ->
                   Decode PCIe bridge -> C2C -> Decode chip.

    Bottleneck: PCIe P2P cross-server link at 64 GB/s (PCIe 5.0 x16).
    Per-token KV bytes are small (1152 B), so latency is the bottleneck
    for small batches; bandwidth dominates for large batches.

    Model components:
      1. Intra-card C2C (chip <-> PCIe bridge): 50 ns hop + serdes
      2. PCIe P2P cross-server: 400 ns fixed + serdes
      Total = 2 * C2C + PCIe P2P

    Args:
        num_tokens: total KV tokens to transfer (batch prompt length)
        kv_bytes_per_token: bytes per token (default 1152, conservative)
        num_chips: chips participating in transfer (default all 32)

    Returns:
        Transfer latency in microseconds.
    """
    total_payload = num_tokens * kv_bytes_per_token

    # C2C intra-card (per direction): 1 hop + serdes
    c2c_serdes_ns = total_payload * 8 / C2C_EFF_GBPS
    c2c_one_way_ns = C2C_HOP_LATENCY_NS + c2c_serdes_ns

    # PCIe P2P cross-server
    p2p_serdes_ns = total_payload * 8 / P2P_EFF_GBPS
    p2p_ns = PCIE_P2P_LATENCY_NS + p2p_serdes_ns

    # Total: prefill C2C (chip->PCIe) + PCIe P2P + decode C2C (PCIe->chip)
    total_ns = 2 * c2c_one_way_ns + p2p_ns
    return total_ns / 1000.0


def kv_transfer_us_per_token(batch_size_tokens: int = 512,
                              kv_bytes_per_token: int = _KV_CONSERVATIVE_BYTES_PER_TOKEN) -> float:
    """Effective per-token KV transfer cost (amortized over batch).

    For batch_size_tokens=1 the fixed overheads dominate (~0.79 us/token).
    For large batches the bandwidth limit is ~0.02 us/token (PCIe 54.4 GB/s).

    Typical disaggregated prefill batch (P=512): ~0.022 us/token.

    Returns:
        Per-token transfer cost in microseconds.
    """
    total = kv_disaggregated_transfer_time_us(batch_size_tokens, kv_bytes_per_token)
    return total / max(1, batch_size_tokens)
