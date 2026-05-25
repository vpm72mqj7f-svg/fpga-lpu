"""
fpga_4chip_pipeline.py — 4-FPGA Chip Distributed Pipeline Model
===============================================================

Models the FULL 4-chip per card architecture:
  - 8 cards × 4 AGM 039-F chips = 32 chips
  - C2C SerDes Dual Ring (Ring A: C0-C1-C2-C3, Ring B: C0-C2, C1-C3)
  - MoE cross-chip dispatch/reduce (384 experts, 12 per chip)
  - Pipeline forwarding (hidden state passing between layer-hosting chips)
  - HBM + SRAM weight caching with hit-rate model
  - Stage-by-stage timing breakdown

This is fundamentally different from the old single-chip-per-card model
— the 4-chip pipeline introduces C2C communication stages and MoE
scatter-gather across the Dual Ring.

Architecture Reference: docs/fpga_inference_cluster_proposal.md
  - §2.1: 8-card × 4-chip physical layout
  - §4.1: 32-chip resource allocation table
  - §4.2: Per-layer MAC breakdown
  - §4.4.1: SRAM cache allocation & hit-rate model
  - §6.2: C2C Dual Ring topology
  - §6.3: C2C protocol (frame format, message types, credit flow)
  - §6.5: Communication bandwidth analysis

Usage:
  python scripts/simulation/fpga_4chip_pipeline.py           # Default analysis
  python scripts/simulation/fpga_4chip_pipeline.py --verbose  # Stage-by-stage
  python scripts/simulation/fpga_4chip_pipeline.py --batch 4  # Batch decode
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum, auto
from collections import defaultdict
import sys
import argparse


# ============================================================================
# Constants & Parameters (from proposal §4.1, §4.3, §4.4.1)
# ============================================================================

# ── Chip-level ──
DSP_COUNT      = 12_300          # AGM 039-F
DSP_FREQ_MHZ   = 450             # MHz
DSP_MAC_PER_CYCLE = 2            # fp4 × fp8 mode
DSP_TMACS      = DSP_COUNT * DSP_FREQ_MHZ * DSP_MAC_PER_CYCLE / 1e6   # 11.07 TMACs/s
HBM_SIZE_GB    = 32              # GB per chip
HBM_BW_GBPS    = 920             # GB/s effective (32 pseudo-channels)
SRAM_M20K_MB   = 29.2            # usable M20K (75% of 38.9 MB)
SRAM_MLAB_MB   = 3.3             # usable MLAB (80% of 4.1 MB)
SRAM_TOTAL_MB  = SRAM_M20K_MB + SRAM_MLAB_MB  # 32.5 MB

# ── Cluster topology ──
NUM_CARDS       = 8
CHIPS_PER_CARD  = 4
TOTAL_CHIPS     = NUM_CARDS * CHIPS_PER_CARD  # 32
NUM_LAYERS      = 61
NUM_EXPERTS     = 384
EXPERTS_PER_CHIP = NUM_EXPERTS // TOTAL_CHIPS  # 12

# ── Model dimensions (DeepSeek V4 Pro) ──
HIDDEN_SIZE         = 7168
INTERMEDIATE_SIZE   = 3072
NUM_ATTN_HEADS      = 128
KV_LORA_RANK        = 512
Q_LORA_RANK         = 1536
O_LORA_RANK         = 1024
QK_ROPE_HEAD_DIM    = 64
QK_NOPE_HEAD_DIM    = 448
V_HEAD_DIM          = 128
NUM_EXPERTS_PER_TOK = 6
SLIDING_WINDOW      = 128

# ── Weight sizes (fp4 unless noted) ──
# Attention weights per layer (TP-shared across card's 4 chips)
ATTN_WEIGHT_MB = {
    'kv_a_down':  KV_LORA_RANK * HIDDEN_SIZE / 2 / (1024*1024),       # 1.75 MB
    'kv_a_up':    NUM_ATTN_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) * KV_LORA_RANK / 2 / (1024*1024),  # 14.25 MB
    'kv_a_rope':  QK_ROPE_HEAD_DIM * HIDDEN_SIZE / 2 / (1024*1024),   # 0.22 MB
    'q_a_down':   Q_LORA_RANK * HIDDEN_SIZE / 2 / (1024*1024),        # 5.25 MB
    'q_a_up':     NUM_ATTN_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) * Q_LORA_RANK / 2 / (1024*1024),  # 48.0 MB
    'o_down':     O_LORA_RANK * NUM_ATTN_HEADS * V_HEAD_DIM / 2 / (1024*1024),  # 12.5 MB
    'o_up':       HIDDEN_SIZE * O_LORA_RANK / 2 / (1024*1024),        # 7.0 MB
}
# Total per-layer attention: ~88.97 MB fp4 → per chip with TP≈2 (card-level TP)
# Each chip handles ~1/2 of attention heads → ~44.5 MB per chip per layer

# Expert weights (per expert, fp4)
EXPERT_WEIGHT_MB = {
    'gate': HIDDEN_SIZE * INTERMEDIATE_SIZE / 2 / (1024*1024),   # 10.5 MB
    'up':   HIDDEN_SIZE * INTERMEDIATE_SIZE / 2 / (1024*1024),   # 10.5 MB
    'down': INTERMEDIATE_SIZE * HIDDEN_SIZE / 2 / (1024*1024),   # 10.5 MB
}
EXPERT_TOTAL_MB = sum(EXPERT_WEIGHT_MB.values())  # 31.5 MB → used as 33 MB in proposal (with overhead)

# Router weight (fp8, not fp4 — precision-sensitive)
ROUTER_WEIGHT_MB = HIDDEN_SIZE * NUM_EXPERTS / (1024*1024)  # ~2.6 MB fp8

# RMSNorm weights (fp16, tiny)
NORM_WEIGHT_MB = 2 * HIDDEN_SIZE * 2 / (1024*1024)  # ~0.03 MB

# ── MAC counts per layer (§4.2) ──
MAC_MLA_Q_DOWN    = HIDDEN_SIZE * Q_LORA_RANK              # 11.01M
MAC_MLA_KV_LATENT = HIDDEN_SIZE * KV_LORA_RANK             #  3.67M
MAC_MLA_KV_ROPE   = HIDDEN_SIZE * QK_ROPE_HEAD_DIM         #  0.46M
MAC_MLA_QK_DOT    = 29.88e6  # Q·K^T (nope+rope)
MAC_MLA_AV_DOT    = 29.36e6  # A·V
MAC_MLA_O_DECOMPRESS = 67.11e6  # 128 × 512 × 1024
MAC_MLA_O_UP      = O_LORA_RANK * HIDDEN_SIZE              #  7.34M
MAC_MLA_TOTAL     = 148.8e6

MAC_EXPERT_GATE   = HIDDEN_SIZE * INTERMEDIATE_SIZE  # 22.02M
MAC_EXPERT_UP     = HIDDEN_SIZE * INTERMEDIATE_SIZE  # 22.02M
MAC_EXPERT_DOWN   = INTERMEDIATE_SIZE * HIDDEN_SIZE  # 22.02M
MAC_EXPERT_TOTAL  = 66.06e6  # 66.06M per expert

MAC_SHARED_EXPERT = MAC_EXPERT_TOTAL  # Same as routed

MAC_MOE_LAYER_TOTAL = MAC_MLA_TOTAL + MAC_SHARED_EXPERT + NUM_EXPERTS_PER_TOK * MAC_EXPERT_TOTAL  # ~611M

# ── C2C parameters (§6.2, §6.3.5) ──
C2C_LINK_BW_GBPS  = 128      # 4 lane × 32 Gbps NRZ
C2C_HOP_LATENCY_NS = 50      # SerDes + PCB trace
C2C_FRAME_OVERHEAD_B = 24    # 16B header + 4B CRC + 4B EOP
C2C_MAX_PAYLOAD_B    = 4088
C2C_MSG_DISPATCH_B   = HIDDEN_SIZE  # 7168 B FP8 (MoE_Dispatch / MoE_Reduce / Pipeline_Fwd)
C2C_DISPATCH_FRAMES  = math.ceil(C2C_MSG_DISPATCH_B / C2C_MAX_PAYLOAD_B)  # 2 frames

# C2C dispatch latency (same card, 7168 B in 2 frames, §6.3.5)
C2C_DISPATCH_LATENCY_NS = 250  # ~250 ns for 7168 B over 128 Gbps
C2C_REDUCE_LATENCY_NS   = 250  # same as dispatch
C2C_FWD_LATENCY_NS      = 250  # Pipeline_Fwd (same size)

# ── PCIe P2P parameters (§6.4, §6.3.5) ──
PCIE_P2P_BW_GBPS    = 64       # PCIe 5.0 x16
PCIE_P2P_LATENCY_NS = 400      # cross-card (including C2C proxy)

# ── HBM load times ──
# Deterministic weights per layer per chip (SRAM cached, no HBM access)
DETERMINISTIC_MB_PER_LAYER = 13.2  # Shared Expert + Attention + Router + RMSNorm (proposal §4.4.1.4)
EXPERT_HBM_LOAD_MB = 33.0         # One expert from HBM (including overhead)

# Expert hit probabilities (per chip, 12 experts / 384 total)
P_EXPERT_PER_CHIP = EXPERTS_PER_CHIP / NUM_EXPERTS  # 12/384 = 0.03125
P_0_HIT = (1 - P_EXPERT_PER_CHIP) ** NUM_EXPERTS_PER_TOK   # 81.6% (actually ~82.5% with 12/chip)
P_1_HIT_REST = NUM_EXPERTS_PER_TOK * P_EXPERT_PER_CHIP * (1 - P_EXPERT_PER_CHIP) ** (NUM_EXPERTS_PER_TOK - 1)  # ~16.9%
# Note: these are per-card probabilities from the proposal. For per-chip, we use:
# 12 experts/chip / 384 → p=0.03125 per chip
P_0_HIT = (1 - P_EXPERT_PER_CHIP) ** NUM_EXPERTS_PER_TOK
P_1_HIT = NUM_EXPERTS_PER_TOK * P_EXPERT_PER_CHIP * (1 - P_EXPERT_PER_CHIP) ** (NUM_EXPERTS_PER_TOK - 1)
P_2P_HIT = 1.0 - P_0_HIT - P_1_HIT

# ── Timing (μs) ──
# DSP compute times per chip (with TP≈2 for attention within card)
# Attention MACs split across 4 chips: MLA ~148.8M / 2 = 74.4M per chip
# Shared Expert MACs split across 4 chips: 66.06M / 2 = 33.03M per chip
# (TP=2 means 2 chips share the attention/shared-expert for each layer-hosting chip pair)
TP_ATTN_PER_LAYER = 2  # Two chips handle attention for a given layer
DSP_ATTN_SHARED_TIME_US = (MAC_MLA_TOTAL + MAC_SHARED_EXPERT) / TP_ATTN_PER_LAYER / (DSP_TMACS * 1e12) * 1e6  # ~9.7 μs
DSP_EXPERT_TIME_US       = MAC_EXPERT_TOTAL / (DSP_TMACS * 1e12) * 1e6  # ~6.0 μs
# Actually let's compute these dynamically


# ============================================================================
# Enums & Data Classes
# ============================================================================

class C2CMessageType(Enum):
    MOE_DISPATCH     = 0x1
    MOE_REDUCE       = 0x2
    PIPELINE_FWD     = 0x3
    PCIE_PROXY       = 0x4
    CREDIT_UPDATE    = 0x5
    WEIGHT_BROADCAST = 0x6
    HEARTBEAT        = 0x7


class StageType(Enum):
    """Pipeline stage types for per-layer processing."""
    WEIGHT_PREFETCH  = auto()   # HBM → SRAM streaming buffer
    MLA_ATTENTION    = auto()   # Q/KV compression, attention, O decompress
    ATTN_NORM        = auto()   # RMSNorm (attn residual)
    MOE_ROUTER       = auto()   # Top-6 expert selection
    MOE_DISPATCH     = auto()   # C2C: send activation to expert chips
    SHARED_EXPERT    = auto()   # Shared Expert FFN (SwiGLU, always local)
    ROUTED_EXPERT    = auto()   # Routed Expert FFN (one per selected expert)
    MOE_REDUCE       = auto()   # C2C: receive & combine expert outputs
    FFN_NORM         = auto()   # RMSNorm (FFN residual)
    PIPELINE_FWD     = auto()   # C2C: forward hidden state to next layer chip


@dataclass
class StageTiming:
    """Timing breakdown for a single pipeline stage."""
    stage_type: StageType
    dsp_time_us: float = 0.0        # DSP compute time
    hbm_time_us: float = 0.0        # HBM read time
    c2c_time_us: float = 0.0        # C2C communication time
    sram_time_us: float = 0.0       # SRAM access time
    total_time_us: float = 0.0      # max of all resources (critical path)
    dsp_util_pct: float = 0.0       # DSP utilization during this stage
    notes: str = ""


@dataclass
class LayerTiming:
    """Complete timing for one Transformer layer on one chip."""
    layer_idx: int
    chip_id: int
    stages: List[StageTiming] = field(default_factory=list)
    c2c_messages: List[Dict] = field(default_factory=list)  # dispatched C2C messages
    total_time_us: float = 0.0
    is_bottleneck: bool = False


@dataclass
class TokenTrace:
    """End-to-end trace of a single token through all 61 layers."""
    token_id: int
    layer_timings: List[LayerTiming] = field(default_factory=list)
    total_latency_us: float = 0.0
    total_dsp_us: float = 0.0
    total_hbm_us: float = 0.0
    total_c2c_us: float = 0.0


# ============================================================================
# Chip Model
# ============================================================================

class Chip:
    """Single AGM 039-F FPGA chip within a 4-chip card."""

    def __init__(self, chip_id: int, card_id: int):
        self.chip_id = chip_id          # 0-3 within card
        self.card_id = card_id          # 0-7
        self.global_id = card_id * CHIPS_PER_CARD + chip_id  # 0-31

        # Assigned layers (from proposal §4.1 allocation table)
        self.assigned_layers: List[int] = []

        # Assigned experts (12 per chip, contiguous ranges)
        self.assigned_experts: List[int] = list(range(
            self.global_id * EXPERTS_PER_CHIP,
            (self.global_id + 1) * EXPERTS_PER_CHIP
        ))

        # Resource models
        self.sram_used_mb = 0.0
        self.hbm_used_mb = 0.0
        self.dsp_busy_us = 0.0

        # C2C neighbors on Ring A
        self.ring_a_prev: Optional[int] = None  # global chip id
        self.ring_a_next: Optional[int] = None

        # Is this chip the PCIe master for the card?
        self.is_pcie_master = (chip_id == 0)

        # Per-layer weight cache state
        self.sram_cached_weights: Set[str] = set()  # weight keys in SRAM

    def __repr__(self):
        return f"Chip(c{self.card_id}.{self.chip_id}, layers={self.assigned_layers})"

    def assign_layers(self, layers: List[int]):
        self.assigned_layers = sorted(layers)

    def compute_dsp_time_us(self, macs: float) -> float:
        """DSP time in microseconds for given MAC count."""
        return macs / (DSP_TMACS * 1e12) * 1e6

    def compute_hbm_time_us(self, mb_to_load: float) -> float:
        """HBM read time in microseconds for given MB."""
        if mb_to_load <= 0:
            return 0.0
        return mb_to_load / (HBM_BW_GBPS / 1024)

    def compute_c2c_time_us(self, payload_bytes: int, num_hops: int = 1) -> float:
        """C2C transfer time including SerDes + framing."""
        frames = math.ceil(payload_bytes / C2C_MAX_PAYLOAD_B)
        serdes_time_ns = frames * (payload_bytes / frames + C2C_FRAME_OVERHEAD_B) * 8 / C2C_LINK_BW_GBPS
        total_ns = C2C_HOP_LATENCY_NS * num_hops + serdes_time_ns
        return total_ns / 1000.0  # ns → μs


# ============================================================================
# C2C Dual Ring Model
# ============================================================================

class C2CDualRing:
    """C2C Dual Ring interconnect within a 4-chip card."""

    # Ring A topology: C0-C1-C2-C3-C0 (bidirectional per hop)
    RING_A_LINKS = [(0,1), (1,2), (2,3), (3,0)]
    # Ring B topology: C0-C2, C1-C3 (redundant cross-links)
    RING_B_LINKS = [(0,2), (1,3)]

    def __init__(self, card_id: int, chips: List[Chip]):
        self.card_id = card_id
        self.chips = {c.chip_id: c for c in chips}

        # Build routing table: (src_chip, dst_chip) → (next_hop, num_hops)
        self._build_routing_table()

        # Bandwidth tracking
        self.link_usage_gbps: Dict[Tuple[int,int], float] = defaultdict(float)

    def _build_routing_table(self):
        """Dijkstra shortest path on Ring A (static topology, compile-time fixed)."""
        self.routes: Dict[Tuple[int,int], Tuple[int,int]] = {}  # (src,dst)→(next_hop,hops)

        for src in range(4):
            for dst in range(4):
                if src == dst:
                    continue
                # Ring A: find shortest path (clockwise or counter-clockwise)
                # Direct neighbors: (0,1)=1hop, (1,2)=1hop, (2,3)=1hop, (3,0)=1hop
                # 2-hop paths: Chip0↔Chip3 via Chip1 or Chip2, Chip1↔Chip2 via Chip0 or Chip3
                dist_clockwise = (dst - src) % 4
                dist_counter   = (src - dst) % 4
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
        """C2C transfer time within the same card."""
        _, hops = self.route(src_chip, dst_chip)
        frames = math.ceil(payload_bytes / C2C_MAX_PAYLOAD_B)
        avg_frame_bytes = payload_bytes / frames + C2C_FRAME_OVERHEAD_B
        serdes_ns = frames * avg_frame_bytes * 8 / C2C_LINK_BW_GBPS
        hop_ns = C2C_HOP_LATENCY_NS * hops
        return (serdes_ns + hop_ns) / 1000.0

    def record_transfer(self, src_chip: int, dst_chip: int, payload_bytes: int):
        """Record a C2C transfer for bandwidth accounting."""
        link = (min(src_chip, dst_chip), max(src_chip, dst_chip))
        # Approximate bandwidth usage
        bw_gbps = payload_bytes * 8 / (C2C_DISPATCH_LATENCY_NS / 1e9) / 1e9  # peak during transfer
        self.link_usage_gbps[link] += bw_gbps


# ============================================================================
# PCIe P2P Fabric Model
# ============================================================================

class PCIeFabric:
    """PCIe 5.0 P2P fabric connecting 8 cards via host backplane."""

    def __init__(self):
        self.transfer_count = 0
        self.total_bytes = 0

    def transfer_time_us(self, src_card: int, dst_card: int, payload_bytes: int) -> float:
        """
        Cross-card transfer time.
        Same CPU socket: PCIe P2P direct (~400 ns)
        Cross CPU socket: PCIe P2P via UPI (~400 ns + UPI overhead, but UPI 20 GB/s > traffic)
        """
        self.transfer_count += 1
        self.total_bytes += payload_bytes
        # Uniform 400 ns for all cross-card (from §6.3.5)
        serdes_ns = payload_bytes * 8 / PCIE_P2P_BW_GBPS
        return max(serdes_ns, PCIE_P2P_LATENCY_NS) / 1000.0  # ns → μs


# ============================================================================
# Cluster Model — Full 32-chip inference system
# ============================================================================

class FPGAInferenceCluster:
    """Complete 8-card × 4-chip FPGA inference cluster."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

        # Create chips
        self.chips: List[Chip] = []
        for card_id in range(NUM_CARDS):
            for chip_id in range(CHIPS_PER_CARD):
                self.chips.append(Chip(chip_id=chip_id, card_id=card_id))

        # Create cards with C2C rings
        self.cards: List[C2CDualRing] = []
        for card_id in range(NUM_CARDS):
            card_chips = [c for c in self.chips if c.card_id == card_id]
            ring = C2CDualRing(card_id, card_chips)
            self.cards.append(ring)

        # Cross-card fabric
        self.pcie_fabric = PCIeFabric()

        # Assign layers (from §4.1 table)
        self._assign_layers()

        # Expert → chip mapping
        self.expert_to_chip: Dict[int, Chip] = {}
        for chip in self.chips:
            for expert_id in chip.assigned_experts:
                self.expert_to_chip[expert_id] = chip

        # Weight placement
        self._place_weights()

        # Statistics
        self.total_tokens_processed = 0
        self.traces: List[TokenTrace] = []

    def _assign_layers(self):
        """Assign 61 layers across 32 chips per §4.1 allocation table."""
        # 29 chips × 2 layers + 3 chips × 1 layer = 61
        layer_per_chip = [2] * 29 + [1] * 3
        # Spread the 1-layer chips evenly
        one_layer_positions = [7, 15, 23]  # chips with 1 layer (roughly at card boundaries)
        layer_per_chip = [2] * 32
        for pos in one_layer_positions:
            layer_per_chip[pos] = 1
        # Adjust: we need exactly 61 layers total
        total = sum(layer_per_chip)
        # 32*2 = 64, need to remove 3 layers = make 3 chips have 1 layer
        # Already done. Total: 29*2 + 3*1 = 58 + 3 = 61 ✓

        layer_idx = 0
        for chip in self.chips:
            n = layer_per_chip[chip.global_id]
            chip.assign_layers(list(range(layer_idx, layer_idx + n)))
            layer_idx += n

        # Special: Embedding on C0.0, lm_head on C7.3
        self.chip_for_embedding = self.chips[0]  # C0.0
        self.chip_for_lm_head  = self.chips[-1]  # C7.3

    def _place_weights(self):
        """Model weight placement across HBM and SRAM."""
        for chip in self.chips:
            # Each chip stores its assigned experts' weights in HBM
            expert_weight_mb = len(chip.assigned_experts) * EXPERT_TOTAL_MB
            # Each chip stores attention weights for its assigned layers
            n_layers = len(chip.assigned_layers)
            attn_weight_per_layer_mb = sum(ATTN_WEIGHT_MB.values()) / TP_ATTN_PER_LAYER  # TP-shared
            attn_total_mb = n_layers * attn_weight_per_layer_mb
            # Router weights (all 384 experts, fp8) — stored on each chip
            router_mb = ROUTER_WEIGHT_MB

            chip.hbm_used_mb = expert_weight_mb + attn_total_mb + router_mb

            # SRAM: cache deterministic weights for current layer (double-buffered)
            chip.sram_used_mb = 18.6  # ~18.6 MB for double-buffered deterministic weights (§4.4.1.3)

    def get_chip_for_layer(self, layer_idx: int) -> Chip:
        """Find which chip hosts a given layer."""
        for chip in self.chips:
            if layer_idx in chip.assigned_layers:
                return chip
        raise ValueError(f"Layer {layer_idx} not assigned to any chip")

    def get_chip_for_expert(self, expert_idx: int) -> Chip:
        """Find which chip hosts a given expert."""
        return self.expert_to_chip[expert_idx]

    def is_same_card(self, chip_a: Chip, chip_b: Chip) -> bool:
        return chip_a.card_id == chip_b.card_id

    def c2c_transfer_time_us(self, src: Chip, dst: Chip, payload_bytes: int) -> float:
        """C2C or PCIe transfer time between any two chips."""
        if src.global_id == dst.global_id:
            return 0.0
        if self.is_same_card(src, dst):
            card = self.cards[src.card_id]
            return card.transfer_time_us(src.chip_id, dst.chip_id, payload_bytes)
        else:
            return self.pcie_fabric.transfer_time_us(src.card_id, dst.card_id, payload_bytes)


# ============================================================================
# Pipeline Stage Implementations
# ============================================================================

class PipelineStage:
    """Base class for pipeline stages."""

    def __init__(self, name: str, stage_type: StageType):
        self.name = name
        self.stage_type = stage_type

    def execute(self, cluster: FPGAInferenceCluster, chip: Chip,
                hidden_state: np.ndarray, layer_idx: int,
                kv_cache=None, expert_selection: List[int] = None) -> Tuple[np.ndarray, StageTiming, Dict]:
        """
        Execute this pipeline stage.

        Returns:
            (output_state, timing, metadata)
        """
        raise NotImplementedError


class WeightPrefetchStage(PipelineStage):
    """Prefetch layer weights from HBM into SRAM streaming buffer."""

    def __init__(self):
        super().__init__("Weight Prefetch", StageType.WEIGHT_PREFETCH)

    def execute(self, cluster, chip, hidden_state, layer_idx, kv_cache=None, expert_selection=None):
        timing = StageTiming(stage_type=self.stage_type)
        n_local_experts = sum(1 for e in (expert_selection or [])
                              if cluster.get_chip_for_expert(e).global_id == chip.global_id)

        # Deterministic weights always in SRAM: attention + shared + router + norms
        # Expert weights stream from HBM during DSP (overlapped, captured in RoutedExpert stage)
        timing.sram_time_us = 0.1  # deterministic weight access from SRAM
        timing.hbm_time_us = 0.0   # expert HBM loading is captured in RoutedExpert stage
        timing.total_time_us = 0.1
        timing.notes = f"SRAM deterministic ({DETERMINISTIC_MB_PER_LAYER:.0f}MB cached), {n_local_experts} local experts to load"

        return hidden_state, timing, {'hbm_mb': 0.0, 'local_experts': n_local_experts}


class MLAAttentionStage(PipelineStage):
    """MLA Attention: Q/KV compression → attention → O decompression."""

    def __init__(self):
        super().__init__("MLA Attention", StageType.MLA_ATTENTION)

    def execute(self, cluster, chip, hidden_state, layer_idx, kv_cache=None, expert_selection=None):
        timing = StageTiming(stage_type=self.stage_type)

        # MACs per chip (TP=2 shares attention across 2 chips in the card)
        macs_per_chip = MAC_MLA_TOTAL / TP_ATTN_PER_LAYER

        timing.dsp_time_us = chip.compute_dsp_time_us(macs_per_chip)
        timing.sram_time_us = 0.2  # KV cache SRAM access
        timing.hbm_time_us = 0.0   # all attention weights in SRAM
        timing.total_time_us = max(timing.dsp_time_us, timing.sram_time_us, timing.hbm_time_us)

        # Simulate attention output (simplified)
        output = hidden_state  # placeholder — actual computation not modeled here
        timing.notes = f"{macs_per_chip/1e6:.1f}M MACs/chip, TP={TP_ATTN_PER_LAYER}"

        return output, timing, {'macs': macs_per_chip}


class AttnNormStage(PipelineStage):
    """RMSNorm after attention residual."""

    def __init__(self):
        super().__init__("Attn RMSNorm", StageType.ATTN_NORM)

    def execute(self, cluster, chip, hidden_state, layer_idx, kv_cache=None, expert_selection=None):
        timing = StageTiming(stage_type=self.stage_type)
        macs = HIDDEN_SIZE * 4  # ~28.7K MACs — negligible
        timing.dsp_time_us = chip.compute_dsp_time_us(macs)
        timing.total_time_us = timing.dsp_time_us
        return hidden_state, timing, {}


class MoERouterStage(PipelineStage):
    """MoE Router: select top-6 experts from 384."""

    def __init__(self, rng: np.random.RandomState):
        super().__init__("MoE Router", StageType.MOE_ROUTER)
        self.rng = rng

    def execute(self, cluster, chip, hidden_state, layer_idx, kv_cache=None, expert_selection=None):
        timing = StageTiming(stage_type=self.stage_type)
        # Router: 7168 × 384 MACs ≈ 2.75M MACs, fp8 precision
        router_macs = HIDDEN_SIZE * NUM_EXPERTS  # 2.75M
        timing.dsp_time_us = chip.compute_dsp_time_us(router_macs)
        timing.sram_time_us = 0.05  # router weights & tables in SRAM
        timing.total_time_us = max(timing.dsp_time_us, timing.sram_time_us)

        # Simulate expert selection with power-law bias
        # Top ~20% experts get ~80% of tokens (power-law)
        if expert_selection is None:
            if self.rng.random() < 0.8:
                # 80% of tokens hit head experts (top 77 = 20% of 384)
                candidates = self.rng.choice(77, size=NUM_EXPERTS_PER_TOK * 2, replace=False)
            else:
                # 20% of tokens hit tail experts
                candidates = self.rng.choice(range(77, 384), size=NUM_EXPERTS_PER_TOK * 2, replace=False)
            self.rng.shuffle(candidates)
            expert_selection = sorted(candidates[:NUM_EXPERTS_PER_TOK])

        timing.notes = f"selected experts: {expert_selection}"
        return hidden_state, timing, {'selected_experts': expert_selection}


class MoEDispatchStage(PipelineStage):
    """C2C Dispatch: send activation to expert-hosting chips."""

    def __init__(self):
        super().__init__("MoE Dispatch", StageType.MOE_DISPATCH)

    def execute(self, cluster, chip, hidden_state, layer_idx, kv_cache=None, expert_selection=None):
        timing = StageTiming(stage_type=self.stage_type)
        messages = []

        if not expert_selection:
            return hidden_state, timing, {'messages': []}

        # Group experts by hosting chip
        experts_by_chip: Dict[int, List[int]] = defaultdict(list)
        for expert_idx in expert_selection:
            host_chip = cluster.get_chip_for_expert(expert_idx)
            experts_by_chip[host_chip.global_id].append(expert_idx)

        total_c2c_us = 0.0
        for dst_chip_id, experts in experts_by_chip.items():
            dst_chip = cluster.chips[dst_chip_id]
            if dst_chip.global_id == chip.global_id:
                continue  # local expert, no C2C

            # MoE_Dispatch: 7168 B FP8 activation vector
            t_us = cluster.c2c_transfer_time_us(chip, dst_chip, C2C_MSG_DISPATCH_B)
            msg = {
                'type': C2CMessageType.MOE_DISPATCH,
                'src': chip.global_id,
                'dst': dst_chip_id,
                'experts': experts,
                'payload_bytes': C2C_MSG_DISPATCH_B,
                'time_us': t_us,
                'cross_card': not cluster.is_same_card(chip, dst_chip),
            }
            messages.append(msg)
            total_c2c_us = max(total_c2c_us, t_us)  # parallel dispatch

        timing.c2c_time_us = total_c2c_us
        timing.total_time_us = timing.c2c_time_us
        timing.notes = f"dispatch to {len(messages)} remote chips, {len(expert_selection)} experts"

        return hidden_state, timing, {'messages': messages}


class SharedExpertStage(PipelineStage):
    """Shared Expert FFN (SwiGLU) — always local, always active."""

    def __init__(self):
        super().__init__("Shared Expert", StageType.SHARED_EXPERT)

    def execute(self, cluster, chip, hidden_state, layer_idx, kv_cache=None, expert_selection=None):
        timing = StageTiming(stage_type=self.stage_type)
        macs = MAC_SHARED_EXPERT / TP_ATTN_PER_LAYER  # TP-shared
        timing.dsp_time_us = chip.compute_dsp_time_us(macs)
        timing.sram_time_us = 0.1   # weights in SRAM
        timing.hbm_time_us = 0.0    # cached
        timing.total_time_us = max(timing.dsp_time_us, timing.sram_time_us)
        return hidden_state, timing, {'macs': macs}


class RoutedExpertStage(PipelineStage):
    """Routed Expert FFN (SwiGLU) — local experts computed on this chip."""

    def __init__(self):
        super().__init__("Routed Expert", StageType.ROUTED_EXPERT)

    def execute(self, cluster, chip, hidden_state, layer_idx, kv_cache=None, expert_selection=None):
        timing = StageTiming(stage_type=self.stage_type)

        # Which of the selected experts are local to this chip?
        local_experts = [e for e in (expert_selection or [])
                         if cluster.get_chip_for_expert(e).global_id == chip.global_id]

        n_local = len(local_experts)
        if n_local == 0:
            timing.total_time_us = 0.0
            timing.notes = "no local experts"
            return hidden_state, timing, {'local_experts': 0}

        # Gate + Up + Down for each local expert
        macs = n_local * MAC_EXPERT_TOTAL
        timing.dsp_time_us = chip.compute_dsp_time_us(macs)

        # HBM time: load expert weights (not in SRAM, streaming from HBM)
        hbm_mb = n_local * EXPERT_HBM_LOAD_MB
        timing.hbm_time_us = chip.compute_hbm_time_us(hbm_mb)

        timing.total_time_us = max(timing.dsp_time_us, timing.hbm_time_us)
        timing.dsp_util_pct = (timing.dsp_time_us / timing.total_time_us * 100) if timing.total_time_us > 0 else 0
        timing.notes = f"{n_local} local experts: {local_experts}"

        return hidden_state, timing, {
            'local_experts': n_local,
            'expert_ids': local_experts,
            'hbm_mb': hbm_mb
        }


class MoEReduceStage(PipelineStage):
    """C2C Reduce: receive expert outputs from remote chips, weighted sum."""

    def __init__(self):
        super().__init__("MoE Reduce", StageType.MOE_REDUCE)

    def execute(self, cluster, chip, hidden_state, layer_idx, kv_cache=None, expert_selection=None):
        timing = StageTiming(stage_type=self.stage_type)
        messages = []

        if not expert_selection:
            return hidden_state, timing, {'messages': []}

        # Identify which expert results need to come back
        experts_by_chip: Dict[int, List[int]] = defaultdict(list)
        for expert_idx in expert_selection:
            host_chip = cluster.get_chip_for_expert(expert_idx)
            if host_chip.global_id != chip.global_id:
                experts_by_chip[host_chip.global_id].append(expert_idx)

        total_c2c_us = 0.0
        for src_chip_id, experts in experts_by_chip.items():
            src_chip = cluster.chips[src_chip_id]
            t_us = cluster.c2c_transfer_time_us(src_chip, chip, C2C_MSG_DISPATCH_B)
            msg = {
                'type': C2CMessageType.MOE_REDUCE,
                'src': src_chip_id,
                'dst': chip.global_id,
                'experts': experts,
                'payload_bytes': C2C_MSG_DISPATCH_B,
                'time_us': t_us,
                'cross_card': not cluster.is_same_card(chip, src_chip),
            }
            messages.append(msg)
            total_c2c_us = max(total_c2c_us, t_us)

        timing.c2c_time_us = total_c2c_us
        timing.total_time_us = timing.c2c_time_us
        timing.notes = f"reduce from {len(messages)} remote chips"

        return hidden_state, timing, {'messages': messages}


class FFNNormStage(PipelineStage):
    """RMSNorm after FFN residual."""

    def __init__(self):
        super().__init__("FFN RMSNorm", StageType.FFN_NORM)

    def execute(self, cluster, chip, hidden_state, layer_idx, kv_cache=None, expert_selection=None):
        timing = StageTiming(stage_type=self.stage_type)
        macs = HIDDEN_SIZE * 4
        timing.dsp_time_us = chip.compute_dsp_time_us(macs)
        timing.total_time_us = timing.dsp_time_us
        return hidden_state, timing, {}


class PipelineForwardStage(PipelineStage):
    """C2C Pipeline Forward: send hidden state to next layer's hosting chip."""

    def __init__(self):
        super().__init__("Pipeline Forward", StageType.PIPELINE_FWD)

    def execute(self, cluster, chip, hidden_state, layer_idx, kv_cache=None, expert_selection=None):
        timing = StageTiming(stage_type=self.stage_type)

        next_layer = layer_idx + 1
        if next_layer >= NUM_LAYERS:
            timing.total_time_us = 0.0
            timing.notes = "final layer, no forward"
            return hidden_state, timing, {'next_chip': None}

        next_chip = cluster.get_chip_for_layer(next_layer)
        if next_chip.global_id == chip.global_id:
            # Same chip — just register transfer, negligible
            timing.total_time_us = 0.001  # ~1 ns
            timing.notes = f"same chip (L{layer_idx}→L{next_layer})"
            return hidden_state, timing, {'next_chip': next_chip.global_id, 'same_chip': True}

        # C2C Pipeline_Fwd: 7168 B FP8 hidden state
        t_us = cluster.c2c_transfer_time_us(chip, next_chip, C2C_MSG_DISPATCH_B)
        msg = {
            'type': C2CMessageType.PIPELINE_FWD,
            'src': chip.global_id,
            'dst': next_chip.global_id,
            'payload_bytes': C2C_MSG_DISPATCH_B,
            'time_us': t_us,
            'cross_card': not cluster.is_same_card(chip, next_chip),
            'from_layer': layer_idx,
            'to_layer': next_layer,
        }
        timing.c2c_time_us = t_us
        timing.total_time_us = t_us
        timing.notes = f"C2C forward L{layer_idx}→L{next_layer}: chip{chip.global_id}→chip{next_chip.global_id}"

        return hidden_state, timing, {'next_chip': next_chip.global_id, 'message': msg}


# ============================================================================
# Full Pipeline Engine
# ============================================================================

class PipelineEngine:
    """Orchestrates full token processing through all 61 layers across 32 chips."""

    def __init__(self, cluster: FPGAInferenceCluster, seed: int = 42):
        self.cluster = cluster
        self.rng = np.random.RandomState(seed)

        # Pipeline stages in order
        self.stages: List[PipelineStage] = [
            WeightPrefetchStage(),
            MLAAttentionStage(),
            AttnNormStage(),
            MoERouterStage(self.rng),
            MoEDispatchStage(),
            SharedExpertStage(),
            RoutedExpertStage(),
            MoEReduceStage(),
            FFNNormStage(),
            PipelineForwardStage(),
        ]

        # Accumulated statistics
        self.stats = defaultdict(list)

    def process_layer(self, chip: Chip, hidden_state: np.ndarray,
                      layer_idx: int, kv_cache=None,
                      expert_selection: List[int] = None) -> LayerTiming:
        """Process a single layer on a single chip through all pipeline stages."""
        layer_timing = LayerTiming(layer_idx=layer_idx, chip_id=chip.global_id)
        current_state = hidden_state
        current_kv = kv_cache

        for stage in self.stages:
            # Skip pipeline forward for the last layer
            if stage.stage_type == StageType.PIPELINE_FWD and layer_idx == NUM_LAYERS - 1:
                output, timing, meta = stage.execute(
                    self.cluster, chip, current_state, layer_idx, current_kv, expert_selection
                )
                layer_timing.stages.append(timing)
                continue

            # MoE Dispatch and Reduce only apply to the chip that processes this layer
            # (the "initiating chip" that runs the router)
            if stage.stage_type == StageType.MOE_DISPATCH:
                output, timing, meta = stage.execute(
                    self.cluster, chip, current_state, layer_idx, current_kv, expert_selection
                )
                layer_timing.stages.append(timing)
                if 'messages' in meta:
                    layer_timing.c2c_messages.extend(meta['messages'])
                continue

            if stage.stage_type == StageType.MOE_REDUCE:
                output, timing, meta = stage.execute(
                    self.cluster, chip, current_state, layer_idx, current_kv, expert_selection
                )
                layer_timing.stages.append(timing)
                if 'messages' in meta:
                    layer_timing.c2c_messages.extend(meta['messages'])
                continue

            output, timing, meta = stage.execute(
                self.cluster, chip, current_state, layer_idx, current_kv, expert_selection
            )
            layer_timing.stages.append(timing)

            # Update expert selection from router stage
            if stage.stage_type == StageType.MOE_ROUTER and 'selected_experts' in meta:
                expert_selection = meta['selected_experts']

        # Compute total layer time (sum of stage critical paths)
        layer_timing.total_time_us = sum(s.total_time_us for s in layer_timing.stages)

        return layer_timing

    def process_token(self, token_id: int = 0) -> TokenTrace:
        """Process a single token through all 61 layers."""
        trace = TokenTrace(token_id=token_id)
        hidden_state = self.rng.randn(1, HIDDEN_SIZE).astype(np.float32) * 0.02

        # Embedding lookup (on C0.0)
        emb_chip = self.cluster.chip_for_embedding

        for layer_idx in range(NUM_LAYERS):
            chip = self.cluster.get_chip_for_layer(layer_idx)
            layer_timing = self.process_layer(chip, hidden_state, layer_idx)
            trace.layer_timings.append(layer_timing)

            # Simulate KV cache growth
            kv_len = min(layer_idx + 1, SLIDING_WINDOW)

        # lm_head projection (on C7.3)
        lm_head_chip = self.cluster.chip_for_lm_head

        # Aggregate trace
        trace.total_latency_us = sum(lt.total_time_us for lt in trace.layer_timings)
        trace.total_dsp_us = sum(
            s.dsp_time_us for lt in trace.layer_timings for s in lt.stages
        )
        trace.total_hbm_us = sum(
            s.hbm_time_us for lt in trace.layer_timings for s in lt.stages
        )
        trace.total_c2c_us = sum(
            s.c2c_time_us for lt in trace.layer_timings for s in lt.stages
        )

        self.cluster.total_tokens_processed += 1
        self.cluster.traces.append(trace)
        return trace

    def analyze_pipeline(self, num_tokens: int = 20) -> Dict:
        """Run pipeline analysis for multiple tokens, compute key metrics."""
        traces = []
        for i in range(num_tokens):
            trace = self.process_token(token_id=i)
            traces.append(trace)

        # Aggregate statistics
        layer_latencies = defaultdict(list)
        stage_latencies = defaultdict(list)
        c2c_breakdown = {'same_card': 0, 'cross_card': 0}
        total_c2c_messages = 0
        expert_hit_dist = {0: 0, 1: 0, 2: 0}

        for trace in traces:
            for lt in trace.layer_timings:
                layer_latencies[lt.layer_idx].append(lt.total_time_us)
                for stage in lt.stages:
                    stage_latencies[stage.stage_type].append(stage.total_time_us)

                # Count expert hits
                routed_stage = next((s for s in lt.stages
                                     if s.stage_type == StageType.ROUTED_EXPERT), None)
                if routed_stage:
                    try:
                        n_local = int(routed_stage.notes.split()[0]) if routed_stage.notes else 0
                    except (ValueError, IndexError):
                        n_local = 0
                    key = min(n_local, 2)  # 0, 1, 2+
                    expert_hit_dist[key] += 1

                # Count C2C messages
                for msg in lt.c2c_messages:
                    total_c2c_messages += 1
                    if msg.get('cross_card'):
                        c2c_breakdown['cross_card'] += 1
                    else:
                        c2c_breakdown['same_card'] += 1

        # Compute averages
        avg_layer_latency = {}
        for layer_idx, lats in sorted(layer_latencies.items()):
            avg_layer_latency[layer_idx] = np.mean(lats)

        avg_stage_latency = {}
        for st, lats in stage_latencies.items():
            avg_stage_latency[st] = np.mean(lats)

        avg_total_latency = np.mean([t.total_latency_us for t in traces])

        # Throughput estimate
        # Pipeline depth = 61 layers, each layer on different chip
        # Pipelining: tokens flow through layers, each chip processes ~2 layers
        # Bottleneck = slowest chip's per-token time
        chip_total_times = defaultdict(list)
        for trace in traces:
            for lt in trace.layer_timings:
                chip_total_times[lt.chip_id].append(lt.total_time_us)

        avg_per_layer_time = {}
        for chip_id, times in chip_total_times.items():
            avg_per_layer_time[chip_id] = np.mean(times)  # avg per-layer on this chip

        # Bottleneck: slowest chip's per-layer time determines pipeline rate
        # In a deep pipeline, all 32 chips work in parallel on different tokens
        # Pipeline rate = 1 / max(per_layer_time)  (tokens enter pipeline at this rate)
        bottleneck_per_layer_us = max(avg_per_layer_time.values()) if avg_per_layer_time else 0
        bottleneck_chip_time_us = (bottleneck_per_layer_us *
                                    max(len(set(lt.layer_idx for lt in t.layer_timings
                                                 if lt.chip_id == max(avg_per_layer_time,
                                                                      key=avg_per_layer_time.get)))
                                        for t in traces)
                                    if traces else 0)
        # Simplified: use the avg layers per chip (~2)
        bottleneck_chip_us = bottleneck_per_layer_us * 2
        throughput_tps = 1e6 / bottleneck_per_layer_us if bottleneck_per_layer_us > 0 else float('inf')

        return {
            'num_tokens': num_tokens,
            'avg_token_latency_us': avg_total_latency,
            'avg_token_latency_ms': avg_total_latency / 1000,
            'bottleneck_chip_us': bottleneck_chip_us,
            'bottleneck_per_layer_us': bottleneck_per_layer_us,
            'throughput_tps': throughput_tps,
            'avg_layer_latency_us': avg_layer_latency,
            'avg_stage_latency_us': {k.name: v for k, v in avg_stage_latency.items()},
            'expert_hit_distribution': expert_hit_dist,
            'c2c_breakdown': c2c_breakdown,
            'total_c2c_messages': total_c2c_messages,
            'dsp_utilization_pct': self._compute_dsp_utilization(traces),
            'hbm_bandwidth_usage_gbps': self._compute_hbm_usage(traces),
            'traces': traces,
        }

    def _compute_dsp_utilization(self, traces: List[TokenTrace]) -> float:
        """Weighted DSP utilization: avg DSP busy time / avg wall time (per layer)."""
        total_wall = 0.0
        total_dsp = 0.0
        for trace in traces:
            for lt in trace.layer_timings:
                total_wall += lt.total_time_us
                total_dsp += sum(s.dsp_time_us for s in lt.stages)
        return (total_dsp / total_wall * 100) if total_wall > 0 else 0

    def _compute_hbm_usage(self, traces: List[TokenTrace]) -> float:
        """Compute aggregate HBM bandwidth usage (GB/s)."""
        total_hbm_mb = sum(t.total_hbm_us * HBM_BW_GBPS / 1024 for t in traces)
        total_time_s = sum(t.total_latency_us for t in traces) / 1e6
        return total_hbm_mb / (1024 * total_time_s) if total_time_s > 0 else 0


# ============================================================================
# Detailed Layer-by-Layer Pipeline Visualization
# ============================================================================

def print_pipeline_analysis(results: Dict, verbose: bool = False):
    """Print formatted pipeline analysis results."""

    print()
    print("=" * 79)
    print("   4-FPGA Chip Distributed Pipeline Analysis")
    print("=" * 79)
    print(f"   Architecture: {NUM_CARDS} cards x {CHIPS_PER_CARD} chips = {TOTAL_CHIPS} AGM 039-F")
    print(f"   Layers: {NUM_LAYERS}, Experts: {NUM_EXPERTS} ({EXPERTS_PER_CHIP}/chip)")
    print(f"   C2C: Dual Ring (128 Gbps/link, ~50 ns/hop)")
    print(f"   PCIe: 5.0 x16 P2P (~400 ns cross-card)")
    print()

    # ── Overall Metrics ──
    print("  --- Overall Performance ---")
    print(f"   Tokens simulated:        {results['num_tokens']:>5d}")
    print(f"   Avg token latency:       {results['avg_token_latency_ms']:>8.2f} ms")
    print(f"   Est. throughput:         {results['throughput_tps']:>8.0f} tok/s (pipelined)")
    print(f"   Bottleneck per-layer:    {results['bottleneck_chip_us']/2:>8.1f} us (chip avg)")
    print(f"   Batch-1 throughput:      {1e6/results['avg_token_latency_us']:>8.0f} tok/s (sequential)")
    print(f"   DSP utilization:         {results['dsp_utilization_pct']:>8.1f} %")
    print(f"   C2C messages/token:      {results['total_c2c_messages'] / max(results['num_tokens'], 1):>8.1f}")
    print()

    # ── Stage Breakdown ──
    print("  --- Stage Timing Breakdown (avg per layer) ---")
    print(f"   {'Stage':<20s} {'Avg us':>8s}  {'% of Layer':>10s}  {'Resource':>12s}")
    print(f"   {'-'*20} {'-'*8}  {'-'*10}  {'-'*12}")

    total_stage_us = sum(results['avg_stage_latency_us'].values())
    resource_map = {
        'MLA_ATTENTION': 'DSP',
        'ATTN_NORM': 'DSP',
        'MOE_ROUTER': 'DSP+SRAM',
        'MOE_DISPATCH': 'C2C',
        'SHARED_EXPERT': 'DSP+SRAM',
        'ROUTED_EXPERT': 'DSP+HBM',
        'MOE_REDUCE': 'C2C',
        'FFN_NORM': 'DSP',
        'PIPELINE_FWD': 'C2C',
        'WEIGHT_PREFETCH': 'HBM',
    }
    resource_icons = {
        'DSP': '[DSP]',
        'C2C': '[C2C]',
        'HBM': '[HBM]',
        'DSP+SRAM': '[DSP+SRAM]',
        'DSP+HBM': '[DSP+HBM]',
    }

    for stage_name, avg_us in sorted(results['avg_stage_latency_us'].items(),
                                       key=lambda x: x[1], reverse=True):
        pct = (avg_us / total_stage_us * 100) if total_stage_us > 0 else 0
        resource = resource_map.get(stage_name, '')
        icon = resource_icons.get(resource, resource)
        print(f"   {stage_name:<20s} {avg_us:>8.2f}  {pct:>9.1f}%  {icon:>12s}")
    print()

    # ── Expert Hit Distribution ──
    print("  --- Expert Local Hit Distribution (per layer per chip) ---")
    hit_dist = results['expert_hit_distribution']
    total_hits = sum(hit_dist.values())
    for k in sorted(hit_dist.keys()):
        pct = hit_dist[k] / total_hits * 100 if total_hits > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {k} local hit(s):  {hit_dist[k]:>6d}  ({pct:>5.1f}%)  {bar}")
    print()

    # ── C2C Communication Breakdown ──
    print("  --- C2C Communication Breakdown ---")
    c2c = results['c2c_breakdown']
    total_c2c = c2c['same_card'] + c2c['cross_card']
    if total_c2c > 0:
        print(f"    Same-card (C2C Ring):    {c2c['same_card']:>6d}  ({c2c['same_card']/total_c2c*100:>5.1f}%)")
        print(f"    Cross-card (PCIe P2P):   {c2c['cross_card']:>6d}  ({c2c['cross_card']/total_c2c*100:>5.1f}%)")
    print(f"    Total messages:          {total_c2c:>6d}")
    print(f"    Same-card latency:       ~250 ns")
    print(f"    Cross-card latency:      ~400 ns")
    print()

    # ── Layer-by-Layer Detail ──
    if verbose:
        print("  --- Layer-by-Layer Timing Detail ---")
        trace_latencies = [t.total_latency_us for t in results['traces']]
        median_idx = np.argsort(trace_latencies)[len(trace_latencies)//2]
        trace = results['traces'][median_idx]

        for lt in trace.layer_timings:
            chip = f"C{lt.chip_id//4}.{lt.chip_id%4}"
            fwd_stage = next((s for s in lt.stages
                             if s.stage_type == StageType.PIPELINE_FWD), None)
            fwd_marker = ""
            if fwd_stage and fwd_stage.c2c_time_us > 0.01:
                fwd_marker = " -> C2C"

            print(f"    L{lt.layer_idx:02d} @ {chip:<5s}  {lt.total_time_us:>8.2f} us{fwd_marker}")

            if verbose:
                for s in lt.stages:
                    if s.total_time_us > 0.5:
                        print(f"      {s.stage_type.name:<24s} {s.total_time_us:>8.2f} us  {s.notes:<30s}")
        print()

    # ── Chip Load Distribution ──
    print("  --- Per-Chip Load Distribution ---")
    chip_times = defaultdict(float)
    chip_layers = defaultdict(list)
    for trace in results['traces']:
        for lt in trace.layer_timings:
            chip_times[lt.chip_id] += lt.total_time_us
            if lt.layer_idx not in chip_layers[lt.chip_id]:
                chip_layers[lt.chip_id].append(lt.layer_idx)

    for chip_id in sorted(chip_times.keys()):
        avg_time = chip_times[chip_id] / results['num_tokens']
        card = chip_id // 4
        chip = chip_id % 4
        layers = chip_layers[chip_id]
        pcie = "PCIe" if chip == 0 else ""
        bar_len = int(avg_time / max(chip_times.values()) * 50) if max(chip_times.values()) > 0 else 0
        bar = '#' * min(bar_len, 50)
        print(f"    C{card}.{chip} {pcie:<4s}  L{min(layers):02d}-{max(layers):02d}  {avg_time:>8.1f} us  {bar}")
    print()

    print()

    # ── Resource Utilization Summary ──
    print("  --- Resource Utilization Summary ---")
    print(f"    DSP:    {DSP_COUNT:,} units @ {DSP_FREQ_MHZ} MHz = {DSP_TMACS:.2f} TMACs/s")
    print(f"    HBM:    {HBM_SIZE_GB} GB @ {HBM_BW_GBPS} GB/s ({HBM_BW_GBPS/1024:.2f} TB/s)")
    print(f"    SRAM:   {SRAM_TOTAL_MB:.0f} MB (deterministic weights cached)")
    print(f"    C2C:    Dual Ring, {C2C_LINK_BW_GBPS} Gbps/link, {C2C_HOP_LATENCY_NS}ns/hop")
    print(f"    PCIe:   5.0 x16 P2P, {PCIE_P2P_BW_GBPS} GB/s")
    print()

    # ── Comparison with H100 ──
    print("  --- FPGA vs H100 Comparison (batch=1 decode) ---")
    h100_hbm_us = 129  # per layer HBM time for H100 (proposal §4.4.1.6)
    fpga_total = results['avg_token_latency_us']
    h100_total = h100_hbm_us * NUM_LAYERS  # ~7869 us
    print(f"    FPGA per-layer (weighted):  ~{fpga_total/NUM_LAYERS:.1f} us")
    print(f"    H100 per-layer (weighted):  ~{h100_hbm_us} us")
    print(f"    FPGA total token latency:   ~{fpga_total:.0f} us ({fpga_total/1000:.2f} ms)")
    print(f"    H100 total token latency:   ~{h100_total:.0f} us ({h100_total/1000:.2f} ms)")
    print(f"    FPGA throughput:            ~{results['throughput_tps']:.0f} tok/s")
    if fpga_total > 0:
        print(f"    FPGA vs H100 advantage:     ~{h100_total/fpga_total:.1f}x faster")
    print()


# ============================================================================
# Sensitivity Analysis
# ============================================================================

def run_sensitivity_analysis(cluster: FPGAInferenceCluster):
    """Analyze system sensitivity to key parameters."""
    print()
    print("=" * 79)
    print("   Sensitivity Analysis")
    print("=" * 79)
    print()
    print("  --- Expert Hit Rate vs Throughput ---")
    print(f"   {'Experts/Chip':>14s} {'P(0 hit)':>10s} {'P(1 hit)':>10s} {'Bottleneck':>12s} {'TPS':>8s}")
    print(f"   {'-'*14} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

    for exp_per_chip in [6, 8, 10, 12, 16, 24]:
        p = exp_per_chip / NUM_EXPERTS
        p0 = (1 - p) ** NUM_EXPERTS_PER_TOK
        p1 = NUM_EXPERTS_PER_TOK * p * (1 - p) ** (NUM_EXPERTS_PER_TOK - 1)

        # Simplified timing model
        dsp_time = (MAC_MLA_TOTAL/TP_ATTN_PER_LAYER + MAC_SHARED_EXPERT/TP_ATTN_PER_LAYER) / (DSP_TMACS*1e12) * 1e6
        if exp_per_chip > 0:
            dsp_time += (p0*0 + p1*1 + (1-p0-p1)*2) * MAC_EXPERT_TOTAL / (DSP_TMACS*1e12) * 1e6
        hbm_time = (p1 * 1 + (1-p0-p1) * 2) * EXPERT_HBM_LOAD_MB / (HBM_BW_GBPS/1024)
        layer_time = max(dsp_time, hbm_time) + 0.25  # + C2C dispatch/reduce (250ns)
        tps = 1e6 / (layer_time * 2)  # 2 layers/chip avg

        print(f"   {exp_per_chip:>14d} {p0:>10.1%} {p1:>10.1%} {layer_time:>11.1f}us {tps:>8.0f}")
    print()

    print("  --- HBM Bandwidth vs Throughput ---")
    print(f"   {'HBM BW (GB/s)':>14s} {'0-hit':>8s} {'1-hit':>8s} {'2-hit':>8s} {'TPS':>8s}")
    print(f"   {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for hbm_bw in [460, 690, 920, 1380, 1840]:
        t0 = 3.4  # 0-hit DSP time (proposal estimate)
        t1 = max(3.4 + 7.8, 33.4 / (hbm_bw/1024))  # 1-hit
        t2 = max(3.4 + 15.6, 66.4 / (hbm_bw/1024))  # 2-hit
        weighted = t0 * P_0_HIT + t1 * P_1_HIT + t2 * P_2P_HIT
        tps = 1e6 / (weighted * 2)  # 2 layers/chip avg
        print(f"   {hbm_bw:>14.0f} {t0:>8.1f} {t1:>8.1f} {t2:>8.1f} {tps:>8.0f}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="4-FPGA Chip Distributed Pipeline Model"
    )
    parser.add_argument('--tokens', type=int, default=20,
                        help='Number of tokens to simulate')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose per-layer timing output')
    parser.add_argument('--sensitivity', '-s', action='store_true',
                        help='Run sensitivity analysis')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size (decode batch)')
    args = parser.parse_args()

    print()
    print("  Initializing 32-chip FPGA Inference Cluster...")

    cluster = FPGAInferenceCluster(seed=42)

    # Print chip → layer mapping
    print(f"  Loaded: {NUM_CARDS} cards × {CHIPS_PER_CARD} chips = {TOTAL_CHIPS} AGM 039-F")
    print(f"  Layers: {NUM_LAYERS} across 32 chips")

    # Verify layer assignment
    all_layers = []
    for chip in cluster.chips:
        all_layers.extend(chip.assigned_layers)
    assert sorted(all_layers) == list(range(NUM_LAYERS)), \
        f"Layer assignment error: {sorted(all_layers)} != {list(range(NUM_LAYERS))}"
    print(f"  Layer assignment verified: {len(all_layers)} layers mapped")

    # Verify expert assignment
    all_experts = []
    for chip in cluster.chips:
        all_experts.extend(chip.assigned_experts)
    assert sorted(all_experts) == list(range(NUM_EXPERTS)), \
        f"Expert assignment error"
    print(f"  Expert assignment verified: {len(all_experts)} experts mapped")

    print()
    print(f"  Running pipeline simulation ({args.tokens} tokens)...")

    engine = PipelineEngine(cluster, seed=42)
    results = engine.analyze_pipeline(num_tokens=args.tokens)

    print_pipeline_analysis(results, verbose=args.verbose)

    if args.sensitivity:
        run_sensitivity_analysis(cluster)

    return results


if __name__ == "__main__":
    results = main()
