"""
fpga_arch/pipeline.py — 10-stage pipeline engine with prefill/decode dual-path timing.

Key enhancements over the original extraction:
  1. Prefill vs Decode separated MAC counts (QK_dot is O(P^2) in prefill, O(B) in decode)
  2. Per-stage batch scaling granularity (projections scale, weight loads don't)
  3. C2C link contention model (parallel same-card, serial cross-card)
  4. Expert hit path enumeration (0/1/2 local) with full C2C dispatch/reduce timing
  5. Pipeline simulator: simulate N tokens flowing through 32 chips with contention
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from enum import Enum, auto
from collections import defaultdict
import numpy as np
import math

from .config import (
    NUM_LAYERS, NUM_EXPERTS, TOTAL_CHIPS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
    Q_LORA_RANK, KV_LORA_RANK, QK_ROPE_HEAD_DIM, O_LORA_RANK,
    NUM_ATTN_HEADS, QK_NOPE_HEAD_DIM, V_HEAD_DIM,
    NUM_EXPERTS_PER_TOK, SLIDING_WINDOW, TOP_K_EXPERTS, EXPERTS_PER_CHIP,
    TP_ATTN_PER_LAYER, DSP_TMACS, HBM_BW_GBPS, HBM_BW_EFF,
    MAC_MLA_TOTAL, MAC_MLA_Q_DOWN, MAC_MLA_KV_LATENT, MAC_MLA_KV_ROPE,
    MAC_MLA_QK_DOT, MAC_MLA_AV_DOT, MAC_MLA_O_DECOMPRESS, MAC_MLA_O_UP,
    MAC_EXPERT_TOTAL, MAC_SHARED_EXPERT, MAC_MOE_LAYER_TOTAL,
    MAC_EXPERT_GATE, MAC_EXPERT_UP, MAC_EXPERT_DOWN,
    C2C_MSG_DISPATCH_B, C2C_MAX_PAYLOAD_B, C2C_FRAME_OVERHEAD_B,
    C2C_LINK_BW_GBPS, C2C_HOP_LATENCY_NS,
    C2C_DISPATCH_LATENCY_NS, C2C_REDUCE_LATENCY_NS, C2C_FWD_LATENCY_NS,
    PCIE_P2P_BW_GBPS, PCIE_P2P_LATENCY_NS,
    DETERMINISTIC_MB_PER_LAYER, EXPERT_HBM_LOAD_MB,
    WEIGHT_GB_PER_CHIP,
    PIPELINE_TPS, BATCH1_TPS, K_PIPELINE,
    P_EXPERT_PER_CHIP, P_0_HIT, P_1_HIT, P_2P_HIT,
    CHIPS_PER_CARD,
    DSP_ATTN_FP8_TMACS, DSP_ATTN_FP4_TMACS, DSP_FFN_TMACS,
    PREFILL_ATTN_DENSITY, PREFILL_ATTN_SPARSITY,
    PREFILL_USE_FP4_ATTN, PREFILL_USE_SPARSE_ATTN,
    CPU_PCIE_LATENCY_US,
)
from .chip import FPGAChip
from .cluster import FPGACluster
from .interconnect import C2CMessageType, C2CMessage


# ============================================================================
# Enums & Data Classes
# ============================================================================

class StageType(Enum):
    WEIGHT_PREFETCH = auto()
    MLA_ATTENTION   = auto()
    ATTN_NORM       = auto()
    MOE_ROUTER      = auto()
    MOE_DISPATCH    = auto()
    SHARED_EXPERT   = auto()
    ROUTED_EXPERT   = auto()
    MOE_REDUCE      = auto()
    FFN_NORM        = auto()
    PIPELINE_FWD    = auto()


class ExpertHitPath(Enum):
    """How many of the top-6 experts are local to the current chip."""
    ZERO  = 0   # all 6 remote — worst case, 82.7%
    ONE   = 1   # 1 local + 5 remote — typical, 16.5%
    TWO_PLUS = 2  # 2+ local — rare, 0.8%


@dataclass
class StageTiming:
    stage_type: StageType
    dsp_time_us: float = 0.0
    hbm_time_us: float = 0.0
    c2c_time_us: float = 0.0
    sram_time_us: float = 0.0
    total_time_us: float = 0.0
    dsp_util_pct: float = 0.0
    notes: str = ""


@dataclass
class LayerTiming:
    layer_idx: int
    chip_id: int
    stages: List[StageTiming] = field(default_factory=list)
    c2c_messages: List[Dict] = field(default_factory=list)
    total_time_us: float = 0.0
    is_bottleneck: bool = False


@dataclass
class TokenTrace:
    token_id: int
    layer_timings: List[LayerTiming] = field(default_factory=list)
    total_latency_us: float = 0.0
    total_dsp_us: float = 0.0
    total_hbm_us: float = 0.0
    total_c2c_us: float = 0.0


@dataclass
class BatchResult:
    batch_size: int
    is_prefill: bool
    total_latency_us: float
    per_token_latency_us: float
    throughput_tps: float
    dsp_utilization_pct: float
    hbm_bandwidth_gbps: float
    c2c_messages: int
    expert_hits: Dict[int, int]


# ============================================================================
# MAC Breakdown — per-operation MAC counts with batch/prefill scaling
# ============================================================================

@dataclass
class MACBreakdown:
    """MAC counts for one layer, one chip (TP-adjusted), batch_size=1 decode.

    For prefill, QK_dot scales as P * KV_len (O(P^2)) instead of 1 * KV_len.
    For decode, all MACs scale linearly with batch_size B.

    Weight loads (HBM_MB) are per-layer, batch-independent (shared across batch).
    """
    # MLA — macros/compute per token (decode), or per-token-equivalent (prefill)
    q_down_macs: float = MAC_MLA_Q_DOWN / TP_ATTN_PER_LAYER       # H × Q_rank / TP
    kv_latent_macs: float = MAC_MLA_KV_LATENT / TP_ATTN_PER_LAYER # H × KV_rank / TP
    kv_rope_macs: float = MAC_MLA_KV_ROPE / TP_ATTN_PER_LAYER     # H × rope_dim / TP
    qk_dot_macs_per_kv: float = 29.88e6 / TP_ATTN_PER_LAYER       # Q·K^T per KV token
    av_dot_macs: float = MAC_MLA_AV_DOT / TP_ATTN_PER_LAYER       # A·V
    o_decompress_macs: float = MAC_MLA_O_DECOMPRESS / TP_ATTN_PER_LAYER
    o_up_macs: float = MAC_MLA_O_UP / TP_ATTN_PER_LAYER

    # Expert FFN — per expert per token
    expert_gate_macs: float = MAC_EXPERT_GATE   # H × inter
    expert_up_macs: float = MAC_EXPERT_UP       # H × inter
    expert_down_macs: float = MAC_EXPERT_DOWN   # inter × H
    expert_total_macs: float = MAC_EXPERT_TOTAL

    # Shared expert — per token (always local, always active)
    shared_expert_macs: float = MAC_SHARED_EXPERT / TP_ATTN_PER_LAYER

    # HBM weight loads (batch-independent, loaded once per batch)
    attn_weight_mb: float = 44.5    # attention weights per chip per layer (TP=2)
    shared_expert_mb: float = 15.0  # shared expert weights per chip
    expert_mb_per_local: float = EXPERT_HBM_LOAD_MB  # 33 MB per local expert

    def compute_prefill(self, prompt_len: int,
                         use_fp4_attn: bool = False,
                         attn_sparsity: float = 0.0,
                         n_requests: int = 0) -> Tuple[float, float, float, float]:
        """Compute DSP MACs and HBM MB for a prefill of prompt_len tokens.

        Returns: (attn_macs, ffn_macs, total_macs, total_hbm_mb)

        attn_macs: QK_dot + AV_dot, fp8×fp8 (or fp8×fp4 with P0)
        ffn_macs:  projections + shared expert + routed experts, always fp8×fp4

        P0 (use_fp4_attn):  K/V activations stay fp4 → 2 MAC/cycle for Q·K^T, A·V.
        P1 (attn_sparsity): Router-guided mask — only attend to (1-sparsity) of KV.

        When n_requests > 1: attention is per-request (causal within each request).
        QK_dot = n_requests × (P/B × P/(2B)) = P²/(2B), not P²/2.
        """
        if n_requests > 1:
            per_req = prompt_len / n_requests
            avg_kv_len = per_req / 2  # causal within each request
        else:
            avg_kv_len = prompt_len / 2  # single request: causal over full sequence
        density = 1.0 - attn_sparsity
        effective_kv_len = avg_kv_len * density

        # ── Projections (fp8 act × fp4 weight → DSP_FFN_TMACS) ──
        proj_macs = (self.q_down_macs + self.kv_latent_macs + self.kv_rope_macs +
                     self.o_decompress_macs + self.o_up_macs) * prompt_len

        # ── Attention dot products ──
        qk_macs = self.qk_dot_macs_per_kv * prompt_len * effective_kv_len
        av_macs = self.av_dot_macs * prompt_len * density

        attn_macs = qk_macs + av_macs

        # ── FFN (fp8 act × fp4 weight → DSP_FFN_TMACS) ──
        shared_macs = self.shared_expert_macs * prompt_len
        routed_macs = self.expert_total_macs * prompt_len * TOP_K_EXPERTS
        ffn_macs = proj_macs + shared_macs + routed_macs

        total_macs = attn_macs + ffn_macs
        total_hbm_mb = self.attn_weight_mb + self.shared_expert_mb

        return attn_macs, ffn_macs, total_macs, total_hbm_mb

    def compute_decode(self, batch_size: int, kv_len: int,
                       n_local_experts: int = 0) -> Tuple[float, float, float, float]:
        """Compute DSP MACs and HBM MB for a decode step.

        Returns: (total_macs, total_hbm_mb, attn_macs, expert_macs)

        All MACs scale with batch_size B. HBM weight loads are batch-independent.
        """
        # Projections: B tokens × per-token MACs
        proj_macs = (self.q_down_macs + self.kv_latent_macs + self.kv_rope_macs +
                     self.o_decompress_macs + self.o_up_macs) * batch_size

        # QK dot: B query tokens × kv_len keys each
        qk_macs = self.qk_dot_macs_per_kv * batch_size * min(kv_len, SLIDING_WINDOW)

        # AV dot: B × V
        av_macs = self.av_dot_macs * batch_size

        attn_macs = proj_macs + qk_macs + av_macs

        # Shared expert: B tokens
        shared_macs = self.shared_expert_macs * batch_size

        # Routed experts: B tokens × top-6 experts
        # HBM loads: only local experts need weight streaming
        routed_local_macs = self.expert_total_macs * batch_size * n_local_experts
        routed_remote_macs = self.expert_total_macs * batch_size * (TOP_K_EXPERTS - n_local_experts)
        expert_macs = shared_macs + routed_local_macs + routed_remote_macs

        total_macs = attn_macs + expert_macs
        total_hbm_mb = (self.attn_weight_mb + self.shared_expert_mb +
                        self.expert_mb_per_local * n_local_experts)

        return total_macs, total_hbm_mb, attn_macs, expert_macs


# ============================================================================
# C2C Link Contention Model
# ============================================================================

class C2CContentionModel:
    """Models link-level contention on the C2C Dual Ring + PCIe fabric.

    Same-card Ring A: 4 unidirectional links (0→1, 1→2, 2→3, 3→0 in one direction).
    Each link is independent — transfers using different links run in parallel.
    Cross-card: all traffic serializes through shared PCIe P2P bus.
    """

    # Ring A clockwise links: (src, dst)
    RING_LINKS = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # Which links does a transfer from src to dst use? (clockwise direction)
    @staticmethod
    def links_used(src_chip: int, dst_chip: int) -> List[Tuple[int, int]]:
        """Return ordered list of (link_src, link_dst) for src→dst clockwise."""
        if src_chip == dst_chip:
            return []
        dist = (dst_chip - src_chip) % 4
        links = []
        for i in range(dist):
            s = (src_chip + i) % 4
            d = (src_chip + i + 1) % 4
            links.append((s, d))
        return links

    @staticmethod
    def transfer_time_ns(payload_bytes: int, hops: int = 1) -> float:
        """Single-hop C2C transfer time in nanoseconds (no contention)."""
        frames = max(1, math.ceil(payload_bytes / C2C_MAX_PAYLOAD_B))
        avg_frame_bytes = payload_bytes / frames + C2C_FRAME_OVERHEAD_B
        serdes_ns = frames * avg_frame_bytes * 8 / C2C_LINK_BW_GBPS
        return serdes_ns + C2C_HOP_LATENCY_NS * hops

    @staticmethod
    def pcie_transfer_time_ns(payload_bytes: int) -> float:
        """PCIe P2P transfer time in nanoseconds."""
        serdes_ns = payload_bytes * 8 / PCIE_P2P_BW_GBPS
        return max(serdes_ns, PCIE_P2P_LATENCY_NS)

    @staticmethod
    def compute_moe_dispatch_time(
        src_chip_id: int, remote_dst_ids: List[int],
        cross_card_ids: Set[int], same_card_ids: Set[int],
        payload_bytes: int = C2C_MSG_DISPATCH_B
    ) -> float:
        """Compute MoE Dispatch time considering parallel same-card + serial cross-card.

        Same-card: parallel if using different ring links, otherwise serialized.
        Cross-card: all serial through shared PCIe bus.

        Returns time in nanoseconds.
        """
        # Same-card transfers: group by link usage, find bottleneck link
        link_loads: Dict[Tuple[int, int], float] = defaultdict(float)
        for dst_id in remote_dst_ids:
            if dst_id in same_card_ids:
                src_local = src_chip_id % CHIPS_PER_CARD
                dst_local = dst_id % CHIPS_PER_CARD
                links = C2CContentionModel.links_used(src_local, dst_local)
                t_ns = C2CContentionModel.transfer_time_ns(payload_bytes, len(links))
                for link in links:
                    link_loads[link] += t_ns
                if not links:  # same chip — negligible
                    pass

        # Same-card bottleneck: max link load
        same_card_time = max(link_loads.values()) if link_loads else 0.0

        # Cross-card: all serial, each at PCIe latency
        n_cross = sum(1 for did in remote_dst_ids if did in cross_card_ids)
        cross_card_time = n_cross * C2CContentionModel.pcie_transfer_time_ns(payload_bytes)

        # Same-card and cross-card can overlap (different physical interfaces)
        return max(same_card_time, cross_card_time)

    @staticmethod
    def compute_pipeline_fwd_time(
        src_chip_id: int, dst_chip_id: int,
        payload_bytes: int = C2C_MSG_DISPATCH_B
    ) -> float:
        """Compute time for a single Pipeline_Fwd message (nanoseconds).

        Same card: C2C ring. Cross card: PCIe P2P.
        """
        src_card = src_chip_id // CHIPS_PER_CARD
        dst_card = dst_chip_id // CHIPS_PER_CARD

        if src_card == dst_card:
            src_local = src_chip_id % CHIPS_PER_CARD
            dst_local = dst_chip_id % CHIPS_PER_CARD
            hops = len(C2CContentionModel.links_used(src_local, dst_local))
            return C2CContentionModel.transfer_time_ns(payload_bytes, max(1, hops))
        else:
            return C2CContentionModel.pcie_transfer_time_ns(payload_bytes)


# ============================================================================
# Detailed Per-Stage Timing Functions
# ============================================================================

def detailed_stage_timing(
    stage: StageType, chip: FPGAChip, cluster: FPGACluster,
    layer_idx: int, batch_size: int, is_prefill: bool,
    prompt_len: int = 0, kv_len: int = 0,
    expert_selection: Optional[List[int]] = None,
    mac_breakdown: Optional[MACBreakdown] = None,
) -> StageTiming:
    """Compute detailed timing for one pipeline stage.

    Considers:
      - DSP: MACs with proper batch/prefill scaling
      - HBM: weight loads (batch-independent, shared)
      - C2C: link contention (parallel same-card, serial cross-card)
      - SRAM: deterministic weight access (negligible)

    Args:
        stage: which pipeline stage
        chip: the chip processing this layer
        cluster: the full cluster (for C2C routing)
        layer_idx: which layer
        batch_size: decode batch size, or 1 for single-token trace
        is_prefill: True for prefill, False for decode
        prompt_len: number of prompt tokens (prefill only)
        kv_len: current KV cache length (decode only)
        expert_selection: top-6 expert indices
        mac_breakdown: pre-computed MAC breakdown (created if None)
    """
    if mac_breakdown is None:
        mac_breakdown = MACBreakdown()

    timing = StageTiming(stage_type=stage)

    # ── Stage-specific timing ──

    if stage == StageType.WEIGHT_PREFETCH:
        # SRAM deterministic weight access only (expert HBM in RoutedExpert)
        n_local = cluster.count_local_experts(chip, expert_selection or [])
        timing.sram_time_us = 0.1
        timing.total_time_us = 0.1
        timing.notes = f"SRAM {DETERMINISTIC_MB_PER_LAYER:.0f}MB, {n_local} local experts"

    elif stage == StageType.MLA_ATTENTION:
        if is_prefill:
            attn_macs, ffn_macs, total_macs, _ = mac_breakdown.compute_prefill(
                prompt_len, use_fp4_attn=PREFILL_USE_FP4_ATTN,
                attn_sparsity=PREFILL_ATTN_SPARSITY if PREFILL_USE_SPARSE_ATTN else 0.0)
            attn_rate = DSP_ATTN_FP4_TMACS if PREFILL_USE_FP4_ATTN else DSP_ATTN_FP8_TMACS
            ffn_rate = DSP_FFN_TMACS
            timing.dsp_time_us = (attn_macs / (attn_rate * 1e12) + ffn_macs / (ffn_rate * 1e12)) * 1e6
        else:
            macs, _, _, _ = mac_breakdown.compute_decode(batch_size, kv_len)
            timing.dsp_time_us = macs / (DSP_TMACS * 1e12) * 1e6

        timing.sram_time_us = 0.2  # KV cache SRAM access
        timing.hbm_time_us = 0.0   # attention weights in SRAM
        timing.total_time_us = max(timing.dsp_time_us, timing.sram_time_us)
        timing.notes = (f"prefill={is_prefill}, batch={batch_size}, "
                        f"DSP={timing.dsp_time_us:.1f}us")

    elif stage == StageType.ATTN_NORM:
        macs = HIDDEN_SIZE * 4 * batch_size if not is_prefill else HIDDEN_SIZE * 4 * prompt_len
        timing.dsp_time_us = macs / (DSP_TMACS * 1e12) * 1e6
        timing.total_time_us = timing.dsp_time_us

    elif stage == StageType.MOE_ROUTER:
        n_tokens = prompt_len if is_prefill else batch_size
        router_macs = HIDDEN_SIZE * NUM_EXPERTS * n_tokens
        timing.dsp_time_us = router_macs / (DSP_TMACS * 1e12) * 1e6
        timing.sram_time_us = 0.05
        timing.total_time_us = max(timing.dsp_time_us, timing.sram_time_us)

    elif stage == StageType.MOE_DISPATCH:
        if not expert_selection:
            timing.total_time_us = 0.0
            return timing

        experts_by_chip = cluster.dispatch_experts(chip, expert_selection)
        remote_ids = [cid for cid in experts_by_chip if cid != chip.global_id]

        if not remote_ids:
            timing.total_time_us = 0.0
            timing.notes = "all experts local"
            return timing

        # Group by same-card vs cross-card
        same_card_ids = {cid for cid in remote_ids
                        if cluster.is_same_card(chip, cluster.chips[cid])}
        cross_card_ids = set(remote_ids) - same_card_ids

        c2c_ns = C2CContentionModel.compute_moe_dispatch_time(
            chip.global_id, remote_ids, cross_card_ids, same_card_ids
        )
        timing.c2c_time_us = c2c_ns / 1000.0
        timing.total_time_us = timing.c2c_time_us
        timing.notes = (f"{len(remote_ids)} remote ({len(same_card_ids)} same-card, "
                        f"{len(cross_card_ids)} cross-card)")

    elif stage == StageType.SHARED_EXPERT:
        n_tokens = prompt_len if is_prefill else batch_size
        macs = mac_breakdown.shared_expert_macs * n_tokens
        timing.dsp_time_us = macs / (DSP_TMACS * 1e12) * 1e6
        timing.sram_time_us = 0.1   # weights in SRAM
        timing.hbm_time_us = 0.0
        timing.total_time_us = max(timing.dsp_time_us, timing.sram_time_us)

    elif stage == StageType.ROUTED_EXPERT:
        n_local = cluster.count_local_experts(chip, expert_selection or [])
        if n_local == 0:
            timing.total_time_us = 0.0
            timing.notes = f"no local experts (chip {chip.global_id})"
            return timing

        n_tokens = prompt_len if is_prefill else batch_size
        # DSP: local experts only (remote experts computed on their chips)
        macs = mac_breakdown.expert_total_macs * n_tokens * n_local
        timing.dsp_time_us = macs / (DSP_TMACS * 1e12) * 1e6

        # HBM: stream expert weights from HBM (batch-independent)
        hbm_mb = mac_breakdown.expert_mb_per_local * n_local
        timing.hbm_time_us = hbm_mb / (HBM_BW_GBPS * HBM_BW_EFF / 1024)

        timing.total_time_us = max(timing.dsp_time_us, timing.hbm_time_us)
        if timing.total_time_us > 0:
            timing.dsp_util_pct = timing.dsp_time_us / timing.total_time_us * 100
        timing.notes = f"{n_local} local experts, {hbm_mb:.0f}MB HBM"

    elif stage == StageType.MOE_REDUCE:
        if not expert_selection:
            timing.total_time_us = 0.0
            return timing

        experts_by_chip = cluster.dispatch_experts(chip, expert_selection)
        remote_ids = [cid for cid in experts_by_chip if cid != chip.global_id]
        if not remote_ids:
            timing.total_time_us = 0.0
            return timing

        same_card_ids = {cid for cid in remote_ids
                        if cluster.is_same_card(chip, cluster.chips[cid])}
        cross_card_ids = set(remote_ids) - same_card_ids

        c2c_ns = C2CContentionModel.compute_moe_dispatch_time(
            chip.global_id, remote_ids, cross_card_ids, same_card_ids
        )
        timing.c2c_time_us = c2c_ns / 1000.0
        timing.total_time_us = timing.c2c_time_us
        timing.notes = f"reduce from {len(remote_ids)} remote chips"

    elif stage == StageType.FFN_NORM:
        n_tokens = prompt_len if is_prefill else batch_size
        macs = HIDDEN_SIZE * 4 * n_tokens
        timing.dsp_time_us = macs / (DSP_TMACS * 1e12) * 1e6
        timing.total_time_us = timing.dsp_time_us

    elif stage == StageType.PIPELINE_FWD:
        next_layer = layer_idx + 1
        if next_layer >= NUM_LAYERS:
            timing.total_time_us = 0.0
            timing.notes = "final layer"
        else:
            next_chip = cluster.get_chip_for_layer(next_layer)
            if next_chip.global_id == chip.global_id:
                timing.total_time_us = 0.001  # same chip, register transfer
                timing.notes = f"same chip L{layer_idx}->L{next_layer}"
            else:
                c2c_ns = C2CContentionModel.compute_pipeline_fwd_time(
                    chip.global_id, next_chip.global_id
                )
                timing.c2c_time_us = c2c_ns / 1000.0
                timing.total_time_us = timing.c2c_time_us
                timing.notes = f"fwd L{layer_idx}->L{next_layer} "

    return timing


# ============================================================================
# Detailed Layer & Pipeline Timing
# ============================================================================

def detailed_layer_timing(
    cluster: FPGACluster, layer_idx: int,
    batch_size: int, is_prefill: bool,
    prompt_len: int = 0, kv_len: int = 0,
    expert_selection: Optional[List[int]] = None,
    expert_path: ExpertHitPath = ExpertHitPath.ONE,
) -> LayerTiming:
    """Compute detailed timing for one layer, all 10 stages.

    Uses proper MAC scaling per stage and C2C contention modeling.

    Args:
        expert_path: which expert hit scenario (ZERO/ONE/TWO_PLUS).
                     Determines how many selected experts are local to this chip.
    """
    chip = cluster.get_chip_for_layer(layer_idx)
    mac_bd = MACBreakdown()

    # Determine expert selection based on path
    if expert_selection is None:
        # Generate experts consistent with the specified path
        n_local = {ExpertHitPath.ZERO: 0, ExpertHitPath.ONE: 1,
                   ExpertHitPath.TWO_PLUS: 2}[expert_path]
        # local experts from this chip's assignment
        local_experts = list(np.random.choice(
            chip.assigned_experts, size=min(n_local, len(chip.assigned_experts)),
            replace=False
        ))
        # remote experts from other chips
        other_experts = [e for e in range(NUM_EXPERTS)
                         if e not in chip.assigned_experts]
        remote_experts = list(np.random.choice(
            other_experts, size=TOP_K_EXPERTS - n_local, replace=False
        ))
        expert_selection = sorted(local_experts + remote_experts)

    layer_timing = LayerTiming(layer_idx=layer_idx, chip_id=chip.global_id)

    stage_order = [
        StageType.WEIGHT_PREFETCH, StageType.MLA_ATTENTION, StageType.ATTN_NORM,
        StageType.MOE_ROUTER, StageType.MOE_DISPATCH, StageType.SHARED_EXPERT,
        StageType.ROUTED_EXPERT, StageType.MOE_REDUCE, StageType.FFN_NORM,
        StageType.PIPELINE_FWD,
    ]

    for st in stage_order:
        timing = detailed_stage_timing(
            st, chip, cluster, layer_idx, batch_size, is_prefill,
            prompt_len, kv_len, expert_selection, mac_bd
        )
        layer_timing.stages.append(timing)

        # Track C2C messages for cross-card analysis
        if timing.c2c_time_us > 0:
            is_cross = False
            if st == StageType.MOE_DISPATCH or st == StageType.MOE_REDUCE:
                experts_by_chip = cluster.dispatch_experts(chip, expert_selection or [])
                remote_ids = [cid for cid in experts_by_chip if cid != chip.global_id]
                is_cross = any(
                    cid // CHIPS_PER_CARD != chip.global_id // CHIPS_PER_CARD
                    for cid in remote_ids
                ) if remote_ids else False
            elif st == StageType.PIPELINE_FWD:
                next_layer = layer_idx + 1
                if next_layer < NUM_LAYERS:
                    next_chip = cluster.get_chip_for_layer(next_layer)
                    is_cross = (next_chip.global_id // CHIPS_PER_CARD !=
                               chip.global_id // CHIPS_PER_CARD)
            layer_timing.c2c_messages.append({
                'stage': st.name,
                'chip_id': chip.global_id,
                'cross_card': is_cross,
                'time_us': timing.c2c_time_us,
            })

    layer_timing.total_time_us = sum(s.total_time_us for s in layer_timing.stages)
    return layer_timing


# ============================================================================
# Pipeline Simulator: tokens flowing through 32-chip pipeline
# ============================================================================

@dataclass
class PipelineSimResult:
    """Result of a pipeline simulation run."""
    num_tokens: int
    is_prefill: bool
    batch_size: int                                      # decode batch size (1 for prefill sim)
    prompt_len: int = 0                                  # prefill prompt length

    # Timing
    avg_token_latency_us: float = 0.0
    bottleneck_per_layer_us: float = 0.0
    bottleneck_chip_id: int = 0
    throughput_tps: float = 0.0

    # Stage breakdown (averaged across all tokens and layers)
    stage_avg_us: Dict[str, float] = field(default_factory=dict)

    # Resource utilization
    dsp_util_pct: float = 0.0
    hbm_read_gbps: float = 0.0

    # Expert hit stats
    expert_hit_dist: Dict[int, int] = field(default_factory=dict)

    # C2C stats
    c2c_messages_per_token: float = 0.0
    cross_card_pct: float = 0.0

    # Per-chip load
    chip_load_us: Dict[int, float] = field(default_factory=dict)


def simulate_pipeline(
    cluster: FPGACluster,
    num_tokens: int = 50,
    batch_size: int = 1,
    is_prefill: bool = False,
    prompt_len: int = 0,
    seed: int = 42,
) -> PipelineSimResult:
    """Simulate tokens flowing through the full 32-chip pipeline.

    For prefill: simulates `num_tokens` prefill requests (each with prompt_len tokens).
                 Prefill processes ALL prompt tokens in parallel through 61 layers.

    For decode: simulates `num_tokens` decode steps, each with `batch_size` tokens
                from concurrent requests flowing in pipeline.

    Returns detailed timing, bottleneck analysis, and resource utilization.

    Key insight: in the 32-chip pipeline, up to 32 tokens can be in-flight
    simultaneously (one per chip at different layers). The bottleneck chip
    determines the pipeline rate: throughput = 1 / max(per_layer_time_on_chip).
    """
    rng = np.random.RandomState(seed)
    mac_bd = MACBreakdown()

    # For each token, track which chip it's on and at what time
    # Simplification: model the steady-state pipeline where tokens enter at rate R
    # and each chip takes T_chip time per token.

    # Step 1: Compute per-layer timing for each layer under the given scenario
    # Step 2: Sum per-chip (chip hosts ~2 layers)
    # Step 3: Bottleneck chip determines throughput

    chip_total_time: Dict[int, float] = defaultdict(float)
    stage_times: Dict[StageType, List[float]] = defaultdict(list)
    expert_hit_counts = {0: 0, 1: 0, 2: 0}
    total_c2c_msgs = 0
    cross_card_msgs = 0
    total_dsp_us = 0.0
    total_hbm_mb = 0.0
    total_wall_us = 0.0

    for token_idx in range(num_tokens):
        # Sample expert hit path based on probabilities
        p = rng.random()
        if p < P_0_HIT:
            hit_path = ExpertHitPath.ZERO
            expert_hit_counts[0] += 1
        elif p < P_0_HIT + P_1_HIT:
            hit_path = ExpertHitPath.ONE
            expert_hit_counts[1] += 1
        else:
            hit_path = ExpertHitPath.TWO_PLUS
            expert_hit_counts[2] += 1

        # Compute kv_len for decode (grows as tokens are generated)
        kv_len = min(token_idx + 1, SLIDING_WINDOW) if not is_prefill else prompt_len

        for layer_idx in range(NUM_LAYERS):
            chip = cluster.get_chip_for_layer(layer_idx)
            lt = detailed_layer_timing(
                cluster, layer_idx, batch_size, is_prefill,
                prompt_len, kv_len, expert_path=hit_path
            )

            chip_total_time[chip.global_id] += lt.total_time_us
            total_wall_us += lt.total_time_us

            for s in lt.stages:
                stage_times[s.stage_type].append(s.total_time_us)
                total_dsp_us += s.dsp_time_us
                if s.hbm_time_us > 0:
                    total_hbm_mb += s.hbm_time_us * HBM_BW_GBPS / 1024

            # Count C2C messages from the layer's tracked message list
            total_c2c_msgs += len(lt.c2c_messages)
            cross_card_msgs += sum(1 for m in lt.c2c_messages if m.get('cross_card'))

    # Average across tokens
    avg_chip_time = {cid: t / num_tokens for cid, t in chip_total_time.items()}
    bottleneck_chip = max(avg_chip_time, key=avg_chip_time.get)
    bottleneck_per_layer = avg_chip_time[bottleneck_chip] / max(1, len(
        cluster.chips[bottleneck_chip].assigned_layers
    ))

    # Throughput: bottleneck_per_layer_us is time to process B tokens through one layer.
    # Pipeline processes 1/bottleneck_per_layer batches/s, each with batch_size tokens.
    throughput = batch_size * 1e6 / bottleneck_per_layer if bottleneck_per_layer > 0 else float('inf')

    # Stage averages
    avg_stages = {}
    for st, times in stage_times.items():
        avg_stages[st.name] = np.mean(times)

    # DSP utilization
    dsp_util = (total_dsp_us / total_wall_us * 100) if total_wall_us > 0 else 0

    return PipelineSimResult(
        num_tokens=num_tokens,
        is_prefill=is_prefill,
        batch_size=batch_size,
        prompt_len=prompt_len,
        avg_token_latency_us=sum(avg_chip_time.values()),
        bottleneck_per_layer_us=bottleneck_per_layer,
        bottleneck_chip_id=bottleneck_chip,
        throughput_tps=throughput,
        stage_avg_us=avg_stages,
        dsp_util_pct=dsp_util,
        hbm_read_gbps=total_hbm_mb / 1024 / (total_wall_us / 1e6) if total_wall_us > 0 else 0,
        expert_hit_dist=expert_hit_counts,
        c2c_messages_per_token=total_c2c_msgs / num_tokens,
        cross_card_pct=cross_card_msgs / max(1, total_c2c_msgs) * 100,
        chip_load_us=dict(sorted(avg_chip_time.items())),
    )


def print_pipeline_result(result: PipelineSimResult):
    """Pretty-print a pipeline simulation result."""
    mode = "PREFILL" if result.is_prefill else "DECODE"
    print(f"  --- Pipeline Simulation: {mode} ---")
    print(f"  Tokens simulated:   {result.num_tokens}")
    if result.is_prefill:
        print(f"  Prompt length:      {result.prompt_len}")
    else:
        print(f"  Batch size:         {result.batch_size}")
    print(f"  Avg token latency:  {result.avg_token_latency_us:.0f} us "
          f"({result.avg_token_latency_us/1000:.2f} ms)")
    print(f"  Bottleneck chip:    {result.bottleneck_chip_id} "
          f"({result.bottleneck_per_layer_us:.1f} us/layer)")
    print(f"  Throughput:         {result.throughput_tps:.0f} tok/s")
    print(f"  DSP utilization:    {result.dsp_util_pct:.1f}%")
    print(f"  HBM read:           {result.hbm_read_gbps:.1f} GB/s")

    print(f"\n  Stage breakdown (avg per stage per layer):")
    stage_names_order = [
        'MLA_ATTENTION', 'ROUTED_EXPERT', 'SHARED_EXPERT',
        'MOE_ROUTER', 'MOE_DISPATCH', 'MOE_REDUCE',
        'PIPELINE_FWD', 'WEIGHT_PREFETCH', 'ATTN_NORM', 'FFN_NORM',
    ]
    for name in stage_names_order:
        if name in result.stage_avg_us:
            us = result.stage_avg_us[name]
            if us > 0.01:
                print(f"    {name:<20s} {us:>8.2f} us")

    print(f"\n  Expert hits: 0={result.expert_hit_dist[0]} "
          f"1={result.expert_hit_dist[1]} 2+={result.expert_hit_dist[2]}")
    print(f"  C2C msgs/token: {result.c2c_messages_per_token:.1f} "
          f"(cross-card: {result.cross_card_pct:.1f}%)")


# ============================================================================
# Pipeline Engine (keeps backward-compatible interface)
# ============================================================================

class PipelineEngine:
    """Orchestrates token processing with dual-path timing.

    Fast path:   throughput_model(B) for scheduler O(1) queries.
    Detailed path: simulate_pipeline() for calibration and deep-dive analysis.
    """

    def __init__(self, cluster: FPGACluster, seed: int = 42):
        self.cluster = cluster
        self.rng = np.random.RandomState(seed)

        # 解法 A: recompute K_PIPELINE if cluster has expert replication
        self._k_pipeline = K_PIPELINE
        if cluster.expert_replication == 'hot':
            self._k_pipeline = self._recompute_k_with_replicas()
        # Patch class-level K so static throughput_model/decode_latency_model
        # callers see the override (used by concurrent_pipeline_model et al.)
        PipelineEngine._active_k_pipeline = self._k_pipeline

    def _recompute_k_with_replicas(self) -> float:
        """Recompute K_PIPELINE given increased local expert hit rate.

        K = PIPELINE_TPS / BATCH1_TPS - 1 captures the pipeline fill overhead.
        More local expert hits → less C2C dispatch/reduce per layer → lower
        per-layer latency at B=1 → higher BATCH1_TPS → lower K.

        Original P(0 local) = 82.7%, P(1) = 16.5%, P(2+) = 0.8% (12 experts/chip).
        With replication, we recompute from the actual cluster layout.
        """
        # Expected local expert hits per layer per token
        total_local_hits = 0
        n_samples = 1000  # Monte Carlo estimate
        for _ in range(n_samples):
            selection = self.cluster.sample_expert_selection()
            # Pick a random chip (each layer is on a specific chip, but average over all)
            chip = self.cluster.chips[self.rng.randint(TOTAL_CHIPS)]
            total_local_hits += self.cluster.count_local_experts(chip, selection)
        avg_local = total_local_hits / n_samples

        # Original avg_local at 12 experts/chip: 6 * 12/384 ≈ 0.187
        # With replication: higher
        orig_local = TOP_K_EXPERTS * EXPERTS_PER_CHIP / NUM_EXPERTS  # 0.1875

        # Per-layer C2C time scales with remote experts: (TOP_K - local)
        # Reduction: K scales roughly as 1 + (C2C_overhead_fraction / compute_fraction)
        # Simplified: K_new = K_old * (TOP_K - avg_local) / (TOP_K - orig_local)
        if avg_local >= TOP_K_EXPERTS:
            return 0.0  # all local — no C2C overhead

        remote_ratio = (TOP_K_EXPERTS - avg_local) / max(0.01, TOP_K_EXPERTS - orig_local)
        k_new = K_PIPELINE * remote_ratio
        return k_new

    @property
    def k_pipeline(self) -> float:
        return self._k_pipeline

    # ── Fast analytical model: Decode ──
    # K_PIPELINE may be reduced when expert replication is enabled.
    # We store the active K on the class so static callers see the override.
    _active_k_pipeline: float = K_PIPELINE

    @staticmethod
    def throughput_model(batch_size: int) -> float:
        """Decode TPS(B) = PIPELINE_TPS * B / (B + K_PIPELINE).

        Uses class-level _active_k_pipeline (overridden when expert replication
        is enabled — see PipelineEngine.__init__).
        """
        if batch_size <= 0:
            return 0.0
        return PIPELINE_TPS * batch_size / (batch_size + PipelineEngine._active_k_pipeline)

    @staticmethod
    def decode_latency_model(batch_size: int) -> float:
        """Per-token decode latency for given batch size (microseconds)."""
        tps = PipelineEngine.throughput_model(batch_size)
        return 1e6 / tps if tps > 0 else float('inf')

    # Backward-compatible alias
    @staticmethod
    def latency_model(batch_size: int) -> float:
        return PipelineEngine.decode_latency_model(batch_size)

    # ── Fast analytical model: Prefill ──

    @staticmethod
    def prefill_latency_model(prompt_tokens: int,
                               use_fp4_attn: bool = False,
                               attn_sparsity: float = 0.0,
                               n_requests: int = 0) -> float:
        """Prefill pipeline latency for `prompt_tokens` tokens (microseconds).

        Prefill is compute-bound: O(P²) QK attention dominates.
        Split-rate model: attention ops (Q·K^T, A·V) use one DSP rate,
        FFN/projection ops use another (2× faster, fp8×fp4).

        P0 (use_fp4_attn):  K/V → fp4, attention goes from 5.54→11.07 TMACS.
        P1 (attn_sparsity): Router-guided sparse mask skips this fraction of KV.

        When n_requests > 1: attention is per-request (causal within each),
        reducing total QK macs from O(P²) to O(P²/B).

        Returns total time for the prefill batch to traverse all layers.
        """
        if prompt_tokens <= 0:
            return 0.0
        mac = MACBreakdown()
        attn_macs, ffn_macs, _, _ = mac.compute_prefill(
            prompt_tokens, use_fp4_attn=use_fp4_attn, attn_sparsity=attn_sparsity,
            n_requests=n_requests)

        # Attention ops: fp8×fp8 baseline (DSP_ATTN_FP8_TMACS) or fp8×fp4 with P0
        attn_rate = DSP_ATTN_FP4_TMACS if use_fp4_attn else DSP_ATTN_FP8_TMACS
        # FFN + projections: always fp8 act × fp4 weight (DSP_FFN_TMACS)
        ffn_rate = DSP_FFN_TMACS

        per_layer_us = (attn_macs / (attn_rate * 1e12) + ffn_macs / (ffn_rate * 1e12)) * 1e6
        return per_layer_us * NUM_LAYERS

    @staticmethod
    def prefill_tps_model(prompt_tokens: int,
                           use_fp4_attn: bool = False,
                           attn_sparsity: float = 0.0,
                           n_requests: int = 0) -> float:
        """Prefill throughput in tokens/s for given prompt length.

        Throughput limited by bottleneck chip (2 layers, ~1/32 of pipeline).
        Uses split-rate model: attention vs FFN DSP rates.
        """
        if prompt_tokens <= 0:
            return 0.0
        mac = MACBreakdown()
        attn_macs, ffn_macs, _, _ = mac.compute_prefill(
            prompt_tokens, use_fp4_attn=use_fp4_attn, attn_sparsity=attn_sparsity,
            n_requests=n_requests)

        attn_rate = DSP_ATTN_FP4_TMACS if use_fp4_attn else DSP_ATTN_FP8_TMACS
        ffn_rate = DSP_FFN_TMACS

        per_layer_us = (attn_macs / (attn_rate * 1e12) + ffn_macs / (ffn_rate * 1e12)) * 1e6
        layers_per_chip = max(1, round(NUM_LAYERS / TOTAL_CHIPS))
        bottleneck_us = per_layer_us * layers_per_chip
        return prompt_tokens * 1e6 / bottleneck_us if bottleneck_us > 0 else float('inf')

    # ── Chunked Prefill ──

    @staticmethod
    def chunked_prefill_model(total_prompt_tokens: int, chunk_size: int = 128,
                               use_fp4_attn: bool = True,
                               attn_sparsity: float = 0.888,
                               n_requests: int = 0) -> Dict:
        """Compute per-chunk latency for chunked prefill.

        Splits `total_prompt_tokens` into ceil(P/chunk_size) chunks.
        First chunk determines TTFT; chunks are pipelined across 32 chips.

        When n_requests > 1: attention is per-request (causal within each).
        avg_kv_len is divided by n_requests, reducing QK macs by 1/B.

        Returns dict with:
          - ttft_ms: time-to-first-token (first chunk latency)
          - chunk_latencies_ms: per-chunk latency list
          - total_prefill_ms: total prefill wall-clock (with pipeline parallelism)
          - effective_tps: total_prompt_tokens / total_prefill_ms * 1000
          - num_chunks: number of chunks
        """
        if total_prompt_tokens <= 0:
            return {'ttft_ms': 0, 'chunk_latencies_ms': [], 'total_prefill_ms': 0,
                    'effective_tps': 0, 'num_chunks': 0}

        mac = MACBreakdown()
        num_chunks = max(1, math.ceil(total_prompt_tokens / chunk_size))
        chunk_latencies = []
        reqs = max(1, n_requests)  # divide KV by this for per-request attention

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_prompt_tokens)
            n_new = end - start  # tokens in this chunk
            accumulated_kv = start  # KV entries from previous chunks

            # avg KV length per token: per-request when batched
            avg_kv_len = (accumulated_kv + n_new / 2) / reqs

            # ── Projections (scale with n_new) ──
            proj_macs = (mac.q_down_macs + mac.kv_latent_macs + mac.kv_rope_macs +
                         mac.o_decompress_macs + mac.o_up_macs) * n_new

            # ── Attention dot products ──
            density = 1.0 - attn_sparsity
            qk_macs = mac.qk_dot_macs_per_kv * n_new * avg_kv_len * density
            av_macs = mac.av_dot_macs * n_new * density

            attn_macs = qk_macs + av_macs

            # ── FFN ──
            shared_macs = mac.shared_expert_macs * n_new
            routed_macs = mac.expert_total_macs * n_new * TOP_K_EXPERTS
            ffn_macs = proj_macs + shared_macs + routed_macs

            # ── Timing ──
            attn_rate = DSP_ATTN_FP4_TMACS if use_fp4_attn else DSP_ATTN_FP8_TMACS
            ffn_rate = DSP_FFN_TMACS
            per_layer_us = (attn_macs / (attn_rate * 1e12) +
                            ffn_macs / (ffn_rate * 1e12)) * 1e6

            # Pipeline latency: all layers × per_layer
            chunk_latency_ms = per_layer_us * NUM_LAYERS / 1000.0
            chunk_latencies.append(chunk_latency_ms)

        # TTFT = first chunk latency
        ttft_ms = chunk_latencies[0]

        # Total prefill wall-clock: pipelined across 32 chips.
        # Chunk 0 traverses all 61 layers (TTFT). Subsequent chunks start
        # as soon as chip 0 finishes the previous chunk.
        #   total = chunk_0_latency + sum_{i=1}^{N-1} bottleneck_time_i
        # where bottleneck_time_i = chunk_i_per_layer × layers_per_chip
        layers_per_chip = max(1, round(NUM_LAYERS / TOTAL_CHIPS))
        total_prefill_ms = chunk_latencies[0]
        for i in range(1, num_chunks):
            total_prefill_ms += chunk_latencies[i] * layers_per_chip / NUM_LAYERS

        effective_tps = (total_prompt_tokens / total_prefill_ms * 1000
                         if total_prefill_ms > 0 else float('inf'))

        return {
            'ttft_ms': ttft_ms,
            'chunk_latencies_ms': chunk_latencies,
            'total_prefill_ms': total_prefill_ms,
            'effective_tps': effective_tps,
            'num_chunks': num_chunks,
        }

    @staticmethod
    def chunked_prefill_ttft(total_prompt_tokens: int, chunk_size: int = 128,
                              use_fp4_attn: bool = True,
                              attn_sparsity: float = 0.888,
                              n_requests: int = 0) -> float:
        """TTFT for chunked prefill (microseconds). First chunk determines TTFT."""
        if total_prompt_tokens <= 0:
            return 0.0
        first_chunk = min(chunk_size, total_prompt_tokens)
        return PipelineEngine.prefill_latency_model(
            first_chunk, use_fp4_attn=use_fp4_attn, attn_sparsity=attn_sparsity,
            n_requests=n_requests)

    # ── Superscalar Prefill Interleaving ──

    @staticmethod
    def prefill_chip0_bottleneck_us(chunk_size: int = 128,
                                     use_fp4_attn: bool = True,
                                     attn_sparsity: float = 0.888,
                                     n_requests: int = 0) -> float:
        """Time chip 0 spends processing one prefill chunk (microseconds).

        This determines the minimum interval between prefill admissions in
        superscalar mode. After chip 0 finishes one chunk, it can accept a
        chunk from a DIFFERENT prefill batch — multiple prefills interleave.

        Returns: bottleneck_per_chip_us = per_layer_us × layers_per_chip
        """
        if chunk_size <= 0:
            return 0.0
        per_layer_us = PipelineEngine.prefill_latency_model(
            chunk_size, use_fp4_attn=use_fp4_attn, attn_sparsity=attn_sparsity,
            n_requests=n_requests) / NUM_LAYERS
        layers_per_chip = max(1, round(NUM_LAYERS / TOTAL_CHIPS))
        return per_layer_us * layers_per_chip

    # ── Chip 0 admission rate modeling (with parallelism options) ──

    @staticmethod
    def chip0_admission_rate(chunk_size: int = 128,
                              use_fp4_attn: bool = True,
                              attn_sparsity: float = 0.888,
                              n_requests: int = 0,
                              chip0_parallelism: int = 1,
                              embedding_offload: bool = False,
                              embedding_cycles_us: float = 50.0) -> Dict:
        """Model chip 0 prefill admission throughput under various architectural
        optimizations.

        Parameters:
          chip0_parallelism: number of independent chip-0 instances (Pipeline
                             Cloning of just the entry stage). E.g. 2 means
                             splitting the cluster's 32 chips into 2 pipelines
                             each with their own chip 0.
          embedding_offload: if True, embedding+tokenize stage runs on a separate
                             unit (CPU or dedicated FPGA logic), removing it
                             from chip 0's critical path.
          embedding_cycles_us: estimated time saved if embedding_offload=True.

        Returns dict:
          per_chunk_us:        time for one chunk through chip 0
          admission_chunks_s:  chunks/sec across all parallel chip 0 instances
          admission_reqs_s:    requests/sec at P=512 (4 chunks each)
          admission_tps:       token throughput (tokens/sec of prefill)
        """
        if chip0_parallelism < 1:
            chip0_parallelism = 1

        per_chunk_us = PipelineEngine.prefill_chip0_bottleneck_us(
            chunk_size=chunk_size, use_fp4_attn=use_fp4_attn,
            attn_sparsity=attn_sparsity, n_requests=n_requests)

        if embedding_offload:
            per_chunk_us = max(1.0, per_chunk_us - embedding_cycles_us)

        chunks_per_sec_one = 1e6 / per_chunk_us if per_chunk_us > 0 else 0
        admission_chunks_s = chunks_per_sec_one * chip0_parallelism

        # At P=512, each prefill needs ceil(512/128) = 4 chunks on chip 0
        chunks_per_prefill = 4
        admission_reqs_s = admission_chunks_s / chunks_per_prefill
        admission_tps = admission_chunks_s * chunk_size

        return {
            'per_chunk_us': per_chunk_us,
            'parallelism': chip0_parallelism,
            'embedding_offload': embedding_offload,
            'admission_chunks_s': admission_chunks_s,
            'admission_reqs_s': admission_reqs_s,
            'admission_tps': admission_tps,
        }

    # ── CPU-FPGA Hybrid Prefill (P2) ──

    @staticmethod
    def cpu_hybrid_prefill_model(prompt_tokens: int,
                                  cpu_tflops: float = 3.0,
                                  use_fp4_attn: bool = True,
                                  attn_sparsity: float = 0.888,
                                  chunk_size: int = 0,
                                  n_requests: int = 0) -> Dict:
        """CPU-FPGA hybrid prefill (P2): CPU handles Q·K^T + A·V, FPGA handles FFN.

        Pipeline model:
          - FPGA: Q/K/V projections → wait for CPU → O projection + FFN
          - CPU:  receives Q,K via PCIe → Q·K^T + A·V → sends attn output back
          - CPU and FPGA work on different layers simultaneously (pipeline overlap).
          - Per-layer bottleneck = max(FPGA_proj+FFN, PCIe_roundtrip+CPU_attn)

        When n_requests > 1: attention is per-request (causal within each),
        reducing CPU attention macs from O(P²) to O(P²/B).

        Returns dict with TTFT, total_prefill_ms, per-layer breakdown.
        """
        if prompt_tokens <= 0:
            return {'ttft_ms': 0, 'total_prefill_ms': 0,
                    'per_layer_us': 0, 'effective_tps': 0, 'num_chunks': 0}

        use_chunked = chunk_size > 0 and prompt_tokens > chunk_size
        mac = MACBreakdown()
        reqs = max(1, n_requests)

        def _per_layer_timing(n_tokens: int, accum_kv_total: int) -> dict:
            """Per-layer timing for n_tokens new tokens, with total accumulated KV.
            CPU and FPGA work in parallel across layers (pipeline overlap)."""
            # Per-request KV for batched attention
            avg_kv = (accum_kv_total + n_tokens / 2) / reqs
            density = 1.0 - attn_sparsity

            # ── FPGA: Q/K/V/O projections (fp8 × fp4) ──
            proj_macs = (mac.q_down_macs + mac.kv_latent_macs + mac.kv_rope_macs +
                         mac.o_decompress_macs + mac.o_up_macs) * n_tokens
            fpga_proj_us = proj_macs / (DSP_FFN_TMACS * 1e12) * 1e6

            # ── CPU: Q·K^T + A·V (AMX matmul) ──
            qk_macs = mac.qk_dot_macs_per_kv * n_tokens * avg_kv * density
            av_macs = mac.av_dot_macs * n_tokens * density
            attn_macs = qk_macs + av_macs
            cpu_attn_us = attn_macs / (cpu_tflops * 1e12) * 1e6

            # ── PCIe: Q + K → CPU, attn output → FPGA ──
            q_size_mb = n_tokens * NUM_ATTN_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) / 1e6
            k_size_mb = q_size_mb
            out_size_mb = n_tokens * NUM_ATTN_HEADS * V_HEAD_DIM / 1e6
            pcie_mb = q_size_mb + k_size_mb + out_size_mb
            pcie_us = pcie_mb / PCIE_P2P_BW_GBPS * 1000.0 + CPU_PCIE_LATENCY_US

            # ── FPGA: FFN (shared + routed experts) ──
            shared_macs = mac.shared_expert_macs * n_tokens
            routed_macs = mac.expert_total_macs * n_tokens * TOP_K_EXPERTS
            ffn_macs = shared_macs + routed_macs
            fpga_ffn_us = ffn_macs / (DSP_FFN_TMACS * 1e12) * 1e6

            # Pipeline overlap: CPU-attn and FPGA-FFN run on different layers
            # FPGA path: proj + FFN (per layer, with CPU gap hidden by pipeline)
            # CPU path: PCIe + attn (per layer, sequential)
            fpga_path = fpga_proj_us + fpga_ffn_us     # FPGA critical path
            cpu_path = pcie_us + cpu_attn_us             # CPU critical path
            bottleneck_us = max(fpga_path, cpu_path)

            # First-layer latency: everything sequential (no overlap yet)
            first_layer_us = fpga_proj_us + pcie_us + cpu_attn_us + fpga_ffn_us

            return {
                'fpga_proj_us': fpga_proj_us,
                'pcie_us': pcie_us,
                'cpu_attn_us': cpu_attn_us,
                'fpga_ffn_us': fpga_ffn_us,
                'bottleneck_us': bottleneck_us,      # steady-state per-layer
                'first_layer_us': first_layer_us,    # pipeline startup
                'attn_macs': attn_macs,
                'ffn_macs': ffn_macs + proj_macs,
            }

        if use_chunked:
            num_chunks = max(1, math.ceil(prompt_tokens / chunk_size))
            chunk_timings = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, prompt_tokens)
                n_new = end - start
                accum_kv = start
                chunk_timings.append(_per_layer_timing(n_new, accum_kv))

            layers_per_chip = max(1, round(NUM_LAYERS / TOTAL_CHIPS))

            # TTFT (first chunk): pipeline startup across all chips
            # First chip: layers_per_chip × first_layer_us (startup)
            # Remaining chips: layers_per_chip × bottleneck_us each (pipelined)
            t0 = chunk_timings[0]
            ttft_ms = (t0['first_layer_us'] * layers_per_chip +
                       t0['bottleneck_us'] * layers_per_chip * (TOTAL_CHIPS - 1)) / 1000.0

            # Total prefill: first chunk TTFT + subsequent chunks pipelined
            total_ms = ttft_ms
            for i in range(1, num_chunks):
                ti = chunk_timings[i]
                # Pipeline contribution: bottleneck chip time
                total_ms += ti['bottleneck_us'] * layers_per_chip / 1000.0

            eff_tps = prompt_tokens / total_ms * 1000 if total_ms > 0 else 0
            per_layer_ref = t0['bottleneck_us']
        else:
            t = _per_layer_timing(prompt_tokens, 0)
            layers_per_chip = max(1, round(NUM_LAYERS / TOTAL_CHIPS))
            ttft_ms = (t['first_layer_us'] * layers_per_chip +
                       t['bottleneck_us'] * layers_per_chip * (TOTAL_CHIPS - 1)) / 1000.0
            total_ms = ttft_ms
            eff_tps = prompt_tokens / total_ms * 1000 if total_ms > 0 else 0
            per_layer_ref = t['bottleneck_us']

        return {
            'ttft_ms': ttft_ms,
            'total_prefill_ms': total_ms,
            'per_layer_us': per_layer_ref,
            'effective_tps': eff_tps,
            'num_chunks': math.ceil(prompt_tokens / chunk_size) if use_chunked else 1,
            'first_chunk_detail': _per_layer_timing(
                min(chunk_size if use_chunked else prompt_tokens, prompt_tokens), 0),
        }

    # ── Detailed pipeline simulation ──

    # ── Concurrent prefill + decode ──

    @staticmethod
    def concurrent_pipeline_model(prefill_tokens: int, decode_batch: int,
                                   use_fp4_attn: bool = True,
                                   attn_sparsity: float = 0.888,
                                   n_requests: int = 0) -> Dict:
        """Throughput model when prefill and decode run concurrently.

        Key physics: prefill is DSP-bound (MLA_ATTENTION Q·K^T dominates),
        decode is HBM-bound (ROUTED_EXPERT weight streaming dominates).
        DSP and HBM are independent hardware units → computations can overlap.

        Resource sharing analysis (per-chip budgets):
          - DSP budget: 11.07 TMACS (fp8×fp4). Prefill uses ~70%, decode ~30%.
            Different MAC operations → DSP array time-multiplexed, ~5% overhead.
          - HBM budget: 920 GB/s. Prefill weight reads ~0.1 GB/s (negligible).
            Decode weight streaming up to ~880 GB/s for large B. Minimal overlap loss.
          - C2C budget: 128 GB/s per link × 4 links. Both use ~15-20% each for
            dispatch/reduce/pipeline_fwd. Combined well under budget.
          - SRAM: deterministic weights double-buffered. Mild bank conflict (~3%).

        Overall contention factor: ~1.10 (10% slowdown when both run).
        Derived from complementary resource profiles with minor shared-resource overhead.
        """
        if prefill_tokens <= 0 or decode_batch <= 0:
            return {'prefill_tps': 0, 'decode_tps': 0, 'combined_tps': 0,
                    'contention_factor': 1.0}

        # Single-workload TPS (sequential baseline)
        prefill_tps_only = PipelineEngine.prefill_tps_model(
            prefill_tokens, use_fp4_attn, attn_sparsity, n_requests=n_requests)
        decode_tps_only = PipelineEngine.throughput_model(decode_batch)

        # Per-layer bottleneck times (for pipeline occupancy analysis)
        prefill_per_layer = PipelineEngine.prefill_latency_model(
            prefill_tokens, use_fp4_attn, attn_sparsity, n_requests=n_requests) / NUM_LAYERS
        decode_per_layer = PipelineEngine.decode_latency_model(
            decode_batch) / NUM_LAYERS

        # ── Resource contention ──
        dsp_contention = 1.05

        # HBM: compute actual bandwidth demand
        mac = MACBreakdown()
        _, _, _, prefill_hbm_mb = mac.compute_prefill(
            prefill_tokens, use_fp4_attn=use_fp4_attn, attn_sparsity=attn_sparsity,
            n_requests=n_requests)
        _, decode_hbm_mb, _, _ = mac.compute_decode(
            decode_batch, kv_len=SLIDING_WINDOW, n_local_experts=1)

        # Prefill HBM: weight reads once per batch, spread over batch duration
        prefill_batch_s = prefill_per_layer * NUM_LAYERS / 1e6
        prefill_hbm_gbps = (prefill_hbm_mb / 1024) / prefill_batch_s if prefill_batch_s > 0 else 0

        # Decode HBM: weight reads once per decode step
        decode_step_s = decode_per_layer * NUM_LAYERS / 1e6
        decode_hbm_gbps = (decode_hbm_mb / 1024) / decode_step_s if decode_step_s > 0 else 0

        # HBM budget: 920 GB/s per chip. Total demand well under for typical batch sizes.
        hbm_total = prefill_hbm_gbps + decode_hbm_gbps
        hbm_contention = max(1.0, hbm_total / (HBM_BW_GBPS * 0.95))  # 95% usable

        # C2C: both use inter-chip links for dispatch/reduce/fwd, ~15% each
        c2c_contention = max(1.0, 0.15 + 0.12)  # 27% → well under budget

        # Overall: bottleneck resource determines contention
        contention = max(dsp_contention, hbm_contention, c2c_contention)

        # Effective TPS with contention
        prefill_tps_eff = prefill_tps_only / contention
        decode_tps_eff = decode_tps_only / contention
        combined = prefill_tps_eff + decode_tps_eff

        return {
            'prefill_tps': prefill_tps_eff,
            'decode_tps': decode_tps_eff,
            'combined_tps': combined,
            'contention_factor': contention,
            'dsp_contention': dsp_contention,
            'hbm_contention': hbm_contention,
            'c2c_contention': c2c_contention,
            'prefill_hbm_gbps': prefill_hbm_gbps,
            'decode_hbm_gbps': decode_hbm_gbps,
            'hbm_total_gbps': hbm_total,
            'prefill_per_layer_us': prefill_per_layer,
            'decode_per_layer_us': decode_per_layer,
            'prefill_tps_only': prefill_tps_only,
            'decode_tps_only': decode_tps_only,
        }

    # ── Detailed pipeline simulation ──

    def simulate_decode(self, batch_size: int, num_tokens: int = 50) -> PipelineSimResult:
        """Simulate decode pipeline with given batch size."""
        return simulate_pipeline(
            self.cluster, num_tokens=num_tokens, batch_size=batch_size, is_prefill=False
        )

    def simulate_prefill(self, prompt_len: int, num_tokens: int = 20) -> PipelineSimResult:
        """Simulate prefill pipeline for given prompt length."""
        return simulate_pipeline(
            self.cluster, num_tokens=num_tokens, batch_size=1,
            is_prefill=True, prompt_len=prompt_len
        )

    def calibrate(self) -> Dict:
        """Run detailed simulation at multiple batch sizes and compare with analytical model.

        The detailed simulation models RAW hardware pipeline throughput (~12-14K tok/s),
        while the analytical model (K_PIPELINE=25.4) includes system overheads
        (scheduling, KV cache, queuing) that reduce effective throughput to 660-9,725 tok/s.

        Returns:
            dict with hardware_tps, analytical_tps, efficiency (system/hardware ratio),
            and per-batch-size simulation results.
        """
        batch_sizes = [1, 2, 4, 8, 16, 32]
        results = {}
        for bs in batch_sizes:
            sim = self.simulate_decode(bs, num_tokens=min(50, bs * 10))
            results[bs] = sim

        tps_hw = {bs: r.throughput_tps for bs, r in results.items()}
        tps_analytical = {bs: PipelineEngine.throughput_model(bs) for bs in batch_sizes}
        efficiency = {bs: tps_analytical[bs] / tps_hw[bs] * 100 if tps_hw[bs] > 0 else 0
                      for bs in batch_sizes}

        return {
            'hardware_tps': tps_hw,
            'analytical_tps': tps_analytical,
            'efficiency_pct': efficiency,
            'bottleneck_curve': {bs: r.bottleneck_per_layer_us for bs, r in results.items()},
            'results': results,
        }

    def execute_batch(self, batch_size: int, is_prefill: bool = False,
                      prompt_len: int = 0) -> BatchResult:
        """Execute a batch through the pipeline (fast analytical path for scheduler)."""
        if is_prefill:
            # Prefill: process all prompt tokens in parallel
            sim = self.simulate_prefill(prompt_len, num_tokens=10)
            per_token = sim.avg_token_latency_us / prompt_len if prompt_len > 0 else sim.avg_token_latency_us
        else:
            sim = self.simulate_decode(batch_size, num_tokens=20)
            per_token = sim.avg_token_latency_us

        tps = self.throughput_model(batch_size)

        return BatchResult(
            batch_size=batch_size,
            is_prefill=is_prefill,
            total_latency_us=sim.avg_token_latency_us,
            per_token_latency_us=per_token,
            throughput_tps=tps,
            dsp_utilization_pct=sim.dsp_util_pct,
            hbm_bandwidth_gbps=sim.hbm_read_gbps,
            c2c_messages=int(sim.c2c_messages_per_token * batch_size),
            expert_hits=sim.expert_hit_dist,
        )


# ============================================================================
# Backward-compatible API (for existing callers)
# ============================================================================

class PipelineStage:
    """Base class for pipeline stages (backward-compatible)."""
    def __init__(self, name: str, stage_type: StageType):
        self.name = name
        self.stage_type = stage_type

    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1,
                kv_cache=None, expert_selection=None):
        raise NotImplementedError


class WeightPrefetchStage(PipelineStage):
    def __init__(self):
        super().__init__("Weight Prefetch", StageType.WEIGHT_PREFETCH)
    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1, kv_cache=None, expert_selection=None):
        timing = detailed_stage_timing(StageType.WEIGHT_PREFETCH, chip, cluster, layer_idx, batch_size, False, expert_selection=expert_selection)
        return hidden_state, timing, {}

class MLAAttentionStage(PipelineStage):
    def __init__(self):
        super().__init__("MLA Attention", StageType.MLA_ATTENTION)
    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1, kv_cache=None, expert_selection=None):
        timing = detailed_stage_timing(StageType.MLA_ATTENTION, chip, cluster, layer_idx, batch_size, False)
        return hidden_state, timing, {}

class AttnNormStage(PipelineStage):
    def __init__(self):
        super().__init__("Attn RMSNorm", StageType.ATTN_NORM)
    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1, kv_cache=None, expert_selection=None):
        timing = detailed_stage_timing(StageType.ATTN_NORM, chip, cluster, layer_idx, batch_size, False)
        return hidden_state, timing, {}

class MoERouterStage(PipelineStage):
    def __init__(self, rng):
        super().__init__("MoE Router", StageType.MOE_ROUTER)
        self.rng = rng
    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1, kv_cache=None, expert_selection=None):
        timing = detailed_stage_timing(StageType.MOE_ROUTER, chip, cluster, layer_idx, batch_size, False)
        return hidden_state, timing, {'selected_experts': expert_selection or []}

class MoEDispatchStage(PipelineStage):
    def __init__(self):
        super().__init__("MoE Dispatch", StageType.MOE_DISPATCH)
    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1, kv_cache=None, expert_selection=None):
        timing = detailed_stage_timing(StageType.MOE_DISPATCH, chip, cluster, layer_idx, batch_size, False, expert_selection=expert_selection)
        return hidden_state, timing, {}

class SharedExpertStage(PipelineStage):
    def __init__(self):
        super().__init__("Shared Expert", StageType.SHARED_EXPERT)
    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1, kv_cache=None, expert_selection=None):
        timing = detailed_stage_timing(StageType.SHARED_EXPERT, chip, cluster, layer_idx, batch_size, False)
        return hidden_state, timing, {}

class RoutedExpertStage(PipelineStage):
    def __init__(self):
        super().__init__("Routed Expert", StageType.ROUTED_EXPERT)
    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1, kv_cache=None, expert_selection=None):
        timing = detailed_stage_timing(StageType.ROUTED_EXPERT, chip, cluster, layer_idx, batch_size, False, expert_selection=expert_selection)
        return hidden_state, timing, {}

class MoEReduceStage(PipelineStage):
    def __init__(self):
        super().__init__("MoE Reduce", StageType.MOE_REDUCE)
    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1, kv_cache=None, expert_selection=None):
        timing = detailed_stage_timing(StageType.MOE_REDUCE, chip, cluster, layer_idx, batch_size, False, expert_selection=expert_selection)
        return hidden_state, timing, {}

class FFNNormStage(PipelineStage):
    def __init__(self):
        super().__init__("FFN RMSNorm", StageType.FFN_NORM)
    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1, kv_cache=None, expert_selection=None):
        timing = detailed_stage_timing(StageType.FFN_NORM, chip, cluster, layer_idx, batch_size, False)
        return hidden_state, timing, {}

class PipelineForwardStage(PipelineStage):
    def __init__(self):
        super().__init__("Pipeline Forward", StageType.PIPELINE_FWD)
    def execute(self, cluster, chip, hidden_state, layer_idx, batch_size=1, kv_cache=None, expert_selection=None):
        timing = detailed_stage_timing(StageType.PIPELINE_FWD, chip, cluster, layer_idx, batch_size, False)
        return hidden_state, timing, {}
