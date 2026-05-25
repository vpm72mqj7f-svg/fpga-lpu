"""
fpga_arch/cluster.py — 32-chip FPGA inference cluster assembly.

Extracted from fpga_4chip_pipeline.py:383-484 with enhancements:
  - cluster_report() for resource utilization summary
  - Expert hit probability computation
  - Per-chip weight/storage tracking
  - 解法 A: Hot Expert Replication — popular experts replicated across
    multiple chips for load balancing and C2C locality
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np

from .config import (
    NUM_CARDS, CHIPS_PER_CARD, TOTAL_CHIPS,
    NUM_LAYERS, NUM_EXPERTS, EXPERTS_PER_CHIP,
    TP_ATTN_PER_LAYER, TOP_K_EXPERTS,
    ATTN_WEIGHT_MB, EXPERT_WEIGHT_MB, EXPERT_TOTAL_MB,
    ROUTER_WEIGHT_MB, NORM_WEIGHT_MB,
    DETERMINISTIC_MB_PER_LAYER, WEIGHT_GB_PER_CHIP,
    HBM_SIZE_GB, SRAM_TOTAL_MB, SRAM_USED_MB, SRAM_FREE_MB,
    P_EXPERT_PER_CHIP, P_0_HIT, P_1_HIT, P_2P_HIT,
)
from .chip import FPGAChip, SRAMBank, HBMBank, DSPArray
from .interconnect import C2CDualRing, PCIeFabric


@dataclass
class ClusterStats:
    """Aggregate cluster resource and performance statistics."""
    total_weight_gb: float = 0.0
    total_kv_cache_gb: float = 0.0
    total_sram_used_mb: float = 0.0
    dsp_available_tflops: float = 0.0
    hbm_available_tbps: float = 0.0
    chip_load_distribution: Dict[int, float] = field(default_factory=dict)
    layer_to_chip: Dict[int, int] = field(default_factory=dict)
    expert_to_chip: Dict[int, int] = field(default_factory=dict)


class FPGACluster:
    """Complete 8-card × 4-chip FPGA inference cluster.

    Handles:
      - Chip creation and layer assignment (61 layers across 32 chips)
      - Expert distribution (384 experts, 12 per chip baseline)
      - 解法 A: Hot Expert Replication for popular experts
      - Weight placement (HBM + SRAM)
      - C2C/PCIe fabric initialization

    expert_replication: 'none' (baseline 12/chip) or 'hot' (Zipf-based replicas).
    """

    def __init__(self, seed: int = 42, expert_replication: str = 'none',
                 zipf_alpha: float = 1.0):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.expert_replication = expert_replication
        self.zipf_alpha = zipf_alpha

        # Statistics (must exist before _assign_layers references it)
        self.stats = ClusterStats()
        self.total_tokens_processed: int = 0

        # Create chips
        self.chips: List[FPGAChip] = []
        for card_id in range(NUM_CARDS):
            for chip_id in range(CHIPS_PER_CARD):
                self.chips.append(FPGAChip(chip_id=chip_id, card_id=card_id))

        # Create cards with C2C rings
        self.cards: List[C2CDualRing] = []
        for card_id in range(NUM_CARDS):
            card_chips = [c for c in self.chips if c.card_id == card_id]
            ring = C2CDualRing(card_id, card_chips)
            self.cards.append(ring)

        # Cross-card fabric
        self.pcie_fabric = PCIeFabric()

        # Assign layers
        self._assign_layers()

        # Expert → chip mapping (supports multi-replica)
        # expert_to_chips[eid] = list of chips hosting this expert
        # expert_to_chip[eid]  = primary chip (backward compat)
        self.expert_to_chips: Dict[int, List[FPGAChip]] = {}
        self.expert_to_chip: Dict[int, FPGAChip] = {}

        if expert_replication == 'hot':
            self._assign_experts_replicated()
        else:
            self._assign_experts_uniform()

        # Weight placement
        self._place_weights()

    def _assign_layers(self):
        """Assign 61 layers across 32 chips: 29×2 + 3×1 = 61."""
        one_layer_positions = {7, 15, 23}  # chips with 1 layer (at card boundaries)
        layer_per_chip = [1 if i in one_layer_positions else 2 for i in range(32)]
        # Total: 29*2 + 3*1 = 58 + 3 = 61

        layer_idx = 0
        for chip in self.chips:
            n = layer_per_chip[chip.global_id]
            chip.assign_layers(list(range(layer_idx, layer_idx + n)))
            layer_idx += n

        self.chip_for_embedding = self.chips[0]   # C0.0
        self.chip_for_lm_head = self.chips[-1]    # C7.3

        # Build layer→chip index
        for chip in self.chips:
            for lid in chip.assigned_layers:
                self.stats.layer_to_chip[lid] = chip.global_id

    def _assign_experts_uniform(self):
        """Baseline: 384 experts / 32 chips = 12 per chip, no replicas."""
        for chip in self.chips:
            for expert_id in chip.assigned_experts:
                self.expert_to_chip[expert_id] = chip
                self.expert_to_chips[expert_id] = [chip]

    def _assign_experts_replicated(self):
        """解法 A: Hot Expert Replication — Zipf-based replica placement.

        Popular experts get multiple replicas spread across cards.
        Placement strategy: for each expert, distribute replicas evenly
        across cards (max 1 replica per card when possible) to minimize
        average C2C distance.
        """
        from .expert_popularity import ExpertPopularity

        pop = ExpertPopularity(num_experts=NUM_EXPERTS, alpha=self.zipf_alpha,
                               seed=self.seed)
        plan = pop.replica_plan(total_chips=TOTAL_CHIPS,
                                hbm_budget_per_chip_gb=2.0,
                                expert_weight_mb=EXPERT_TOTAL_MB)
        self._pop = pop

        # Clear default assignment
        for chip in self.chips:
            chip.assigned_experts = []

        # Place replicas: spread across cards first, then across chips within card
        card_chips = defaultdict(list)
        for chip in self.chips:
            card_chips[chip.card_id].append(chip)

        for expert_id in range(NUM_EXPERTS):
            n_replicas = plan[expert_id]
            # Pick n_replicas cards, then chip 0 from each card
            # For >8 replicas (shouldn't happen with 32 chips), fill remaining
            target_cards = list(range(NUM_CARDS))
            self.rng.shuffle(target_cards)
            selected_cards = target_cards[:min(n_replicas, NUM_CARDS)]

            replicas = []
            for card_id in selected_cards:
                # Pick chip with fewest assigned experts so far (load balance)
                candidates = card_chips[card_id]
                candidates.sort(key=lambda c: len(c.assigned_experts))
                chosen = candidates[0]
                replicas.append(chosen)
                chosen.assigned_experts.append(expert_id)

            # If need more replicas than cards, distribute extras
            if n_replicas > NUM_CARDS:
                remaining = n_replicas - NUM_CARDS
                for _ in range(remaining):
                    all_chips_sorted = sorted(self.chips,
                                              key=lambda c: len(c.assigned_experts))
                    chosen = all_chips_sorted[0]
                    if expert_id not in chosen.assigned_experts:
                        replicas.append(chosen)
                        chosen.assigned_experts.append(expert_id)

            self.expert_to_chips[expert_id] = replicas
            self.expert_to_chip[expert_id] = replicas[0]  # primary

    def _place_weights(self):
        """Model weight placement across HBM and SRAM."""
        for chip in self.chips:
            expert_weight_mb = len(chip.assigned_experts) * EXPERT_TOTAL_MB
            n_layers = len(chip.assigned_layers)
            attn_per_layer_mb = sum(ATTN_WEIGHT_MB.values()) / TP_ATTN_PER_LAYER
            attn_total_mb = n_layers * attn_per_layer_mb
            router_mb = ROUTER_WEIGHT_MB

            chip.place_weights(expert_weight_mb, attn_total_mb, router_mb)
            chip.sram.deterministic_mb = DETERMINISTIC_MB_PER_LAYER

    def get_chip_for_layer(self, layer_idx: int) -> FPGAChip:
        for chip in self.chips:
            if layer_idx in chip.assigned_layers:
                return chip
        raise ValueError(f"Layer {layer_idx} not assigned to any chip")

    def get_chip_for_expert(self, expert_idx: int) -> FPGAChip:
        """Primary chip for this expert (backward compat)."""
        return self.expert_to_chip[expert_idx]

    def get_chips_for_expert(self, expert_idx: int) -> List[FPGAChip]:
        """All chips hosting a replica of this expert."""
        return self.expert_to_chips.get(expert_idx, [self.expert_to_chip[expert_idx]])

    def closest_replica(self, src_chip: FPGAChip, expert_idx: int) -> FPGAChip:
        """Find the closest replica of expert_idx to src_chip.

        Same-card → C2C ring (1-3 hops). Cross-card → PCIe.
        If expert has only 1 replica, returns that chip regardless.
        """
        replicas = self.get_chips_for_expert(expert_idx)
        if len(replicas) == 1:
            return replicas[0]
        # Same-card first
        same_card = [c for c in replicas if c.card_id == src_chip.card_id]
        if same_card:
            return same_card[0]  # any same-card chip is close enough
        # Cross-card: pick by card distance (topology-agnostic: min card_id diff)
        return min(replicas, key=lambda c: abs(c.card_id - src_chip.card_id))

    def is_same_card(self, chip_a: FPGAChip, chip_b: FPGAChip) -> bool:
        return chip_a.card_id == chip_b.card_id

    def c2c_transfer_time_us(self, src: FPGAChip, dst: FPGAChip, payload_bytes: int) -> float:
        """C2C or PCIe transfer time between any two chips."""
        if src.global_id == dst.global_id:
            return 0.0
        if self.is_same_card(src, dst):
            card = self.cards[src.card_id]
            return card.transfer_time_us(src.chip_id, dst.chip_id, payload_bytes)
        else:
            return self.pcie_fabric.transfer_time_us(src.card_id, dst.card_id, payload_bytes)

    def dispatch_experts(self, chip: FPGAChip, expert_selection: List[int]) -> Dict[int, List[int]]:
        """Group selected experts by hosting chip. Returns {chip_global_id: [expert_ids]}.

        With replication: picks closest replica per expert from src chip's perspective.
        """
        by_chip: Dict[int, List[int]] = defaultdict(list)
        for eid in expert_selection:
            host = self.closest_replica(chip, eid)
            by_chip[host.global_id].append(eid)
        return dict(by_chip)

    def count_local_experts(self, chip: FPGAChip, expert_selection: List[int]) -> int:
        """Count experts in selection that have a replica on the given chip."""
        local = 0
        for eid in expert_selection:
            replicas = self.get_chips_for_expert(eid)
            if any(c.global_id == chip.global_id for c in replicas):
                local += 1
        return local

    def sample_expert_selection(self) -> List[int]:
        """Sample top-6 experts, using popularity model if replication is on."""
        if self.expert_replication == 'hot' and hasattr(self, '_pop'):
            return self._pop.sample_experts(top_k=TOP_K_EXPERTS, rng=self.rng)
        # Fallback: simple 80/20 split (original behavior)
        if self.rng.random() < 0.8:
            candidates = self.rng.choice(77, size=TOP_K_EXPERTS * 2, replace=False)
        else:
            candidates = self.rng.choice(range(77, NUM_EXPERTS), size=TOP_K_EXPERTS * 2, replace=False)
        self.rng.shuffle(candidates)
        return sorted(int(c) for c in candidates[:TOP_K_EXPERTS])

    def cluster_report(self) -> str:
        """Generate a human-readable cluster resource report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  FPGA Inference Cluster Report")
        lines.append("=" * 60)
        lines.append(f"  Chips: {NUM_CARDS} cards × {CHIPS_PER_CARD} chips = {TOTAL_CHIPS}")
        lines.append(f"  Layers: {NUM_LAYERS} across 32 chips (29×2 + 3×1)")
        lines.append(f"  Experts: {NUM_EXPERTS} total, {EXPERTS_PER_CHIP} per chip (baseline)")
        if self.expert_replication == 'hot' and hasattr(self, '_pop'):
            lines.append(f"  Replication: {self._pop.plan_summary().replace(chr(10), chr(10) + '  ')}")
        lines.append(f"  Expert hit probs: P(0)={P_0_HIT:.3f} P(1)={P_1_HIT:.3f} P(2+)={P_2P_HIT:.4f}")
        lines.append("")

        # Aggregate resources
        total_weight_gb = sum(c.hbm.weight_storage_gb for c in self.chips)
        total_kv_gb = sum(c.hbm.kv_cache_gb for c in self.chips)
        total_sram = sum(c.sram.used_mb for c in self.chips)
        lines.append("  --- Aggregate Resources ---")
        lines.append(f"  Weight storage:  {total_weight_gb:.1f} GB ({(total_weight_gb/TOTAL_CHIPS):.2f} GB/chip)")
        lines.append(f"  KV cache:        {total_kv_gb:.1f} GB")
        lines.append(f"  SRAM used:       {total_sram:.0f} MB ({(total_sram/TOTAL_CHIPS):.1f} MB/chip)")
        lines.append(f"  DSP total:       {TOTAL_CHIPS * DSPArray().tmacs:.1f} TMACs")
        lines.append(f"  HBM total:       {TOTAL_CHIPS * HBM_SIZE_GB} GB")
        lines.append("")

        # Per-chip breakdown
        lines.append("  --- Per-Chip Layer Assignment ---")
        for chip in self.chips:
            layers = chip.assigned_layers
            if layers:
                lines.append(
                    f"  C{chip.card_id}.{chip.chip_id} (gid={chip.global_id:02d}): "
                    f"L{min(layers):02d}-{max(layers):02d} ({len(layers)} layers), "
                    f"{len(chip.assigned_experts)} experts"
                )
        lines.append("")

        # Bottleneck analysis
        lines.append("  --- Bottleneck Analysis ---")
        max_layers = max(len(c.assigned_layers) for c in self.chips)
        bottleneck_chips = [c for c in self.chips if len(c.assigned_layers) == max_layers]
        for bc in bottleneck_chips:
            lines.append(f"  Max-layer chip: C{bc.card_id}.{bc.chip_id} with {len(bc.assigned_layers)} layers")

        lines.append("=" * 60)
        return "\n".join(lines)
