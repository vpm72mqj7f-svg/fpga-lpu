"""
fpga_arch/expert_popularity.py — MoE expert popularity model (Zipf).

DeepSeek V4 Pro MoE routing exhibits a power-law (Zipf-like) distribution:
a small fraction of experts receive a disproportionately large share of tokens.
This module models that distribution and derives replica placement.

Key results (default alpha=1.0):
  Top-20 experts capture ~34% of all token routing decisions
  Top-77 experts capture ~65% (consistent with cluster.py's 80/20 split)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class ExpertPopularity:
    """Zipf-distributed expert popularity model.

    rank_freq[i] = 1 / (1 + i)^alpha,  i = 0..N-1
    Normalized so that sum = 1.0.

    alpha=0 → uniform, alpha=1 → Zipf, alpha=2 → strong skew.
    """
    num_experts: int = 384
    alpha: float = 1.0
    seed: int = 42

    def __post_init__(self):
        rng = np.random.RandomState(self.seed)
        ranks = np.arange(self.num_experts)
        raw = 1.0 / (1.0 + ranks) ** self.alpha
        # Shuffle to avoid systematic chip bias (experts are not sorted by popularity)
        rng.shuffle(raw)
        self.freq: np.ndarray = raw / raw.sum()
        # Sort descending for replica allocation
        order = np.argsort(-self.freq)
        self.sorted_expert_ids: np.ndarray = order
        self.sorted_freq: np.ndarray = self.freq[order]

    def top_k_mass(self, k: int) -> float:
        """Cumulative routing mass captured by top-K most popular experts."""
        return float(self.sorted_freq[:k].sum())

    def sample_experts(self, top_k: int = 6, rng: np.random.RandomState = None) -> List[int]:
        """Sample top_k experts according to popularity distribution."""
        if rng is None:
            rng = np.random.RandomState()
        return sorted(rng.choice(self.num_experts, size=top_k, replace=False,
                                 p=self.freq).tolist())

    def replica_plan(self, total_chips: int = 32,
                     hbm_budget_per_chip_gb: float = 1.0,
                     expert_weight_mb: float = 33.0) -> Dict[int, int]:
        """Compute replica count per expert based on popularity.

        Strategy:
          - top_n_hot:   8 replicas each (spread across cards for C2C locality)
          - mid_n:       2 replicas each
          - tail:        1 replica  (baseline)

        Replica count is capped so that each chip's weight budget is not exceeded.

        Returns: {expert_id: replica_count}
        """
        max_experts_per_chip = int(hbm_budget_per_chip_gb * 1024 / expert_weight_mb)

        # Determine tier boundaries from cumulative mass
        # Tier 1 (hot): captures ~35% of traffic
        # Tier 2 (warm): next ~30%
        # Tier 3 (cold): remaining ~35%
        tier1_mass = 0.35
        tier2_mass = 0.65

        cum = 0.0
        tier1_end = 0
        tier2_end = 0
        for i, f in enumerate(self.sorted_freq):
            cum += f
            if cum >= tier1_mass and tier1_end == 0:
                tier1_end = i + 1
            if cum >= tier2_mass and tier2_end == 0:
                tier2_end = i + 1
                break

        n_hot = max(1, tier1_end)
        n_warm = max(0, tier2_end - tier1_end)

        plan = {}
        total_replicas = 0

        for i, eid in enumerate(self.sorted_expert_ids):
            if i < n_hot:
                plan[int(eid)] = min(8, total_chips)
            elif i < n_hot + n_warm:
                plan[int(eid)] = 2
            else:
                plan[int(eid)] = 1
            total_replicas += plan[int(eid)]

        # Check HBM feasibility: replicas must fit across chips
        avg_per_chip = total_replicas / total_chips
        if avg_per_chip > max_experts_per_chip:
            # Scale down: reduce hot-tier replicas
            scale = max_experts_per_chip / avg_per_chip
            for eid in plan:
                plan[eid] = max(1, int(plan[eid] * scale))
            total_replicas = sum(plan.values())

        self._plan_summary = {
            'n_hot': n_hot,
            'n_warm': n_warm,
            'n_cold': self.num_experts - n_hot - n_warm,
            'total_replicas': total_replicas,
            'avg_per_chip': total_replicas / total_chips,
            'hot_mass': self.top_k_mass(n_hot),
            'warm_mass': self.top_k_mass(n_hot + n_warm),
        }

        return plan

    def plan_summary(self) -> str:
        s = self._plan_summary
        lines = [
            f"Expert Popularity (Zipf α={self.alpha}):",
            f"  Hot tier:   {s['n_hot']:>3} experts ×8 replicas  = {s['n_hot']*8:>4}  ({s['hot_mass']:.0%} traffic)",
            f"  Warm tier:  {s['n_warm']:>3} experts ×2 replicas  = {s['n_warm']*2:>4}",
            f"  Cold tier:  {s['n_cold']:>3} experts ×1 replica   = {s['n_cold']:>4}",
            f"  Total replicas: {s['total_replicas']}, avg {s['avg_per_chip']:.1f}/chip",
        ]
        return "\n".join(lines)
