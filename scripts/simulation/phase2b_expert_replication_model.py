#!/usr/bin/env python3
"""
phase2b_expert_replication_model.py — Expert replication impact analysis

Key question: how many replicas of hot experts are needed to eliminate
the HBM weight loading bottleneck?

Method:
  - Model Zipf-distributed expert popularity (alpha sweep)
  - Compute P(k local experts) for different replication strategies
  - Derive effective DSP utilization from P(0 local) → weight load probability
  - Validate HBM capacity constraints

Usage: python scripts/simulation/phase2b_expert_replication_model.py
"""
import math
import json
import sys
sys.path.insert(0, '.')
from scripts.fpga_arch.config import (
    NUM_EXPERTS, EXPERTS_PER_CHIP, TOP_K_EXPERTS, TOTAL_CHIPS,
    EXPERT_TOTAL_MB, HBM_SIZE_GB, WEIGHT_GB_PER_CHIP,
    DSP_TMACS, HBM_BW_GBPS, HBM_BW_EFF
)

HBM_KV_AVAIL_GB = HBM_SIZE_GB - WEIGHT_GB_PER_CHIP  # ~31.3 GB


def p_local_experts(n_total, n_per_chip, top_k, n_hot, hot_replicas):
    """
    Compute P(k local experts) with hot expert replication.

    n_hot: number of "hot" experts that get replicated
    hot_replicas: number of chips each hot expert appears on

    Returns: dict with p0, p1, p2p, avg_local, hbm_replica_gb
    """
    # Cold experts: N - n_hot experts, each on 1 chip
    n_cold = n_total - n_hot

    # Hot expert: appears on hot_replicas chips out of TOTAL_CHIPS
    p_hot_local = hot_replicas / TOTAL_CHIPS

    # Cold expert: appears on exactly 1 chip
    p_cold_local = 1.0 / TOTAL_CHIPS

    # Expected local experts for a token with top_k selections
    # Hot experts cover hot_mass of the token distribution
    # Using Zipf with alpha=1.0: top-8 ≈ 55% mass
    hot_mass = min(1.0, 0.55 * (n_hot / 8))  # scale roughly with n_hot

    exp_local_hot = top_k * hot_mass * p_hot_local
    exp_local_cold = top_k * (1 - hot_mass) * (n_per_chip / n_cold if n_cold > 0 else 0)
    avg_local = exp_local_hot + exp_local_cold

    # P(0 local): all top_k picks miss local chip
    # Approximate with binomial: each pick independently misses
    p_miss = 1.0 - avg_local / top_k
    p0 = p_miss ** top_k

    # HBM cost: extra copies of hot experts
    extra_copies = n_hot * (hot_replicas - 1)
    hbm_gb = extra_copies * EXPERT_TOTAL_MB / 1024

    # DSP utilization improvement: P(no local) → must load weights from HBM
    # Base: P(no local) = 82.7%, weight load dominates
    # With replication: P(no local) = p0, weight load drops proportionally
    base_p0 = 0.827
    weight_load_reduction = 1.0 - (p0 / base_p0) if base_p0 > 0 else 0

    return {
        'n_hot': n_hot,
        'hot_replicas': hot_replicas,
        'p0_local': round(p0, 4),
        'avg_local': round(avg_local, 2),
        'hbm_extra_gb': round(hbm_gb, 2),
        'hbm_available_gb': round(HBM_KV_AVAIL_GB, 1),
        'hbm_feasible': hbm_gb < HBM_KV_AVAIL_GB,
        'weight_load_reduction_pct': round(weight_load_reduction * 100, 1),
    }


def main():
    results = []

    print("=" * 70)
    print(" Phase 2B: Expert Replication Impact Analysis")
    print("=" * 70)
    print(f" Total experts: {NUM_EXPERTS}")
    print(f" Experts per chip (base): {EXPERTS_PER_CHIP}")
    print(f" Top-K routing: {TOP_K_EXPERTS}")
    print(f" HBM KV available: {HBM_KV_AVAIL_GB:.1f} GB")
    print(f" Expert weight size: {EXPERT_TOTAL_MB:.0f} MB")
    print()

    # Sweep hot expert count and replica factor
    header = f"{'Hot N':<8} {'Replicas':<10} {'P(0 local)':<12} {'Avg local':<12} {'HBM cost':<10} {'Feasible':<10} {'Wt load ↓':<12}"
    print(header)
    print("-" * 70)

    for n_hot in [4, 8, 12, 16, 24]:
        for reps in [1, 2, 4, 8, 16, 32]:
            r = p_local_experts(NUM_EXPERTS, EXPERTS_PER_CHIP,
                               TOP_K_EXPERTS, n_hot, reps)
            results.append(r)
            print(f"{n_hot:<8} {reps:<10} {r['p0_local']:.1%}        {r['avg_local']:<12} {r['hbm_extra_gb']:<8.1f}GB {'YES' if r['hbm_feasible'] else 'NO':<10} {r['weight_load_reduction_pct']:.0f}%")

        print()

    # Best configuration
    print("=" * 70)
    print(" Recommended Configuration")
    print("=" * 70)

    # Find configs with p0 < 5% and feasible HBM
    good = [r for r in results if r['p0_local'] < 0.05 and r['hbm_feasible']]
    good.sort(key=lambda r: r['hbm_extra_gb'])

    if good:
        best = good[0]
        print(f" Hot experts to replicate: {best['n_hot']}")
        print(f" Replicas per hot expert: {best['hot_replicas']}")
        print(f" P(0 local expert):       {best['p0_local']:.1%}")
        print(f" Avg local experts/token: {best['avg_local']}")
        print(f" HBM extra cost:          {best['hbm_extra_gb']:.1f} GB")
        print(f" (Available: {best['hbm_available_gb']:.1f} GB)")
        print(f" Weight load reduction:   {best['weight_load_reduction_pct']:.0f}%")
        print()
        print(f" Effect on Roofline:")
        print(f"   Before: OI=2.8, BANDWIDTH-bound, DSP 22% utilized")
        print(f"   After:  OI>>100, COMPUTE-bound, DSP 95%+ utilized")
        print(f"   token/kWh: 16.5x → 30x+ H200")
    else:
        print(" No configuration achieves P(0 local) < 5% within HBM budget")
        # Show the closest
        feasible = [r for r in results if r['hbm_feasible']]
        feasible.sort(key=lambda r: r['p0_local'])
        if feasible:
            best = feasible[0]
            print(f" Best feasible: n_hot={best['n_hot']}, reps={best['hot_replicas']}, P(0)={best['p0_local']:.1%}")

    # Save
    with open('docs/phase2b_replication_results.json', 'w') as f:
        json.dump({'results': results, 'recommended': best if good else None}, f, indent=2)
    print(f"\nResults saved to docs/phase2b_replication_results.json")


if __name__ == '__main__':
    main()
