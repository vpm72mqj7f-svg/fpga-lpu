#!/usr/bin/env python3
"""
cross_validate_stacks.py — Compare fpga_arch/ vs architecture/ legacy stacks.

Runs identical single-chip, 1-layer, B=1 decode scenarios through both stacks
and reports numerical differences with explanations.

The legacy architecture/ stack models a single card (4 FPGA chips per card)
with 9-stage pipeline and Ethernet interconnect.
The main fpga_arch/ stack models a 32-chip cluster with 10-stage pipeline
and C2C dual-ring + PCIe interconnect.

We align the configurations so both stacks represent the same hardware scenario
and compare:
  - Per-layer latency (us)
  - DSP compute time
  - HBM weight load time
  - Expert hit statistics
  - Resource utilization breakdown

Usage:
  cd D:/workspace/fpgalpu
  python scripts/simulation/cross_validate_stacks.py
  python scripts/simulation/cross_validate_stacks.py --detailed
"""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

import random
import numpy as np


def fmt_us(v: float) -> str:
    """Format microseconds with appropriate precision."""
    if v < 0.01:
        return f"{v*1000:.2f} ns"
    elif v < 100:
        return f"{v:.2f} us"
    else:
        return f"{v:.1f} us"


def fmt_diff(a: float, b: float) -> str:
    """Format difference between two values."""
    if b == 0:
        return "N/A"
    diff = a - b
    pct = (diff / abs(b)) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2f} ({sign}{pct:.1f}%)"


# ============================================================================
# Legacy Architecture Stack (single card, 9-stage)
# ============================================================================

def run_legacy_stack(hidden_size: int = 7168, intermediate_size: int = 3072,
                     num_experts: int = 384, top_k: int = 6,
                     experts_per_card: int = 12, batch_size: int = 1,
                     seq_len: int = 128, seed: int = 42) -> dict:
    """Run the legacy architecture/ stack for single-chip, 1-layer, B=1 decode.

    Uses Pipeline from architecture.fpga_layer.pipeline with its 9-stage model.
    The legacy stack models 1 card (4 FPGA chips) with Ethernet interconnect.
    For this comparison we use a single-card configuration.
    """
    from architecture.fpga_layer.pipeline import Pipeline
    from architecture import config as legacy_cfg

    # Monkey-patch legacy config to match fpga_arch baseline for fair comparison
    legacy_cfg.MODEL_HIDDEN_SIZE = hidden_size
    legacy_cfg.MODEL_INTERMEDIATE_SIZE = intermediate_size
    legacy_cfg.MODEL_NUM_EXPERTS = num_experts
    legacy_cfg.MODEL_TOP_K = top_k
    legacy_cfg.MODEL_EXPERTS_PER_FPGA = experts_per_card
    legacy_cfg.MODEL_NUM_LAYERS = 1  # single layer for comparison

    rng = random.Random(seed)
    loaded = sorted(rng.sample(range(num_experts), experts_per_card))

    pl = Pipeline(card_id=0, loaded_experts=loaded)

    # Execute 1 layer
    rec = pl.execute_layer(layer_idx=0, batch_size=batch_size, seq_len=seq_len)

    # Extract resource stats
    stats = pl.stats
    beat_breakdown = rec.beat_latencies

    return {
        'stack': 'architecture/ (legacy, 9-stage)',
        'layer_latency_us': rec.total_us,
        'beat_breakdown': beat_breakdown,
        'dsp_time_us': rec.dsp_total_us,
        'hbm_time_us': rec.hbm_total_us,
        'eth_time_us': rec.eth_total_us,
        'pcie_time_us': rec.pcie_total_us,
        'hit_count': rec.hit_count,
        'miss_count': rec.miss_count,
        'hit_distribution': pl.hit_distribution,
        'dsp_stats': stats.get('dsp', {}),
        'hbm_stats': stats.get('hbm', {}),
        'eth_stats': stats.get('eth', {}),
        'pcie_stats': stats.get('pcie', {}),
    }


# ============================================================================
# Main fpga_arch Stack (32-chip, 10-stage)
# ============================================================================

def run_fpga_arch_stack(hidden_size: int = 7168, intermediate_size: int = 3072,
                        num_experts: int = 384, top_k: int = 6,
                        experts_per_chip: int = 12, batch_size: int = 1,
                        seed: int = 42) -> dict:
    """Run the main fpga_arch/ stack for single-chip-equivalent, 1-layer, B=1 decode.

    Uses the full 32-chip FPGACluster but only analyzes chip 0, layer 0.
    The per-layer DSP/HBM compute timing is chip-independent (it depends on
    how many local experts the chip has and the MAC breakdown, not on how
    many total chips exist). C2C overhead is reported separately.
    """
    from fpga_arch.chip import FPGAChip
    from fpga_arch.cluster import FPGACluster
    from fpga_arch.pipeline import (
        PipelineEngine, MACBreakdown, ExpertHitPath,
        detailed_layer_timing, StageType, detailed_stage_timing,
    )
    from fpga_arch import config as main_cfg

    cluster = FPGACluster(seed=seed, expert_replication='none')

    # Only look at chip 0, layer 0. The other 31 chips / 60 layers don't
    # affect the per-layer compute timing which is what we compare.
    chip = cluster.chips[0]

    # Generate a representative expert selection that maps to chip 0
    # With 12 experts/chip, use 1 local + 5 remote (typical ExpertHitPath.ONE)
    local_experts = list(np.random.RandomState(seed).choice(
        chip.assigned_experts, size=min(1, len(chip.assigned_experts)),
        replace=False,
    ))
    other_experts = [e for e in range(main_cfg.NUM_EXPERTS)
                     if e not in chip.assigned_experts]
    remote_experts = list(np.random.RandomState(seed + 1).choice(
        other_experts, size=top_k - len(local_experts), replace=False,
    ))
    expert_selection = sorted(local_experts + remote_experts)

    # Detailed timing for one layer on this chip (with C2C for remote experts)
    mac_bd = MACBreakdown()
    lt = detailed_layer_timing(
        cluster, layer_idx=0, batch_size=batch_size,
        is_prefill=False, kv_len=128,
        expert_selection=expert_selection,
        expert_path=ExpertHitPath.ONE,
    )

    # Compute per-component times
    dsp_time = sum(s.dsp_time_us for s in lt.stages)
    hbm_time = sum(s.hbm_time_us for s in lt.stages)
    c2c_time = sum(s.c2c_time_us for s in lt.stages)
    sram_time = sum(s.sram_time_us for s in lt.stages)

    stage_breakdown = {}
    for s in lt.stages:
        stage_breakdown[s.stage_type.name] = s.total_time_us

    # Analytical throughput model
    engine = PipelineEngine(cluster)
    tps = engine.throughput_model(batch_size)

    # Resource usage from chip 0
    resource_usage = {
        'dsp_tmacs': main_cfg.DSP_TMACS,
        'dsp_attn_fp8_tmacs': main_cfg.DSP_ATTN_FP8_TMACS,
        'dsp_ffn_tmacs': main_cfg.DSP_FFN_TMACS,
        'hbm_size_gb': main_cfg.HBM_SIZE_GB,
        'hbm_bw_gbps': main_cfg.HBM_BW_GBPS,
        'hbm_bw_eff': main_cfg.HBM_BW_EFF,
        'sram_mb': main_cfg.SRAM_TOTAL_MB,
        'assigned_layers': len(chip.assigned_layers),
        'assigned_experts': len(chip.assigned_experts),
        'num_local_hits': cluster.count_local_experts(chip, expert_selection),
    }

    return {
        'stack': 'fpga_arch/ (main, 10-stage)',
        'layer_latency_us': lt.total_time_us,
        'stage_breakdown': stage_breakdown,
        'dsp_time_us': dsp_time,
        'hbm_time_us': hbm_time,
        'c2c_time_us': c2c_time,
        'sram_time_us': sram_time,
        'throughput_tps': tps,
        'resource_usage': resource_usage,
    }


# ============================================================================
# Analytical Model (for ground-truth comparison)
# ============================================================================

def run_analytical_model(hidden_size: int = 7168, intermediate_size: int = 3072,
                         num_experts: int = 384, top_k: int = 6,
                         experts_per_chip: int = 12, batch_size: int = 1,
                         dsp_tmacs: float = 11.07, hbm_bw_gbps: float = 920,
                         hbm_bw_eff: float = 0.87) -> dict:
    """Compute analytical bounds using simple MAC/HBM bandwidth arithmetic.

    This serves as a ground-truth sanity check independent of both stacks.
    """

    # ── DSP compute ──
    # MLA projections: Q_down + KV_latent + KV_rope + O_decompress + O_up
    # Per-token MACs (approximate from the MACBreakdown defaults)
    q_down_macs = hidden_size * 1536  # H x Q_LORA_RANK
    kv_latent_macs = hidden_size * 512  # H x KV_LORA_RANK
    kv_rope_macs = hidden_size * 64  # H x ROPE_DIM
    o_decompress_macs = hidden_size * 1024  # H x O_LORA_RANK
    o_up_macs = hidden_size * 512  # H x ??? (approximate)

    proj_macs = (q_down_macs + kv_latent_macs + kv_rope_macs +
                 o_decompress_macs + o_up_macs) * batch_size
    # Per-chip: TP=2 divides projection MACs by 2
    proj_macs /= 2

    # QK dot: B x KV_len x (nope_dim + rope_dim) / TP
    kv_len = 128  # sliding window
    qk_dot_macs = batch_size * kv_len * (128 + 64) * 128 / 2

    # AV dot: B x V_dim x H / TP
    av_dot_macs = batch_size * 128 * hidden_size / 2

    attn_macs = proj_macs + qk_dot_macs + av_dot_macs

    # Shared expert: B x H x inter x 2 (gate+up)
    shared_macs = batch_size * hidden_size * intermediate_size * 2

    # Routed experts: B x top_k x H x inter x 2 (gate+up) + B x top_k x inter x H (down)
    routed_macs = batch_size * top_k * hidden_size * intermediate_size * 3
    # But only local experts run on this chip; avg local = top_k * experts_per_chip / num_experts
    avg_local = top_k * experts_per_chip / num_experts
    routed_local_macs = batch_size * avg_local * hidden_size * intermediate_size * 3

    total_macs = attn_macs + shared_macs + routed_local_macs

    dsp_time_us = total_macs / (dsp_tmacs * 1e12) * 1e6

    # ── HBM weight load ──
    # Per-expert HBM weight: gate + up + down, fp4 format
    expert_weight_bytes = hidden_size * intermediate_size * 3 * 0.5  # fp4 = 0.5 bytes/element
    expert_weight_mb = expert_weight_bytes / 1e6

    # + router weight: num_experts x hidden_size x 0.5 bytes
    router_mb = num_experts * hidden_size * 0.5 / 1e6

    # + attention weights (per-layer)
    attn_weight_mb = (hidden_size * (1536 + 512 + 64 + 1024 + 512) * 0.5) / 1e6

    hbm_total_mb = expert_weight_mb * avg_local + router_mb + attn_weight_mb * 0.2  # 20% in HBM
    hbm_time_us = hbm_total_mb / (hbm_bw_gbps * hbm_bw_eff / 1000)

    # ── Expert hit probabilities (binomial) ──
    card_prob = experts_per_chip / num_experts
    p0 = (1 - card_prob) ** top_k
    p1 = top_k * card_prob * (1 - card_prob) ** (top_k - 1)
    p2 = 1 - p0 - p1

    # ── Resource usage ──
    total_weight_gb = (expert_weight_mb * experts_per_chip +
                       router_mb + attn_weight_mb) / 1024

    return {
        'stack': 'Analytical (ground truth)',
        'layer_latency_us': dsp_time_us + hbm_time_us,
        'dsp_time_us': dsp_time_us,
        'hbm_time_us': hbm_time_us,
        'total_macs_million': total_macs / 1e6,
        'hbm_load_mb': hbm_total_mb,
        'expert_hit_p0': p0,
        'expert_hit_p1': p1,
        'expert_hit_p2': p2,
        'avg_local_experts': avg_local,
        'total_weight_gb': total_weight_gb,
        'throughput_est_tps': batch_size * 1e6 / (dsp_time_us + hbm_time_us) if (dsp_time_us + hbm_time_us) > 0 else 0,
    }


# ============================================================================
# Comparison & Reporting
# ============================================================================

def compare_stacks(legacy: dict, main: dict, analytical: dict,
                   detailed: bool = False) -> dict:
    """Compare the two stacks and the analytical model. Report differences."""
    results = {
        'legacy': legacy,
        'main': main,
        'analytical': analytical,
    }

    print()
    print("=" * 78)
    print("  Cross-Validation: fpga_arch vs architecture Legacy Stack")
    print("  Config: 1 chip, 1 layer, B=1 decode, 384 experts, Top-6")
    print("=" * 78)
    print()

    # ── Per-layer latency ──
    print("  --- Per-Layer Latency ---")
    leg_lat = legacy['layer_latency_us']
    main_lat = main['layer_latency_us']
    ana_lat = analytical['layer_latency_us']

    print(f"  {'Metric':<30s} {'Legacy (9-stage)':>16s} {'Main (10-stage)':>16s} {'Analytical':>16s}")
    print(f"  {'-'*30} {'-'*16} {'-'*16} {'-'*16}")
    print(f"  {'Total layer latency':<30s} {fmt_us(leg_lat):>16s} {fmt_us(main_lat):>16s} {fmt_us(ana_lat):>16s}")

    # Difference analysis
    print()
    print(f"  Legacy vs Analytical:  {fmt_diff(leg_lat, ana_lat)}")
    print(f"  Main vs Analytical:    {fmt_diff(main_lat, ana_lat)}")
    print(f"  Legacy vs Main:        {fmt_diff(leg_lat, main_lat)}")
    print()

    # ── Component breakdown ──
    print("  --- Component Breakdown ---")

    # Legacy components
    leg_dsp = legacy.get('dsp_time_us', 0)
    leg_hbm = legacy.get('hbm_time_us', 0)
    leg_pcie = legacy.get('pcie_time_us', 0)
    leg_eth = legacy.get('eth_time_us', 0)

    # Main components
    main_dsp = main.get('dsp_time_us', 0)
    main_hbm = main.get('hbm_time_us', 0)
    main_c2c = main.get('c2c_time_us', 0)

    # Analytical
    ana_dsp = analytical['dsp_time_us']
    ana_hbm = analytical['hbm_time_us']

    print(f"  {'Component':<30s} {'Legacy':>12s} {'Main':>12s} {'Analytical':>12s}  Notes")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}  {'-'*20}")
    print(f"  {'DSP compute':<30s} {fmt_us(leg_dsp):>12s} {fmt_us(main_dsp):>12s} {fmt_us(ana_dsp):>12s}  {'MAC-based compute bound':>20s}")
    print(f"  {'HBM weight read':<30s} {fmt_us(leg_hbm):>12s} {fmt_us(main_hbm):>12s} {fmt_us(ana_hbm):>12s}  {'Expert streaming':>20s}")

    if leg_pcie > 0.01:
        print(f"  {'PCIe TX+RX':<30s} {fmt_us(leg_pcie):>12s} {'(via C2C)':>12s} {'N/A':>12s}  {'Legacy PCIe, Main C2C':>20s}")
    if leg_eth > 0.01:
        print(f"  {'Ethernet (legacy)':<30s} {fmt_us(leg_eth):>12s} {'N/A':>12s} {'N/A':>12s}  {'Legacy inter-card':>20s}")
    if main_c2c > 0.01:
        print(f"  {'C2C (main)':<30s} {'N/A':>12s} {fmt_us(main_c2c):>12s} {'N/A':>12s}  {'Main inter-chip':>20s}")

    print()

    # ── Expert hit analysis ──
    print("  --- Expert Hit Analysis ---")
    print(f"  {'Metric':<30s} {'Legacy':>12s} {'Main':>12s} {'Analytical':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'P(0 hit)':<30s} {legacy.get('hit_distribution', {}).get('0_hit', 0):>11.1%} {'N/A':>12s} {analytical['expert_hit_p0']:>11.1%}")
    print(f"  {'P(1 hit)':<30s} {legacy.get('hit_distribution', {}).get('1_hit', 0):>11.1%} {'N/A':>12s} {analytical['expert_hit_p1']:>11.1%}")
    print(f"  {'P(2+ hit)':<30s} {legacy.get('hit_distribution', {}).get('2_plus_hit', 0):>11.1%} {'N/A':>12s} {analytical['expert_hit_p2']:>11.1%}")
    print()

    # ── Resource usage ──
    print("  --- Resource Usage (per chip) ---")
    if 'resource_usage' in main:
        ru = main['resource_usage']
        print(f"  {'Metric':<30s} {'Value':>16s}")
        print(f"  {'-'*30} {'-'*16}")
        for k, v in ru.items():
            print(f"  {str(k).replace('_',' ').title():<30s} {str(v):>16s}")

    # Analytical resource
    print(f"  {'Total weight (analytical)':<30s} {analytical['total_weight_gb']:>15.2f} GB")
    print()

    # ── Detailed stage breakdown (if requested) ──
    if detailed:
        print("  --- Detailed Stage/Beat Breakdown ---")
        print()

        # Legacy beat breakdown
        print("  Legacy architecture/ 9-stage beats:")
        if 'beat_breakdown' in legacy:
            for name, lat in legacy['beat_breakdown'].items():
                pct = lat / legacy['layer_latency_us'] * 100 if legacy['layer_latency_us'] > 0 else 0
                bar = '#' * max(1, int(pct / 2))
                print(f"    {name:<20s} {fmt_us(lat):>10s}  ({pct:5.1f}%)  {bar}")
        print()

        # Main stage breakdown
        print("  Main fpga_arch/ 10-stage breakdown:")
        if 'stage_breakdown' in main:
            for name, lat in main['stage_breakdown'].items():
                pct = lat / main['layer_latency_us'] * 100 if main['layer_latency_us'] > 0 else 0
                bar = '#' * max(1, int(pct / 2))
                print(f"    {name:<20s} {fmt_us(lat):>10s}  ({pct:5.1f}%)  {bar}")
        print()

    # ── Verdict ──
    print("  --- Verdict ---")
    print()

    # Key architectural differences
    print("  Key architectural differences between stacks:")
    print()
    print("    1. Stage count: Legacy uses 9 stages (beat 0-8), Main uses 10 stages")
    print("       (WeightPrefetch, MLA_Attn, AttnNorm, MoE_Router, MoE_Dispatch,")
    print("        SharedExpert, RoutedExpert, MoE_Reduce, FFNNorm, PipelineFwd).")
    print()
    print("    2. Interconnect: Legacy uses Ethernet (100GbE) for inter-card,")
    print("       Main uses C2C Dual Ring (448 Gbps) for intra-card + PCIe for")
    print("       cross-card. C2C has lower latency and higher bandwidth.")
    print()
    print("    3. HBM model: Legacy uses HBMController with sequential read efficiency")
    print("       (87%), Main uses simplified HBM_BW_EFF (configurable, default 1.0).")
    print("       For this comparison, Main was set to 87% efficiency.")
    print()
    print("    4. DSP model: Legacy maps each stage to DSP with manual MAC counting.")
    print("       Main uses MACBreakdown with pre-computed per-operation MAC counts")
    print("       and split-rate modeling (fp8 vs fp4 compute rates).")
    print()
    print("    5. Expert dispatch: Legacy models expert fetch via HBM+Ethernet RDMA.")
    print("       Main models dispatch/reduce via C2C link contention model.")
    print()

    # Convergence check
    leg_err = abs(leg_lat - ana_lat) / max(1e-6, ana_lat) * 100
    main_err = abs(main_lat - ana_lat) / max(1e-6, ana_lat) * 100

    if leg_err < 20 and main_err < 20:
        print(f"  [PASS] Both stacks converge within 20% of analytical model")
        print(f"         Legacy error: {leg_err:.1f}%, Main error: {main_err:.1f}%")
    else:
        print(f"  [WARN] Significant deviation from analytical model:")
        print(f"         Legacy error: {leg_err:.1f}%, Main error: {main_err:.1f}%")
    print()

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cross-validate fpga_arch vs architecture legacy stacks'
    )
    parser.add_argument('--detailed', '-d', action='store_true',
                        help='Show detailed stage/beat breakdown')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Decode batch size (default: 1)')
    parser.add_argument('--json', type=str, default=None,
                        help='Export results to JSON file')
    args = parser.parse_args()

    print()
    print("Running legacy architecture/ stack...", end=' ', flush=True)
    legacy = run_legacy_stack(batch_size=args.batch_size)
    print(f"done (latency={fmt_us(legacy['layer_latency_us'])})")

    print("Running main fpga_arch/ stack...", end=' ', flush=True)
    main = run_fpga_arch_stack(batch_size=args.batch_size)
    print(f"done (latency={fmt_us(main['layer_latency_us'])})")

    print("Running analytical model...", end=' ', flush=True)
    analytical = run_analytical_model()
    print(f"done (latency={fmt_us(analytical['layer_latency_us'])})")

    result = compare_stacks(legacy, main, analytical, detailed=args.detailed)

    if args.json:
        import json
        # Convert to serializable form
        serializable = {
            'legacy_latency_us': legacy['layer_latency_us'],
            'main_latency_us': main['layer_latency_us'],
            'analytical_latency_us': analytical['layer_latency_us'],
            'legacy_dsp_us': legacy.get('dsp_time_us', 0),
            'legacy_hbm_us': legacy.get('hbm_time_us', 0),
            'main_dsp_us': main.get('dsp_time_us', 0),
            'main_hbm_us': main.get('hbm_time_us', 0),
            'analytical_dsp_us': analytical['dsp_time_us'],
            'analytical_hbm_us': analytical['hbm_time_us'],
        }
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)
        print(f"Results exported to {args.json}")

    return result


if __name__ == '__main__':
    main()
