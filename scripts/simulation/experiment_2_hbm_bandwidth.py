"""
实验 2: HBM 带宽仿真
Experiment 2: HBM Bandwidth Simulation

验证 MoE 专家加载模式下的有效 HBM 带宽,
与 H100 HBM3 做对比。
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moe_router import MoERouter, analyze_expert_distribution, DEFAULT_CFG


def run_experiment_2(expert_size_mb=33.0, hbm_bw_gbps=920.0,
                     experts_per_card=13, num_experts=384,
                     top_k=6, num_tokens=2000):
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  实验 2: HBM 带宽仿真 — MoE 专家加载".center(50) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print(f"  配置: 专家大小={expert_size_mb:.0f} MB (fp4), HBM 理论带宽={hbm_bw_gbps:.0f} GB/s")
    print(f"        专家/卡={experts_per_card}, 总专家={num_experts}, Top-K={top_k}")
    print(f"        模拟 Token 数={num_tokens}")
    print()
    print("  关键假设:")
    print(f"    · 确定性权重 (Attention + Shared Expert + Router = ~9.3 MB) 常驻 SRAM")
    print(f"    · 只有被选中的 routed expert 从 HBM 加载")
    print(f"    · HBM 顺序读效率 87% → 有效带宽 = {hbm_bw_gbps * 0.87:.0f} GB/s")
    print()

    # ── Step 1: 专家激活分布 ──
    print("  ┌─ Step 1: 专家激活分布 (Power-Law) ─────────────────────┐")
    router = MoERouter(DEFAULT_CFG)
    dist = analyze_expert_distribution(router, num_tokens=num_tokens)

    print(f"  │ Top-20% 专家承载:    {dist['top_20pct_concentration']:.1%} 的流量          │")
    print(f"  │ 单卡专家命中概率:     {dist['per_card_hit_prob']:.4f}                     │")
    print( "  │                                                          │")
    p0, p1, p2 = dist['p_0_hit'], dist['p_1_hit'], dist['p_2_plus_hit']
    print(f"  │ 每 Token 每层 本地命中分布:                              │")
    print(f"  │   P(0 命中) = {p0:.1%}  ← SRAM 全部命中, 零 HBM 读取  │")
    print(f"  │   P(1 命中) = {p1:.1%}  ← 需加载 1 个专家             │")
    print(f"  │   P(2+命中) = {p2:.1%}  ← 需加载 2+ 个专家            │")
    print( "  └──────────────────────────────────────────────────────────┘")
    print()

    # ── Step 2: HBM 访问时间 ──
    print("  ┌─ Step 2: HBM 访问时间 (含 SRAM 缓存) ──────────────────┐")
    seq_bw = hbm_bw_gbps * 0.87  # GB/s

    # 0 命中: 全部在 SRAM
    hbm_mb_0 = 0.0
    hbm_us_0 = 0.0

    # 1 命中: 加载 1 个专家 + router 开销
    hbm_mb_1 = expert_size_mb + 0.37
    hbm_us_1 = hbm_mb_1 / (seq_bw / 1000)

    # 2 命中: 加载 2 个专家 + router 开销
    hbm_mb_2 = 2 * expert_size_mb + 0.37
    hbm_us_2 = hbm_mb_2 / (seq_bw / 1000)

    weighted_time = p0 * hbm_us_0 + p1 * hbm_us_1 + p2 * hbm_us_2
    avg_mb = p0 * hbm_mb_0 + p1 * hbm_mb_1 + p2 * hbm_mb_2
    effective_bw = avg_mb / max(weighted_time, 1e-6) * 1000

    print(f"  │ 情况 A — 0 命中 ({p0:.1%}): {hbm_mb_0:5.1f} MB → {hbm_us_0:6.1f} μs              │")
    print(f"  │ 情况 B — 1 命中 ({p1:.1%}): {hbm_mb_1:5.1f} MB → {hbm_us_1:6.1f} μs              │")
    print(f"  │ 情况 C — 2+命中 ({p2:.1%}): {hbm_mb_2:5.1f} MB → {hbm_us_2:6.1f} μs              │")
    print(f"  │                                                          │")
    print(f"  │ 加权平均 HBM 时间: {weighted_time:.1f} μs/层                     │")
    print(f"  │ 加权平均 HBM 数据: {avg_mb:.1f} MB/层                        │")
    print(f"  │ 有效 HBM 带宽:     {effective_bw:.0f} GB/s ({effective_bw/hbm_bw_gbps:.1%} 理论峰值)   │")
    print( "  └──────────────────────────────────────────────────────────┘")
    print()

    # ── Step 3: H100 对比 ──
    print("  ┌─ Step 3: FPGA vs H100 HBM 对比 ────────────────────────┐")
    h100_bw = 3350  # GB/s
    h100_eff = h100_bw * 0.80  # HBM3 顺序读效率
    # H100: BF16, 所有权重从 HBM 加载, 6 experts + 确定性权重
    h100_mb = (6 * expert_size_mb + 9.3) * 4  # BF16 = 4× fp4
    h100_us = h100_mb / (h100_eff / 1000)

    print(f"  │                 FPGA (fp4)          H100 (BF16)         │")
    print(f"  │ 权重格式       fp4 E2M1             BF16                │")
    print(f"  │ 数据量/层      {avg_mb:6.1f} MB          {h100_mb:6.0f} MB         │")
    print(f"  │ 有效带宽       {seq_bw:5.0f} GB/s          {h100_eff:5.0f} GB/s        │")
    print(f"  │ HBM 时间/层    {weighted_time:6.1f} μs          {h100_us:6.0f} μs         │")
    print(f"  │                                                          │")
    ratio = weighted_time / h100_us
    print(f"  │ FPGA HBM 时间 = H100 的 {ratio:.1%}                            │")
    print( "  └──────────────────────────────────────────────────────────┘")
    print()

    # ── 判定 ──
    target_60 = hbm_bw_gbps * 0.60
    target_40 = hbm_bw_gbps * 0.40

    print("  ╔" + "═" * 50 + "╗")
    if effective_bw >= target_60:
        print("  ║" + "  结论: [PASS] — 有效带宽 >= 60% 理论峰值".center(46) + "║")
    elif effective_bw >= target_40:
        print("  ║" + "  结论: [WARN] — 有效带宽在 40-60% 之间".center(46) + "║")
    else:
        print("  ║" + "  结论: [FAIL] — 触发 Go/No-Go #3".center(46) + "║")
    print("  ╚" + "═" * 50 + "╝")
    print()
    print(f"  关键发现:")
    print(f"    · SRAM 缓存消除 {p0:.1%} 的 HBM 读取 (P(0 命中))")
    print(f"    · 有效带宽 = {effective_bw:.0f} GB/s (需 ≥{target_60:.0f} GB/s 即 60% 理论峰值)")
    print(f"    · FPGA HBM 访问时间仅为 H100 的 {ratio:.1%} — 因为 fp4 密度 4× + SRAM 缓存")
    print()

    return {
        'effective_bw_gbps': effective_bw,
        'weighted_hbm_time_us': weighted_time,
        'p_0_hit': p0,
        'p_1_hit': p1,
        'p_2_hit': p2,
        'h100_time_us': h100_us,
    }


if __name__ == '__main__':
    run_experiment_2()
