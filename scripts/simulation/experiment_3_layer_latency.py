"""
实验 3: 单层端到端延迟估算
Experiment 3: Single-Layer End-to-End Latency Estimation

合并 DSP 计算时间 + HBM 访问时间,
计算加权平均层延迟和系统吞吐量。
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_experiment_3(expert_size_mb=33.0, hbm_bw_gbps=920.0,
                     dsp_tops=8.44, dsp_efficiency=0.85,
                     num_experts=384, experts_per_card=13, top_k=6):
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  实验 3: 层延迟估算 — DSP + HBM 端到端".center(50) + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # ── TP 配置 ──
    tp_avg = (7 * 14 + 8 * 16) / 30  # ≈ 7.53, 混合 TP=7 和 TP=8 的节点

    print(f"  配置: FPGA DSP = {dsp_tops:.2f} TMACs/s, 效率 = {dsp_efficiency:.1%}")
    print(f"        HBM = {hbm_bw_gbps:.0f} GB/s (87% 效率 → {hbm_bw_gbps*0.87:.0f} GB/s)")
    print(f"        TP ≈ {tp_avg:.1f} (混合 TP=7/8 节点), 30 卡总量")
    print(f"        专家/卡 = {experts_per_card}/{num_experts}, Top-K = {top_k}")
    print()

    # ── Expert 命中概率 ──
    p_per_expert = experts_per_card / num_experts
    p0 = (1 - p_per_expert) ** top_k
    p1 = top_k * p_per_expert * (1 - p_per_expert) ** (top_k - 1)
    p2_plus = 1 - p0 - p1

    print("  ┌─ Expert 命中分布 (Binomial 模型) ─────────────────────┐")
    print(f"  │ 单专家命中概率: {p_per_expert:.4f}                                │")
    print(f"  │ P(0 命中) = {p0:.1%}  ← 零 HBM 读取                 │")
    print(f"  │ P(1 命中) = {p1:.1%}  ← 加载 1 个专家              │")
    print(f"  │ P(2+命中) = {p2_plus:.1%}  ← 加载 2 个专家              │")
    print( "  └──────────────────────────────────────────────────────┘")
    print()

    # ── DSP 计算时间 ──
    eff_dsp = dsp_tops  # TMACs/s

    # 每卡每层 MACs (TP 分摊):
    #   MLA Attention:    97M / tp_avg  ≈ 12.9M  (TP 分摊)
    #   Shared Expert:    66M / tp_avg  ≈  8.8M  (TP 分摊)
    #   1 Routed Expert:  66M  (本地, 不 TP 分摊)
    mla_macs    = 97 / tp_avg   # ≈ 12.9M
    shared_macs = 66 / tp_avg   # ≈  8.8M
    expert_macs = 66            # 不 TP 分摊

    macs = {
        0: mla_macs + shared_macs,                   # ≈ 21.7M
        1: mla_macs + shared_macs + expert_macs,      # ≈ 87.7M
        2: mla_macs + shared_macs + 2 * expert_macs,  # ≈ 153.7M
    }

    dsp_time = {}
    for k in [0, 1, 2]:
        # MACs (百万) / TMACs/s (万亿/秒) = μs (单位直接抵消)
        dsp_time[k] = macs[k] / eff_dsp  # M_MACs / TMACs/s = μs

    print("  ┌─ DSP 计算时间 (TP 分摊后, 每卡) ─────────────────────┐")
    print(f"  │ {'情况':>8s}  {'MACs/层':>10s}  {'DSP 时间':>10s}  {'占比':>8s}  │")
    print(f"  │ {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}  │")
    for k in [0, 1, 2]:
        label = f"{k}-命中"
        mac_str = f"{macs[k]:.1f} M"
        time_str = f"{dsp_time[k]:.1f} μs"
        pct = f"{p0 if k==0 else (p1 if k==1 else p2_plus):.1%}"
        print(f"  │ {label:>8s}  {mac_str:>10s}  {time_str:>10s}  {pct:>8s}  │")
    print( "  └──────────────────────────────────────────────────────┘")
    print()

    # ── HBM 访问时间 ──
    seq_bw = hbm_bw_gbps * 0.87

    hbm_mb = {
        0: 0.0,
        1: expert_size_mb + 0.37,
        2: 2 * expert_size_mb + 0.37,
    }

    hbm_time = {}
    for k in [0, 1, 2]:
        hbm_time[k] = hbm_mb[k] / (seq_bw / 1000)

    print("  ┌─ HBM 访问时间 ───────────────────────────────────────┐")
    print(f"  │ 顺序读带宽: {seq_bw:.0f} GB/s (87% × {hbm_bw_gbps:.0f})                           │")
    print(f"  │ {'情况':>8s}  {'数据量':>10s}  {'HBM 时间':>10s}             │")
    print(f"  │ {'─'*8}  {'─'*10}  {'─'*10}             │")
    for k in [0, 1, 2]:
        label = f"{k}-命中"
        mb_str = f"{hbm_mb[k]:.1f} MB"
        time_str = f"{hbm_time[k]:.1f} μs"
        print(f"  │ {label:>8s}  {mb_str:>10s}  {time_str:>10s}             │")
    print( "  └──────────────────────────────────────────────────────┘")
    print()

    # ── 每情况延迟 = max(DSP, HBM) ──
    probs = {0: p0, 1: p1, 2: p2_plus}
    latency = {}
    dsp_util = {}

    print("  ┌─ 每情况端到端延迟 (瓶颈 = max(DSP, HBM)) ─────────────┐")
    print(f"  │ {'情况':>8s}  {'DSP':>8s}  {'HBM':>8s}  {'延迟':>8s}  {'DSP利用':>8s}  │")
    print(f"  │ {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  │")
    for k in [0, 1, 2]:
        latency[k] = max(dsp_time[k], hbm_time[k])
        dsp_util[k] = dsp_time[k] / latency[k] if latency[k] > 0 else 1.0
        label = f"{k}-命中"
        d_str = f"{dsp_time[k]:.1f}"
        h_str = f"{hbm_time[k]:.1f}"
        l_str = f"{latency[k]:.1f}"
        u_str = f"{dsp_util[k]:.1%}"
        print(f"  │ {label:>8s}  {d_str:>8s}  {h_str:>8s}  {l_str:>8s}  {u_str:>8s}  │")

    # 加权平均
    weighted_lat = p0 * latency[0] + p1 * latency[1] + p2_plus * latency[2]
    weighted_dsp_busy = p0 * dsp_time[0] + p1 * dsp_time[1] + p2_plus * dsp_time[2]
    weighted_dsp_util = weighted_dsp_busy / weighted_lat

    print(f"  │                                                      │")
    print(f"  │ 加权平均延迟:  {weighted_lat:.1f} μs/层                         │")
    print(f"  │ 加权 DSP 利用: {weighted_dsp_util:.1%}                              │")
    print( "  └──────────────────────────────────────────────────────┘")
    print()

    # ── 吞吐量 ──
    # 架构: 30 卡全 TP 模式, 每卡处理每层的 TP 分片
    # 所有卡同时工作于同一层, 61 层串行处理
    # 单 token 延迟 = 61 * weighted_lat (无流水线加速)
    # 吞吐 = 1 / 单token延迟 (batch_size=1, 串行模式)
    # 注意: 真实系统可流水线叠加多 token, 吞吐可提升 5-10x
    total_layers = 61
    num_cards = 30
    us_per_token = weighted_lat * total_layers  # 单 token 串行延迟
    tok_per_sec_serial = 1e6 / us_per_token     # 串行吞吐 (batch=1)

    print("  ┌─ 系统吞吐量估算 ─────────────────────────────────────┐")
    print(f"  │ 总层数: {total_layers}, 总卡数: {num_cards}, TP≈{tp_avg:.1f}                        │")
    print(f"  │                                                      │")
    print(f"  │ 单 Token 延迟 = {total_layers} 层 x {weighted_lat:.1f} μs/层")
    print(f"  │                = {us_per_token:.0f} μs = {us_per_token/1000:.1f} ms")
    print(f"  │                                                      │")
    print(f"  │ 串行吞吐 (batch=1):        {tok_per_sec_serial:.0f} tok/s")
    print(f"  │ 流水线吞吐 (x{num_cards} batch):  ~{tok_per_sec_serial * num_cards:.0f} tok/s")
    print(f"  │  (注: 实际吞吐受通信/调度开销影响, 在两者之间)      │")
    print( "  └──────────────────────────────────────────────────────┘")
    tok_per_sec = tok_per_sec_serial  # 用串行保守估计作为返回值
    print()

    # ── 判定 ──
    print("  ╔" + "═" * 50 + "╗")
    if weighted_lat <= 15.0:
        print("  ║" + f"  结论: [PASS] — {weighted_lat:.1f} μs <= 15 μs 目标".center(46) + "║")
    elif weighted_lat <= 25.0:
        print("  ║" + f"  结论: [WARN] — {weighted_lat:.1f} μs 在 15-25 μs 区间".center(46) + "║")
    else:
        print("  ║" + "  结论: [FAIL] — 延迟过高".center(46) + "║")
    print("  ╚" + "═" * 50 + "╝")
    print()
    hbm_weighted = p0 * hbm_time[0] + p1 * hbm_time[1] + p2_plus * hbm_time[2]
    bottleneck = "DSP" if weighted_dsp_busy >= hbm_weighted else "HBM"
    print(f"  关键发现:")
    print(f"    - 加权平均层延迟 = {weighted_lat:.1f} μs (目标 <= 15 μs)")
    print(f"    - 瓶颈在 {bottleneck} (DSP={weighted_dsp_busy:.1f} μs, HBM={hbm_weighted:.1f} μs)")
    print(f"    · DSP 加权利用率 = {weighted_dsp_util:.1%}")
    print(f"    · 估算吞吐量 = {tok_per_sec:.0f} tok/s ({num_cards}-卡 TP)")
    print()

    return {
        'weighted_latency_us': weighted_lat,
        'weighted_dsp_utilization': weighted_dsp_util,
        'throughput_tok_s': tok_per_sec,
        'lat_0hit_us': latency[0],
        'lat_1hit_us': latency[1],
        'lat_2hit_us': latency[2],
    }


if __name__ == '__main__':
    run_experiment_3()
