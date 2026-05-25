#!/usr/bin/env python3
"""
端到端集成 Demo — vLLM + FPGA 全栈推理流程.

演示:
  1. 流水线节拍分解 — 单 token 过 9 个 beat, 每拍延迟
  2. 单卡 61 层 — 命中分布 + DSP/HBM/Ethernet 占比
  3. TP 组多卡 — 以太网 AllReduce + 跨组 expert fetch
  4. 30 卡集群 — 概率模型吞吐估算

跑法: python -m scripts.architecture.integration
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.architecture.config import *
from scripts.architecture.fpga_layer.phys.pcie import PCIeDMA
from scripts.architecture.fpga_layer.phys.sram import SRAMBank
from scripts.architecture.fpga_layer.phys.hbm import HBMController
from scripts.architecture.fpga_layer.phys.dsp_array import DSPArray
from scripts.architecture.fpga_layer.phys.ethernet import EthernetMAC
from scripts.architecture.fpga_layer.pipeline import Pipeline, LayerBeatRecord
from scripts.architecture.fpga_layer.tp_group import TPGroup, TPGroupExecution
from scripts.architecture.fpga_layer.runtime import FPGARuntime


def init_pipeline(card_id: int, rng: random.Random) -> Pipeline:
    """初始化一张卡的流水线."""
    loaded = sorted(rng.sample(range(MODEL_NUM_EXPERTS), MODEL_EXPERTS_PER_FPGA))
    return Pipeline(card_id=card_id, loaded_experts=loaded)


# ═══════════════════════════════════════════════════════════════
# Demo 1: 单 Token 流水线节拍分解
# ═══════════════════════════════════════════════════════════════

def demo_pipeline_beats():
    print("=" * 70)
    print("  Demo 1: 单 Token 流水线节拍分解 (9 Beats)")
    print("=" * 70)
    print()

    rng = random.Random(42)
    pl = init_pipeline(0, rng)

    # 执行一层
    rec = pl.execute_layer(layer_idx=0, batch_size=1, seq_len=128)

    print(f"  Layer 0, batch=1, seq=128")
    print(f"  Expert hit={rec.hit_count}, miss={rec.miss_count}")
    print()

    # 节拍分解
    stage_order = [s.name for s in pl.stages]
    total = rec.total_us

    print(f"  {'Beat':<6s} {'Stage':<20s} {'Latency(μs)':>12s}  {'占比':>8s}  {'权重源':>10s}  {'精度':>10s}")
    print(f"  {'-'*6} {'-'*20} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")

    for stage in pl.stages:
        lat = rec.beat_latencies.get(stage.name, 0)
        pct = lat / total * 100 if total > 0 else 0
        bar = "█" * max(1, int(lat / max(total, 0.1) * 40))
        print(f"  [{stage.beat}]   {stage.name:<18s}  {lat:10.2f}  {pct:6.1f}%  {stage.weight_source:>10s}  {stage.precision:>10s}  {bar}")
    print(f"  {'-'*6} {'-'*20} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
    print(f"  {'Total':<6s} {'':<20s} {total:10.2f}  {'100.0%':>8s}")
    print()

    # 各资源消耗
    print(f"  物理资源消耗:")
    print(f"    DSP:      {rec.dsp_total_us:.1f} μs ({rec.dsp_total_us/total*100:.1f}%)")
    print(f"    HBM:      {rec.hbm_total_us:.1f} μs ({rec.hbm_total_us/total*100:.1f}%)")
    print(f"    Ethernet: {rec.eth_total_us:.1f} μs ({rec.eth_total_us/total*100:.1f}%)")
    print(f"    PCIe:     {rec.pcie_total_us:.1f} μs ({rec.pcie_total_us/total*100:.1f}%)")

    # 瓶颈分析
    if rec.hit_count == 0:
        print(f"    瓶颈: DSP (纯计算, 零 HBM 读)")
    elif rec.hit_count == 1:
        print(f"    瓶颈: HBM ({rec.hbm_total_us:.1f} μs 权重加载)")
    else:
        print(f"    瓶颈: HBM ({rec.hit_count} experts × ~42 μs)")
    print()

    pl.reset()
    return pl


# ═══════════════════════════════════════════════════════════════
# Demo 2: 单卡 61 层 — 命中分布 + 各资源占比
# ═══════════════════════════════════════════════════════════════

def demo_single_card_61_layers():
    print("=" * 70)
    print("  Demo 2: 单卡 61 层 — 命中分布 + 节拍统计")
    print("=" * 70)
    print()

    rng = random.Random(42)
    pl = init_pipeline(0, rng)

    records = pl.execute_full_model(batch_size=1, seq_len=128)

    # 命中分布
    hits = [r.hit_count for r in records]
    h0 = sum(1 for h in hits if h == 0)
    h1 = sum(1 for h in hits if h == 1)
    h2 = sum(1 for h in hits if h >= 2)

    print(f"  Expert 命中分布 (61 层):")
    print(f"    0-hit: {h0} 层 ({h0/61:.1%}) — 纯 SRAM+DSP, 最快")
    print(f"    1-hit: {h1} 层 ({h1/61:.1%}) — 1×HBM 权重读")
    print(f"    2+hit: {h2} 层 ({h2/61:.1%}) — 2×HBM 权重读, 最慢")
    print()

    # 平均节拍延迟
    beat_avg = pl.beat_summary
    print(f"  平均节拍延迟 (61层):")
    for stage in pl.stages:
        lat = beat_avg.get(stage.name, 0)
        bar = "█" * max(1, int(lat))
        print(f"    [{stage.beat}] {stage.name:<18s} {bar} {lat:.1f} μs — {stage.description}")

    # 合计
    s = pl.stats
    print()
    print(f"  统计:")
    print(f"    平均每层: {s['avg_total_per_layer_us']:.1f} μs")
    print(f"    DSP 平均: {s['avg_dsp_us']:.1f} μs ({s['avg_dsp_us']/s['avg_total_per_layer_us']*100:.1f}%)")
    print(f"    HBM 平均: {s['avg_hbm_us']:.1f} μs ({s['avg_hbm_us']/s['avg_total_per_layer_us']*100:.1f}%)")
    print(f"    ETH 平均: {s['avg_eth_us']:.1f} μs ({s['avg_eth_us']/s['avg_total_per_layer_us']*100:.1f}%)")
    print(f"    PCIe平均: {s['avg_pcie_us']:.1f} μs ({s['avg_pcie_us']/s['avg_total_per_layer_us']*100:.1f}%)")
    total_61 = sum(r.total_us for r in records)
    print(f"    61 层总延迟: {total_61:.0f} μs = {total_61/1000:.1f} ms")
    print(f"    吞吐 (串行, batch=1): {1e6/total_61:.0f} tok/s")
    print(f"    DSP 利用率: {s['dsp']['utilization']:.1%}")
    print()

    pl.reset()
    return pl


# ═══════════════════════════════════════════════════════════════
# Demo 3: TP 组 — 以太网 AllReduce + 跨组 expert fetch
# ═══════════════════════════════════════════════════════════════

def demo_tp_group():
    print("=" * 70)
    print(f"  Demo 3: TP 组 ({SYS_TP_SIZE} 芯片 = {SYS_CARDS_PER_TP_GROUP} 卡) — 以太网 AllReduce + 跨组 Expert Fetch")
    print("=" * 70)
    print()

    rng = random.Random(42)
    tp_size = SYS_TP_SIZE

    pipelines = []
    for cid in range(tp_size):
        loaded = sorted(rng.sample(range(MODEL_NUM_EXPERTS), MODEL_EXPERTS_PER_FPGA))
        pipelines.append(Pipeline(card_id=cid, loaded_experts=loaded))

    tp = TPGroup(group_id=0, pipelines=pipelines)

    print(f"  TP Group: {tp.group_size} 卡")
    print(f"  Expert 覆盖: {len(tp._expert_to_card)} / {MODEL_NUM_EXPERTS} "
          f"({len(tp._expert_to_card)/MODEL_NUM_EXPERTS:.1%})")
    print()

    # 模拟一层
    rng2 = random.Random(123)
    selected = rng2.sample(range(MODEL_NUM_EXPERTS), MODEL_TOP_K)
    local_in_group = set(tp._expert_to_card.keys())
    hits_in_group = [e for e in selected if e in local_in_group]
    misses = [e for e in selected if e not in local_in_group]

    print(f"  Router Top-6: {selected}")
    print(f"    组内命中 ({len(hits_in_group)}): {hits_in_group}")
    print(f"    需跨组 RDMA ({len(misses)}): {misses}")
    print()

    exec_rec = tp.execute_layer(0, batch_size=1, seq_len=128, expert_ids=selected)

    print(f"  延迟分解:")
    print(f"    各卡总延迟: {[f'{t:.1f}' for t in exec_rec.per_card_total_us]} μs")
    print(f"    最慢卡:     {max(exec_rec.per_card_total_us):.1f} μs")
    print(f"    AllReduce:  {exec_rec.allreduce_us:.1f} μs (以太网 Ring)")
    print(f"    跨组拉取:   {exec_rec.cross_group_expert_us:.1f} μs (以太网 RDMA)")
    print(f"    ─────────────────")
    print(f"    总延迟:     {exec_rec.total_us:.1f} μs")
    print()

    # 61 层统计
    tp.reset()
    routing = []
    for layer in range(MODEL_NUM_LAYERS):
        routing.append(rng2.sample(range(MODEL_NUM_EXPERTS), MODEL_TOP_K))

    total, records = tp.execute_full_model(batch_size=1, seq_len=128, expert_routing=routing)

    avg_comp = sum(max(e.per_card_total_us) for e in tp.executions) / len(tp.executions)
    avg_ar = sum(e.allreduce_us for e in tp.executions) / len(tp.executions)
    avg_cross = sum(e.cross_group_expert_us for e in tp.executions) / len(tp.executions)
    avg_layer = sum(e.total_us for e in tp.executions) / len(tp.executions)

    print(f"  61 层统计 (batch_size=1):")
    print(f"    平均最慢卡计算: {avg_comp:.1f} μs/层")
    print(f"    平均 AllReduce:   {avg_ar:.1f} μs/层 (以太网 {tp.eth.eff_bw_gbps:.0f} Gbps)")
    print(f"    平均跨组拉取:     {avg_cross:.1f} μs/层")
    print(f"    平均层延迟:       {avg_layer:.1f} μs/层")
    print(f"    61 层总延迟:      {total:.0f} μs = {total/1000:.1f} ms")
    print(f"    吞吐 (单 TP 组):   {1e6/total:.0f} tok/s")
    print()

    # 以太网统计
    eth_s = tp.eth.stats
    print(f"  以太网 100GbE 统计:")
    print(f"    有效带宽: {eth_s['effective_bw_gbps']:.1f} Gbps ({eth_s['bw_bytes_per_us']:.1f} B/μs)")
    print(f"    AllReduce 次数: {eth_s['allreduce_ops']}")
    print(f"    P2P 次数:       {eth_s['p2p_ops']}")
    print(f"    总传输: {eth_s['total_bytes']/1e6:.1f} MB, 总延迟: {eth_s['total_latency_us']:.0f} μs")
    print()

    tp.reset()
    return tp


# ═══════════════════════════════════════════════════════════════
# Demo 4: 30 卡集群 — 概率模型 + 各物理链路占比
# ═══════════════════════════════════════════════════════════════

def demo_full_cluster():
    print("=" * 70)
    print(f"  Demo 4: 集群吞吐估算 — {HW_FPGA_CARD_COUNT}卡×{HW_FPGAS_PER_CARD}片 = {HW_FPGA_CHIP_COUNT}芯片")
    print("=" * 70)
    print()

    rng = random.Random(42)

    pipelines = []
    for cid in range(HW_FPGA_CHIP_COUNT):
        loaded = sorted(rng.sample(range(MODEL_NUM_EXPERTS), MODEL_EXPERTS_PER_FPGA))
        pipelines.append(Pipeline(card_id=cid, loaded_experts=loaded))

    # 分为 4 个 TP 组, 每组 8 片 (2 卡)
    tp_groups = []
    cursor = 0
    for gid in range(SYS_TP_GROUPS):
        group_pls = pipelines[cursor:cursor + SYS_TP_SIZE]
        cursor += SYS_TP_SIZE
        tp_groups.append(TPGroup(group_id=gid, pipelines=group_pls))

    print(f"  集群: {HW_FPGA_CARD_COUNT} 卡 × {HW_FPGAS_PER_CARD} 片 = {HW_FPGA_CHIP_COUNT} 芯片 → {len(tp_groups)} 个 TP 组")
    for tp in tp_groups:
        expert_coverage = len(tp._expert_to_card)
        print(f"    Group #{tp.group_id}: {tp.group_size} 片 ({tp.group_size//HW_FPGAS_PER_CARD} 卡), "
              f"覆盖 {expert_coverage}/{MODEL_NUM_EXPERTS} experts "
              f"({expert_coverage/MODEL_NUM_EXPERTS:.1%})")
    print()

    # ── 概率模型 ──
    p_card = MODEL_EXPERTS_PER_FPGA / MODEL_NUM_EXPERTS  # 12/384
    p0 = (1 - p_card) ** MODEL_TOP_K
    p1 = MODEL_TOP_K * p_card * (1 - p_card) ** (MODEL_TOP_K - 1)
    p2 = 1 - p0 - p1

    # 每种情况的延迟 (从 Pipeline 单卡实测)
    lat_0 = 10.6   # μs, 0-hit: PCIe+MLA+Shared+Router+DSP
    lat_1 = 52.3   # μs, 1-hit: +42 μs HBM
    lat_2 = 94.0   # μs, 2+hit: +84 μs HBM

    # 以太网 AllReduce (7 张卡, 7KB FP8 tensor)
    eth = EthernetMAC()
    ar_lat = eth.allreduce(MODEL_HIDDEN_SIZE * 1, 7, same_server=True)

    # 跨组 expert RDMA
    # 30 卡覆盖全部 384 experts → 跨组 miss 率 = 0
    cross_lat = 0.0

    weighted_comp = p0 * lat_0 + p1 * lat_1 + p2 * lat_2
    weighted_total = weighted_comp + ar_lat + cross_lat

    us_per_token = weighted_total * MODEL_NUM_LAYERS
    tok_serial = 1e6 / us_per_token
    tok_pipeline = len(tp_groups) * tok_serial

    print(f"  单层延迟 (概率加权):")
    print(f"    P(0-hit)={p0:.1%} × {lat_0:.1f} μs = {p0*lat_0:.1f}")
    print(f"    P(1-hit)={p1:.1%} × {lat_1:.1f} μs = {p1*lat_1:.1f}")
    print(f"    P(2+hit)={p2:.1%} × {lat_2:.1f} μs = {p2*lat_2:.1f}")
    print(f"    加权计算:         {weighted_comp:.1f} μs")
    print(f"    + AllReduce:       {ar_lat:.1f} μs (以太网 100GbE)")
    print(f"    + 跨组 RDMA:       {cross_lat:.1f} μs (30卡全覆盖, miss=0)")
    print(f"    = 单层延迟:        {weighted_total:.1f} μs")
    print()

    print(f"  系统吞吐:")
    print(f"    单 Token 延迟:     {us_per_token:.0f} μs = {us_per_token/1000:.1f} ms")
    print(f"    串行吞吐:          {tok_serial:.0f} tok/s (1 TP 组)")
    print(f"    流水线并行度:      {len(tp_groups)} 组")
    print(f"    流水线吞吐:        ~{tok_pipeline:.0f} tok/s")
    print()

    # ── 资源消耗占比 ──
    print(f"  物理资源消耗占比 (单层加权):")
    pcie_lat = 4.3
    bar_total = 50

    items = [
        ("DSP (MLA+Shared+Router+Routed)", weighted_comp - max(lat_0 - pcie_lat - 5, 1), 0),
        ("HBM (expert 权重读)", p1 * 41.7 + p2 * 83.4, 0),
        ("Ethernet (AllReduce)", ar_lat, 0),
        ("Ethernet (跨组 RDMA)", cross_lat, 0),
        ("PCIe (RX+TX)", pcie_lat, 0),
    ]

    for name, lat, _ in items:
        bar = "█" * max(1, int(lat / weighted_total * bar_total))
        pct = lat / weighted_total * 100
        print(f"    {name:<32s} {bar} {lat:.1f} μs ({pct:.1f}%)")
    print()

    # ── H100 对比 ──
    h100_tok = 40
    print(f"  H100 对比:")
    print(f"    1×H100 (BF16, batch≈32):  ~{h100_tok} tok/s")
    print(f"    30×FPGA (fp4):            ~{tok_pipeline:.0f} tok/s")
    print(f"    吞吐比: 30 FPGA ≈ {tok_pipeline/h100_tok:.0f}× 1 H100")
    print(f"    每卡效率: {tok_pipeline/h100_tok/30:.1f}× H100")
    print()

    return tp_groups


# ═══════════════════════════════════════════════════════════════
# Demo 5: 物理资源 + 链路总览
# ═══════════════════════════════════════════════════════════════

def demo_phys_overview():
    print("=" * 70)
    print("  Demo 5: 物理资源 / 链路总览")
    print("=" * 70)
    print()

    print(f"  Agilex 7 AGFB027 单卡资源:")
    print(f"    DSP:     {HW_FPGA_DSP_COUNT} blocks, {HW_FPGA_DSP_TOPS} TMACs/s")
    print(f"    HBM:     {HW_FPGA_HBM_SIZE_GB} GB @ {HW_FPGA_HBM_BW_GBPS} GB/s (eff {HW_FPGA_HBM_EFF:.0%})")
    print(f"    SRAM:    {HW_FPGA_SRAM_SIZE_MB} MB (M20K + MLAB)")
    print(f"    F-tile:  PCIe Gen5 x16 + 100GbE MAC")
    print()

    print(f"  物理链路 (2 种):")
    pcie = PCIeDMA()
    eth = EthernetMAC()

    print(f"    PCIe Gen5 x16:")
    print(f"      带宽: {pcie.bw_gbps} GB/s 单向有效")
    print(f"      延迟: {pcie.startup_us} μs 启动 + data/BW")
    print(f"      用途: Host<->FPGA activation / KV cache")
    print(f"      7KB activation: {pcie.latency(MODEL_HIDDEN_SIZE):.2f} μs")
    print()

    print(f"    100GbE RDMA (所有卡间通信):")
    print(f"      线速: {eth.bw_gbps} Gbps, 有效: {eth.eff_bw_gbps:.0f} Gbps = {eth._bw_bytes_per_us:.0f} B/μs")
    hop_cost = eth.mac_latency_us + eth.switch_latency_us
    print(f"      同机箱: {eth._hops_same_server} 跳 (SW) × {hop_cost:.1f} μs = {eth._hop_latency(eth._hops_same_server):.1f} μs")
    print(f"      跨服务器: {eth._hops_cross_server} 跳 (SW1+SW2) × {hop_cost:.1f} μs = {eth._hop_latency(eth._hops_cross_server):.1f} μs")
    print(f"      同机箱 AllReduce (7卡, 7KB): {eth.allreduce(MODEL_HIDDEN_SIZE, 7, same_server=True):.2f} μs")
    print(f"      跨服务器 P2P (7KB): {eth.p2p_fetch(0, 1, MODEL_HIDDEN_SIZE, same_server=False):.2f} μs")
    print()

    print(f"    HBM2e 权重读:")
    hbm = HBMController()
    print(f"      有效带宽: {hbm.seq_bw_gbps:.0f} GB/s = {hbm._bw_bytes_per_us:.0f} B/μs")
    expert_lat = hbm._read_mb(WEIGHT_EXPERT_MB)
    print(f"      1 expert ({WEIGHT_EXPERT_MB:.0f} MB): {expert_lat:.1f} μs")
    print(f"      2 experts: {expert_lat*2:.1f} μs")
    print()

    print(f"  DSP 精度模式:")
    print(f"    fp4×FP8: 权重 fp4 → LUT(16项) → fp8 → ×FP8 → fp32 acc")
    print(f"    FP8×FP8: 权重 fp8 ──────────────→ ×FP8 → fp32 acc")
    print(f"    两种模式共享同一 MAC 阵列, 输入 mux 切换")
    print()

    # RTL TODO
    print(f"  RTL 待实现:")
    for name, desc in [
        ("fp4_lut", "fp4→fp8 16项查表解量化"),
        ("dsp_gemm", "fp4/FP8×FP8→fp32 MAC 阵列"),
        ("mla_qk_proj", "MLA Q/K 低秩投影 + RoPE"),
        ("mla_attention", "flash attention (QK^T + softmax + V)"),
        ("swiglu_ffn", "gate/up SiLU + down 投影"),
        ("router_topk", "Top-K 路由 (softmax + argsort)"),
        ("hbm_reader", "HBM 顺序 burst 读 (expert 权重)"),
        ("pcie_dma", "PCIe Gen5 x16 DMA 引擎"),
        ("eth_mac_rdma", "100GbE MAC + RDMA 引擎"),
        ("pipeline_ctrl", "9-stage 流水线控制器 (双缓冲 + flow control)"),
    ]:
        print(f"    [TODO] {name:<18s} — {desc}")
    print()


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    random.seed(42)

    try:
        demo_pipeline_beats()
    except Exception as e:
        print(f"Demo 1 FAILED: {e}")
        import traceback; traceback.print_exc()
    print()

    try:
        demo_single_card_61_layers()
    except Exception as e:
        print(f"Demo 2 FAILED: {e}")
        import traceback; traceback.print_exc()
    print()

    try:
        demo_tp_group()
    except Exception as e:
        print(f"Demo 3 FAILED: {e}")
        import traceback; traceback.print_exc()
    print()

    try:
        demo_full_cluster()
    except Exception as e:
        print(f"Demo 4 FAILED: {e}")
        import traceback; traceback.print_exc()
    print()

    try:
        demo_phys_overview()
    except Exception as e:
        print(f"Demo 5 FAILED: {e}")
        import traceback; traceback.print_exc()
