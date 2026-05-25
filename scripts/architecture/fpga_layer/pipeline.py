"""
流水线控制器 — 串 9 个 Stage, 管理节拍间 overlap.

核心概念:
  1. 每层推理 = 9 个节拍 (Beat 0→8), token 逐拍推进
  2. 相邻 token 的节拍可以 overlap:
     Token T 在 Beat 6 (DSP) 时, Token T+1 可以进入 Beat 1 (DSP 不同核)
  3. 关键路径 = max(各 Beat 延迟之和, 节拍间不可 overlap 的部分)

节拍间的物理依赖:
  Beat 0-4:  严格串行 (activation 路径上每一步依赖前一步结果)
  Beat 5:    与 Beat 6 一定程度上可 overlap (HBM 读取 + DSP 计算可流水)
  Beat 6-7:  严格串行
  Beat 8:    可与下一层的 Beat 0 overlap (PCIe TX + PCIe RX 同时进行)

延迟模型:
  - 无 overlap: total = sum(beat_0..beat_8) + AllReduce
  - 有 overlap: total = max(compute_critical_path, memory_critical_path)

单层最简路径 (保守, 无 overlap):
  total = PCIe_RX + MLA_QK + MLA_Attn + SharedFFN + Router
        + max(HBM_read, Ethernet_fetch) + RoutedFFN + Aggregate + PCIe_TX
        + AllReduce(以太网)
"""

from dataclasses import dataclass, field
from .. import config
from .phys.pcie import PCIeDMA
from .phys.sram import SRAMBank
from .phys.hbm import HBMController
from .phys.dsp_array import DSPArray
from .phys.ethernet import EthernetMAC
from .stages.base import StageContext
from .stages.pcie_rx import PCIeRxStage
from .stages.mla_qk import MLAQKStage
from .stages.mla_attention import MLAAttentionStage
from .stages.shared_expert import SharedExpertStage
from .stages.router import RouterStage
from .stages.expert_fetch import ExpertFetchStage
from .stages.routed_expert import RoutedExpertStage
from .stages.aggregate import AggregateStage
from .stages.pcie_tx import PCIeTxStage


@dataclass
class LayerBeatRecord:
    """一层各节拍的延迟记录."""
    layer_idx: int
    beat_latencies: dict[str, float] = field(default_factory=dict)
    total_us: float = 0.0
    hit_count: int = 0
    miss_count: int = 0
    dsp_total_us: float = 0.0
    hbm_total_us: float = 0.0
    eth_total_us: float = 0.0
    pcie_total_us: float = 0.0


class Pipeline:
    """单卡 FPGA 流水线控制器.

    初始化时连接物理资源和 9 个 Stage,
    execute_layer() 执行整条流水线并返回节拍级延迟分解.
    """

    def __init__(self, card_id: int = 0, loaded_experts: list[int] = None):
        self.card_id = card_id

        # ── 物理资源 ──
        self.pcie = PCIeDMA()
        self.sram = SRAMBank()
        self.hbm = HBMController(card_id=card_id)
        self.dsp = DSPArray()
        self.eth = EthernetMAC()

        # 加载 experts 到 HBM
        if loaded_experts:
            self.hbm.load(loaded_experts)

        # ── 9 个流水线 Stage ──
        self.stages = [
            PCIeRxStage(self.pcie),
            MLAQKStage(self.dsp),
            MLAAttentionStage(self.dsp),
            SharedExpertStage(self.dsp),
            RouterStage(self.dsp, self.hbm),
            ExpertFetchStage(self.hbm, self.eth, card_id),
            RoutedExpertStage(self.dsp),
            AggregateStage(self.dsp),
            PCIeTxStage(self.pcie),
        ]

        # 执行记录
        self.layer_records: list[LayerBeatRecord] = []

    def execute_layer(self, layer_idx: int, batch_size: int, seq_len: int) -> LayerBeatRecord:
        """执行一层 — token 过全部 9 个节拍.

        Returns:
            LayerBeatRecord 含每个节拍的延迟
        """
        rec = LayerBeatRecord(layer_idx=layer_idx)

        ctx = StageContext(
            batch_size=batch_size,
            seq_len=seq_len,
            layer_idx=layer_idx,
        )

        self.dsp.reset()  # 每层独立统计 DSP

        for stage in self.stages:
            lat = stage.latency_us(ctx)
            ctx = stage.forward(ctx)
            rec.beat_latencies[stage.name] = lat

        # 分类汇总 — DSP 从节拍延迟中算 (加权源=sram/hbm 的节拍)
        rec.dsp_total_us = sum(
            lat for name, lat in rec.beat_latencies.items()
            if name in ('mla_qk', 'mla_attention', 'shared_expert', 'router', 'routed_expert', 'aggregate'))
        rec.hbm_total_us = ctx.hbm_fetch_us
        rec.eth_total_us = ctx.ethernet_fetch_us
        rec.pcie_total_us = rec.beat_latencies.get('pcie_rx', 0) + rec.beat_latencies.get('pcie_tx', 0)
        rec.hit_count = len(ctx.hit_experts)
        rec.miss_count = len(ctx.miss_experts)
        rec.total_us = sum(rec.beat_latencies.values())

        self.layer_records.append(rec)
        return rec

    def execute_full_model(self, batch_size: int, seq_len: int) -> list[LayerBeatRecord]:
        """执行全部 61 层."""
        records = []
        for layer in range(config.MODEL_NUM_LAYERS):
            rec = self.execute_layer(layer, batch_size, seq_len)
            records.append(rec)
        return records

    # ── 汇总统计 ──

    @property
    def beat_summary(self) -> dict:
        """各节拍的平均延迟 (跨所有已执行层)."""
        if not self.layer_records:
            return {}
        n = len(self.layer_records)
        summary = {}
        for rec in self.layer_records:
            for name, lat in rec.beat_latencies.items():
                summary[name] = summary.get(name, 0.0) + lat
        return {k: v / n for k, v in summary.items()}

    @property
    def hit_distribution(self) -> dict:
        """Expert 命中分布."""
        hits = []
        for rec in self.layer_records:
            hits.append(rec.hit_count)
        if not hits:
            return {}
        n = len(hits)
        h0 = sum(1 for h in hits if h == 0)
        h1 = sum(1 for h in hits if h == 1)
        h2 = sum(1 for h in hits if h >= 2)
        return {
            '0_hit': h0 / n,
            '1_hit': h1 / n,
            '2_plus_hit': h2 / n,
        }

    @property
    def stats(self) -> dict:
        n = max(len(self.layer_records), 1)
        avg_total = sum(r.total_us for r in self.layer_records) / n
        avg_dsp = sum(r.dsp_total_us for r in self.layer_records) / n
        avg_hbm = sum(r.hbm_total_us for r in self.layer_records) / n
        avg_eth = sum(r.eth_total_us for r in self.layer_records) / n
        avg_pcie = sum(r.pcie_total_us for r in self.layer_records) / n
        beat_avg = self.beat_summary

        return {
            'card_id': self.card_id,
            'layers_executed': len(self.layer_records),
            'avg_total_per_layer_us': avg_total,
            'avg_dsp_us': avg_dsp,
            'avg_hbm_us': avg_hbm,
            'avg_eth_us': avg_eth,
            'avg_pcie_us': avg_pcie,
            'hit_distribution': self.hit_distribution,
            'beat_breakdown': beat_avg,
            'dsp': self.dsp.stats,
            'hbm': self.hbm.stats,
            'eth': self.eth.stats,
            'pcie': self.pcie.stats,
        }

    def print_layer(self, layer_idx: int = 0):
        """打印一层的节拍分解."""
        if layer_idx >= len(self.layer_records):
            return
        rec = self.layer_records[layer_idx]
        print(f"  Layer {rec.layer_idx}: total={rec.total_us:.1f} μs")
        for stage in self.stages:
            lat = rec.beat_latencies.get(stage.name, 0)
            bar = "█" * max(1, int(lat / max(rec.total_us, 0.1) * 30))
            print(f"    [{stage.beat}] {stage.name:<18s} {bar} {lat:.2f} μs")
        print(f"    hit={rec.hit_count}, miss={rec.miss_count}")

    def reset(self):
        self.dsp.reset()
        self.hbm.reset()
        self.eth.reset()
        self.pcie.reset()
        self.layer_records.clear()
