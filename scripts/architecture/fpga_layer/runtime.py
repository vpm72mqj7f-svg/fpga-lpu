"""
FPGA Runtime — 单卡 FPGA 顶层, 兼容旧接口.

内部使用 Pipeline (stages 流水线) 执行推理.
对外暴露与旧 runtime.py 相同的 execute_layer / execute_full_model 接口.
"""

from dataclasses import dataclass, field
from .. import config
from .pipeline import Pipeline, LayerBeatRecord


@dataclass
class LayerExecution:
    """一层执行的兼容记录 (与旧 runtime.py 接口一致)."""
    layer_idx: int
    batch_size: int
    seq_len: int
    expert_ids: list[int] = field(default_factory=list)
    sram_read_us: float = 0.0
    hbm_read_us: float = 0.0
    dsp_total_us: float = 0.0
    pcie_xfer_us: float = 0.0
    total_us: float = 0.0
    beat_breakdown: dict = field(default_factory=dict)


class FPGARuntime:
    """单卡 FPGA 顶层 — 内部使用 Pipeline 流水线."""

    def __init__(self, card_id: int = 0, loaded_experts: list[int] = None):
        self.card_id = card_id
        self.pipeline = Pipeline(card_id=card_id, loaded_experts=loaded_experts)
        self.executions: list[LayerExecution] = []

    def execute_layer(
        self,
        layer_idx: int,
        batch_size: int,
        seq_len: int,
        expert_ids: list[int],
        input_bytes: int = 0,
    ) -> LayerExecution:
        """执行一层 — 返回兼容旧格式."""
        rec = self.pipeline.execute_layer(layer_idx, batch_size, seq_len)

        exec_rec = LayerExecution(
            layer_idx=layer_idx,
            batch_size=batch_size,
            seq_len=seq_len,
            expert_ids=list(expert_ids),
            dsp_total_us=rec.dsp_total_us,
            hbm_read_us=rec.hbm_total_us,
            pcie_xfer_us=rec.pcie_total_us,
            total_us=rec.total_us,
            beat_breakdown=rec.beat_latencies,
        )
        self.executions.append(exec_rec)
        return exec_rec

    def execute_full_model(self, batch_size: int, seq_len: int,
                           expert_routing: list[list[int]]) -> tuple[float, list[LayerExecution]]:
        """执行全部 61 层."""
        records = []
        total = 0.0
        for layer_idx in range(config.MODEL_NUM_LAYERS):
            eids = expert_routing[layer_idx] if layer_idx < len(expert_routing) else []
            rec = self.execute_layer(layer_idx, batch_size, seq_len, eids, 0)
            total += rec.total_us
            records.append(rec)
        return total, records

    @property
    def stats(self) -> dict:
        total = len(self.executions)
        avg = sum(e.total_us for e in self.executions) / max(total, 1) if total > 0 else 0
        return {
            'card_id': self.card_id,
            'total_layers': total,
            'avg_total_us': avg,
            **self.pipeline.stats,
        }

    def reset(self):
        self.pipeline.reset()
        self.executions.clear()
