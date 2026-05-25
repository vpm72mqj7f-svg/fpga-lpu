"""
节拍 7 — 汇总: Shared + Routed 结果求和 + RMS Norm.

目的: 将 Shared Expert 输出和所有 Routed Expert 输出按 router weight 加权求和,
      然后过 RMS Norm, 产生本层的最终 hidden state.

计算步骤:
  1. combined = shared_out + sum(router_weight[i] * routed_out[i])
     shared_out:       [batch, 7168] FP8 (来自节拍3)
     routed_out[i]:    [batch, 7168] FP8 (节拍6本地 + 节拍5远端)
  2. hidden_state = RMS_Norm(combined)  → [batch, 7168] FP8

权重: 无
DSP:  FP8×FP8 element-wise MAD + RMS Norm (极小)
物理资源: DSP(FP8×FP8 element-wise ops)
"""

from .base import PipelineStage, StageContext
from ... import config
from ..phys.dsp_array import DSPArray


class AggregateStage(PipelineStage):
    name: str = "aggregate"
    beat: int = 7
    description: str = "Shared + Sum(weighted Routed) + RMS Norm"
    input_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    output_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    weight_source: str = "none"
    precision: str = "fp8×fp8"
    dsp_macs_million: float = 0.0

    def __init__(self, dsp: DSPArray):
        self.dsp = dsp

    def _compute_latency(self, ctx: StageContext) -> float:
        batch = ctx.batch_size
        n_experts = len(ctx.hit_experts) + len(ctx.miss_experts)
        # element-wise sum: n_experts × batch × hidden adds
        sum_macs = n_experts * batch * config.MODEL_HIDDEN_SIZE / 1e6
        # RMS Norm: batch × hidden × 2 (square + scale)
        norm_macs = batch * config.MODEL_HIDDEN_SIZE * 2 / 1e6
        total = sum_macs + norm_macs
        self.dsp_macs_million = total
        return max(total / self.dsp.tops, 0.01)

    def _transform(self, ctx: StageContext) -> StageContext:
        ctx = ctx.clone()
        batch = ctx.batch_size

        self.dsp.rms_norm(batch, config.MODEL_HIDDEN_SIZE)
        ctx.allreduce_needed = True
        ctx.allreduce_bytes = batch * config.MODEL_HIDDEN_SIZE * 1  # FP8

        return ctx
