"""
节拍 6 — Routed Expert FFN (SwiGLU).

目的: 对 Router 选中的 expert 执行前向传播.
      每个 expert 独立计算, 本地 expert 用 HBM 权重, 远端 expert 结果由节拍5 的以太网拉回.

计算步骤 (per expert):
  1. gate = W_gate [fp4] @ x [FP8]  → [batch, 3072]  FP8
  2. up   = W_up [fp4] @ x [FP8]    → [batch, 3072]  FP8
  3. h    = SiLU(gate) * up          → [batch, 3072]  FP8
  4. out  = W_down [fp4] @ h [FP8]  → [batch, 7168]  FP8
  5. weighted = out * router_weight  → [batch, 7168]  FP8

MACs: 每个 expert ~66M, 多个 expert 串行 (或可流水, 此处保守估计串行)
权重: HBM (本地命中) / 远端 FPGA (miss, 结果已在节拍5拉回)
物理资源: HBM(权重) + DSP(fp4×FP8)
"""

from .base import PipelineStage, StageContext
from ... import config
from ..phys.dsp_array import DSPArray


class RoutedExpertStage(PipelineStage):
    name: str = "routed_expert"
    beat: int = 6
    description: str = "Routed Expert SwiGLU ×N: HBM fp4 weights × FP8 act"
    input_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    output_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    weight_source: str = "hbm"
    precision: str = "fp4×fp8"
    dsp_macs_million: float = 0.0

    def __init__(self, dsp: DSPArray):
        self.dsp = dsp
        self.D = config.MODEL_HIDDEN_SIZE
        self.I = config.MODEL_INTERMEDIATE_SIZE

    def _compute_latency(self, ctx: StageContext) -> float:
        # 只计算本地命中的 expert (miss 的结果来自远端)
        n_local = len(ctx.hit_experts)
        if n_local == 0:
            return 0.0
        # 每个 expert: gate(3072×7168) + up(3072×7168) + down(7168×3072)
        per_expert = (self.I * self.D * 2 + self.I * self.D * 2 + self.D * self.I * 2) / 1e6
        total = per_expert * n_local * ctx.batch_size
        self.dsp_macs_million = total
        return total / self.dsp.tops

    def _transform(self, ctx: StageContext) -> StageContext:
        ctx = ctx.clone()
        batch = ctx.batch_size

        for eid in ctx.hit_experts:
            self.dsp.gemm(self.I, self.D, batch, weight_precision='fp4',
                           name=f'expert_{eid}_gate')
            self.dsp.gemm(self.I, self.D, batch, weight_precision='fp4',
                           name=f'expert_{eid}_up')
            self.dsp.silu_gate(batch, self.I)
            self.dsp.gemm(self.D, self.I, batch, weight_precision='fp4',
                           name=f'expert_{eid}_down')

        return ctx
