"""
节拍 3 — Shared Expert FFN (SwiGLU).

目的: 对所有 token 执行 shared expert 前向传播.
      这是 DeepSeek MoE 的"共享底座" — 不经过路由, 每 token 都过.

计算步骤:
  1. gate = W_gate [fp4] @ x [FP8]  → [batch, 3072]  FP8
  2. up   = W_up [fp4] @ x [FP8]    → [batch, 3072]  FP8
  3. h    = SiLU(gate) * up          → [batch, 3072]  FP8 (element-wise)
  4. out  = W_down [fp4] @ h [FP8]  → [batch, 7168]  FP8

权重: 全部在 SRAM (~33 MB fp4, gate + up + down)
DSP:  fp4×FP8 (GEMM) + FP8×FP8 (SiLU gate)
MACs: ~66M / 层 (全量), TP 分摊后 ~8.8M/卡
物理资源: SRAM(读权重) + DSP(fp4×FP8)
"""

from .base import PipelineStage, StageContext
from ... import config
from ..phys.dsp_array import DSPArray


class SharedExpertStage(PipelineStage):
    name: str = "shared_expert"
    beat: int = 3
    description: str = "Shared Expert SwiGLU: gate/up/down (SRAM fp4, ~66M MACs)"
    input_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    output_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    weight_source: str = "sram"
    precision: str = "fp4×fp8"
    dsp_macs_million: float = 0.0

    def __init__(self, dsp: DSPArray):
        self.dsp = dsp
        self.D = config.MODEL_HIDDEN_SIZE
        self.I = config.MODEL_INTERMEDIATE_SIZE

    def _compute_latency(self, ctx: StageContext) -> float:
        batch = ctx.batch_size
        # gate: [3072, 7168] @ [7168, batch] + up: [3072, 7168] @ [7168, batch]
        # down: [7168, 3072] @ [3072, batch]
        gate_macs = self.I * self.D * batch * 2 / 1e6
        up_macs = self.I * self.D * batch * 2 / 1e6
        down_macs = self.D * self.I * batch * 2 / 1e6
        total = gate_macs + up_macs + down_macs
        self.dsp_macs_million = total
        return total / self.dsp.tops / config.SYS_TP_SIZE

    def _transform(self, ctx: StageContext) -> StageContext:
        ctx = ctx.clone()
        batch = ctx.batch_size
        self.dsp.gemm(self.I, self.D, batch, weight_precision='fp4', name='shared_gate')
        self.dsp.gemm(self.I, self.D, batch, weight_precision='fp4', name='shared_up')
        self.dsp.silu_gate(batch, self.I)
        self.dsp.gemm(self.D, self.I, batch, weight_precision='fp4', name='shared_down')
        return ctx
