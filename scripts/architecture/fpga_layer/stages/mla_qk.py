"""
节拍 1 — MLA Q/K 低秩投影 + RoPE.

目的: 用 SRAM 中的 fp4 权重, 将 FP8 activation 投影到 MLA 低秩空间,
      并注入 RoPE 位置编码.

计算步骤:
  1. q_latent = W_Q_down [fp4] @ x [FP8]   → [batch, 1536]  FP8
  2. kv_latent = W_KV_down [fp4] @ x [FP8] → [batch, 512]   FP8
  3. RoPE: 在 q_latent[:, :64] 和 kv_latent[:, :64] 上施加旋转位置编码

权重: 全部在 SRAM (~1.5 MB fp4)
DSP:  fp4×FP8 模式, MACs ≈ 1536*7168*2 + 512*7168*2 per batch
物理资源: SRAM(读权重) + DSP(fp4×FP8)
"""

from .base import PipelineStage, StageContext
from ... import config
from ..phys.dsp_array import DSPArray


class MLAQKStage(PipelineStage):
    name: str = "mla_qk"
    beat: int = 1
    description: str = "Q/K low-rank proj + RoPE: SRAM fp4 weights × FP8 act"
    input_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    output_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    weight_source: str = "sram"
    precision: str = "fp4×fp8"
    dsp_macs_million: float = 0.0

    def __init__(self, dsp: DSPArray):
        self.dsp = dsp
        self._q_down_macs = config.MODEL_HIDDEN_SIZE * config.MODEL_Q_LORA_RANK * 2 / 1e6
        self._kv_down_macs = config.MODEL_HIDDEN_SIZE * config.MODEL_KV_LORA_RANK * 2 / 1e6
        self._rope_macs = config.MODEL_QK_ROPE_DIM * 4 / 1e6  # sin/cos + rotate

    def _compute_latency(self, ctx: StageContext) -> float:
        total_macs = (self._q_down_macs + self._kv_down_macs) * ctx.batch_size
        total_macs += self._rope_macs * ctx.batch_size
        self.dsp_macs_million = total_macs
        # TP 分摊
        macs_per_card = total_macs / config.SYS_TP_SIZE
        return macs_per_card / self.dsp.tops

    def _transform(self, ctx: StageContext) -> StageContext:
        ctx = ctx.clone()
        # 将 Q down-proj 和 KV down-proj 记录到 DSP
        batch = ctx.batch_size
        self.dsp.gemm(config.MODEL_Q_LORA_RANK, config.MODEL_HIDDEN_SIZE, batch,
                       weight_precision='fp4', activation_precision='fp8',
                       name='mla_q_down')
        self.dsp.gemm(config.MODEL_KV_LORA_RANK, config.MODEL_HIDDEN_SIZE, batch,
                       weight_precision='fp4', activation_precision='fp8',
                       name='mla_kv_down')
        return ctx
