"""
节拍 2 — MLA Attention 核心.

目的: 用低秩 Q/K/V 计算多头注意力.

计算步骤:
  1. K_full = W_K_up [fp4] @ kv_latent [FP8]  → [batch, 128, head_dim]  ← fp4×FP8
  2. V_full = W_V_up [fp4] @ kv_latent [FP8]  → [batch, 128, head_dim]  ← fp4×FP8
  3. Q_full = W_Q_up [fp4] @ q_latent [FP8]   → [batch, 128, head_dim]  ← fp4×FP8
  4. score = Q @ K^T / sqrt(d)                → [batch, 128, seq, seq]   ← FP8×FP8
  5. attn  = softmax(score) @ V               → [batch, 128, seq ,dim]   ← FP8×FP8
  6. O     = W_O_proj [fp4] @ attn [FP8]      → [batch, 7168]           ← fp4×FP8

权重: up-proj + O-proj 全部在 SRAM (~4.7 MB fp4)
DSP:  fp4×FP8 (步骤1-3,6) + FP8×FP8 (步骤4-5)
物理资源: SRAM(读权重) + DSP(两种精度)
"""

from .base import PipelineStage, StageContext
from ... import config
from ..phys.dsp_array import DSPArray


class MLAAttentionStage(PipelineStage):
    name: str = "mla_attention"
    beat: int = 2
    description: str = "Q/K/V up-proj + attention score + O-proj (SRAM fp4 + DSP)"
    input_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    output_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    weight_source: str = "sram"
    precision: str = "fp4×fp8"
    dsp_macs_million: float = 0.0

    def __init__(self, dsp: DSPArray):
        self.dsp = dsp

    def _compute_latency(self, ctx: StageContext) -> float:
        batch, seq = ctx.batch_size, ctx.seq_len

        # Steps 1-3: up-proj MACs (fp4×FP8)
        k_up = config.MODEL_NUM_HEADS * config.MODEL_V_HEAD_DIM * config.MODEL_KV_LORA_RANK * 2 / 1e6
        v_up = config.MODEL_NUM_HEADS * config.MODEL_V_HEAD_DIM * config.MODEL_KV_LORA_RANK * 2 / 1e6
        q_up = config.MODEL_NUM_HEADS * config.MODEL_V_HEAD_DIM * config.MODEL_Q_LORA_RANK * 2 / 1e6
        up_macs = (k_up + v_up + q_up) * batch

        # Steps 4-5: attention score + output (FP8×FP8)
        head_dim = config.MODEL_V_HEAD_DIM
        score_macs = batch * config.MODEL_NUM_HEADS * seq * seq * head_dim * 2 / 1e6
        attn_out_macs = batch * config.MODEL_NUM_HEADS * seq * seq * head_dim * 2 / 1e6

        # Step 6: O-proj (fp4×FP8)
        o_macs = batch * config.MODEL_HIDDEN_SIZE * config.MODEL_NUM_HEADS * head_dim * 2 / 1e6

        total = up_macs + score_macs + attn_out_macs + o_macs
        self.dsp_macs_million = total
        return total / self.dsp.tops / config.SYS_TP_SIZE

    def _transform(self, ctx: StageContext) -> StageContext:
        ctx = ctx.clone()
        batch, seq = ctx.batch_size, ctx.seq_len
        head_dim = config.MODEL_V_HEAD_DIM

        # fp4×FP8 GEMMs
        self.dsp.gemm(config.MODEL_NUM_HEADS * head_dim, config.MODEL_KV_LORA_RANK,
                       batch, weight_precision='fp4', name='mla_k_up')
        self.dsp.gemm(config.MODEL_NUM_HEADS * head_dim, config.MODEL_KV_LORA_RANK,
                       batch, weight_precision='fp4', name='mla_v_up')
        self.dsp.gemm(config.MODEL_NUM_HEADS * head_dim, config.MODEL_Q_LORA_RANK,
                       batch, weight_precision='fp4', name='mla_q_up')

        # FP8×FP8 attention
        self.dsp.attention_score(batch, config.MODEL_NUM_HEADS, seq, head_dim)
        self.dsp.attention_output(batch, config.MODEL_NUM_HEADS, seq, head_dim)

        # O-proj
        self.dsp.gemm(config.MODEL_HIDDEN_SIZE, config.MODEL_NUM_HEADS * head_dim,
                       batch, weight_precision='fp4', name='mla_o_proj')

        return ctx
