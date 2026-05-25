"""
节拍 4 — MoE Router: Top-K expert 选择.

目的: 用 SRAM 中的 router 权重, 计算每个 token 的 expert affinity,
      选出 Top-K 个 expert, 输出 expert_ids + router_weights.

计算步骤:
  1. logits = W_router [fp4] @ x [FP8]  → [batch, 384]  FP8
  2. probs  = softmax(logits)            → [batch, 384]
  3. top_k  = argsort(probs)[-K:]         → [batch, K]   expert indices
  4. weights = gather(probs, top_k)      → [batch, K]   router weights

权重: W_router 在 SRAM (~0.37 MB fp4), 384×7168
DSP:  稀疏 GEMM + Top-K 排序
      正常 GEMM MACs = 7168*384*2*batch, 但路由矩阵稀疏,
      实际 DSP 消耗约为全量 GEMM 的 10%
物理资源: SRAM(读权重) + DSP(fp4×FP8)
"""

import random
from .base import PipelineStage, StageContext
from ... import config
from ..phys.dsp_array import DSPArray


class RouterStage(PipelineStage):
    name: str = "router"
    beat: int = 4
    description: str = "MoE Router: Top-K expert selection (SRAM fp4 ~0.37MB)"
    input_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    output_shape: tuple = (1, config.MODEL_NUM_EXPERTS)
    weight_source: str = "sram"
    precision: str = "fp4×fp8"
    dsp_macs_million: float = 0.0

    def __init__(self, dsp: DSPArray, hbm=None):
        self.dsp = dsp
        self.hbm = hbm  # 用于 expert 命中判断
        self._rng = random.Random()

    def _compute_latency(self, ctx: StageContext) -> float:
        batch = ctx.batch_size
        # router GEMM: [384, 7168] @ [7168, batch], 稀疏仅 10%
        full_macs = config.MODEL_NUM_EXPERTS * config.MODEL_HIDDEN_SIZE * batch * 2 / 1e6
        self.dsp_macs_million = full_macs * 0.1
        return max(self.dsp_macs_million / self.dsp.tops, 0.05)

    def _transform(self, ctx: StageContext) -> StageContext:
        ctx = ctx.clone()
        batch = ctx.batch_size

        # 记录 router GEMM
        self.dsp.gemm(config.MODEL_NUM_EXPERTS, config.MODEL_HIDDEN_SIZE, batch,
                       weight_precision='fp4', activation_precision='fp8',
                       name='router_forward')

        # 模拟 Top-K 选择: 从 HBM loaded experts 中按概率选
        if self.hbm is not None and self.hbm._loaded:
            # 从本卡 loaded experts 中随机选
            local = list(self.hbm._loaded)
            # 加入一些全局 expert (模拟跨卡)
            all_experts = list(range(config.MODEL_NUM_EXPERTS))
            # 按命中概率选
            p_card = config.MODEL_EXPERTS_PER_FPGA / config.MODEL_NUM_EXPERTS
            chosen = []
            for _ in range(config.MODEL_TOP_K):
                if self._rng.random() < p_card and local:
                    chosen.append(self._rng.choice(local))
                else:
                    chosen.append(self._rng.choice(all_experts))
            ctx.top_k_experts = chosen
            ctx.router_weights = [1.0 / config.MODEL_TOP_K] * config.MODEL_TOP_K
        else:
            # fallback: 纯随机
            ctx.top_k_experts = self._rng.sample(
                range(config.MODEL_NUM_EXPERTS), config.MODEL_TOP_K)
            ctx.router_weights = [1.0 / config.MODEL_TOP_K] * config.MODEL_TOP_K

        return ctx
