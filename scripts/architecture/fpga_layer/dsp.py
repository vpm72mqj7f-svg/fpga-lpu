"""
DSP 计算核 — 模拟 FPGA DSP 阵列的 GEMM/Attention/SwiGLU 计算.

每个 kernel 返回计算延迟 (μs) + 逻辑输出 shape (不实际计算).
延迟公式: time_us = MACs (百万) / DSP_TOPS (TMACs/s)

FPGA DSP 数据通路:
  权重 buffer (fp4) → dequant → DSP (fp8 × fp8 → fp32 acc)
  激活 buffer (FP8) → DMA ────┘

RTL 状态: 全部留空 — 此处只定义接口和时序模型.
"""

from dataclasses import dataclass
import numpy as np
from .. import config


@dataclass
class KernelResult:
    """一次 DSP kernel 执行的结果."""
    name: str
    macs_million: float      # MACs (百万)
    time_us: float           # 计算延迟 μs
    output_shape: tuple      # 输出 tensor shape
    mla_cache_bytes: int = 0  # 如果是 MLA, 产生的 KV cache 大小


class DSPKernels:
    """FPGA DSP 阵列 — 所有计算核的集合.

    核类型:
      - fp4_gemm:    通用 fp4 矩阵乘 (W @ x + bias)
      - mla_attention:  MLA 注意力 (Q/K 压缩 + attention + output proj)
      - swiglu_ffn:   SwiGLU FFN (gate/up/down)
      - rms_norm:     RMS 归一化
    """

    def __init__(self):
        self.tops = config.HW_FPGA_DSP_TOPS
        self.tp_avg = config.SYS_TP_AVG
        self.kernel_calls = []

    def _time(self, macs_million: float) -> float:
        """MACs (百万) / TMACs/s = μs."""
        return macs_million / self.tops

    # ── fp4 GEMM ──

    def fp4_gemm(self, m: int, k: int, n: int, name: str = "gemm") -> KernelResult:
        """W: [M, K] fp4, x: [K, N] FP8 → [M, N] FP32.

        MACs = M × K × N × 2 (每个 fp4 元素 = 1 multiply + 1 accumulate)
        """
        macs = m * k * n * 2 / 1e6  # 百万
        t = self._time(macs)
        r = KernelResult(name=name, macs_million=macs, time_us=t, output_shape=(m, n))
        self.kernel_calls.append(r)
        return r

    # ── MLA Attention ──

    def mla_attention(self, batch_size: int, seq_len: int) -> KernelResult:
        """MLA Attention — TP 分摊后的单卡延迟.

        包含: Q/K 压缩, K/V up-proj, Q up-proj, attention, O proj.
        每卡 MACs = 97M / tp_avg (TP 分摊)
        """
        macs_per_card = config.MACS_MLA_M / self.tp_avg  # 百万
        t = self._time(macs_per_card)

        kv_bytes = batch_size * config.KV_BYTES_PER_TOKEN

        r = KernelResult(
            name="mla_attention",
            macs_million=macs_per_card,
            time_us=t,
            output_shape=(batch_size, config.MODEL_HIDDEN_SIZE),
            mla_cache_bytes=kv_bytes,
        )
        self.kernel_calls.append(r)
        return r

    # ── Shared Expert FFN (SwiGLU) ──

    def shared_expert_ffn(self, batch_size: int) -> KernelResult:
        """Shared Expert — 对所有 token 执行, TP 分摊.

        每卡 MACs = 66M / tp_avg
        """
        macs_per_card = config.MACS_SHARED_EXPERT_M / self.tp_avg
        t = self._time(macs_per_card)

        r = KernelResult(
            name="shared_expert_ffn",
            macs_million=macs_per_card,
            time_us=t,
            output_shape=(batch_size, config.MODEL_HIDDEN_SIZE),
        )
        self.kernel_calls.append(r)
        return r

    # ── Routed Expert FFN (SwiGLU) ──

    def routed_expert_ffn(self, batch_size: int, num_experts: int = 1) -> KernelResult:
        """Routed Expert(s) — 不 TP 分摊, 每卡独立计算自己的 expert.

        每个 expert MACs = 66M (全量, 不除以 tp_avg).
        多个 expert 串行执行 (或者可流水, 此处保守估计串行).
        """
        macs = config.MACS_ROUTED_EXPERT_M * num_experts  # 百万
        t = self._time(macs)

        r = KernelResult(
            name=f"routed_expert_ffn_x{num_experts}",
            macs_million=macs,
            time_us=t,
            output_shape=(batch_size, config.MODEL_HIDDEN_SIZE),
        )
        self.kernel_calls.append(r)
        return r

    # ── 辅助 ──

    def rms_norm(self, batch_size: int) -> KernelResult:
        """RMS Norm — 向量运算, MACs 很小 (~0.01M)."""
        macs = batch_size * config.MODEL_HIDDEN_SIZE * 2 / 1e6
        t = self._time(macs)
        r = KernelResult(name="rms_norm", macs_million=macs, time_us=t,
                         output_shape=(batch_size, config.MODEL_HIDDEN_SIZE))
        self.kernel_calls.append(r)
        return r

    def router_forward(self, batch_size: int) -> KernelResult:
        """MoE Router — logits = x @ W_router.T, W_router: [384, 7168].

        这是小矩阵, 可以 DSP 算也可以 CPU 算. 此处用 DSP 估计.
        MACs = batch × 7168 × 384 × 2 (在 FPGA 上算).
        """
        macs = batch_size * config.MODEL_HIDDEN_SIZE * config.MODEL_NUM_EXPERTS * 2 / 1e6
        t = self._time(macs * 0.1)  # 稀疏, 实际远小于 GEMM
        r = KernelResult(name="router_forward", macs_million=macs * 0.1,
                         time_us=max(t, 0.05), output_shape=(batch_size, config.MODEL_NUM_EXPERTS))
        self.kernel_calls.append(r)
        return r

    @property
    def stats(self) -> dict:
        total_time = sum(k.time_us for k in self.kernel_calls)
        total_macs = sum(k.macs_million for k in self.kernel_calls)
        return {
            'total_kernel_calls': len(self.kernel_calls),
            'total_time_us': total_time,
            'total_macs_million': total_macs,
        }

    def reset(self):
        self.kernel_calls.clear()
