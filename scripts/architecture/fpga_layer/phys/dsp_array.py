"""
DSP 计算阵列 — Agilex 7 可变精度 MAC 通路.

两种精度模式共享同一乘法器:
  - FP8 模式: 权重(fp8) × 激活(FP8) → 累加(fp32)
  - fp4 模式: 权重(fp4) → [16项LUT] → fp8 → × 激活(FP8) → 累加(fp32)

硬件资源: 9375 DSP blocks, 8.44 TMACs/s
RTL 接口: 权重输入 + 激活输入 + 模式选择 → MAC 阵列 → 累加输出
"""

from dataclasses import dataclass, field
from ... import config


@dataclass
class DSPOp:
    """一次 DSP 操作的定义."""
    name: str
    m: int            # 输出 dim
    k: int            # 输入 dim
    n: int            # batch dim
    weight_precision: str  # 'fp4' | 'fp8'
    activation_precision: str  # 'fp8'
    macs_million: float
    time_us: float


class DSPArray:
    """FPGA DSP 阵列 — fp4×FP8 / FP8×FP8 双模 MAC 通路.

    延迟公式: time_us = MACs(百万) / TOPS(TMACs/s)
    fp4 模式不增加计算延迟, 只在前端多 16 项 LUT (0.05 ns).
    """

    def __init__(self):
        self.tops = config.HW_FPGA_DSP_TOPS  # 8.44 TMACs/s
        self.dsp_count = config.HW_FPGA_DSP_COUNT  # 9375
        self.ops: list[DSPOp] = []

    def gemm(self, m: int, k: int, n: int,
             weight_precision: str = 'fp4',
             activation_precision: str = 'fp8',
             name: str = "gemm") -> DSPOp:
        """通用矩阵乘: W[m,k] @ X[k,n] → Y[m,n].

        MACs = m × k × n × 2  (每个元素 1 mul + 1 add)
        """
        macs = m * k * n * 2 / 1e6  # 百万
        t = macs / self.tops
        op = DSPOp(name=name, m=m, k=k, n=n,
                   weight_precision=weight_precision,
                   activation_precision=activation_precision,
                   macs_million=macs, time_us=t)
        self.ops.append(op)
        return op

    def attention_score(self, batch: int, heads: int, seq_len: int, head_dim: int) -> DSPOp:
        """Q @ K^T: [B, H, S, D] @ [B, H, D, S] → [B, H, S, S].
        FP8×FP8 纯激活间乘法, 无权重.
        """
        macs = batch * heads * seq_len * seq_len * head_dim * 2 / 1e6
        t = macs / self.tops
        op = DSPOp(name="attention_score", m=seq_len, k=head_dim, n=seq_len,
                   weight_precision='fp8', activation_precision='fp8',
                   macs_million=macs, time_us=t)
        self.ops.append(op)
        return op

    def attention_output(self, batch: int, heads: int, seq_len: int, head_dim: int) -> DSPOp:
        """attn @ V: [B, H, S, S] @ [B, H, D, D] → [B, H, S, D].
        FP8×FP8 纯激活间乘法.
        """
        macs = batch * heads * seq_len * seq_len * head_dim * 2 / 1e6
        t = macs / self.tops
        op = DSPOp(name="attention_output", m=seq_len, k=seq_len, n=head_dim,
                   weight_precision='fp8', activation_precision='fp8',
                   macs_million=macs, time_us=t)
        self.ops.append(op)
        return op

    def rms_norm(self, batch: int, hidden: int) -> DSPOp:
        """RMS Norm: 向量逐元素乘加, MACs 极小."""
        macs = batch * hidden * 2 / 1e6  # square + scale
        t = max(macs / self.tops, 0.01)  # 最小 0.01 μs
        op = DSPOp(name="rms_norm", m=hidden, k=1, n=batch,
                   weight_precision='fp8', activation_precision='fp8',
                   macs_million=macs, time_us=t)
        self.ops.append(op)
        return op

    def silu_gate(self, batch: int, hidden: int) -> DSPOp:
        """SiLU 激活: x * sigmoid(x), MACs 极小."""
        macs = batch * hidden * 4 / 1e6  # approx 4 ops per element
        t = max(macs / self.tops, 0.01)
        op = DSPOp(name="silu_gate", m=hidden, k=1, n=batch,
                   weight_precision='fp8', activation_precision='fp8',
                   macs_million=macs, time_us=t)
        self.ops.append(op)
        return op

    @property
    def utilization(self) -> float:
        total_macs = sum(op.macs_million for op in self.ops)
        total_time = sum(op.time_us for op in self.ops)
        return total_macs / (self.tops * total_time) if total_time > 0 else 0.0

    @property
    def stats(self) -> dict:
        total_macs = sum(op.macs_million for op in self.ops)
        total_time = sum(op.time_us for op in self.ops)
        return {
            'total_ops': len(self.ops),
            'total_macs_million': total_macs,
            'total_time_us': total_time,
            'utilization': self.utilization,
        }

    def reset(self):
        self.ops.clear()
