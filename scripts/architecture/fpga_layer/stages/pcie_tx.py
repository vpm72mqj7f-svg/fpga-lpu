"""
节拍 8 — PCIe TX: FPGA → Host, hidden state 返回.

目的: 将本层最终 hidden state [batch, 7168] FP8 通过 PCIe DMA 传回 Host.

数据通路: FPGA SRAM staging buffer → PCIe Gen5 x16 DMA → Host DRAM

输入: hidden_state [batch, 7168] FP8 (来自节拍7 aggregate)
输出: (PCIe 传输完成)
消耗的物理资源: PCIe (发送)
DSP: 无
权重: 无

延迟 = PCIe DMA 启动(2μs) + hidden_state 传输时间
"""

from .base import PipelineStage, StageContext
from ... import config


class PCIeTxStage(PipelineStage):
    name: str = "pcie_tx"
    beat: int = 8
    description: str = "FPGA→Host: 返回 hidden_state [batch,7168]FP8 via PCIe"
    input_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    output_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    weight_source: str = "none"
    precision: str = "none"
    dsp_macs_million: float = 0.0

    def __init__(self, pcie):
        self.pcie = pcie
        self._out_bytes = 0

    def _compute_latency(self, ctx: StageContext) -> float:
        self._out_bytes = ctx.batch_size * ctx.hidden * 1  # FP8
        return self.pcie.latency(self._out_bytes)

    def _transform(self, ctx: StageContext) -> StageContext:
        ctx = ctx.clone()
        latency = self._compute_latency(ctx)
        self.pcie.recv(self._out_bytes, f"L{ctx.layer_idx}_tx")
        ctx.activation_ready = False  # 已送出, 下一层从 RX 重新开始
        return ctx
