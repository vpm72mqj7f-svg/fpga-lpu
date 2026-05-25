"""
节拍 0 — PCIe RX: Host → FPGA, activation 到达.

目的: 接收 vLLM Host 下发的 FP8 activation tensor, 写入片上 staging buffer.

数据通路: Host DRAM → PCIe Gen5 x16 DMA → FPGA SRAM staging buffer

输入: (无, 外部 PCIe)
输出: activation [batch, 7168] FP8, 在 staging buffer 中就绪
消耗的物理资源: PCIe (接收), SRAM (写 staging buffer)
DSP: 无
权重: 无

延迟 = PCIe DMA 启动(2μs) + activation 传输时间
"""

from .base import PipelineStage, StageContext
from ... import config


class PCIeRxStage(PipelineStage):
    name: str = "pcie_rx"
    beat: int = 0
    description: str = "Host→FPGA: 接收 activation [batch,7168]FP8 via PCIe"
    input_shape: tuple = (0, 0)
    output_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    weight_source: str = "none"
    precision: str = "none"
    dsp_macs_million: float = 0.0

    def __init__(self, pcie):
        self.pcie = pcie
        self._act_bytes = 0

    def _compute_latency(self, ctx: StageContext) -> float:
        self._act_bytes = ctx.batch_size * ctx.hidden * 1  # FP8 = 1 byte/elem
        return self.pcie.latency(self._act_bytes)

    def _transform(self, ctx: StageContext) -> StageContext:
        ctx = ctx.clone()
        ctx.activation_ready = True
        latency = self._compute_latency(ctx)
        self.pcie.send(self._act_bytes, f"L{ctx.layer_idx}_rx")
        return ctx
