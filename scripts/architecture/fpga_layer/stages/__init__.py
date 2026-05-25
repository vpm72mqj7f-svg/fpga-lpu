"""
流水线阶段层 — 每个节拍一个模块, token 逐拍推进.

Stage 流水线:
  Beat 0: pcie_rx        Host→FPGA, activation 到达
  Beat 1: mla_qk         Q/K 低秩投影 + RoPE (SRAM fp4)
  Beat 2: mla_attention  Q/K/V up-proj + attention + O-proj
  Beat 3: shared_expert  Shared Expert SwiGLU (SRAM fp4)
  Beat 4: router         MoE Router Top-K (SRAM fp4)
  Beat 5: expert_fetch   获取 expert 权重 (HBM 命中/以太网 miss)
  Beat 6: routed_expert  Routed Expert SwiGLU (HBM fp4)
  Beat 7: aggregate      Shared + Routed 求和 + RMS Norm
  Beat 8: pcie_tx        FPGA→Host, hidden state 返回
"""

from .base import PipelineStage, StageContext
from .pcie_rx import PCIeRxStage
from .mla_qk import MLAQKStage
from .mla_attention import MLAAttentionStage
from .shared_expert import SharedExpertStage
from .router import RouterStage
from .expert_fetch import ExpertFetchStage
from .routed_expert import RoutedExpertStage
from .aggregate import AggregateStage
from .pcie_tx import PCIeTxStage
