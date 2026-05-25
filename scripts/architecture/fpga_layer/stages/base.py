"""
Stage 基类 — 流水线的每个节拍都是一个 Stage.

每个 Stage 必须声明:
  1. beat: 节拍编号 (0-8)
  2. 目的: 这个节拍完成什么计算
  3. 输入/输出 activation shape
  4. 消耗的物理资源 (DSP/SRAM/HBM/PCIe/Ethernet)
  5. 权重来源和精度

关键约束: 每个 Stage 只依赖前一个 Stage 的输出,
不直接访问其他 Stage 的内部状态.
"""

from dataclasses import dataclass, field
from ... import config


@dataclass
class StageContext:
    """在 Stage 间传递的上下文 — 一个 token batch 的当前状态."""
    batch_size: int
    seq_len: int
    hidden: int = config.MODEL_HIDDEN_SIZE

    # 当前 activation (概念上的, 不实际存储)
    activation_ready: bool = True

    # Router 输出 (由 router Stage 填写)
    top_k_experts: list[int] = field(default_factory=list)
    router_weights: list[float] = field(default_factory=list)

    # Expert fetch 状态 (由 expert_fetch Stage 填写)
    hit_experts: list[int] = field(default_factory=list)
    miss_experts: list[int] = field(default_factory=list)
    hbm_fetch_us: float = 0.0
    ethernet_fetch_us: float = 0.0

    # Layer 编号 (用于追踪)
    layer_idx: int = 0

    # TP 组内 AllReduce (由 aggregate + tp_group 填写)
    allreduce_needed: bool = True
    allreduce_bytes: int = 0

    def clone(self) -> 'StageContext':
        return StageContext(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            hidden=self.hidden,
            activation_ready=self.activation_ready,
            top_k_experts=list(self.top_k_experts),
            router_weights=list(self.router_weights),
            hit_experts=list(self.hit_experts),
            miss_experts=list(self.miss_experts),
            hbm_fetch_us=self.hbm_fetch_us,
            ethernet_fetch_us=self.ethernet_fetch_us,
            layer_idx=self.layer_idx,
            allreduce_needed=self.allreduce_needed,
            allreduce_bytes=self.allreduce_bytes,
        )


class PipelineStage:
    """流水线阶段基类.

    每个子类必须定义:
      - name, beat: 标识
      - description: 这个节拍做什么 (一行)
      - input_shape, output_shape: activation shape
      - weight_source: 'sram' | 'hbm' | 'ethernet' | 'none'
      - precision: 'fp4×fp8' | 'fp8×fp8' | 'none'
      - dsp_macs_million: DSP 消耗

    子类实现:
      - _compute_latency(ctx) → float
      - _transform(ctx) → StageContext
    """

    name: str = ""
    beat: int = -1
    description: str = ""
    input_shape: tuple = ()
    output_shape: tuple = ()
    weight_source: str = "none"   # 'sram' | 'hbm' | 'ethernet' | 'none'
    precision: str = "none"       # 'fp4×fp8' | 'fp8×fp8' | 'none'
    dsp_macs_million: float = 0.0

    def latency_us(self, ctx: StageContext) -> float:
        """计算本阶段的延迟 (μs)."""
        return self._compute_latency(ctx)

    def forward(self, ctx: StageContext) -> StageContext:
        """执行本阶段, 返回下一阶段的上下文."""
        return self._transform(ctx)

    def _compute_latency(self, ctx: StageContext) -> float:
        raise NotImplementedError

    def _transform(self, ctx: StageContext) -> StageContext:
        """默认: 透传上下文."""
        return ctx.clone()

    @property
    def summary(self) -> str:
        """单行摘要: [Beat X] Name — description"""
        return f"[Beat {self.beat}] {self.name:<20s} — {self.description}"

    def __repr__(self):
        return self.summary
