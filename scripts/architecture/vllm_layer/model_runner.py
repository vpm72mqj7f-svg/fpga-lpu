"""
Model Runner — 模型图执行器, 将每个 layer 的计算分派到 FPGA.

这是 vLLM 和 FPGA 之间的桥梁:
  vLLM (host)                      FPGA (device)
  ─────────────────────────────────────────────────
  ModelRunner.execute_layer()  →   FPGA Runtime.execute_layer()
       ↓                                ↓
  torch.nn.Module 图结构          DSP kernels + HBM + SRAM

核心变化: 每一层的 torch.nn.Linear 被替换为 FPGA PCIe 调用.
"""

from dataclasses import dataclass, field
import random
from .. import config
from .scheduler import ScheduledBatch, StepType


@dataclass
class LayerDispatch:
    """一层分派到 FPGA 的完整记录."""
    layer_idx: int
    # 输入
    batch_size: int
    seq_len: int
    expert_ids: list[int]
    # FPGA 返回的延迟 (μs)
    fpga_time_us: float = 0.0
    pcie_time_us: float = 0.0
    # 输出
    hbm_hit: bool = True


class ModelRunner:
    """vLLM Model Runner — 适配 FPGA 后端.

    职责:
      1. 维护 61 层模型图结构 (config, 不包含实际权重)
      2. 对每个 layer 生成 FPGA 调用
      3. 聚合层间结果
      4. 上报延迟统计

    与原始 vLLM 的关键区别:
      - 没有 torch.nn.Module (权重在 FPGA 侧)
      - forward() 变成 PCIe 调用序列
      - KV cache 管理通过 KVManager
    """

    def __init__(self):
        self.num_layers = config.MODEL_NUM_LAYERS
        self.hidden_size = config.MODEL_HIDDEN_SIZE
        self.num_experts = config.MODEL_NUM_EXPERTS
        self.top_k = config.MODEL_TOP_K

        # 运行统计
        self.total_layers_executed = 0
        self.total_fpga_time_us = 0.0
        self.total_pcie_time_us = 0.0
        self.layer_records: list[LayerDispatch] = []

    def _simulate_router(self, batch_size: int) -> list[int]:
        """模拟 MoE Router 输出: 返回本 batch 需要的 expert IDs (去重).

        真实系统: Router 在 SRAM 中 (384 × 7168 权重), FPGA DSP 直接算.
        此处简化: 基于二项分布随机采样.
        """
        p_per_expert = config.MODEL_EXPERTS_PER_FPGA / config.MODEL_NUM_EXPERTS

        all_experts = set()
        for _ in range(batch_size):
            for _ in range(self.top_k):
                if random.random() < p_per_expert:
                    # 本地命中: 从该卡 13 个 expert 中选一个
                    eid = random.randint(0, config.MODEL_EXPERTS_PER_FPGA - 1)
                    all_experts.add(eid)

        return sorted(all_experts)

    def execute_layer(
        self,
        layer_idx: int,
        batch: ScheduledBatch,
        fpga_runtime,  # FPGARuntime instance
    ) -> LayerDispatch:
        """执行一层. 调用 fpga_runtime.execute_layer().

        batch 中包含当前 step 的所有请求:
          - prefill: batch_size = num_tokens (prompt tokens 总数)
          - decode:  batch_size = num_seqs (每个序列 1 token)
        """
        batch_size = batch.num_tokens
        seq_len = 0  # 由 KV cache 管理, FPGA 侧需要但此处简化

        # Router 模拟 → expert IDs
        expert_ids = self._simulate_router(batch_size)

        # 输入 activation 大小 (FP8: 1 byte/element)
        input_bytes = batch_size * self.hidden_size

        # 调用 FPGA
        exec_rec = fpga_runtime.execute_layer(
            layer_idx=layer_idx,
            batch_size=batch_size,
            seq_len=seq_len,
            expert_ids=expert_ids,
            input_bytes=input_bytes,
        )

        dispatch = LayerDispatch(
            layer_idx=layer_idx,
            batch_size=batch_size,
            seq_len=seq_len,
            expert_ids=expert_ids,
            fpga_time_us=exec_rec.total_us,
            pcie_time_us=exec_rec.pcie_xfer_us,
            hbm_hit=(exec_rec.hbm_read_us < 10.0),  # < 10 μs 视为全部命中
        )

        self.total_layers_executed += 1
        self.total_fpga_time_us += exec_rec.total_us
        self.total_pcie_time_us += exec_rec.pcie_xfer_us
        self.layer_records.append(dispatch)

        return dispatch

    def execute_model(
        self,
        batch: ScheduledBatch,
        fpga_runtime,
    ) -> list[LayerDispatch]:
        """执行全部 61 层. 返回每层分派记录."""
        records = []
        for layer_idx in range(self.num_layers):
            rec = self.execute_layer(layer_idx, batch, fpga_runtime)
            records.append(rec)
        return records

    @property
    def stats(self) -> dict:
        n = max(self.total_layers_executed, 1)
        return {
            'total_layers_executed': self.total_layers_executed,
            'total_fpga_time_us': self.total_fpga_time_us,
            'total_pcie_time_us': self.total_pcie_time_us,
            'avg_fpga_time_per_layer_us': self.total_fpga_time_us / n,
            'avg_pcie_time_per_layer_us': self.total_pcie_time_us / n,
            'hbm_hit_ratio': sum(1 for r in self.layer_records if r.hbm_hit) / max(len(self.layer_records), 1),
        }

    def reset(self):
        self.total_layers_executed = 0
        self.total_fpga_time_us = 0.0
        self.total_pcie_time_us = 0.0
        self.layer_records.clear()
