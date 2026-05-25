"""
TP Group — Tensor Parallel 组协调器.

管理 group_size 张 FPGA 卡协同处理同一层.
所有卡间通信统一走 100GbE 以太网 (通过交换机).

职责:
  1. 接收 vLLM 下发的 expert routing → 分配 expert 到持有卡
  2. 各卡并行执行流水线 (pipeline.execute_layer)
  3. AllReduce 汇总 TP 组内各卡部分结果 (reduce-scatter + all-gather)
  4. 跨组 expert 结果通过以太网 RDMA 共享

通信全部走 EthernetMAC, 无 SerDes:
  - AllReduce: Ring over Ethernet, 2×(N-1) steps
  - Expert fetch: P2P RDMA, 1 hop (同交换机) or 3 hops (跨交换机)
"""

from dataclasses import dataclass, field
from .. import config
from .phys.ethernet import EthernetMAC
from .pipeline import Pipeline


@dataclass
class TPGroupExecution:
    """一个 TP group 执行一层的完整记录."""
    layer_idx: int
    batch_size: int
    seq_len: int
    group_size: int
    # 最慢卡的节拍分解
    beat_breakdown: dict[str, float] = field(default_factory=dict)
    # 每卡总延迟
    per_card_total_us: list[float] = field(default_factory=list)
    # 通信延迟
    allreduce_us: float = 0.0
    cross_group_expert_us: float = 0.0
    # 总延迟
    total_us: float = 0.0


class TPGroup:
    """Tensor Parallel Group — 以太网互联."""

    def __init__(self, group_id: int, pipelines: list[Pipeline]):
        self.group_id = group_id
        self.pipelines = pipelines  # 每个 Pipeline = 一张卡的完整流水线
        self.group_size = len(pipelines)
        self.eth = EthernetMAC(num_cards=self.group_size)

        # expert → pipeline index (哪个 expert 在哪个 Pipeline 的 HBM 中)
        self._expert_to_card: dict[int, int] = {}
        for i, pl in enumerate(self.pipelines):
            for eid in pl.hbm._loaded:
                self._expert_to_card[eid] = i

        self.executions: list[TPGroupExecution] = []

    @property
    def cards(self):
        return self.pipelines

    def execute_layer(
        self,
        layer_idx: int,
        batch_size: int,
        seq_len: int,
        expert_ids: list[int],
    ) -> TPGroupExecution:
        """TP 组内执行一层.

        1. Expert → Card 分配
        2. 各卡并行跑 pipeline.execute_layer()
        3. 以太网 AllReduce 汇总
        4. 跨组 expert 结果 P2P 拉取
        """
        exec_rec = TPGroupExecution(
            layer_idx=layer_idx, batch_size=batch_size,
            seq_len=seq_len, group_size=self.group_size)

        # ── Step 1: 分配 expert 到卡 ──
        card_experts: dict[int, list[int]] = {i: [] for i in range(self.group_size)}
        for eid in expert_ids:
            card_idx = self._expert_to_card.get(eid, 0)
            card_experts[card_idx].append(eid)

        # ── Step 2: 各卡并行执行 ──
        per_card_total = []
        per_card_beats = []

        for i, pl in enumerate(self.pipelines):
            # 将 expert fetch 和 routed expert 的 expert 集合设为分配到的
            local_experts = card_experts[i]
            # 执行流水线 (Router 已提前填好 expert_ids)
            rec = pl.execute_layer(layer_idx, batch_size, seq_len)
            # 补充 expert 信息
            rec.hit_count = sum(1 for e in local_experts if pl.hbm.is_local(e))
            rec.miss_count = len(local_experts) - rec.hit_count
            per_card_total.append(rec.total_us)
            per_card_beats.append(rec.beat_latencies)

        exec_rec.per_card_total_us = per_card_total

        # 取最慢卡的节拍分解
        slowest = per_card_total.index(max(per_card_total))
        exec_rec.beat_breakdown = per_card_beats[slowest]

        # ── Step 3: 以太网 AllReduce ──
        reduce_bytes = batch_size * config.MODEL_HIDDEN_SIZE * 1  # FP8
        exec_rec.allreduce_us = self.eth.allreduce(reduce_bytes, self.group_size,
                                                     same_server=True)

        # ── Step 4: 跨组 expert 结果 P2P ──
        local_experts_in_group = self._expert_to_card.keys()
        miss_count = sum(1 for e in expert_ids if e not in local_experts_in_group)
        if miss_count > 0:
            share_bytes = miss_count * batch_size * config.MODEL_HIDDEN_SIZE
            exec_rec.cross_group_expert_us = self.eth.p2p_fetch(
                -1, 0, share_bytes, same_server=False)

        # ── 总延迟 = 最慢卡计算 + AllReduce + 跨组拉取 ──
        exec_rec.total_us = max(per_card_total) + exec_rec.allreduce_us + exec_rec.cross_group_expert_us

        self.executions.append(exec_rec)
        return exec_rec

    def execute_full_model(
        self, batch_size: int, seq_len: int,
        expert_routing: list[list[int]],
    ) -> tuple[float, list[TPGroupExecution]]:
        """执行全部 61 层."""
        records = []
        total = 0.0

        for layer_idx in range(config.MODEL_NUM_LAYERS):
            eids = expert_routing[layer_idx] if layer_idx < len(expert_routing) else []
            rec = self.execute_layer(layer_idx, batch_size, seq_len, eids)
            total += rec.total_us
            records.append(rec)

        return total, records

    @property
    def stats(self) -> dict:
        n = max(len(self.executions), 1)
        avg_comp = sum(max(e.per_card_total_us) for e in self.executions) / n
        avg_ar = sum(e.allreduce_us for e in self.executions) / n
        avg_cross = sum(e.cross_group_expert_us for e in self.executions) / n
        avg_total = sum(e.total_us for e in self.executions) / n
        return {
            'group_id': self.group_id,
            'group_size': self.group_size,
            'layers': n * config.MODEL_NUM_LAYERS if n > 0 else 0,
            'avg_comp_us': avg_comp,
            'avg_allreduce_us': avg_ar,
            'avg_cross_group_us': avg_cross,
            'avg_total_layer_us': avg_total,
            'eth_stats': self.eth.stats,
        }

    def reset(self):
        self.eth.reset()
        self.executions.clear()
        for pl in self.pipelines:
            pl.reset()
