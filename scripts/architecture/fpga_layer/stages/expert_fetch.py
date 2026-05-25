"""
节拍 5 — Expert 权重获取.

目的: 根据 Router 选出的 expert IDs, 获取对应权重.
      命中 → HBM 本地读; Miss → 以太网 RDMA 从其他卡拉.

计算步骤:
  1. 查 HBM: 每个 expert_id 是否在本地
  2. 命中: HBM 顺序 burst 读, 每个 expert ~33 MB @ 800 GB/s → 41.7 μs
  3. Miss:  以太网 RDMA 拉取 expert 结果 (不是权重, 是 activation)
            expert 结果 = [batch, 7168] FP8
            100GbE RDMA: ~3μs + 7KB / 10.6 GB/s ≈ 3.7 μs

关键区分:
  - 拉的是 "expert 计算结果" 而不是 "expert 权重"
  - 远端卡执行 routed expert 计算, 结果通过以太网送回来
  - 本地卡负责自己的 expert 计算 (下一拍), 同时等待远端结果

物理资源: HBM(命中) / 以太网 RDMA(miss)
DSP: 无 (纯数据传输)
"""

from .base import PipelineStage, StageContext
from ... import config
from ..phys.hbm import HBMController
from ..phys.ethernet import EthernetMAC


class ExpertFetchStage(PipelineStage):
    name: str = "expert_fetch"
    beat: int = 5
    description: str = "Fetch expert weights: HBM(hit) / 100GbE(miss) RDMA"
    input_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    output_shape: tuple = (1, config.MODEL_HIDDEN_SIZE)
    weight_source: str = "hbm"
    precision: str = "fp4×fp8"
    dsp_macs_million: float = 0.0

    def __init__(self, hbm: HBMController, eth: EthernetMAC, card_id: int = 0):
        self.hbm = hbm
        self.eth = eth
        self.card_id = card_id

    def _compute_latency(self, ctx: StageContext) -> float:
        """延迟 = max(HBM 读, 以太网拉取) — 两者可并行."""
        hbm_lat = 0.0
        eth_lat = 0.0

        for eid in ctx.top_k_experts:
            if self.hbm.is_local(eid):
                hbm_lat += self.hbm._read_mb(config.WEIGHT_EXPERT_MB)
            else:
                # 远端 expert 结果: [batch, 7168] FP8 = batch × 7KB
                result_bytes = ctx.batch_size * config.MODEL_HIDDEN_SIZE
                # 找持有该 expert 的卡 (模拟: 用 eid 计算)
                owner = eid % config.HW_FPGA_CHIP_COUNT
                eth_lat += self.eth.p2p_fetch(owner, self.card_id, result_bytes,
                                               same_server=False)

        # HBM 和 以太网可并行
        return max(hbm_lat, eth_lat)

    def _transform(self, ctx: StageContext) -> StageContext:
        ctx = ctx.clone()

        hits = []
        misses = []
        hbm_us = 0.0
        eth_us = 0.0

        for eid in ctx.top_k_experts:
            if self.hbm.is_local(eid):
                hits.append(eid)
                hbm_us += self.hbm.read(eid)
            else:
                misses.append(eid)
                result_bytes = ctx.batch_size * config.MODEL_HIDDEN_SIZE
                owner = eid % config.HW_FPGA_CHIP_COUNT
                eth_us += self.eth.p2p_fetch(owner, self.card_id, result_bytes,
                                              same_server=False)

        ctx.hit_experts = hits
        ctx.miss_experts = misses
        ctx.hbm_fetch_us = hbm_us
        ctx.ethernet_fetch_us = eth_us

        return ctx
