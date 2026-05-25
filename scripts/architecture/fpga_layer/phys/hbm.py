"""
HBM2e 控制器 — 32 GB, 存储全部 Routed Expert 权重.

每卡加载 13 个 expert (384 / 30).
Expert 权重 (fp4): gate/up/down ≈ 33 MB / expert.
共 13 × 33 ≈ 429 MB, 占 HBM 1.3%.

剩余 ~31 GB 可用于:
  - KV cache 池 (~12.8 GB = 40%)
  - 更多 expert 副本 (如果不同层请求不同 expert)
  - 临时的 cross-group expert cache

访问模式: 顺序 burst 读 (expert 权重是一整块, 不需要随机寻址)
有效带宽: 920 × 0.87 = 800 GB/s

RTL 接口:
  - Expert ID → 地址译码 → HBM 控制器 → 顺序 burst → 权重 buffer
  - 支持单 expert 读和多 expert 流水读
"""

from ... import config


class HBMController:
    """HBM2e 控制器 — Expert 权重存储."""

    def __init__(self, card_id: int = 0):
        self.card_id = card_id
        self.total_gb = config.HW_FPGA_HBM_SIZE_GB

        # 有效带宽 = 理论 × 效率
        self.seq_bw_gbps = config.HW_FPGA_HBM_BW_GBPS * config.HW_FPGA_HBM_EFF  # 800 GB/s
        self._bw_bytes_per_us = self.seq_bw_gbps * 1e3  # GB/s → bytes/μs

        self._loaded: set[int] = set()
        self.expert_size_mb = config.WEIGHT_EXPERT_MB  # 33 MB

        # 统计
        self.hits = 0
        self.misses = 0
        self.total_bytes = 0
        self.total_time_us = 0.0

    def load(self, expert_ids: list[int]):
        """初始化: 从 flash/NIC 加载 expert 权重到 HBM."""
        for eid in expert_ids:
            self._loaded.add(eid)

    @property
    def loaded_experts(self) -> set[int]:
        return self._loaded

    def is_local(self, expert_id: int) -> bool:
        """Expert 权重是否在本卡 HBM 中."""
        return expert_id in self._loaded

    def read(self, expert_id: int) -> float:
        """读一个 expert 的权重. 返回延迟 μs.

        命中: 顺序 burst 读 33 MB @ 800 GB/s.
        Miss: 返回 0, 由上层 stages/expert_fetch 决定走以太网.
        """
        if expert_id not in self._loaded:
            self.misses += 1
            return 0.0  # 不在这卡, 不走 HBM, 由 expert_fetch stage 走以太网

        self.hits += 1
        lat = self._read_mb(self.expert_size_mb)
        return lat

    def read_multi(self, expert_ids: list[int]) -> tuple[float, list[int], list[int]]:
        """读多个 expert. 返回 (延迟, 命中列表, miss列表)."""
        hits = []
        misses = []
        total_lat = 0.0
        for eid in expert_ids:
            if eid in self._loaded:
                hits.append(eid)
                total_lat += self._read_mb(self.expert_size_mb)
            else:
                misses.append(eid)
        self.hits += len(hits)
        self.misses += len(misses)
        return total_lat, hits, misses

    def _read_mb(self, size_mb: float) -> float:
        """HBM 顺序 burst 读延迟."""
        size_bytes = size_mb * 1e6
        lat = size_bytes / self._bw_bytes_per_us
        self.total_bytes += size_bytes
        self.total_time_us += lat
        return lat

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 1.0

    @property
    def stats(self) -> dict:
        return {
            'card_id': self.card_id,
            'loaded_experts': len(self._loaded),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'total_read_time_us': self.total_time_us,
            'total_read_bytes': self.total_bytes,
        }

    def reset(self):
        self.hits = 0
        self.misses = 0
        self.total_bytes = 0
        self.total_time_us = 0.0
