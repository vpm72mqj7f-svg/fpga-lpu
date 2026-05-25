"""
HBM 管理器 — 32 GB HBM2e, 存储全部 expert 权重.

核心功能:
  1. 追踪每个 expert 是否已在 HBM 中
  2. 模拟 HBM 顺序读取延迟 (800 GB/s 有效)
  3. 支持 expert 加载/驱逐

关键假设:
  - 确定性权重在 SRAM 中, 不经过 HBM
  - 每个 FPGA 卡加载 13 个 expert (384/30)
  - 被选中的 routed expert 从 HBM 按需读取
"""

from .. import config


class HBMManager:
    """FPGA HBM2e 管理器.

    每个 FPGA 卡持有一部分 expert 权重在 HBM 中.
    推理时根据 router 输出决定从 HBM 加载哪些 expert.
    """

    def __init__(self, card_id: int = 0):
        self.card_id = card_id
        self.total_gb = config.HW_FPGA_HBM_SIZE_GB
        self.seq_bw_gbps = config.HW_FPGA_HBM_BW_GBPS * config.HW_FPGA_HBM_EFF
        self._bw_bytes_per_us = self.seq_bw_gbps * 1e3  # GB/s → bytes/μs: ×10^9/10^6 = ×10^3

        # 该卡上驻留的 expert 列表 (初始化时从 flash/网络加载)
        self._loaded_experts: set[int] = set()
        self._expert_size_mb = config.WEIGHT_EXPERT_MB

        # 统计
        self.total_reads = 0
        self.total_read_bytes = 0
        self.total_read_time_us = 0.0
        self.hit_count = 0
        self.miss_count = 0

    def load_experts(self, expert_ids: list[int]) -> float:
        """初始化: 加载指定 expert 到 HBM (模拟从 flash/NIC 加载)."""
        for eid in expert_ids:
            self._loaded_experts.add(eid)
        return 0.0  # 初始加载时间不计入推理延迟

    def is_expert_loaded(self, expert_id: int) -> bool:
        return expert_id in self._loaded_experts

    def read_expert(self, expert_id: int) -> float:
        """从 HBM 读取一个 expert 权重. 返回延迟 μs.

        如果 expert 不在该卡上 → 返回 penalty (需要从其他卡取, 模拟 AllReduce).
        """
        if expert_id not in self._loaded_experts:
            self.miss_count += 1
            # Cache miss: 从其他卡拉取 (2× PCIe 延迟 + 网络)
            return 50.0  # 惩罚: ~50 μs

        self.hit_count += 1
        return self._read_mb(self._expert_size_mb + 0.37)  # + router table overhead

    def read_multi_experts(self, expert_ids: list[int]) -> float:
        """读取多个 expert. 返回总延迟 μs."""
        total = 0.0
        for eid in expert_ids:
            total += self.read_expert(eid)
        self.total_reads += len(expert_ids)
        return total

    def _read_mb(self, size_mb: float) -> float:
        """模拟从 HBM 读取 size_mb MB 的延迟 (μs).

        HBM 顺序读: 一次 burst 即可, 不需要 seek.
        """
        bytes_read = size_mb * 1e6
        lat_us = bytes_read / self._bw_bytes_per_us
        self.total_read_bytes += bytes_read
        self.total_read_time_us += lat_us
        return lat_us

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 1.0

    @property
    def stats(self) -> dict:
        return {
            'card_id': self.card_id,
            'loaded_experts': len(self._loaded_experts),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.hit_rate,
            'total_read_time_us': self.total_read_time_us,
            'avg_read_time_us': self.total_read_time_us / max(self.hit_count, 1),
        }
