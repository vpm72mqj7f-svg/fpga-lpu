"""
SRAM 缓存 — 片上 43 MB, 永久驻留确定性权重.

驻留内容 (~9.3 MB):
  - MLA Attention weights     6.2 MB
  - Shared Expert weights     2.7 MB (one expert, fp4)
  - Router weights            0.37 MB

访问延迟: ~0 (片上, 与 DSP 同频, 可忽略)
"""

from .. import config


class SRAMCache:
    """FPGA 片上 SRAM — 确定性权重永久缓存.

    确定性权重 = 所有请求都需要的权重, 不随 expert routing 变化.
    这些权重在初始化时一次性从 HBM 加载到 SRAM, 之后零延迟访问.
    """

    def __init__(self):
        self.total_mb = config.HW_FPGA_SRAM_SIZE_MB
        self.used_mb = config.WEIGHT_DETERMINISTIC_MB
        self.free_mb = self.total_mb - self.used_mb

        # 缓存状态
        self._weights_loaded = False
        self._load_time_us = 0.0

    def initialize(self, hbm_read_fn) -> float:
        """从 HBM 加载确定性权重到 SRAM (仅在启动时执行一次).

        hbm_read_fn: HBM 读取函数, 参数 (size_mb) → 延迟 μs
        """
        if self._weights_loaded:
            return 0.0

        load_time = hbm_read_fn(self.used_mb)
        self._weights_loaded = True
        self._load_time_us = load_time
        return load_time

    @property
    def is_ready(self) -> bool:
        return self._weights_loaded

    def contains(self, weight_name: str) -> bool:
        """检查某权重是否在 SRAM 中."""
        return weight_name in ('attention', 'shared_expert', 'router')

    def read_latency(self, size_mb: float = 0) -> float:
        """SRAM 读取延迟 (~0, 片上访问)."""
        return 0.0
