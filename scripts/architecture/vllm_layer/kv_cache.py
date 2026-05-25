"""
KV Cache Manager — PagedAttention 风格的 KV 块管理器.

MLA 压缩优势: 每 token 只需 576 bytes (vs MHA 的 32 KB).
因此 32 GB HBM 可存 ~55M token 的 KV cache.

采用 Hybrid 方案: FPGA HBM 保存热 KV blocks, CPU RAM 保存冷 blocks.
swap_in/swap_out 接口与 vLLM 的 block_manager 一致.
"""

from dataclasses import dataclass, field
from .. import config


@dataclass
class KVBlock:
    """一个 KV cache block (16 tokens)."""
    block_id: int
    num_tokens: int = 0
    # 位置: 'fpga_hbm' | 'cpu_ram' | 'free'
    location: str = 'free'
    last_accessed: int = 0  # 用于 LRU 驱逐

    @property
    def size_bytes(self) -> int:
        return self.num_tokens * config.KV_BYTES_PER_TOKEN


class KVBlockTable:
    """每序列的 KV block 映射表.

    类似 vLLM 的 BlockTable: 每个 sequence 维护一个 block_id 列表.
    """

    def __init__(self, seq_id: int, max_blocks: int = 2048):
        self.seq_id = seq_id
        self.block_ids: list[int] = []   # 按 token 顺序排列
        self.max_blocks = max_blocks

    @property
    def num_tokens(self) -> int:
        return sum(1 for _ in self.block_ids)  # 每 block 最多 16 tokens
        # 简化: 当前 block 全部满 16 tokens

    def append_block(self, block_id: int):
        self.block_ids.append(block_id)

    def sliding_window_blocks(self, window_size: int) -> list[int]:
        """返回滑动窗口内的 block IDs."""
        n = (window_size + config.SW_BLOCK_SIZE - 1) // config.SW_BLOCK_SIZE
        return self.block_ids[-n:] if len(self.block_ids) > n else list(self.block_ids)


class KVManager:
    """全局 KV Cache 管理器 — PagedAttention 风格.

    支持:
      - FPGA HBM 作为热缓存 (容量有限)
      - CPU RAM 作为冷存储 (容量大)
      - swap_in / swap_out 迁移 blocks
    """

    def __init__(self, hbm_size_mb: float = None):
        if hbm_size_mb is None:
            hbm_size_mb = config.HW_FPGA_HBM_SIZE_GB * 1024 * 0.4  # 40% HBM 用于 KV
        # 预留给 KV cache 的 HBM 大小
        self.hbm_kv_bytes = int(hbm_size_mb * 1e6)
        self.cpu_kv_bytes = 256 * 1024**3  # 256 GB CPU RAM for KV

        self._total_blocks = self.hbm_kv_bytes // (config.SW_BLOCK_SIZE * config.KV_BYTES_PER_TOKEN)
        self._blocks: dict[int, KVBlock] = {}
        self._next_block_id = 0
        self._tables: dict[int, KVBlockTable] = {}
        self._step_counter = 0  # 用于 LRU

        # 统计
        self.swap_ins = 0
        self.swap_outs = 0

    def allocate_table(self, seq_id: int) -> KVBlockTable:
        """为新序列分配 block table."""
        table = KVBlockTable(seq_id)
        self._tables[seq_id] = table
        return table

    def allocate_block(self) -> int:
        """分配一个新 KV block. 返回 block_id."""
        bid = self._next_block_id
        self._next_block_id += 1
        self._blocks[bid] = KVBlock(block_id=bid, num_tokens=0, location='fpga_hbm')
        return bid

    def swap_in(self, block_ids: list[int]) -> float:
        """CPU RAM → FPGA HBM. 返回延迟 μs.

        PCIe 传输 + HBM 写入. 模拟 ~32 GB/s 有效带宽.
        """
        total_bytes = sum(
            self._blocks[bid].size_bytes
            for bid in block_ids
            if bid in self._blocks and self._blocks[bid].location == 'cpu_ram'
        )
        if total_bytes == 0:
            return 0.0
        # PCIe read from CPU: ~2 μs + size/32GBps
        lat_us = 2.0 + total_bytes / (config.HW_PCIE_BW_GBPS * 1e3 / 1e9)
        for bid in block_ids:
            if bid in self._blocks:
                self._blocks[bid].location = 'fpga_hbm'
        self.swap_ins += 1
        return lat_us

    def swap_out(self, block_ids: list[int]) -> float:
        """FPGA HBM → CPU RAM. 返回延迟 μs."""
        total_bytes = sum(
            self._blocks[bid].size_bytes
            for bid in block_ids
            if bid in self._blocks and self._blocks[bid].location == 'fpga_hbm'
        )
        if total_bytes == 0:
            return 0.0
        lat_us = 2.0 + total_bytes / (config.HW_PCIE_BW_GBPS * 1e3 / 1e9)
        for bid in block_ids:
            if bid in self._blocks:
                self._blocks[bid].location = 'cpu_ram'
        self.swap_outs += 1
        return lat_us

    def get_table(self, seq_id: int) -> KVBlockTable | None:
        return self._tables.get(seq_id)

    def step(self):
        self._step_counter += 1

    @property
    def stats(self) -> dict:
        hbm_blocks = sum(1 for b in self._blocks.values() if b.location == 'fpga_hbm')
        cpu_blocks = sum(1 for b in self._blocks.values() if b.location == 'cpu_ram')
        return {
            'total_blocks': len(self._blocks),
            'hbm_blocks': hbm_blocks,
            'cpu_blocks': cpu_blocks,
            'hbm_usage_mb': hbm_blocks * config.SW_BLOCK_SIZE * config.KV_BYTES_PER_TOKEN / 1e6,
            'swap_ins': self.swap_ins,
            'swap_outs': self.swap_outs,
            'num_sequences': len(self._tables),
        }
