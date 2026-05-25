"""
fpga_arch/chip.py — FPGA chip model with resource tracking.

Extracted from fpga_4chip_pipeline.py:235-289 with enhancements:
  - SRAMBank / HBMBank / DSPArray dataclasses for fine-grained tracking
  - KV block allocation / free (PagedAttention-compatible)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Tuple
import math

from .config import (
    DSP_COUNT, DSP_FREQ_MHZ, DSP_MAC_PER_CYCLE, DSP_TMACS,
    HBM_SIZE_GB, HBM_BW_GBPS, HBM_BW_EFF,
    SRAM_M20K_MB, SRAM_MLAB_MB, SRAM_TOTAL_MB,
    CHIPS_PER_CARD, EXPERTS_PER_CHIP, NUM_EXPERTS,
    C2C_MAX_PAYLOAD_B, C2C_LINK_BW_GBPS, C2C_FRAME_OVERHEAD_B, C2C_HOP_LATENCY_NS,
    KV_LORA_RANK, QK_ROPE_HEAD_DIM, MLA_KV_BYTES,
    DETERMINISTIC_MB_PER_LAYER, WEIGHT_GB_PER_CHIP,
)


@dataclass
class SRAMBank:
    """SRAM resource tracker for a single chip.

    Three consumers compete for SRAM:
      - deterministic: double-buffered attn + shared expert + router + norms (~21 MB)
      - kv_scratch: KV cache working set during attention
      - expert_buffer: streaming buffer for expert weights from HBM
    """
    total_mb: float = SRAM_TOTAL_MB
    deterministic_mb: float = 0.0
    kv_scratch_mb: float = 0.0
    expert_buffer_mb: float = 0.0

    @property
    def used_mb(self) -> float:
        return self.deterministic_mb + self.kv_scratch_mb + self.expert_buffer_mb

    @property
    def free_mb(self) -> float:
        return self.total_mb - self.used_mb

    @property
    def utilization_pct(self) -> float:
        return self.used_mb / self.total_mb * 100 if self.total_mb > 0 else 0


@dataclass
class HBMBank:
    """HBM resource tracker for a single chip.

    Three consumers:
      - weight_storage: fp4 expert + attention + router weights (~0.7 GB)
      - kv_cache: per-token KV cache blocks (PagedAttention)
      - misc: residuals, router tables
    """
    total_gb: float = HBM_SIZE_GB
    weight_storage_gb: float = 0.0
    kv_cache_gb: float = 0.0
    misc_gb: float = 0.0

    @property
    def used_gb(self) -> float:
        return self.weight_storage_gb + self.kv_cache_gb + self.misc_gb

    @property
    def free_gb(self) -> float:
        return self.total_gb - self.used_gb

    @property
    def utilization_pct(self) -> float:
        return self.used_gb / self.total_gb * 100 if self.total_gb > 0 else 0

    def read_time_us(self, mb: float) -> float:
        """HBM read latency for given MB at effective bandwidth."""
        if mb <= 0:
            return 0.0
        return mb / (HBM_BW_GBPS * HBM_BW_EFF / 1024)


@dataclass
class DSPArray:
    """DSP compute resource tracker."""
    num_dsp: int = DSP_COUNT
    freq_mhz: float = DSP_FREQ_MHZ
    mac_per_cycle: int = DSP_MAC_PER_CYCLE
    busy_us: float = 0.0
    total_us: float = 0.0
    total_macs: float = 0.0

    @property
    def tmacs(self) -> float:
        return self.num_dsp * self.freq_mhz * self.mac_per_cycle / 1e6

    @property
    def utilization_pct(self) -> float:
        return self.busy_us / self.total_us * 100 if self.total_us > 0 else 0

    def compute_time_us(self, macs: float) -> float:
        """DSP time in microseconds for given MAC count."""
        return macs / (self.tmacs * 1e12) * 1e6

    def record_compute(self, macs: float, wall_us: float):
        """Record a DSP compute operation."""
        dsp_time = self.compute_time_us(macs)
        self.busy_us += dsp_time
        self.total_us += wall_us
        self.total_macs += macs


class FPGAChip:
    """Single AGM 039-F FPGA chip within a 4-chip card.

    Extracted from fpga_4chip_pipeline.py:235-289 with resource tracking
    enhancements for KV cache management and batch-aware compute.
    """

    def __init__(self, chip_id: int, card_id: int):
        self.chip_id = chip_id          # 0-3 within card
        self.card_id = card_id          # 0-7
        self.global_id = card_id * CHIPS_PER_CARD + chip_id  # 0-31

        # Assigned layers
        self.assigned_layers: List[int] = []

        # Assigned experts (contiguous range)
        start = self.global_id * EXPERTS_PER_CHIP
        self.assigned_experts: List[int] = list(range(start, start + EXPERTS_PER_CHIP))

        # Resource banks
        self.sram = SRAMBank()
        self.hbm = HBMBank()
        self.dsp = DSPArray()

        # C2C neighbors on Ring A (set by cluster assembly)
        self.ring_a_prev: Optional[int] = None   # global chip id
        self.ring_a_next: Optional[int] = None

        # PCIe master for the card (chip 0 of each card)
        self.is_pcie_master = (chip_id == 0)

        # Per-layer weight cache state
        self.sram_cached_weights: Set[str] = set()

        # KV cache block tracking (PagedAttention)
        # Each block: 16 tokens × MLA_KV_BYTES bytes × 2 layers (K+V)
        self._kv_blocks: Dict[int, 'KVBlock'] = {}
        self._kv_block_usage: Dict[int, int] = {}  # block_id -> ref_count
        self._kv_block_lru: List[int] = []          # LRU order of block_ids

    def __repr__(self):
        return f"FPGAChip(c{self.card_id}.{self.chip_id}, gid={self.global_id}, layers={self.assigned_layers})"

    def assign_layers(self, layers: List[int]):
        self.assigned_layers = sorted(layers)

    # ── Resource setup ──

    def place_weights(self, expert_weight_mb: float, attn_weight_mb: float, router_mb: float):
        """Initialize HBM/SRAM weight placement."""
        self.hbm.weight_storage_gb = (expert_weight_mb + attn_weight_mb + router_mb) / 1024
        self.sram.deterministic_mb = DETERMINISTIC_MB_PER_LAYER

    # ── KV cache block management ──

    def allocate_kv_blocks(self, num_blocks: int, block_tokens: int = 16) -> List[int]:
        """Allocate KV cache blocks. Returns list of block IDs or raises if OOM."""
        bytes_per_block = block_tokens * MLA_KV_BYTES * 2  # K + V
        gb_per_block = bytes_per_block / (1024**3)

        allocated = []
        for _ in range(num_blocks):
            if self.hbm.free_gb < gb_per_block:
                # Try LRU eviction
                if not self._try_evict_lru(gb_per_block):
                    raise RuntimeError(
                        f"KV cache OOM on chip {self.global_id}: "
                        f"{self.hbm.free_gb:.3f} GB free, need {gb_per_block:.6f} GB/block"
                    )
            block_id = len(self._kv_blocks)
            self._kv_blocks[block_id] = KVBlock(block_id, block_tokens)
            self._kv_block_usage[block_id] = 1
            self._kv_block_lru.append(block_id)
            self.hbm.kv_cache_gb += gb_per_block
            allocated.append(block_id)
        return allocated

    def free_kv_blocks(self, block_ids: List[int]):
        """Release KV cache blocks."""
        bytes_per_block = next(iter(self._kv_blocks.values())).num_tokens * MLA_KV_BYTES * 2
        gb_per_block = bytes_per_block / (1024**3)
        for bid in block_ids:
            if bid in self._kv_blocks:
                del self._kv_blocks[bid]
                self._kv_block_usage.pop(bid, None)
                if bid in self._kv_block_lru:
                    self._kv_block_lru.remove(bid)
                self.hbm.kv_cache_gb = max(0, self.hbm.kv_cache_gb - gb_per_block)

    def _try_evict_lru(self, needed_gb: float) -> bool:
        """Evict least-recently-used blocks until needed_gb is available."""
        bytes_per_block = 16 * MLA_KV_BYTES * 2
        gb_per_block = bytes_per_block / (1024**3)
        freed = 0.0
        evicted = []
        for bid in list(self._kv_block_lru):
            if self._kv_block_usage.get(bid, 0) == 0:
                evicted.append(bid)
                freed += gb_per_block
                if freed >= needed_gb:
                    break
        if freed >= needed_gb:
            self.free_kv_blocks(evicted)
            return True
        return False

    def access_kv_block(self, block_id: int):
        """Mark a KV block as recently used (move to end of LRU)."""
        if block_id in self._kv_block_lru:
            self._kv_block_lru.remove(block_id)
            self._kv_block_lru.append(block_id)

    @property
    def kv_cache_used_gb(self) -> float:
        return self.hbm.kv_cache_gb

    @property
    def kv_cache_used_blocks(self) -> int:
        return len(self._kv_blocks)


@dataclass
class KVBlock:
    """A single KV cache block (16 tokens)."""
    block_id: int
    num_tokens: int = 16
