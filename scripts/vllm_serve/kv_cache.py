"""
vllm_serve/kv_cache.py — PagedAttention KV cache block manager.

Block-based allocation with LRU eviction. Tracks per-chip HBM pressure.
Each block: BLOCK_SIZE tokens × KV_BYTES_PER_TOKEN bytes × 2 (K+V).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

from .config import (
    BLOCK_SIZE, MAX_NUM_BLOCKS, MAX_SEQ_LEN,
    KV_BLOCK_TOKENS, KV_BYTES_PER_TOKEN, KV_GB_PER_BLOCK,
    KV_MAX_GB_PER_SEQ, KV_BLOCKS_PER_CHIP,
)


@dataclass
class KVBlock:
    """A single KV cache block."""
    block_id: int
    chip_id: int          # which chip owns this block
    num_tokens: int = KV_BLOCK_TOKENS
    ref_count: int = 0    # number of active sequences referencing this block
    last_access_us: float = 0.0


class KVCacheManager:
    """Manages KV cache blocks across multiple FPGA chips.

    Implements PagedAttention-style allocation:
      - Blocks are allocated per-chip (KV cache sharded by layer)
      - LRU eviction when HBM is full
      - Prefill: allocate all prompt blocks at once
      - Decode: allocate one block per step per sequence
    """

    def __init__(self, num_chips: int = 32, max_blocks_per_chip: int = KV_BLOCKS_PER_CHIP):
        self.num_chips = num_chips
        self.max_blocks_per_chip = max_blocks_per_chip

        # All blocks across all chips
        self._blocks: Dict[int, KVBlock] = {}
        self._next_block_id = 0

        # Free blocks per chip
        self._free_blocks: Dict[int, List[int]] = defaultdict(list)

        # Blocks per chip (allocated)
        self._chip_blocks: Dict[int, Set[int]] = defaultdict(set)

        # Per-sequence block lists
        self._seq_blocks: Dict[int, List[int]] = {}  # request_id -> [block_ids]

        # LRU tracking per chip
        self._lru: Dict[int, List[int]] = defaultdict(list)

        # Initialize blocks
        for chip_id in range(num_chips):
            for _ in range(max_blocks_per_chip):
                bid = self._next_block_id
                self._next_block_id += 1
                self._blocks[bid] = KVBlock(block_id=bid, chip_id=chip_id)
                self._free_blocks[chip_id].append(bid)

    # ── Allocation ──

    def allocate_prefill(self, request_id: int, prompt_len: int,
                         chip_ids: List[int],
                         current_time_us: float) -> List[int]:
        """Allocate KV blocks for prefill on specified chips.

        KV blocks are distributed across chips proportionally.
        Total blocks = prompt_len / BLOCK_SIZE, split across chip_ids.
        """
        total_blocks = max(1, prompt_len // KV_BLOCK_TOKENS)
        blocks_per_chip = max(1, total_blocks // max(1, len(chip_ids)))
        # Ensure total allocation doesn't exceed needed blocks
        actual_total = blocks_per_chip * len(chip_ids)

        all_blocks = []
        for chip_id in chip_ids:
            chip_blocks = []
            for _ in range(blocks_per_chip):
                bid = self._allocate_block(chip_id, current_time_us)
                if bid is None:
                    # Rollback
                    self._free_blocks_chip(chip_id, chip_blocks)
                    for prev_chip, prev_blocks in self._snapshot_rollback(all_blocks, chip_ids):
                        self._free_blocks_chip(prev_chip, prev_blocks)
                    raise RuntimeError(
                        f"KV cache OOM on chip {chip_id}: "
                        f"need {blocks_per_chip} blocks, {len(self._free_blocks[chip_id])} free"
                    )
                chip_blocks.append(bid)
            all_blocks.extend(chip_blocks)

        self._seq_blocks[request_id] = all_blocks
        return all_blocks

    def allocate_decode(self, request_id: int, decode_step: int,
                        chip_ids: List[int],
                        current_time_us: float) -> List[int]:
        """Allocate KV blocks for decode step.

        Only allocates a new block when the current block is full
        (every KV_BLOCK_TOKENS steps). Returns empty list if no allocation needed.
        """
        # Only need a new block every KV_BLOCK_TOKENS decode steps
        if decode_step % KV_BLOCK_TOKENS != 0:
            return []

        new_blocks = []
        for chip_id in chip_ids:
            bid = self._allocate_block(chip_id, current_time_us)
            if bid is None:
                evicted = self._evict_lru(chip_id, 1, current_time_us)
                if evicted:
                    bid = self._allocate_block(chip_id, current_time_us)
                if bid is None:
                    raise RuntimeError(f"KV cache OOM on chip {chip_id} during decode")
            new_blocks.append(bid)

        if request_id in self._seq_blocks:
            self._seq_blocks[request_id].extend(new_blocks)
        else:
            self._seq_blocks[request_id] = new_blocks
        return new_blocks

    def _allocate_block(self, chip_id: int, current_time_us: float) -> Optional[int]:
        """Allocate a single block from free pool."""
        free = self._free_blocks.get(chip_id, [])
        if not free:
            return None
        bid = free.pop()
        self._chip_blocks[chip_id].add(bid)
        self._blocks[bid].ref_count += 1
        self._blocks[bid].last_access_us = current_time_us
        # Add to LRU
        self._lru[chip_id].append(bid)
        return bid

    def _free_blocks_chip(self, chip_id: int, block_ids: List[int]):
        """Return blocks to free pool."""
        for bid in block_ids:
            if bid in self._chip_blocks.get(chip_id, set()):
                self._chip_blocks[chip_id].discard(bid)
                self._blocks[bid].ref_count = max(0, self._blocks[bid].ref_count - 1)
                self._free_blocks[chip_id].append(bid)
                if bid in self._lru[chip_id]:
                    self._lru[chip_id].remove(bid)

    def _snapshot_rollback(self, all_blocks: List[int], chip_ids: List[int]):
        """Group blocks by chip for rollback."""
        groups = defaultdict(list)
        # Naive grouping: iterate in order
        for i, bid in enumerate(all_blocks):
            chip_id = self._blocks[bid].chip_id
            groups[chip_id].append(bid)
        return [(cid, blks) for cid, blks in groups.items()]

    # ── Free ──

    def free_request(self, request_id: int):
        """Release all blocks for a finished request."""
        if request_id not in self._seq_blocks:
            return
        for bid in self._seq_blocks[request_id]:
            chip_id = self._blocks[bid].chip_id
            self._free_blocks_chip(chip_id, [bid])
        del self._seq_blocks[request_id]

    # ── LRU eviction ──

    def _evict_lru(self, chip_id: int, num_blocks: int,
                   current_time_us: float) -> int:
        """Evict least-recently-used blocks. Returns number actually evicted."""
        lru_list = self._lru.get(chip_id, [])
        evicted = 0
        for bid in list(lru_list):
            if evicted >= num_blocks:
                break
            if self._blocks[bid].ref_count <= 0:
                self._free_blocks_chip(chip_id, [bid])
                evicted += 1
        return evicted

    # ── Access tracking ──

    def access_block(self, block_id: int, current_time_us: float):
        """Mark block as recently accessed (touch LRU)."""
        if block_id in self._blocks:
            self._blocks[block_id].last_access_us = current_time_us
            chip_id = self._blocks[block_id].chip_id
            if block_id in self._lru[chip_id]:
                self._lru[chip_id].remove(block_id)
                self._lru[chip_id].append(block_id)

    # ── Stats ──

    @property
    def total_blocks_allocated(self) -> int:
        return sum(len(v) for v in self._chip_blocks.values())

    @property
    def total_blocks_free(self) -> int:
        return sum(len(v) for v in self._free_blocks.values())

    @property
    def utilization_pct(self) -> float:
        total = self.total_blocks_allocated + self.total_blocks_free
        return self.total_blocks_allocated / max(total, 1) * 100

    def chip_usage_gb(self, chip_id: int) -> float:
        n_blocks = len(self._chip_blocks.get(chip_id, set()))
        return n_blocks * KV_GB_PER_BLOCK

    def stats_summary(self) -> str:
        lines = [
            f"KV Cache: {self.total_blocks_allocated}/{self.total_blocks_allocated + self.total_blocks_free} blocks allocated",
            f"  Utilization: {self.utilization_pct:.1f}%",
        ]
        for chip_id in sorted(self._chip_blocks.keys())[:8]:
            n = len(self._chip_blocks[chip_id])
            free = len(self._free_blocks.get(chip_id, []))
            gb = self.chip_usage_gb(chip_id)
            lines.append(f"  Chip {chip_id:02d}: {n} used, {free} free, {gb:.3f} GB")
        return "\n".join(lines)
