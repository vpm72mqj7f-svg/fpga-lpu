"""
vllm_serve/weight_layout.py — Weight Layout Compiler (WLC) prototype.

Maps logical model weights (layers / experts / shared weights) to physical
FPGA chip HBM regions. This is a software-side compiler model, not RTL.

Supported placement features:
  - Baseline expert placement: 384 experts / 32 chips = 12 experts/chip
  - Hot Expert Replication: Zipf-popular experts replicated across cards
  - Pipeline Cloning: split 32 chips into N independent pipelines
  - HBM budget validation: weight + KV reservation must fit in 32 GB/chip
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict

from fpga_arch.config import (
    TOTAL_CHIPS, NUM_LAYERS, NUM_EXPERTS, EXPERT_TOTAL_MB,
    ATTN_TOTAL_MB_PER_LAYER, ROUTER_WEIGHT_MB, HBM_SIZE_GB,
)
from fpga_arch.expert_popularity import ExpertPopularity

if TYPE_CHECKING:
    import numpy as np


@dataclass
class HBMRegion:
    """One contiguous HBM allocation."""
    name: str
    start_mb: float
    size_mb: float
    kind: str
    metadata: Dict = field(default_factory=dict)

    @property
    def end_mb(self) -> float:
        return self.start_mb + self.size_mb


@dataclass
class ChipLayout:
    """All HBM allocations for a physical chip."""
    chip_id: int
    pipeline_id: int
    layers: List[int] = field(default_factory=list)
    experts: List[int] = field(default_factory=list)
    regions: List[HBMRegion] = field(default_factory=list)

    @property
    def used_mb(self) -> float:
        return sum(r.size_mb for r in self.regions)

    @property
    def used_gb(self) -> float:
        return self.used_mb / 1024

    @property
    def free_gb(self) -> float:
        return HBM_SIZE_GB - self.used_gb

    def add_region(self, name: str, size_mb: float, kind: str, metadata: Dict = None):
        start = self.regions[-1].end_mb if self.regions else 0.0
        self.regions.append(HBMRegion(name=name, start_mb=start, size_mb=size_mb,
                                      kind=kind, metadata=metadata or {}))


@dataclass
class LayoutReport:
    pipeline_clones: int
    replication: str
    chip_layouts: List[ChipLayout]
    expert_to_chips: Dict[int, List[int]]

    @property
    def max_used_gb(self) -> float:
        return max(c.used_gb for c in self.chip_layouts)

    @property
    def min_free_gb(self) -> float:
        return min(c.free_gb for c in self.chip_layouts)

    @property
    def total_weight_gb(self) -> float:
        return sum(c.used_gb for c in self.chip_layouts)

    def summary(self) -> str:
        counts = [len(c.experts) for c in self.chip_layouts]
        lines = [
            f"Weight Layout: clone={self.pipeline_clones}, replication={self.replication}",
            f"  chips: {len(self.chip_layouts)}, total allocated: {self.total_weight_gb:.1f} GB",
            f"  max/chip: {self.max_used_gb:.2f} GB, min free/chip: {self.min_free_gb:.2f} GB",
            f"  experts/chip: min={min(counts)}, avg={sum(counts)/len(counts):.1f}, max={max(counts)}",
        ]
        for c in self.chip_layouts[:8]:
            lines.append(
                f"  chip {c.chip_id:02d} P{c.pipeline_id}: "
                f"layers={c.layers}, experts={len(c.experts)}, used={c.used_gb:.2f} GB"
            )
        return "\n".join(lines)


class WeightLayoutCompiler:
    """Compile logical model placement into per-chip HBM maps.

    Supports optional weight preloading via the weight_preloader C library
    (Linux only, requires libaio). Use preload_weights() after compile()
    to load layer weights from SSD into host pinned memory for Phase 1
    startup preloading.
    """

    def __init__(self, pipeline_clones: int = 1, replication: str = "none",
                 zipf_alpha: float = 1.0, kv_reserve_gb: float = 22.0,
                 seed: int = 42):
        if pipeline_clones not in (1, 2, 4):
            raise ValueError("pipeline_clones must be 1, 2, or 4")
        if replication not in ("none", "hot"):
            raise ValueError("replication must be 'none' or 'hot'")
        self.pipeline_clones = pipeline_clones
        self.replication = replication
        self.zipf_alpha = zipf_alpha
        self.kv_reserve_gb = kv_reserve_gb
        self.seed = seed
        self._last_layout: Optional[LayoutReport] = None
        self._preloader_bridge = None
        self._preloader_weight_dir: str = ""

    def compile(self) -> LayoutReport:
        chips_per_pipeline = TOTAL_CHIPS // self.pipeline_clones
        chip_layouts = [ChipLayout(chip_id=i, pipeline_id=i // chips_per_pipeline)
                        for i in range(TOTAL_CHIPS)]

        self._assign_layers(chip_layouts, chips_per_pipeline)
        expert_to_chips = self._assign_experts(chip_layouts)
        self._allocate_hbm_regions(chip_layouts)
        self._validate(chip_layouts)

        report = LayoutReport(
            pipeline_clones=self.pipeline_clones,
            replication=self.replication,
            chip_layouts=chip_layouts,
            expert_to_chips=expert_to_chips,
        )
        self._last_layout = report
        return report

    def preload_weights(self, weight_dir: str = "",
                        layout: Optional[LayoutReport] = None,
                        num_preload: int = 4):
        """Preload layer weights from SSD into host pinned memory (Phase 1 startup).

        Uses the weight_preloader C library (Linux + libaio). On non-Linux
        platforms or if the shared library is not built, raises
        NotImplementedError with a build instruction.

        This is the integration point between the weight layout compiler
        and the C weight preloader bridge. After compiling the chip layout,
        call this to actually load the weight data into memory for DMA
        transfer to FPGA HBM.

        Args:
            weight_dir: path to SSD weight files (e.g. "/data/weights/deepseek_v4")
            layout: LayoutReport from compile(). Uses last compiled layout if None.
            num_preload: max layers to keep in host memory (default 4, ~800 MB)

        Returns:
            WeightPreloaderBridge instance (for accessing loaded tensors).

        Raises:
            RuntimeError: if no layout has been compiled.
            NotImplementedError: if the weight preloader C library is not available.
        """
        if layout is None:
            if self._last_layout is None:
                raise RuntimeError(
                    "No layout compiled. Call compile() first, "
                    "or pass an explicit layout."
                )
            layout = self._last_layout

        # Defer import so the weight_layout module is usable without the bridge
        try:
            from prefill.weight_preloader_bridge import (
                WeightPreloaderBridge, integrate_with_weight_layout,
            )
        except ImportError as e:
            raise NotImplementedError(
                f"Cannot import weight_preloader_bridge: {e}. "
                "Ensure scripts/prefill/ is on PYTHONPATH."
            )

        # Use provided weight_dir or fall back to previously configured
        effective_dir = weight_dir or self._preloader_weight_dir
        if not effective_dir:
            raise ValueError(
                "weight_dir must be specified (e.g. '/data/weights/deepseek_v4')"
            )

        bridge = WeightPreloaderBridge()

        if not bridge.is_available:
            raise NotImplementedError(
                "weight_preloader shared library not available. "
                "Build with: cd c_ref/prefill && "
                "gcc -O3 -shared -fPIC -o build/libweight_preloader.so "
                "weight_preloader.c -laio -lpthread"
            )

        # Collect all unique layer indices from all chips
        all_layers: set = set()
        for chip_layout in layout.chip_layouts:
            all_layers.update(chip_layout.layers)

        bridge.init(effective_dir, num_layers=61, num_preload=num_preload)

        # Load layers in order (LRU-friendly)
        for layer_idx in sorted(all_layers):
            bridge.load_layer(layer_idx)

        self._preloader_bridge = bridge
        self._preloader_weight_dir = effective_dir
        return bridge

    def get_weight_tensor(self, layer_idx: int,
                          tensor_name: str) -> "Optional[np.ndarray]":
        """Get a loaded weight tensor (requires preload_weights() first).

        Args:
            layer_idx: layer index (0..60)
            tensor_name: e.g. "W_Q", "W_gate", "W_down"

        Returns:
            2D numpy array of uint8 fp8 weights, or None if not available.
        """
        import numpy as np
        if self._preloader_bridge is None:
            return None
        return self._preloader_bridge.get_tensor_reshaped(layer_idx, tensor_name)

    def _assign_layers(self, chips: List[ChipLayout], chips_per_pipeline: int):
        """Assign all 61 layers to each pipeline clone."""
        for p in range(self.pipeline_clones):
            start_chip = p * chips_per_pipeline
            # Distribute 61 layers across this pipeline's chips.
            # Example: clone=1 → 32 chips, mostly 2 layers/chip.
            # clone=2 → 16 chips, mostly 4 layers/chip.
            base = NUM_LAYERS // chips_per_pipeline
            extra = NUM_LAYERS % chips_per_pipeline
            layer = 0
            for local_idx in range(chips_per_pipeline):
                n = base + (1 if local_idx < extra else 0)
                chips[start_chip + local_idx].layers = list(range(layer, layer + n))
                layer += n

    def _assign_experts(self, chips: List[ChipLayout]) -> Dict[int, List[int]]:
        """Assign expert replicas per pipeline.

        Each pipeline clone holds a complete logical expert set so it can serve
        requests independently. Hot replication is applied within each pipeline.
        """
        expert_to_chips: Dict[int, List[int]] = defaultdict(list)
        chips_per_pipeline = TOTAL_CHIPS // self.pipeline_clones

        for p in range(self.pipeline_clones):
            pipeline_chips = chips[p * chips_per_pipeline:(p + 1) * chips_per_pipeline]
            if self.replication == "none":
                self._assign_experts_uniform(pipeline_chips, expert_to_chips)
            else:
                self._assign_experts_hot(pipeline_chips, expert_to_chips)

        for c in chips:
            c.experts.sort()
        return dict(expert_to_chips)

    def _assign_experts_uniform(self, pipeline_chips: List[ChipLayout],
                                expert_to_chips: Dict[int, List[int]]):
        for eid in range(NUM_EXPERTS):
            chip = pipeline_chips[eid % len(pipeline_chips)]
            chip.experts.append(eid)
            expert_to_chips[eid].append(chip.chip_id)

    def _assign_experts_hot(self, pipeline_chips: List[ChipLayout],
                            expert_to_chips: Dict[int, List[int]]):
        pop = ExpertPopularity(num_experts=NUM_EXPERTS, alpha=self.zipf_alpha,
                               seed=self.seed)
        plan = pop.replica_plan(total_chips=len(pipeline_chips),
                                hbm_budget_per_chip_gb=2.0,
                                expert_weight_mb=EXPERT_TOTAL_MB)
        for eid in range(NUM_EXPERTS):
            n_rep = min(plan[eid], len(pipeline_chips))
            # Choose least-loaded chips for this expert to balance HBM.
            chosen = sorted(pipeline_chips, key=lambda c: len(c.experts))[:n_rep]
            for chip in chosen:
                chip.experts.append(eid)
                expert_to_chips[eid].append(chip.chip_id)

    def _allocate_hbm_regions(self, chips: List[ChipLayout]):
        for c in chips:
            # Deterministic per-layer weights (attention + router)
            for layer in c.layers:
                c.add_region(
                    name=f"layer{layer:02d}_attn",
                    size_mb=ATTN_TOTAL_MB_PER_LAYER,
                    kind="attention",
                    metadata={"layer": layer},
                )
                c.add_region(
                    name=f"layer{layer:02d}_router",
                    size_mb=ROUTER_WEIGHT_MB,
                    kind="router",
                    metadata={"layer": layer},
                )
            # Expert weights (replicas)
            for eid in c.experts:
                c.add_region(
                    name=f"expert{eid:03d}",
                    size_mb=EXPERT_TOTAL_MB,
                    kind="expert",
                    metadata={"expert": eid},
                )
            # Reserve runtime region for KV + activations.
            c.add_region(
                name="runtime_kv_activation",
                size_mb=self.kv_reserve_gb * 1024,
                kind="runtime",
                metadata={"kv_reserve_gb": self.kv_reserve_gb},
            )

    def _validate(self, chips: List[ChipLayout]):
        overflow = [c for c in chips if c.used_gb > HBM_SIZE_GB]
        if overflow:
            msg = "; ".join(f"chip {c.chip_id}: {c.used_gb:.2f} GB" for c in overflow[:4])
            raise RuntimeError(f"HBM overflow: {msg}")


def demo_layouts() -> str:
    """Return a multi-layout smoke-test report."""
    reports = []
    for clone in (1, 2, 4):
        for repl in ("none", "hot"):
            # More clones means each pipeline has fewer chips and more layer weight
            # per chip; keep a slightly smaller KV reserve for clone=4.
            kv = 22.0 if clone <= 2 else 20.0
            rpt = WeightLayoutCompiler(pipeline_clones=clone, replication=repl,
                                       kv_reserve_gb=kv).compile()
            reports.append(rpt.summary())
    return "\n\n".join(reports)


if __name__ == "__main__":
    print(demo_layouts())
