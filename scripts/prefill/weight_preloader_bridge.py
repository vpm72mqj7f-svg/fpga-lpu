#!/usr/bin/env python3
"""
weight_preloader_bridge.py — ctypes bridge to the C weight preloader.

Loads libweight_preloader.so/.dll and wraps the key C functions for
fp4 weight loading from SSD into host pinned memory. Used by the
WeightLayoutCompiler to pre-load weights before FPGA inference.

The weight preloader:
  - Reads fp4 (4-bit) weights from SSD using async I/O (libaio)
  - Unpacks fp4 to fp8 E4M3 in host memory
  - Pins memory with mlock for DMA to FPGA HBM
  - Manages an LRU cache of preloaded layers (~800 MB for 4 layers)

Build the shared library first:
  Linux:   cd c_ref/prefill && gcc -O3 -shared -fPIC -o build/libweight_preloader.so weight_preloader.c -laio -lpthread
  Windows: not supported (requires libaio + mlock, Linux-only)

Usage:
    from prefill.weight_preloader_bridge import WeightPreloaderBridge

    wp = WeightPreloaderBridge()
    wp.init("/data/weights/deepseek_v4", num_layers=61, num_preload=4)
    wp.load_layer(0)
    data, size = wp.get_tensor(0, "W_Q")

    # Integration with WeightLayoutCompiler:
    compiler = WeightLayoutCompiler()
    compiler.preload_weights(bridge=wp)  # preload all layers in layout
"""

import ctypes
import os
import sys
import warnings
from typing import Optional, Dict, Tuple

import numpy as np

# ── Library discovery ──────────────────────────────────────────────────

_LIB_NAME = "weight_preloader"
_LIB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "..", "c_ref", "prefill", "build"))

# Only supported on Linux (requires libaio + mlock)
_IS_LINUX = sys.platform.startswith("linux")


def _find_lib():
    """Locate the shared library. Returns path or None."""
    if not _IS_LINUX:
        return None
    candidates = [
        os.path.join(_LIB_DIR, "libweight_preloader.so"),
        os.path.join(_LIB_DIR, "weight_preloader.so"),
        f"lib{_LIB_NAME}.so",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


_lib = None
_lib_path = _find_lib()

_LOAD_WARNING_ISSUED = False

if _lib_path:
    try:
        _lib = ctypes.CDLL(_lib_path)
    except OSError as e:
        warnings.warn(f"weight_preloader: failed to load {_lib_path}: {e}")
elif not _IS_LINUX:
    if not _LOAD_WARNING_ISSUED:
        warnings.warn(
            "weight_preloader: not supported on this platform "
            f"({sys.platform}). Requires Linux with libaio and mlock. "
            "Weight preload functions will raise NotImplementedError."
        )
        _LOAD_WARNING_ISSUED = True
else:
    if not _LOAD_WARNING_ISSUED:
        warnings.warn(
            "weight_preloader shared library not built. "
            "Build with: cd c_ref/prefill && "
            "gcc -O3 -shared -fPIC -o build/libweight_preloader.so "
            "weight_preloader.c -laio -lpthread"
        )
        _LOAD_WARNING_ISSUED = True


# ── C Struct Definitions ───────────────────────────────────────────────

class WeightTensor(ctypes.Structure):
    """C weight_tensor_t: describes one weight matrix in a layer."""
    _fields_ = [
        ("name",       ctypes.c_char_p),
        ("rows",       ctypes.c_size_t),
        ("cols",       ctypes.c_size_t),
        ("elem_bytes", ctypes.c_size_t),   # 1 for fp8, 0.5 for fp4 (packed)
        ("is_fp4",     ctypes.c_int),      # needs unpacking
    ]


class LayerWeights(ctypes.Structure):
    """C layer_weights_t: all weight tensors for one layer."""
    _fields_ = [
        ("layer_idx",   ctypes.c_int),
        ("tensors",     WeightTensor * 20),  # up to 20 tensors per layer
        ("num_tensors", ctypes.c_int),
        ("data",        ctypes.POINTER(ctypes.c_uint8)),
        ("total_bytes", ctypes.c_size_t),
    ]


class WeightPreloader(ctypes.Structure):
    """C weight_preloader_t: preloader state.

    The io_context field is an opaque pointer (libaio io_context_t) which
    we represent as a void pointer. We don't access it from Python.
    """
    _fields_ = [
        ("weight_dir",   ctypes.c_char_p),
        ("num_layers",   ctypes.c_int),
        ("num_preload",  ctypes.c_int),
        ("layers",       ctypes.POINTER(LayerWeights)),
        ("total_memory", ctypes.c_size_t),
        ("lock",         ctypes.c_byte * 64),  # pthread_mutex_t (opaque)
    ]


# ── Function Signatures ────────────────────────────────────────────────

def _setup_functions(lib):
    """Set up ctypes function signatures for the weight preloader."""
    # int weight_preloader_init(weight_preloader_t *wp,
    #     const char *weight_dir, int num_layers, int num_preload)
    lib.weight_preloader_init.argtypes = [
        ctypes.POINTER(WeightPreloader),
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.weight_preloader_init.restype = ctypes.c_int

    # int weight_preloader_load_layer(weight_preloader_t *wp,
    #     int layer_idx, io_context_t *io_ctx)
    # io_context_t is an opaque pointer from libaio.
    lib.weight_preloader_load_layer.argtypes = [
        ctypes.POINTER(WeightPreloader),
        ctypes.c_int,
        ctypes.c_void_p,  # io_context_t *
    ]
    lib.weight_preloader_load_layer.restype = ctypes.c_int

    # const uint8_t *weight_preloader_get_tensor(
    #     const weight_preloader_t *wp, int layer_idx,
    #     const char *tensor_name, size_t *out_bytes)
    lib.weight_preloader_get_tensor.argtypes = [
        ctypes.POINTER(WeightPreloader),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.weight_preloader_get_tensor.restype = ctypes.POINTER(ctypes.c_uint8)

    # void weight_preloader_destroy(weight_preloader_t *wp)
    lib.weight_preloader_destroy.argtypes = [
        ctypes.POINTER(WeightPreloader),
    ]
    lib.weight_preloader_destroy.restype = None


if _lib:
    _setup_functions(_lib)


# ── Null-check guard ───────────────────────────────────────────────────

def _require_lib():
    if _lib is None:
        raise NotImplementedError(
            "weight_preloader shared library not available. "
            "Build with: cd c_ref/prefill && "
            "gcc -O3 -shared -fPIC -o build/libweight_preloader.so "
            "weight_preloader.c -laio -lpthread"
        )


# ── Public API ─────────────────────────────────────────────────────────

# Known tensor names per layer (matching DeepSeek V4 Pro MLA architecture)
TENSOR_NAMES = [
    "W_Q", "W_K", "W_V", "W_K_up", "W_V_up",
    "W_gate", "W_up", "W_down", "W_router",
]
# Per-expert tensors: W_gate, W_up, W_down for each of 12 experts per chip
EXPERT_TENSOR_NAMES = ["W_gate", "W_up", "W_down"]

# Per-layer fp8 weight sizes (bytes), from the C code layout.
# These match the tensors[] array in weight_preloader.c:134-145.
TENSOR_BYTES = {
    "W_Q":      7168 * 7168,       # 51.4 MB
    "W_K":        512 * 7168,      #  3.7 MB
    "W_V":        512 * 7168,      #  3.7 MB
    "W_K_up":    7168 * 512,       #  3.7 MB
    "W_V_up":    7168 * 512,       #  3.7 MB
    "W_gate":    3072 * 7168,      # 22.0 MB
    "W_up":      3072 * 7168,      # 22.0 MB
    "W_down":    7168 * 3072,      # 22.0 MB
    "W_router":   384 * 7168,      #  2.8 MB
}
TENSOR_TOTAL_BYTES = sum(TENSOR_BYTES.values())  # ~136 MB per layer


def weight_preloader_init(wp_ptr, weight_dir: str,
                          num_layers: int = 61,
                          num_preload: int = 4) -> int:
    """Initialize weight preloader state.

    Args:
        wp_ptr: ctypes.POINTER(WeightPreloader) to initialized state
        weight_dir: path to SSD weight files
        num_layers: total layers (default 61, DeepSeek V4)
        num_preload: max layers kept in memory (default 4, ~800 MB)

    Returns: 0 on success, -1 on error.
    """
    _require_lib()
    weight_dir_bytes = weight_dir.encode("utf-8")
    return _lib.weight_preloader_init(
        wp_ptr, weight_dir_bytes, num_layers, num_preload,
    )


def weight_preloader_load_layer(wp_ptr, layer_idx: int,
                                io_ctx: Optional[int] = None) -> int:
    """Load a single layer's weights from SSD into pinned memory.

    Args:
        wp_ptr: ctypes.POINTER(WeightPreloader)
        layer_idx: which layer to load (0..60)
        io_ctx: libaio io_context_t pointer (NULL = slow fallback path)

    Returns: 0 on success, -1 on error.
    """
    _require_lib()
    ctx = ctypes.c_void_p(io_ctx) if io_ctx else ctypes.c_void_p(0)
    return _lib.weight_preloader_load_layer(wp_ptr, layer_idx, ctx)


def weight_preloader_get_tensor(wp_ptr, layer_idx: int,
                                tensor_name: str) -> Tuple[Optional[np.ndarray], int]:
    """Get a specific weight tensor for a loaded layer.

    Args:
        wp_ptr: ctypes.POINTER(WeightPreloader)
        layer_idx: layer index (0..60)
        tensor_name: e.g. "W_Q", "W_gate", "W_down"

    Returns:
        (numpy array of uint8 fp8 data, size_in_bytes), or (None, 0) if not found.
    """
    _require_lib()
    name_bytes = tensor_name.encode("utf-8")
    out_bytes = ctypes.c_size_t(0)

    data_ptr = _lib.weight_preloader_get_tensor(
        wp_ptr, layer_idx, name_bytes, ctypes.byref(out_bytes),
    )

    if not data_ptr or out_bytes.value == 0:
        return None, 0

    # Copy data from pinned memory into numpy array
    size = out_bytes.value
    result = np.empty(size, dtype=np.uint8)
    ctypes.memmove(result.ctypes.data_as(ctypes.c_void_p), data_ptr, size)
    return result, size


def weight_preloader_destroy(wp_ptr):
    """Release all memory and destroy the weight preloader state."""
    _require_lib()
    _lib.weight_preloader_destroy(wp_ptr)
    # Zero out the struct to prevent dangling pointer reuse
    ctypes.memset(wp_ptr, 0, ctypes.sizeof(WeightPreloader))


# ── High-Level Python Wrapper ──────────────────────────────────────────

class WeightPreloaderBridge:
    """High-level Python wrapper around the C weight preloader.

    Manages lifetime of the C weight_preloader_t struct and provides
    a Pythonic API for loading and accessing layer weights.

    Usage:
        wp = WeightPreloaderBridge()
        wp.init("/data/weights/deepseek_v4", num_layers=61, num_preload=4)
        wp.load_layer(0)  # load layer 0
        wq, size = wp.get_tensor(0, "W_Q")
        print(f"W_Q layer 0: {size} bytes, shape=({7168},{7168})")
        wp.close()
    """

    def __init__(self):
        self._wp = WeightPreloader()
        self._initialized = False
        self._loaded_layers: set = set()

    @property
    def is_available(self) -> bool:
        return _lib is not None

    def init(self, weight_dir: str, num_layers: int = 61,
             num_preload: int = 4):
        """Initialize weight preloader.

        Must be called before any load_layer() or get_tensor() calls.

        Args:
            weight_dir: path to SSD weight files (e.g. "/data/weights/deepseek_v4")
            num_layers: total model layers
            num_preload: max layers to keep in host memory (LRU eviction)
        """
        _require_lib()
        ret = weight_preloader_init(
            ctypes.byref(self._wp), weight_dir, num_layers, num_preload,
        )
        if ret != 0:
            raise RuntimeError(
                f"weight_preloader_init failed for {weight_dir}"
            )
        self._initialized = True

    def load_layer(self, layer_idx: int):
        """Load a single layer's weights from SSD into memory.

        Uses async I/O (libaio) on Linux. Falls back to synchronous
        read if async I/O is unavailable. Automatically evicts LRU
        layers when over the preload limit.
        """
        _require_lib()
        if not self._initialized:
            raise RuntimeError("Bridge not initialized. Call init() first.")
        ret = weight_preloader_load_layer(
            ctypes.byref(self._wp), layer_idx,
        )
        if ret != 0:
            raise RuntimeError(
                f"weight_preloader_load_layer failed for layer {layer_idx}"
            )
        self._loaded_layers.add(layer_idx)

    def load_layers(self, layer_indices: list):
        """Load multiple layers sequentially.

        Args:
            layer_indices: list of layer indices to load (e.g. [0, 1, 2, 3])
        """
        for idx in layer_indices:
            self.load_layer(idx)

    def get_tensor(self, layer_idx: int,
                   tensor_name: str) -> Tuple[Optional[np.ndarray], int]:
        """Get a weight tensor for a loaded layer.

        Args:
            layer_idx: layer index (must already be loaded)
            tensor_name: tensor name (e.g. "W_Q", "W_gate", "W_down")

        Returns:
            (uint8 numpy array of fp8 weights, size_in_bytes).
            Returns (None, 0) if tensor not found.
        """
        _require_lib()
        if not self._initialized:
            raise RuntimeError("Bridge not initialized. Call init() first.")
        if layer_idx not in self._loaded_layers:
            raise RuntimeError(
                f"Layer {layer_idx} not loaded. Call load_layer({layer_idx}) first."
            )
        return weight_preloader_get_tensor(
            ctypes.byref(self._wp), layer_idx, tensor_name,
        )

    def get_tensor_reshaped(self, layer_idx: int,
                            tensor_name: str) -> Optional[np.ndarray]:
        """Get a tensor reshaped to its canonical dimensions.

        Uses TENSOR_BYTES lookup table to determine rows x cols.
        For fp4-packed tensors, bytes = rows*cols/2; for fp8, bytes = rows*cols.
        All loaded tensors are unpacked to fp8, so bytes = rows * cols.

        Returns:
            2D numpy array (rows, cols) of uint8 fp8 weights, or None.
        """
        data, size = self.get_tensor(layer_idx, tensor_name)
        if data is None:
            return None
        # Canonical dimensions: rows * cols = size (fp8 unpacked)
        expected = TENSOR_BYTES.get(tensor_name, size)
        if tensor_name.startswith("W_"):
            # Determine shape from known tensor dimensions
            rows, cols = _tensor_shape(tensor_name)
            if rows * cols == size:
                return data.reshape(rows, cols)
        # Fallback: return flat
        return data

    def close(self):
        """Release all resources. Safe to call multiple times."""
        if self._initialized and _lib is not None:
            weight_preloader_destroy(ctypes.byref(self._wp))
            self._initialized = False
            self._loaded_layers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def stats(self) -> dict:
        """Return preloader state summary."""
        return {
            "initialized": self._initialized,
            "layers_loaded": sorted(self._loaded_layers),
            "total_layers": len(self._loaded_layers),
            "memory_estimate_mb": len(self._loaded_layers) * TENSOR_TOTAL_BYTES / (1024 * 1024),
        }


def _tensor_shape(name: str) -> Tuple[int, int]:
    """Return (rows, cols) for a known tensor name based on C layout."""
    shapes = {
        "W_Q":      (7168, 7168),
        "W_K":      (512,  7168),
        "W_V":      (512,  7168),
        "W_K_up":   (7168, 512),
        "W_V_up":   (7168, 512),
        "W_gate":   (3072, 7168),
        "W_up":     (3072, 7168),
        "W_down":   (7168, 3072),
        "W_router": (384,  7168),
    }
    return shapes.get(name, (0, 0))


# ── Integration Point: WeightLayoutCompiler ────────────────────────────

def integrate_with_weight_layout(compiler, weight_dir: str = "",
                                 num_preload: int = 4):
    """Attach weight preloading capability to a WeightLayoutCompiler instance.

    Adds a .preload_weights() method that uses the bridge to load all
    layers referenced in the compiled layout. Designed to be called
    after compiler.compile().

    Usage:
        from vllm_serve.weight_layout import WeightLayoutCompiler
        from prefill.weight_preloader_bridge import integrate_with_weight_layout

        compiler = WeightLayoutCompiler(pipeline_clones=1)
        layout = compiler.compile()
        integrate_with_weight_layout(compiler, weight_dir="/data/weights/v4")
        compiler.preload_weights()

    Args:
        compiler: WeightLayoutCompiler instance
        weight_dir: path to SSD weight files
        num_preload: max layers to keep in host memory
    """
    wp = WeightPreloaderBridge()

    def _preload_weights(layout=None):
        """Preload all layers in the compiled layout into host memory.

        If layout is None, uses the last compiled layout (must call
        compiler.compile() first).

        Returns the WeightPreloaderBridge instance for further use.
        """
        nonlocal wp
        if layout is None:
            if not hasattr(compiler, '_last_layout'):
                raise RuntimeError(
                    "No layout available. Call compiler.compile() first."
                )
            layout = compiler._last_layout

        if not wp.is_available:
            raise NotImplementedError(
                "weight_preloader shared library not available. "
                "See prefill.weight_preloader_bridge for build instructions."
            )

        # Collect all unique layer indices from all chip layouts
        all_layers = set()
        for chip_layout in layout.chip_layouts:
            all_layers.update(chip_layout.layers)

        if not wp._initialized:
            wp.init(weight_dir, num_layers=61, num_preload=num_preload)

        # Load layers in sorted order (LRU-friendly)
        for layer_idx in sorted(all_layers):
            wp.load_layer(layer_idx)

        return wp

    compiler.preload_weights = _preload_weights
    compiler._weight_preloader = wp
    return compiler


# ── Module-level smoke test ────────────────────────────────────────────

def smoke_test():
    """Verify the bridge module loads correctly (without C library)."""
    wp = WeightPreloaderBridge()
    print(f"WeightPreloaderBridge created")
    print(f"  is_available: {wp.is_available}")
    print(f"  platform: {sys.platform}")
    print(f"  library path: {_lib_path or 'N/A'}")

    # Verify tensor metadata
    print(f"\nTensor layout per layer:")
    total_mb = 0
    for name in TENSOR_NAMES:
        size = TENSOR_BYTES[name]
        total_mb += size
        rows, cols = _tensor_shape(name)
        print(f"  {name:12s}: {rows:5d} x {cols:5d} = {size/1e6:7.1f} MB (fp8)")
    print(f"  {'TOTAL':12s}:              {total_mb/1e6:7.1f} MB per layer")
    print(f"  {'x61 layers':12s}:              {total_mb*61/1e9:7.1f} GB total")

    if not wp.is_available:
        print("\n  (C library not available — expected on non-Linux platforms)")

    # Test graceful fallback
    try:
        wp.init("/nonexistent/path")
        print("  ERROR: init() should have raised NotImplementedError")
    except NotImplementedError:
        print("  OK: init() correctly raises NotImplementedError")
    except Exception as e:
        print(f"  OK: init() raises {type(e).__name__}: {e}")

    # Test integration with compiler (offline — no actual layout compilation)
    print("\nIntegration test (offline):")
    try:
        from vllm_serve.weight_layout import WeightLayoutCompiler
        compiler = WeightLayoutCompiler(pipeline_clones=1)
        integrate_with_weight_layout(compiler, weight_dir="/data/weights/v4")
        print("  integrate_with_weight_layout: OK (compiler.preload_weights attached)")
        # preload_weights will fail gracefully since lib is not available
        try:
            compiler.preload_weights()
        except NotImplementedError:
            print("  compiler.preload_weights: correctly raises NotImplementedError")
    except ImportError as e:
        print(f"  Skipped (import error: {e})")

    print("\nWeight preloader bridge smoke test complete.")


if __name__ == "__main__":
    smoke_test()
