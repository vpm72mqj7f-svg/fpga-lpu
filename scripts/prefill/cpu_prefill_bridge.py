#!/usr/bin/env python3
"""
cpu_prefill_bridge.py — ctypes bridge to the C CPU prefill engine.

Loads libcpu_prefill.so/.dll and wraps the key C functions with
numpy array <-> C pointer conversion. Falls back gracefully with a
warning if the shared library is not yet built.

Build the shared library first:
  Linux:   cd c_ref/prefill && bash build.sh
  Windows: cd c_ref\\prefill && build.bat

Usage:
    from prefill.cpu_prefill_bridge import CpuPrefillEngine

    engine = CpuPrefillEngine()
    engine.init(num_threads=0)
    output, kv_k, kv_v = engine.prefill_chunk(hidden_state, chunk_size)

    # Or at module level:
    import prefill.cpu_prefill_bridge as bridge
    bridge.cpu_gemm_fp8(M, K, N, A, B, C, scale_A, scale_B)
"""

import ctypes
import os
import warnings
from typing import Optional, Tuple

import numpy as np

# ── Library discovery ──────────────────────────────────────────────────

_LIB_NAME = "cpu_prefill"
_LIB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "..", "c_ref", "prefill", "build"))

def _find_lib():
    """Locate the shared library. Returns path or None."""
    candidates = [
        os.path.join(_LIB_DIR, "libcpu_prefill.so"),
        os.path.join(_LIB_DIR, "cpu_prefill.dll"),
        os.path.join(_LIB_DIR, "libcpu_prefill.dylib"),
        # Fallback: system library path
        f"lib{_LIB_NAME}.so",
        f"{_LIB_NAME}.dll",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

_lib = None
_lib_path = _find_lib()

if _lib_path:
    try:
        _lib = ctypes.CDLL(_lib_path)
    except OSError as e:
        warnings.warn(f"cpu_prefill: failed to load {_lib_path}: {e}")
else:
    warnings.warn(
        "cpu_prefill shared library not built. "
        "Run: cd c_ref/prefill && bash build.sh (Linux) or build.bat (Windows). "
        "CPU prefill functions will raise NotImplementedError until the library is built."
    )

# ── C Struct Definitions ───────────────────────────────────────────────

class CpuPrefillBackend(ctypes.c_int):
    CPU_PREFILL_AMX = 0
    CPU_PREFILL_AVX512 = 1
    CPU_PREFILL_SCALAR = 2

class CpuPrefillConfig(ctypes.Structure):
    _fields_ = [
        ("backend",           ctypes.c_int),
        ("num_threads",       ctypes.c_int),
        ("max_chunk_size",    ctypes.c_int),
        ("hidden_dim",        ctypes.c_int),
        ("intermediate_dim",  ctypes.c_int),
        ("kv_latent_dim",     ctypes.c_int),
        ("num_experts",       ctypes.c_int),
        ("top_k",             ctypes.c_int),
        ("num_layers",        ctypes.c_int),
    ]

class CpuPrefillStats(ctypes.Structure):
    _fields_ = [
        ("total_us",            ctypes.c_double),
        ("gemm_us",             ctypes.c_double),
        ("attention_us",        ctypes.c_double),
        ("moe_us",              ctypes.c_double),
        ("effective_tflops",    ctypes.c_double),
        ("chunks_processed",    ctypes.c_int),
        ("tokens_prefilled",    ctypes.c_int),
    ]

# ── Default Configuration ──────────────────────────────────────────────

DEFAULT_CFG = dict(
    backend=CpuPrefillBackend.CPU_PREFILL_SCALAR,
    num_threads=0,
    max_chunk_size=128,
    hidden_dim=7168,
    intermediate_dim=3072,
    kv_latent_dim=512,
    num_experts=384,
    top_k=6,
    num_layers=61,
)

# ── Function Signatures ────────────────────────────────────────────────

def _setup_functions(lib):
    """Set up ctypes function signatures."""
    # double cpu_gemm_fp8(int M, int K, int N,
    #     const uint8_t *A, const uint8_t *B, float *C,
    #     const float *scale_A, const float *scale_B)
    lib.cpu_gemm_fp8.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.cpu_gemm_fp8.restype = ctypes.c_double

    # double cpu_batched_gemv_fp8(int batch, int K, int N,
    #     const uint8_t *A, const uint8_t *B, float *C,
    #     const float *scale_A, const float *scale_B)
    lib.cpu_batched_gemv_fp8.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.cpu_batched_gemv_fp8.restype = ctypes.c_double

    # double cpu_prefill_layer(const cpu_prefill_config_t *cfg,
    #     int layer_idx, int chunk_size,
    #     const uint8_t *hidden_state, uint8_t *output_state,
    #     uint8_t *kv_cache_k, uint8_t *kv_cache_v)
    lib.cpu_prefill_layer.argtypes = [
        ctypes.POINTER(CpuPrefillConfig),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.cpu_prefill_layer.restype = ctypes.c_double

    # double cpu_prefill_all_layers(const cpu_prefill_config_t *cfg,
    #     int chunk_size,
    #     const uint8_t *input_tokens, uint8_t *output_state,
    #     uint8_t *kv_cache_k_all, uint8_t *kv_cache_v_all)
    lib.cpu_prefill_all_layers.argtypes = [
        ctypes.POINTER(CpuPrefillConfig),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.cpu_prefill_all_layers.restype = ctypes.c_double

    # int cpu_weight_cache_load(const cpu_prefill_config_t *cfg,
    #     int layer_idx, const void *weight_data, size_t weight_size)
    lib.cpu_weight_cache_load.argtypes = [
        ctypes.POINTER(CpuPrefillConfig),
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.cpu_weight_cache_load.restype = ctypes.c_int

    # void cpu_weight_cache_unload(int layer_idx)
    lib.cpu_weight_cache_unload.argtypes = [ctypes.c_int]
    lib.cpu_weight_cache_unload.restype = None

    # void cpu_weight_cache_unload_all(void)
    lib.cpu_weight_cache_unload_all.argtypes = []
    lib.cpu_weight_cache_unload_all.restype = None

    # const cpu_prefill_stats_t *cpu_prefill_get_stats(void)
    lib.cpu_prefill_get_stats.argtypes = []
    lib.cpu_prefill_get_stats.restype = ctypes.POINTER(CpuPrefillStats)

    # void cpu_prefill_reset_stats(void)
    lib.cpu_prefill_reset_stats.argtypes = []
    lib.cpu_prefill_reset_stats.restype = None

if _lib:
    _setup_functions(_lib)

# ── Helper: numpy → ctypes pointer conversion ──────────────────────────

def _as_u8_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    """Return ctypes pointer to uint8 array data (must be contiguous uint8)."""
    if arr.dtype != np.uint8:
        raise TypeError(f"Expected uint8 array, got {arr.dtype}")
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

def _as_f32_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    """Return ctypes pointer to float32 array data."""
    if arr.dtype != np.float32:
        raise TypeError(f"Expected float32 array, got {arr.dtype}")
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def _ensure_u8(arr: np.ndarray) -> np.ndarray:
    """Ensure array is uint8, contiguous, C-order."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return np.ascontiguousarray(arr)

def _ensure_f32(arr: np.ndarray) -> np.ndarray:
    """Ensure array is float32, contiguous, C-order."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)

# ── Null-check guards for unavailable library ───────────────────────────

def _require_lib():
    if _lib is None:
        raise NotImplementedError(
            "cpu_prefill shared library not available. "
            "Build with: cd c_ref/prefill && bash build.sh"
        )

# ── Public API ─────────────────────────────────────────────────────────

_default_cfg = CpuPrefillConfig(**DEFAULT_CFG)

def cpu_gemm_fp8(M: int, K: int, N: int,
                 A: np.ndarray, B: np.ndarray, C: np.ndarray,
                 scale_A: Optional[np.ndarray] = None,
                 scale_B: Optional[np.ndarray] = None) -> float:
    """fp8 GEMM: C[M,N] = A[M,K] x B[K,N].

    A: uint8 [M * K]  — fp8 E4M3 activations
    B: uint8 [K * N]  — fp8 E4M3 weights
    C: float32 [M * N] — output (modified in-place)

    Returns: GFLOPS achieved.
    """
    _require_lib()
    A = _ensure_u8(A)
    B = _ensure_u8(B)
    C = _ensure_f32(C)

    sA = _as_f32_ptr(_ensure_f32(scale_A)) if scale_A is not None \
         else ctypes.POINTER(ctypes.c_float)()
    sB = _as_f32_ptr(_ensure_f32(scale_B)) if scale_B is not None \
         else ctypes.POINTER(ctypes.c_float)()

    return _lib.cpu_gemm_fp8(M, K, N,
                              _as_u8_ptr(A), _as_u8_ptr(B),
                              _as_f32_ptr(C), sA, sB)

def cpu_batched_gemv_fp8(batch: int, K: int, N: int,
                         A: np.ndarray, B: np.ndarray, C: np.ndarray,
                         scale_A: Optional[np.ndarray] = None,
                         scale_B: Optional[np.ndarray] = None) -> float:
    """Batched fp8 GEMV: C[b,N] = A[b,:] @ B[:,N] for each b.

    A: uint8 [batch * K]
    B: uint8 [K * N]
    C: float32 [batch * N] — output (modified in-place)

    Returns: GFLOPS achieved.
    """
    _require_lib()
    A = _ensure_u8(A)
    B = _ensure_u8(B)
    C = _ensure_f32(C)

    sA = _as_f32_ptr(_ensure_f32(scale_A)) if scale_A is not None \
         else ctypes.POINTER(ctypes.c_float)()
    sB = _as_f32_ptr(_ensure_f32(scale_B)) if scale_B is not None \
         else ctypes.POINTER(ctypes.c_float)()

    return _lib.cpu_batched_gemv_fp8(batch, K, N,
                                      _as_u8_ptr(A), _as_u8_ptr(B),
                                      _as_f32_ptr(C), sA, sB)

def prefill_layer(cfg: Optional[CpuPrefillConfig],
                  layer_idx: int, chunk_size: int,
                  hidden_state: np.ndarray,
                  output_state: np.ndarray,
                  kv_cache_k: np.ndarray,
                  kv_cache_v: np.ndarray) -> float:
    """Prefill one transformer layer.

    hidden_state: uint8 [chunk_size * hidden_dim]
    output_state: uint8 [chunk_size * hidden_dim]  (output written here)
    kv_cache_k:   uint8 [chunk_size * kv_latent_dim]  (output)
    kv_cache_v:   uint8 [chunk_size * kv_latent_dim]  (output)

    Returns: elapsed microseconds.
    """
    _require_lib()
    _cfg = cfg if cfg is not None else _default_cfg

    return _lib.cpu_prefill_layer(
        ctypes.byref(_cfg), layer_idx, chunk_size,
        _as_u8_ptr(_ensure_u8(hidden_state)),
        _as_u8_ptr(_ensure_u8(output_state)),
        _as_u8_ptr(_ensure_u8(kv_cache_k)),
        _as_u8_ptr(_ensure_u8(kv_cache_v)),
    )

def prefill_all_layers(cfg: Optional[CpuPrefillConfig],
                       chunk_size: int,
                       input_tokens: np.ndarray,
                       output_state: np.ndarray,
                       kv_cache_k_all: np.ndarray,
                       kv_cache_v_all: np.ndarray) -> float:
    """Prefill all layers for one chunk.

    input_tokens:    uint8 [chunk_size * hidden_dim]
    output_state:    uint8 [chunk_size * hidden_dim] (output)
    kv_cache_k_all:  uint8 [num_layers * chunk_size * kv_latent_dim] (output)
    kv_cache_v_all:  uint8 [num_layers * chunk_size * kv_latent_dim] (output)

    Returns: total elapsed microseconds.
    """
    _require_lib()
    _cfg = cfg if cfg is not None else _default_cfg

    return _lib.cpu_prefill_all_layers(
        ctypes.byref(_cfg), chunk_size,
        _as_u8_ptr(_ensure_u8(input_tokens)),
        _as_u8_ptr(_ensure_u8(output_state)),
        _as_u8_ptr(_ensure_u8(kv_cache_k_all)),
        _as_u8_ptr(_ensure_u8(kv_cache_v_all)),
    )

def weight_cache_load(cfg: Optional[CpuPrefillConfig],
                      layer_idx: int,
                      weight_data: np.ndarray,
                      weight_size: int) -> int:
    """Pre-load weights for one layer into CPU memory.

    weight_data: raw fp4 weights (uint8 buffer)

    Returns: 0 on success, -1 on error.
    """
    _require_lib()
    _cfg = cfg if cfg is not None else _default_cfg

    data = _ensure_u8(weight_data)
    return _lib.cpu_weight_cache_load(
        ctypes.byref(_cfg), layer_idx,
        data.ctypes.data_as(ctypes.c_void_p),
        weight_size,
    )

def weight_cache_unload(layer_idx: int):
    """Unload weights for one layer."""
    _require_lib()
    _lib.cpu_weight_cache_unload(layer_idx)

def weight_cache_unload_all():
    """Unload all cached weights."""
    _require_lib()
    _lib.cpu_weight_cache_unload_all()

def get_stats() -> CpuPrefillStats:
    """Get current performance statistics."""
    _require_lib()
    result_ptr = _lib.cpu_prefill_get_stats()
    return result_ptr.contents

def reset_stats():
    """Reset performance statistics."""
    _require_lib()
    _lib.cpu_prefill_reset_stats()

# ── High-Level Python Wrapper ──────────────────────────────────────────

class CpuPrefillEngine:
    """High-level Python wrapper around the C CPU prefill engine.

    Usage:
        engine = CpuPrefillEngine(num_threads=0)
        engine.init()

        hidden = np.random.randint(0, 256, size=(128, 7168), dtype=np.uint8)
        output, kv_k, kv_v = engine.prefill_chunk(hidden, chunk_size=128)
    """

    def __init__(self, num_threads: int = 0,
                 backend: int = CpuPrefillBackend.CPU_PREFILL_SCALAR,
                 hidden_dim: int = 7168,
                 intermediate_dim: int = 3072,
                 kv_latent_dim: int = 512,
                 num_experts: int = 384,
                 top_k: int = 6,
                 num_layers: int = 61,
                 max_chunk_size: int = 128):
        self.cfg = CpuPrefillConfig(
            backend=backend,
            num_threads=num_threads,
            max_chunk_size=max_chunk_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            kv_latent_dim=kv_latent_dim,
            num_experts=num_experts,
            top_k=top_k,
            num_layers=num_layers,
        )
        self._initialized = False

    @property
    def is_available(self) -> bool:
        return _lib is not None

    def init(self):
        """Initialize engine. Required before calling prefill functions."""
        _require_lib()
        self._initialized = True

    def prefill_layer(self, layer_idx: int,
                      hidden_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prefill a single layer. Returns (output, kv_k, kv_v)."""
        _require_lib()
        if not self._initialized:
            raise RuntimeError("CpuPrefillEngine not initialized. Call init().")

        chunk_size = hidden_state.shape[0]
        H = self.cfg.hidden_dim
        KL = self.cfg.kv_latent_dim

        output = np.zeros((chunk_size, H), dtype=np.uint8)
        kv_k = np.zeros((chunk_size, KL), dtype=np.uint8)
        kv_v = np.zeros((chunk_size, KL), dtype=np.uint8)

        prefill_layer(self.cfg, layer_idx, chunk_size,
                      hidden_state.ravel(), output.ravel(),
                      kv_k.ravel(), kv_v.ravel())

        return output, kv_k, kv_v

    def prefill_chunk(self, hidden_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prefill all layers for one chunk. Returns (output, kv_k_all, kv_v_all)."""
        _require_lib()
        if not self._initialized:
            raise RuntimeError("CpuPrefillEngine not initialized. Call init().")

        chunk_size = hidden_state.shape[0]
        H = self.cfg.hidden_dim
        L = self.cfg.num_layers
        KL = self.cfg.kv_latent_dim

        output = np.zeros((chunk_size, H), dtype=np.uint8)
        kv_k_all = np.zeros((L * chunk_size, KL), dtype=np.uint8)
        kv_v_all = np.zeros((L * chunk_size, KL), dtype=np.uint8)

        elapsed_us = prefill_all_layers(
            self.cfg, chunk_size,
            hidden_state.ravel(), output.ravel(),
            kv_k_all.ravel(), kv_v_all.ravel(),
        )

        # Reshape KV to [num_layers, chunk_size, kv_latent_dim]
        kv_k_all = kv_k_all.reshape(L, chunk_size, KL)
        kv_v_all = kv_v_all.reshape(L, chunk_size, KL)

        return output, kv_k_all, kv_v_all

    def load_weights(self, layer_idx: int, weight_data: np.ndarray):
        """Load and cache weights for a layer."""
        _require_lib()
        size = weight_data.nbytes
        ret = weight_cache_load(self.cfg, layer_idx, weight_data, size)
        if ret != 0:
            raise RuntimeError(f"weight_cache_load failed for layer {layer_idx}")

    def unload_weights(self, layer_idx: int = -1):
        """Unload weights. layer_idx=-1 unloads all."""
        _require_lib()
        if layer_idx < 0:
            weight_cache_unload_all()
        else:
            weight_cache_unload(layer_idx)

    @property
    def stats(self) -> dict:
        """Performance statistics as a dict."""
        _require_lib()
        s = get_stats()
        return {
            'total_us': s.total_us,
            'gemm_us': s.gemm_us,
            'attention_us': s.attention_us,
            'moe_us': s.moe_us,
            'effective_tflops': s.effective_tflops,
            'chunks_processed': s.chunks_processed,
            'tokens_prefilled': s.tokens_prefilled,
        }

    def reset_stats(self):
        """Reset performance counters."""
        _require_lib()
        reset_stats()
