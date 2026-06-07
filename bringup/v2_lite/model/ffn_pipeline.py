# =============================================================================
# ffn_pipeline.py — DeepSeek V2-Lite FFN Pipeline (Python reference model)
#
# Matches v2_lite_ffn_engine.sv behaviour:
#   hidden=2048, inter=1408, 66 experts (64 routed + 2 shared), TOP_K=6
#   All data paths are FP8 E4M3 (DATA_W=8).
#
# Pipeline per token:
#   1. Receive attention output (RN) from CPU   → activ[2048]  FP8
#   2. Router selects TOP_K experts             → pick first K (simulated)
#   3. Per expert: gate → SiLU → up x gate → down → [2048] FP8
#   4. Merge: sum of TOP_K outputs + shared experts → [2048] FP8
#   5. FP8 encode for output
#
# Weight storage (per expert, ~3.0 MB in FP8):
#   gate_w : [HIDDEN, INTER] = [2048, 1408]  ~2.88 M params
#   up_w   : [HIDDEN, INTER] = [2048, 1408]  ~2.88 M params
#   down_w : [INTER, HIDDEN] = [1408, 2048]  ~2.88 M params
# =============================================================================

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Support both direct execution and package import
try:
    from .fp8_e4m3 import FP8_E4M3
except ImportError:
    from fp8_e4m3 import FP8_E4M3


# ---------------------------------------------------------------------------
# Vectorised FP8 array conversion (wraps FP8_E4M3 scalar static methods)
# ---------------------------------------------------------------------------
_E4M3 = FP8_E4M3

# Precompiled vectorised ufuncs for per-element encode / decode
_v_decode = np.vectorize(_E4M3.decode, otypes=[np.float32])
_v_encode = np.vectorize(_E4M3.encode, otypes=[np.uint8])


def fp8_to_float(arr: np.ndarray) -> np.ndarray:
    """Convert uint8 ndarray of FP8 E4M3 bytes → float32 (same shape)."""
    arr = np.asarray(arr, dtype=np.uint8)
    return _v_decode(arr)


def float_to_fp8(arr: np.ndarray) -> np.ndarray:
    """Convert float32 ndarray → uint8 FP8 E4M3 bytes (same shape)."""
    arr = np.asarray(arr, dtype=np.float32)
    return _v_encode(arr)


def pack_fp8_scalar(v: float) -> np.uint8:
    """Single float → single FP8 byte."""
    return np.uint8(_E4M3.encode(v))


def unpack_fp8_scalar(b: int) -> float:
    """Single FP8 byte → float."""
    return _E4M3.decode(b)


# =============================================================================
# FFNPipeline
# =============================================================================

class FFNPipeline:
    """DeepSeek V2-Lite FFN compute pipeline (Python reference).

    V2-Lite: 16B parameters, 27 layers, 66 experts (64 routed + 2 shared).

    Activations travel as FP8 E4M3 bytes (uint8) to match hardware wire format.
    Internal computation uses float32 for golden-model precision.
    Weights are stored as FP8 bytes (uint8) and lazily cached as float32.
    """

    def __init__(
        self,
        hidden: int = 2048,
        inter: int = 1408,
        num_experts: int = 66,
        top_k: int = 6,
    ):
        """Initialise the V2-Lite FFN pipeline.

        Args:
            hidden:      hidden dimension (2048 for V2-Lite).
            inter:       intermediate dimension (1408 for V2-Lite).
            num_experts: total experts (66 = 64 routed + 2 shared).
            top_k:       number of routed experts activated per token.
        """
        self.hidden = hidden
        self.inter = inter
        self.num_experts = num_experts       # 64 routed + 2 shared
        self.top_k = top_k

        # FP8 weight storage (uint8) — matches HBM2 layout
        self.expert_weights: Dict[int, Dict[str, np.ndarray]] = {}
        # Shared experts (always active)
        self.shared_experts: List[Dict[str, np.ndarray]] = []

        # Float32 weight cache (lazy, populated on first forward() call)
        self._expert_f32: Dict[int, Dict[str, np.ndarray]] = {}
        self._shared_f32_list: List[Dict[str, np.ndarray]] = []

        # Router: optional [HIDDEN, NUM_EXPERTS-2] FP8 weight matrix
        self.router_w: Optional[np.ndarray] = None
        self._router_w_f32: Optional[np.ndarray] = None

        # Bookkeeping
        self.last_timing_ms: float = 0.0
        self.last_selected_experts: List[int] = []
        self._last_output_f32: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_expert_weights(
        self,
        expert_id: int,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> None:
        """Load FP8 weights for one routed expert.

        Args:
            expert_id:  0-based index [0 .. num_experts-3] (routed experts).
            gate_w:     uint8 [HIDDEN, INTER] — FP8 E4M3 gate projection.
            up_w:       uint8 [HIDDEN, INTER] — FP8 E4M3 up projection.
            down_w:     uint8 [INTER, HIDDEN] — FP8 E4M3 down projection.
        """
        self._check_expert_id(expert_id)
        self._check_shape(gate_w, (self.hidden, self.inter), 'gate_w')
        self._check_shape(up_w,   (self.hidden, self.inter), 'up_w')
        self._check_shape(down_w, (self.inter, self.hidden), 'down_w')

        self.expert_weights[expert_id] = {
            'gate_w': np.asarray(gate_w, dtype=np.uint8),
            'up_w':   np.asarray(up_w,   dtype=np.uint8),
            'down_w': np.asarray(down_w, dtype=np.uint8),
        }
        self._expert_f32.pop(expert_id, None)

    def load_shared_expert(
        self,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> None:
        """Load FP8 weights for one shared expert (always active).

        V2-Lite has 2 shared experts. Call this twice to load both.
        """
        self._check_shape(gate_w, (self.hidden, self.inter), 'gate_w')
        self._check_shape(up_w,   (self.hidden, self.inter), 'up_w')
        self._check_shape(down_w, (self.inter, self.hidden), 'down_w')

        self.shared_experts.append({
            'gate_w': np.asarray(gate_w, dtype=np.uint8),
            'up_w':   np.asarray(up_w,   dtype=np.uint8),
            'down_w': np.asarray(down_w, dtype=np.uint8),
        })
        self._shared_f32_list.append(None)

    def load_router_weights(self, router_w: np.ndarray) -> None:
        """(Optional) Load router weight matrix [HIDDEN, NUM_EXPERTS-2] in FP8."""
        n_routed = self.num_experts - 2
        self._check_shape(router_w, (self.hidden, n_routed), 'router_w')
        self.router_w = np.asarray(router_w, dtype=np.uint8)
        self._router_w_f32 = None

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        """SiLU (Sigmoid Linear Unit): x * sigmoid(x).

        Numerically stable split-path:
          x >= 0:  x / (1 + exp(-x))
          x <  0:  x * exp(x) / (1 + exp(x))
        """
        x = np.asarray(x, dtype=np.float32)
        out = np.empty_like(x)
        mask_neg = x < 0
        out[~mask_neg] = x[~mask_neg] / (1.0 + np.exp(-x[~mask_neg]))
        exp_x = np.exp(x[mask_neg])
        out[mask_neg] = x[mask_neg] * exp_x / (1.0 + exp_x)
        return out

    # ------------------------------------------------------------------
    # Projection primitives (float32)
    # ------------------------------------------------------------------

    @staticmethod
    def gate_projection(activ: np.ndarray, gate_w: np.ndarray) -> np.ndarray:
        """activ[HIDDEN] @ gate_w[HIDDEN, INTER] → gate_out[INTER]."""
        return activ @ gate_w

    @staticmethod
    def up_projection(activ: np.ndarray, up_w: np.ndarray) -> np.ndarray:
        """activ[HIDDEN] @ up_w[HIDDEN, INTER] → up_out[INTER]."""
        return activ @ up_w

    @staticmethod
    def down_projection(combined: np.ndarray, down_w: np.ndarray) -> np.ndarray:
        """combined[INTER] @ down_w[INTER, HIDDEN] → down_out[HIDDEN]."""
        return combined @ down_w

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def _select_experts(self, activ_f32: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Select TOP_K routed experts. Returns (expert_ids, routing_weights)."""
        if self.router_w is not None:
            if self._router_w_f32 is None:
                self._router_w_f32 = fp8_to_float(self.router_w)
            logits = activ_f32 @ self._router_w_f32
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs /= probs.sum()
            top_idx = np.argpartition(-probs, min(self.top_k, len(probs)))[:self.top_k]
            top_scores = probs[top_idx]
            top_scores /= top_scores.sum()
            order = np.argsort(-top_scores)
            return top_idx[order].tolist(), top_scores[order]
        else:
            available = sorted(self.expert_weights.keys())
            expert_ids = available[:self.top_k]
            n = len(expert_ids)
            route_w = np.full(n, 1.0 / n, dtype=np.float32) if n > 0 else np.array([], dtype=np.float32)
            return expert_ids, route_w

    # ------------------------------------------------------------------
    # Single-expert forward (float32)
    # ------------------------------------------------------------------

    def _expert_forward(
        self, activ_f32: np.ndarray, w_f32: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """gate → SiLU → up x gate → down for one expert. Returns [HIDDEN] f32."""
        gate = self.gate_projection(activ_f32, w_f32['gate_w'])
        gate_act = self.silu(gate)
        up = self.up_projection(activ_f32, w_f32['up_w'])
        combined = gate_act * up
        return self.down_projection(combined, w_f32['down_w'])

    # ------------------------------------------------------------------
    # Weight cache
    # ------------------------------------------------------------------

    def _get_shared_f32_list(self) -> List[Dict[str, np.ndarray]]:
        """Return decoded float32 weights for all shared experts."""
        result = []
        for i, sh in enumerate(self.shared_experts):
            if i < len(self._shared_f32_list) and self._shared_f32_list[i] is not None:
                result.append(self._shared_f32_list[i])
            else:
                decoded = {k: fp8_to_float(v) for k, v in sh.items()}
                if i < len(self._shared_f32_list):
                    self._shared_f32_list[i] = decoded
                else:
                    self._shared_f32_list.append(decoded)
                result.append(decoded)
        return result

    def _get_expert_f32(self, expert_id: int) -> Dict[str, np.ndarray]:
        if expert_id in self._expert_f32:
            return self._expert_f32[expert_id]
        if expert_id not in self.expert_weights:
            raise KeyError(
                f"Expert {expert_id} not loaded. "
                f"Call load_expert_weights({expert_id}) first.")
        self._expert_f32[expert_id] = {
            k: fp8_to_float(v) for k, v in self.expert_weights[expert_id].items()
        }
        return self._expert_f32[expert_id]

    def invalidate_cache(self) -> None:
        """Clear float32 weight cache (after updating FP8 weights in-place)."""
        self._expert_f32.clear()
        self._shared_f32_list.clear()
        self._shared_f32_list = [None] * len(self.shared_experts)
        self._router_w_f32 = None

    # ------------------------------------------------------------------
    # Main forward pass
    # ------------------------------------------------------------------

    def forward(self, activ_fp8: np.ndarray) -> np.ndarray:
        """Full FFN forward pass for one token.

        Args:
            activ_fp8:  uint8 [HIDDEN] — attention output (RN) from CPU,
                        encoded as FP8 E4M3 bytes.

        Returns:
            ffn_out_fp8: uint8 [HIDDEN] — FFN output, FP8 E4M3 encoded,
                          ready for PCIe TX to CPU.
        """
        t0 = time.perf_counter()

        activ = np.asarray(activ_fp8, dtype=np.uint8)
        if activ.shape != (self.hidden,):
            raise ValueError(
                f"Expected activ_fp8 shape ({self.hidden},), got {activ.shape}")

        # FP8 → float32
        activ_f32 = fp8_to_float(activ)

        # Shared experts (always active)
        ffn_f32 = np.zeros(self.hidden, dtype=np.float32)
        for sh_f32 in self._get_shared_f32_list():
            sh_out = self._expert_forward(activ_f32, sh_f32)
            ffn_f32 += sh_out

        # Router
        expert_ids, route_w = self._select_experts(activ_f32)
        self.last_selected_experts = expert_ids

        # Routed experts
        for eid, rw in zip(expert_ids, route_w):
            e_f32 = self._get_expert_f32(eid)
            e_out = self._expert_forward(activ_f32, e_f32)
            np.add(ffn_f32, rw * e_out, out=ffn_f32)

        # float32 → FP8
        ffn_out_fp8 = float_to_fp8(ffn_f32)

        self._last_output_f32 = ffn_f32
        self.last_timing_ms = (time.perf_counter() - t0) * 1000.0
        return ffn_out_fp8

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_output(self, expected: np.ndarray, tolerance: float = 1e-3) -> bool:
        """Verify last forward() output against expected tensor.

        Args:
            expected:   uint8 [HIDDEN] FP8 bytes, or float32 [HIDDEN].
            tolerance:  relative tolerance for float comparison.

        Returns:
            True if max relative error <= tolerance.

        Raises:
            RuntimeError if forward() has not been called yet.
            ValueError if shapes mismatch or NaN/inf in expected.
        """
        if self._last_output_f32 is None:
            raise RuntimeError(
                "No forward() output to verify. Call forward() first.")

        expected = np.asarray(expected)
        if expected.shape != (self.hidden,):
            raise ValueError(
                f"Expected shape ({self.hidden},), got {expected.shape}")

        exp_f32 = expected if expected.dtype == np.float32 else fp8_to_float(expected)

        if not np.all(np.isfinite(exp_f32)):
            raise ValueError("Expected tensor contains NaN or Inf values.")

        actual_f32 = self._last_output_f32
        abs_err = np.abs(actual_f32 - exp_f32)
        denom = np.maximum(np.abs(exp_f32), 1e-8)
        max_rel_err = float(np.max(abs_err / denom))
        return max_rel_err <= tolerance

    @staticmethod
    def compare_outputs(
        actual: np.ndarray,
        expected: np.ndarray,
        tolerance: float = 1e-3,
    ) -> Dict[str, float]:
        """Compare two FFN output tensors (FP8 bytes or float32).

        Returns dict with max_abs_err, mean_abs_err, max_rel_err, match.
        """
        a = actual if actual.dtype == np.float32 else fp8_to_float(actual)
        b = expected if expected.dtype == np.float32 else fp8_to_float(expected)

        abs_err = np.abs(a - b)
        denom = np.maximum(np.abs(b), 1e-8)

        return {
            'max_abs_err':  float(np.max(abs_err)),
            'mean_abs_err': float(np.mean(abs_err)),
            'max_rel_err':  float(np.max(abs_err / denom)),
            'match':        float(np.max(abs_err / denom)) <= tolerance,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_expert_id(self, expert_id: int) -> None:
        max_id = self.num_experts - 3  # 64 routed experts: 0..63
        if not (0 <= expert_id <= max_id):
            raise ValueError(
                f"expert_id {expert_id} out of range [0, {max_id}]")

    @staticmethod
    def _check_shape(arr: np.ndarray, expected: Tuple[int, int], name: str) -> None:
        arr = np.asarray(arr)
        if arr.shape != expected:
            raise ValueError(
                f"{name}: expected {expected}, got {arr.shape}")

    @property
    def num_loaded_experts(self) -> int:
        """Number of routed experts with weights loaded."""
        return len(self.expert_weights)

    @property
    def is_ready(self) -> bool:
        """True when at least TOP_K routed experts + shared experts are loaded."""
        return (self.num_loaded_experts >= self.top_k and
                len(self.shared_experts) > 0)

    def __repr__(self) -> str:
        return (
            f"FFNPipeline(hidden={self.hidden}, inter={self.inter}, "
            f"experts={self.num_experts} ({self.num_experts - 2}r + 2sh), "
            f"top_k={self.top_k}, "
            f"loaded={self.num_loaded_experts}, shared_loaded={len(self.shared_experts)}, "
            f"ready={self.is_ready})"
        )


# =============================================================================
# Synthetic weight builders
# =============================================================================

def make_random_fp8_weights(shape: Tuple[int, int], seed: int = 42) -> np.ndarray:
    """Create a random FP8 weight matrix (uint8 bytes).

    Args:
        shape: (rows, cols).
        seed:  RNG seed.

    Returns:
        uint8 ndarray, FP8 E4M3 encoded.
    """
    rng = np.random.RandomState(seed)
    f32 = rng.randn(*shape).astype(np.float32) * 0.5
    return float_to_fp8(f32)


def make_identity_fp8_weights(shape: Tuple[int, int]) -> np.ndarray:
    """FP8 weight matrix with 1.0 on diagonal, 0 elsewhere. For path checks."""
    f32 = np.zeros(shape, dtype=np.float32)
    n = min(shape[0], shape[1])
    for i in range(n):
        f32[i, i] = 1.0
    return float_to_fp8(f32)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 64)
    print("V2-Lite FFN Pipeline — Self-Test Suite")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Test 1: FP8 round-trip sanity
    # ------------------------------------------------------------------
    print("\n[Test 1] FP8 E4M3 round-trip...")
    test_vals = np.array(
        [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -3.5, 120.0, -120.0],
        dtype=np.float32,
    )
    fp8_bytes = float_to_fp8(test_vals)
    recovered = fp8_to_float(fp8_bytes)
    errs = np.abs(test_vals - recovered)
    max_err = float(np.max(errs))
    print(f"  Input:      {test_vals}")
    print(f"  Recovered:  {np.round(recovered, 3)}")
    print(f"  Max error:  {max_err:.4f}")
    assert unpack_fp8_scalar(pack_fp8_scalar(0.0)) == 0.0
    assert unpack_fp8_scalar(pack_fp8_scalar(float('nan'))) == 0.0  # NaN→0
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 2: Tiny model (H=16, I=8, K=2)
    # ------------------------------------------------------------------
    print("\n[Test 2] Tiny model forward pass (H=16, I=8, K=2)...")
    H, I, N, K = 16, 8, 5, 2

    pipe = FFNPipeline(hidden=H, inter=I, num_experts=N, top_k=K)

    for eid in range(3):
        pipe.load_expert_weights(
            eid,
            gate_w=make_random_fp8_weights((H, I), seed=100 + eid),
            up_w=make_random_fp8_weights((H, I), seed=200 + eid),
            down_w=make_random_fp8_weights((I, H), seed=300 + eid),
        )

    pipe.load_shared_expert(
        gate_w=make_random_fp8_weights((H, I), seed=400),
        up_w=make_random_fp8_weights((H, I), seed=500),
        down_w=make_random_fp8_weights((I, H), seed=600),
    )

    activ_fp8 = float_to_fp8(np.full(H, 0.1, dtype=np.float32))
    out_fp8 = pipe.forward(activ_fp8)
    out_f32 = fp8_to_float(out_fp8)

    print(f"  Pipeline          : {pipe}")
    print(f"  Output shape      : {out_fp8.shape}")
    print(f"  Output dtype      : {out_fp8.dtype}")
    print(f"  Selected experts  : {pipe.last_selected_experts}")
    print(f"  Has NaN           : {np.any(np.isnan(out_f32))}")
    print(f"  |max|             : {np.max(np.abs(out_f32)):.6f}")
    print(f"  Timing            : {pipe.last_timing_ms:.3f} ms")

    assert out_fp8.shape == (H,), f"Bad shape: {out_fp8.shape}"
    assert not np.any(np.isnan(out_f32)), "Output has NaN"
    assert np.max(np.abs(out_f32)) > 0, "Output is all zeros"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 3: Output varies with different inputs
    # ------------------------------------------------------------------
    print("\n[Test 3] Output varies with different inputs...")
    out2_fp8 = pipe.forward(float_to_fp8(np.full(H, 0.5, dtype=np.float32)))
    diff = np.max(np.abs(fp8_to_float(out_fp8) - fp8_to_float(out2_fp8)))
    print(f"  Max delta (0.1 vs 0.5): {diff:.6f}")
    assert diff > 0, "Output unchanged for different input"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 4: Small model (H=256, I=128, K=6)
    # ------------------------------------------------------------------
    print("\n[Test 4] Small model forward pass (H=256, I=128, K=6)...")
    H_s, I_s, N_s, K_s = 256, 128, 9, 6

    pipe_s = FFNPipeline(hidden=H_s, inter=I_s, num_experts=N_s, top_k=K_s)

    for eid in range(N_s - 2):  # 7 routed experts (indices 0-6)
        pipe_s.load_expert_weights(
            eid,
            gate_w=make_random_fp8_weights((H_s, I_s), seed=1000 + eid),
            up_w=make_random_fp8_weights((H_s, I_s), seed=2000 + eid),
            down_w=make_random_fp8_weights((I_s, H_s), seed=3000 + eid),
        )

    # Load 2 shared experts
    pipe_s.load_shared_expert(
        gate_w=make_random_fp8_weights((H_s, I_s), seed=4000),
        up_w=make_random_fp8_weights((H_s, I_s), seed=4001),
        down_w=make_random_fp8_weights((I_s, H_s), seed=4002),
    )
    pipe_s.load_shared_expert(
        gate_w=make_random_fp8_weights((H_s, I_s), seed=5000),
        up_w=make_random_fp8_weights((H_s, I_s), seed=5001),
        down_w=make_random_fp8_weights((I_s, H_s), seed=5002),
    )

    print(f"  Pipeline: {pipe_s}")

    activ_s = float_to_fp8(np.random.RandomState(42).randn(H_s).astype(np.float32) * 0.5)
    out_s_fp8 = pipe_s.forward(activ_s)
    out_s_f32 = fp8_to_float(out_s_fp8)

    print(f"  Output shape      : {out_s_fp8.shape}")
    print(f"  Selected experts  : {pipe_s.last_selected_experts}")
    print(f"  Has NaN           : {np.any(np.isnan(out_s_f32))}")
    print(f"  |max|             : {np.max(np.abs(out_s_f32)):.6f}")
    print(f"  Timing            : {pipe_s.last_timing_ms:.3f} ms")

    assert out_s_fp8.shape == (H_s,), f"Bad shape: {out_s_fp8.shape}"
    assert not np.any(np.isnan(out_s_f32)), "Output has NaN"
    assert np.max(np.abs(out_s_f32)) > 0, "Output is all zeros"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 5: Production-scale (H=2048, I=1408, K=6)
    # ------------------------------------------------------------------
    print("\n[Test 5] Production-scale (H=2048, I=1408, K=6)...")
    big = FFNPipeline(hidden=2048, inter=1408, num_experts=66, top_k=6)
    print(f"  {big}")

    big.load_expert_weights(0,
        make_random_fp8_weights((2048, 1408), seed=1),
        make_random_fp8_weights((2048, 1408), seed=2),
        make_random_fp8_weights((1408, 2048), seed=3),
    )
    big.load_expert_weights(1,
        make_random_fp8_weights((2048, 1408), seed=4),
        make_random_fp8_weights((2048, 1408), seed=5),
        make_random_fp8_weights((1408, 2048), seed=6),
    )
    big.load_expert_weights(2,
        make_random_fp8_weights((2048, 1408), seed=7),
        make_random_fp8_weights((2048, 1408), seed=8),
        make_random_fp8_weights((1408, 2048), seed=9),
    )
    big.load_shared_expert(
        make_random_fp8_weights((2048, 1408), seed=10),
        make_random_fp8_weights((2048, 1408), seed=11),
        make_random_fp8_weights((1408, 2048), seed=12),
    )
    big.load_shared_expert(
        make_random_fp8_weights((2048, 1408), seed=13),
        make_random_fp8_weights((2048, 1408), seed=14),
        make_random_fp8_weights((1408, 2048), seed=15),
    )

    mem_mb = (3 + 2) * (2048 * 1408 * 3) / 1e6
    print(f"  Weight memory (3 routed + 2 shared): {mem_mb:.0f} MB FP8")
    print(f"  Full model 66 experts: {66 * 2048 * 1408 * 3 / 1e9:.1f} GB FP8")

    activ = float_to_fp8(np.full(2048, 0.01, dtype=np.float32))
    t0 = time.perf_counter()
    out = big.forward(activ)
    t_ms = (time.perf_counter() - t0) * 1000.0
    out_f32 = fp8_to_float(out)

    assert out.shape == (2048,), f"Bad shape: {out.shape}"
    assert not np.any(np.isnan(out_f32)), "Production output has NaN"
    assert np.max(np.abs(out_f32)) < 1000.0, "Output overflow"
    print(f"  Output |max|  : {np.max(np.abs(out_f32)):.4f}")
    print(f"  Timing        : {t_ms:.1f} ms")
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 6: Identity weights — path correctness
    # ------------------------------------------------------------------
    print("\n[Test 6] Identity weight propagation (H=4, I=4, K=1)...")
    ipipe = FFNPipeline(hidden=4, inter=4, num_experts=4, top_k=1)

    i_gate = make_identity_fp8_weights((4, 4))
    i_up = make_identity_fp8_weights((4, 4))
    i_down = make_identity_fp8_weights((4, 4))

    ipipe.load_expert_weights(0, i_gate, i_up, i_down)
    ipipe.load_shared_expert(i_gate, i_up, i_down)

    inp_fp8 = float_to_fp8(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    out_i = fp8_to_float(ipipe.forward(inp_fp8))

    print(f"  Input :  [1.0, 2.0, 3.0, 4.0]")
    print(f"  Output:  {np.round(out_i, 3)}")
    assert np.all(out_i >= -1e-6), "Identity test: expected non-negative output"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Test 7: Router with learned weights
    # ------------------------------------------------------------------
    print("\n[Test 7] Router with learned weights (H=8, I=4, experts=5, K=2)...")
    rpipe = FFNPipeline(hidden=8, inter=4, num_experts=5, top_k=2)

    n_routed = 3  # 5 total - 2 shared
    for eid in range(n_routed):
        rpipe.load_expert_weights(eid,
            make_random_fp8_weights((8, 4), seed=700 + eid),
            make_random_fp8_weights((8, 4), seed=800 + eid),
            make_random_fp8_weights((4, 8), seed=900 + eid),
        )
    rpipe.load_shared_expert(
        make_random_fp8_weights((8, 4), seed=1000),
        make_random_fp8_weights((8, 4), seed=1100),
        make_random_fp8_weights((4, 8), seed=1200),
    )

    # Router bias towards expert 1
    router_w_f32 = np.zeros((8, n_routed), dtype=np.float32)
    router_w_f32[:, 1] = 10.0
    rpipe.load_router_weights(float_to_fp8(router_w_f32))

    activ = float_to_fp8(np.ones(8, dtype=np.float32) * 0.1)
    rout = rpipe.forward(activ)
    sel = rpipe.last_selected_experts
    print(f"  Selected experts: {sel}")
    assert 1 in sel, f"Expert 1 should be selected, got {sel}"
    print("  [PASS]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("All tests PASSED.")
    print(f"  Tiny model  (H=16,  I=8)       : {pipe.last_timing_ms:.2f} ms/token")
    print(f"  Small model (H=256, I=128)     : {pipe_s.last_timing_ms:.1f} ms/token")
    print(f"  Full model  (H=2048, I=1408)   : {big.last_timing_ms:.0f} ms/token")
    print("=" * 64)
