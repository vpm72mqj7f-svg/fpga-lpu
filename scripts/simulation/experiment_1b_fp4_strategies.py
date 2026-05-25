#!/usr/bin/env python3
"""
Experiment 1B: fp4 precision strategy sweep.

Goal: raise production-size Expert FFN cosine similarity from ~0.992 to >=0.995.

Strategies tested:
  1. Smaller group size: 128 / 64 / 32 / 16
  2. SmoothQuant alpha: scale = (median_rms / col_rms) ** alpha
  3. Outlier fallback: keep top quantization-error input channels in fp8/fp32

This is still a functional NumPy model. The fallback strategy estimates the
hardware cost by reporting the fraction of columns that must bypass fp4.
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp4_utils import quantize_fp4_e2m1, dequantize_fp4_e2m1, compute_output_diff
from experiment_1_fp4_precision import _make_weights_with_outliers


@dataclass
class StrategyResult:
    name: str
    group_size: int
    alpha: float
    fallback_ratio: float
    mean_cosine: float
    min_cosine: float
    mean_relative_error: float
    gate_mse: float
    up_mse: float
    down_mse: float
    fallback_cols_gate: int
    fallback_cols_up: int
    fallback_cols_down: int
    pass_995: bool


def smooth_weights(W: np.ndarray, x: np.ndarray | None, alpha: float) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Partial per-input-channel smoothing.

    alpha=0: no smoothing.
    alpha=1: fully equalize column RMS to median.
    """
    if alpha <= 0:
        return W.copy(), None if x is None else x.copy(), np.ones(W.shape[1], dtype=np.float32)
    col_rms = np.sqrt(np.mean(W ** 2, axis=0))
    target = np.maximum(np.median(col_rms), 1e-8)
    scale = (target / np.maximum(col_rms, 1e-8)) ** alpha
    # Clamp to avoid exploding activations.
    scale = np.clip(scale, 0.05, 20.0).astype(np.float32)
    W_s = W * scale[None, :]
    if x is not None:
        x_s = x / (scale[None, :] + 1e-12)
        return W_s, x_s, scale
    return W_s, None, scale


def fp4_weight(W: np.ndarray, group_size: int) -> Tuple[np.ndarray, float, np.ndarray]:
    idx, sc = quantize_fp4_e2m1(W, group_size=group_size)
    Wq = dequantize_fp4_e2m1(idx, sc, group_size=group_size)
    err = (Wq - W) ** 2
    rel_mse = float(np.mean(err) / np.maximum(np.mean(W ** 2), 1e-12))
    col_err = np.mean(err, axis=0) / np.maximum(np.mean(W ** 2, axis=0), 1e-12)
    return Wq, rel_mse, col_err


def fp4_linear(W: np.ndarray, x: np.ndarray, group_size: int,
               fallback_ratio: float = 0.0) -> Tuple[np.ndarray, float, int]:
    """Linear layer with fp4 weight and optional outlier input-channel fallback.

    Fallback replaces selected input columns in the dequantized weight with the
    original float weight. This approximates storing those columns in fp8/fp16.
    """
    Wq, mse, col_err = fp4_weight(W, group_size)
    k = int(round(W.shape[1] * fallback_ratio))
    if k > 0:
        idx = np.argpartition(-col_err, k - 1)[:k]
        Wq[:, idx] = W[:, idx]
    else:
        idx = []
    y = x @ Wq.T
    return y, mse, len(idx)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def run_strategy(hidden_size: int, intermediate_size: int, num_tokens: int,
                 group_size: int, alpha: float, fallback_ratio: float,
                 seed: int = 42) -> StrategyResult:
    rng = np.random.RandomState(seed)
    gate_W = _make_weights_with_outliers((intermediate_size, hidden_size))
    up_W = _make_weights_with_outliers((intermediate_size, hidden_size))
    down_W = _make_weights_with_outliers((hidden_size, intermediate_size))
    x = rng.randn(num_tokens, hidden_size).astype(np.float32)

    # Reference
    gate_ref = x @ gate_W.T
    up_ref = x @ up_W.T
    hidden_ref = silu(gate_ref) * up_ref
    out_ref = hidden_ref @ down_W.T

    # Smooth gate and up with shared input activation x.
    gate_W_s, x_gate, _ = smooth_weights(gate_W, x, alpha)
    up_W_s, x_up, _ = smooth_weights(up_W, x, alpha)

    gate_q, gate_mse, gate_fb = fp4_linear(gate_W_s, x_gate, group_size, fallback_ratio)
    up_q, up_mse, up_fb = fp4_linear(up_W_s, x_up, group_size, fallback_ratio)
    hidden_q = silu(gate_q) * up_q

    # Down consumes hidden_q; smooth/down fallback independently.
    down_W_s, hidden_s, _ = smooth_weights(down_W, hidden_q, alpha)
    out_q, down_mse, down_fb = fp4_linear(down_W_s, hidden_s, group_size, fallback_ratio)

    diff = compute_output_diff(out_q, out_ref)
    name = f"g{group_size}_a{alpha:.2f}_fb{fallback_ratio:.3f}"
    return StrategyResult(
        name=name,
        group_size=group_size,
        alpha=alpha,
        fallback_ratio=fallback_ratio,
        mean_cosine=diff['mean_cosine'],
        min_cosine=diff['min_cosine'],
        mean_relative_error=diff['mean_relative_error'],
        gate_mse=gate_mse,
        up_mse=up_mse,
        down_mse=down_mse,
        fallback_cols_gate=gate_fb,
        fallback_cols_up=up_fb,
        fallback_cols_down=down_fb,
        pass_995=diff['mean_cosine'] >= 0.995,
    )


def run_sweep(hidden_size: int = 7168, intermediate_size: int = 3072,
              num_tokens: int = 128, seed: int = 42) -> List[StrategyResult]:
    configs = []
    for group_size in (128, 64, 32, 16):
        for alpha in (0.0, 0.5, 0.75, 1.0, 1.25):
            # fallback 0 means pure fp4. Add fallback ratios only for more promising group sizes.
            for fb in (0.0, 0.005, 0.01, 0.02, 0.05):
                configs.append((group_size, alpha, fb))

    results: List[StrategyResult] = []
    for i, (g, a, fb) in enumerate(configs, 1):
        print(f"[{i:03d}/{len(configs)}] group={g:3d} alpha={a:4.2f} fallback={fb:5.3f}", flush=True)
        res = run_strategy(hidden_size, intermediate_size, num_tokens, g, a, fb, seed)
        results.append(res)
        print(f"    cosine={res.mean_cosine:.6f}  rel_err={res.mean_relative_error:.5f}  PASS={res.pass_995}", flush=True)
    return sorted(results, key=lambda r: r.mean_cosine, reverse=True)


def main():
    results = run_sweep()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'docs')
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'fp4_strategy_sweep_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    print("\n=== TOP 15 STRATEGIES ===")
    print(f"{'rank':>4} {'name':<22} {'cosine':>10} {'min':>10} {'rel_err':>10} {'fallback':>9} {'PASS':>6}")
    for i, r in enumerate(results[:15], 1):
        print(f"{i:>4} {r.name:<22} {r.mean_cosine:>10.6f} {r.min_cosine:>10.6f} "
              f"{r.mean_relative_error:>10.5f} {r.fallback_ratio:>9.3%} {str(r.pass_995):>6}")

    best = results[0]
    print("\nBEST:")
    print(json.dumps(asdict(best), indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")

    return 0 if best.pass_995 else 1


if __name__ == '__main__':
    raise SystemExit(main())
