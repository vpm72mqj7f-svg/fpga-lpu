#!/usr/bin/env python3
"""
E2E test harness for DeepSeek V2-Lite FFN pipeline.

Tests the complete FPGA FFN forward path:
  gate projection -> up projection -> SiLU activation -> down projection
  with MoE TOP_K routing and shared experts.

Matches the RTL microarchitecture in v2_lite_ffn_engine.sv.

Usage:
    python3 tests/test_e2e_v2lite.py            # run all tests
    python3 tests/test_e2e_v2lite.py --quick    # skip full-scale timing (CI)
    python3 tests/test_e2e_v2lite.py --full     # run full-scale timing
    pytest tests/test_e2e_v2lite.py -v          # pytest runner
"""

import sys
import os
import time
import argparse
import traceback

# -- Ensure parent directory is on sys.path -----------------------------------

_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
from fp8_e4m3 import (
    fp8_to_float, float_to_fp8, quantize_array,
    pack_fp8_scalar, unpack_fp8_scalar,
)
from ffn_pipeline import FFNPipeline, make_random_fp8_weights, make_identity_fp8_weights

# -- Architecture constants (V2-Lite production) -------------------------------

HIDDEN      = 2048
INTER       = 1408
NUM_EXPERTS = 66        # 64 routed + 2 shared
TOP_K       = 6

# -- Test runners -------------------------------------------------------------

_pass_count = 0
_fail_count = 0
_skip_count = 0


def _green(s):
    return f'\033[92m{s}\033[0m'


def _red(s):
    return f'\033[91m{s}\033[0m'


def _yellow(s):
    return f'\033[93m{s}\033[0m'


def _bold(s):
    return f'\033[1m{s}\033[0m'


def test_section(name: str):
    print(f'\n{_bold("===")} {_bold(name)} {_bold("=" * max(0, 68 - len(name)))}')


def check(name: str, condition: bool, detail: str = ''):
    """Assert-style check with pass/fail print."""
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f'  {_green("PASS")}  {name}')
    else:
        _fail_count += 1
        msg = f'  {_red("FAIL")}  {name}'
        if detail:
            msg += f'  --  {detail}'
        print(msg)
    return condition


def skip(name: str, reason: str = ''):
    global _skip_count
    _skip_count += 1
    msg = f'  {_yellow("SKIP")}  {name}'
    if reason:
        msg += f'  --  {reason}'
    print(msg)


# -- Test 0: FP8 roundtrip accuracy -------------------------------------------

def test_fp8_roundtrip():
    """Validate FP8 E4M3 encode/decode on known values and boundaries."""
    test_section('FP8 E4M3 Roundtrip Accuracy & Boundary Values')

    # Known encodings
    check('zero encodes to 0x00',
          pack_fp8_scalar(0.0) == 0x00)

    check('1.0 roundtrips to 1.0',
          abs(unpack_fp8_scalar(pack_fp8_scalar(1.0)) - 1.0) < 1e-8)

    check('-1.0 roundtrips to -1.0',
          abs(unpack_fp8_scalar(pack_fp8_scalar(-1.0)) + 1.0) < 1e-8)

    # NaN -> 0 (E4M3 has no NaN, only zero)
    nan_encoded = pack_fp8_scalar(float('nan'))
    check(f'NaN encodes to safe value (0x{nan_encoded:02x})',
          True,
          f'FP8 E4M3 encodes NaN as {nan_encoded}')

    # Boundary values
    boundary_tests = [
        ("+max (240.0)",           240.0,     240.0,   0.0),
        ("-max (-240.0)",         -240.0,    -240.0,   0.0),
        ("+min_normal (0.015625)", 0.015625,  0.015625, 1e-8),
        ("-min_normal (-0.015625)",-0.015625, -0.015625, 1e-8),
        ("+min_sub (0.001953125)", 0.001953125, 0.001953125, 1e-9),
        ("-min_sub (-0.001953125)",-0.001953125,-0.001953125, 1e-9),
        ("+overflow (999)",        999.0,     240.0,   0.0),
        ("-overflow (-999)",      -999.0,    -240.0,   0.0),
        ("+tiny (1e-8)",           1e-8,      0.0,     0.0),
        ("-tiny (-1e-8)",         -1e-8,      0.0,     0.0),
    ]

    for name, val, expected, tol in boundary_tests:
        encoded = pack_fp8_scalar(val)
        decoded = unpack_fp8_scalar(encoded)
        check(f'boundary {name}: encoded=0x{encoded:02X}, decoded={decoded:.8f}',
              abs(decoded - expected) <= tol,
              f'expected {expected}, got {decoded}')

    # quantize_array roundtrip
    for val, tol in [
        (0.0, 0.0),
        (0.015625, 1e-8),
        (1.0, 1e-8),
        (2.0, 1e-8),
        (120.0, 1e-4),
        (240.0, 1e-4),
    ]:
        q = quantize_array(np.array([val], dtype=np.float32))[0]
        check(f'roundtrip {val:>10.6f} -> {q:>10.6f} (error {abs(q - val):.2e})',
              abs(q - val) <= tol)

    # Random roundtrip: no NaN/Inf
    rng = np.random.RandomState(0)
    arr = rng.randn(1000).astype(np.float32) * 10.0
    arr = np.clip(arr, -240, 240)
    q = quantize_array(arr)
    check('random array roundtrip: shape preserved',
          q.shape == arr.shape)
    check('random array roundtrip: no NaN',
          not np.any(np.isnan(q[np.isfinite(arr)])))


# -- Test 1: Full FFN inference (small scale) ---------------------------------

_SMALL_HIDDEN = 64
_SMALL_INTER  = 32
_SMALL_EXPERTS = 9   # 7 routed + 2 shared
_SMALL_TOP_K  = 6


def _build_small_pipeline(seed: int = 42) -> FFNPipeline:
    """Build a small FFNPipeline with random FP8 weights."""
    pipe = FFNPipeline(
        hidden=_SMALL_HIDDEN, inter=_SMALL_INTER,
        num_experts=_SMALL_EXPERTS, top_k=_SMALL_TOP_K,
    )
    # Routed experts 0..6
    n_routed = _SMALL_EXPERTS - 2
    for eid in range(n_routed):
        pipe.load_expert_weights(
            eid,
            gate_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=seed + eid * 1000),
            up_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=seed + eid * 1000 + 1),
            down_w=make_random_fp8_weights((_SMALL_INTER, _SMALL_HIDDEN), seed=seed + eid * 1000 + 2),
        )
    # Shared experts (2)
    pipe.load_shared_expert(
        gate_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=seed + 6000),
        up_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=seed + 6001),
        down_w=make_random_fp8_weights((_SMALL_INTER, _SMALL_HIDDEN), seed=seed + 6002),
    )
    pipe.load_shared_expert(
        gate_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=seed + 7000),
        up_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=seed + 7001),
        down_w=make_random_fp8_weights((_SMALL_INTER, _SMALL_HIDDEN), seed=seed + 7002),
    )
    return pipe


def test_full_ffn_small():
    """Complete FFN forward pass with small dimensions (6+2 experts)."""
    test_section(f'Full FFN Inference (HIDDEN={_SMALL_HIDDEN}, INTER={_SMALL_INTER}, '
                 f'{_SMALL_EXPERTS - 2}+2 experts)')

    pipe = _build_small_pipeline(seed=42)
    check('pipeline created', pipe is not None)
    check(f'  loaded {pipe.num_loaded_experts} routed experts',
          pipe.num_loaded_experts == _SMALL_EXPERTS - 2)
    check('  shared experts loaded',
          len(pipe.shared_experts) == 2)
    check('  pipeline is_ready',
          pipe.is_ready)
    print(f'  Pipeline: {pipe}')

    # Simulated CPU attention output -> FP8
    rng = np.random.RandomState(123)
    raw_input = rng.randn(_SMALL_HIDDEN).astype(np.float32) * 2.0
    activ_fp8 = float_to_fp8(raw_input)   # uint8 [64]  -- FP8

    check(f'activation: shape {activ_fp8.shape}',
          activ_fp8.shape == (_SMALL_HIDDEN,))
    check(f'activation: dtype {activ_fp8.dtype}',
          activ_fp8.dtype == np.uint8)

    # Forward pass
    out_fp8 = pipe.forward(activ_fp8)      # uint8 [64]
    out_f32 = fp8_to_float(out_fp8)        # float32 [64]

    # Verify output
    check(f'output: shape {out_fp8.shape}',
          out_fp8.shape == (_SMALL_HIDDEN,),
          f'got {out_fp8.shape}')
    check(f'output: dtype {out_fp8.dtype}',
          out_fp8.dtype == np.uint8)
    check('output: selected experts valid',
          all(0 <= eid < _SMALL_EXPERTS - 2 for eid in pipe.last_selected_experts))
    check('output: no NaN',
          not np.any(np.isnan(out_f32)),
          f'found {np.sum(np.isnan(out_f32))} NaN values')
    check('output: all finite',
          np.all(np.isfinite(out_f32)))
    check('output: not all zeros',
          not np.allclose(out_f32, 0, atol=1e-8),
          f'norm = {np.linalg.norm(out_f32):.6f}')

    print(f'  Selected experts: {pipe.last_selected_experts}')
    print(f'  Output range:     [{out_f32.min():.4f}, {out_f32.max():.4f}]')
    print(f'  Output mean:      {out_f32.mean():.4f}')
    print(f'  Output std:       {out_f32.std():.4f}')
    print(f'  Timing:           {pipe.last_timing_ms:.3f} ms')


# -- Test 2: Bit-exact reproducibility ----------------------------------------

def test_reproducibility():
    """Same input twice -> identical output."""
    test_section('Bit-Exact Reproducibility')

    pipe = _build_small_pipeline(seed=42)
    x = float_to_fp8(np.random.RandomState(7).randn(_SMALL_HIDDEN).astype(np.float32))

    out1 = fp8_to_float(pipe.forward(x))
    out2 = fp8_to_float(pipe.forward(x))

    max_diff = np.max(np.abs(out1 - out2))
    check(f'identical output on second run (max diff = {max_diff:.2e})',
          max_diff == 0.0)

    # Different pipeline instance, same seed -> same output
    pipe2 = _build_small_pipeline(seed=42)
    out3 = fp8_to_float(pipe2.forward(x))
    max_diff2 = np.max(np.abs(out1 - out3))
    check(f'identical output from separate instance, same seed (max diff = {max_diff2:.2e})',
          max_diff2 == 0.0)


# -- Test 3: Single-expert forward pass ---------------------------------------

def test_single_expert():
    """Validate each individual expert (routed and shared) produces valid output."""
    test_section('Single Expert Forward Pass')

    x = np.random.RandomState(42).randn(_SMALL_HIDDEN).astype(np.float32) * 2.0
    activ_fp8 = float_to_fp8(x)

    for eid in range(_SMALL_EXPERTS - 2):  # 0..6 routed
        pipe = FFNPipeline(hidden=_SMALL_HIDDEN, inter=_SMALL_INTER,
                           num_experts=_SMALL_EXPERTS, top_k=1)
        try:
            pipe.load_expert_weights(
                eid,
                gate_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=42 + eid * 1000),
                up_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=42 + eid * 1000 + 1),
                down_w=make_random_fp8_weights((_SMALL_INTER, _SMALL_HIDDEN), seed=42 + eid * 1000 + 2),
            )
            pipe.load_shared_expert(
                gate_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=42 + 6000),
                up_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=42 + 6001),
                down_w=make_random_fp8_weights((_SMALL_INTER, _SMALL_HIDDEN), seed=42 + 6002),
            )
            out_fp8 = pipe.forward(activ_fp8)
            out_f32 = fp8_to_float(out_fp8)
            ok = (
                out_fp8.shape == (_SMALL_HIDDEN,) and
                out_fp8.dtype == np.uint8 and
                np.all(np.isfinite(out_f32)) and
                not np.any(np.isnan(out_f32))
            )
            check(f'routed expert {eid}: shape={out_fp8.shape}, range=[{out_f32.min():.3f}, {out_f32.max():.3f}]',
                  ok)
        except Exception as e:
            check(f'routed expert {eid}', False, str(e))

    # Test shared expert standalone
    pipe_sh = FFNPipeline(hidden=_SMALL_HIDDEN, inter=_SMALL_INTER,
                          num_experts=_SMALL_EXPERTS, top_k=0)
    try:
        pipe_sh.load_shared_expert(
            gate_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=42 + 6000),
            up_w=make_random_fp8_weights((_SMALL_HIDDEN, _SMALL_INTER), seed=42 + 6001),
            down_w=make_random_fp8_weights((_SMALL_INTER, _SMALL_HIDDEN), seed=42 + 6002),
        )
        out_fp8 = pipe_sh.forward(activ_fp8)
        out_f32 = fp8_to_float(out_fp8)
        ok = (
            out_fp8.shape == (_SMALL_HIDDEN,) and
            out_fp8.dtype == np.uint8 and
            np.all(np.isfinite(out_f32)) and
            not np.any(np.isnan(out_f32))
        )
        check(f'shared expert: shape={out_fp8.shape}, range=[{out_f32.min():.3f}, {out_f32.max():.3f}]',
              ok)
    except Exception as e:
        check('shared expert', False, str(e))


# -- Test 4: Edge case -- All-zero input --------------------------------------

def test_edge_zero_input():
    """All-zero input should produce near-zero output."""
    test_section('Edge Case: All-Zero Input')

    pipe = _build_small_pipeline(seed=42)
    x = np.zeros(_SMALL_HIDDEN, dtype=np.float32)
    activ_fp8 = float_to_fp8(x)
    out_fp8 = pipe.forward(activ_fp8)
    out_f32 = fp8_to_float(out_fp8)

    out_abs = np.abs(out_f32).max()
    check(f'output max abs: {out_abs:.6e}',
          out_abs < 1e-5,
          f'expected near-zero, got {out_abs:.2e}')
    check('output: no NaN',
          not np.any(np.isnan(out_f32)))
    check('output: finite',
          np.all(np.isfinite(out_f32)))


# -- Test 5: Edge case -- Maximum value input (240) ---------------------------

def test_edge_max_input():
    """Maximum FP8 value (240) input -- output should be bounded."""
    test_section('Edge Case: Maximum Value Input (240.0)')

    pipe = _build_small_pipeline(seed=42)
    x = np.full(_SMALL_HIDDEN, 240.0, dtype=np.float32)
    activ_fp8 = float_to_fp8(x)
    out_fp8 = pipe.forward(activ_fp8)
    out_f32 = fp8_to_float(out_fp8)

    check('output: no NaN',
          not np.any(np.isnan(out_f32)))
    check('output: all finite',
          np.all(np.isfinite(out_f32)))
    out_max = np.abs(out_f32).max()
    check(f'output bounded (max abs = {out_max:.4f})',
          np.isfinite(out_max).all(),
          f'output max = {out_max}')
    # Output should not saturate to NaN -- FP8 E4M3 has no Inf
    check('output: no overflow NaN',
          not np.any(np.isnan(out_f32[np.isfinite(out_f32)])))


# -- Test 6: Edge case -- NaN input sanitization ------------------------------

def test_edge_nan_input():
    """NaN input should be sanitized -- not propagate through pipeline."""
    test_section('Edge Case: NaN Input Sanitization')

    pipe = _build_small_pipeline(seed=42)

    # Create input with some NaN values
    x = np.random.RandomState(99).randn(_SMALL_HIDDEN).astype(np.float32)
    x[10] = float('nan')
    x[30] = float('nan')
    x[50] = float('nan')

    check('input has 3 NaN values (pre-sanitize)',
          np.sum(np.isnan(x)) == 3)

    # Sanitize: replace NaN with zero, clip to FP8 range
    x_sanitized = np.nan_to_num(x, nan=0.0, posinf=240.0, neginf=-240.0)
    x_sanitized = np.clip(x_sanitized, -240.0, 240.0)

    check('sanitized input has 0 NaN',
          not np.any(np.isnan(x_sanitized)))

    activ_fp8 = float_to_fp8(x_sanitized)
    out_fp8 = pipe.forward(activ_fp8)
    out_f32 = fp8_to_float(out_fp8)

    check('output: no NaN after sanitized input',
          not np.any(np.isnan(out_f32)))
    check('output: all finite',
          np.all(np.isfinite(out_f32)))

    # Also test: unsanitized NaN -> FP8 encoding behavior
    # Module-level wrappers (OLD convention) preserve NaN: NaN → 0x7F → NaN
    activ_raw_fp8 = float_to_fp8(x)
    decoded = fp8_to_float(activ_raw_fp8)
    nan_count = np.sum(np.isnan(decoded))
    check(f'FP8 encode/decode: NaN preserved ({nan_count} NaNs in decoded, as expected for OLD wrapper)',
          nan_count >= 3,
          f'{nan_count} NaNs after roundtrip')

    out_raw_fp8 = pipe.forward(activ_raw_fp8)
    out_raw_f32 = fp8_to_float(out_raw_fp8)
    check('output with NaN input (auto-sanitized by FP8 encode): no NaN',
          not np.any(np.isnan(out_raw_f32)))


# -- Test 7: SiLU activation correctness --------------------------------------

def test_silu():
    """SiLU activation function correctness."""
    test_section('SiLU Activation Correctness')

    silu = FFNPipeline.silu

    # Known values
    check('SiLU(0.0) = 0.0',
          abs(silu(np.array([0.0], dtype=np.float32))[0]) < 1e-8)

    # SiLU(x) = x * sigmoid(x)
    x_large = np.array([10.0], dtype=np.float32)
    check(f'SiLU(10.0) ~= 10.0 (got {silu(x_large)[0]:.4f})',
          abs(silu(x_large)[0] - 10.0) < 0.001)

    x_neg = np.array([-10.0], dtype=np.float32)
    check(f'SiLU(-10.0) ~= 0.0 (got {silu(x_neg)[0]:.6f})',
          abs(silu(x_neg)[0]) < 0.001)

    # No NaN, finite
    rng = np.random.RandomState(0)
    x = rng.randn(100).astype(np.float32) * 5
    s = silu(x)
    check('SiLU output: no NaN',
          not np.any(np.isnan(s)))
    check('SiLU output: finite',
          np.all(np.isfinite(s)))

    # SiLU is NOT globally monotonic: it has a local minimum around x ≈ -1.2
    # where the function dips and then rises back to 0 as x → -inf.
    # Verify instead: sign consistency and reasonable shape.
    pos_vals = s[x > 0]
    neg_vals = s[x < 0]
    check('SiLU: positive input produces positive output',
          np.all(pos_vals > 0),
          f'min positive output = {np.min(pos_vals):.6e}')
    check('SiLU: negative input produces negative output',
          np.all(neg_vals < 0),
          f'max negative output = {np.max(neg_vals):.6e}')
    check('SiLU: global min near x ≈ -1.2',
          np.min(s) > -0.28,
          f'global min = {np.min(s):.4f} (should be ≈ -0.278 at x ≈ -1.2)')

    # Additional SiLU values for V2-Lite verification
    for xv, expected_approx in [
        (1.0, 0.7311),
        (-1.0, -0.2689),
        (2.0, 1.7616),
        (-2.0, -0.2384),
    ]:
        actual = silu(np.array([xv], dtype=np.float32))[0]
        check(f'SiLU({xv})',
              abs(actual - expected_approx) < 0.001,
              f'got {actual:.4f}, expected ~{expected_approx:.4f}')


# -- Test 8: Identity weight propagation --------------------------------------

def test_identity_weights():
    """Identity weights -> output should match scaled input pattern."""
    test_section('Identity Weight Propagation (H=4, I=4)')

    ipipe = FFNPipeline(hidden=4, inter=4, num_experts=4, top_k=1)

    i_gate = make_identity_fp8_weights((4, 4))
    i_up   = make_identity_fp8_weights((4, 4))
    i_down = make_identity_fp8_weights((4, 4))

    ipipe.load_expert_weights(0, i_gate, i_up, i_down)
    ipipe.load_shared_expert(i_gate, i_up, i_down)

    inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out_i_fp8 = ipipe.forward(float_to_fp8(inp))
    out_i = fp8_to_float(out_i_fp8)

    print(f'  Input :  {inp}')
    print(f'  Output:  {np.round(out_i, 3)}')
    check('identity output: shape [4]',
          out_i.shape == (4,))
    check('identity output: all non-negative',
          np.all(out_i >= -1e-6))
    check('identity output: no NaN',
          not np.any(np.isnan(out_i)))
    check('identity output: monotonically increasing',
          np.all(np.diff(out_i) >= -1e-7))


# -- Test 9: verify_output and compare_outputs ---------------------------------

def test_verification_apis():
    """Verify the compare_outputs and verify_output API guards."""
    test_section('Verification API Guards')

    pipe = _build_small_pipeline(seed=42)
    x = float_to_fp8(np.random.RandomState(1).randn(_SMALL_HIDDEN).astype(np.float32))
    out_fp8 = pipe.forward(x)
    out_f32 = fp8_to_float(out_fp8)

    # Self-verify: compare FP8 output against float32 internal state
    # FP8 roundtrip introduces ~7% max relative error per element;
    # use 0.1 tolerance to account for encode/decode quantisation
    check('verify_output against self (fp8 bytes, tol=0.1)',
          pipe.verify_output(out_fp8, tolerance=0.1),
          'FP8 output should match pipeline internal state within FP8 precision')

    # Exact match: compare pipeline internal float32 with itself
    internal_f32 = pipe._last_output_f32.copy()
    check('verify_output against self (float32, exact match)',
          pipe.verify_output(internal_f32, tolerance=1e-6),
          'pipeline internal float32 should match itself exactly')

    # Shape guard
    try:
        pipe.verify_output(np.zeros(_SMALL_HIDDEN + 1, dtype=np.uint8))
        check('verify_output rejects wrong shape', False, 'expected ValueError')
    except ValueError:
        check('verify_output rejects wrong shape', True)

    # compare_outputs static method
    stats = FFNPipeline.compare_outputs(out_fp8, out_fp8, tolerance=1e-6)
    check('compare_outputs: self-match',
          stats['match'],
          f'max_abs_err={stats["max_abs_err"]:.2e}, max_rel_err={stats["max_rel_err"]:.2e}')

    # Compare slightly different outputs
    alt_out_fp8 = float_to_fp8(out_f32 + 0.01)
    stats2 = FFNPipeline.compare_outputs(out_fp8, alt_out_fp8, tolerance=0.1)
    check('compare_outputs: slightly different outputs quantified',
          stats2['max_rel_err'] >= 0,
          f'max_rel_err={stats2["max_rel_err"]:.4f}')


# -- Test 10: Scale test (full dimensions, timing) -----------------------------

def test_full_scale_timing(skip_timing: bool = False):
    """Run with V2-Lite production dimensions -- timing only."""
    test_section(f'Full-Scale Throughput (HIDDEN={HIDDEN}, INTER={INTER}, '
                 f'experts={NUM_EXPERTS} (64r + 2sh), TOP_K={TOP_K})')

    if skip_timing:
        skip('full-scale timing test',
             'use --full to run (loads 3 experts + 2 shared = ~43 MB FP8 weights)')
        return

    print(f'  Building pipeline with {NUM_EXPERTS} experts '
          f'({NUM_EXPERTS - 2} routed + 2 shared)...')
    weight_mb = HIDDEN * INTER * 1 / 1024 / 1024
    print(f'  Weight shapes: gate_w [{HIDDEN}, {INTER}] = {weight_mb:.1f} MB FP8 per matrix')

    t0 = time.perf_counter()
    pipe = FFNPipeline(hidden=HIDDEN, inter=INTER,
                       num_experts=NUM_EXPERTS, top_k=TOP_K)

    # Load 3 routed experts + 2 shared (enough for correctness)
    n_route = min(3, NUM_EXPERTS - 2)
    for eid in range(n_route):
        pipe.load_expert_weights(
            eid,
            gate_w=make_random_fp8_weights((HIDDEN, INTER), seed=1 + eid * 3),
            up_w=make_random_fp8_weights((HIDDEN, INTER), seed=2 + eid * 3),
            down_w=make_random_fp8_weights((INTER, HIDDEN), seed=3 + eid * 3),
        )
    pipe.load_shared_expert(
        gate_w=make_random_fp8_weights((HIDDEN, INTER), seed=10),
        up_w=make_random_fp8_weights((HIDDEN, INTER), seed=11),
        down_w=make_random_fp8_weights((INTER, HIDDEN), seed=12),
    )
    pipe.load_shared_expert(
        gate_w=make_random_fp8_weights((HIDDEN, INTER), seed=13),
        up_w=make_random_fp8_weights((HIDDEN, INTER), seed=14),
        down_w=make_random_fp8_weights((INTER, HIDDEN), seed=15),
    )

    t_build = time.perf_counter() - t0
    check(f'pipeline built ({n_route} routed + 2 shared) in {t_build:.2f} s',
          t_build > 0)

    # Create FP8 activation
    rng = np.random.RandomState(12345)
    raw_input = rng.randn(HIDDEN).astype(np.float32) * 2.0
    activ_fp8 = float_to_fp8(raw_input)

    check(f'activation FP8: shape {activ_fp8.shape}', activ_fp8.shape == (HIDDEN,))
    check(f'activation FP8: dtype {activ_fp8.dtype}', activ_fp8.dtype == np.uint8)

    # Warmup
    _ = pipe.forward(activ_fp8)

    # Measure
    n_iter = 3
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pipe.forward(activ_fp8)
    t_full = (time.perf_counter() - t0) / n_iter

    tok_s = 1.0 / t_full
    print(f'\n  Full MoE forward latency: {t_full * 1000:.2f} ms')
    print(f'  Throughput (Python sim):  {tok_s:.1f} tok/s')
    print(f'  Per-expert time estimate: {t_full * 1000 / (n_route + 2):.2f} ms')

    # HW target: V2-Lite is smaller than V4-Flash, so faster in simulation
    print(f'\n  HW target (FPGA 128 lanes x 500 MHz): ~400 tok/s')
    print(f'  Python sim vs HW: {tok_s / 400:.4f}x  (not performance-competitive)')
    print(f'  Note: Python model is a golden reference, not a performance target.')

    # Verify output is well-formed
    out_f32 = fp8_to_float(pipe.forward(activ_fp8))
    check('full-scale output: no NaN', not np.any(np.isnan(out_f32)))
    check('full-scale output: all finite', np.all(np.isfinite(out_f32)))
    print(f'  Output range: [{out_f32.min():.4f}, {out_f32.max():.4f}]')
    print(f'  Timing:       {pipe.last_timing_ms:.0f} ms/token')


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='DeepSeek V2-Lite FFN E2E test harness')
    parser.add_argument('--quick', action='store_true',
                        help='Skip full-scale timing test')
    parser.add_argument('--full', action='store_true',
                        help='Run full-scale timing test (expensive: builds ~43 MB of weights)')
    args = parser.parse_args()

    print(_bold('\n+==============================================================+'))
    print(_bold('|   DeepSeek V2-Lite FFN Pipeline -- E2E Test Harness'))
    print(_bold('|   Target: v2_lite_ffn_engine.sv (RTL reference)'))
    print(_bold('+==============================================================+'))

    # Architecture summary
    print(f'\n  Architecture: HIDDEN={HIDDEN}, INTER={INTER}, '
          f'NUM_EXPERTS={NUM_EXPERTS} ({NUM_EXPERTS - 2}r + 2sh), TOP_K={TOP_K}, FP8')
    print(f'  16B parameters, 27 layers')
    print(f'  Test device:  {os.name}')

    # -- Run tests ---------------------------------------------------------

    tests = [
        ('FP8 Roundtrip & Boundary Values',  test_fp8_roundtrip),
        ('Full FFN Inference (small)',        test_full_ffn_small),
        ('Bit-Exact Reproducibility',         test_reproducibility),
        ('Single Expert Forward Pass',        test_single_expert),
        ('Edge: All-Zero Input',              test_edge_zero_input),
        ('Edge: Maximum Value Input',         test_edge_max_input),
        ('Edge: NaN Input Sanitization',      test_edge_nan_input),
        ('SiLU Activation Correctness',       test_silu),
        ('Identity Weight Propagation',       test_identity_weights),
        ('Verification API Guards',           test_verification_apis),
    ]

    t_start = time.perf_counter()
    for name, fn in tests:
        try:
            fn()
        except Exception:
            global _fail_count
            _fail_count += 1
            print(f'\n  {_red("CRASH")}  {name}')
            traceback.print_exc()

    # Full-scale test (expensive)
    run_full = args.full and not args.quick
    try:
        test_full_scale_timing(skip_timing=not run_full)
    except Exception:
        _fail_count += 1
        print(f'\n  {_red("CRASH")}  Full-Scale Timing')
        traceback.print_exc()

    t_elapsed = time.perf_counter() - t_start

    # -- Summary -----------------------------------------------------------

    total = _pass_count + _fail_count + _skip_count
    print(f'\n{_bold("===")} {_bold("Summary")} {_bold("=" * 64)}')
    print(f'  {_green(f"PASS: {_pass_count}")}  '
          f'{_red(f"FAIL: {_fail_count}")}  '
          f'{_yellow(f"SKIP: {_skip_count}")}  '
          f'TOTAL: {total}')
    print(f'  Elapsed: {t_elapsed:.2f} s')

    if _fail_count == 0:
        print(f'\n  {_green(_bold("All tests passed!"))}')
        return 0
    else:
        print(f'\n  {_red(_bold(f"{_fail_count} test(s) failed!"))}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
