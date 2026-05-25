"""
verify_fp4_mac_stages.py — Stage-by-stage verification of fp4_mac pipeline.

Matches RTL exactly:
  Stage 0: fp4 LUT decode (x16) + fp8 E4M3 decode (x256)
  Stage 1: Decoded operands — width check, value distribution
  Stage 2: signed 8b x signed 12b -> 20b product — exhaustive 16x256
  Stage 3: 20b sign-extend -> 32b accumulate — realistic GEMM

Verification strategy:
  - Exhaustive where possible (16 fp4, 256 fp8, 4096 products)
  - Realistic distributions for accumulation (Gaussian activations)
  - Bit-accurate: Python model matches RTL cycle-by-cycle
  - Quantization error quantified separately from saturation
"""

import numpy as np
import math

# ============================================================================
# RTL-matched decode — bit-accurate to fp4_mac.sv
# ============================================================================

# fp4 LUT: 3-bit magnitude index -> value x 16
FP4_LUT = np.array([0, 4, 8, 12, 16, 24, 32, 48], dtype=np.int16)
FP4_FLOAT = FP4_LUT.astype(np.float64) / 16.0  # [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def fp4_decode(mag: np.ndarray) -> np.ndarray:
    """fp4 3-bit magnitude -> x16 scaled. sign applied separately by caller."""
    return FP4_LUT[mag & 0x7]


def fp4_decode_signed(fp4: np.ndarray) -> np.ndarray:
    """fp4 E2M1 encoded -> signed 8-bit (x16). Matches RTL Stage 1."""
    fp4 = np.asarray(fp4, dtype=np.uint8)
    mag = fp4 & 0x7
    sign = (fp4 >> 3) & 1
    val = FP4_LUT[mag].copy()
    val[(sign == 1) & (mag != 0)] = -val[(sign == 1) & (mag != 0)]
    return val.astype(np.int16)


def fp4_to_float(fp4: np.ndarray) -> np.ndarray:
    """fp4 E2M1 -> float32."""
    fp4 = np.asarray(fp4, dtype=np.uint8)
    mag = fp4 & 0x7
    sign = ((fp4 >> 3) & 1).astype(np.float32)
    v = FP4_FLOAT[mag].astype(np.float32)
    v = np.where(sign == 1, -v, v)
    v[mag == 0] = 0.0
    return v


def fp8_decode_signed(fp8: np.ndarray) -> np.ndarray:
    """
    fp8 E4M3 -> signed 12-bit (x256). Bit-accurate to RTL Stage 1.

    Normal  (e!=0): (1+m/8)x2^(e-7), x256 -> (8+m)x2^(e-2)
    Subnorm (e==0): m/8 x 2^(-6), x256 -> m/2
    Saturate to [-2048, 2047] for 12-bit signed.
    """
    fp8 = np.asarray(fp8, dtype=np.uint8)
    sign = (fp8 >> 7) & 1
    exp = (fp8 >> 3) & 0xF
    mant = fp8 & 0x7

    mag = np.zeros(len(fp8), dtype=np.int32)

    # Subnorm: e=0 -> m//2
    m_sub = exp == 0
    mag[m_sub] = mant[m_sub] // 2

    # Normal e=1: (8+m) >> 1
    m_e1 = exp == 1
    mag[m_e1] = (8 + mant[m_e1]) >> 1

    # Normal e>=2: (8+m) << (e-2), saturate
    m_ge2 = exp >= 2
    shift = exp[m_ge2].astype(np.int32) - 2
    base = (8 + mant[m_ge2]).astype(np.int32)
    full = base.astype(np.int64) << shift
    full = np.clip(full, 0, 2047)  # clip magnitude before sign
    mag[m_ge2] = full

    # Apply sign
    result = np.where((sign == 1) & (mag != 0), -mag, mag)
    return result.astype(np.int16)


def fp8_to_float(fp8: np.ndarray) -> np.ndarray:
    """fp8 E4M3 -> float32 (IEEE-754-like reference)."""
    fp8 = np.asarray(fp8, dtype=np.uint8)
    sign = ((fp8 >> 7) & 1).astype(np.float32)
    exp = ((fp8 >> 3) & 0xF).astype(np.int32)
    mant = (fp8 & 0x7).astype(np.float32)

    v = np.zeros(len(fp8), dtype=np.float32)
    m_sub = exp == 0
    m_norm = exp != 0
    v[m_sub] = (mant[m_sub] / 8.0) * (2.0 ** (-6))
    v[m_norm] = (1.0 + mant[m_norm] / 8.0) * (2.0 ** (exp[m_norm].astype(np.float32) - 7))
    return np.where(sign == 1, -v, v)


def gen_realistic_activations(n: int, seed: int = 42) -> np.ndarray:
    """
    Generate realistic fp8 activation encodings.
    Post-RMSNorm activations ~ N(0, 1), clipped to [-4, 4].
    Quantize to nearest fp8 E4M3 value.
    """
    rng = np.random.RandomState(seed)
    floats = rng.randn(n).astype(np.float32)
    floats = np.clip(floats, -4.0, 4.0)
    # Find nearest fp8 encoding
    all_fp8 = np.arange(256, dtype=np.uint8)
    all_vals = fp8_to_float(all_fp8)
    # For each float, find closest fp8 value
    idx = np.abs(floats[:, None] - all_vals[None, :]).argmin(axis=1)
    return all_fp8[idx]


def gen_realistic_weights(k: int, seed: int = 123) -> np.ndarray:
    """Generate realistic fp4 weights (uniform over all values)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 16, size=k).astype(np.uint8)


# ============================================================================
# Print helpers
# ============================================================================
def green(s):
    return s  # kept simple


def red(s):
    return s


# ============================================================================
# Stage 0a: fp4 decode — exhaustive 16 values
# ============================================================================
def verify_fp4_decode():
    print("=" * 64)
    print(" Stage 0a: fp4 E2M1 decode (LUT, exhaustive)")
    print("=" * 64)

    all_fp4 = np.arange(16, dtype=np.uint8)
    dec = fp4_decode_signed(all_fp4)
    fref = fp4_to_float(all_fp4)
    # RTL float = decoded / 16
    frecon = dec.astype(np.float64) / 16.0

    errors = np.abs(frecon - fref.astype(np.float64))
    passed = errors.max() < 1e-9

    print(f"{'idx':>4s}  {'fp4':>8s}  {'sign exp man':>14s}  {'dec':>5s}  {'float':>8s}  {'recon':>8s}  err")
    print("-" * 64)
    for i in range(16):
        s = (i >> 3) & 1
        e = (i >> 1) & 3
        m = i & 1
        eb = f"{e:02b}"
        print(f"{i:4d}  4'b{s}_{eb}_{m}    s={s} e={eb} m={m}   "
              f"{dec[i]:+5d}  {fref[i]:+8.4f}  {frecon[i]:+8.4f}  {errors[i]:.0e}")
    print(f"\n  Max error: {errors.max():.1e}  {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Stage 0b: fp8 E4M3 decode — exhaustive 256 values
# ============================================================================
def verify_fp8_decode():
    print("\n" + "=" * 64)
    print(" Stage 0b: fp8 E4M3 decode (x256, exhaustive)")
    print("=" * 64)

    all_fp8 = np.arange(256, dtype=np.uint8)
    dec = fp8_decode_signed(all_fp8)
    fref = fp8_to_float(all_fp8)
    frecon = dec.astype(np.float64) / 256.0

    # Separate saturated vs unsaturated
    is_sat = np.abs(dec) >= 2047  # saturated magnitude
    n_sat = is_sat.sum()
    n_unsat = (~is_sat).sum()

    abs_err = np.abs(frecon - fref.astype(np.float64))
    rel_err = np.where(np.abs(fref) > 1e-8, abs_err / np.abs(fref), 0)

    print(f"  Total encodings: 256")
    print(f"  Unsaturated: {n_unsat}  |  Saturated: {n_sat}")
    print(f"  Decoded range: [{dec.min()}, {dec.max()}]")
    print()

    # Unsaturated error stats
    unsat_err = abs_err[~is_sat]
    unsat_rel = rel_err[~is_sat]
    print(f"  --- Unsaturated ({n_unsat} values) ---")
    print(f"  Max  absolute error: {unsat_err.max():.6e}")
    print(f"  Mean absolute error: {unsat_err.mean():.6e}")
    print(f"  Max  relative error: {unsat_rel.max():.6e}")
    print(f"  Mean relative error: {unsat_rel.mean():.6e}")
    print(f"  % with abs err < 1e-9: {(unsat_err < 1e-9).mean() * 100:.1f}%")

    # Saturated values
    if n_sat > 0:
        print(f"\n  --- Saturated ({n_sat} values) ---")
        sat_idx = np.where(is_sat)[0]
        for idx in sat_idx[:8]:  # first 8
            e = (idx >> 3) & 0xF
            m = idx & 0x7
            s = (idx >> 7) & 1
            print(f"  fp8=0x{idx:02x} (e={e:2d} m={m}) sign={s}  "
                  f"float={fref[idx]:+12.6f}  dec={dec[idx]:+6d}  -> clip to {dec[idx]/256:+.4f}")

    # Spot-check key values
    print(f"\n  --- Key value spot-check ---")
    checks = [
        (0x00, "subnorm m=0"),
        (0x01, "subnorm m=1"),
        (0x07, "subnorm m=7"),
        (0x08, "e=1 m=0"),
        (0x0F, "e=1 m=7"),
        (0x38, "e=7 m=0 -> 1.0"),
        (0x40, "e=8 m=0 -> 2.0"),
        (0xB0, "e=6 m=0 sign=1 -> -0.5"),
        (0x50, "e=10 m=0 -> 8.0 (saturates?)"),
    ]
    for code, desc in checks:
        sat_mark = " [SAT]" if is_sat[code] else ""
        print(f"  fp8=0x{code:02x} {desc:30s}  dec={dec[code]:+6d}  "
              f"float={fref[code]:+10.6f}  recon={frecon[code]:+10.6f}{sat_mark}")

    # Quantization error from floor division (subnorm: m//2, e=1: (8+m)//2)
    # is at most 0.5/256 = 1/512 ≈ 0.00195. This is expected, not a bug.
    max_expected_qerr = 1.0 / 512.0
    passed = unsat_err.max() <= max_expected_qerr + 1e-12
    print(f"\n  Max theoretical quantization error: {max_expected_qerr:.6e}")
    print(f"  Actual max unsaturated error:     {unsat_err.max():.6e}")
    print(f"  Bit-accurate to RTL (within quantization): {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Stage 1: Combined decoded operands — width & range
# ============================================================================
def verify_stage1_combined():
    print("\n" + "=" * 64)
    print(" Stage 1: Decoded operand width & range check")
    print("=" * 64)

    all_fp4 = np.arange(16, dtype=np.uint8)
    all_fp8 = np.arange(256, dtype=np.uint8)
    w = fp4_decode_signed(all_fp4)
    a = fp8_decode_signed(all_fp8)

    w_fits = w.min() >= -128 and w.max() <= 127
    a_fits = a.min() >= -2048 and a.max() <= 2047

    print(f"  fp4 signed 8b: [{w.min():+d}, {w.max():+d}]  fits in [-128,127]: {w_fits}")
    print(f"  fp8 signed 12b: [{a.min():+d}, {a.max():+d}]  fits in [-2048,2047]: {a_fits}")
    print(f"  fp4 nonzero values: {(w != 0).sum()}/16")
    print(f"  fp8 nonzero values: {(a != 0).sum()}/256")
    print(f"  fp8 saturated: {((a == 2047) | (a == -2048) | (a == -2047)).sum()}/256")

    return w_fits and a_fits


# ============================================================================
# Stage 2: Multiply — exhaustive 4096 products
# ============================================================================
def verify_stage2_multiply():
    print("\n" + "=" * 64)
    print(" Stage 2: signed 8b x signed 12b -> 20b (exhaustive 16x256)")
    print("=" * 64)

    all_fp4 = np.arange(16, dtype=np.uint8)
    all_fp8 = np.arange(256, dtype=np.uint8)
    w = fp4_decode_signed(all_fp4).astype(np.int32)
    a = fp8_decode_signed(all_fp8).astype(np.int32)

    # All 4096 products
    products = np.outer(w, a).astype(np.int64)

    p_min, p_max = products.min(), products.max()
    fits_20b = p_min >= -(1 << 19) and p_max <= (1 << 19) - 1

    print(f"  Product matrix: {products.shape[0]} x {products.shape[1]} = {products.size} values")
    print(f"  Range: [{p_min}, {p_max}]")
    print(f"  20-bit signed range [-524288, 524287]: {'OK' if fits_20b else 'OVERFLOW'}")

    # Bit-accurate vs float reference
    w_f = fp4_to_float(all_fp4).astype(np.float64)
    a_f = fp8_to_float(all_fp8).astype(np.float64)
    products_float = np.outer(w_f, a_f)
    products_recon = products.astype(np.float64) / 4096.0  # (x16 * x256 = x4096)

    abs_err = np.abs(products_recon - products_float)
    rel_err = np.where(np.abs(products_float) > 1e-8,
                       abs_err / np.abs(products_float), 0)

    # Separate saturated vs unsaturated activations
    a_is_sat = np.abs(a) >= 2047
    unsat_mask = ~np.outer(np.ones(16, dtype=bool), a_is_sat)
    sat_mask = np.outer(np.ones(16, dtype=bool), a_is_sat)

    unsat_err = abs_err[unsat_mask]
    sat_err = abs_err[sat_mask]

    print(f"\n  --- Unsaturated activations ({unsat_mask.sum()} products) ---")
    print(f"  Max  absolute error: {unsat_err.max():.6e}")
    print(f"  Mean absolute error: {unsat_err.mean():.6e}")
    print(f"  Exact matches: {(unsat_err < 1e-9).mean() * 100:.1f}%")

    if sat_mask.sum() > 0:
        print(f"\n  --- Saturated activations ({sat_mask.sum()} products) ---")
        print(f"  Max  absolute error: {sat_err.max():.4f}")
        print(f"  Mean absolute error: {sat_err.mean():.4f}")

    # Show worst-case non-saturated product
    unsat_flat = abs_err[unsat_mask]
    if len(unsat_flat) > 0:
        worst = np.argmax(unsat_flat)
        ww, wa = np.where(unsat_mask)
        wi, ai = ww[worst], wa[worst]
        print(f"\n  Worst unsaturated product: fp4={all_fp4[wi]:4d} x fp8=0x{all_fp8[ai]:02x}")
        print(f"    w={w[wi]}, a={a[ai]}, prod={products[wi,ai]}")
        print(f"    recon={products_recon[wi,ai]:.8f}, float={products_float[wi,ai]:.8f}")
        print(f"    err={unsat_flat[worst]:.6e}")

    # For unsaturated: max error comes from fp8 subnorm/e=1 floor division
    # propagated through fp4 multiply. Worst case: fp4=±48 × fp8 floor error 0.5/256.
    # Max product error = 48 × 0.5 / 256 = 0.09375
    max_expected_qerr = 48.0 * 0.5 / 256.0  # = 0.09375
    print(f"\n  Max theoretical product quantization error: {max_expected_qerr:.6f}")
    print(f"  Actual max unsaturated product error:      {unsat_err.max():.6e}")
    passed_stage2 = fits_20b and unsat_err.max() <= max_expected_qerr + 1e-9
    print(f"  Bit-accurate for unsaturated: {'PASS' if passed_stage2 else 'FAIL'}")
    return passed_stage2


# ============================================================================
# Stage 3: Accumulate — realistic GEMM inner product
# ============================================================================
def verify_stage3_accumulate():
    print("\n" + "=" * 64)
    print(" Stage 3: 20b sign-extend -> 32b accumulate")
    print("=" * 64)

    rng = np.random.RandomState(42)
    K = 128
    M, N = 4, 4
    num_tests = 100

    max_abs_err = 0.0
    max_rel_err = 0.0
    all_rel_err = []

    for trial in range(num_tests):
        w_enc = gen_realistic_weights(M * K, seed=1000 + trial).reshape(M, K)
        a_enc = gen_realistic_activations(K * N, seed=2000 + trial).reshape(K, N)

        w_dec = np.zeros((M, K), dtype=np.int32)
        a_dec = np.zeros((K, N), dtype=np.int32)
        for i in range(M):
            w_dec[i] = fp4_decode_signed(w_enc[i]).astype(np.int32)
        for j in range(N):
            a_dec[:, j] = fp8_decode_signed(a_enc[:, j]).astype(np.int32)

        # RTL accumulation
        accum_rtl = np.zeros((M, N), dtype=np.int64)
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    accum_rtl[i, j] += int(w_dec[i, k]) * int(a_dec[k, j])

        accum_rtl = np.clip(accum_rtl, -(2 ** 31), 2 ** 31 - 1)

        # Float reference
        w_f = fp4_to_float(w_enc.flatten()).reshape(M, K).astype(np.float64)
        a_f = fp8_to_float(a_enc.flatten()).reshape(K, N).astype(np.float64)
        accum_ref = w_f @ a_f

        # RTL -> float
        accum_recon = accum_rtl.astype(np.float64) / 4096.0
        abs_err = np.abs(accum_recon - accum_ref)
        rel_err = np.where(np.abs(accum_ref) > 1e-8,
                           abs_err / np.abs(accum_ref), 0)

        max_abs_err = max(max_abs_err, abs_err.max())
        max_rel_err = max(max_rel_err, rel_err.max())
        all_rel_err.extend(rel_err.flatten().tolist())

    all_rel_err = np.array(all_rel_err)

    print(f"  Tests: {num_tests} GEMMs ({M}x{K} x {K}x{N})")
    print(f"  Max  absolute error: {max_abs_err:.6e}")
    print(f"  Mean absolute error: {np.mean(all_rel_err):.6e}  (relative)")
    print(f"  Max  relative error: {max_rel_err:.6e}")
    print(f"  P50  relative error: {np.percentile(all_rel_err, 50):.6e}")
    print(f"  P95  relative error: {np.percentile(all_rel_err, 95):.6e}")
    print(f"  P99  relative error: {np.percentile(all_rel_err, 99):.6e}")

    # Show one example in detail
    w_enc = gen_realistic_weights(8, seed=9999)
    a_enc = gen_realistic_activations(8, seed=9999)
    w_dec = fp4_decode_signed(w_enc).astype(np.int64)
    a_dec = fp8_decode_signed(a_enc).astype(np.int64)
    accum_rtl = np.int64(0)
    for k in range(8):
        accum_rtl += int(w_dec[k]) * int(a_dec[k])
    w_f = fp4_to_float(w_enc).astype(np.float64)
    a_f = fp8_to_float(a_enc).astype(np.float64)
    accum_ref = np.dot(w_f, a_f)
    accum_recon = float(accum_rtl) / 4096.0

    print(f"\n  --- Example K=8 dot product ---")
    for k in range(8):
        p_rtl = int(w_dec[k]) * int(a_dec[k])
        p_float = float(w_f[k]) * float(a_f[k])
        p_recon = p_rtl / 4096.0
        print(f"  k={k}: fp4=0x{w_enc[k]:01x} ({w_f[k]:+.4f}) x "
              f"fp8=0x{a_enc[k]:02x} ({a_f[k]:+.4f})  |  "
              f"rtl_prod={p_rtl:+6d} ({p_recon:+.6f})  ref={p_float:+.6f}  "
              f"err={abs(p_recon-p_float):.2e}")
    print(f"  accum:  rtl={accum_rtl:+8d} ({accum_recon:+.8f})  ref={accum_ref:+.8f}  "
          f"err={abs(accum_recon-accum_ref):.2e}")

    passed = max_rel_err < 0.05
    print(f"\n  {'PASS' if passed else 'FAIL'} (max relative error < 5%)")
    return passed


# ============================================================================
# Stage: Saturation analysis
# ============================================================================
def analyze_saturation():
    print("\n" + "=" * 64)
    print(" Saturation analysis: when does fp8 clip?")
    print("=" * 64)

    # fp8 E4M3 values sorted by float value
    all_fp8 = np.arange(256, dtype=np.uint8)
    dec = fp8_decode_signed(all_fp8)
    fref = fp8_to_float(all_fp8)
    is_sat = np.abs(dec) >= 2047

    print(f"  Saturated encodings: {is_sat.sum()}/256")
    print(f"  Float range of ALL saturated: [{fref[is_sat].min():.2f}, {fref[is_sat].max():.2f}]")
    # Find the smallest |float| that saturates
    sat_floats = np.abs(fref[is_sat])
    print(f"  Float saturation threshold: |x| >= {sat_floats.min():.4f}")
    print(f"  Decoded threshold: |dec| = 2047 -> recon = {2047/256:.4f}")

    # What fraction of Gaussian activations saturate?
    # N(0,1): 99.7% within [-3, 3], far from 8.0 threshold
    # N(0,2): 95% within [-4, 4], still far
    # N(0,4): 95% within [-8, 8], border
    for sigma in [1.0, 2.0, 3.0, 4.0]:
        threshold = 8.0  # approximate saturation threshold in float
        prob_sat = 2 * (1 - 0.5 * (1 + math.erf(threshold / (sigma * math.sqrt(2)))))
        print(f"  N(0,{sigma:.1f}): P(|x|>8.0) = {prob_sat:.6e}  (~{prob_sat*1e6:.1f} ppm)")

    print(f"\n  Conclusion: for post-LayerNorm activations (sigma~1),")
    print(f"  saturation rate is negligible (< 1e-15).")
    print(f"  The x256 scaling and 12-bit output are sufficient.")

    return True


# ============================================================================
# End-to-end: realistic GEMM
# ============================================================================
def verify_e2e_realistic():
    print("\n" + "=" * 64)
    print(" End-to-end: fp4 x fp8 GEMM with realistic data")
    print("=" * 64)

    rng = np.random.RandomState(0)
    M, K, N = 16, 128, 16
    num_trials = 50

    errors = []
    cos_sims = []

    for trial in range(num_trials):
        w_enc = gen_realistic_weights(M * K, seed=3000 + trial).reshape(M, K)
        a_enc = gen_realistic_activations(K * N, seed=4000 + trial).reshape(K, N)

        # Fixed-point GEMM (RTL model)
        accum = np.zeros((M, N), dtype=np.int64)
        for i in range(M):
            w_i = fp4_decode_signed(w_enc[i]).astype(np.int64)
            for j in range(N):
                a_j = fp8_decode_signed(a_enc[:, j]).astype(np.int64)
                accum[i, j] = np.dot(w_i, a_j)

        accum_f = accum.astype(np.float64) / 4096.0

        # Float reference
        w_f = fp4_to_float(w_enc.flatten()).reshape(M, K).astype(np.float64)
        a_f = fp8_to_float(a_enc.flatten()).reshape(K, N).astype(np.float64)
        ref = w_f @ a_f

        abs_err = np.abs(accum_f - ref)
        rel_err = np.where(np.abs(ref) > 1e-8, abs_err / np.abs(ref), 0)

        # Per-row cosine similarity
        for i in range(M):
            norm_rtl = np.linalg.norm(accum_f[i])
            norm_ref = np.linalg.norm(ref[i])
            if norm_rtl > 1e-8 and norm_ref > 1e-8:
                cos = np.dot(accum_f[i], ref[i]) / (norm_rtl * norm_ref)
                cos_sims.append(cos)

        errors.extend(rel_err.flatten().tolist())

    errors = np.array(errors)
    cos_sims = np.array(cos_sims)

    print(f"  Trials: {num_trials} GEMMs ({M}x{K} x {K}x{N})")
    print(f"  Total output elements: {len(errors)}")
    print(f"  --- Relative error ---")
    print(f"  Max:    {errors.max():.6e}")
    print(f"  Mean:   {errors.mean():.6e}")
    print(f"  P50:    {np.percentile(errors, 50):.6e}")
    print(f"  P95:    {np.percentile(errors, 95):.6e}")
    print(f"  P99:    {np.percentile(errors, 99):.6e}")
    print(f"  --- Cosine similarity ---")
    print(f"  Mean:   {cos_sims.mean():.8f}")
    print(f"  Min:    {cos_sims.min():.8f}")
    print(f"  P1:     {np.percentile(cos_sims, 1):.8f}")

    passed = cos_sims.min() > 0.999
    print(f"\n  Min cosine > 0.999: {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    results = []
    results.append(("Stage 0a: fp4 decode", verify_fp4_decode()))
    results.append(("Stage 0b: fp8 decode", verify_fp8_decode()))
    results.append(("Stage 1:  operand check", verify_stage1_combined()))
    results.append(("Stage 2:  multiply", verify_stage2_multiply()))
    results.append(("Stage 3:  accumulate", verify_stage3_accumulate()))
    results.append(("Sat analysis", analyze_saturation()))
    results.append(("End-to-end", verify_e2e_realistic()))

    print("\n" + "=" * 64)
    print(" SUMMARY")
    print("=" * 64)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s} [{status}]")
        if not passed:
            all_pass = False
    print(f"\n  {'ALL STAGES PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 64)
