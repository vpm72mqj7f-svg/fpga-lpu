"""
gen_tb_vectors.py — Generate golden test vectors for tb_fp4_mac.sv.

Outputs a SystemVerilog package file (tb_golden_pkg.sv) containing:
  - Test case definitions (fp4 weight, fp8 activation, expected product)
  - Expected accumulated results for each test sequence

This provides an independent golden reference — not sharing code with the RTL.
"""

import numpy as np
import sys
import os

# Add parent for fp4_utils imports if needed
sys.path.insert(0, os.path.dirname(__file__))

# Duplicate the RTL-matched decode here (independent from verify script)
# to avoid any shared-bug risk.

FP4_LUT = np.array([0, 4, 8, 12, 16, 24, 32, 48], dtype=np.int16)


def fp4_decode_signed(fp4):
    fp4 = np.asarray(fp4, dtype=np.uint8)
    mag = fp4 & 0x7
    sign = (fp4 >> 3) & 1
    val = FP4_LUT[mag].copy()
    val[(sign == 1) & (mag != 0)] = -val[(sign == 1) & (mag != 0)]
    return val.astype(np.int16)


def fp8_decode_signed(fp8):
    fp8 = np.asarray(fp8, dtype=np.uint8)
    sign = (fp8 >> 7) & 1
    exp = (fp8 >> 3) & 0xF
    mant = fp8 & 0x7

    mag = np.zeros(len(fp8), dtype=np.int32)
    m_sub = exp == 0
    mag[m_sub] = mant[m_sub] // 2
    m_e1 = exp == 1
    mag[m_e1] = (8 + mant[m_e1]) >> 1
    m_ge2 = exp >= 2
    shift = exp[m_ge2].astype(np.int32) - 2
    base = (8 + mant[m_ge2]).astype(np.int32)
    full = base.astype(np.int64) << shift
    full = np.clip(full, 0, 2047)
    mag[m_ge2] = full
    result = np.where((sign == 1) & (mag != 0), -mag, mag)
    return result.astype(np.int16)


def product_rtl(w_enc, a_enc, s_enc=0x38):
    """RTL product: fp4_decoded x fp8_decoded x fp8_scale (scale ×256, >>>8)."""
    w = int(fp4_decode_signed(np.array([w_enc], dtype=np.uint8))[0])
    a = int(fp8_decode_signed(np.array([a_enc], dtype=np.uint8))[0])
    s = int(fp8_decode_signed(np.array([s_enc], dtype=np.uint8))[0])
    p = (w * a * s) >> 8
    if p >= (1 << 31):
        p -= (1 << 32)
    elif p < -(1 << 31):
        p += (1 << 32)
    return p


def compute_accum(weight_list, activ_list, scale_list=None):
    """Compute RTL accumulator value for a sequence of (w, scale, a) triples."""
    if scale_list is None:
        scale_list = [0x38] * len(weight_list)  # fp8 +1.0
    accum = 0
    for w, a, s in zip(weight_list, activ_list, scale_list):
        p_32 = product_rtl(w, a, s)
        accum = (accum + p_32) & 0xFFFFFFFF  # 32-bit wrap
        if accum >= (1 << 31):
            accum -= (1 << 32)
    return accum


def format_sv_hex(val, bits):
    """Format integer as SystemVerilog hex literal with bit width."""
    mask = (1 << bits) - 1
    uv = val & mask
    return f"{bits}'h{uv:0{(bits+3)//4}x}"


def gen_golden_pkg():
    """Generate tb_golden_pkg.sv with all test vectors and expected results."""

    tests = []

    # --- Test 1: Single multiply ---
    tests.append({
        "name": "T1_SINGLE",
        "desc": "fp4 +1.0 x fp8 +1.0",
        "weights": [0x4],   # 0_10_0 = +1.0
        "activs":  [0x38],   # 0_0111_000 = +1.0
        "expected_accum": compute_accum([0x4], [0x38]),
    })

    # --- Test 2: Accumulate 4 ---
    w = [0x2, 0x2, 0x5, 0x5]  # +0.25, +0.25, +1.5, +1.5
    a = [0x30, 0x30, 0x28, 0x28]  # +0.5, +0.5, +0.25, +0.25
    tests.append({
        "name": "T2_ACCUM4",
        "desc": "4-term accumulation",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 3: Positive fp4 sweep (all 8) ---
    w = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]
    a = [0x38] * 8
    tests.append({
        "name": "T3_POS_SWEEP",
        "desc": "all 8 positive fp4 x +1.0",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 4: Negative fp4 sweep (all 7 nonzero) ---
    w = [0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF]
    a = [0x38] * 7
    tests.append({
        "name": "T4_NEG_SWEEP",
        "desc": "all 7 negative fp4 x +1.0",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 5: Mixed signs ---
    w = [0xC, 0x6]   # -1.0, +2.0
    a = [0x38, 0xB0]  # +1.0, -0.5
    tests.append({
        "name": "T5_MIXED_SIGN",
        "desc": "positive + negative cancellation",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 6: Zeros ---
    w = [0x0, 0x8, 0x4]  # +0, -0, +1.0
    a = [0x38, 0x38, 0x00]  # +1.0, +1.0, 0
    tests.append({
        "name": "T6_ZEROS",
        "desc": "fp4 zero, fp4 signed zero, fp8 zero",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 7: fp8 subnorm ---
    w = [0x4, 0x4, 0x4]  # +1.0
    a = [0x01, 0x04, 0x07]  # subnorm m=1, m=4, m=7
    tests.append({
        "name": "T7_SUBNORM",
        "desc": "fp8 subnorm values",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 8: fp8 e=1 edge (right-shift quantization) ---
    w = [0x4, 0x4, 0x4, 0x4]  # +1.0
    a = [0x08, 0x09, 0x0E, 0x0F]  # e=1 m=0,1,6,7
    tests.append({
        "name": "T8_E1_EDGE",
        "desc": "fp8 e=1 right-shift boundary",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 9: fp8 saturation boundary ---
    w = [0x4, 0x4, 0x4]  # +1.0
    a = [0x48, 0x4F, 0x50]  # e=9 m=0 (4.0), e=9 m=7 (7.5), e=10 m=0 (8.0, sats)
    tests.append({
        "name": "T9_SAT_EDGE",
        "desc": "fp8 near and at saturation",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 10: Large accumulation (no overflow) ---
    w = [0x7] * 32  # all +3.0 (largest positive)
    a = [0x38] * 32  # all +1.0
    tests.append({
        "name": "T10_LARGE_ACCUM",
        "desc": "32-term max-weight accumulation",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 11: Back-to-back streaming (no bubbles) ---
    w = [0x4] * 16
    a = [0x38] * 16
    tests.append({
        "name": "T11_STREAM",
        "desc": "16 back-to-back, no bubbles",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 12: Alternating signs (cancellation) ---
    w = [0x6, 0xE] * 8  # +2.0, -2.0 alternating
    a = [0x38] * 16      # all +1.0
    tests.append({
        "name": "T12_CANCEL",
        "desc": "alternating +2.0/-2.0 x +1.0 -> 0",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 13: fp8 value sweep (e=0,1,2,3,4,5,6,7) ---
    w = [0x4] * 8
    a = [0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38]  # various e
    tests.append({
        "name": "T13_FP8_SWEEP",
        "desc": "fp8 exponents 0..7",
        "weights": w,
        "activs": a,
        "expected_accum": compute_accum(w, a),
    })

    # --- Test 14: Non-unity scales ---
    # scale 0x40 ≈ +2.0, 0x30 ≈ +0.5
    w = [0x4, 0x4, 0x6, 0x6]  # +1.0, +1.0, +2.0, +2.0
    a = [0x38, 0x38, 0x38, 0x38]  # +1.0
    s = [0x40, 0x30, 0x40, 0x30]
    tests.append({
        "name": "T14_SCALE",
        "desc": "non-unity fp8 scale values",
        "weights": w,
        "activs": a,
        "scales": s,
        "expected_accum": compute_accum(w, a, s),
    })

    return tests


def write_sv_package(tests, filepath):
    """Write the golden test vectors as a SystemVerilog package."""

    lines = []
    lines.append("//=============================================================================")
    lines.append("// tb_golden_pkg.sv — AUTO-GENERATED by gen_tb_vectors.py")
    lines.append("// Golden reference values for tb_fp4_mac.sv")
    lines.append("// DO NOT EDIT BY HAND")
    lines.append("//=============================================================================")
    lines.append("")
    lines.append("package tb_golden_pkg;")
    lines.append("")

    # Test count
    lines.append(f"    localparam int NUM_TESTS = {len(tests)};")
    lines.append("")

    # Test data: lengths, weights, activs, scales, expected accum
    # Note: avoid SystemVerilog string localparam arrays for Icarus compatibility.
    for i, t in enumerate(tests):
        n = len(t["weights"])
        lines.append(f"    // {t['name']}: {t['desc']} ({n} terms)")
        lines.append(f"    localparam int T{i}_LEN = {n};")

        # Packed vectors, element 0 in low bits for Icarus compatibility
        w_str = ", ".join(f"4'h{w:01x}" for w in reversed(t["weights"]))
        lines.append(f"    localparam logic [{n}*4-1:0] T{i}_W_PACK = {{{w_str}}};")

        a_str = ", ".join(f"8'h{aa:02x}" for aa in reversed(t["activs"]))
        lines.append(f"    localparam logic [{n}*8-1:0] T{i}_A_PACK = {{{a_str}}};")

        scales = t.get("scales", [0x38] * n)
        s_str = ", ".join(f"8'h{ss:02x}" for ss in reversed(scales))
        lines.append(f"    localparam logic [{n}*8-1:0] T{i}_S_PACK = {{{s_str}}};")

        # Expected accumulated result
        acc = t["expected_accum"]
        lines.append(f"    localparam logic [31:0] T{i}_EXPECTED = 32'h{acc & 0xFFFFFFFF:08x};")
        lines.append("")

    lines.append("endpackage")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {filepath} ({len(tests)} test cases)")


if __name__ == "__main__":
    tests = gen_golden_pkg()

    # Print summary to console
    print("=" * 64)
    print(" Golden Test Vector Summary")
    print("=" * 64)
    for t in tests:
        n = len(t["weights"])
        acc = t["expected_accum"]
        acc_f = acc / 4096.0
        print(f"  {t['name']:20s}  n={n:2d}  accum=0x{acc&0xFFFFFFFF:08x} ({acc:+12d})  float={acc_f:+.6f}")

    # Write SV package
    outpath = os.path.join(os.path.dirname(__file__), "..", "..", "rtl", "sim", "tb_golden_pkg.sv")
    outpath = os.path.abspath(outpath)
    write_sv_package(tests, outpath)
    print(f"\n  Run: python {__file__}")
