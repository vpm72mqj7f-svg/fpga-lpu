"""
Compute expected values for T15-T19 corner case test vectors.
Uses the RTL-matched decode functions from gen_tb_vectors.py.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# Copy RTL-matched functions from gen_tb_vectors.py for independence
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
    if scale_list is None:
        scale_list = [0x38] * len(weight_list)
    # The RTL uses sat_acc for each accumulation, but the gen_tb_vectors
    # uses simple 32-bit wrapping. We need to match the RTL saturating
    # behavior. For most practical values, saturation is never reached,
    # but for our overflow test we need to model it correctly.
    accum = 0
    for w, a, s in zip(weight_list, activ_list, scale_list):
        p_32 = product_rtl(w, a, s)

        # RTL sat_acc logic: if both positive and sum goes negative => saturate positive
        #                    if both negative and sum goes positive => saturate negative
        sum_raw = (accum + p_32) & 0xFFFFFFFF
        if sum_raw >= (1 << 31):
            sum_raw -= (1 << 32)

        old_sign = accum < 0
        val_sign = p_32 < 0
        sum_sign = sum_raw < 0

        if (not old_sign) and (not val_sign) and sum_sign:
            # Positive saturation
            accum = (1 << 31) - 1
        elif old_sign and val_sign and (not sum_sign):
            # Negative saturation
            accum = -(1 << 31)
        else:
            accum = sum_raw

    return accum


def format_packed_hex(values, bits_per_elem):
    """Format reversed values as SV packed hex."""
    parts = []
    for v in reversed(values):
        fmt = f"{bits_per_elem}'h"
        if bits_per_elem == 4:
            parts.append(f"{fmt}{v & 0xF:01x}")
        elif bits_per_elem == 8:
            parts.append(f"{fmt}{v & 0xFF:02x}")
    return ", ".join(parts)


def print_test(idx, name, desc, weights, activs, scales, expected):
    n = len(weights)
    print(f"  // {name}: {desc} ({n} terms)")
    print(f"  localparam int T{idx}_LEN = {n};")
    print(f"  localparam logic [{n}*4-1:0] T{idx}_W_PACK = {{{{{format_packed_hex(weights, 4)}}}}};")
    print(f"  localparam logic [{n}*8-1:0] T{idx}_A_PACK = {{{{{format_packed_hex(activs, 8)}}}}};")
    print(f"  localparam logic [{n}*8-1:0] T{idx}_S_PACK = {{{{{format_packed_hex(scales, 8)}}}}};")
    print(f"  localparam logic [31:0] T{idx}_EXPECTED = 32'h{expected & 0xFFFFFFFF:08x};")
    print()


# ============================================================================
# T15: fp8 subnormals (e=0) with non-unity scale and varied fp4 weights
# ============================================================================
# fp8 subnormals: 0x01(m=1→0), 0x02(m=2→1), 0x03(m=3→1), 0x04(m=4→2),
#                  0x05(m=5→2), 0x06(m=6→3), 0x07(m=7→3)
# fp4 weights: +0.25, +0.5, +1.0, +2.0, -0.5, -1.0
# scale: non-unity fp8 0x30 (+0.5), mixed with 0x40 (+2.0)
w15 = [0x1, 0x2, 0x4, 0x6, 0xA, 0xC, 0x3, 0x2]  # +0.25, +0.5, +1, +2, -0.5, -1, +0.75, +0.5
a15 = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x01]  # subnorm m=1..7,1
s15 = [0x40, 0x40, 0x30, 0x30, 0x40, 0x40, 0x30, 0x38]  # +2.0, +2.0, +0.5, +0.5, +2.0, +2.0, +0.5, +1.0
acc15 = compute_accum(w15, a15, s15)
print_test(14, "T15_FP8_SUBNORM_SCALED", "fp8 subnormals x varied fp4 x non-unity scale",
           w15, a15, s15, acc15)

# ============================================================================
# T16: scale=0 edge case (fp8 0x00 = zero scale)
# ============================================================================
# All products should be zero regardless of weight/activation
w16 = [0x4, 0x6, 0x7, 0xC]  # +1, +2, +3, -1
a16 = [0x38, 0x48, 0x50, 0x38]  # +1, +4, +8(sat), +1
s16 = [0x00, 0x00, 0x00, 0x00]  # all scale=0
acc16 = compute_accum(w16, a16, s16)
print_test(15, "T16_SCALE_ZERO", "scale=0 zeroes all products",
           w16, a16, s16, acc16)

# ============================================================================
# T17: scale=0 mixed with normal (scale=0 mid-stream, then normal)
# ============================================================================
# Tests that scale=0 produces zero product, and accumulator continues correctly
w17 = [0x4, 0x4, 0x6, 0x4]  # +1, +1, +2, +1
a17 = [0x38, 0x38, 0x38, 0x38]  # all +1
s17 = [0x38, 0x00, 0x38, 0x38]  # normal, ZERO, normal, normal
acc17 = compute_accum(w17, a17, s17)
print_test(16, "T17_SCALE_ZERO_MID", "scale=0 in middle of accumulation",
           w17, a17, s17, acc17)

# ============================================================================
# T18: fp8 near-saturation activation values (e=9, e=10 edge)
# ============================================================================
# fp8 values that push the activation decode near/at the 2047 saturation limit
# 0x48: e=9,m=0 → base=8<<7=1024 (below sat)
# 0x4F: e=9,m=7 → base=15<<7=1920 (below sat)
# 0x50: e=10,m=0 → base=8<<8=2048 (>2047, clips to 2047)
# 0x57: e=10,m=7 → base=15<<8=3840 (>2047, clips to 2047)
# 0x58: e=11,m=0 → base=8<<9=4096 (>2047, clips to 2047)
w18 = [0x7, 0x7, 0x6, 0x4, 0x4]  # +3, +3, +2, +1, +1
a18 = [0x48, 0x4F, 0x50, 0x57, 0x58]  # near/at saturation
s18 = [0x38] * 5  # unity scale
acc18 = compute_accum(w18, a18, s18)
print_test(17, "T18_FP8_NEAR_SAT", "fp8 activations near/at decode saturation",
           w18, a18, s18, acc18)

# ============================================================================
# T19: fp8 subnormals with negative fp4 weights (sign interaction)
# ============================================================================
# Subnormals are tiny (m/2 after decode). Test negative fp4 × subnormal fp8
# to verify sign handling in the subnormal decode path.
w19 = [0xC, 0xC, 0xA, 0x6, 0x4]  # -1, -1, -0.5, +2, +1
a19 = [0x07, 0x03, 0x01, 0x05, 0x07]  # subnorm m=7→3, m=3→1, m=1→0, m=5→2, m=7→3
s19 = [0x38] * 5  # unity scale
acc19 = compute_accum(w19, a19, s19)
print_test(18, "T19_FP8_SUBNORM_SIGNED", "fp8 subnormals x mixed-sign fp4 weights",
           w19, a19, s19, acc19)

# Summary
print()
print("=== SUMMARY ===")
for i, (name, acc) in enumerate([
    ("T15", acc15), ("T16", acc16), ("T17", acc17), ("T18", acc18), ("T19", acc19)
], start=15):
    print(f"  {name}: expected = 0x{acc & 0xFFFFFFFF:08x} ({acc})")
