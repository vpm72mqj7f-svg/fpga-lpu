"""
FP8 E4M3 Arithmetic Module — Golden Reference Model
====================================================
DeepSeek V4-Flash FFN Engine

Format: 1 sign | 4 exponent (bias=7) | 3 mantissa
  Bit layout: [s][e3 e2 e1 e0][m2 m1 m0] = bits[7:0]

Encoding:
  exp=0,  mant=0  : +0 / -0   (sign preserved)
  exp=0,  mant>0  : subnormal = (-1)^s * 2^{-6} * mant/8
  exp=1..14       : normal    = (-1)^s * 2^{exp-7} * (1 + mant/8)
  exp=15          : NaN       — saturates to 0 on decode (E4M3 has no Inf)

Range:
  Max normal:    +/- 240.0       (0bS_1110_111)
  Min normal:    +/- 2^{-6}      = 0.015625    (0bS_0001_000)
  Max subnormal: +/- 7 * 2^{-9}  = 0.013671875
  Min subnormal: +/- 2^{-9}      = 0.001953125

Rounding: round-to-nearest, ties-to-even (RNE) on encode / multiply / add.
"""

from __future__ import annotations
import math
import numpy as np

# ---------------------------------------------------------------------------
# Format constants (module-level for quick access)
# ---------------------------------------------------------------------------
_EXP_BIAS: int = 7
_MANT_BITS: int = 3
_MANT_DIV: float = float(1 << _MANT_BITS)     # 8.0
_MAX_NORMAL: float = 240.0
_MIN_NORMAL: float = 2.0 ** -6                # 0.015625
_MIN_SUBNORMAL: float = 2.0 ** -9              # 0.001953125


class FP8_E4M3:
    """FP8 E4M3 arithmetic — golden reference model.

    All arithmetic operations (multiply, add) decode operands to float,
    compute in float32, then re-encode to FP8 with round-to-nearest-even.
    ``multiply_acc`` returns the exact float32 product for accumulation
    in wider precision.
    """

    # ------------------------------------------------------------------
    # Decode: 8-bit encoding → Python float
    # ------------------------------------------------------------------

    @staticmethod
    def decode(bits: int) -> float:
        """Convert an 8-bit FP8 E4M3 encoding to a Python float.

        Args:
            bits: 8-bit integer (only lower 8 bits are used).

        Returns:
            The floating-point value.  NaN encodings (exp=15) return 0.0.
            Subnormals (exp=0, mant>0) are fully supported.  Sign of zero
            is preserved for ±0 inputs.
        """
        b: int = bits & 0xFF
        sign: int = (b >> 7) & 1
        exp: int = (b >> 3) & 0xF
        mant: int = b & 0x7

        if exp == 0:
            if mant == 0:
                return -0.0 if sign else 0.0       # ±0
            # Subnormal: value = mant/8 * 2^{-6} = mant * 2^{-9}
            val: float = mant / _MANT_DIV * _MIN_NORMAL
        elif exp == 15:
            return 0.0                             # NaN → 0
        else:
            # Normal: value = (1 + mant/8) * 2^{exp-7}
            val = (1.0 + mant / _MANT_DIV) * (2.0 ** (exp - _EXP_BIAS))

        return -val if sign else val

    # ------------------------------------------------------------------
    # Encode: Python float → 8-bit FP8  (RNE)
    # ------------------------------------------------------------------

    @staticmethod
    def encode(value: float) -> int:
        """Convert a Python float to FP8 E4M3 (round-to-nearest, ties-to-even).

        Args:
            value: Any finite float.  NaN / Inf inputs are saturated to 0.

        Returns:
            8-bit integer encoding.
        """
        # --- NaN / Inf → 0 ---
        if math.isnan(value) or math.isinf(value):
            return 0x00

        # --- Zero ---
        if value == 0.0:
            return 0x80 if math.copysign(1.0, value) < 0 else 0x00

        sign: int = 1 if value < 0.0 else 0
        abs_val: float = abs(value)

        # --- Overflow: clamp to max normal ---
        if abs_val >= _MAX_NORMAL:
            return (sign << 7) | 0x77

        # --- Exact-match enumeration over all representable FP8 values ---
        # There are only 15×8 = 120 non-zero, non-NaN encodings.
        # Brute-force guarantees bit-exact round-to-nearest-even for a
        # golden model without floating-point edge-case bugs.
        best_bits: int = 0
        best_error: float = float("inf")

        for e in range(15):          # 0 … 14  (skip NaN exp=15)
            for m in range(8):       # 0 … 7
                if e == 0 and m == 0:
                    continue          # zero handled above

                # Exact FP8 value for this (exp, mant) pair
                if e == 0:
                    v: float = m / _MANT_DIV * _MIN_NORMAL
                else:
                    v = (1.0 + m / _MANT_DIV) * (2.0 ** (e - _EXP_BIAS))

                err: float = abs(abs_val - v)

                if err < best_error:
                    best_error = err
                    best_bits = (e << 3) | m
                elif err == best_error:
                    # Ties-to-even: prefer the encoding whose mantissa
                    # LSB is 0.  This correctly handles normal→normal,
                    # sub→sub, and subnormal→normal boundaries.
                    if (m & 1) == 0:
                        best_bits = (e << 3) | m

        return (sign << 7) | best_bits

    # ------------------------------------------------------------------
    # Multiply:  fp8 × fp8 → fp8  (rounded)
    # ------------------------------------------------------------------

    @staticmethod
    def multiply(a: int, b: int) -> int:
        """Multiply two FP8 values, returning FP8 (round-to-nearest-even).

        Args:
            a: 8-bit FP8 encoding of first operand.
            b: 8-bit FP8 encoding of second operand.

        Returns:
            8-bit FP8 encoding of a × b.
        """
        fa: float = FP8_E4M3.decode(a)
        fb: float = FP8_E4M3.decode(b)
        return FP8_E4M3.encode(fa * fb)

    # ------------------------------------------------------------------
    # Multiply-accumulate:  fp8 × fp8 → float  (exact, no rounding)
    # ------------------------------------------------------------------

    @staticmethod
    def multiply_acc(a: int, b: int) -> float:
        """Multiply two FP8 values, returning an exact float32 product.

        Use this for dot-product accumulation where rounding is deferred
        until the final sum.  No FP8 rounding is applied to the product.

        Args:
            a: 8-bit FP8 encoding of first operand.
            b: 8-bit FP8 encoding of second operand.

        Returns:
            Exact float32 product a × b.
        """
        fa: float = FP8_E4M3.decode(a)
        fb: float = FP8_E4M3.decode(b)
        return fa * fb

    # ------------------------------------------------------------------
    # Add:  fp8 + fp8 → fp8  (rounded)
    # ------------------------------------------------------------------

    @staticmethod
    def add(a: int, b: int) -> int:
        """Add two FP8 values, returning FP8 (round-to-nearest-even).

        Args:
            a: 8-bit FP8 encoding of first operand.
            b: 8-bit FP8 encoding of second operand.

        Returns:
            8-bit FP8 encoding of a + b.
        """
        fa: float = FP8_E4M3.decode(a)
        fb: float = FP8_E4M3.decode(b)
        return FP8_E4M3.encode(fa + fb)

    # ------------------------------------------------------------------
    # Convert to FP16:  fp8 → IEEE 754 half-precision
    # ------------------------------------------------------------------

    @staticmethod
    def to_fp16(bits: int) -> int:
        """Convert an FP8 E4M3 value to IEEE 754 half-precision (FP16).

        FP16 format: 1 sign | 5 exponent (bias=15) | 10 mantissa.
        This is a lossless up-conversion since FP16 has more range and
        precision than FP8 E4M3.

        Args:
            bits: 8-bit FP8 encoding.

        Returns:
            16-bit FP16 encoding (bits[15:0]).
        """
        val: float = FP8_E4M3.decode(bits)
        return _float_to_fp16(val)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_sign(bits: int) -> int:
        """Extract sign bit (0 or 1)."""
        return (bits >> 7) & 1

    @staticmethod
    def get_exponent(bits: int) -> int:
        """Extract raw exponent field (0..15)."""
        return (bits >> 3) & 0xF

    @staticmethod
    def get_mantissa(bits: int) -> int:
        """Extract raw mantissa field (0..7)."""
        return bits & 0x7


# =============================================================================
# FP16 conversion helper (module-level, shared with to_fp16)
# =============================================================================

def _float_to_fp16(val: float) -> int:
    """Convert a Python float to IEEE 754 binary16 (round-to-nearest-even).

    This is a standalone helper used by ``FP8_E4M3.to_fp16``.
    """
    if val == 0.0:
        return 0x8000 if math.copysign(1.0, val) < 0 else 0x0000

    if math.isnan(val) or math.isinf(val):
        return 0x0000     # Should not reach here from fp8 inputs

    sign: int = 1 if val < 0.0 else 0
    abs_val: float = abs(val)

    # FP16 constants
    FP16_MAX: float = 65504.0            # (1 + 1023/1024) * 2^15
    FP16_MIN_SUB: float = 2.0 ** -24    # 2^{-14} * 1/1024

    if abs_val >= FP16_MAX:
        return (sign << 15) | 0x7BFF    # max fp16

    if abs_val < FP16_MIN_SUB:
        return sign << 15               # flush to zero

    # Biased exponent for fp16
    e16: int = int(math.floor(math.log2(abs_val))) + 15

    if e16 <= 0:
        # Subnormal: value = m / 1024 * 2^{-14}
        m16: int = int(round(abs_val * float(1 << 24)))   # * 2^{14} * 1024
        if m16 >= 1024:
            e16 = 1
            m16 = 0
        else:
            e16 = 0
        m16 = max(0, min(1023, m16))
    else:
        if e16 >= 31:
            return (sign << 15) | 0x7C00    # Inf
        # Normal: value = (1 + m/1024) * 2^{e16-15}
        m16 = int(round((abs_val / (2.0 ** (e16 - 15)) - 1.0) * 1024.0))
        if m16 >= 1024:
            m16 = 0
            e16 += 1
            if e16 >= 31:
                return (sign << 15) | 0x7C00
        m16 = max(0, min(1023, m16))

    return (sign << 15) | (e16 << 10) | m16


# =============================================================================
# Quick inline FP16 decoder (for round-trip tests only)
# =============================================================================

def _fp16_to_float(bits: int) -> float:
    """Decode a 16-bit FP16 value to float.  Used for cross-checking."""
    s: int = (bits >> 15) & 1
    e: int = (bits >> 10) & 0x1F
    m: int = bits & 0x3FF
    sign: float = -1.0 if s else 1.0
    if e == 0:
        return sign * (m / 1024.0) * (2.0 ** -14)
    return sign * (1.0 + m / 1024.0) * (2.0 ** (e - 15))


# =============================================================================
# Module-level NumPy-vectorized convenience wrappers
# =============================================================================
#
# IMPORTANT — NaN convention compatibility:
#   The FP8_E4M3 class uses the NEW golden-model convention (exp=15 → 0.0).
#   These numpy wrappers use the OLD convention (exp=15 → float('nan')) via
#   a precomputed 256-entry LUT, to preserve backward compatibility with
#   ffn_pipeline.py and test_e2e_ffn.py which both expect:
#       assert np.isnan(unpack_fp8_scalar(pack_fp8_scalar(float('nan'))))
#   and:
#       np.isnan(unpack_fp8_scalar(...))  checks on NaN-encoded weights.
#
#   NEW code should use FP8_E4M3 class methods directly.
#   These wrappers are for existing callers that rely on NaN-preservation.
# =============================================================================

# -- float32 LUT indexed by FP8 byte [0..255] (old NaN convention) --

_FP8_TO_F32_LUT: "np.ndarray" = np.zeros(256, dtype=np.float32)

for _b in range(256):
    _s: int = (_b >> 7) & 1
    _e: int = (_b >> 3) & 0xF
    _m: int = _b & 0x7
    if _e == 0:
        _v: float = 0.0 if _m == 0 else (_m / 8.0) * (2.0 ** -6)
    elif _e == 15:
        _v = float("nan")                     # OLD: keep NaN as NaN
    else:
        _v = (1.0 + _m / 8.0) * (2.0 ** (_e - 7))
    _FP8_TO_F32_LUT[_b] = -_v if _s else _v


def fp8_to_float(fp8_bytes: np.ndarray) -> np.ndarray:
    """Convert uint8 FP8 array to float32 (vectorized, old NaN convention)."""
    fp8_bytes = np.asarray(fp8_bytes, dtype=np.uint8)
    return _FP8_TO_F32_LUT[fp8_bytes]


def _float_to_fp8_scalar(v: float) -> int:
    """Convert a single float32 to FP8 byte (old NaN convention, RNE)."""
    if np.isnan(v):
        return 0x7F                            # OLD: NaN → 0x7F
    if v == 0.0:
        return 0x00
    sign: int = int(v < 0.0)
    abs_v: float = abs(v)
    if abs_v >= 240.0:
        return 0x77 | (sign << 7)

    frac, iexp = np.frexp(abs_v)               # frac in [0.5, 1.0)
    biased: int = iexp - 1 + _EXP_BIAS

    if biased <= 0:
        sub_val: float = abs_v * (2.0 ** 6) * 8.0
        mant: int = int(np.rint(sub_val))
        if mant >= 8:
            biased = 1
            mant = 0
        elif mant < 0:
            mant = 0
        exp: int = 0
    elif biased >= 15:
        exp = 14
        mant = 7
    else:
        mant_cont: float = (abs_v / (2.0 ** (biased - _EXP_BIAS)) - 1.0) * 8.0
        mant = int(np.rint(mant_cont))
        if mant >= 8:
            mant = 0
            biased += 1
        elif mant < 0:
            mant = 0
        if biased >= 15:
            exp = 14
            mant = 7
        else:
            exp = biased

    return (sign << 7) | ((exp & 0xF) << 3) | (mant & 0x7)


_float_to_fp8_vec = np.vectorize(_float_to_fp8_scalar, otypes=[np.uint8])


def float_to_fp8(values: np.ndarray) -> np.ndarray:
    """Convert float32 array to FP8 bytes (vectorized, old NaN convention)."""
    values = np.asarray(values, dtype=np.float32)
    return _float_to_fp8_vec(values)


def pack_fp8_scalar(value: float) -> np.uint8:
    """Convert a single Python float to a single FP8 byte (np.uint8)."""
    return np.uint8(_float_to_fp8_scalar(value))


def unpack_fp8_scalar(bits: int) -> float:
    """Convert a single FP8 byte to a Python float."""
    return float(_FP8_TO_F32_LUT[bits & 0xFF])


def quantize_array(arr: np.ndarray) -> np.ndarray:
    """Quantise float32 → FP8 → float32 (simulates FP8 precision loss)."""
    return fp8_to_float(float_to_fp8(arr))


def decode_fp8(fp8_bytes):
    """Alias for fp8_to_float."""
    return fp8_to_float(fp8_bytes)


def encode_fp8(values):
    """Alias for float_to_fp8."""
    return float_to_fp8(values)


# =============================================================================
# Comprehensive Tests
# =============================================================================
if __name__ == "__main__":
    # Convenience aliases
    encode = FP8_E4M3.encode
    decode = FP8_E4M3.decode
    mul = FP8_E4M3.multiply
    mul_acc = FP8_E4M3.multiply_acc
    add = FP8_E4M3.add
    to_fp16 = FP8_E4M3.to_fp16
    ONE = encode(1.0)
    ZERO = encode(0.0)
    NEG_ZERO = encode(-0.0)

    _counts = {'passed': 0, 'failed': 0}  # immutable closure capture

    def check(label: str, cond: bool, detail: str = "") -> None:
        if cond:
            _counts['passed'] += 1
        else:
            _counts['failed'] += 1
            print(f"  FAIL [{label}]: {detail}")

    # ==================================================================
    # 1. Encode / Decode Round-Trip
    # ==================================================================
    print("=" * 60)
    print("1. Encode / Decode Round-Trip")
    print("=" * 60)

    test_values: dict[str, float] = {
        "+0":              0.0,
        "-0":             -0.0,
        "+1.0":            1.0,
        "+1.5":            1.5,
        "+2.0":            2.0,
        "+3.5":            3.5,
        "+7.5":            7.5,
        "+max":            _MAX_NORMAL,         # 240.0
        "+overflow":       999.0,
        "+min_normal":     _MIN_NORMAL,         # 0.015625
        "+min_subnormal":  _MIN_SUBNORMAL,      # 0.001953125
        "+0.0078125":      0.0078125,            # m=4 subnormal
        "+0.00390625":     0.00390625,           # m=2 subnormal
        "-1.0":           -1.0,
        "-max":           -_MAX_NORMAL,
        "-min_normal":    -_MIN_NORMAL,
        "-min_subnormal": -_MIN_SUBNORMAL,
    }

    for name, val in test_values.items():
        bits: int = encode(val)
        dec: float = decode(bits)

        # Compute expected decoded value directly from the bit fields
        s: int = FP8_E4M3.get_sign(bits)
        e: int = FP8_E4M3.get_exponent(bits)
        m: int = FP8_E4M3.get_mantissa(bits)

        if e == 0 and m == 0:
            expected: float = -0.0 if s else 0.0
        elif e == 0:
            expected = -(m / 8.0 * _MIN_NORMAL) if s else (m / 8.0 * _MIN_NORMAL)
        else:
            expected = (-(1.0 + m / 8.0) * (2.0 ** (e - 7))
                        if s else (1.0 + m / 8.0) * (2.0 ** (e - 7)))

        check(f"encode({name}) decode matches",
              dec == expected,
              f"bits=0x{bits:02X} dec={dec} expected={expected}")

    print(f"  Round-trip: {_counts['passed']} pass, {_counts['failed']} fail\n")

    # ==================================================================
    # 2. Exact Bit Patterns for Key Values
    # ==================================================================
    print("=" * 60)
    print("2. Exact Bit Patterns")
    print("=" * 60)

    bit_tests = [
        ("+0",           0.0,             0x00),
        ("-0",          -0.0,             0x80),
        ("+1.0",         1.0,             0x38),   # e=7,m=0 → 0b0_0111_000
        ("+min_normal",  _MIN_NORMAL,     0x08),   # e=1,m=0 → 0b0_0001_000
        ("+min_sub",     _MIN_SUBNORMAL,  0x01),   # e=0,m=1 → 0b0_0000_001
        ("+max",         _MAX_NORMAL,     0x77),   # e=14,m=7→ 0b0_1110_111
        ("-max",        -_MAX_NORMAL,     0xF7),   # 0b1_1110_111
        ("NaN",          float("nan"),    0x00),   # saturates to 0
        ("Inf",          float("inf"),    0x00),
        ("+overflow",    999.0,           0x77),
        ("-overflow",   -999.0,           0xF7),
    ]

    for name, val, expected_bits in bit_tests:
        bits: int = encode(val)
        check(f"bits({name})",
              bits == expected_bits,
              f"got 0x{bits:02X} expected 0x{expected_bits:02X}")

    print(f"  Bit-patterns: {_counts['passed']} pass, {_counts['failed']} fail\n")

    # ==================================================================
    # 3. NaN Handling (exp=15 → saturate to 0)
    # ==================================================================
    print("=" * 60)
    print("3. NaN Handling (exp=15 → saturate to 0)")
    print("=" * 60)

    nan_encodings: list[int] = [
        0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F,
        0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF,
    ]
    for nbits in nan_encodings:
        val: float = decode(nbits)
        check(f"decode(NaN 0x{nbits:02X}) == 0",
              val == 0.0,
              f"got {val}")

    # Multiply with NaN operand produces 0
    check("mul(NaN, 1.0) == 0",
          mul(0x7F, ONE) == 0x00,
          f"got 0x{mul(0x7F, ONE):02X}")
    check("mul(1.0, NaN) == 0",
          mul(ONE, 0x7F) == 0x00,
          f"got 0x{mul(ONE, 0x7F):02X}")

    print(f"  NaN: {_counts['passed']} pass, {_counts['failed']} fail\n")

    # ==================================================================
    # 4. Subnormal Encoding / Decoding
    # ==================================================================
    print("=" * 60)
    print("4. Subnormal Handling")
    print("=" * 60)

    sub_tests = [
        (_MIN_SUBNORMAL,             0x01),   # 2^{-9}
        (2 * _MIN_SUBNORMAL,         0x02),
        (3 * _MIN_SUBNORMAL,         0x03),
        (4 * _MIN_SUBNORMAL,         0x04),
        (5 * _MIN_SUBNORMAL,         0x05),
        (6 * _MIN_SUBNORMAL,         0x06),
        (7 * _MIN_SUBNORMAL,         0x07),   # max subnormal
    ]
    for val, expected_bits in sub_tests:
        bits: int = encode(val)
        check(f"encode({val:.10f}) sub",
              bits == expected_bits,
              f"got 0x{bits:02X} expected 0x{expected_bits:02X}")

    # Decode all subnormal encodings
    for m in range(1, 8):
        expected: float = m * _MIN_SUBNORMAL
        bits: int = 0x00 | m
        dec: float = decode(bits)
        check(f"decode(sub m={m})",
              dec == expected,
              f"got {dec:.10f} expected {expected:.10f}")

    # Ties-to-even in subnormals: midpoint between m=3 (odd) and m=4 (even)
    mid_3_4: float = (3 * _MIN_SUBNORMAL + 4 * _MIN_SUBNORMAL) / 2.0
    check("sub tie-break 3→4 (even)",
          encode(mid_3_4) == 0x04,
          f"got 0x{encode(mid_3_4):02X}")

    # Midpoint between m=4 (even) and m=5 (odd) → m=4 (even)
    mid_4_5: float = (4 * _MIN_SUBNORMAL + 5 * _MIN_SUBNORMAL) / 2.0
    check("sub tie-break 4→5 (even)",
          encode(mid_4_5) == 0x04,
          f"got 0x{encode(mid_4_5):02X}")

    print(f"  Subnormal: {_counts['passed']} pass, {_counts['failed']} fail\n")

    # ==================================================================
    # 5. Multiply: fp8 × fp8 → fp8
    # ==================================================================
    print("=" * 60)
    print("5. Multiply (fp8 × fp8 → fp8)")
    print("=" * 60)

    mul_tests = [
        ("1.0", 1.0, "1.0", 1.0, "1.0", 1.0),
        ("1.0", 1.0, "0.0", 0.0, "0.0", 0.0),
        ("0.5", 0.5, "0.5", 0.5, "0.25", 0.25),
        ("2.0", 2.0, "3.0", 3.0, "6.0", 6.0),
        ("-1.0", -1.0, "2.0", 2.0, "-2.0", -2.0),
        ("-1.0", -1.0, "-1.0", -1.0, "1.0", 1.0),
        ("max", _MAX_NORMAL, "0.5", 0.5, "120.0", 120.0),
        ("min_sub", _MIN_SUBNORMAL, "min_sub", _MIN_SUBNORMAL,
         "underflow→0", 0.0),
        ("min_sub", _MIN_SUBNORMAL, "1024.0", 1024.0, "2.0", 2.0),
    ]

    for a_nm, a_v, b_nm, b_v, _exp_nm, exp_v in mul_tests:
        a_bits: int = encode(a_v)
        b_bits: int = encode(b_v)
        result_bits: int = mul(a_bits, b_bits)
        result_val: float = decode(result_bits)
        expected_bits: int = encode(exp_v)

        check(f"mul({a_nm}, {b_nm})",
              result_bits == expected_bits,
              f"got 0x{result_bits:02X} ({result_val}) "
              f"expected 0x{expected_bits:02X} ({exp_v})")

    # Overflow in multiply: 120 × 3 = 360 → clamp to 240
    over_bits: int = mul(encode(120.0), encode(3.0))
    over_val: float = decode(over_bits)
    check("mul(120, 3) saturates to 240",
          over_bits == encode(_MAX_NORMAL),
          f"got 0x{over_bits:02X} ({over_val})")

    # Signed zero results
    check("mul(+0, +1) == +0", mul(ZERO, ONE) == 0x00,
          f"got 0x{mul(ZERO, ONE):02X}")
    check("mul(-0, +1) == -0", mul(NEG_ZERO, ONE) == 0x80,
          f"got 0x{mul(NEG_ZERO, ONE):02X}")
    check("mul(+0, -1) == -0", mul(ZERO, encode(-1.0)) == 0x80,
          f"got 0x{mul(ZERO, encode(-1.0)):02X}")

    print(f"  Multiply: {_counts['passed']} pass, {_counts['failed']} fail\n")

    # ==================================================================
    # 6. multiply_acc: fp8 × fp8 → float32 (no rounding)
    # ==================================================================
    print("=" * 60)
    print("6. multiply_acc (fp8 × fp8 → float32, no rounding)")
    print("=" * 60)

    acc_tests = [
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.25),
        (_MAX_NORMAL, 0.5, 120.0),
        (_MIN_SUBNORMAL, _MIN_SUBNORMAL,
         _MIN_SUBNORMAL * _MIN_SUBNORMAL),
        (2.0, 3.0, 6.0),
        (-1.0, 2.0, -2.0),
    ]

    for a_v, b_v, exp_v in acc_tests:
        a_bits: int = encode(a_v)
        b_bits: int = encode(b_v)
        result: float = mul_acc(a_bits, b_bits)
        check(f"mul_acc({a_v}, {b_v})",
              result == exp_v,
              f"got {result} expected {exp_v}")

    # multiply_acc with NaN operand → 0.0
    check("mul_acc(NaN, 1.0) == 0",
          mul_acc(0x7F, ONE) == 0.0,
          f"got {mul_acc(0x7F, ONE)}")

    print(f"  multiply_acc: {_counts['passed']} pass, {_counts['failed']} fail\n")

    # ==================================================================
    # 7. Add: fp8 + fp8 → fp8
    # ==================================================================
    print("=" * 60)
    print("7. Add (fp8 + fp8 → fp8)")
    print("=" * 60)

    add_tests = [
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0),
        (1.0, -1.0, 0.0),
        (0.5, 0.5, 1.0),
        (120.0, 120.0, 240.0),
        (_MAX_NORMAL, 1.0, _MAX_NORMAL),            # saturate
        (-_MAX_NORMAL, -1.0, -_MAX_NORMAL),         # saturate negative
        (_MIN_SUBNORMAL, _MIN_SUBNORMAL,
         2 * _MIN_SUBNORMAL),
        (1.0, _MIN_SUBNORMAL, 1.0),  # sub lost at fp8 precision
    ]

    for a_v, b_v, exp_v in add_tests:
        a_bits: int = encode(a_v)
        b_bits: int = encode(b_v)
        result_bits: int = add(a_bits, b_bits)
        result_val: float = decode(result_bits)
        expected_bits: int = encode(exp_v)

        check(f"add({a_v}, {b_v})",
              result_bits == expected_bits,
              f"got 0x{result_bits:02X} ({result_val}) "
              f"expected 0x{expected_bits:02X} ({exp_v})")

    # Signed zero addition
    check("add(+0, +0) == +0", add(ZERO, ZERO) == 0x00,
          f"got 0x{add(ZERO, ZERO):02X}")
    check("add(+0, -0) == +0", add(ZERO, NEG_ZERO) == 0x00,
          f"got 0x{add(ZERO, NEG_ZERO):02X}")

    print(f"  Add: {_counts['passed']} pass, {_counts['failed']} fail\n")

    # ==================================================================
    # 8. to_fp16: FP8 → IEEE 754 half-precision
    # ==================================================================
    print("=" * 60)
    print("8. to_fp16 (FP8 E4M3 → IEEE 754 half)")
    print("=" * 60)

    fp16_tests = [
        # (value, expected_fp16_bits, description)
        (0.0,             0x0000, "0.0"),
        (-0.0,            0x8000, "-0.0"),
        (1.0,             0x3C00, "1.0"),       # e=15, m=0
        (-1.0,            0xBC00, "-1.0"),
        (2.0,             0x4000, "2.0"),       # e=16, m=0
        (0.5,             0x3800, "0.5"),       # e=14, m=0
        (0.25,            0x3400, "0.25"),      # e=13, m=0
        (_MIN_NORMAL,     0x2400, "min_normal"),# 2^{-6} → e=9, m=0
        (_MAX_NORMAL,     0x5B80, "240.0"),     # 240 → e=22, m=896
        (_MIN_SUBNORMAL,  0x1000, "min_sub"),   # 2^{-9} → e=6, m=0
    ]

    for val, exp_fp16, desc in fp16_tests:
        fp8_bits: int = encode(val)
        result: int = to_fp16(fp8_bits)
        check(f"to_fp16({desc})",
              result == exp_fp16,
              f"got 0x{result:04X} expected 0x{exp_fp16:04X}")

    # FP8 NaN → FP16 zero
    check("to_fp16(NaN) == 0",
          to_fp16(0x7F) == 0x0000,
          f"got 0x{to_fp16(0x7F):04X}")

    # Round-trip fidelity: fp8 → fp16 → float should match fp8 → float
    for val in [0.0, 1.0, 2.0, 0.5, _MIN_NORMAL, _MAX_NORMAL,
                _MIN_SUBNORMAL, 0.25, 10.0, 64.0, 128.0, -1.0]:
        fp8_bits = encode(val)
        fp16_bits = to_fp16(fp8_bits)
        f16_val = _fp16_to_float(fp16_bits)
        orig_val = decode(fp8_bits)
        check(f"fp16 round-trip({val})",
              abs(f16_val - orig_val) < 0.001,
              f"fp16 decoded={f16_val} orig={orig_val}")

    print(f"  to_fp16: {_counts['passed']} pass, {_counts['failed']} fail\n")

    # ==================================================================
    # 9. Edge Cases & Rounding
    # ==================================================================
    print("=" * 60)
    print("9. Edge Cases & Rounding")
    print("=" * 60)

    # Tiny values flush to zero
    for tiny in [1e-10, 1e-8, 1e-7]:
        check(f"tiny {tiny} → 0",
              encode(tiny) == 0x00,
              f"got 0x{encode(tiny):02X}")

    # Large values saturate at ±240
    for huge in [241.0, 1000.0, 1e10]:
        bits_h = encode(huge)
        check(f"huge {huge} saturates",
              decode(bits_h) == _MAX_NORMAL,
              f"got bits=0x{bits_h:02X} val={decode(bits_h)}")

    for huge_neg in [-241.0, -1000.0, -1e10]:
        bits_hn = encode(huge_neg)
        check(f"huge neg {huge_neg} saturates",
              decode(bits_hn) == -_MAX_NORMAL,
              f"got bits=0x{bits_hn:02X} val={decode(bits_hn)}")

    # --- Subnormal ↔ normal boundary rounding ---
    # max subnormal = 7/512 = 0.013671875
    # min normal    = 2/128 = 1/64 = 0.015625
    # midpoint      = (0.013671875 + 0.015625) / 2 = 0.0146484375
    sub_max: float = 7.0 / 512.0
    norm_min: float = _MIN_NORMAL
    midpoint_sn: float = (sub_max + norm_min) / 2.0

    check("just below sub/normal boundary → max sub",
          encode(midpoint_sn - 1e-9) == 0x07,
          f"got 0x{encode(midpoint_sn - 1e-9):02X}")
    check("just above sub/normal boundary → min normal",
          encode(midpoint_sn + 1e-9) == 0x08,
          f"got 0x{encode(midpoint_sn + 1e-9):02X}")
    check("exact sub/normal midpoint → min normal (ties-to-even)",
          encode(midpoint_sn) == 0x08,
          f"got 0x{encode(midpoint_sn):02X}")

    # --- Mantissa rounding ties-to-even at e=7 (scale=1) ---
    # m=2: (1+2/8)=1.25, m=3: (1+3/8)=1.375, midpoint=1.3125 → m=2 (even)
    check("ties-to-even m=2 vs m=3",
          encode(1.3125) == 0x3A,
          f"got 0x{encode(1.3125):02X} expected 0x3A")
    # m=4: 1.5, m=5: 1.625, midpoint=1.5625 → m=4 (even)
    check("ties-to-even m=4 vs m=5",
          encode(1.5625) == 0x3C,
          f"got 0x{encode(1.5625):02X} expected 0x3C")
    # m=6: 1.75, m=7: 1.875, midpoint=1.8125 → m=6 (even)
    check("ties-to-even m=6 vs m=7",
          encode(1.8125) == 0x3E,
          f"got 0x{encode(1.8125):02X} expected 0x3E")

    print(f"  Edge-cases: {_counts['passed']} pass, {_counts['failed']} fail\n")

    # ==================================================================
    # 10. Full Encoding Range:  encode(decode(bits)) == bits
    # ==================================================================
    print("=" * 60)
    print("10. Full Value Range: encode(decode(bits)) == bits")
    print("=" * 60)

    for b in range(256):
        e: int = (b >> 3) & 0xF
        if e == 15:
            # NaN encodings: decode returns 0, re-encode gives 0x00.
            # This is intentional — NaN inputs are not recoverable.
            continue
        val: float = decode(b)
        re_enc: int = encode(val)
        check(f"round-trip 0x{b:02X}",
              re_enc == b,
              f"val={val} re_encoded=0x{re_enc:02X}")

    print(f"  Range consistency: {_counts['passed']} pass, {_counts['failed']} fail\n")

    # ==================================================================
    # Summary
    # ==================================================================
    total: int = _counts['passed'] + _counts['failed']
    print("=" * 60)
    if _counts['failed'] == 0:
        print(f"RESULTS: {_counts['passed']}/{total} passed — ALL PASSED")
    else:
        print(f"RESULTS: {_counts['passed']}/{total} passed  ({_counts['failed']} FAILED!)")
    print("=" * 60)

    if _counts['failed']:
        raise SystemExit(1)
