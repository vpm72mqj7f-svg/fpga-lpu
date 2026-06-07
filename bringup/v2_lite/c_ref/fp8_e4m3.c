/*
 * fp8_e4m3.c -- FP8 E4M3 reference implementation
 *
 * All arithmetic is performed in IEEE 754 single precision and then rounded
 * back to FP8 with round-to-nearest-even (ties-to-even) semantics.
 *
 * The E4M3 format has no infinity; exponent=15 is reserved for NaN only.
 * Per convention we treat a NaN encoding as zero on decode.
 */

#include "fp8_e4m3.h"
#include <math.h>
#include <stdint.h>

/* ---------------------------------------------------------------------------
 *  internal helpers
 * ---------------------------------------------------------------------------
 */

static const float FP8_MAX_NORMAL     = 240.0f;        /* 2^7 × 1.875        */
static const float FP8_MIN_SUBNORMAL  = 1.0f / 512.0f;  /* 2^-9               */
static const float FP8_HALF_SUBNORMAL = FP8_MIN_SUBNORMAL * 0.5f; /* 2^-10   */

/*
 * round_to_nearest_even:
 *   Given a floating-point value `x` and an integer resolution `step` (which
 *   must be a power-of-two fraction), return the integer that represents the
 *   quantized value.  Ties are broken toward the even integer.
 *
 *   This is equivalent to performing "multiply by 1/step, round to nearest
 *   integer with ties-to-even" in infinite precision, but we approximate it
 *   with double-precision arithmetic to avoid rounding noise.
 */
static int round_to_nearest_even(double x, double step)
{
    double scaled = x / step;
    double flr    = floor(scaled);
    double frac   = scaled - flr;

    if (frac < 0.5) {
        return (int)flr;
    } else if (frac > 0.5) {
        return (int)flr + 1;
    } else {
        /* tie — round to even */
        int i = (int)flr;
        return (i & 1) ? i + 1 : i;
    }
}

/* ---------------------------------------------------------------------------
 *  fp8_encode  — IEEE 754 float → FP8 E4M3
 * ---------------------------------------------------------------------------
 *
 *  Encoding strategy:
 *    1. Handle special inputs (NaN / Inf → 0, zero, overflow).
 *    2. Determine the "ideal" exponent field E = floor(log2(|v|)) + 7.
 *    3. If E ≤ 0 → subnormal range (quantize with step = 2^-9).
 *    4. If E ∈ [1,14] → normal range (quantize mantissa to 3 bits).
 *    5. Round-to-nearest-even with carry propagation across exponent boundary.
 */
uint8_t fp8_encode(float value)
{
    /* ---- NaN / Inf → 0 ---- */
    if (isnan(value) || isinf(value)) {
        return 0;
    }

    /* ---- sign extraction ---- */
    uint8_t sign = (value < 0.0f) ? 0x80 : 0;
    float absval  = sign ? -value : value;

    /* ---- zero ---- */
    if (absval == 0.0f) {
        return sign;
    }

    /* ---- overflow → saturate to max normal ---- */
    if (absval >= FP8_MAX_NORMAL) {
        return sign | 0x77;  /* S | E=14 | M=7 */
    }

    /* ---- underflow (round to zero) ---- */
    if (absval < FP8_HALF_SUBNORMAL) {
        return sign;
    }

    /*
     * Determine exponent field E such that 2^(E-7) is the scale.
     * For normal values:  floor(log2(absval)) = E - 7 + floor(log2(1+M/8))
     *  Since (1+M/8) ∈ [1.0, 1.875), floor(log2(1+M/8)) is 0.
     *  Therefore E = floor(log2(absval)) + 7.
     */
    int log2_val = (int)floorf(log2f(absval));
    int E        = log2_val + 7;

    /* ---- subnormal (E ≤ 0) ---- */
    if (E <= 0) {
        /*
         * Value = M/8 × 2^-6 = M × 2^-9.
         * Quantize M = round(absval × 512) with ties-to-even.
         * M is a 3-bit integer ∈ [1, 7].
         */
        int M = round_to_nearest_even((double)absval, (double)FP8_MIN_SUBNORMAL);
        if (M >= 8) {
            /* Rounding crosses into normal — represent as min normal */
            return sign | (1 << 3) | 0;  /* E=1, M=0 */
        }
        if (M == 0) {
            return sign;  /* underflow to zero */
        }
        return sign | (uint8_t)M;
    }

    /* ---- overflow after rounding adjustment ---- */
    if (E >= 15) {
        return sign | 0x77;
    }

    /* ---- normal (E ∈ [1,14]) ---- */
    /*
     * absval = 2^(E-7) × (1 + M/8)
     *   →  (1 + M/8) = absval / 2^(E-7)
     *   →  M = ((absval / 2^(E-7)) - 1) × 8
     *
     * Mantissa step size at this exponent is 2^(E-7) / 8.
     */
    float  scale  = ldexpf(1.0f, E - 7);
    double frac   = (double)absval / (double)scale;  /* in [1.0, 2.0) */
    double mstep  = 1.0 / 8.0;                       /* mantissa LSB as fraction of (1+M/8) */
    int    M      = round_to_nearest_even(frac - 1.0, mstep);

    if (M >= 8) {
        /* carry into exponent */
        M = 0;
        E++;
        if (E >= 15) {
            return sign | 0x77;  /* overflow to max */
        }
    }

    return sign | ((uint8_t)E << 3) | (uint8_t)M;
}

/* ---------------------------------------------------------------------------
 *  fp8_decode  — FP8 E4M3 → IEEE 754 float
 * ---------------------------------------------------------------------------
 */
float fp8_decode(uint8_t bits)
{
    uint8_t sign = (bits >> 7) & 1;
    uint8_t exp  = (bits >> 3) & 0xF;
    uint8_t mant =  bits       & 0x7;

    /* NaN → 0 (per E4M3 convention: no Inf, NaN maps to zero) */
    if (exp == 15) {
        return 0.0f;
    }

    if (exp == 0) {
        if (mant == 0) {
            /* signed zero */
            return sign ? -0.0f : 0.0f;
        }
        /* subnormal:  (-1)^S × 2^(-6) × (M / 8) */
        float val = ((float)mant) * (1.0f / 8.0f) * (1.0f / 64.0f);
        return sign ? -val : val;
    }

    /* normal:  (-1)^S × 2^(E-7) × (1 + M/8) */
    float val = ldexpf(1.0f + ((float)mant) * (1.0f / 8.0f), (int)exp - 7);
    return sign ? -val : val;
}

/* ---------------------------------------------------------------------------
 *  fp8_mul  —  fp8 × fp8 → fp8  (with rounding)
 * ---------------------------------------------------------------------------
 */
uint8_t fp8_mul(uint8_t a, uint8_t b)
{
    float fa = fp8_decode(a);
    float fb = fp8_decode(b);
    return fp8_encode(fa * fb);
}

/* ---------------------------------------------------------------------------
 *  fp8_mul_to_float  —  fp8 × fp8 → float  (exact product, no fp8 rounding)
 * ---------------------------------------------------------------------------
 */
float fp8_mul_to_float(uint8_t a, uint8_t b)
{
    float fa = fp8_decode(a);
    float fb = fp8_decode(b);
    return fa * fb;
}

/* ---------------------------------------------------------------------------
 *  fp8_add  —  fp8 + fp8 → fp8
 * ---------------------------------------------------------------------------
 */
uint8_t fp8_add(uint8_t a, uint8_t b)
{
    float fa = fp8_decode(a);
    float fb = fp8_decode(b);
    return fp8_encode(fa + fb);
}

/* ---------------------------------------------------------------------------
 *  fp8_to_fp16  —  FP8 E4M3 → IEEE 754 binary16
 * ---------------------------------------------------------------------------
 *
 *  IEEE 754 binary16 (half precision):
 *    sign[15] | exp[14:10] (5 bits, bias=15) | mantissa[9:0] (10 bits)
 *    Normal:    2^(E-15) × (1 + M/1024),   E ∈ [1, 30]
 *    Subnormal: 2^(-14)  × (M/1024),        E=0, M ∈ [1, 1023]
 *    Inf/NaN:   E=31
 */
uint16_t fp8_to_fp16(uint8_t bits)
{
    uint8_t  sign8  = (bits >> 7) & 1;
    uint8_t  exp8   = (bits >> 3) & 0xF;
    uint8_t  mant8  =  bits       & 0x7;

    uint16_t sign16 = (uint16_t)sign8 << 15;

    /* ---- zero ---- */
    if (exp8 == 0 && mant8 == 0) {
        return sign16;
    }

    /* ---- NaN (E=15) → map to fp16 NaN ---- */
    if (exp8 == 15) {
        /* fp16 NaN: E=31, mantissa != 0 */
        return sign16 | (31U << 10) | 1U;
    }

    if (exp8 == 0) {
        /* ---- subnormal fp8: value = mant8 × 2^-9 ----
         *
         * Normalize: find leading 1 in mant8 (mant8 ∈ [1, 7]).
         *   k = position of MSB (0-indexed from LSB), k ∈ [0, 2].
         *   mant8 = (1 << k) + rem,   rem < (1 << k).
         *   value = (1 + rem/2^k) × 2^(k-9).
         *
         * fp16 normal representation:
         *   E16 = (k - 9) + 15 = k + 6      (always ≥ 6 ≥ 1, so always normal)
         *   M16 = (rem / 2^k) × 1024 = rem × 2^(10-k) = rem << (10 - k)
         */
        int k = 2;
        while (k >= 0 && ((mant8 >> k) & 1) == 0) {
            k--;
        }
        /* k ∈ [0, 2] (guaranteed because mant8 > 0) */

        uint16_t rem   = mant8 & ((1U << k) - 1); /* lower bits                    */
        uint16_t exp16 = (uint16_t)(k + 6);        /* E16 ∈ [6, 8], always normal  */
        uint16_t man16;

        if (k > 0) {
            man16 = rem << (10 - k);
        } else {
            man16 = 0;  /* k=0 → mant8=1, rem=0 */
        }

        return sign16 | (exp16 << 10) | (man16 & 0x3FF);
    }

    /* ---- normal fp8 ----
     *   value = 2^(E8-7) × (1 + M8/8)
     *         = 2^(E8-7) × (1 + M8×128 / 1024)
     *
     *   E16 = (E8 - 7) + 15 = E8 + 8      (E16 ∈ [9, 22])
     *   M16 = M8 × 128 = M8 << 7
     */
    uint16_t exp16 = (uint16_t)exp8 + 8;
    uint16_t man16 = (uint16_t)mant8 << 7;

    return sign16 | (exp16 << 10) | (man16 & 0x3FF);
}

/* ---------------------------------------------------------------------------
 *  utility predicates
 * ---------------------------------------------------------------------------
 */

int fp8_is_zero(uint8_t bits)
{
    /* Zero if exponent and mantissa are both 0 (sign is ignored) */
    return ((bits & 0x7F) == 0);
}

int fp8_is_nan(uint8_t bits)
{
    /* NaN if exponent field is 15 regardless of mantissa */
    return ((bits >> 3) & 0xF) == 15;
}
