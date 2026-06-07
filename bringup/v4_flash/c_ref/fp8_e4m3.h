/*
 * fp8_e4m3.h -- FP8 E4M3 reference implementation (header)
 *
 * FP8 E4M3 format: 1 sign, 4 exponent (bias=7), 3 mantissa
 *   Exponent=15: NaN (no Inf in E4M3; treated as zero)
 *   Exponent=0:  subnormal (mantissa != 0) or signed zero (mantissa == 0)
 *
 * Value (normal):    (-1)^S × 2^(E-7) × (1 + M/8)
 * Value (subnormal): (-1)^S × 2^(-6) × (M/8)
 *
 * Range:
 *   Max normal:         ±240         (S=0/1, E=14, M=7)
 *   Min positive normal: 2^-6 ≈ 0.015625  (S=0, E=1, M=0)
 *   Min pos. subnormal:  2^-9 ≈ 0.001953  (S=0, E=0, M=1)
 */

#ifndef FP8_E4M3_H
#define FP8_E4M3_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 *  encode / decode
 * ---------------------------------------------------------------------------
 *  fp8_encode:  IEEE 754 float → FP8 E4M3 (round-to-nearest-even; NaN → 0)
 *  fp8_decode:  FP8 E4M3 → IEEE 754 float (NaN → 0.0f)
 */
uint8_t fp8_encode(float value);
float   fp8_decode(uint8_t bits);

/* ---------------------------------------------------------------------------
 *  arithmetic  (all use round-to-nearest-even for the final result)
 * ---------------------------------------------------------------------------
 *  fp8_mul:          fp8 × fp8 → fp8  (with rounding)
 *  fp8_mul_to_float: fp8 × fp8 → float (exact product, no fp8 rounding)
 *  fp8_add:          fp8 + fp8 → fp8
 */
uint8_t fp8_mul(uint8_t a, uint8_t b);
float   fp8_mul_to_float(uint8_t a, uint8_t b);
uint8_t fp8_add(uint8_t a, uint8_t b);

/* ---------------------------------------------------------------------------
 *  type conversion
 * ---------------------------------------------------------------------------
 *  fp8_to_fp16:  FP8 E4M3 → IEEE 754 binary16 (half precision)
 */
uint16_t fp8_to_fp16(uint8_t bits);

/* ---------------------------------------------------------------------------
 *  utility predicates
 * ---------------------------------------------------------------------------
 */
int fp8_is_zero(uint8_t bits);
int fp8_is_nan(uint8_t bits);

#ifdef __cplusplus
}
#endif

#endif /* FP8_E4M3_H */
