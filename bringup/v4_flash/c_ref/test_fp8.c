/*
 * test_fp8.c -- Comprehensive test suite for FP8 E4M3 reference implementation
 *
 * Output format: hex dump ready for cross-comparison with Python golden model.
 * All key values are printed as hex bytes so they can be diffed against a
 * Python-generated reference file.
 */

#include "fp8_e4m3.h"
#include <math.h>
#include <stdio.h>

/* ---------------------------------------------------------------------------
 *  test helpers
 * ---------------------------------------------------------------------------
 */

static int tests_run  = 0;
static int tests_pass = 0;
static int tests_fail = 0;

#define TEST(name)  do { tests_run++; printf("  %-50s ", name); } while (0)
#define OK()        do { tests_pass++; printf("PASS\n"); } while (0)
#define FAIL(fmt, ...) \
    do { \
        tests_fail++; \
        printf("FAIL: " fmt "\n", ##__VA_ARGS__); \
    } while (0)

#define ASSERT_EQ_INT(expected, actual, name_str) \
    do { \
        if ((expected) != (actual)) { \
            FAIL("%s: expected %d, got %d", name_str, (int)(expected), (int)(actual)); \
        } else { OK(); } \
    } while (0)

#define ASSERT_EQ_UINT8(expected, actual, name_str) \
    do { \
        if ((expected) != (actual)) { \
            FAIL("%s: expected 0x%02X, got 0x%02X", name_str, \
                 (unsigned)(expected), (unsigned)(actual)); \
        } else { OK(); } \
    } while (0)

#define ASSERT_EQ_UINT16(expected, actual, name_str) \
    do { \
        if ((expected) != (actual)) { \
            FAIL("%s: expected 0x%04X, got 0x%04X", name_str, \
                 (unsigned)(expected), (unsigned)(actual)); \
        } else { OK(); } \
    } while (0)

#define ASSERT_FLOAT_EQ(expected, actual, tol, name_str) \
    do { \
        float diff = (actual) - (expected); \
        if (diff < 0) diff = -diff; \
        if (diff > (tol)) { \
            FAIL("%s: expected %.9g, got %.9g (diff %.9g)", name_str, \
                 (double)(expected), (double)(actual), (double)diff); \
        } else { OK(); } \
    } while (0)

/* ---------------------------------------------------------------------------
 *  test_decode_encode  — verify round-trip for known reference values
 * ---------------------------------------------------------------------------
 */
static void test_decode_encode(void)
{
    printf("\n--- Decode / Encode Round-trip ---\n");

    /* Zero */
    TEST("decode(0x00)");  ASSERT_FLOAT_EQ(0.0f,  fp8_decode(0x00), 0.0f, "0x00");
    TEST("decode(0x80)");  ASSERT_FLOAT_EQ(-0.0f, fp8_decode(0x80), 0.0f, "0x80");
    TEST("encode(0.0)");   ASSERT_EQ_UINT8(0x00, fp8_encode(0.0f),  "encode(0.0)");
    TEST("encode(-0.0)");  ASSERT_EQ_UINT8(0x80, fp8_encode(-0.0f), "encode(-0.0)");

    /* Min positive subnormal: 2^-9 */
    TEST("decode min subnormal");
    ASSERT_FLOAT_EQ(1.0f/512.0f, fp8_decode(0x01), 1e-12f, "0x01");

    TEST("encode min subnormal");
    ASSERT_EQ_UINT8(0x01, fp8_encode(1.0f/512.0f), "encode(2^-9)");

    /* Min positive normal: 2^-6 */
    TEST("decode min normal");
    ASSERT_FLOAT_EQ(1.0f/64.0f, fp8_decode(0x08), 1e-12f, "0x08");

    TEST("encode min normal");
    ASSERT_EQ_UINT8(0x08, fp8_encode(1.0f/64.0f), "encode(2^-6)");

    /* 1.0 */
    TEST("decode 1.0");
    ASSERT_FLOAT_EQ(1.0f, fp8_decode(0x38), 1e-12f, "1.0 (0x38)");

    TEST("encode 1.0");
    ASSERT_EQ_UINT8(0x38, fp8_encode(1.0f), "encode(1.0)");

    /* Max normal: 240 */
    TEST("decode max normal");
    ASSERT_FLOAT_EQ(240.0f, fp8_decode(0x77), 1e-9f, "0x77");

    TEST("encode max normal");
    ASSERT_EQ_UINT8(0x77, fp8_encode(240.0f), "encode(240)");

    /* Overflow: values >= 240 saturate */
    TEST("overflow → max");
    ASSERT_EQ_UINT8(0x77, fp8_encode(1000.0f), "encode(1000)");

    /* NaN → 0 */
    TEST("NaN → 0");
    ASSERT_EQ_UINT8(0x00, fp8_encode(NAN), "encode(NaN)");

    /* Infinity → 0 */
    TEST("Inf → 0");
    ASSERT_EQ_UINT8(0x00, fp8_encode(INFINITY), "encode(Inf)");

    /* Underflow: value < half of min subnormal → zero */
    TEST("underflow to zero");
    ASSERT_EQ_UINT8(0x00, fp8_encode(1.0f/2048.0f), "encode(2^-11)");

    /* -1.0 */
    TEST("decode -1.0");
    ASSERT_FLOAT_EQ(-1.0f, fp8_decode(0xB8), 1e-12f, "-1.0 (0xB8)");

    TEST("encode -1.0");
    ASSERT_EQ_UINT8(0xB8, fp8_encode(-1.0f), "encode(-1.0)");

    /* Subnormal round-to-even at boundary */
    TEST("subnormal: 3.5 × 2^-9 → M=4 even");
    ASSERT_EQ_UINT8(0x04, fp8_encode(3.5f / 512.0f), "encode(3.5/512)");

    TEST("subnormal: 3.5 × 2^-9 (neg) → M=4 even");
    ASSERT_EQ_UINT8(0x84, fp8_encode(-3.5f / 512.0f), "encode(-3.5/512)");

    /* Mantissa tie: round to even */
    /* 2^(E-7) × 1.5: should round mantissa=4 to even */
    /* E=3 → scale=2^-4=0.0625. Value=0.0625+0.5*step=0.0625*1.5=0.09375 */
    /* Actually: value at M=4 is 2^(3-7)*(1+4/8)=0.0625*1.5=0.09375 */
    TEST("round-tie-even normal");
    ASSERT_EQ_UINT8(0x1C, fp8_encode(0.09375f), "encode(0.09375) E=3 M=4");
}

/* ---------------------------------------------------------------------------
 *  test_predicates
 * ---------------------------------------------------------------------------
 */
static void test_predicates(void)
{
    printf("\n--- Predicates ---\n");

    TEST("is_zero(0x00)");
    ASSERT_EQ_INT(1, fp8_is_zero(0x00), "0x00");

    TEST("is_zero(0x80)");
    ASSERT_EQ_INT(1, fp8_is_zero(0x80), "-0 (0x80)");

    TEST("is_zero(0x01)");
    ASSERT_EQ_INT(0, fp8_is_zero(0x01), "subnormal");

    TEST("is_zero(0x38)");
    ASSERT_EQ_INT(0, fp8_is_zero(0x38), "1.0");

    TEST("is_nan(0x78)");
    ASSERT_EQ_INT(1, fp8_is_nan(0x78), "NaN E=15 M=0");

    TEST("is_nan(0x7F)");
    ASSERT_EQ_INT(1, fp8_is_nan(0x7F), "NaN E=15 M=7");

    TEST("is_nan(0x77)");
    ASSERT_EQ_INT(0, fp8_is_nan(0x77), "max normal (not NaN)");
}

/* ---------------------------------------------------------------------------
 *  test_mul  — multiplication corner cases
 * ---------------------------------------------------------------------------
 */
static void test_mul(void)
{
    printf("\n--- Multiplication ---\n");

    /* 0 × x = 0 */
    TEST("mul(0, 1.0)");
    ASSERT_EQ_UINT8(0x00, fp8_mul(0x00, 0x38), "0 × 1.0");

    TEST("mul(1.0, 0)");
    ASSERT_EQ_UINT8(0x00, fp8_mul(0x38, 0x00), "1.0 × 0");

    TEST("mul(-0, 1.0)");
    ASSERT_EQ_UINT8(0x80, fp8_mul(0x80, 0x38), "-0 × 1.0");

    /* 1.0 × 1.0 = 1.0 */
    TEST("mul(1.0, 1.0)");
    ASSERT_EQ_UINT8(0x38, fp8_mul(0x38, 0x38), "1.0 × 1.0");

    /* 1.0 × -1.0 = -1.0 */
    TEST("mul(1.0, -1.0)");
    ASSERT_EQ_UINT8(0xB8, fp8_mul(0x38, 0xB8), "1.0 × -1.0");

    /* -1.0 × -1.0 = 1.0 */
    TEST("mul(-1.0, -1.0)");
    ASSERT_EQ_UINT8(0x38, fp8_mul(0xB8, 0xB8), "-1.0 × -1.0");

    /* 2.0 × 2.0 = 4.0:  2.0=0x40, 4.0=0x48 */
    TEST("mul(2.0, 2.0)");
    ASSERT_EQ_UINT8(0x48, fp8_mul(0x40, 0x40), "2.0 × 2.0");

    /* Max × max → overflow to max */
    TEST("mul(max, max) overflow");
    ASSERT_EQ_UINT8(0x77, fp8_mul(0x77, 0x77), "max × max");

    /* Subnormal × subnormal → underflow */
    /* 2^-9 × 2^-9 = 2^-18, well below min subnormal */
    TEST("mul(sub, sub) underflow");
    ASSERT_EQ_UINT8(0x00, fp8_mul(0x01, 0x01), "sub × sub underflow");

    /* Subnormal × normal */
    /* 2^-9 × 1.0 = 2^-9 (min subnormal) */
    TEST("mul(2^-9, 1.0)");
    ASSERT_EQ_UINT8(0x01, fp8_mul(0x01, 0x38), "2^-9 × 1.0");

    /* 0.5 × 0.5 = 0.25 */
    /* 0.5 = E=6 M=0 (0x30) */
    TEST("mul(0.5, 0.5)");
    ASSERT_EQ_UINT8(0x28, fp8_mul(0x30, 0x30), "0.5 × 0.5");

    /* NaN × anything */
    TEST("mul(NaN, 1.0)");
    ASSERT_EQ_UINT8(0x00, fp8_mul(0x78, 0x38), "NaN × 1.0");
}

/* ---------------------------------------------------------------------------
 *  test_mul_to_float  — exact product preservation
 * ---------------------------------------------------------------------------
 */
static void test_mul_to_float(void)
{
    printf("\n--- Multiply → Float (exact) ---\n");

    TEST("fp8_mul_to_float(1.0, 1.0)");
    ASSERT_FLOAT_EQ(1.0f, fp8_mul_to_float(0x38, 0x38), 1e-12f, "1.0*1.0");

    TEST("fp8_mul_to_float(2.0, 3.0)");
    /* 2.0 = 0x40; approximate 3.0 in fp8: closest is E=9 M=4 → 2^2 × 1.5 = 6? No.
     * Actually, 3.0: log2(3)=1.585, floor=1, E=8. 2^1 × (1+M/8) = 2 × frac = 3, frac=1.5, M=4.
     * E=8: 2^(8-7) × 1.5 = 2 × 1.5 = 3.0. So 0x44 = 3.0.
     * 2.0 × 3.0 = 6.0 */
    ASSERT_FLOAT_EQ(6.0f, fp8_mul_to_float(0x40, 0x44), 1e-12f, "2.0*3.0");

    TEST("fp8_mul_to_float(sub, normal)");
    ASSERT_FLOAT_EQ(1.0f/512.0f, fp8_mul_to_float(0x01, 0x38), 1e-15f, "2^-9 * 1.0");
}

/* ---------------------------------------------------------------------------
 *  test_add  — addition
 * ---------------------------------------------------------------------------
 */
static void test_add(void)
{
    printf("\n--- Addition ---\n");

    /* 0 + x = x */
    TEST("add(0, 1.0)");
    ASSERT_EQ_UINT8(0x38, fp8_add(0x00, 0x38), "0 + 1.0");

    /* 1.0 + 1.0 = 2.0 */
    TEST("add(1.0, 1.0)");
    ASSERT_EQ_UINT8(0x40, fp8_add(0x38, 0x38), "1.0 + 1.0");

    /* 1.0 + (-1.0) = 0 */
    TEST("add(1.0, -1.0)");
    ASSERT_EQ_UINT8(0x00, fp8_add(0x38, 0xB8), "1.0 + (-1.0)");

    /* 240 + 240 → overflow → max */
    TEST("add(max, max) overflow");
    ASSERT_EQ_UINT8(0x77, fp8_add(0x77, 0x77), "max + max");

    /* Subnormal + subnormal */
    /* 2^-9 + 2^-9 = 2^-8 */
    TEST("add(2^-9, 2^-9)");
    ASSERT_EQ_UINT8(0x02, fp8_add(0x01, 0x01), "sub + sub");

    /* -0 + -0 = -0 */
    TEST("add(-0, -0)");
    ASSERT_EQ_UINT8(0x80, fp8_add(0x80, 0x80), "-0 + -0");
}

/* ---------------------------------------------------------------------------
 *  test_fp16  — fp8 → fp16 conversion
 * ---------------------------------------------------------------------------
 */
static void test_fp16(void)
{
    printf("\n--- FP8 → FP16 Conversion ---\n");

    /* Zero */
    TEST("fp16(0x00)");
    ASSERT_EQ_UINT16(0x0000, fp8_to_fp16(0x00), "0 → 0x0000");

    TEST("fp16(0x80)");
    ASSERT_EQ_UINT16(0x8000, fp8_to_fp16(0x80), "-0 → 0x8000");

    /* 1.0 → fp16: 0 01111 0000000000 = 0x3C00 */
    TEST("fp16(1.0)");
    ASSERT_EQ_UINT16(0x3C00, fp8_to_fp16(0x38), "1.0 → 0x3C00");

    /* 2.0 → fp16: 0 10000 0000000000 = 0x4000 */
    TEST("fp16(2.0)");
    ASSERT_EQ_UINT16(0x4000, fp8_to_fp16(0x40), "2.0 → 0x4000");

    /* -1.0 → fp16: 1 01111 0000000000 = 0xBC00 */
    TEST("fp16(-1.0)");
    ASSERT_EQ_UINT16(0xBC00, fp8_to_fp16(0xB8), "-1.0 → 0xBC00");

    /* Min subnormal: 2^-9 = 2^(6-15) × 1.0  → E=6, M=0 → 0x1800 */
    TEST("fp16(2^-9)");
    ASSERT_EQ_UINT16(0x1800, fp8_to_fp16(0x01), "2^-9 → 0x1800");

    /* Max subnormal: 7 × 2^-9 = 2^(8-15) × (1+768/1024) → E=8, M=768=0x300 → 0x2300 */
    TEST("fp16(max sub)");
    ASSERT_EQ_UINT16(0x2300, fp8_to_fp16(0x07), "max subnormal → 0x2300");

    /* Max normal: 240 = 128 × 1.875 = 2^(22-15) × (1+896/1024) → E=22, M=896=0x380 → 0x5B80 */
    /* Actually: 240 = 2^7 × 1.875. fp16: E=7+15=22, M=(1.875-1)*1024=896=0x380 → 0x5B80 */
    TEST("fp16(max normal)");
    ASSERT_EQ_UINT16(0x5B80, fp8_to_fp16(0x77), "max → 0x5B80");

    /* NaN → fp16 NaN (we map to fp16 NaN for toolchain compatibility) */
    TEST("fp16(NaN)");
    ASSERT_EQ_UINT16(0x7C01, fp8_to_fp16(0x78), "NaN → 0x7C01");

    /* 0.5 → fp16: E=6, M=0 → 0x3800 */
    TEST("fp16(0.5)");
    ASSERT_EQ_UINT16(0x3800, fp8_to_fp16(0x30), "0.5 → 0x3800");
}

/* ---------------------------------------------------------------------------
 *  test_enumeration  — exhaustive encode→decode round-trip for all 256 codes
 * ---------------------------------------------------------------------------
 *
 *  For all legal FP8 bit patterns, decode then re-encode should produce
 *  the same bit pattern.  NaN codes are excluded from this round-trip check
 *  because they are deliberately mapped to zero.
 */
static void test_enumeration(void)
{
    printf("\n--- Exhaustive Round-trip (256 codes) ---\n");
    int roundtrip_ok  = 0;
    int roundtrip_nan = 0;
    int roundtrip_fail = 0;

    for (unsigned i = 0; i < 256; i++) {
        uint8_t bits = (uint8_t)i;
        int is_nan = fp8_is_nan(bits);

        float   decoded = fp8_decode(bits);
        uint8_t encoded = fp8_encode(decoded);

        if (is_nan) {
            /* NaN codes are deliberately mapped to zero by decode;
             * re-encoding zero gives 0x00, not the original NaN code.
             * This is expected behaviour. */
            if (encoded == 0x00 || encoded == 0x80) {
                roundtrip_nan++;
            } else {
                roundtrip_fail++;
                printf("  FAIL  NaN 0x%02X → %.9g → 0x%02X\n", bits,
                       (double)decoded, encoded);
            }
        } else {
            if (encoded == bits) {
                roundtrip_ok++;
            } else {
                roundtrip_fail++;
                printf("  FAIL  0x%02X → %.9g → 0x%02X\n", bits,
                       (double)decoded, encoded);
            }
        }
    }

    printf("  Round-trip OK:  %d / 256\n", roundtrip_ok);
    printf("  NaN → zero:     %d / 16\n", roundtrip_nan);
    printf("  Failures:       %d\n", roundtrip_fail);
    tests_run++;
    if (roundtrip_fail == 0) {
        tests_pass++;
        printf("  PASS\n");
    } else {
        tests_fail++;
        printf("  FAIL\n");
    }
}

/* ---------------------------------------------------------------------------
 *  print_hex_table  — print encode/decode reference table for Python comparison
 * ---------------------------------------------------------------------------
 */
static void print_hex_table(void)
{
    printf("\n=== FP8 E4M3 ENCODE TABLE (C reference) ===\n");
    printf("# Comparison format: value(float) → fp8_hex\n");

    /* Key values for cross-validation */
    static const float test_values[] = {
         0.0f,       -0.0f,
         1.0f/512.0f,            /* min subnormal      */
         2.0f/512.0f,
         3.0f/512.0f,
         4.0f/512.0f,
         7.0f/512.0f,            /* max subnormal     */
         8.0f/512.0f,            /* = 1/64 = min normal */
         16.0f/512.0f,
         1.0f,
         2.0f,
         3.0f,
         4.0f,
         8.0f,
        16.0f,
        32.0f,
        64.0f,
       128.0f,
       240.0f,                   /* max normal         */
        NAN,                      /* NaN → 0            */
        INFINITY,                 /* Inf → 0            */
    };
    int n = sizeof(test_values) / sizeof(test_values[0]);

    for (int i = 0; i < n; i++) {
        float v = test_values[i];
        uint8_t enc = fp8_encode(v);
        printf("  encode(% .10e) = 0x%02X\n", (double)v, (unsigned)enc);
    }

    printf("\n=== FP8 E4M3 DECODE TABLE (C reference) ===\n");
    printf("# Comparison format: fp8_hex → value(float)\n");

    for (unsigned b = 0; b < 256; b++) {
        uint8_t bits = (uint8_t)b;
        /* skip NaN codes for brevity (they all decode to 0) */
        if (fp8_is_nan(bits)) {
            printf("  decode(0x%02X) = %.9e  [NaN]\n", b,
                   (double)fp8_decode(bits));
        } else {
            printf("  decode(0x%02X) = % .10e\n", b,
                   (double)fp8_decode(bits));
        }
    }

    printf("\n=== FP8 MUL TABLE (C reference) ===\n");
    printf("# Comparison format: mul(hex_a, hex_b) = hex_result\n");

    static const uint8_t mul_test_vals[] = {
        0x00, 0x01, 0x07, 0x08, 0x30, 0x38, 0x40, 0x44,
        0x48, 0x50, 0x58, 0x77, 0x80, 0xB8,
    };
    int na = sizeof(mul_test_vals) / sizeof(mul_test_vals[0]);

    for (int a = 0; a < na; a++) {
        for (int b = 0; b < na; b++) {
            uint8_t ra = mul_test_vals[a];
            uint8_t rb = mul_test_vals[b];
            uint8_t res = fp8_mul(ra, rb);
            printf("  mul(0x%02X, 0x%02X) = 0x%02X\n",
                   (unsigned)ra, (unsigned)rb, (unsigned)res);
        }
    }

    printf("\n=== FP8 ADD TABLE (C reference) ===\n");
    printf("# Comparison format: add(hex_a, hex_b) = hex_result\n");

    static const uint8_t add_test_vals[] = {
        0x00, 0x01, 0x08, 0x30, 0x38, 0x40, 0x77, 0x80, 0xB8,
    };
    int add_n = sizeof(add_test_vals) / sizeof(add_test_vals[0]);

    for (int a = 0; a < add_n; a++) {
        for (int b = 0; b < add_n; b++) {
            uint8_t ra = add_test_vals[a];
            uint8_t rb = add_test_vals[b];
            uint8_t res = fp8_add(ra, rb);
            printf("  add(0x%02X, 0x%02X) = 0x%02X\n",
                   (unsigned)ra, (unsigned)rb, (unsigned)res);
        }
    }

    printf("\n=== FP8 → FP16 TABLE (C reference) ===\n");
    printf("# Comparison format: fp8_to_fp16(fp8_hex) = fp16_hex\n");

    for (unsigned b = 0; b < 256; b++) {
        uint8_t bits = (uint8_t)b;
        uint16_t f16 = fp8_to_fp16(bits);
        printf("  fp8_to_fp16(0x%02X) = 0x%04X\n", b, (unsigned)f16);
    }
}

/* ---------------------------------------------------------------------------
 *  main
 * ---------------------------------------------------------------------------
 */
int main(void)
{
    printf("=== FP8 E4M3 Reference Implementation Test Suite ===\n");
    printf("C reference for cross-validation against Python golden model\n");

    test_decode_encode();
    test_predicates();
    test_mul();
    test_mul_to_float();
    test_add();
    test_fp16();
    test_enumeration();

    /* Print full tables for Python cross-comparison */
    print_hex_table();

    printf("\n=== Summary ===\n");
    printf("Total:  %d\n", tests_run);
    printf("Passed: %d\n", tests_pass);
    printf("Failed: %d\n", tests_fail);

    return (tests_fail == 0) ? 0 : 1;
}
