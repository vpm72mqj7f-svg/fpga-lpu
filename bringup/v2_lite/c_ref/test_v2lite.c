/*
 * test_v2lite.c -- V2-Lite FFN dimension verification + FP8 test vectors
 *
 * V2-Lite is a reduced-scale FPGA FFN engine using the same FP8 E4M3
 * arithmetic as V4-Flash.  This test verifies:
 *   1. Key FP8 encode/decode constants (zero, one, max, negative max)
 *   2. FP8 multiply corner cases
 *   3. Produces a hex table for cross-validation with Python golden model
 *
 * V2-Lite dimensions (for reference — not exercised by this arithmetic test):
 *   HIDDEN    = 2048
 *   INTERMED  = 5120
 *   EXPERTS   = 3
 *   CHANNELS  = 8
 *   TILE_M    = 64
 *   TILE_K    = 64
 *   TILE_N    = 16
 */

#include "fp8_e4m3.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ========================================================================
 *  V2-Lite architecture constants (documentation + sanity checks)
 * ======================================================================== */

#define V2L_HIDDEN    2048
#define V2L_INTERMED  5120
#define V2L_EXPERTS   3
#define V2L_CHANNELS  8
#define V2L_TILE_M    64
#define V2L_TILE_K    64
#define V2L_TILE_N    16

/* Sanity: verify tile dimensions divide HIDDEN / INTERMED evenly */
_Static_assert(V2L_HIDDEN   % V2L_TILE_K == 0, "TILE_K must divide HIDDEN");
_Static_assert(V2L_INTERMED % V2L_TILE_N == 0, "TILE_N must divide INTERMED");

/* ========================================================================
 *  test helpers
 * ======================================================================== */

static int tests_run  = 0;
static int tests_pass = 0;
static int tests_fail = 0;

#define TEST(name)  do { tests_run++; printf("  %-55s ", name); } while (0)
#define OK()        do { tests_pass++; printf("PASS\n"); } while (0)
#define FAIL(fmt, ...) \
    do { \
        tests_fail++; \
        printf("FAIL: " fmt "\n", ##__VA_ARGS__); \
    } while (0)

#define ASSERT_EQ_UINT8(expected, actual, name_str) \
    do { \
        if ((expected) != (actual)) { \
            FAIL("%s: expected 0x%02X, got 0x%02X", name_str, \
                 (unsigned)(expected), (unsigned)(actual)); \
        } else { OK(); } \
    } while (0)

#define ASSERT_EQ_INT(expected, actual, name_str) \
    do { \
        if ((expected) != (actual)) { \
            FAIL("%s: expected %d, got %d", name_str, \
                 (int)(expected), (int)(actual)); \
        } else { OK(); } \
    } while (0)

/* ========================================================================
 *  test_v2lite_encodings  —  V2-Lite specific FP8 encode checks
 * ======================================================================== */
static void test_v2lite_encodings(void)
{
    printf("\n--- V2-Lite FP8 Encode Verification ---\n");

    /* Requirement: fp8_encode(0.0) == 0x00 */
    TEST("fp8_encode(0.0) == 0x00");
    ASSERT_EQ_UINT8(0x00, fp8_encode(0.0f), "encode(0.0)");

    /* Requirement: fp8_encode(1.0) == 0x38 */
    TEST("fp8_encode(1.0) == 0x38");
    ASSERT_EQ_UINT8(0x38, fp8_encode(1.0f), "encode(1.0)");

    /* Requirement: fp8_encode(240.0) == 0x77 (max normal) */
    TEST("fp8_encode(240.0) == 0x77 (max)");
    ASSERT_EQ_UINT8(0x77, fp8_encode(240.0f), "encode(240.0)");

    /* Requirement: fp8_encode(-240.0) == 0xF7 */
    TEST("fp8_encode(-240.0) == 0xF7");
    ASSERT_EQ_UINT8(0xF7, fp8_encode(-240.0f), "encode(-240.0)");

    /* Additional V2-Lite relevant checks */
    TEST("fp8_encode(-0.0) == 0x80");
    ASSERT_EQ_UINT8(0x80, fp8_encode(-0.0f), "encode(-0.0)");

    TEST("fp8_encode(min_subnormal) == 0x01");
    ASSERT_EQ_UINT8(0x01, fp8_encode(1.0f / 512.0f), "encode(2^-9)");

    TEST("fp8_encode(min_normal) == 0x08");
    ASSERT_EQ_UINT8(0x08, fp8_encode(1.0f / 64.0f), "encode(2^-6)");

    TEST("fp8_encode(overflow) saturates to 0x77");
    ASSERT_EQ_UINT8(0x77, fp8_encode(500.0f), "encode(500)");

    TEST("fp8_encode(NaN) == 0x00");
    ASSERT_EQ_UINT8(0x00, fp8_encode(NAN), "encode(NaN)");

    TEST("fp8_encode(Inf) == 0x00");
    ASSERT_EQ_UINT8(0x00, fp8_encode(INFINITY), "encode(Inf)");
}

/* ========================================================================
 *  test_v2lite_mul  —  V2-Lite specific FP8 multiply checks
 * ======================================================================== */
static void test_v2lite_mul(void)
{
    printf("\n--- V2-Lite FP8 Multiply Verification ---\n");

    /* Requirement: fp8_mul(0x38, 0x38) == 0x38  (1.0 × 1.0 = 1.0) */
    TEST("fp8_mul(0x38, 0x38) == 0x38  (1.0 × 1.0 = 1.0)");
    ASSERT_EQ_UINT8(0x38, fp8_mul(0x38, 0x38), "1.0 × 1.0");

    /* Requirement: fp8_mul(0x00, 0x38) == 0x00  (0 × 1 = 0) */
    TEST("fp8_mul(0x00, 0x38) == 0x00  (0 × 1 = 0)");
    ASSERT_EQ_UINT8(0x00, fp8_mul(0x00, 0x38), "0 × 1.0");

    /* Additional V2-Lite relevant checks */
    TEST("fp8_mul(0x38, 0x00) == 0x00  (1 × 0 = 0)");
    ASSERT_EQ_UINT8(0x00, fp8_mul(0x38, 0x00), "1.0 × 0");

    TEST("fp8_mul(0x80, 0x38) == 0x80  (-0 × 1 = -0)");
    ASSERT_EQ_UINT8(0x80, fp8_mul(0x80, 0x38), "-0 × 1.0");

    TEST("fp8_mul(0x38, 0xB8) == 0xB8  (1.0 × -1.0 = -1.0)");
    ASSERT_EQ_UINT8(0xB8, fp8_mul(0x38, 0xB8), "1.0 × -1.0");

    TEST("fp8_mul(0xB8, 0xB8) == 0x38  (-1.0 × -1.0 = 1.0)");
    ASSERT_EQ_UINT8(0x38, fp8_mul(0xB8, 0xB8), "-1.0 × -1.0");

    TEST("fp8_mul(0x77, 0x77) == 0x77  (max × max overflow → max)");
    ASSERT_EQ_UINT8(0x77, fp8_mul(0x77, 0x77), "max × max");

    TEST("fp8_mul(0x01, 0x01) == 0x00  (sub × sub → underflow)");
    ASSERT_EQ_UINT8(0x00, fp8_mul(0x01, 0x01), "sub × sub");
}

/* ========================================================================
 *  test_v2lite_dimensions  —  verify V2-Lite architecture dimensions
 * ======================================================================== */
static void test_v2lite_dimensions(void)
{
    printf("\n--- V2-Lite Architecture Dimension Verification ---\n");

    /* HIDDEN = 2048 */
    TEST("V2L_HIDDEN == 2048");
    ASSERT_EQ_INT(2048, V2L_HIDDEN, "HIDDEN");

    /* INTERMED = 5120 */
    TEST("V2L_INTERMED == 5120");
    ASSERT_EQ_INT(5120, V2L_INTERMED, "INTERMED");

    /* EXPERTS = 3 */
    TEST("V2L_EXPERTS == 3");
    ASSERT_EQ_INT(3, V2L_EXPERTS, "EXPERTS");

    /* CHANNELS = 8 */
    TEST("V2L_CHANNELS == 8");
    ASSERT_EQ_INT(8, V2L_CHANNELS, "CHANNELS");

    /* TILE_M = 64 */
    TEST("V2L_TILE_M == 64");
    ASSERT_EQ_INT(64, V2L_TILE_M, "TILE_M");

    /* TILE_K = 64 */
    TEST("V2L_TILE_K == 64");
    ASSERT_EQ_INT(64, V2L_TILE_K, "TILE_K");

    /* TILE_N = 16 */
    TEST("V2L_TILE_N == 16");
    ASSERT_EQ_INT(16, V2L_TILE_N, "TILE_N");

    /* Divisibility checks (already compile-time, but verify-derived) */
    TEST("HIDDEN divisible by TILE_K");
    ASSERT_EQ_INT(0, V2L_HIDDEN % V2L_TILE_K, "2048 % 64");

    TEST("INTERMED divisible by TILE_N");
    ASSERT_EQ_INT(0, V2L_INTERMED % V2L_TILE_N, "5120 % 16");
}

/* ========================================================================
 *  print_v2lite_hex_table  —  hex dump for Python test vector comparison
 * ========================================================================
 *
 *  This table is designed to be diff'd against a Python-generated golden
 *  reference for V2-Lite FFN weight/product validation.
 */
static void print_v2lite_hex_table(void)
{
    printf("\n=================================================================\n");
    printf("=== V2-Lite FP8 Test Vector Table (C reference) ===\n");
    printf("# Format:  operation(args) = result_hex\n");
    printf("# Compatible with Python test vector comparison\n");
    printf("=================================================================\n");

    /* ------------------------------------------------------------------
     * Section 1: V2-Lite Dimension Summary
     * ------------------------------------------------------------------ */
    printf("\n--- V2L_DIMS ---\n");
    printf("  HIDDEN      = %d\n", V2L_HIDDEN);
    printf("  INTERMED    = %d\n", V2L_INTERMED);
    printf("  EXPERTS     = %d\n", V2L_EXPERTS);
    printf("  CHANNELS    = %d\n", V2L_CHANNELS);
    printf("  TILE_M      = %d\n", V2L_TILE_M);
    printf("  TILE_K      = %d\n", V2L_TILE_K);
    printf("  TILE_N      = %d\n", V2L_TILE_N);

    /* ------------------------------------------------------------------
     * Section 2: Encode table (key values for V2-Lite)
     * ------------------------------------------------------------------ */
    printf("\n--- V2L_ENCODE ---\n");
    printf("# float_value → fp8_hex\n");

    static const struct {
        float value;
        const char *label;
    } encode_tests[] = {
        {  0.0f,              "zero" },
        { -0.0f,              "neg_zero" },
        {  1.0f / 512.0f,     "min_subnormal (2^-9)" },
        {  7.0f / 512.0f,     "max_subnormal" },
        {  1.0f / 64.0f,      "min_normal (2^-6)" },
        {  1.0f,              "one" },
        { -1.0f,              "neg_one" },
        {  2.0f,              "two" },
        {  3.0f,              "three" },
        {  4.0f,              "four" },
        {  8.0f,              "eight" },
        { 16.0f,              "sixteen" },
        { 32.0f,              "thirtytwo" },
        { 64.0f,              "sixtyfour" },
        {128.0f,              "128" },
        {240.0f,              "max_normal" },
        {-240.0f,              "neg_max_normal" },
        {500.0f,              "overflow→max" },
    };
    int n_enc = sizeof(encode_tests) / sizeof(encode_tests[0]);
    for (int i = 0; i < n_enc; i++) {
        uint8_t enc = fp8_encode(encode_tests[i].value);
        printf("  ENCODE(%.10e)  # %s\n    = 0x%02X\n",
               (double)encode_tests[i].value, encode_tests[i].label, (unsigned)enc);
    }

    /* ------------------------------------------------------------------
     * Section 3: Decode table (all 256 codes)
     * ------------------------------------------------------------------ */
    printf("\n--- V2L_DECODE ---\n");
    printf("# fp8_hex → float_value\n");

    for (unsigned b = 0; b < 256; b++) {
        uint8_t bits = (uint8_t)b;
        float decoded = fp8_decode(bits);
        int is_nan = fp8_is_nan(bits);
        int is_zero = fp8_is_zero(bits);
        const char *tag = "";
        if (is_nan)  tag = " [NaN]";
        else if (is_zero && (bits & 0x80)) tag = " [-0]";
        else if (is_zero) tag = " [0]";
        printf("  DECODE(0x%02X) = % .10e%s\n", b, (double)decoded, tag);
    }

    /* ------------------------------------------------------------------
     * Section 4: Multiply reference table (V2-Lite relevant subset)
     * ------------------------------------------------------------------ */
    printf("\n--- V2L_MUL ---\n");
    printf("# mul(fp8_a, fp8_b) = fp8_result\n");

    /*
     * V2-Lite FFN weights are FP8; key dot-product operands to validate.
     * Test matrix: all combinations of these representative values.
     */
    static const struct {
        uint8_t code;
        const char *label;
    } mul_ops[] = {
        { 0x00, "0" },
        { 0x80, "-0" },
        { 0x01, "2^-9" },
        { 0x08, "2^-6" },
        { 0x30, "0.5" },
        { 0x38, "1.0" },
        { 0x40, "2.0" },
        { 0x44, "3.0" },
        { 0x48, "4.0" },
        { 0x50, "8.0" },
        { 0x58, "16.0" },
        { 0x60, "32.0" },
        { 0x68, "64.0" },
        { 0x70, "128.0" },
        { 0x77, "240.0" },
        { 0xB8, "-1.0" },
        { 0xF7, "-240.0" },
    };
    int n_mul = sizeof(mul_ops) / sizeof(mul_ops[0]);

    for (int a = 0; a < n_mul; a++) {
        for (int b = 0; b < n_mul; b++) {
            uint8_t res = fp8_mul(mul_ops[a].code, mul_ops[b].code);
            printf("  MUL(0x%02X, 0x%02X) = 0x%02X    # %s × %s\n",
                   (unsigned)mul_ops[a].code, (unsigned)mul_ops[b].code,
                   (unsigned)res,
                   mul_ops[a].label, mul_ops[b].label);
        }
    }

    /* ------------------------------------------------------------------
     * Section 5: Add reference table (V2-Lite accumulator validation)
     * ------------------------------------------------------------------ */
    printf("\n--- V2L_ADD ---\n");
    printf("# add(fp8_a, fp8_b) = fp8_result\n");

    static const struct {
        uint8_t code;
        const char *label;
    } add_ops[] = {
        { 0x00, "0" },
        { 0x80, "-0" },
        { 0x01, "2^-9" },
        { 0x08, "2^-6" },
        { 0x30, "0.5" },
        { 0x38, "1.0" },
        { 0x40, "2.0" },
        { 0x48, "4.0" },
        { 0x77, "240.0" },
        { 0xB8, "-1.0" },
    };
    int n_add = sizeof(add_ops) / sizeof(add_ops[0]);

    for (int a = 0; a < n_add; a++) {
        for (int b = 0; b < n_add; b++) {
            uint8_t res = fp8_add(add_ops[a].code, add_ops[b].code);
            printf("  ADD(0x%02X, 0x%02X) = 0x%02X    # %s + %s\n",
                   (unsigned)add_ops[a].code, (unsigned)add_ops[b].code,
                   (unsigned)res,
                   add_ops[a].label, add_ops[b].label);
        }
    }

    /* ------------------------------------------------------------------
     * Section 6: FP8 to FP16 conversion table
     * ------------------------------------------------------------------ */
    printf("\n--- V2L_FP8_TO_FP16 ---\n");
    printf("# fp8_to_fp16(fp8_hex) = fp16_hex\n");

    for (unsigned b = 0; b < 256; b++) {
        uint8_t bits = (uint8_t)b;
        uint16_t f16 = fp8_to_fp16(bits);
        printf("  FP8TOFP16(0x%02X) = 0x%04X\n", b, (unsigned)f16);
    }
}

/* ========================================================================
 *  main
 * ======================================================================== */
int main(void)
{
    printf("============================================================\n");
    printf("===  V2-Lite FFN — FP8 E4M3 C Reference Verification      ===\n");
    printf("============================================================\n");
    printf("\n");
    printf("Architecture: HIDDEN=%d  INTERMED=%d  EXPERTS=%d  CHANNELS=%d\n",
           V2L_HIDDEN, V2L_INTERMED, V2L_EXPERTS, V2L_CHANNELS);
    printf("Tile:         M=%d       K=%d         N=%d\n",
           V2L_TILE_M, V2L_TILE_K, V2L_TILE_N);
    printf("FP8 format:   E4M3  (bias=7, no Inf, NaN mapped to zero)\n");
    printf("\n");

    test_v2lite_encodings();
    test_v2lite_mul();
    test_v2lite_dimensions();

    /* Print hex table for Python cross-comparison */
    print_v2lite_hex_table();

    printf("\n============================================================\n");
    printf("===  Summary                                                ===\n");
    printf("============================================================\n");
    printf("  Total:  %d\n", tests_run);
    printf("  Passed: %d\n", tests_pass);
    printf("  Failed: %d\n", tests_fail);

    if (tests_fail == 0) {
        printf("\n  V2-Lite FP8 reference: ALL TESTS PASSED\n");
    } else {
        printf("\n  V2-Lite FP8 reference: %d TEST(S) FAILED\n", tests_fail);
    }

    return (tests_fail == 0) ? 0 : 1;
}
