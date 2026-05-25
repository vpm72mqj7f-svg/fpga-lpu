#include "fp4_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return 1; } \
} while (0)

static float frand(int *seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return ((float)(*seed % 20000) / 10000.0f) - 1.0f;
}

static int test_codes(void) {
    CHECK(fp4_e2m1_dequant_code(0x0) == 0.0f, "zero code");
    CHECK(fabsf(fp4_e2m1_dequant_code(0x4) - 1.0f) < 1e-6f, "one code");
    CHECK(fabsf(fp4_e2m1_dequant_code(0x7) - 3.0f) < 1e-6f, "max code");
    CHECK(fabsf(fp4_e2m1_dequant_code(0xF) + 3.0f) < 1e-6f, "negative max code");
    CHECK(fp4_e2m1_quant_value(0.74f) == 3, "nearest 0.75");
    CHECK(fp4_e2m1_quant_value(-1.6f) == (0x8 | 5), "nearest -1.5");
    return 0;
}

static int test_quant_dequant(void) {
    float src[32];
    uint8_t codes[32];
    float scales[2];
    float dst[32];
    for (int i = 0; i < 32; ++i) src[i] = (float)(i - 16) * 0.1f;
    fp4_quantize_grouped(src, codes, scales, 2, 16, 16);
    fp4_dequantize_grouped(codes, scales, dst, 2, 16, 16);
    float cos = fp4_cosine_similarity(src, dst, 32);
    CHECK(cos > 0.99f, "quant/dequant cosine");
    CHECK(scales[0] > 0.0f && scales[1] > 0.0f, "scales positive");
    return 0;
}

static int test_gemm(void) {
    enum { M = 8, K = 16, N = 4 };
    float w[M * K];
    float a[K * N];
    float ref[M * N];
    float out[M * N];
    uint8_t codes[M * K];
    float scales[M * ((K + 15) / 16)];
    int seed = 42;
    for (int i = 0; i < M * K; ++i) w[i] = frand(&seed) * 0.2f;
    for (int i = 0; i < K * N; ++i) a[i] = frand(&seed);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += w[m * K + k] * a[k * N + n];
            ref[m * N + n] = acc;
        }
    }

    fp4_quantize_grouped(w, codes, scales, M, K, 16);
    fp4_gemm_ref(codes, scales, a, out, M, K, N, 16);
    float cos = fp4_cosine_similarity(ref, out, M * N);
    float rel = fp4_relative_l2_error(out, ref, M * N);
    printf("gemm cosine=%.6f rel=%.6f\n", cos, rel);
    CHECK(cos > 0.98f, "gemm cosine");
    CHECK(rel < 0.25f, "gemm relative error");
    return 0;
}

int main(void) {
    CHECK(test_codes() == 0, "test_codes");
    CHECK(test_quant_dequant() == 0, "test_quant_dequant");
    CHECK(test_gemm() == 0, "test_gemm");
    printf("PASS fp4_ref tests\n");
    return 0;
}
