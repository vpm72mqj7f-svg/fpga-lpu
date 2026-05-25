#include "fp4_ref.h"

#include <math.h>
#include <string.h>

static const float FP4_POS_VALUES[8] = {
    0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f
};

float fp4_e2m1_dequant_code(uint8_t code) {
    uint8_t mag = code & 0x7u;
    uint8_t sign = (code >> 3) & 0x1u;
    float v = FP4_POS_VALUES[mag > 7 ? 7 : mag];
    return sign ? -v : v;
}

uint8_t fp4_e2m1_quant_value(float x) {
    int sign = x < 0.0f;
    float ax = fabsf(x);
    int best = 0;
    float best_diff = fabsf(ax - FP4_POS_VALUES[0]);
    for (int i = 1; i < 8; ++i) {
        float diff = fabsf(ax - FP4_POS_VALUES[i]);
        if (diff < best_diff) {
            best = i;
            best_diff = diff;
        }
    }
    return (uint8_t)(best | (sign ? 0x8u : 0u));
}

void fp4_quantize_grouped(const float *src, uint8_t *codes, float *scales,
                          size_t rows, size_t cols, size_t group_size) {
    if (group_size == 0) group_size = FP4_GROUP_SIZE_DEFAULT;
    size_t groups = (cols + group_size - 1) / group_size;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t g = 0; g < groups; ++g) {
            size_t start = g * group_size;
            size_t end = start + group_size;
            if (end > cols) end = cols;

            float amax = 0.0f;
            for (size_t c = start; c < end; ++c) {
                float ax = fabsf(src[r * cols + c]);
                if (ax > amax) amax = ax;
            }
            if (amax < 1e-12f) amax = 1e-12f;
            float scale = amax / FP4_E2M1_MAX;
            scales[r * groups + g] = scale;

            for (size_t c = start; c < end; ++c) {
                float scaled = src[r * cols + c] / scale;
                if (scaled > FP4_E2M1_MAX) scaled = FP4_E2M1_MAX;
                if (scaled < -FP4_E2M1_MAX) scaled = -FP4_E2M1_MAX;
                codes[r * cols + c] = fp4_e2m1_quant_value(scaled);
            }
        }
    }
}

void fp4_dequantize_grouped(const uint8_t *codes, const float *scales,
                            float *dst, size_t rows, size_t cols,
                            size_t group_size) {
    if (group_size == 0) group_size = FP4_GROUP_SIZE_DEFAULT;
    size_t groups = (cols + group_size - 1) / group_size;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            size_t g = c / group_size;
            float scale = scales[r * groups + g];
            dst[r * cols + c] = fp4_e2m1_dequant_code(codes[r * cols + c]) * scale;
        }
    }
}

void fp4_gemm_ref(const uint8_t *weight_codes, const float *weight_scales,
                  const float *activation, float *out,
                  size_t m, size_t k, size_t n, size_t group_size) {
    if (group_size == 0) group_size = FP4_GROUP_SIZE_DEFAULT;
    size_t groups = (k + group_size - 1) / group_size;
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) {
                size_t g = kk / group_size;
                float w = fp4_e2m1_dequant_code(weight_codes[row * k + kk]) *
                          weight_scales[row * groups + g];
                float a = activation[kk * n + col];
                acc += w * a;
            }
            out[row * n + col] = acc;
        }
    }
}

float fp4_cosine_similarity(const float *a, const float *b, size_t n) {
    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * (double)b[i];
        aa += (double)a[i] * (double)a[i];
        bb += (double)b[i] * (double)b[i];
    }
    double denom = sqrt(aa * bb);
    if (denom < 1e-30) return 0.0f;
    return (float)(dot / denom);
}

float fp4_relative_l2_error(const float *a, const float *b, size_t n) {
    double diff = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        diff += d * d;
        ref += (double)b[i] * (double)b[i];
    }
    if (ref < 1e-30) return 0.0f;
    return (float)sqrt(diff / ref);
}
