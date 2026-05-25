#ifndef FPGA_LPU_FP4_REF_H
#define FPGA_LPU_FP4_REF_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FP4_GROUP_SIZE_DEFAULT 16
#define FP4_E2M1_MAX 3.0f

typedef struct {
    uint8_t code;
} fp4_t;

float fp4_e2m1_dequant_code(uint8_t code);
uint8_t fp4_e2m1_quant_value(float x);

void fp4_quantize_grouped(const float *src, uint8_t *codes, float *scales,
                          size_t rows, size_t cols, size_t group_size);

void fp4_dequantize_grouped(const uint8_t *codes, const float *scales,
                            float *dst, size_t rows, size_t cols,
                            size_t group_size);

void fp4_gemm_ref(const uint8_t *weight_codes, const float *weight_scales,
                  const float *activation, float *out,
                  size_t m, size_t k, size_t n, size_t group_size);

float fp4_cosine_similarity(const float *a, const float *b, size_t n);
float fp4_relative_l2_error(const float *a, const float *b, size_t n);

#ifdef __cplusplus
}
#endif

#endif
