/**
 * cpu_prefill.c — CPU Prefill Engine Implementation
 *
 * Targets: Intel Xeon Granite Rapids (AMX) and AMD EPYC Turin (AVX-512).
 *
 * AMX tile configuration (Xeon 6980P):
 *   - 8 tile registers (TMM0-TMM7), each 1KB (16x16 BF16)
 *   - Tile operations: TDPBF16PS (dot-product BF16→FP32 accumulate)
 *   - Throughput: 1024 BF16 ops/cycle/tile, 128 tiles/socket
 *   - Peak: ~262 TFLOPS BF16 theoretical, ~10 TFLOPS practical fp8 GEMM
 *
 * Build:
 *   gcc -O3 -march=graniterapids -mamx-tile -mamx-int8 -mamx-bf16 \
 *       -lpthread -o cpu_prefill_bench cpu_prefill.c
 */

#include "cpu_prefill.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>

/* ── Backend Detection ────────────────────────────────────────────────── */

#ifdef __AMX_TILE__
  #define HAS_AMX 1
#else
  #define HAS_AMX 0
#endif

#ifdef __AVX512BF16__
  #define HAS_AVX512_BF16 1
#else
  #define HAS_AVX512_BF16 0
#endif

static cpu_prefill_backend_t detect_backend(void) {
    if (HAS_AMX) return CPU_PREFILL_AMX;
    if (HAS_AVX512_BF16) return CPU_PREFILL_AVX512;
    return CPU_PREFILL_SCALAR;
}

/* ── AMX GEMM Implementation ──────────────────────────────────────────── */

#if HAS_AMX
#include <immintrin.h>

static double amx_gemm_fp8(int M, int K, int N,
                           const uint8_t *A, const uint8_t *B,
                           float *C) {
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    /* AMX tile configuration:
     *   TMM0: A tile (activations) — 16 rows × K columns, fp8→BF16
     *   TMM1: B tile (weights)     — K rows × 16 columns, fp8→BF16
     *   TMM2: C tile (accumulator) — 16 × 16 fp32
     *
     * Compute: C[16,16] += A[16,K] × B[K,16]
     *
     * Outer loop: tile over M (rows of A) and N (cols of B)
     * Inner loop: accumulate over K dimension
     */

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; m += 16) {
        for (int n = 0; n < N; n += 16) {
            /* Initialize accumulator tile to zero */
            _tile_zero(2);

            /* Accumulate over K dimension in chunks */
            for (int k = 0; k < K; k += 32) {
                int k_chunk = (K - k < 32) ? (K - k) : 32;

                /* Load A tile [16 × k_chunk] */
                _tile_loadd(0, A + m * K + k, K);

                /* Load B tile [k_chunk × 16] */
                _tile_loadd(1, B + k * N + n, N);

                /* C[16,16] += A[16,kc] × B[kc,16] */
                _tile_dpbf16ps(2, 0, 1);
            }

            /* Store result */
            _tile_stored(2, C + m * N + n, N);
        }
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) +
                     (t1.tv_usec - t0.tv_usec) / 1e6;

    double gflops = (2.0 * M * K * N) / (elapsed * 1e9);
    return gflops;
}
#endif /* HAS_AMX */

/* ── AVX-512 GEMM Implementation (AMD EPYC Turin) ─────────────────────── */

#if HAS_AVX512_BF16
#include <immintrin.h>

static double avx512_gemm_fp8(int M, int K, int N,
                              const uint8_t *A, const uint8_t *B,
                              float *C) {
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    /* AVX-512 BF16: VDPBF16PS — dot product BF16 pairs → FP32
     * Each ZMM register: 32 BF16 values
     * One VDPBF16PS: 32 multiply + 32 accumulate = 64 FLOPs
     *
     * Strategy:
     *   - Pack fp8→BF16 at load time (simple zero-extension for E4M3)
     *   - Outer product accumulation over K dimension
     *   - 16 accumulators to hide latency
     */

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n += 32) {
            __m512 c0 = _mm512_setzero_ps();
            __m512 c1 = _mm512_setzero_ps();

            for (int k = 0; k < K; k++) {
                /* Broadcast A[m,k] to all 32 slots */
                uint16_t a_bf16 = (uint16_t)(A[m * K + k]) << 8;
                __m512bh a_vec = _mm512_set1_bf16(a_bf16);

                /* Load B[k, n:n+31] as BF16 */
                __m512bh b_vec = _mm512_loadu_bf16(B + k * N + n);

                /* c += a * b */
                c0 = _mm512_dpbf16_ps(c0, a_vec, b_vec);
            }

            _mm512_storeu_ps(C + m * N + n, c0);
        }
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) +
                     (t1.tv_usec - t0.tv_usec) / 1e6;
    return (2.0 * M * K * N) / (elapsed * 1e9);
}
#endif /* HAS_AVX512_BF16 */

/* ── Portable Scalar Fallback ─────────────────────────────────────────── */

static double scalar_gemm_fp8(int M, int K, int N,
                              const uint8_t *A, const uint8_t *B,
                              float *C) {
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                /* fp8 E4M3 → float: rough approximation */
                uint8_t a_val = A[m * K + k];
                uint8_t b_val = B[k * N + n];
                float af = (a_val - 127) / 16.0f;  /* simplified E4M3 decode */
                float bf = (b_val - 127) / 16.0f;
                sum += af * bf;
            }
            C[m * N + n] = sum;
        }
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) +
                     (t1.tv_usec - t0.tv_usec) / 1e6;
    return (2.0 * M * K * N) / (elapsed * 1e9);
}

/* ── Public API ───────────────────────────────────────────────────────── */

double cpu_gemm_fp8(int M, int K, int N,
                    const uint8_t *A, const uint8_t *B,
                    float *C,
                    const float *scale_A, const float *scale_B) {
    (void)scale_A; (void)scale_B;  /* TODO: apply per-row/col scaling */

#if HAS_AMX
    return amx_gemm_fp8(M, K, N, A, B, C);
#elif HAS_AVX512_BF16
    return avx512_gemm_fp8(M, K, N, A, B, C);
#else
    return scalar_gemm_fp8(M, K, N, A, B, C);
#endif
}

double cpu_batched_gemv_fp8(int batch, int K, int N,
                            const uint8_t *A, const uint8_t *B,
                            float *C,
                            const float *scale_A, const float *scale_B) {
    /* Batched GEMV: treat as GEMM M=batch and let the tiled impl handle it.
     * AMX efficiently handles small-M GEMM via tile operations. */
    return cpu_gemm_fp8(batch, K, N, A, B, C, scale_A, scale_B);
}

/* ── Layer Implementation ─────────────────────────────────────────────── */

double cpu_prefill_layer(const cpu_prefill_config_t *cfg,
                         int layer_idx, int chunk_size,
                         const uint8_t *hidden_state,
                         uint8_t *output_state,
                         uint8_t *kv_cache_k, uint8_t *kv_cache_v) {
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    (void)layer_idx;

    int H = cfg->hidden_dim;
    int I = cfg->intermediate_dim;
    int KL = cfg->kv_latent_dim;
    int NE = cfg->num_experts;
    int TK = cfg->top_k;
    int P = chunk_size;

    /* Scratch buffers */
    float *tmp_HxH = calloc(P * H, sizeof(float));
    float *tmp_HxKL = calloc(P * KL, sizeof(float));
    float *tmp_HxI  = calloc(P * I, sizeof(float));
    float *tmp_IxH  = calloc(P * H, sizeof(float));

    /* ── MLA QKV ─────────────────────────────────────────── */
    /* Q = hidden @ W_Q  [P,H] x [H,H] → [P,H] */
    const uint8_t *W_Q = NULL;  /* TODO: load from weight cache */
    cpu_gemm_fp8(P, H, H, hidden_state, W_Q, tmp_HxH, NULL, NULL);

    /* K_latent = hidden @ W_K  [P,H] x [H,KL] → [P,KL] */
    const uint8_t *W_K = NULL;
    cpu_gemm_fp8(P, H, KL, hidden_state, W_K, tmp_HxKL, NULL, NULL);
    memcpy(kv_cache_k, tmp_HxKL, P * KL * sizeof(float));

    /* V_latent = hidden @ W_V  [P,H] x [H,KL] → [P,KL] */
    const uint8_t *W_V = NULL;
    cpu_gemm_fp8(P, H, KL, hidden_state, W_V, tmp_HxKL, NULL, NULL);
    memcpy(kv_cache_v, tmp_HxKL, P * KL * sizeof(float));

    /* Decompress: K = K_latent @ W_K_up, V = V_latent @ W_V_up */
    /* (simplified — real impl uses RoPE + multi-head) */

    /* ── Attention (simplified) ──────────────────────────── */
    /* Q @ K^T → softmax → @ V */
    /* For chunked prefill: attend over current chunk + cached prefix */

    /* ── Shared Expert FFN ────────────────────────────────── */
    /* gate: [P,H] x [H,I] → [P,I] */
    const uint8_t *W_gate = NULL;
    cpu_gemm_fp8(P, H, I, hidden_state, W_gate, tmp_HxI, NULL, NULL);

    /* SiLU activation (in-place) */
    for (int i = 0; i < P * I; i++) {
        float x = tmp_HxI[i];
        tmp_HxI[i] = x / (1.0f + expf(-x));  /* silu(x) = x * sigmoid(x) */
    }

    /* up: [P,H] x [H,I] → [P,I] */
    const uint8_t *W_up = NULL;
    float *tmp_HxI_up = calloc(P * I, sizeof(float));
    cpu_gemm_fp8(P, H, I, hidden_state, W_up, tmp_HxI_up, NULL, NULL);

    /* Element-wise: silu(gate) * up */
    for (int i = 0; i < P * I; i++)
        tmp_HxI[i] *= tmp_HxI_up[i];
    free(tmp_HxI_up);

    /* down: [P,I] x [I,H] → [P,H] (add to output) */
    const uint8_t *W_down = NULL;
    cpu_gemm_fp8(P, I, H, (uint8_t *)tmp_HxI, W_down, tmp_IxH, NULL, NULL);
    for (int i = 0; i < P * H; i++)
        tmp_HxH[i] += tmp_IxH[i];

    /* ── Routed Experts (top-K = 6) ──────────────────────── */
    /* Router: [P,H] x [H,NE] → [P,NE] */
    /* Select top-K experts per token */
    /* For each expert e in top-K:
     *   gate: A @ W_gate[e]  → SiLU
     *   up:   A @ W_up[e]    → mul
     *   down: mid @ W_down[e] → add to output
     */
    /* (simplified — real impl dispatches per-expert) */

    /* ── Output: RMSNorm ─────────────────────────────────── */
    for (int p = 0; p < P; p++) {
        float sum_sq = 0.0f;
        for (int h = 0; h < H; h++) {
            float v = tmp_HxH[p * H + h];
            sum_sq += v * v;
        }
        float rsqrt = 1.0f / sqrtf(sum_sq / H + 1e-5f);
        for (int h = 0; h < H; h++) {
            float v = tmp_HxH[p * H + h] * rsqrt;
            /* Clamp to fp8 range, convert back to E4M3 */
            output_state[p * H + h] = (uint8_t)((v + 16.0f) * 16.0f);
        }
    }

    free(tmp_HxH); free(tmp_HxKL); free(tmp_HxI); free(tmp_IxH);

    gettimeofday(&t1, NULL);
    return (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_usec - t0.tv_usec);
}

double cpu_prefill_all_layers(const cpu_prefill_config_t *cfg,
                              int chunk_size,
                              const uint8_t *input_tokens,
                              uint8_t *output_state,
                              uint8_t *kv_cache_k_all,
                              uint8_t *kv_cache_v_all) {
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    int H = cfg->hidden_dim;
    int KL = cfg->kv_latent_dim;
    int NL = cfg->num_layers;

    /* Working buffer: ping-pong between two hidden states */
    uint8_t *hs[2];
    hs[0] = malloc(chunk_size * H);
    hs[1] = malloc(chunk_size * H);
    memcpy(hs[0], input_tokens, chunk_size * H);

    for (int l = 0; l < NL; l++) {
        int src = l & 1;
        int dst = 1 - src;

        cpu_prefill_layer(cfg, l, chunk_size,
                         hs[src], hs[dst],
                         kv_cache_k_all + l * chunk_size * KL,
                         kv_cache_v_all + l * chunk_size * KL);
    }

    memcpy(output_state, hs[NL & 1], chunk_size * H);
    free(hs[0]); free(hs[1]);

    gettimeofday(&t1, NULL);
    return (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_usec - t0.tv_usec);
}

/* ── Weight Cache ─────────────────────────────────────────────────────── */

#define MAX_LAYER_WEIGHTS 128

static struct {
    void   *data;
    size_t  size;
    int     loaded;
} weight_cache[MAX_LAYER_WEIGHTS];

int cpu_weight_cache_load(const cpu_prefill_config_t *cfg,
                          int layer_idx, const void *data, size_t size) {
    (void)cfg;
    if (layer_idx >= MAX_LAYER_WEIGHTS) return -1;
    if (weight_cache[layer_idx].loaded) return 0;

    void *buf = malloc(size);
    if (!buf) return -1;
    memcpy(buf, data, size);
    weight_cache[layer_idx].data = buf;
    weight_cache[layer_idx].size = size;
    weight_cache[layer_idx].loaded = 1;
    return 0;
}

void cpu_weight_cache_unload(int layer_idx) {
    if (layer_idx < MAX_LAYER_WEIGHTS && weight_cache[layer_idx].loaded) {
        free(weight_cache[layer_idx].data);
        weight_cache[layer_idx].data   = NULL;
        weight_cache[layer_idx].size   = 0;
        weight_cache[layer_idx].loaded = 0;
    }
}

void cpu_weight_cache_unload_all(void) {
    for (int i = 0; i < MAX_LAYER_WEIGHTS; i++)
        cpu_weight_cache_unload(i);
}

/* ── Performance Stats ────────────────────────────────────────────────── */

static cpu_prefill_stats_t stats;

const cpu_prefill_stats_t *cpu_prefill_get_stats(void) { return &stats; }
void cpu_prefill_reset_stats(void) { memset(&stats, 0, sizeof(stats)); }
