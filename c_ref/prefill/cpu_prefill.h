/**
 * cpu_prefill.h — CPU Prefill Engine for DeepSeek V4 Pro
 *
 * Three-tier prefill architecture:
 *   Tier 1: CPU prefill (Intel AMX / AMD AVX-512) — short/medium prompts
 *   Tier 2: FPGA chunked prefill — long prompts, CPU fallback
 *   Tier 3: GPU prefill (optional) — ultra-low-latency TTFT
 *
 * This module implements Tier 1: optimized fp8 GEMM on host CPU.
 * Uses Intel AMX (Advanced Matrix Extensions) on Xeon Granite Rapids,
 * or AVX-512 BF16 on AMD EPYC Turin.
 */

#ifndef CPU_PREFILL_H
#define CPU_PREFILL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Configuration ───────────────────────────────────────────────────── */

typedef enum {
    CPU_PREFILL_AMX,        // Intel AMX (Granite Rapids+)
    CPU_PREFILL_AVX512,     // AMD AVX-512 BF16 (EPYC Turin)
    CPU_PREFILL_SCALAR      // Portable scalar fallback
} cpu_prefill_backend_t;

typedef struct {
    cpu_prefill_backend_t backend;
    int                   num_threads;       // 0 = auto (all cores)
    int                   max_chunk_size;    // tokens per prefill chunk (default 128)
    int                   hidden_dim;        // model hidden dimension
    int                   intermediate_dim;  // FFN intermediate dimension
    int                   kv_latent_dim;     // MLA K/V latent dimension
    int                   num_experts;       // total MoE experts
    int                   top_k;             // routed experts per token
    int                   num_layers;        // transformer layers
} cpu_prefill_config_t;

/* ── GEMM Operations ──────────────────────────────────────────────────── */

/**
 * fp8 GEMM: C[M,N] = A[M,K] x B[K,N]
 *
 * A: activations  (fp8 E4M3, M × K)
 * B: weights      (fp8 E4M3, K × N) — pre-quantized from fp4
 * C: output       (fp32, M × N)
 *
 * Uses AMX tile operations or AVX-512 VNNI depending on backend.
 * Returns: GFLOPS achieved (for performance monitoring).
 */
double cpu_gemm_fp8(
    int M, int K, int N,
    const uint8_t *A,           // [M * K] fp8
    const uint8_t *B,           // [K * N] fp8
    float         *C,           // [M * N] fp32
    const float   *scale_A,     // [M] per-row scale
    const float   *scale_B      // [N] per-col scale
);

/**
 * Batched fp8 GEMV (used for routed expert with top-K):
 *   C[b, n] = A[b, :] · B[:, n]   for b in 0..batch-1, n in 0..N-1
 *
 * Equivalent to GEMM with M=batch, but optimized for the case
 * where batch << N (common in MoE with top-K routing).
 */
double cpu_batched_gemv_fp8(
    int batch, int K, int N,
    const uint8_t *A,           // [batch * K] fp8
    const uint8_t *B,           // [K * N] fp8
    float         *C,           // [batch * N] fp32
    const float   *scale_A,
    const float   *scale_B
);

/* ── Full Layer Operations ────────────────────────────────────────────── */

/**
 * Prefill one transformer layer for `chunk_size` tokens.
 *
 * Weights must be pre-loaded into the weight cache (see below).
 * KV cache entries are produced as output.
 *
 * Returns: elapsed microseconds.
 */
double cpu_prefill_layer(
    const cpu_prefill_config_t *cfg,
    int                         layer_idx,
    int                         chunk_size,      // tokens in this chunk
    const uint8_t              *hidden_state,    // [chunk_size * hidden_dim] fp8
    uint8_t                    *output_state,    // [chunk_size * hidden_dim] fp8
    uint8_t                    *kv_cache_k,      // [chunk_size * kv_latent_dim] fp8
    uint8_t                    *kv_cache_v       // [chunk_size * kv_latent_dim] fp8
);

/**
 * Prefill all layers for one chunk.
 *
 * Returns: total elapsed microseconds.
 * Output: final hidden state + full KV cache for all layers.
 */
double cpu_prefill_all_layers(
    const cpu_prefill_config_t *cfg,
    int                         chunk_size,
    const uint8_t              *input_tokens,     // [chunk_size * hidden_dim] fp8
    uint8_t                    *output_state,     // [chunk_size * hidden_dim] fp8
    uint8_t                    *kv_cache_k_all,   // [num_layers * chunk_size * kv_latent_dim]
    uint8_t                    *kv_cache_v_all    // [num_layers * chunk_size * kv_latent_dim]
);

/* ── Weight Cache ─────────────────────────────────────────────────────── */

/**
 * Pre-load weights for one layer into CPU memory.
 * Weights are stored in fp8 format (converted from fp4 at load time).
 *
 * Weights should be pinned (mlock) for DMA to FPGA later if needed.
 */
int cpu_weight_cache_load(
    const cpu_prefill_config_t *cfg,
    int                         layer_idx,
    const void                 *weight_data,      // raw fp4 weights from disk
    size_t                      weight_size
);

void cpu_weight_cache_unload(int layer_idx);
void cpu_weight_cache_unload_all(void);

/* ── Performance Monitoring ───────────────────────────────────────────── */

typedef struct {
    double total_us;            // total prefill time
    double gemm_us;             // time spent in GEMM
    double attention_us;        // time spent in attention
    double moe_us;              // time spent in MoE routing
    double effective_tflops;    // achieved TFLOPS
    int    chunks_processed;
    int    tokens_prefilled;
} cpu_prefill_stats_t;

const cpu_prefill_stats_t *cpu_prefill_get_stats(void);
void cpu_prefill_reset_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* CPU_PREFILL_H */
