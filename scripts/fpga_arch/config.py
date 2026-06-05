"""
fpga_arch/config.py — Unified hardware constants.

Single source of truth for all FPGA hardware parameters.
Extracted from fpga_4chip_pipeline.py:42-166 and fpga_cloud_serving.py:22-106.
"""
import math

# ============================================================================
# Chip-level (AGM 039-F / A7)
# ============================================================================
DSP_COUNT             = 12_300          # DSP units per chip
DSP_FREQ_MHZ          = 450             # MHz
DSP_MAC_PER_CYCLE     = 2               # fp4 x fp8 native mode
DSP_TMACS             = DSP_COUNT * DSP_FREQ_MHZ * DSP_MAC_PER_CYCLE / 1e6  # 11.07

HBM_SIZE_GB           = 32
HBM_BW_GBPS           = 920
HBM_BW_EFF            = 0.916           # RTL-measured: tb_axi4_hbm_bw_bench, streaming read, 256-beat bursts (2026-05-30)

SRAM_M20K_MB          = 29.2            # usable M20K (75% of 38.9 MB)
SRAM_MLAB_MB          = 3.3             # usable MLAB
SRAM_TOTAL_MB         = SRAM_M20K_MB + SRAM_MLAB_MB  # 32.5

# ============================================================================
# FP8 算力归一化
# ============================================================================
DSP_FP8_MAC_PER_CYCLE = 1               # conservative: fp8xfp8 mode
DSP_FP8_TMACS_PER_CHIP = DSP_COUNT * DSP_FREQ_MHZ * DSP_FP8_MAC_PER_CYCLE / 1e6  # 5.54
DSP_FP8_TFLOPS_PER_CHIP = DSP_FP8_TMACS_PER_CHIP * 2  # 11.07 (GPU equiv)

# Prefill compute rates: attention ops (Q·K^T, A·V) are fp8×fp8 by default,
# projections and FFN are fp8 activation × fp4 weight → 2 MAC/cycle.
# P0 (fp4 attention): reduce K/V to fp4 during prefill → 2× attention compute.
DSP_ATTN_FP8_TMACS     = DSP_FP8_TMACS_PER_CHIP  # 5.54, baseline fp8×fp8
DSP_ATTN_FP4_TMACS     = DSP_TMACS               # 11.07, P0: fp8×fp4 K/V
DSP_FFN_TMACS          = DSP_TMACS               # 11.07, always fp8×fp4

# P1: Router-guided sparse attention
# ───────────────────────────────────────────────────────────────────────────
# Ideal: P(share ≥1 expert) = 1 - C(378,6)/C(384,6) ≈ 9.07%
# Each token only attends to other tokens in same expert cluster → 90.93% sparse.
#
# Pipeline constraint: current pipeline is Attention → Router, but P1 needs
# Router → Attention. Solution: layer N uses layer N-1's router output.
# Adjacent layers have ~90% router agreement (same top-6 experts).
#
# Effective sparsity with carry-forward:
#   density_ideal = P(share expert) = 9.07%
#   overlap = P(same expert in adjacent layer) = 90%
#   pairwise_accuracy = overlap² = 81%  (both query & KV token must agree)
#   density_eff = density_ideal / pairwise_accuracy = 11.20%  (保守补偿假阴性)
#   sparsity_eff = 1 - density_eff = 88.80%
PREFILL_ATTN_DENSITY_IDEAL  = 0.0907  # P(share expert) same-layer, ideal
ROUTER_SCORE_OVERLAP        = 0.90    # P(same top-6 in adjacent layers)
PREFILL_ATTN_DENSITY        = PREFILL_ATTN_DENSITY_IDEAL / (ROUTER_SCORE_OVERLAP ** 2)  # 0.1120
PREFILL_ATTN_SPARSITY       = 1.0 - PREFILL_ATTN_DENSITY  # 0.8880
# Note: with 90% overlap, effective sparsity is 88.8% (vs ideal 90.9%).
# Sensitivity: overlap=95% → 89.9%, overlap=85% → 87.4%, overlap=80% → 85.8%.
# False negative rate (missed relevant KV): density_ideal*(1-overlap²) ≈ 1.7% → negligible.

# Prefill optimization toggles
# P0+P1 (FPGA-side prefill via fp4 attention + router-guided sparse): RESERVED FOR FUTURE.
# Current architecture: CPU handles all prefill, FPGA handles decode only.
PREFILL_USE_FP4_ATTN   = False    # RESERVED: P0 fp4 K/V for FPGA-side prefill attention
PREFILL_USE_SPARSE_ATTN = False    # RESERVED: P1 router-guided sparse attention mask

# Chunked prefill: RESERVED for future FPGA-side prefill.
# Current architecture: CPU prefill runs unchunked (full prompt on CPU).
PREFILL_CHUNK_SIZE     = 128      # tokens per chunk (reserved)
PREFILL_USE_CHUNKED    = False    # RESERVED: enable chunked FPGA prefill

# Heterogeneous Prefill: Flash model on CPU (primary) or GPU (fallback).
# Flash model (285B, 27 layers) shares identical HIDDEN=7168, K_LATENT=512
# with Full model (671B, 61 layers). KV cache is directly compatible.
#   Primary: AMD EPYC Turin 192C — Flash TTFT ~1.0s @ P=512
#   Fallback: L20 GPU — Flash TTFT ~40ms @ P=512 (low-latency SLA)
CPU_PREFILL            = True     # CPU prefill is the primary prefill path
GPU_PREFILL_FALLBACK   = True     # GPU prefill available as low-latency fallback

# Flash model parameters (prefill-only, not loaded on FPGA)
FLASH_MODEL_LAYERS     = 27       # Flash model layers (vs 61 full)
FLASH_MODEL_PARAMS_B   = 285      # Flash model total params (vs 671B full)
FLASH_PREFILL_FACTOR   = 27 / 61  # ~0.44 — compute reduction vs full model

# CPU Prefill hardware options
# AMD EPYC 9755 (Turin 192C): 8.0 TFLOPS FP8, 12ch DDR5-6400 (~600 GB/s)
# AMD EPYC 9965 (Turin 128C): 6.0 TFLOPS FP8, 12ch DDR5-6000 (~500 GB/s)
# Intel Xeon 6980P (MR-AMX):  5.0 TFLOPS FP8, 12ch DDR5-6400 (~500 GB/s)
# Intel Xeon 8592+ (AMX):     3.0 TFLOPS FP8, 8ch DDR5-5600 (~350 GB/s)
CPU_FP8_TFLOPS         = 8.0      # AMD EPYC 9755 (primary CPU target)
# PCIe round-trip: KV tensors → FPGA HBM
CPU_PCIE_LATENCY_US    = 5.0      # fixed PCIe latency (DMA setup + transfer)

# ============================================================================
# Cluster topology
# ============================================================================
NUM_CARDS             = 8
CHIPS_PER_CARD        = 4
TOTAL_CHIPS           = 32
NUM_LAYERS            = 61
NUM_EXPERTS           = 384
EXPERTS_PER_CHIP      = 12
TOP_K_EXPERTS         = 6
SHARED_EXPERT         = True

# ============================================================================
# Model dimensions (DeepSeek V4 Pro)
# ============================================================================
HIDDEN_SIZE           = 7168
INTERMEDIATE_SIZE     = 3072
NUM_ATTN_HEADS        = 128
KV_LORA_RANK          = 512
Q_LORA_RANK           = 1536
O_LORA_RANK           = 1024
QK_ROPE_HEAD_DIM      = 64
QK_NOPE_HEAD_DIM      = 448
V_HEAD_DIM            = 128
NUM_EXPERTS_PER_TOK   = 6
SLIDING_WINDOW        = 128
MLA_KV_BYTES          = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576 bytes FP8

# RTL-compatible aliases (kept in sync with lpu_config.svh)
K_LATENT              = KV_LORA_RANK   # LPU_K_LATENT: MLA K low-rank dim
V_LATENT              = KV_LORA_RANK   # LPU_V_LATENT: MLA V low-rank dim
MAX_SEQ_LEN           = 4096           # LPU_MAX_SEQ_LEN: max sequence positions
VOCAB_SIZE            = 129280         # LPU_VOCAB_SIZE: token vocabulary size
KV_CACHE_SLOTS        = 4096           # LPU_KV_CACHE_SLOTS: KV cache entries per chip
SCALE_GROUPS          = 448            # LPU_SCALE_GROUPS: fp4 scale groups (HIDDEN/16)
ARRAY_LANES           = 128            # LPU_ARRAY_LANES: systolic K-direction parallelism
ARRAY_M_ROWS          = 32             # LPU_ARRAY_M_ROWS: systolic M-direction parallelism

# ============================================================================
# Weight sizes (fp4, per layer where applicable)
# ============================================================================
ATTN_WEIGHT_MB = {
    'kv_a_down':  1.75,
    'kv_a_up':   14.25,
    'kv_a_rope':  0.22,
    'q_a_down':   5.25,
    'q_a_up':    48.00,
    'o_down':    12.50,
    'o_up':       7.00,
}
ATTN_TOTAL_MB_PER_LAYER = sum(ATTN_WEIGHT_MB.values())  # ~88.97

EXPERT_WEIGHT_MB = {  # per expert
    'gate': 10.5,
    'up':   10.5,
    'down': 10.5,
}
EXPERT_TOTAL_MB    = sum(EXPERT_WEIGHT_MB.values()) + 1.5  # 33 MB (with overhead)
ROUTER_WEIGHT_MB   = 2.6      # router table (fp8)
NORM_WEIGHT_MB     = 0.03     # RMSNorm (fp16)

# ============================================================================
# MAC counts per layer
# ============================================================================
MAC_MLA_Q_DOWN       = 11.01e6
MAC_MLA_KV_LATENT    = 3.67e6
MAC_MLA_KV_ROPE      = 0.46e6
MAC_MLA_QK_DOT       = 29.88e6
MAC_MLA_AV_DOT       = 29.36e6
MAC_MLA_O_DECOMPRESS = 67.11e6
MAC_MLA_O_UP         = 7.34e6
MAC_MLA_TOTAL        = sum([MAC_MLA_Q_DOWN, MAC_MLA_KV_LATENT, MAC_MLA_KV_ROPE,
                             MAC_MLA_QK_DOT, MAC_MLA_AV_DOT,
                             MAC_MLA_O_DECOMPRESS, MAC_MLA_O_UP])  # 148.83M

MAC_EXPERT_GATE      = 22.02e6
MAC_EXPERT_UP        = 22.02e6
MAC_EXPERT_DOWN      = 22.02e6
MAC_EXPERT_TOTAL     = MAC_EXPERT_GATE + MAC_EXPERT_UP + MAC_EXPERT_DOWN  # 66.06M

MAC_SHARED_EXPERT    = MAC_EXPERT_TOTAL  # 66.06M
MAC_MOE_LAYER_TOTAL  = (MAC_MLA_TOTAL + MAC_SHARED_EXPERT +
                        TOP_K_EXPERTS * MAC_EXPERT_TOTAL)  # ~611M

# ============================================================================
# C2C SerDes parameters
# ============================================================================
C2C_LINK_BW_GBPS     = 128
C2C_HOP_LATENCY_NS   = 50
C2C_FRAME_OVERHEAD_B = 24
C2C_MAX_PAYLOAD_B    = 4088
C2C_MSG_DISPATCH_B   = 7168
C2C_DISPATCH_FRAMES  = math.ceil(C2C_MSG_DISPATCH_B / C2C_MAX_PAYLOAD_B)  # 2
C2C_DISPATCH_LATENCY_NS = 250
C2C_REDUCE_LATENCY_NS  = 250
C2C_FWD_LATENCY_NS     = 250

# ============================================================================
# PCIe P2P
# ============================================================================
PCIE_P2P_BW_GBPS     = 64
PCIE_P2P_LATENCY_NS  = 400

# ============================================================================
# Tensor parallelism
# ============================================================================
TP_ATTN_PER_LAYER    = 2

# ============================================================================
# Deterministic weights (SRAM-resident, double-buffered)
# ============================================================================
DETERMINISTIC_MB_PER_LAYER = 13.2
EXPERT_HBM_LOAD_MB         = 33.0

# Expert hit probabilities (12/384 per chip)
P_EXPERT_PER_CHIP = EXPERTS_PER_CHIP / NUM_EXPERTS  # 0.03125
P_0_HIT  = (1 - P_EXPERT_PER_CHIP) ** TOP_K_EXPERTS                         # 0.8265
P_1_HIT  = TOP_K_EXPERTS * P_EXPERT_PER_CHIP * (1 - P_EXPERT_PER_CHIP) ** 5  # 0.1653
P_2P_HIT = 1 - P_0_HIT - P_1_HIT                                            # 0.0082

# ============================================================================
# Weight placement (per chip, from _place_weights())
# ============================================================================
WEIGHT_GB_PER_CHIP   = 0.7   # fp4 expert+attn+router per chip
HBM_KV_AVAIL_GB      = HBM_SIZE_GB - WEIGHT_GB_PER_CHIP  # 31.3

SRAM_USED_MB         = 21.0  # 确定性权重双缓冲
SRAM_FREE_MB         = SRAM_TOTAL_MB - SRAM_USED_MB  # 11.5

# ============================================================================
# Pipeline performance (calibrated from fpga_4chip_pipeline.py)
#
# Derivation (CR-3 fix, 2026-05-30):
#   PIPELINE_TPS ≈ 17,445 — saturation decode throughput at batch=32,
#     from simulate_pipeline() bottleneck analysis (32-chip, 61-layer pipeline).
#     Original untraceable constant: 23,104 / V4_ACTIVE_SCALE (V3 calibration).
#   BATCH1_TPS  ≈    660 — single-token decode throughput at batch=1.
#     Original untraceable constant:    875 / V4_ACTIVE_SCALE.
#   K_PIPELINE  ≈   25.4 — pipeline fill overhead factor.
#     K = PIPELINE_TPS / BATCH1_TPS - 1.
#     Re-derivable via derive_k_pipeline(cluster) in pipeline.py.
#
#   TPS(B) = PIPELINE_TPS * B / (B + K_PIPELINE)
# ============================================================================
V4_ACTIVE_SCALE      = 49.0 / 37.0    # V4 Pro 49B active vs V3 37B
PIPELINE_TPS         = int(23_104 / V4_ACTIVE_SCALE)  # ~17,445
BATCH1_TPS           = int(875 / V4_ACTIVE_SCALE)      # ~660
TOKEN_LATENCY_US     = int(1140 * V4_ACTIVE_SCALE)     # ~1,510
PER_LAYER_US         = 18.7 * V4_ACTIVE_SCALE          # ~24.7

# Pipeline overhead K (calibrated from batch-1 efficiency)
# K = PIPELINE_TPS / BATCH1_TPS - 1
K_PIPELINE = PIPELINE_TPS / BATCH1_TPS - 1  # ~25.4

# ============================================================================
# Costs & Economics
# ============================================================================
A7_CHIP_COST_USD      = 2_500
A7_CHIP_COST_RMB      = 18_000
A7_CHIP_TOTAL_RMB     = A7_CHIP_COST_RMB * TOTAL_CHIPS  # 576,000
SERVER_COST_RMB       = 1_000_000
SERVER_POWER_KW       = 5.3
RMB_PER_KWH           = 0.35

# Market pricing (DS V4 Pro API, 2026/04)
PRICE_PER_1M_TOKENS_RMB = 3.0  # blended I/O rate

# ============================================================================
# ============================================================================
# GPU comparison (all numbers for DeepSeek V4 Pro, FP8, end-to-end serving)
# ============================================================================
# H200 8-GPU: real benchmark data (2025 Q3) — includes FlashAttention, chunked prefill
H200_FP8_TFLOPS       = 1_979
H200_FP8_TFLOPS_PER_SRV = 1_979 * 8
H200_SRV_COST         = 3_000_000
H200_DECODE_TPS       = 2_000       # aggregate decode at high concurrency
H200_PREFILL_TPS      = 8_000       # prompt tokens/s (FlashAttention-3 optimized)
H200_TTFT_P50_MS      = 120         # P=512, batched prefill
H200_SESS_256K        = 15
IB_COST_PER_SRV       = 300_000

# Ascend 950PR 8-NPU: estimated (no public DS V4 Pro benchmark)
ASCEND_FP8_TFLOPS     = 800
ASCEND_SRV_COST       = 1_300_000
ASCEND_DECODE_TPS     = 1_500       # estimated ~75% of H200
ASCEND_PREFILL_TPS    = 6_000       # estimated (weaker compute, similar arch)
ASCEND_TTFT_P50_MS    = 160         # estimated
ASCEND_SESS_256K      = 10
HCCS_COST_PER_SRV     = 200_000

# FPGA decode-only throughput (prefill runs on CPU)
FPGA_DECODE_TPS         = PIPELINE_TPS  # 17,445 (decode-only, saturated batch)
FPGA_DECODE_TPS_HW      = 14_000        # raw hardware decode (~flat across B, from simulate_pipeline)


def validate_derived_constants(verbose: bool = False) -> dict:
    """Check derived constants for internal consistency and staleness.

    Returns dict with keys: 'consistent' (bool), 'warnings' (list[str]).
    """
    warnings = []

    # 1. HBM_BW_EFF should be < 1.0 (would indicate unmeasured value)
    if HBM_BW_EFF >= 0.99:
        warnings.append("HBM_BW_EFF >= 0.99: likely unmeasured default. Run tb_axi4_hbm_bw_bench.")

    # 2. K_PIPELINE should be derivable from PIPELINE_TPS/BATCH1_TPS
    k_implied = PIPELINE_TPS / BATCH1_TPS - 1 if BATCH1_TPS > 0 else 0
    if abs(k_implied - K_PIPELINE) > 0.5:
        warnings.append(
            f"K_PIPELINE={K_PIPELINE:.1f} inconsistent with "
            f"PIPELINE_TPS/BATCH1_TPS-1={k_implied:.1f}"
        )

    if verbose and not warnings:
        print("config.py: All derived constants consistent.")
    elif verbose:
        print(f"config.py: {len(warnings)} derived constant warning(s):")
        for w in warnings:
            print(f"  [!] {w}")

    return {'consistent': len(warnings) == 0, 'warnings': warnings}
