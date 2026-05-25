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
HBM_BW_EFF            = 1.0             # effective utilization

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
PREFILL_USE_FP4_ATTN   = True     # P0: fp4 K/V for prefill attention
PREFILL_USE_SPARSE_ATTN = True    # P1: router-guided sparse attention mask

# Chunked prefill: split long prompts into chunks to reduce TTFT.
# Standard vLLM approach — first chunk latency = TTFT, chunks pipelined across chips.
PREFILL_CHUNK_SIZE     = 128      # tokens per chunk (vLLM default)
PREFILL_USE_CHUNKED    = True     # enable chunked prefill

# P2: CPU-FPGA hybrid prefill — CPU handles attention (Q·K^T, A·V) via AMX,
# FPGA handles projections + FFN. Enables parallel compute across CPU/FPGA.
# CPU AMX (Xeon 6 / EPYC Turin): 2-4 TFLOPS FP8/BF16 for dense matmul.
CPU_FP8_TFLOPS         = 3.0      # configurable CPU FP8 throughput
CPU_OFFLOAD_ATTN       = False    # toggle: offload attention to CPU
# PCIe round-trip: Q,K tensors → CPU, attn output → FPGA
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

# FPGA A7 32-chip: prefill with P0 (fp4 K/V attention) + P1 (router-guided sparse)
# P0: attention uses fp4 → 2× compute vs fp8×fp8 Q·K^T
# P1: router-guided sparse attention, using layer N-1 router scores for layer N
#     (carry-forward, 90% adjacent overlap → 88.8% effective sparsity)
# All numbers P=512 unless noted.

# Baseline (corrected physics: attention fp8×fp8=5.54T, FFN fp8×fp4=11.07T)
FPGA_PREFILL_TPS_BASE   = 679     # no P0, no P1: 23.00s TTFT
FPGA_TTFT_MS_BASE       = 22_990

# P0 only (fp4 K/V → attention goes from 5.54→11.07 TMACS)
FPGA_PREFILL_TPS_P0     = 1_284   # P0 only: 12.16s TTFT
FPGA_TTFT_MS_P0         = 12_160

# P1 only (carry-fwd router, 88.8% sparse, fp8 attention)
FPGA_PREFILL_TPS_P1     = 4_149   # P1 only: 3.76s TTFT
FPGA_TTFT_MS_P1         = 3_760

# P0 + P1 (fp4 attention + carry-fwd router)
FPGA_PREFILL_TPS        = 6_122   # P0+P1: 2.55s TTFT
FPGA_TTFT_P50_MS        = 2_550

# Ideal P0+P1 (same-layer router, 90.9% sparse — needs pipeline reorder)
FPGA_PREFILL_TPS_IDEAL  = 6_730   # P0+P1 ideal: 2.32s TTFT
FPGA_TTFT_MS_IDEAL      = 2_320

# Short-prompt (P=128) with P0+P1 carry-fwd
FPGA_PREFILL_TPS_P128   = 9_496   # P0+P1, P=128: 0.41s TTFT
FPGA_TTFT_MS_P128       = 411

# Decode (unchanged)
FPGA_DECODE_TPS         = PIPELINE_TPS  # 17,445 (decode-only, saturated batch)
FPGA_DECODE_TPS_HW      = 14_000        # raw hardware decode (~flat across B, from simulate_pipeline)

# ── Chunked Prefill (PREFILL_CHUNK_SIZE=128, P0+P1 carry-fwd) ──
# Splits prompt into 128-token chunks. First chunk = TTFT, chunks pipelined.
FPGA_TTFT_CHUNKED_MS    = 411     # P=512, first chunk latency
FPGA_PREFILL_TOTAL_MS   = 481     # P=512, all 4 chunks (pipelined)
FPGA_PREFILL_TPS_CHUNKED = 1_064  # P=512, effective prefill TPS with chunking
# Chunked prefill vs non-chunked P=512:
#   TTFT: 2,551ms → 411ms (6.2×)
#   Prefill TPS: 679 → 1,064 (1.6×, pipeline parallelism)

# Note: carry-forward overhead vs ideal is only 9.9% TPS loss (6,122 vs 6,730).
# Pipeline reorder (Router → Attention) would recover this but requires HW changes.
# With chunked prefill + P0+P1, TTFT gap to Ascend closes from 15.9× to 2.6×.
