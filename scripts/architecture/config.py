"""
全局配置 — 所有参数与方案文档一致。

命名: HW_ = 硬件参数, SW_ = 软件参数, SYS_ = 系统参数
"""

# ═══════════════════════════════════════════════════════════
# FPGA 硬件参数 (Agilex 7 AGFB027)
# ═══════════════════════════════════════════════════════════
HW_FPGA_DSP_TOPS = 8.44       # TMACs/s (1 MAC = 1 op)
HW_FPGA_DSP_COUNT = 9375      # DSP 数量
HW_FPGA_HBM_SIZE_GB = 32      # HBM2e 总容量
HW_FPGA_HBM_BW_GBPS = 920     # HBM2e 理论带宽
HW_FPGA_HBM_EFF = 0.87        # 顺序读效率
HW_FPGA_SRAM_SIZE_MB = 43     # 片上 SRAM (M20K + MLAB)

# 系统拓扑: 8 张 PCIe 卡, 每卡 4 片 FPGA
HW_FPGA_CARD_COUNT = 8        # 物理卡数
HW_FPGAS_PER_CARD = 4         # 每卡 FPGA 芯片数
HW_FPGA_CHIP_COUNT = HW_FPGA_CARD_COUNT * HW_FPGAS_PER_CARD  # 32 片

# 卡内互联: SerDes 片间直连 (PCB 走线)
HW_SERDES_LANE_GBPS = 56      # PAM4 per lane
HW_SERDES_LANES = 8           # 每链路 lane 数
HW_SERDES_BW_GBPS = HW_SERDES_LANE_GBPS * HW_SERDES_LANES  # 448 Gbps = 56 GB/s
HW_SERDES_HOP_US = 0.01       # 片内 PCB 走线延迟

# ═══════════════════════════════════════════════════════════
# PCIe 参数
# ═══════════════════════════════════════════════════════════
HW_PCIE_BW_GBPS = 32          # PCIe Gen5 x16 单向有效带宽
HW_PCIE_LATENCY_US = 2.0      # DMA 启动延迟 (固定开销)
HW_PCIE_MTU_KB = 256          # 单次 DMA 最大传输

# ═══════════════════════════════════════════════════════════
# 模型参数 (DeepSeek V4 Pro)
# ═══════════════════════════════════════════════════════════
MODEL_HIDDEN_SIZE = 7168
MODEL_INTERMEDIATE_SIZE = 3072
MODEL_NUM_LAYERS = 61
MODEL_NUM_HEADS = 128
MODEL_KV_LORA_RANK = 512
MODEL_Q_LORA_RANK = 1536
MODEL_QK_ROPE_DIM = 64
MODEL_V_HEAD_DIM = 128
MODEL_NUM_EXPERTS = 384
MODEL_TOP_K = 6
MODEL_EXPERTS_PER_FPGA = 12   # 384/32
MODEL_SLIDING_WINDOW = 128

# ═══════════════════════════════════════════════════════════
# 权重大小 (MB, fp4 格式)
# ═══════════════════════════════════════════════════════════
WEIGHT_EXPERT_MB = 33.0           # 单个 routed expert per-layer (gate+up+down)
WEIGHT_SHARED_EXPERT_MB = 33.0    # Shared expert per-layer (gate+up+down)
WEIGHT_ATTENTION_MB = 6.2         # MLA 所有权重 per-layer
WEIGHT_ROUTER_MB = 0.37           # Router 权重表
WEIGHT_DETERMINISTIC_MB = WEIGHT_ATTENTION_MB + WEIGHT_ROUTER_MB + WEIGHT_SHARED_EXPERT_MB  # ≈ 39.6 MB

# ═══════════════════════════════════════════════════════════
# 每层 MACs (百万), 用于 DSP 延迟估算
# ═══════════════════════════════════════════════════════════
MACS_MLA_M = 97          # MLA Attention MACs
MACS_SHARED_EXPERT_M = 66  # Shared Expert MACs
MACS_ROUTED_EXPERT_M = 66  # 单个 Routed Expert MACs

# ═══════════════════════════════════════════════════════════
# vLLM Scheduler 参数
# ═══════════════════════════════════════════════════════════
SW_MAX_BATCH_TOKENS = 8192     # 每步最多处理的 token 数
SW_MAX_NUM_SEQS = 256          # 最大并发序列数
SW_BLOCK_SIZE = 16             # KV cache block 大小 (tokens)
SW_MAX_PROMPT_LEN = 32768      # 最大 prompt 长度

# ═══════════════════════════════════════════════════════════
# KV Cache 参数 (MLA 压缩)
# ═══════════════════════════════════════════════════════════
KV_BYTES_PER_TOKEN = 576       # (kv_lora_rank 512 + rope 64) × FP8
KV_BLOCKS_PER_GB = (1024**3) // (KV_BYTES_PER_TOKEN * SW_BLOCK_SIZE)

# ═══════════════════════════════════════════════════════════
# 系统拓扑 (8卡 × 4片 = 32 芯片)
# ═══════════════════════════════════════════════════════════
# TP 组: 每组 8 片 (2 张卡的 FPGA), 共 4 组
# 每组内 8 片 × 12 experts = 96 expert 覆盖 (25%)
# 卡内 4 片走 SerDes, 卡间走 100GbE
SYS_TP_SIZE = 8                # 每 TP 组芯片数
SYS_TP_GROUPS = HW_FPGA_CHIP_COUNT // SYS_TP_SIZE  # 32/8 = 4
# 每组内: 卡内 SerDes (4片) + 跨卡 Ethernet (2卡)
SYS_CARDS_PER_TP_GROUP = SYS_TP_SIZE // HW_FPGAS_PER_CARD  # 2 卡/组

# 每片 FPGA 的 HBM 容量检查
_HBM_MB = HW_FPGA_HBM_SIZE_GB * 1024
_EXPERT_TOTAL_MB = MODEL_EXPERTS_PER_FPGA * MODEL_NUM_LAYERS * WEIGHT_EXPERT_MB
assert _EXPERT_TOTAL_MB < _HBM_MB, \
    f"HBM overflow: {_EXPERT_TOTAL_MB:.0f} MB > {_HBM_MB} MB @ {MODEL_EXPERTS_PER_FPGA} experts/FPGA"
