"""
流水线 Stage 硬件接口定义 — Intel Agilex 7 (Avalon-ST/MM).

每个 Stage 对应一个 RTL 模块. 全部使用 Intel FPGA 标准接口:
  - 数据通路: Avalon-ST (source_valid/sink_ready/endofpacket)
  - 权重访存: Avalon-MM (address/read/write/readdata/writedata)
  - 控制:      conduit (start/done/idle + 层参数)

与 AXI 的差异:
  - tvalid/tready → source_valid / sink_ready
  - tlast         → endofpacket (+ startofpacket)
  - tkeep         → empty (最后一拍无效字节数, 反向语义)
  - tid/tuser     → channel (最多 128 个逻辑通道)
  - awaddr/araddr → address (统一读写地址)
  - wvalid/rvalid → write / readdatavalid

数据宽度:
  - FP8 activation: 8b/elem, tile = 128 lanes × 8b = 1024b
  - fp4 weight:     4b/elem, 128 lanes × 4b = 512b
  - FP32 accum:     32b/elem
"""

from dataclasses import dataclass
from ... import config

# ═══════════════════════════════════════════════════════════
# 通用总线定义 (Avalon)
# ═══════════════════════════════════════════════════════════

@dataclass
class AvalonStPort:
    """Avalon-ST 端口 — 节拍间 activation 流.

    Intel 标准命名:
      source_data/source_valid/source_error
      sink_ready
      startofpacket/endofpacket
      channel
      empty
    """
    data_width: int = 1024      # 128 lanes × 8b FP8
    channel_width: int = 4      # 逻辑通道 (token index 等)
    empty_width: int = 4        # empty = 7 表示最后一拍 1 字节有效


@dataclass
class AvalonMmPort:
    """Avalon-MM 端口 — SRAM/HBM 权重读取.

    Intel 标准命名:
      address / read / write
      readdata / readdatavalid
      writedata / byteenable
      waitrequest (反压)
    """
    addr_width: int = 20
    data_width: int = 512       # 128 fp4 = 512b
    read_latency: int = 2       # SRAM=2 cycle, HBM=~2000 cycle


# ═══════════════════════════════════════════════════════════
# 节拍 0: PCIe RX
# ═══════════════════════════════════════════════════════════
# 目的: 接收 Host 下发的 activation, 写入片上 staging SRAM
# 输入:  PCIe Hard IP Avalon-ST RX (来自 Host DMA)
# 输出:  Avalon-ST → SRAM staging buffer → Beat 1
# 权重:  无
# 备注:  PCIe RX 和 staging write 双缓冲 ping-pong

PCIeRxInterface = {
    "module": "pcie_rx",
    "beat": 0,
    "clock_domain": "pcie_clk (250 MHz)",

    # ── PCIe RX: 来自 Intel P-Tile Hard IP ──
    "pcie_rx_st": {
        "source_data":   "256b  (Gen5 x16: 32B/cycle @250MHz)",
        "source_valid":  "1b",
        "sink_ready":    "1b    FPGA 侧就绪",
        "startofpacket": "1b   TLP 起始",
        "endofpacket":   "1b   TLP 结束",
        "empty":         "4b   最后一拍无效字节数",
        "bar":           "3b   BAR 选择",
    },

    # ── 输出: activation → Beat 1 ──
    "act_out_st": {
        "source_data":   "1024b (128×FP8 activation tile 行)",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b   一个 tensor 起始",
        "endofpacket":   "1b   一个 tensor 结束",
        "channel":       "4b   token index (batch内)",
        "empty":         "4b",
    },

    # ── staging buffer 写 (SRAM Avalon-MM) ──
    "staging_wr_mm": {
        "address":    "14b   staging buffer 地址",
        "write":      "1b    写使能",
        "writedata":  "1024b",
        "byteenable": "128b  128 字节使能",
        "waitrequest": "1b   SRAM 反压",
    },

    "ctrl": {
        "start":      "1b   Host 发起新层传输",
        "done":       "1b   本层 activation 写入完毕",
        "act_bytes":  "16b  本层总字节数 = batch × 7168",
        "layer_idx":  "8b",
    },
}


# ═══════════════════════════════════════════════════════════
# 节拍 1: MLA Q/K 低秩投影 + RoPE
# ═══════════════════════════════════════════════════════════
# 目的: Q = W_Q_down @ x, KV = W_KV_down @ x + RoPE
# 权重:  SRAM W_Q_down [1536,7168] fp4 + W_KV_down [512,7168] fp4
# DSP:   两个 fp4×FP8 GEMM tile 流

MLAQKInterface = {
    "module": "mla_qk",
    "beat": 1,
    "clock_domain": "dsp_clk (500 MHz)",

    "act_in_st": {
        "source_data":   "1024b  activation tile",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b    token index",
    },

    # ── 权重读 (SRAM Avalon-MM) ──
    "weight_rd_mm": {
        "address":        "20b  {weight_sel[4b], row[8b], col[8b]}",
        "read":           "1b",
        "readdata":       "512b (128×fp4 tile 行)",
        "readdatavalid":  "1b",
        "waitrequest":    "1b",
        "weight_sel":     "4b   0=W_Q_down, 1=W_KV_down",
    },

    # ── RoPE 参数 ──
    "rope": {
        "sin_table": "64×8b   预计算 sin (QK_ROPE_DIM=64)",
        "cos_table": "64×8b   预计算 cos",
        "position":  "16b     token 在序列中的位置",
    },

    # ── 输出: q_latent + kv_latent 合并 ──
    "latent_out_st": {
        "source_data":   "1024b ({q_latent[128], kv_latent[128]} 合并)",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b",
        "q_done":        "1b   Q 流结束标志",
        "kv_done":       "1b   KV 流结束标志",
    },

    "ctrl": {
        "start":      "1b",
        "done":       "1b",
        "batch_size": "8b",
        "position":   "16b  RoPE 位置",
    },
}


# ═══════════════════════════════════════════════════════════
# 节拍 2: MLA Attention
# ═══════════════════════════════════════════════════════════
# 目的: K/V/Q up-proj + Flash Attention + O-proj
# 权重:  SRAM W_K_up, W_V_up, W_Q_up, W_O_proj (fp4)
# KV cache: HBM 读历史 KV, 写当前 KV
# 最复杂节拍 — 含 Flash Attention 状态机

MLAAttentionInterface = {
    "module": "mla_attention",
    "beat": 2,
    "clock_domain": "dsp_clk (500 MHz)",

    "latent_in_st": {
        "source_data":   "1024b ({q_latent, kv_latent} 合并)",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b",
    },

    # ── 权重读 (SRAM) ──
    "weight_rd_mm": {
        "address":        "20b",
        "read":           "1b",
        "readdata":       "512b",
        "readdatavalid":  "1b",
        "waitrequest":    "1b",
        "weight_sel":     "4b   0=K_up, 1=V_up, 2=Q_up, 3=O_proj",
    },

    # ── KV Cache (HBM Avalon-MM) ──
    "kv_cache_mm": {
        # 读历史 KV
        "rd_address":       "32b  {layer[8b], head[7b], seq_pos[16b]}",
        "rd_read":          "1b",
        "rd_readdata":      "2048b ({K[128], V[128]} × 128 heads × 8b)",
        "rd_readdatavalid": "1b",
        "rd_waitrequest":   "1b",
        # 写当前 KV
        "wr_address":       "32b",
        "wr_write":         "1b",
        "wr_writedata":     "2048b",
        "wr_waitrequest":   "1b",
        "cache_len":        "16b   当前 KV cache 长度",
    },

    # ── Flash Attention 状态机 ──
    "flash_attn": {
        "q_tile_idx":      "8b    Q 分块循环索引",
        "kv_tile_start":   "16b   K/V tile 起始位置",
        "kv_tile_end":     "16b   K/V tile 结束位置",
        "softmax_scale":   "8b    1/sqrt(d)",
        "running_max":     "32b   online softmax 当前 max (FP32)",
        "running_sum":     "32b   online softmax 当前 sum (FP32)",
        "state":           "3b    0=IDLE, 1=LOAD_Q, 2=LOAD_KV, 3=SCORE, 4=SOFTMAX, 5=OUTPUT",
    },

    "attn_out_st": {
        "source_data":   "1024b (128×FP8 attention output)",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b",
    },

    "ctrl": {
        "start":      "1b",
        "done":       "1b",
        "batch_size": "8b",
        "seq_len":    "16b",
        "is_prefill": "1b   1=全量 KV 计算, 0=增量 decode",
    },
}


# ═══════════════════════════════════════════════════════════
# 节拍 3: Shared Expert SwiGLU
# ═══════════════════════════════════════════════════════════
# 目的: Shared Expert FFN (gate/up SiLU + down)
# 权重:  SRAM W_gate [3072,7168] + W_up [3072,7168] + W_down [7168,3072]
# DSP:   3 个 fp4×FP8 GEMM + SiLU

SharedExpertInterface = {
    "module": "shared_expert",
    "beat": 3,
    "clock_domain": "dsp_clk (500 MHz)",

    "act_in_st": {
        "source_data":   "1024b",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b",
    },

    "weight_rd_mm": {
        "address":       "20b",
        "read":          "1b",
        "readdata":      "512b (128×fp4)",
        "readdatavalid": "1b",
        "waitrequest":   "1b",
        "weight_sel":    "3b   0=gate, 1=up, 2=down",
    },

    # ── SiLU LUT ──
    "silu": {
        "din":      "1024b 128×FP8 输入",
        "din_valid": "1b",
        "dout":     "1024b SiLU(gate) * up 输出",
        "dout_valid": "1b",
        "lut_addr": "8b    SiLU LUT 读地址 (256 项)",
        "lut_q":    "8b    SiLU LUT 输出 FP8",
    },

    "shared_out_st": {
        "source_data":   "1024b",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b",
    },

    "ctrl": {
        "start":      "1b",
        "done":       "1b",
        "batch_size": "8b",
    },
}


# ═══════════════════════════════════════════════════════════
# 节拍 4: MoE Router
# ═══════════════════════════════════════════════════════════
# 目的: W_router @ x → softmax → Top-K (6 from 384)
# 权重:  SRAM W_router [384, 7168] fp4 (~0.37MB)
# 备注:  softmax + Top-K 用少量 DSP + LUT

RouterInterface = {
    "module": "router",
    "beat": 4,
    "clock_domain": "dsp_clk (500 MHz)",

    "act_in_st": {
        "source_data":   "1024b",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b",
    },

    "weight_rd_mm": {
        "address":       "20b",
        "read":          "1b",
        "readdata":      "512b (128×fp4 router weight tile)",
        "readdatavalid": "1b",
        "waitrequest":   "1b",
        "weight_sel":    "1b   0=W_router",
    },

    # ── Top-K 排序 ──
    "topk": {
        "logits_din":    "32b   FP32 partial logit (逐 expert 累加)",
        "logits_valid":  "1b",
        "logits_eid":    "9b    expert index (0..383)",
        "top_k_id":      "9b×6  per-cycle 输出一个 expert ID",
        "top_k_weight":  "8b×6  FP8 router weight",
        "top_k_valid":   "1b",
        "top_k_done":    "1b    6 个全部输出完成",
        "state":         "3b    0=WAIT_LOGITS, 1=ACCUM, 2=SORT, 3=EMIT",
    },

    "expert_info_out_st": {
        "source_data":   "72b   单拍发一个 expert info (eid+weight)",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b   第一个 expert",
        "endofpacket":   "1b   第六个 expert (TOP_K=6)",
        "channel":       "4b    token index",
    },

    "ctrl": {
        "start":       "1b",
        "done":        "1b",
        "batch_size":  "8b",
        "top_k":       "4b    MODEL_TOP_K=6",
        "num_experts": "10b   MODEL_NUM_EXPERTS=384",
    },
}


# ═══════════════════════════════════════════════════════════
# 节拍 5: Expert Fetch
# ═══════════════════════════════════════════════════════════
# 目的: 根据 Router 输出的 expert IDs 获取权重/结果
#       命中 → HBM 本地读; Miss → 以太网 RDMA 远端结果
# 纯数据传输, 零 DSP. HBM 和 Ethernet 可并行.

ExpertFetchInterface = {
    "module": "expert_fetch",
    "beat": 5,
    "clock_domain": "hbm_clk (800 MHz) / eth_clk (322 MHz)",

    "expert_info_in_st": {
        "source_data":   "72b   单拍一个 expert (eid+weight)",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b   第六个 expert 结束",
        "channel":       "4b",
    },

    # ── HBM 读通道 (Avalon-MM) ──
    "hbm_rd_mm": {
        "address":        "28b   HBM 地址 = expert_base[eid] + offset",
        "read":           "1b",
        "readdata":       "1024b (128×fp4 per beat)",
        "readdatavalid":  "1b",
        "waitrequest":    "1b",
        "burstcount":     "8b   一次 burst 长度",
        "local_experts":  "384b  bitmap: 哪些 expert 在本地 HBM",
    },

    # ── 以太网 RDMA (100GbE MAC Avalon-ST) ──
    "eth_rdma": {
        # 请求发送
        "tx_st_data":       "256b  RDMA read request",
        "tx_st_valid":      "1b",
        "tx_st_sop":        "1b",
        "tx_st_eop":        "1b",
        "tx_st_ready":      "1b    MAC 侧就绪",
        "req_expert_id":    "9b",
        "req_owner_card":   "5b    目标 FPGA 卡号 (0..29)",
        # 结果接收
        "rx_st_data":       "1024b 远端 expert 结果 FP8",
        "rx_st_valid":      "1b",
        "rx_st_sop":        "1b",
        "rx_st_eop":        "1b",
        "rx_size_bytes":    "16b   batch × 7168",
        "eth_link_up":      "1b    链路状态",
        "credit_avail":     "8b    RDMA 流控 credit",
    },

    # ── 权重 buffer 就绪状态 ──
    "ready_map": {
        "expert_ready":  "6b    bitmap: 每位标记一个 expert 数据已就绪",
        "expert_source": "6b    bitmap: 0=HBM 本地, 1=Ethernet 远端",
        "all_ready":     "1b    全部 6 路就绪 → 启动 Beat 6",
    },

    "ctrl": {
        "start":         "1b",
        "done":          "1b",
        "top_k":         "4b",
        "hbm_bw_budget": "16b   HBM 带宽预算 (避免与 KV cache 争抢)",
    },
}


# ═══════════════════════════════════════════════════════════
# 节拍 6: Routed Expert SwiGLU
# ═══════════════════════════════════════════════════════════
# 目的: 每个本地命中 expert 做 SwiGLU FFN
# 远端 expert 结果已由节拍5以太网就绪, 本拍只需加权
# 权重:  HBM buffer (节拍5 已预取到片上 buffer)
# DSP:   每个 expert ~66M MACs

RoutedExpertInterface = {
    "module": "routed_expert",
    "beat": 6,
    "clock_domain": "dsp_clk (500 MHz)",

    "act_in_st": {
        "source_data":   "1024b activation (与 shared expert 同输入)",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b",
    },

    # ── 本地权重 buffer (节拍5 HBM 预取后写入) ──
    "weight_buf_rd": {
        "buf_sel":    "4b   选择哪个 expert 的 buffer",
        "buf_addr":   "12b  buffer 内 tile 地址",
        "buf_rdata":  "512b 128×fp4 weight tile",
        "buf_rvalid": "1b   (SRAM buffer 读延迟 ~2 cycle)",
        "weight_sel": "2b   0=gate, 1=up, 2=down",
    },

    # ── 远端结果 buffer (节拍5 Eth RDMA 写入) ──
    "remote_buf_rd": {
        "expert_id":      "9b",
        "buf_rdata":      "1024b 远端计算的 FFN 输出 FP8",
        "buf_rvalid":     "1b",
        "buf_rd_done":    "1b   该 expert 全部 tile 读完",
        "router_weight":  "8b   该 expert 的路由权重 FP8",
    },

    # ── SiLU ──
    "silu": {
        "din":       "1024b",
        "din_valid": "1b",
        "dout":      "1024b SiLU(gate) * up",
        "dout_valid": "1b",
    },

    "expert_out_st": {
        "source_data":   "1024b weighted expert output",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b   per expert 起始",
        "endofpacket":   "1b   per expert 结束",
        "channel":       "4b    token index",
        "expert_id":     "9b    对应的 expert ID",
        "is_remote":     "1b    1=远端结果 (跳过计算, 直接加权)",
    },

    "ctrl": {
        "start":        "1b",
        "done":         "1b",
        "batch_size":   "8b",
        "num_local":    "4b   本地命中 expert 数",
        "num_remote":   "4b   远端 expert 数",
        "active_expert": "3b  当前处理的 expert 序号 (0..5)",
    },
}


# ═══════════════════════════════════════════════════════════
# 节拍 7: Aggregate
# ═══════════════════════════════════════════════════════════
# 目的: shared_out + sum(weight[i] × expert_out[i]) + RMS Norm
# 输入:  shared expert 输出 + 多个 routed expert 输出 (仲裁)
# 输出:  final hidden_state [batch, 7168] FP8
# 权重:  无
# DSP:   element-wise MAD + RMS Norm (极小)

AggregateInterface = {
    "module": "aggregate",
    "beat": 7,
    "clock_domain": "dsp_clk (500 MHz)",

    # ── 两路输入 (仲裁合并) ──
    "shared_in_st": {
        "source_data":   "1024b shared expert 输出",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
    },

    "expert_in_st": {
        "source_data":   "1024b expert 输出 (最多 6 路仲裁)",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b",
        "expert_id":     "9b",
        "router_weight": "8b    FP8 路由权重",
    },

    # ── 累加器 + RMS Norm ──
    "accum": {
        "partial_sum":  "1024b FP8 运行累加和",
        "accum_valid":  "1b   当前 cycle 有 expert_out 需要累加",
        "accum_done":   "1b   全部 expert 累加完成",
        "norm_in":      "1024b 送入 RMS Norm 的完整求和结果",
        "rms_scale":    "1024b 128×FP8 可学习 scale",
        "norm_out":     "1024b 128×FP8 归一化输出",
        "norm_valid":   "1b",
    },

    "hidden_out_st": {
        "source_data":   "1024b final hidden_state [batch, 7168] FP8",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b",
    },

    "ctrl": {
        "start":       "1b",
        "done":        "1b",
        "batch_size":  "8b",
        "num_experts": "4b   需要累加的 expert 总数 (0..6)",
        "tp_reduce":   "1b   1=需要 AllReduce (TP>1), 0=单卡直接输出",
    },
}


# ═══════════════════════════════════════════════════════════
# 节拍 8: PCIe TX
# ═══════════════════════════════════════════════════════════
# 目的: 将 final hidden state 通过 PCIe DMA 返回 Host
# 输入:  hidden_state [batch, 7168] FP8
# 输出:  PCIe Hard IP Avalon-ST TX
# 备注:  与下一层 Beat 0 (PCIe RX) 可 overlap (全双工)

PCIeTxInterface = {
    "module": "pcie_tx",
    "beat": 8,
    "clock_domain": "pcie_clk (250 MHz)",

    "hidden_in_st": {
        "source_data":   "1024b final hidden state",
        "source_valid":  "1b",
        "sink_ready":    "1b",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "channel":       "4b",
    },

    # ── PCIe TX: 到 Intel P-Tile Hard IP ──
    "pcie_tx_st": {
        "source_data":   "256b  PCIe TX (32B/cycle @250MHz)",
        "source_valid":  "1b",
        "sink_ready":    "1b   (来自 PCIe IP core)",
        "startofpacket": "1b",
        "endofpacket":   "1b",
        "empty":         "4b",
    },

    "ctrl": {
        "start":     "1b",
        "done":      "1b",
        "out_bytes": "16b  本层输出字节数 = batch × 7168",
        "layer_idx": "8b",
    },
}


# ═══════════════════════════════════════════════════════════
# 流水线顶层 — 连接全部 9 Stage
# ═══════════════════════════════════════════════════════════

PipelineTopInterface = {
    "module": "fpga_pipeline_top",
    "description": "单卡 FPGA 推理流水线顶层 — 9 Stage 串联, Avalon-ST 互联",

    # ── 外部接口 ──
    "pcie": {
        "rx_st": "← P-Tile RX (Host→FPGA), 连 Beat 0",
        "tx_st": "→ P-Tile TX (FPGA→Host), 连 Beat 8",
    },

    "sram": {
        "staging_mm":    "↔ Beat 0 (activation staging buffer)",
        "weight_mm":     "↔ Beat 1/2/3/4 (确定性权重: MLA+Shared+Router)",
        "weight_buf_mm": "↔ Beat 5→6 (HBM 预取后写入/读出的权重 buffer)",
    },

    "hbm": {
        "weight_mm":  "↔ Beat 5 (expert 权重读取)",
        "kv_cache_mm": "↔ Beat 2 (KV cache 读写)",
    },

    "ethernet": {
        "rdma_tx_st": "→ Beat 5 (远端 expert 请求 MAC TX)",
        "rdma_rx_st": "← Beat 5 (远端 expert 结果 MAC RX)",
        "allreduce":  "↔ Beat 7 后 TP 组 AllReduce (走 Pipeline Top 的 eth_mac)",
    },

    # ── Stage 间 Avalon-ST 互联 ──
    "interconnect": {
        "Beat0→1": "pcie_rx.act_out_st        → mla_qk.act_in_st         (1024b)",
        "Beat1→2": "mla_qk.latent_out_st       → mla_attention.latent_in_st (1024b)",
        "Beat2→3": "mla_attention.attn_out_st  → shared_expert.act_in_st   (1024b)",
        "Beat3→4": "shared_expert.shared_out_st → router.act_in_st          (1024b)",
        "Beat4→5": "router.expert_info_out_st  → expert_fetch.expert_info_in_st (72b)",
        "Beat5→6": "expert_fetch.ready_map     → routed_expert (all_ready 启动) + weight_buf/buf_rd",
        "Beat6→7": "routed_expert.expert_out_st→ aggregate.expert_in_st    (1024b)",
        "Beat7→8": "aggregate.hidden_out_st    → pcie_tx.hidden_in_st       (1024b)",
    },

    # ── 流控 ──
    "flow_control": {
        "stall_in":  "1b/Stage  下游 sink_ready=0 → 反压传播",
        "stall_out": "1b/Stage  上游 stall → 本拍 waitrequest",
        "bubble_cnt": "8b       流水线气泡计数 (性能监控)",
        "token_id":  "8b       当前 batch 中 token 序号",
    },

    # ── 全局控制 (conduit) ──
    "global_ctrl": {
        "start":          "1b   启动新层推理",
        "done":           "1b   本层 9 拍全部完成",
        "layer_idx":      "8b   当前层 (0..60)",
        "batch_size":     "8b",
        "seq_len":        "16b",
        "is_last_layer":  "1b   本层是最后一层",
        "tp_group_size":  "5b   TP 组大小 (用于 AllReduce 步数计算)",
        "card_id":        "5b   本卡在 TP 组中的 rank",
    },
}


# ═══════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════

ALL_STAGE_INTERFACES = [
    PCIeRxInterface,
    MLAQKInterface,
    MLAAttentionInterface,
    SharedExpertInterface,
    RouterInterface,
    ExpertFetchInterface,
    RoutedExpertInterface,
    AggregateInterface,
    PCIeTxInterface,
    PipelineTopInterface,
]
