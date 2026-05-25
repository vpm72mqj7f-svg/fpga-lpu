"""
4-FPGA 单卡架构 — 一张 PCIe 卡级联 4 片 Agilex 7.

两种配置模式:
  A) TP 组模式: expert 分片 (每片 96), AllReduce via SerDes
  B) 独立推理模式: expert 全量复制 (每片 384), 零推理时通信

SerDes: 片内 PCB 直连, 8 lane × 56 Gbps = 56 GB/s, ~0.01 μs/hop
互联拓扑: mesh 全互联 (每片 3 条 SerDes 对)

物理约束:
  - 功耗: 4 × 50W ≈ 200W, 需 6-pin + 8-pin 辅助供电
  - 散热: 主动风扇 + 大面积散热片, 双槽宽度
  - PCIe: Gen5 x16 → PCIe Switch → 4 × Gen5 x4
  - FPGA: Agilex 7 AGFB027 (9375 DSP, 32GB HBM, 43MB SRAM)
"""

from dataclasses import dataclass, field
from .. import config
from .pipeline import Pipeline
from .phys.ethernet import EthernetMAC


@dataclass
class CardTopConfig:
    """单卡 4-FPGA 配置."""
    num_fpgas: int = 4
    fpga_model: str = "AGFB027"
    experts_per_fpga: int = 384 // 4      # 方案A: 96; 方案B: 384
    hbm_per_fpga_gb: int = 32
    sram_per_fpga_mb: int = 43
    dsp_tops_per_fpga: float = 8.44

    # 互联
    serdes_lanes: int = 8
    serdes_per_lane_gbps: float = 56.0
    serdes_hop_us: float = 0.01       # 片内 PCB 走线 ~10ps

    # 功耗 / 散热
    tdp_per_fpga_w: float = 50.0
    card_total_tdp_w: float = 220.0   # 4×50 + 20 aux

    # PCIe
    host_pcie: str = "Gen5 x16"       # Host ←→ PCIe Switch
    fpga_pcie: str = "Gen5 x4"        # Switch → 每片 FPGA


class FourFPGACard:
    """单卡 4 FPGA — 独立推理引擎 (方案 B).

    vLLM host 通过 PCIe Switch 下发请求到 4 片 FPGA.
    每片 FPGA 独立完成全模型推理 (61 层, 384 experts 全在本地 HBM).
    推理时 FPGA 间零通信, SerDes 仅启动时权重同步.

    RTL 结构:
      Host (vLLM)
        ↕ PCIe Gen5 x16
      [PCIe Switch]  (芯片, 不是 FPGA 逻辑)
        ↕ 4 × PCIe Gen5 x4
      ┌──────────────────────────────────────┐
      │  FPGA0           ...        FPGA3    │
      │  ┌─────────┐            ┌─────────┐  │
      │  │Pipeline │            │Pipeline │  │
      │  │ 9-stage │            │ 9-stage │  │
      │  │SRAM|HBM │←→SerDes→←→│SRAM|HBM │  │
      │  │(384exp) │   mesh    │(384exp) │  │
      │  └─────────┘            └─────────┘  │
      └──────────────────────────────────────┘
    """

    def __init__(self, mode: str = "independent",
                 loaded_experts_per_fpga: int = None):
        """
        Args:
            mode: 'independent' (方案B) or 'tp_group' (方案A)
            loaded_experts_per_fpga: 每个 FPGA 加载的 expert 数
        """
        self.mode = mode
        self.num_fpgas = 4
        self.cfg = CardTopConfig()

        if loaded_experts_per_fpga is None:
            if mode == "independent":
                loaded_experts_per_fpga = config.MODEL_NUM_EXPERTS  # 384
            else:
                loaded_experts_per_fpga = config.MODEL_NUM_EXPERTS // self.num_fpgas  # 96

        self.experts_per_fpga = loaded_experts_per_fpga
        self.total_experts_coverage = (
            loaded_experts_per_fpga if mode == "independent"
            else loaded_experts_per_fpga * self.num_fpgas
        )

        # SerDes 互联带宽
        self.serdes_bw_gbps = self.cfg.serdes_lanes * self.cfg.serdes_per_lane_gbps
        self.serdes_bw_gbs = self.serdes_bw_gbps / 8  # 56 GB/s

    def capacity_check(self) -> dict:
        """检查单卡硬件容量."""
        hbm_mb = self.cfg.hbm_per_fpga_gb * 1024

        # 确定性权重 (每片 FPGA SRAM)
        sram_used = config.WEIGHT_DETERMINISTIC_MB
        sram_ok = sram_used <= self.cfg.sram_per_fpga_mb

        # Expert 权重 (HBM)
        hbm_used = self.experts_per_fpga * config.WEIGHT_EXPERT_MB
        hbm_free = hbm_mb - hbm_used
        hbm_ok = hbm_used <= hbm_mb

        # KV cache in remaining HBM (40% of free)
        kv_cache_hbm_mb = hbm_free * 0.4
        kv_tokens = kv_cache_hbm_mb * 1e6 / config.KV_BYTES_PER_TOKEN

        return {
            'mode': self.mode,
            'num_fpgas': self.num_fpgas,
            'experts_per_fpga': self.experts_per_fpga,
            'total_expert_coverage': self.total_experts_coverage,
            # SRAM
            'sram_mb': self.cfg.sram_per_fpga_mb,
            'sram_used_mb': sram_used,
            'sram_free_mb': self.cfg.sram_per_fpga_mb - sram_used,
            'sram_ok': sram_ok,
            # HBM
            'hbm_mb': hbm_mb,
            'hbm_used_mb': hbm_used,
            'hbm_free_mb': hbm_free,
            'hbm_pct': hbm_used / hbm_mb * 100,
            'hbm_ok': hbm_ok,
            # KV cache
            'kv_cache_mb': kv_cache_hbm_mb,
            'kv_tokens_m': kv_tokens / 1e6,
            # SerDes
            'serdes_bw_gbps': self.serdes_bw_gbps,
            'serdes_bw_gbs': self.serdes_bw_gbs,
            'serdes_hop_us': self.cfg.serdes_hop_us,
            # Power
            'tdp_per_fpga_w': self.cfg.tdp_per_fpga_w,
            'card_total_tdp_w': self.cfg.card_total_tdp_w,
        }

    def expert_hit_model(self, batch_size: int = 1) -> dict:
        """Expert 命中模型.

        方案B (independent): 100% 命中, 但每 token 需读 K=6 experts from HBM.
        batch_size 越大, HBM 读取摊销越好.
        """
        if self.mode == "independent":
            # 所有 expert 在 HBM 中, 每次选 6 个
            n_experts_per_layer = config.MODEL_TOP_K
            hbm_read_per_expert_us = config.WEIGHT_EXPERT_MB * 1e6 / (
                config.HW_FPGA_HBM_BW_GBPS * config.HW_FPGA_HBM_EFF * 1e3)

            # HBM 读可以流水: 6 experts 数据总量 / 带宽
            total_expert_mb = n_experts_per_layer * config.WEIGHT_EXPERT_MB
            hbm_time_us = total_expert_mb * 1e6 / (
                config.HW_FPGA_HBM_BW_GBPS * config.HW_FPGA_HBM_EFF * 1e3)

            # 批量摊销: HBM 时间不随 batch 增长
            hbm_per_token = hbm_time_us / batch_size

            return {
                'mode': 'independent (all-local)',
                'hit_rate': 1.0,
                'n_experts_per_layer': n_experts_per_layer,
                'hbm_read_total_us': hbm_time_us,
                'hbm_read_per_token_us': hbm_per_token,
                'batch_amortize_factor': batch_size,
            }
        else:
            # 方案A: TP 分片, 每次只有 1/4 expert 在本地
            p_local = self.experts_per_fpga / config.MODEL_NUM_EXPERTS
            p0 = (1 - p_local) ** config.MODEL_TOP_K
            p1 = config.MODEL_TOP_K * p_local * (1 - p_local) ** (config.MODEL_TOP_K - 1)
            p2 = 1 - p0 - p1

            hbm_per = config.WEIGHT_EXPERT_MB * 1e6 / (
                config.HW_FPGA_HBM_BW_GBPS * config.HW_FPGA_HBM_EFF * 1e3)

            return {
                'mode': 'tp_group (sharded)',
                'p_local': p_local,
                'p_0_hit': p0,
                'p_1_hit': p1,
                'p_2_plus': p2,
                'hbm_read_per_expert_us': hbm_per,
                'weighted_hbm_us': (p1 * hbm_per + p2 * hbm_per * 2) / 1,
            }

    def print_capacity(self):
        """打印单卡容量报告."""
        c = self.capacity_check()
        h = self.expert_hit_model()

        print(f"  4-FPGA 单卡架构 — {self.mode}")
        print(f"  模式: {'独立推理 (方案B)' if self.mode == 'independent' else 'TP组 (方案A)'}")
        print()
        print(f"  硬件资源 (每片):")
        print(f"    DSP:     {c['hbm_mb']/1024:.0f} GB HBM, {self.cfg.sram_per_fpga_mb} MB SRAM")
        print(f"    SRAM:    使用 {c['sram_used_mb']:.0f} MB (确定性权重), 余 {c['sram_free_mb']:.0f} MB")
        print(f"    HBM:     使用 {c['hbm_used_mb']:.0f} MB expert ({c['hbm_pct']:.0f}%), "
              f"余 {c['hbm_free_mb']:.0f} MB")
        print(f"    KV cache: {c['kv_cache_mb']:.0f} MB ({c['kv_tokens_m']:.1f}M tokens)")
        print()
        print(f"  互联:")
        print(f"    SerDes: {c['serdes_bw_gbps']:.0f} Gbps ({c['serdes_bw_gbs']:.0f} GB/s), "
              f"{c['serdes_hop_us']:.3f} μs/hop")
        print(f"    PCIe:   Host x16 → Switch → 4 × x4")
        print()
        print(f"  功耗: {c['card_total_tdp_w']}W (需辅助供电)")
        print()
        if self.mode == "independent":
            print(f"  Expert 命中:")
            print(f"    本地命中率:    100% ({self.experts_per_fpga} experts 全量)")
            print(f"    每层 HBM 读:   {h['n_experts_per_layer']} experts × "
                  f"{h['hbm_read_total_us']:.0f} μs = {h['hbm_read_total_us']:.0f} μs")
            print(f"    batch=1 摊销:  {h['hbm_read_per_token_us']:.1f} μs/token")
            print(f"    batch=16 摊销: {h['hbm_read_total_us']/16:.1f} μs/token")
            print(f"    推理时 SerDes: 零流量")
        else:
            print(f"  Expert 命中 (本地):")
            print(f"    P(0-hit)={h['p_0_hit']:.3f}  P(1-hit)={h['p_1_hit']:.3f}  "
                  f"P(2+)={h['p_2_plus']:.3f}")
            print(f"    AllReduce: 每层 ~0.06 μs (可忽略)")
        print()
