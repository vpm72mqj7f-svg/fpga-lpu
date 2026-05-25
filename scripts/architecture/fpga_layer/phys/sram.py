"""
片上 SRAM — 43 MB M20K + MLAB, 永久驻留确定性权重.

驻留内容 (~39.6 MB, fp4):
  - MLA Attention weights (Q/K/V/O 投影 + RoPE): 6.2 MB
  - Shared Expert weights (gate/up/down):         33.0 MB
  - Router weights:                                0.37 MB
  剩余 ~3.4 MB 可用于临时 buffer (activation staging).

访问特性:
  - 片上 SRAM, 与 DSP 同频, 读延迟 ~0 (pipeline 可忽略)
  - 双缓冲, DSP 算当前层时预读下一层权重
  - 权重按 tile 组织, 每个 tile 匹配 DSP 阵列宽度
"""

from ... import config


class SRAMBank:
    """片上 SRAM — 确定性权重常驻, 零延迟访问."""

    def __init__(self):
        self.total_mb = config.HW_FPGA_SRAM_SIZE_MB
        self.used_mb = config.WEIGHT_DETERMINISTIC_MB
        self.free_mb = self.total_mb - self.used_mb

        # 常驻权重清单
        self.resident_weights = {
            'mla_q_down':     config.MODEL_Q_LORA_RANK * config.MODEL_HIDDEN_SIZE / 1e6 * 0.5,  # fp4=0.5B/el
            'mla_kv_down':    config.MODEL_KV_LORA_RANK * config.MODEL_HIDDEN_SIZE / 1e6 * 0.5,
            'mla_k_up':       config.MODEL_KV_LORA_RANK * config.MODEL_HIDDEN_SIZE / 1e6 * 0.5,
            'mla_v_up':       config.MODEL_KV_LORA_RANK * config.MODEL_HIDDEN_SIZE / 1e6 * 0.5,
            'mla_q_up':       config.MODEL_Q_LORA_RANK * config.MODEL_HIDDEN_SIZE / 1e6 * 0.5,
            'mla_o_proj':     config.MODEL_HIDDEN_SIZE * config.MODEL_HIDDEN_SIZE / 1e6 * 0.5,
            'shared_gate':    config.MODEL_HIDDEN_SIZE * config.MODEL_INTERMEDIATE_SIZE / 1e6 * 0.5,
            'shared_up':      config.MODEL_HIDDEN_SIZE * config.MODEL_INTERMEDIATE_SIZE / 1e6 * 0.5,
            'shared_down':    config.MODEL_INTERMEDIATE_SIZE * config.MODEL_HIDDEN_SIZE / 1e6 * 0.5,
            'router':         config.MODEL_HIDDEN_SIZE * config.MODEL_NUM_EXPERTS / 1e6 * 0.5,
        }
        self._loaded = True  # 假设启动时已加载

    def read_latency(self, weight_name: str = "") -> float:
        """SRAM 读延迟 ~0 (片上, 可忽略)."""
        return 0.0

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def stats(self) -> dict:
        return {
            'total_mb': self.total_mb,
            'used_mb': self.used_mb,
            'free_mb': self.free_mb,
            'num_resident_weights': len(self.resident_weights),
        }

    def reset(self):
        pass
