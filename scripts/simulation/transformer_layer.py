"""
Single Transformer Layer simulation — DeepSeek V4 Pro MoE layer.

Combines: MLA Attention → RMSNorm → MoE Router → Expert FFN (SwiGLU) + Shared Expert

This is the "golden model" for fp4 precision validation: BF16 reference vs fp4 simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from .mla_attention import MLAAttention
    from .moe_router import MoERouter
    from .fp4_utils import quantize_fp4_e2m1, fp4_gemm_simulate
except ImportError:
    from mla_attention import MLAAttention
    from moe_router import MoERouter
    from fp4_utils import quantize_fp4_e2m1, fp4_gemm_simulate


class ExpertFFN(nn.Module):
    """Single MoE Expert FFN with SwiGLU activation."""

    def __init__(self, hidden_size: int = 7168, intermediate_size: int = 3072):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerLayer(nn.Module):
    """
    One DeepSeek V4 Pro Transformer layer (BF16 reference).

    Flow: Input → MLA Attention → RMSNorm → Shared Expert + MoE Routed Experts → Output
    """

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or {}

        self.hidden_size = cfg.get('hidden_size', 7168)
        self.n_routed_experts = cfg.get('n_routed_experts', 384)
        self.intermediate_size = cfg.get('moe_intermediate_size', 3072)
        self.num_experts_per_tok = cfg.get('num_experts_per_tok', 6)

        self.attention = MLAAttention(cfg)
        self.attn_norm = nn.RMSNorm(self.hidden_size)
        self.ffn_norm = nn.RMSNorm(self.hidden_size)

        self.router = MoERouter(cfg)

        # Shared expert (always activated)
        self.shared_expert = ExpertFFN(self.hidden_size, self.intermediate_size)

        # Routed experts (sparse, Top-K per token)
        self.routed_experts = nn.ModuleList([
            ExpertFFN(self.hidden_size, self.intermediate_size)
            for _ in range(self.n_routed_experts)
        ])

    def forward(self, hidden_states: torch.Tensor, kv_cache: tuple = None,
                return_kv_cache: bool = False):
        """
        Args:
            hidden_states: [B, H] single token decode
            kv_cache: optional KV cache tuple
            return_kv_cache: whether to return updated KV cache

        Returns:
            output: [B, H]
            kv_cache: updated if requested
        """
        residual = hidden_states

        # MLA Attention
        attn_out = self.attention(hidden_states, kv_cache=kv_cache,
                                  return_kv_cache=return_kv_cache)
        if return_kv_cache:
            attn_out, new_kv_cache = attn_out
        hidden_states = self.attn_norm(residual + attn_out)

        # FFN
        residual = hidden_states
        shared_out = self.shared_expert(hidden_states)

        topk_indices, topk_weights = self.router(hidden_states)
        routed_out = torch.zeros_like(hidden_states)
        for b in range(hidden_states.shape[0]):
            for k in range(self.num_experts_per_tok):
                expert_idx = topk_indices[b, k].item()
                expert_out = self.routed_experts[expert_idx](hidden_states[b:b + 1])
                routed_out[b] += expert_out.squeeze(0) * topk_weights[b, k].item()

        hidden_states = self.ffn_norm(residual + shared_out + routed_out)

        if return_kv_cache:
            return hidden_states, new_kv_cache
        return hidden_states


class TransformerLayerFP4(nn.Module):
    """
    Same layer but all linear weights are fp4 quantized.
    Uses fp4_gemm_simulate for each GEMM to emulate FPGA behavior.
    """

    def __init__(self, bf16_layer: TransformerLayer, group_size: int = 128):
        super().__init__()
        self.bf16_layer = bf16_layer
        self.group_size = group_size
        self.hidden_size = bf16_layer.hidden_size
        self.n_routed_experts = bf16_layer.n_routed_experts
        self.intermediate_size = bf16_layer.intermediate_size
        self.num_experts_per_tok = bf16_layer.num_experts_per_tok

        # Keep attention and router in fp8/bf16 (as in the FPGA design)
        self.attention = bf16_layer.attention
        self.router = bf16_layer.router
        self.attn_norm = bf16_layer.attn_norm
        self.ffn_norm = bf16_layer.ffn_norm

        # Quantize all expert weights to fp4
        self._shared_expert_w = self._quantize_expert(bf16_layer.shared_expert)
        self._routed_expert_w = [
            self._quantize_expert(exp) for exp in bf16_layer.routed_experts
        ]

    def _quantize_expert(self, expert: ExpertFFN):
        """Quantize expert's three projection weights to fp4."""
        result = {}
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            w = getattr(expert, name).weight.data
            fp4_idx, fp8_scale = quantize_fp4_e2m1(w.T, self.group_size)
            result[name] = (fp4_idx, fp8_scale)
        return result

    def _fp4_linear(self, x: torch.Tensor, fp4_data: tuple,
                    name: str) -> torch.Tensor:
        """Simulate fp4 linear layer."""
        w_idx, w_scale = fp4_data
        return fp4_gemm_simulate(w_idx, w_scale, x.T, self.group_size).T

    def _fp4_expert_forward(self, x: torch.Tensor, fp4_data: dict) -> torch.Tensor:
        """Forward through an fp4-quantized expert."""
        gate = F.silu(self._fp4_linear(x, fp4_data['gate_proj'], 'gate'))
        up = self._fp4_linear(x, fp4_data['up_proj'], 'up')
        return self._fp4_linear(gate * up, fp4_data['down_proj'], 'down')

    def forward(self, hidden_states: torch.Tensor, kv_cache: tuple = None,
                return_kv_cache: bool = False):
        """Forward pass with fp4 weights, matching the BF16 reference flow."""
        residual = hidden_states

        # Attention (fp8, not fp4)
        attn_out = self.attention(hidden_states, kv_cache=kv_cache,
                                  return_kv_cache=return_kv_cache)
        if return_kv_cache:
            attn_out, new_kv_cache = attn_out
        hidden_states = self.attn_norm(residual + attn_out)

        # FFN with fp4 weights
        residual = hidden_states
        shared_out = self._fp4_expert_forward(hidden_states, self._shared_expert_w)

        topk_indices, topk_weights = self.router(hidden_states)
        routed_out = torch.zeros_like(hidden_states)
        for b in range(hidden_states.shape[0]):
            for k in range(self.num_experts_per_tok):
                expert_idx = topk_indices[b, k].item()
                fp4_data = self._routed_expert_w[expert_idx]
                expert_out = self._fp4_expert_forward(
                    hidden_states[b:b + 1], fp4_data
                )
                routed_out[b] += expert_out.squeeze(0) * topk_weights[b, k].item()

        hidden_states = self.ffn_norm(residual + shared_out + routed_out)

        if return_kv_cache:
            return hidden_states, new_kv_cache
        return hidden_states
