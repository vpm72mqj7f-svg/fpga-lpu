"""
MLA (Multi-head Latent Attention) — NumPy functional simulation.

Key parameters:
  hidden_size=7168, num_attention_heads=128, num_kv_heads=1
  q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64
  head_dim=512 (nope=448 + rope=64)
  v_head_dim=128

KV Cache: (kv_lora_rank + qk_rope_head_dim) × FP8 = 576 bytes/token
Standard MHA: 2 × 128 × 128 × FP16 = 32 KB/token → MLA = 56× compression
"""

import numpy as np
import math


class MLAAttention:
    """NumPy functional reference of DeepSeek MLA."""

    def __init__(self, config=None):
        cfg = config or {}

        self.hidden_size = cfg.get('hidden_size', 7168)
        self.num_heads = cfg.get('num_attention_heads', 128)
        self.kv_lora_rank = cfg.get('kv_lora_rank', 512)
        self.q_lora_rank = cfg.get('q_lora_rank', 1536)
        self.qk_rope_head_dim = cfg.get('qk_rope_head_dim', 64)
        self.qk_nope_head_dim = cfg.get('qk_nope_head_dim', 448)
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 512
        self.v_head_dim = cfg.get('v_head_dim', 128)
        self.o_lora_rank = cfg.get('o_lora_rank', 1024)
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.sliding_window = cfg.get('sliding_window', 128)

        self._init_weights()

    def _init_weights(self):
        rng = np.random.RandomState(42)
        h = self.hidden_size
        kv_r, q_r = self.kv_lora_rank, self.q_lora_rank
        nh = self.num_heads
        nope_dim = self.qk_nope_head_dim
        rope_dim = self.qk_rope_head_dim
        v_dim = self.v_head_dim
        o_r = self.o_lora_rank

        # KV compression
        self.kv_a_down_W = rng.randn(kv_r, h).astype(np.float32) * 0.02  # [512, 7168]
        self.kv_a_up_W = rng.randn(nh * (nope_dim + v_dim), kv_r).astype(np.float32) * 0.02  # [128*576, 512]
        self.kv_a_rope_W = rng.randn(rope_dim, h).astype(np.float32) * 0.02  # [64, 7168]

        # Q compression
        self.q_a_down_W = rng.randn(q_r, h).astype(np.float32) * 0.02  # [1536, 7168]
        self.q_a_up_W = rng.randn(nh * (nope_dim + rope_dim), q_r).astype(np.float32) * 0.02

        # Output
        self.o_down_W = rng.randn(o_r, nh * v_dim).astype(np.float32) * 0.02
        self.o_up_W = rng.randn(h, o_r).astype(np.float32) * 0.02

    def _rms_norm(self, x):
        """Simple RMSNorm."""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
        return x / rms

    def _linear(self, x, W):
        """x: [B, in], W: [out, in] → [B, out]"""
        return x @ W.T

    def _softmax(self, x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    def forward(self, hidden, kv_cache=None, return_kv_cache=False):
        """
        hidden: [B, H] single token decode
        kv_cache: (k_cache [B, S, 128, 512], v_cache [B, S, 128, 128]) or None
        """
        B = hidden.shape[0]
        if hidden.ndim == 3:
            hidden = hidden[:, 0, :]

        h = hidden
        nh = self.num_heads
        nope_dim = self.qk_nope_head_dim
        rope_dim = self.qk_rope_head_dim
        v_dim = self.v_head_dim

        # ── KV compression ──
        kv_normed = self._rms_norm(h)
        kv_c = self._linear(kv_normed, self.kv_a_down_W)         # [B, 512]
        kv_up = self._linear(kv_c, self.kv_a_up_W)               # [B, 128*(448+128)]

        k_nope = kv_up[:, :nh * nope_dim].reshape(B, nh, nope_dim)
        v = kv_up[:, nh * nope_dim:].reshape(B, nh, v_dim)

        k_rope = self._linear(kv_normed, self.kv_a_rope_W)       # [B, 64]
        k_rope = k_rope[:, None, :].repeat(nh, axis=1)           # [B, 128, 64]

        k_full = np.concatenate([k_nope, k_rope], axis=-1)       # [B, 128, 512]

        # ── Q compression ──
        q_normed = self._rms_norm(h)
        q_c = self._linear(q_normed, self.q_a_down_W)            # [B, 1536]
        q_up = self._linear(q_c, self.q_a_up_W)                  # [B, 128*(448+64)]
        q_full = q_up.reshape(B, nh, nope_dim + rope_dim)        # [B, 128, 512]

        q_full = self._rms_norm(q_full)
        k_full = self._rms_norm(k_full)

        # ── KV Cache management ──
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_cache = np.concatenate([k_cache, k_full[:, None, :, :]], axis=1)  # [B, S+1, 128, 512]
            v_cache = np.concatenate([v_cache, v[:, None, :, :]], axis=1)       # [B, S+1, 128, 128]
            if self.sliding_window > 0 and k_cache.shape[1] > self.sliding_window:
                k_cache = k_cache[:, -self.sliding_window:, :, :]
                v_cache = v_cache[:, -self.sliding_window:, :, :]
        else:
            k_cache = k_full[:, None, :, :]
            v_cache = v[:, None, :, :]

        # ── Attention ──
        # Q: [B, 128, 512], K: [B, S, 128, 512]
        attn_scores = np.einsum('bhd,bshd->bhs', q_full, k_cache) * self.scaling
        attn_probs = self._softmax(attn_scores, axis=-1)

        # Output: [B, 128, 128]
        attn_out = np.einsum('bhs,bshv->bhv', attn_probs, v_cache)
        attn_out = attn_out.reshape(B, -1)

        # ── Output projection ──
        o_c = self._linear(attn_out, self.o_down_W)
        o_c = self._rms_norm(o_c)
        output = self._linear(o_c, self.o_up_W)

        if return_kv_cache:
            return output, (k_cache, v_cache)
        return output


def compute_kv_cache_comparison(num_tokens, kv_rank=512, rope_dim=64,
                                num_heads=128, head_dim=128):
    """Compare MLA vs MHA KV Cache size."""
    mla_bytes = num_tokens * (kv_rank + rope_dim)  # FP8
    mha_bytes = num_tokens * 2 * num_heads * head_dim * 2  # FP16
    return {
        'num_tokens': num_tokens,
        'mla_mb': mla_bytes / (1024 * 1024),
        'mha_mb': mha_bytes / (1024 * 1024),
        'compression_ratio': mha_bytes / mla_bytes,
    }
