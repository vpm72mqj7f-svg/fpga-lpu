"""
MoE Router simulation for DeepSeek V4 Pro (NumPy).

Config: 384 routed experts + 1 shared expert, Top-6 routing per token.
"""

import numpy as np
import math


# Default config from DeepSeek V4 Pro config.json
DEFAULT_CFG = {
    'hidden_size': 7168,
    'n_routed_experts': 384,
    'n_shared_experts': 1,
    'num_experts_per_tok': 6,
    'scoring_func': 'sqrtsoftplus',
    'routed_scaling_factor': 2.5,
}


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sqrt_softplus(x):
    return np.sqrt(softplus(x))


class MoERouter:
    """NumPy MoE Router simulating DeepSeek V4 Pro routing logic."""

    def __init__(self, config=None):
        cfg = config or DEFAULT_CFG
        self.hidden_size = cfg['hidden_size']
        self.n_experts = cfg['n_routed_experts']
        self.top_k = cfg['num_experts_per_tok']
        self.scoring_func = cfg['scoring_func']
        self.scale = cfg['routed_scaling_factor']

        # Router weight matrix [n_experts, hidden_size]
        rng = np.random.RandomState(42)
        self.router_weight = rng.randn(self.n_experts, self.hidden_size).astype(np.float32) * 0.02

    def forward(self, hidden, return_probs=False):
        """hidden: [B, H] → expert_indices [B, top_k], expert_weights [B, top_k]"""
        if hidden.ndim == 3:
            hidden = hidden.squeeze(1)

        logits = hidden @ self.router_weight.T  # [B, 384]

        if self.scoring_func == 'sqrtsoftplus':
            scores = sqrt_softplus(logits)
        elif self.scoring_func == 'sigmoid':
            scores = 1.0 / (1.0 + np.exp(-logits))
        else:
            scores = np.exp(logits - logits.max(axis=-1, keepdims=True))
            scores /= scores.sum(axis=-1, keepdims=True)

        # Top-K
        topk_indices = np.argpartition(-scores, self.top_k, axis=-1)[:, :self.top_k]
        # Sort top-k by score descending
        batch_idx = np.arange(hidden.shape[0])[:, None]
        topk_scores = scores[batch_idx, topk_indices]
        sort_order = np.argsort(-topk_scores, axis=-1)
        topk_indices = topk_indices[batch_idx, sort_order]
        topk_weights = scores[batch_idx, topk_indices]

        # Normalize
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
        topk_weights = topk_weights * self.scale

        if return_probs:
            return topk_indices, topk_weights, scores
        return topk_indices, topk_weights


def analyze_expert_distribution(router, num_tokens=5000):
    """
    Analyze expert activation distribution.
    Returns power-law concentration and per-card hit probabilities.
    """
    rng = np.random.RandomState(123)
    hidden = rng.randn(num_tokens, router.hidden_size).astype(np.float32)

    indices, weights, probs = router.forward(hidden, return_probs=True)

    # Count expert selections
    expert_counts = np.zeros(router.n_experts, dtype=np.int64)
    for i in range(num_tokens):
        for j in range(router.top_k):
            expert_counts[indices[i, j]] += 1

    sorted_counts = np.sort(expert_counts)[::-1]
    total = expert_counts.sum()

    # Top-20% concentration
    top_20pct = int(router.n_experts * 0.2)
    concentration = sorted_counts[:top_20pct].sum() / total

    # Per-card hit simulation (30 cards, ~13 experts each)
    experts_per_card = router.n_experts / 30
    selection_probs = expert_counts.astype(np.float64) / total

    # One card: random 13 experts
    rng2 = np.random.RandomState(456)
    card_experts = rng2.choice(router.n_experts, size=int(experts_per_card), replace=False)
    card_prob = selection_probs[card_experts].sum()

    # Binomial hit probabilities
    top_k = router.top_k
    p0 = (1 - card_prob) ** top_k
    p1 = top_k * card_prob * (1 - card_prob) ** (top_k - 1)
    p2_plus = 1 - p0 - p1

    return {
        'total_tokens': num_tokens,
        'top_20pct_concentration': concentration,
        'per_card_hit_prob': card_prob,
        'p_0_hit': p0,
        'p_1_hit': p1,
        'p_2_plus_hit': p2_plus,
        'expert_counts': sorted_counts,
    }
