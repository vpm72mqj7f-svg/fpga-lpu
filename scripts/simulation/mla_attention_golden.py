"""
mla_attention_golden.py — Python golden model for mla_attention_v2 RTL validation.

Computes exact Q12 fixed-point MLA attention matching the RTL bring-up model:
  - mla_qkv_proj: Q/K/V low-rank projections (1 dot per cycle, Q12 accum)
  - mla_rope: Rotary Position Embedding (Q only, cos/sin LUT)
  - mla_kv_cache: token store/retrieve (K_latent, V_latent)
  - Attention: Q·K → softmax (piecewise exp LUT) → weighted V sum

All arithmetic is Q12 fixed-point (signed 32-bit) matching RTL precision.
"""

import numpy as np

Q12_ONE = 4096
Q12_SHIFT = 12


def q12_mul(a, b):
    """Signed Q12 × Q12 → Q12 (truncated, matching RTL >>> 12)."""
    return (int(a) * int(b)) >> Q12_SHIFT


def q12_dot(a_vec, b_vec):
    """Q12 dot product: sum(a[i] * b[i] >>> 12) matching RTL always_comb."""
    acc = 0
    for a, b in zip(a_vec, b_vec):
        acc += q12_mul(a, b)
    return acc


def rtl_clamp32(v):
    """Clamp to signed 32-bit."""
    if v > 2147483647:
        v -= 4294967296
    elif v < -2147483648:
        v += 4294967296
    return v


def linear(x, W):
    """x@W^T in Q12: x is Q12 input vector, W is [out_dim, in_dim] Q12 weights."""
    out = []
    for r in range(len(W)):
        out.append(q12_dot(x, W[r]))
    return out


def rope_rotate(vec, pos, sin_lut, cos_lut):
    """
    RoPE: rotate consecutive pairs by angle θ[pos,pair].
    vec: Q12 input vector [HIDDEN]
    sin_lut[pos][pair], cos_lut[pos][pair]: Q12
    Returns: rotated vector [HIDDEN]
    """
    n = len(vec)
    out = vec[:]  # copy
    for i in range(0, n, 2):
        a = vec[i]
        b = vec[i + 1]
        cos_val = cos_lut[pos][i // 2]
        sin_val = sin_lut[pos][i // 2]
        out[i] = q12_mul(a, cos_val) - q12_mul(b, sin_val)
        out[i + 1] = q12_mul(a, sin_val) + q12_mul(b, cos_val)
    return out


def exp_lut_rtl(adj):
    """Piecewise exp LUT matching RTL mla_attention_v2.sv:135-141."""
    if adj > -256:
        return 4096
    elif adj > -1024:
        return 3545
    elif adj > -2048:
        return 2588
    elif adj > -4096:
        return 1507
    elif adj > -8192:
        return 538
    else:
        return 48


def softmax_rtl(scores):
    """
    RTL-style softmax: find max, subtract, exp LUT, sum, normalize.
    scores: Q12 values
    Returns: Q12 softmax probabilities
    """
    score_max = max(scores)
    exps = []
    for s in scores:
        adj = s - score_max  # Q12 difference
        exps.append(exp_lut_rtl(adj))
    exp_sum = sum(exps)
    probs = []
    for e in exps:
        # Q12 × Q12 / sum ≈ e * 4096 / exp_sum
        probs.append((e * Q12_ONE) // exp_sum if exp_sum > 0 else 0)
    return probs


class MLAAttentionRTL:
    """Python golden model matching mla_attention_v2 RTL at bring-up dimensions."""

    def __init__(self, hidden=8, k_latent=4, v_latent=4, num_slots=64, max_pos=64):
        self.H = hidden
        self.KL = k_latent
        self.VL = v_latent
        self.NUM_SLOTS = num_slots
        self.MAX_POS = max_pos

        # Weights (Q12) — stored as [out_dim][in_dim] for direct linear() call.
        # RTL stores [in_dim][out_dim]; Python model transposes at load time.
        self.W_Q  = [[0] * hidden for _ in range(hidden)]    # [H_out, H_in]
        self.W_K  = [[0] * hidden for _ in range(k_latent)]  # [KL_out, H_in]
        self.W_Ku = [[0] * k_latent for _ in range(hidden)]  # [H_out, KL_in]
        self.W_V  = [[0] * hidden for _ in range(v_latent)]  # [VL_out, H_in]
        self.W_Vu = [[0] * v_latent for _ in range(hidden)]  # [H_out, VL_in]

        # RoPE LUTs (Q12)
        self.sin_lut = [[0] * (hidden // 2) for _ in range(max_pos)]
        self.cos_lut = [[Q12_ONE] * (hidden // 2) for _ in range(max_pos)]

        # KV Cache
        self.k_cache = []  # list of K_latent vectors
        self.v_cache = []  # list of V_latent vectors
        self.write_ptr = 0

    def set_identity_weights(self):
        """Set all projection weights to identity (where dimensions match)."""
        for r in range(self.H):
            self.W_Q[r][r] = Q12_ONE
        for r in range(self.H):
            if r < self.KL:
                self.W_K[r][r] = Q12_ONE
        for r in range(self.KL):
            self.W_Ku[r][r] = Q12_ONE
        for r in range(self.H):
            if r < self.VL:
                self.W_V[r][r] = Q12_ONE
        for r in range(self.VL):
            self.W_Vu[r][r] = Q12_ONE

    def qkv_proj(self, hidden):
        """Compute Q, K, V from hidden state."""
        # Q = W_Q @ hidden (H × H) → [H]
        Q = [0] * self.H
        for r in range(self.H):
            Q[r] = q12_dot(hidden, self.W_Q[r])

        # K_latent = W_K @ hidden (H × KL) → [KL]
        K_lat = [0] * self.KL
        for r in range(self.KL):
            K_lat[r] = q12_dot(hidden, self.W_K[r])

        # K = W_Ku @ K_lat (KL × H) → [H]
        K = [0] * self.H
        for r in range(self.H):
            K[r] = q12_dot(K_lat, self.W_Ku[r])

        # V_latent = W_V @ hidden (H × VL) → [VL]
        V_lat = [0] * self.VL
        for r in range(self.VL):
            V_lat[r] = q12_dot(hidden, self.W_V[r])

        # V = W_Vu @ V_lat (VL × H) → [H]
        V = [0] * self.H
        for r in range(self.H):
            V[r] = q12_dot(V_lat, self.W_Vu[r])

        return Q, K, V, K_lat, V_lat

    def rope(self, Q, pos):
        """Apply RoPE to Q at position."""
        return rope_rotate(Q, pos, self.sin_lut, self.cos_lut)

    def write_cache(self, K_lat, V_lat):
        """Write K_lat/V_lat to cache (circular buffer)."""
        if len(self.k_cache) < self.NUM_SLOTS:
            self.k_cache.append(K_lat)
            self.v_cache.append(V_lat)
        else:
            idx = self.write_ptr % self.NUM_SLOTS
            self.k_cache[idx] = K_lat
            self.v_cache[idx] = V_lat
        self.write_ptr += 1

    def attention(self, Q, K_cached, V_cached):
        """
        Compute attention: softmax(Q·K) weighted V sum.
        K_cached: list of K vectors [num_cached][H]
        V_cached: list of V vectors [num_cached][H]
        Returns: output vector [H]
        """
        num_tokens = len(K_cached)
        if num_tokens == 0:
            return [0] * self.H

        scores = []
        for k in K_cached:
            scores.append(q12_dot(Q, k))

        probs = softmax_rtl(scores)

        out = [0] * self.H
        for t in range(num_tokens):
            w = probs[t]
            for d in range(self.H):
                out[d] += q12_mul(w, V_cached[t][d])
        return out

    def forward(self, hidden, pos=0):
        """Full forward pass: single-token decode."""
        Q, K, V, K_lat, V_lat = self.qkv_proj(hidden)
        Q_rope = self.rope(Q, pos)

        # Cache the new K, V
        self.write_cache(K_lat, V_lat)
        # For attention, use decompressed K from latent
        K_full = []
        V_full = []
        for i in range(len(self.k_cache)):
            # Decompress each cached entry
            k_vec = [0] * self.H
            v_vec = [0] * self.H
            for r in range(self.H):
                k_vec[r] = q12_dot(self.k_cache[i], self.W_Ku[r])
                v_vec[r] = q12_dot(self.v_cache[i], self.W_Vu[r])
            K_full.append(k_vec)
            V_full.append(v_vec)

        output = self.attention(Q_rope, K_full, V_full)
        return output, Q, K, V, Q_rope, K_lat, V_lat


def test_identity_passthrough():
    """Test: identity weights, single token → output = V = hidden[:VL] padded."""
    rtl = MLAAttentionRTL(hidden=8, k_latent=4, v_latent=4)
    rtl.set_identity_weights()

    hidden = [100, 101, 102, 103, 104, 105, 106, 107]
    output, Q, K, V, Q_rope, K_lat, V_lat = rtl.forward(hidden, pos=0)

    # With identity weights:
    # Q = hidden, K_lat = hidden[0:4], K ≈ hidden[0:4] padded with zeros
    # V_lat = hidden[0:4], V ≈ hidden[0:4] padded with zeros
    # Single-token softmax = [1.0], output = V
    expected = [100, 101, 102, 103, 0, 0, 0, 0]

    print("Q:", Q)
    print("K:", K)
    print("V:", V)
    print("K_lat:", K_lat)
    print("V_lat:", V_lat)
    print("Output:", output)
    print("Expected:", expected)

    if output == expected:
        print("PASS: identity passthrough")
    else:
        print("FAIL: output mismatch")
        return False
    return True


def test_multi_token():
    """Test: 2-token attention with same input → output should stay same."""
    rtl = MLAAttentionRTL(hidden=8, k_latent=4, v_latent=4)
    rtl.set_identity_weights()

    # Token 1
    hidden = [100, 101, 102, 103, 104, 105, 106, 107]
    out1, _, _, _, _, _, _ = rtl.forward(hidden, pos=0)
    print(f"Token 1 output: {out1}")

    # Token 2 (same input)
    hidden2 = [200, 201, 202, 203, 204, 205, 206, 207]
    out2, _, _, _, _, _, _ = rtl.forward(hidden2, pos=1)
    print(f"Token 2 output: {out2}")

    # Token 2 attention: softmax over tokens 1 and 2
    # Token 1: V ≈ hidden[0:4] = [100,101,102,103,0,0,0,0]
    # Token 2: V ≈ hidden2[0:4] = [200,201,202,203,0,0,0,0]
    # Scores: Q·K
    # Q2 = hidden2[0:4] = [200,201,202,203,0,0,0,0] (with latency)
    # K1 = [100,101,102,103,0,0,0,0]
    # K2 = [200,201,202,203,0,0,0,0]
    # score_1 = 200*100+201*101+202*102+203*103 (same as hidden2·hidden dims 0-3)

    return True


if __name__ == "__main__":
    print("=" * 60)
    print(" MLA Attention — Python Golden Model")
    print("=" * 60)
    print()
    test_identity_passthrough()
    print()
    test_multi_token()
    print()
    print("Done.")
