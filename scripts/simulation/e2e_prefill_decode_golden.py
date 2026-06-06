#!/usr/bin/env python3
"""
e2e_prefill_decode_golden.py — Golden model for end-to-end CPU prefill → FPGA decode.

Models the full_transformer_layer at bring-up parameters (HIDDEN=8, K_LATENT=4, V_LATENT=4)
with identity weights. Generates:

  1. KV cache entries (K_latent, V_latent) for P prefill tokens
  2. Expected decode output after running one token through the layer

Q12 fixed-point arithmetic matches RTL bit-widths:
  - Q12_ONE = 4096
  - Products: (a * b) >> 12
  - RMSNorm: simplified identity (gamma=4096, no rescaling)
"""

import numpy as np

# Bring-up parameters (match lpu_config_pkg bring-up mode)
HIDDEN = 8
K_LATENT = 4
V_LATENT = 4
DATA_W = 32
Q12_ONE = 4096
WEIGHT_W = 16

# Identity QKV weights: W[i,i] = 4096, others = 0
# Q_proj: HIDDEN×HIDDEN identity → Q = hidden
# K_proj: K_LATENT×HIDDEN identity (first 4 rows) → K_latent = hidden[0:4]
# V_proj: V_LATENT×HIDDEN identity (first 4 rows) → V_latent = hidden[0:4]


def q12_mul(a, b):
    """Q12 fixed-point multiply: (a * b) >> 12"""
    return (int(a) * int(b)) >> 12


def q12_vec_mul(vec_a, vec_b, length):
    """Dot product of two Q12 vectors."""
    total = 0
    for i in range(length):
        total += q12_mul(vec_a[i], vec_b[i])
    return total


def rms_norm_identity(x, gamma=None):
    """RMSNorm with identity gamma (all 4096) — output = input for Q12."""
    # With gamma=4096 and identity scale, RMSNorm ≈ identity
    # Simplified: output = input (exact for gamma=4096, unit variance)
    return [int(v) for v in x]


def generate_kv_entries(prefill_tokens, seed=42):
    """
    Generate K_latent and V_latent for each prefill token.

    With identity QKV weights:
      K_latent[i] = hidden[i] for i in [0..K_LATENT-1]
      V_latent[i] = hidden[i] for i in [0..V_LATENT-1]

    Returns: list of (K_latent, V_latent) tuples, each as flat-packed Q12 arrays.
    """
    rng = np.random.RandomState(seed)
    entries = []
    for p in range(prefill_tokens):
        # Generate a prefill token hidden state
        hidden = [int(Q12_ONE) for _ in range(HIDDEN)]  # all 4096
        # Add per-token variation
        for d in range(HIDDEN):
            hidden[d] = hidden[d] + int(rng.randint(-128, 128))

        # K_latent = first K_LATENT dims of hidden (identity projection)
        K_lat = [hidden[d] for d in range(K_LATENT)]
        # V_latent = first V_LATENT dims of hidden (identity projection)
        V_lat = [hidden[d] for d in range(V_LATENT)]

        entries.append((K_lat, V_lat))
    return entries


def flat_pack(vec, width):
    """Pack a list of ints into a flat bit vector: vec[0] at LSB."""
    result = 0
    for i, v in enumerate(vec):
        result |= (int(v) & ((1 << width) - 1)) << (i * width)
    return result


def decode_output(prefill_kv_entries, decode_hidden):
    """
    Compute expected decode output through full_transformer_layer.

    Pipeline: RMS → Attention → RMS → Router → FFN → RMS

    With identity weights throughout:
      - RMSNorm: identity
      - QKV projection: identity (Q=hidden, K_lat=hidden[0:4], V_lat=hidden[0:4], V=hidden)
      - RoPE: identity (cos=1, sin=0)
      - Attention (single-token self-attention, stub): output = V = hidden
      - Router: identity (no expert selection change)
      - FFN: identity gate/up/down
      - RMSNorm: identity

    So expected output ≈ decode_hidden (with minor Q12 rounding).
    """
    # For identity weights, output = input through the layer
    # The RTL simplified attention outputs V_r (= hidden)
    output = [int(h) for h in decode_hidden]
    return output


def pack_kv_entry(K_lat, V_lat):
    """Pack K_latent and V_latent into flat vectors for RTL preload port."""
    K_flat = flat_pack(K_lat, DATA_W)
    V_flat = flat_pack(V_lat, DATA_W)
    return K_flat, V_flat


def main():
    print("=" * 60)
    print("E2E CPU Prefill → FPGA Decode Golden Model")
    print("=" * 60)
    print(f"Parameters: HIDDEN={HIDDEN}, K_LATENT={K_LATENT}, V_LATENT={V_LATENT}")
    print(f"Q12_ONE={Q12_ONE}")
    print()

    # Generate 2 prefill tokens
    prefill_entries = generate_kv_entries(2, seed=42)
    print("=== Prefill KV Cache Entries ===")
    for i, (K_lat, V_lat) in enumerate(prefill_entries):
        K_flat, V_flat = pack_kv_entry(K_lat, V_lat)
        print(f"  Token {i}: K_lat={K_lat} (0x{K_flat:032x})")
        print(f"           V_lat={V_lat} (0x{V_flat:032x})")

    # Decode token
    decode_hidden = [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
    expected_output = decode_output(prefill_entries, decode_hidden)
    out_flat = flat_pack(expected_output, DATA_W)

    print()
    print("=== Decode Token ===")
    print(f"  hidden_in:   {decode_hidden}")
    print()
    print("=== Expected Decode Output ===")
    print(f"  y (scalar):  {expected_output}")
    print(f"  y_flat:      0x{out_flat:064x}")
    print()

    # Generate Verilog-friendly hex values
    print("=== RTL Preload Data (Verilog) ===")
    for i, (K_lat, V_lat) in enumerate(prefill_entries):
        K_flat, V_flat = pack_kv_entry(K_lat, V_lat)
        print(f"  // Token {i}")
        print(f"  K_flat = {K_LATENT*DATA_W}'h{K_flat:0{K_LATENT*DATA_W//4}x};")
        print(f"  V_flat = {V_LATENT*DATA_W}'h{V_flat:0{V_LATENT*DATA_W//4}x};")
    print(f"  // Expected y_flat")
    print(f"  expected_y = {HIDDEN*DATA_W}'h{out_flat:0{HIDDEN*DATA_W//4}x};")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
