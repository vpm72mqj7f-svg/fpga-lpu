"""
fp4 E2M1 quantization utilities for DeepSeek V4 Pro simulation.

Format: E2M1 (1 sign, 2 exponent, 1 mantissa)
  Normal:  (-1)^s × 2^(e-1) × (1 + m/2),  e ∈ {1,2,3}
  Subnorm: (-1)^s × 2^0 × m/2,             e = 0

Values: 0, ±0.25, ±0.5, ±0.75, ±1.0, ±1.5, ±2.0, ±3.0
"""

import numpy as np

# fp4 E2M1 values (positive, magnitude)
FP4_POS_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)
FP4_MAX = 3.0


def fp4_e2m1_values():
    """Return all 16 fp4 values (positive, negative, zero)."""
    pos = list(FP4_POS_VALUES)
    neg = [-x for x in pos if x != 0]
    return sorted(neg + pos)


def fp4_e2m1_info():
    """Return a string describing the fp4 format."""
    vals = fp4_e2m1_values()
    return (f"fp4 E2M1: {len(vals)} values = {vals}\n"
            f"  Normal:  (-1)^s × 2^(e-1) × (1 + m/2), e∈{{1,2,3}}\n"
            f"  Subnorm: (-1)^s × m/2, e=0\n"
            f"  Max: ±3.0, Min nonzero: ±0.25, Zero: 0")


def quantize_fp4_e2m1(tensor, group_size=128):
    """
    Quantize a float32 tensor to fp4 with per-group FP8 scaling.

    Args:
        tensor: numpy float32 array, shape [..., K]
        group_size: elements per scaling group (last dim)

    Returns:
        fp4_indices: uint8 array with 4-bit index per element (0-15)
        fp8_scales: float32 per-group scales
    """
    shape = tensor.shape
    flat = tensor.reshape(-1, shape[-1]) if tensor.ndim > 1 else tensor.reshape(1, -1)
    N = flat.shape[-1]
    num_groups = (N + group_size - 1) // group_size

    fp4_indices = np.zeros(flat.shape, dtype=np.uint8)
    fp8_scales = np.zeros(flat.shape[:-1] + (num_groups,), dtype=np.float32)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, N)
        group = flat[..., start:end]

        # Max absolute value for scaling
        amax = np.max(np.abs(group), axis=-1, keepdims=True)
        amax = np.maximum(amax, 1e-12)
        scale = amax / FP4_MAX
        fp8_scales[..., g] = scale.squeeze(-1)

        # Quantize
        scaled = group / scale
        scaled = np.clip(scaled, -FP4_MAX, FP4_MAX)

        # Find nearest positive fp4 value
        abs_scaled = np.abs(scaled)
        # [..., N, 8] differences
        diffs = np.abs(abs_scaled[..., None] - FP4_POS_VALUES)
        nearest_idx = np.argmin(diffs, axis=-1)

        # Apply sign
        sign = (scaled >= 0).astype(np.uint8)
        indices = nearest_idx.astype(np.uint8)
        indices = np.where(sign, indices, indices | 0x8)

        fp4_indices[..., start:end] = indices

    return fp4_indices, fp8_scales


def dequantize_fp4_e2m1(fp4_indices, fp8_scales, group_size=128, original_last_dim=None):
    """
    Dequantize fp4 weights back to float32.

    Args:
        fp4_indices: uint8 array with 4-bit indices per element
        fp8_scales: per-group scaling factors
        group_size: elements per scaling group
        original_last_dim: original last dimension size

    Returns:
        float32 array
    """
    N = original_last_dim if original_last_dim is not None else fp4_indices.shape[-1]

    # Extract sign and magnitude
    mag_idx = fp4_indices & 0x7
    sign = (fp4_indices >> 3) & 0x1

    values = FP4_POS_VALUES[mag_idx.clip(0, 7).astype(int)]
    values = np.where(sign.astype(bool), -values, values)

    # Apply per-group scales
    num_groups = fp8_scales.shape[-1]
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, values.shape[-1])
        scale_slice = fp8_scales[..., g:g + 1]
        values[..., start:end] *= scale_slice

    return values[..., :N]


def fp4_gemm_simulate(weight_fp4_idx, weight_scales, activation, group_size=128):
    """
    Simulate fp4 weight × fp8 activation GEMM as on FPGA DSP.

    FPGA data path: weight(fp4)→dequant→fp8 × activation(fp8) → FP32 accumulate

    Args:
        weight_fp4_idx: [M, K] uint8 fp4 indices
        weight_scales: [M, K//group_size] per-group scales
        activation: [K, N] float32 (simulating fp8 with noise)
        group_size: elements per scaling group

    Returns:
        [M, N] float32 output
    """
    # Dequantize fp4 weights
    weight_fp32 = dequantize_fp4_e2m1(weight_fp4_idx, weight_scales, group_size)

    # Truncate/pad to match activation
    K_act = activation.shape[0]
    if weight_fp32.shape[-1] < K_act:
        weight_fp32 = np.pad(weight_fp32, ((0, 0), (0, K_act - weight_fp32.shape[-1])))
    else:
        weight_fp32 = weight_fp32[..., :K_act]

    # Simulate FP8 activation quantization noise (~0.4% relative)
    act_amax = np.max(np.abs(activation))
    if act_amax > 0:
        noise_scale = act_amax * (0.5 / 128)
        act_quant = activation + np.random.randn(*activation.shape).astype(np.float32) * noise_scale
    else:
        act_quant = activation

    # GEMM with FP32 accumulation
    return weight_fp32 @ act_quant


def compute_output_diff(fp4_output, bf16_ref):
    """
    Compute per-token difference metrics between fp4 and bf16 reference.

    Args:
        fp4_output: [B, H] float32
        bf16_ref: [B, H] float32

    Returns:
        dict with metrics
    """
    B = fp4_output.shape[0]

    # Per-token cosine similarity
    fp4_norm = np.linalg.norm(fp4_output, axis=-1)
    ref_norm = np.linalg.norm(bf16_ref, axis=-1)
    dot = np.sum(fp4_output * bf16_ref, axis=-1)
    denom = np.maximum(fp4_norm * ref_norm, 1e-8)
    cos_sim = dot / denom

    # Per-token relative error
    rel_err = np.linalg.norm(fp4_output - bf16_ref, axis=-1) / np.maximum(ref_norm, 1e-8)

    # Max absolute error per token
    max_abs = np.max(np.abs(fp4_output - bf16_ref), axis=-1)

    return {
        'cosine_similarity': cos_sim,
        'mean_cosine': float(np.mean(cos_sim)),
        'min_cosine': float(np.min(cos_sim)),
        'mean_relative_error': float(np.mean(rel_err)),
        'max_absolute_error': float(np.mean(max_abs)),
        'per_token_diff': fp4_output - bf16_ref,
    }
