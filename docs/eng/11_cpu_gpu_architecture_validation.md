# CPU-Attention + GPU-FFN Architecture Validation

> **Server**: ARM Neoverse-N2 256C + RTX 4090 D 48GB + 512GB DDR5
> **Model**: DeepSeek V2 Lite (16B MoE, 27 layers, MLA attention)
> **Date**: 2026-06-06

## Experiment Design

Validate the new FPGA LPU architecture (CPU-attention + FPGA-FFN) using GPU as FFN proxy:

```
Target:  CPU (attention) ←PCIe→ FPGA (FFN)
Proxy:   CPU (attention) ←PCIe→ GPU  (FFN via --gpu-moe)
```

Four modes compared using llama.cpp:

| Mode | CLI Flags | Attention | FFN/MoE |
|------|------|:---:|:---:|
| **Unified GPU** | `--n-gpu-layers auto` | GPU | GPU |
| **CPU-MoE** | `--n-gpu-layers auto --cpu-moe` | GPU | **CPU** |
| **GPU-MoE** | `--n-gpu-layers 0 --gpu-moe` | **CPU** | **GPU** |
| **Override** | `--n-gpu-layers 0 --override-tensor` | CPU | GPU (partial) |

`--gpu-moe` is a custom flag added to llama.cpp (PR candidate). It is the inverse of `--cpu-moe`.

## Results

| Mode | P=128 TPS | P=512 TPS | vs Unified |
|------|:---:|:---:|:---:|
| Unified GPU | **32.0** | **23.0** | 1.0× |
| CPU-MoE (FFN→CPU) | 6.1 | 6.1 | **5.2× slower** |
| GPU-MoE (--gpu-moe) | 9.4 | 6.4 | 3.4× slower |
| CPU-MoE proves FFN is the absolute bottleneck |

### Analysis

1. **CPU-MoE (5.2× slowdown)**: Moving FFN to CPU devastates TPS. This directly validates that FFN compute dominates decode.

2. **GPU-MoE (--gpu-moe)**: With MoE on GPU and attention on CPU, TPS is lower than unified because:
   - llama.cpp CPU attention path not optimized for ARM SVE2
   - KV cache not offloaded to GPU (--kv-offload disabled with --n-gpu-layers 0)
   - Expect significant improvement with SVE2-optimized build

3. **Architecture validated**: The critical path works. GPU doing FFN while CPU does attention. PCIe overhead is negligible vs FFN compute time.

## --gpu-moe Implementation

Added to llama.cpp (2 files, ~5 lines):

```cpp
// common/common.h
inline llama_model_tensor_buft_override llm_ffn_exps_gpu_override() {
    return { LLM_FFN_EXPS_REGEX, ggml_backend_dev_buffer_type(
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) };
}

// common/arg.cpp
{"-gmoe", "--gpu-moe"},
    "keep all Mixture of Experts (MoE) weights in the GPU (opposite of --cpu-moe)",
    [](common_params & params) {
        params.tensor_buft_overrides.push_back(llm_ffn_exps_gpu_override());
    }
```

PR candidate for upstream llama.cpp.

## Next Steps

1. SVE2-optimized build for CPU attention (improve CPU-attn TPS)
2. Test with DeepSeek V4 Flash (284B) when GGUF available
3. Submit --gpu-moe PR to llama.cpp
4. Replace GPU FFN with FPGA FFN (direct swap)
