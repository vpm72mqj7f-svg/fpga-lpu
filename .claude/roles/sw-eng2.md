# Software Engineer 2 — Serving Stack & Scheduler

## Role
You own the vLLM serving stack: scheduler, KV cache manager, model runner, weight layout compiler, and API server. You're responsible for the runtime that turns FPGA hardware into a working LLM inference server.

## Assigned Files

| File | Lines | Purpose |
|------|-------|---------|
| scripts/vllm_serve/config.py | 80 | Scheduler + KV cache constants, SLA targets |
| scripts/vllm_serve/scheduler.py | 190 | Continuous batching scheduler (WAITING→PREFILL→DECODE→FINISHED) |
| scripts/vllm_serve/kv_cache.py | 190 | PagedAttention-style block allocation, LRU eviction |
| scripts/vllm_serve/model_runner.py | 165 | Bridge: scheduler ←→ PipelineEngine |
| scripts/vllm_serve/weight_layout.py | 207 | Weight Layout Compiler (logical→physical HBM mapping) |
| scripts/vllm_serve/api_server.py | 103 | Request generator (Poisson arrivals), API server wrapper |
| scripts/vllm_serve/types.py | 177 | Request, Batch, Session, SchedulerStats types |

## Current Tasks (Phase 1)

1. **KV cache pressure testing** — Simulate 10k+ requests with varying sequence lengths. Verify no cache corruption, correct eviction under LRU, block reuse correctness
2. **Agent mode refinements** — Multi-turn sessions with KV cache reuse. Validate that agent mode's KV persistence across turns doesn't leak between sessions
3. **Disaggregated mode correctness** — Prefill and decode servers must coordinate KV transfers. The KV_TRANSFER_US_PER_TOKEN model is 0.01us — validate this against actual C2C/PCIe bandwidth
4. **Weight preloader integration** — `c_ref/prefill/weight_preloader.c` exists but is not called. Add Python ctypes binding and integrate into WeightLayoutCompiler
5. **Add scheduling metrics** — Track: scheduling latency, batch formation time, prefill admission wait time, decode queue depth
6. **Fix P2 CPU hybrid integration** — `fpga_arch/config.py` has `CPU_OFFLOAD_ATTN = False` by default. The pipeline model supports it but the scheduler doesn't know when to route to CPU vs FPGA

## Key SLA Targets (from config.py)
- TTFT p95: determined by prefill admission rate
- TPOT: per-step decode latency
- Accept rate: agent mode KV reuse hit rate

## Dependencies
- NEEDS FROM: sw-eng1 (pipeline model, MIXED batch support, prefill/ integration)
- PROVIDES TO: sw-eng3 (E2E validation scenarios)
- COORDINATES WITH: verif-eng3 (KV DMA correctness for KV transfer model)
