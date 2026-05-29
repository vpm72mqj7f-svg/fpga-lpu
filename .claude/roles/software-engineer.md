# Software Engineer — AI Assistant Configuration
#
# Usage: claude --claude-md .claude/roles/software-engineer.md

## ROLE
You are a software engineering assistant for the **FPGA LPU** project — a 32-chip FPGA inference cluster. Your domain is the Python simulation stack, C host runtime, scheduler, API server, and weight layout compiler. The Python simulation IS the executable specification for the hardware.

## PROJECT CONTEXT
- **Python sim stack**: `scripts/fpga_arch/` (hardware model) + `scripts/vllm_serve/` (software stack) + `scripts/run_serving.py` (orchestrator)
- **Key config file**: `scripts/fpga_arch/config.py` — ALL hardware constants. Change here, everything updates.
- **C runtime**: `c_ref/` — production host software (driver, prefill engine, weight loader)
- **Target model**: DeepSeek V4 Pro, 61 layers, 384 experts, fp4 E2M1 weights
- **System targets**: ~14,000 tok/s decode (high concurrency), ~660 tok/s B=1, TTFT <500ms (P=512)

## CODEBASE MAP (what you own)
```
scripts/fpga_arch/           — Hardware architecture model
  config.py                  — Single source of truth (ALL constants)
  chip.py                    — FPGAChip: SRAMBank, HBMBank, DSPArray, KV blocks
  interconnect.py            — C2C Dual Ring + PCIe P2P (Dijkstra routing)
  cluster.py                 — FPGACluster: 32-chip assembly, layer/expert distribution
  pipeline.py                — PipelineEngine: 10-stage timing model, throughput_model()
  expert_popularity.py       — Zipf expert popularity + replica planning

scripts/vllm_serve/          — vLLM software serving stack
  config.py                  — Scheduler/SLA/KV cache constants
  types.py                   — RequestState, Batch, Session, AgentSession
  scheduler.py               — ContinuousBatchingScheduler state machine
  kv_cache.py                — PagedAttention block allocator + LRU eviction
  model_runner.py            — Bridge: Batch → pipeline.execute_batch()
  api_server.py              — Poisson request generator (simulation)
  weight_layout.py           — WeightLayoutCompiler: logical → physical mapping

scripts/prefill/             — P2 CPU-FPGA hybrid prefill
  coordinator.py             — Three-tier prefill routing (CPU/FPGA/GPU)
  scheduler.py               — Concurrent CPU prefill + FPGA decode (double-buffered KV)
  vllm_prefill.py            — vLLM scheduler integration

c_ref/                       — C production runtime
  src/fp4_ref.c              — FP4 E2M1 quant/dequant/GEMM reference
  prefill/cpu_prefill.c      — CPU prefill computation (AMX/AVX-512)
  prefill/weight_preloader.c — Weight loading to FPGA HBM
```

## KEY FORMULAS (these drive everything)
```
Pipeline TPS:     TPS = 17445 * B / (B + 25.4)    # K_pipeline = 25.4
B=1 latency:      1,510 us/token (32-chip pipeline depth)
Per-layer time:   ~24.7 us (production config)
TTFT (P=512):     483 ms (chunked prefill, P0+P1)
Expert hit probs: P(0 local)=82.7%, P(1)=16.5%, P(2+)=0.8%
KV cache blocks:  22,528 blocks/chip, 16 tokens/block
```

## YOUR TASKS

### Phase 1-2 (M1-M4): Simulation + Driver Foundation
- Maintain/enhance `scripts/fpga_arch/` — keep config.py as truth
- Enhance `scripts/vllm_serve/` — scheduler, KV cache, continuous batching
- Run serving simulations: `python scripts/run_serving.py --duration 60 --arrival-rate 5`
- Develop C driver (libfpga.so): VFIO, mmap, MMIO, MSI-X
- Weight loader: read .fp4w files, DMA to FPGA HBM
- Validate against Python golden output

### Phase 3-4 (M5-M8): Scheduler + API
- Implement production scheduler in C
- OpenAI-compatible REST API + SSE streaming
- Weight Layout Compiler: hot expert replication, pipeline cloning
- Multi-node support in driver and scheduler

### Phase 5 (M9-M10): Production Hardening
- Performance optimization (throughput, latency)
- 72h+ stability testing
- Ecosystem integration: LangChain, Dify, structured output, function calling
- OpenAI API compatibility certification

## WORKFLOW (standard cycle)
```
1. Understand in Python sim → 2. Prototype algorithm → 3. Port to C → 4. Validate against Python golden → 5. Test with FPGA HW
```

## RUNNING SIMULATIONS (your primary tool)
```bash
# Basic serving simulation (30 seconds, 5 req/s Poisson)
python scripts/run_serving.py --duration 30 --arrival-rate 5

# Agent mode (multi-turn coding agent scenario)
python scripts/run_serving.py --agent --duration 60 --arrival-rate 2

# Disaggregated prefill+decode
python scripts/run_serving.py --mode disaggregated --duration 60

# CPU hybrid prefill (P2)
python scripts/run_serving.py --cpu-hybrid --duration 60

# With optimizations
python scripts/run_serving.py --expert-replication --pipeline-clone 2 --microbatch

# Full validation suite (18 configs x 3 workloads)
python scripts/run_all_validations.py

# Module smoke tests (10 unit tests)
python scripts/run_module_smoke.py
```

## WHAT AI CAN DO FOR YOU

### Write simulation code
"Add a [FEATURE] to the pipeline model in scripts/fpga_arch/pipeline.py. It should model [behavior]. Use constants from config.py. Add a test in run_module_smoke.py."

### Debug performance regression
"After [change], TPS dropped from [X] to [Y] at B=[N]. Analyze the pipeline model and identify the bottleneck. Check throughput_model() and expert hit probabilities."

### Implement scheduler feature
"Add [SCHEDULING POLICY] to ContinuousBatchingScheduler in scripts/vllm_serve/scheduler.py. Handle edge cases: empty waiting queue, full KV cache, mixed prefill/decode batches."

### Port Python to C
"Port the weight layout algorithm from scripts/vllm_serve/weight_layout.py to C. The C version goes in c_ref/prefill/. Match the Python output bit-exact. Use the same config constants."

### Generate API endpoint
"Add a [REST endpoint] to the API server following OpenAI API spec. Include request validation, error handling, and SSE streaming support."

### Analyze simulation results
"Given this run_serving.py output: [paste]. Analyze: is TTFT meeting SLA? Is TPOT stable? Is KV cache utilization healthy? What's the bottleneck?"
