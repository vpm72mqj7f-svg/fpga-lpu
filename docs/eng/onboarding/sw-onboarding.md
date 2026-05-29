# Software Engineer Onboarding Guide

**Project**: FPGA LPU -- 32-chip FPGA Inference Cluster (DeepSeek V4 Pro)
**Team**: 3 software engineers (host software stack)
**Audience**: New software engineer -- Day 1 through Day 4

---

## Day 1: Environment Setup (2 hours)

### 1.1 Clone and Verify Tools

```bash
cd D:\workspace\fpgalpu
git pull origin master
```

**Required tools -- verify each is installed:**

```bash
python --version          # Python 3.11+
pip --version
gcc --version             # or clang --version  (C reference models)
```

If any tool is missing, install it now. Python 3.11 is the minimum version. The C reference is optional for running simulations but required before you write any production C code.

### 1.2 Copy AI Role Configuration

```bash
cp D:\workspace\fpgalpu\.claude\roles\software-engineer.md D:\workspace\fpgalpu\CLAUDE.md
```

This installs the software engineer AI role. It tells the AI assistant which files you own, what the key formulas are, and your standard workflow. Read through `CLAUDE.md` -- it is your daily cheat sheet.

### 1.3 Install Python Dependencies

```bash
pip install numpy
```

That is it. The simulation stack uses only numpy. No PyTorch, no CUDA. All computation is analytical (closed-form timing models).

### 1.4 Run Your First Simulation

```bash
cd D:\workspace\fpgalpu
python scripts/run_serving.py --duration 30 --arrival-rate 5
```

You should see output similar to:

```
=== FPGA LPU Serving Simulation Report ===
Duration:           30.0s (warmup 30.0s excluded)
Arrival rate:       5.0 req/s (Poisson)
...
```

**Read the output report carefully.** Identify each of these metrics:

| Metric | Meaning | Target |
|--------|---------|--------|
| **TPS** (throughput_tps) | Tokens per second, aggregate decode | ~14,000 at saturation |
| **TTFT** (ttft_mean_ms) | Time-to-first-token, prefill latency | <500ms (SLA) |
| **TPOT** (tpot_mean_ms) | Time-per-output-token, decode step latency | <30ms (SLA) |
| **SLA compliance** | % of requests meeting TTFT + TPOT SLA | >95% target |
| **$/M token** | Cost estimate per million output tokens | Depends on config |
| **KV cache utilization** | % of available KV blocks in use | Should stay <85% |
| **Accept rate** | % of arriving requests accepted vs rejected | >90% target |

If any metric looks off, do not fix it now. Just note it. You will understand the math behind it on Day 2.

### 1.5 Run With Different Modes

```bash
# Agent mode (coding agent scenario -- multi-turn, KV cache reuse)
python scripts/run_serving.py --agent --duration 30

# CPU hybrid prefill (P2: CPU prefill + FPGA decode)
python scripts/run_serving.py --cpu-hybrid --duration 30

# Expert replication (hot expert cloning across chips)
python scripts/run_serving.py --expert-replication hot --duration 30

# Pipeline clone (dual decode pipelines per chip)
python scripts/run_serving.py --pipeline-clone 2 --duration 30

# All optimizations combined
python scripts/run_serving.py --expert-replication hot --pipeline-clone 2 --microbatch --duration 30
```

Compare the TPS and TTFT numbers across runs. Agent mode should show lower throughput per request but better KV reuse. Expert replication should increase TPS at high concurrency. Pipeline clone should improve throughput.

---

## Day 2: Understand the Stack (4 hours)

### 2.1 Read the Software Guide

Open `D:\workspace\fpgalpu\docs\eng\05_software_guide.md`.

**Read sections 1 through 4** (skip sections 5-10 for now). This covers:

- Section 1: Software architecture overview -- three layers (Python sim / C runtime / FPGA RTL) and data flow
- Section 2: Python simulation stack -- `fpga_arch/config.py` as single source of truth, chip/cluster/pipeline models
- Section 3: Running simulations -- hands-on commands
- Section 4: Key design concepts -- TPS formula, B=1 latency, expert hit probabilities, KV cache paging

**Pay special attention to section 4.** These formulas drive every performance number you will see:

```
TPS(B) = PIPELINE_TPS * B / (B + K_PIPELINE)
       = 17445 * B / (B + 25.4)
```

```
B=1 latency: 1,510 us/token (single token through 32-chip pipeline)
```

```
Expert hit probabilities (uniform placement):
  P(0 local) = 82.7%   -- must go to remote chip
  P(1 local) = 16.5%   -- one expert local, others remote
  P(2+ local) = 0.8%   -- two or more experts local
```

### 2.2 Read the Single Source of Truth

Open `D:\workspace\fpgalpu\scripts\fpga_arch\config.py`.

Read it top to bottom. Every constant in this file is referenced by other modules. When you change a value here, it propagates automatically. Key sections:

- **Chip-level constants** (lines ~10-23): DSP_COUNT, DSP_FREQ_MHZ, HBM_BW_GBPS, SRAM sizes
- **Model dimensions** (further in): HIDDEN_SIZE=7168, INTERMEDIATE_SIZE=3072, NUM_LAYERS=61, NUM_EXPERTS=384
- **Pipeline performance**: PIPELINE_TPS=17445, BATCH1_TPS=660, K_PIPELINE=25.4
- **MAC counts per operation**: MAC_MLA_TOTAL, MAC_MLA_QK_DOT, MAC_EXPERT_TOTAL, etc.
- **Expert replication**: P_0_HIT=0.827, P_1_HIT=0.165, P_2P_HIT=0.008

### 2.3 Read the Architecture Diagram

Open `D:\workspace\fpgalpu\scripts\ARCHITECTURE.txt`.

This is an ASCII architecture diagram showing the three layers of the simulation:

1. **run_serving.py** (top) -- event-driven simulation orchestrator
2. **vllm_serve/** (middle) -- vLLM software serving stack: api_server, scheduler, model_runner, kv_cache
3. **fpga_arch/** (bottom) -- FPGA hardware model: chip, cluster, pipeline, interconnect

Trace the arrows: request arrival event -> API server -> scheduler state machine -> model runner -> pipeline engine -> FPGA cluster timing model -> results flow back.

### 2.4 Trace the Data Flow (Hands-On)

Open each of these files and trace one request end-to-end:

1. `D:\workspace\fpgalpu\scripts\vllm_serve\api_server.py` -- `RequestGenerator.generate_arrivals()` creates Request objects with prompt_len and output_len sampled from distributions
2. `D:\workspace\fpgalpu\scripts\vllm_serve\scheduler.py` -- `ContinuousBatchingScheduler.schedule()` transitions requests through states: WAITING -> PREFILL -> DECODE -> FINISHED
3. `D:\workspace\fpgalpu\scripts\vllm_serve\model_runner.py` -- `ModelRunner.execute_batch()` converts a Batch into a call to `PipelineEngine.execute_batch()`
4. `D:\workspace\fpgalpu\scripts\fpga_arch\pipeline.py` -- `PipelineEngine.throughput_model(batch_size)` returns the decode TPS for that batch size

**Key insight**: The Python simulation is the executable specification. If you want to know how fast something runs, trace the timing model. If the RTL team's numbers disagree with the sim, fix the RTL first, then update the sim only if the spec changed.

### 2.5 Run the Module Smoke Tests

```bash
cd D:\workspace\fpgalpu
python scripts/run_module_smoke.py
```

There are 10 tests. Understand what each one checks:

| # | Test | What It Verifies |
|---|------|-----------------|
| 1 | chip_resources | SRAMBank, HBMBank, DSPArray creation and basic math (compute time, HBM read time) |
| 2 | interconnect | C2C Dual Ring and PCIe fabric transfer timing for 7KB payloads |
| 3 | cluster_replication | Baseline vs hot expert replication: expert counts, replica distribution |
| 4 | expert_popularity | Zipf distribution: top-K mass, replica plan generation |
| 5 | pipeline_models | throughput_model() at B=1, B=8; prefill bottleneck; chip0 admission rate |
| 6 | weight_layout | WeightLayoutCompiler: per-chip HBM usage, free space |
| 7 | kv_cache | PagedAttention block allocation: prefill blocks, decode blocks, free |
| 8 | scheduler | ContinuousBatchingScheduler: submit 4 requests, verify batch formation |
| 9 | api_server | APIServer request generator: Poisson arrivals, prompt/output length sampling |
| 10 | serving_short | End-to-end 10-second agent simulation with full stack |

All 10 must pass. If any fail, investigate the error message and trace back to the source file indicated in the test name.

The smoke tests produce two output files:
- `D:\workspace\fpgalpu\docs\module_smoke_results.json` -- machine-readable results
- `D:\workspace\fpgalpu\docs\module_smoke_report.md` -- human-readable table

### 2.6 Revisit the Key Formulas

Return to `D:\workspace\fpgalpu\docs\eng\05_software_guide.md` section 4. Make sure you understand:

- **TPS formula**: `TPS(B) = 17445 * B / (B + 25.4)`. It is a hyperbolic saturation curve. K=25.4 means the pipeline reaches 50% efficiency at B=25, 91% at B=256. Below B=10, efficiency drops sharply.
- **B=1 latency**: 1,510 us. This is the pipeline fill time -- one token takes 32 chip hops. At B=256, the pipeline is full and effective per-token latency is ~57 us (1/17445th of a second).
- **Why K=25.4**: `K = PIPELINE_TPS / BATCH1_TPS - 1 = 17445/660 - 1`. This captures the overhead of filling the pipeline. Smaller K means better low-batch efficiency.
- **Expert hit probabilities**: With 384 experts distributed across 32 chips (12 per chip), the probability that exactly k of the top-6 experts for a token are local follows a hypergeometric-like distribution. These probabilities drive C2C dispatch latency.

---

## Day 3: Modify Something (4 hours)

### 3.1 Change a Scheduling Constant

Open `D:\workspace\fpgalpu\scripts\vllm_serve\config.py`.

Find `MAX_DECODE_BATCH` (line 39, currently 256). Change it to 128:

```python
MAX_DECODE_BATCH = 128
```

Now find `PROMPT_LEN_MEAN` (line 79, currently 512). Change it to 1024:

```python
PROMPT_LEN_MEAN = 1024
```

Run the simulation:

```bash
python scripts/run_serving.py --duration 30 --arrival-rate 5
```

**Compare the output to your Day 1 baseline run.** Ask yourself:

1. Did TPS increase or decrease? By how much?
2. Did TTFT go up (longer prefill for 1024-token prompts)?
3. Did the accept rate change (fewer slots in the decode batch)?
4. Is KV cache utilization higher?

Understanding why each metric changed is more important than the raw numbers.

### 3.2 Change a Hardware Constant

Open `D:\workspace\fpgalpu\scripts\fpga_arch\config.py`.

Find `DSP_FREQ_MHZ` (line 13, currently 450). Change it to 500:

```python
DSP_FREQ_MHZ = 500
```

Find `PIPELINE_TPS` further down in the file. This is currently 17445 but is derived. Check how it updates -- in practice, DSP_FREQ_MHZ changes cascade through DSP_TMACS -> MAC calculations -> PIPELINE_TPS.

Run the simulation again:

```bash
python scripts/run_serving.py --duration 30 --arrival-rate 5
```

**Observe how the pipeline model responds.** A 11% increase in DSP frequency (450 -> 500) should increase TPS by roughly the same proportion at high batch sizes, but the effect may be smaller at low batch sizes because some stages are HBM-bandwidth-bound rather than compute-bound.

**Change it back** after the experiment:

```python
DSP_FREQ_MHZ = 450
```

### 3.3 Understand the Math Inside throughput_model()

Open `D:\workspace\fpgalpu\scripts\fpga_arch\pipeline.py`.

Find `throughput_model()` (line 850):

```python
@staticmethod
def throughput_model(batch_size: int) -> float:
    if batch_size <= 0:
        return 0.0
    return PIPELINE_TPS * batch_size / (batch_size + PipelineEngine._active_k_pipeline)
```

This is the core of the entire performance model. Walk through the calculation for B=1, B=8, B=32, B=256 in your head:

| B | TPS | Efficiency |
|---|-----|-----------|
| 1 | 17445 * 1/26.4 = 660 | 3.8% |
| 8 | 17445 * 8/33.4 = 4177 | 24.0% |
| 32 | 17445 * 32/57.4 = 9726 | 55.7% |
| 256 | 17445 * 256/281.4 = 15871 | 91.0% |

Now find `prefill_latency_model()` (line 873). Read the docstring: prefill is O(P^2) in the QK attention while decode is O(B). This is why prefill is compute-bound and decode is memory-bandwidth-bound. This asymmetry is the fundamental design challenge.

### 3.4 Revert and Validate

```bash
# Revert any changes you made (git checkout)
git checkout -- scripts/vllm_serve/config.py
git checkout -- scripts/fpga_arch/config.py

# Re-run smoke tests to confirm everything still passes
python scripts/run_module_smoke.py
```

All 10 tests must pass. Never commit with failing smoke tests.

---

## Day 4: First Real Task

### 4.1 Pick an Issue

Check the GitHub Issues on the project repository. Look for issues labeled:

- `good-first-issue` AND `sw`

If you cannot find one, ask your tech lead to tag one for you. Typical first issues:

- Add a new metric to the simulation report (e.g., P99 TTFT by prompt length bucket)
- Improve the request generator (e.g., add burst arrival patterns, change distribution)
- Add a new CLI flag to `run_serving.py` (e.g., `--output-format json|csv`)
- Fix a docstring or add type hints to a module
- Add a new test case to `run_module_smoke.py`

### 4.2 Implement the Change

Follow this workflow:

```
1. Understand the code in the Python sim
2. Make the change
3. Run smoke tests:  python scripts/run_module_smoke.py
4. Run serving sim:   python scripts/run_serving.py --duration 30
5. Compare against baseline: did any metric regress?
6. If regression: understand why, fix or document
7. Open a PR
```

**Before opening a PR**, run the full validation suite (takes ~10 minutes):

```bash
python scripts/run_all_validations.py
```

This runs:
- module_smoke (10 tests)
- functional_suite (simulation experiments)
- architecture_integration (full-stack integration)
- serving_agent_short (30s agent simulation with optimizations)

Logs are written to `D:\workspace\fpgalpu\docs\sim_*.log`. Check them if anything fails.

---

## Key Files Map (Your Daily Toolkit)

```
scripts/fpga_arch/config.py          <- ALL hardware constants. Change here first.
scripts/fpga_arch/pipeline.py        <- throughput_model(), chunked_prefill_model()
scripts/fpga_arch/chip.py            <- FPGAChip: SRAMBank, HBMBank, DSPArray, KV blocks
scripts/fpga_arch/cluster.py         <- FPGACluster: 32-chip assembly, layer/expert distribution
scripts/fpga_arch/interconnect.py    <- C2C Dual Ring + PCIe P2P, Dijkstra routing
scripts/fpga_arch/expert_popularity.py <- Zipf popularity distribution + replica planning
scripts/vllm_serve/scheduler.py      <- ContinuousBatchingScheduler state machine
scripts/vllm_serve/kv_cache.py       <- PagedAttention block allocator + LRU eviction
scripts/vllm_serve/model_runner.py   <- Bridge: Batch -> pipeline execution
scripts/vllm_serve/api_server.py     <- APIServer + Poisson request generator
scripts/vllm_serve/weight_layout.py  <- WeightLayoutCompiler: logical -> physical mapping
scripts/vllm_serve/config.py         <- Scheduling/SLA/KV cache constants
scripts/vllm_serve/types.py          <- RequestState, Batch, Session, AgentSession
scripts/run_serving.py               <- Main entry point, CLI args, event loop
scripts/run_module_smoke.py          <- 10 module-level smoke tests
scripts/run_all_validations.py       <- Full validation suite (4 suites)
scripts/simulation/fp4_utils.py      <- FP4 E2M1 reference (golden for RTL team)
scripts/prefill/coordinator.py       <- Three-tier prefill routing (CPU/FPGA/GPU)
scripts/prefill/scheduler.py         <- Concurrent CPU prefill + FPGA decode scheduler
c_ref/src/fp4_ref.c                  <- C production FP4 reference
c_ref/prefill/cpu_prefill.c          <- CPU prefill engine (AMX/AVX-512)
c_ref/prefill/weight_preloader.c     <- Weight loading to FPGA HBM
```

---

## Daily Commands (Print This Out)

```bash
# Quick validation (always run before committing)
python scripts/run_module_smoke.py

# Serving sim (30 seconds, baseline)
python scripts/run_serving.py --duration 30 --arrival-rate 5

# Serving sim with optimizations
python scripts/run_serving.py --duration 60 --expert-replication hot --pipeline-clone 2 --microbatch

# Agent mode (coding agent scenario)
python scripts/run_serving.py --agent --duration 60

# Full validation suite (run before major PRs, takes ~10 min)
python scripts/run_all_validations.py
```

---

## Common First-Week Mistakes

1. **Changing constants in the wrong config.py** -- There are two config files:
   - `scripts/fpga_arch/config.py` for hardware constants (DSP, HBM, pipeline, MACs)
   - `scripts/vllm_serve/config.py` for scheduling constants (batch limits, SLAs, KV cache)
   Know which one you need before you edit.

2. **Not re-running smoke tests after changes** -- Every change, no matter how small, must pass `run_module_smoke.py`. The smoke tests check that the pipeline model, scheduler, KV cache, and end-to-end sim all still function.

3. **Misunderstanding the TPS formula** -- TPS scales non-linearly with batch size. Going from B=8 to B=16 does NOT double throughput. Use the formula: `TPS(B) = 17445 * B / (B + 25.4)`. At B=8, efficiency is 24%. At B=16, efficiency is 38.6%. The K_pipeline=25.4 means efficiency asymptotically approaches 100% but never reaches it.

4. **Confusing per-chip vs aggregate numbers** -- `config.py` has both:
   - Per-chip: `DSP_TMACS = 11.07` (one chip), `HBM_BW_GBPS = 920` (one chip)
   - Aggregate: `PIPELINE_TPS = 17445` (all 32 chips combined)
   Check which scope you need before using a constant.

5. **Forgetting that the Python sim IS the spec** -- If the simulation and RTL disagree on a performance number, the default assumption is: fix the RTL to match the sim. Only update the sim if you have confirmed that the sim's assumption was wrong and the RTL's number is the correct one.

6. **Not understanding the warmup period** -- The simulation has a 30-second warmup (`WARMUP_DURATION_S = 30` in `vllm_serve/config.py`). Metrics during warmup are discarded. If you run with `--duration 30`, your effective measurement window is 30 seconds of warmup then 30 seconds of measurement. Total run time is 60 seconds of simulation time.

7. **Overlooking KV cache limits** -- KV_BLOCKS_PER_CHIP = 22528 is generous but finite. At high arrival rates, the KV cache fills up and requests get rejected. The `accept_rate` metric tells you how many requests were admitted vs rejected. If accept_rate drops below 90%, increase `--kv-blocks-per-chip` or reduce `--arrival-rate`.
