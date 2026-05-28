# FPGA LPU: 32-Chip LLM Inference Cluster -- Engineering Project Plan

**Document Version:** 1.0
**Date:** 2026-05-28
**Target Model:** DeepSeek V4 Pro (61 layers, 384 experts, MLA attention, fp4 weights)
**Target Platform:** Intel Agilex 7 M-Series (AGM 039-F) on DK-DEV-AGM039EA boards
**Project Duration:** 10 months (M1-M10)
**Team Size:** 10 engineers (4 RTL, 3 verification/test, 3 C/software)

---

## 1. Executive Summary

This project delivers a 32-chip FPGA inference cluster capable of serving DeepSeek V4 Pro at ~14,000 tok/s decode throughput under high concurrency and ~660 tok/s at batch-size 1. The architecture exploits a fundamental mismatch between GPU design and LLM decode physics: GPU Tensor Cores sit idle 95-98% of the time during decode because the workload is memory-bandwidth-bound (6 MACs per byte loaded from HBM), not compute-bound. FPGA HBM2e at 920 GB/s per chip matches or exceeds GPU HBM bandwidth per dollar while eliminating the stranded compute silicon. Four RTL engineers deliver ~50 SystemVerilog modules across the critical path (fp4 systolic array, MLA attention pipeline, KV cache manager, MoE router/dispatch, chip-to-chip interconnect); three verification engineers build the Icarus-to-Quartus verification continuum with Python golden models; three C/software engineers build the vLLM-compatible serving stack with continuous batching, PagedAttention KV cache, CPU-FPGA hybrid prefill, and OpenAI-compatible API endpoint. The 10-month timeline is organized into five 2-month phases with explicit go/no-go gates.

---

## 2. Team Structure and Roles

### 2.1 Role Definitions

| Role | Headcount | Primary Responsibility | Key Tools |
|------|-----------|----------------------|-----------|
| **RTL Engineer** | 4 | SystemVerilog module design, synthesis, timing closure, Quartus compilation | Quartus Prime Pro 24.3, Icarus Verilog, Signal Tap, Platform Designer |
| **Verification Engineer** | 3 | Testbench development, golden model cross-validation, CI automation, hardware bring-up testing | Python (NumPy), Cocotb (optional), Icarus, Quartus, Signal Tap |
| **C/Software Engineer** | 3 | FPGA driver, vLLM serving integration, CPU prefill, PCIe DMA runtime, API server, cluster orchestration | C (AVX-512/AMX), Python, Intel OFS drivers, PCIe DMA library |

### 2.2 Named Role Assignments

Each engineer has a primary role and a secondary domain for redundancy.

| Engineer ID | Role | Primary Module Focus | Secondary Module Focus | Geographic TZ |
|-------------|------|---------------------|----------------------|---------------|
| RTL-1 | Sr. RTL Lead | fp4 systolic array family, DSP characterization | GEMM engine, prefill engine | TBD |
| RTL-2 | RTL Engineer | MLA attention pipeline, KV cache | RMSNorm, SiLU activation | TBD |
| RTL-3 | RTL Engineer | MoE router, expert FFN engine, shared expert | Token embedding, lm_head, MTP | TBD |
| RTL-4 | RTL Engineer | Chip2Chip router, PCIe DMA wrapper, KV DMA engine | Inference control FSM, pipeline forward | TBD |
| VRF-1 | Sr. Verification Lead | Test architecture, golden model pipeline, CI | Full-layer testbenches, cluster simulation | TBD |
| VRF-2 | Verification Engineer | DSP/MAC testbenches, fp4 precision validation | FPGA validation scripts, HBM characterization | TBD |
| VRF-3 | Verification Engineer | Full-layer and cluster testbenches, bring-up test scripts | MoE routing tests, C2C ring tests | TBD |
| SW-1 | Sr. Software Lead | vLLM scheduler, KV cache manager, API server | Weight layout compiler, model runner | TBD |
| SW-2 | Software Engineer | FPGA driver, PCIe DMA runtime, weight streaming | CPU prefill integration, coordinator | TBD |
| SW-3 | Software Engineer | CPU prefill engine (AMX/AVX-512), performance analysis | Cluster orchestration, monitoring | TBD |

### 2.3 Cross-Training Matrix

Every role has a designated backup who can step in within 48 hours.

| Primary | Backup 1 | Backup 2 |
|---------|----------|----------|
| RTL-1   | RTL-2   | VRF-2   |
| RTL-2   | RTL-3   | VRF-1   |
| RTL-3   | RTL-4   | VRF-3   |
| RTL-4   | RTL-1   | SW-2   |
| VRF-1   | VRF-2   | RTL-2   |
| VRF-2   | VRF-3   | RTL-1   |
| VRF-3   | VRF-1   | RTL-3   |
| SW-1    | SW-2    | SW-3    |
| SW-2    | SW-3    | SW-1    |
| SW-3    | SW-1    | SW-2    |

---

## 3. Phase-by-Phase Plan

### 3.1 Phase 1: Single-Card Verification (Month 1-2)

**Objective:** Prove the fundamental physics on a single FPGA board. Answer: does fp4 compute match Python golden models on real silicon? Does HBM2e deliver 920 GB/s? Does a single transformer layer produce correct outputs?

**Target Platform:** 1x DK-DEV-AGM039EA board with 1x AGM 039-F chip.

#### Phase 1 Deliverables

| Deliverable | Description | Owner | Success Criterion |
|-------------|-------------|-------|-------------------|
| P1-D1 | PCIe 5.0 x16 link up at Gen5 speed | RTL-4, SW-2 | `lspci` shows Intel Agilex device; Gen5 x16 link trained |
| P1-D2 | HBM2e sequential read/write at >= 800 GB/s | RTL-4, VRF-2 | Measured BW >= 800 GB/s read, >= 700 GB/s write |
| P1-D3 | fp4 x fp8 MAC golden vector runner: 15/15 tests pass on hardware | RTL-1, VRF-2 | All 15 MAC tests match Python golden within 1 ULP |
| P1-D4 | Quartus synthesis passes for `full_transformer_layer.sv` with bring-up dimensions | RTL-1, RTL-2, RTL-3 | Zero synthesis errors, timing closed at 390 MHz DSP |
| P1-D5 | Single-layer inference on hardware matches C golden model | RTL-2, VRF-1 | C0 test exact match; C1 within +/- 4 ULP |
| P1-D6 | fp4 precision comparison vs PyTorch: cosine similarity >= 0.995 | VRF-2, RTL-1 | Cosine similarity across 1000 random vectors >= 0.995 |

#### Phase 1 Work Assignment

| Week | RTL Team (RTL-1..4) | Verification Team (VRF-1..3) | Software Team (SW-1..3) |
|------|---------------------|------------------------------|------------------------|
| W1 | Quartus project setup; Golden Top from BSP verification; RTL code freeze (tag `phase1-freeze`); `quartus_map` on `hw/quartus/fpga_lpu.qpf` | Freeze Python golden model (`scripts/simulation/gen_tb_vectors.py`, `gen_ffn_tb_vectors.py`, `gen_layer_golden.py`); generate committed golden vectors under `hw/test_vectors/` | Install Intel FPGA SDK/oneAPI driver; verify `lspci`; set up PCIe BAR mapping library |
| W2 | HBM example design instantiation in QSYS; UART TX module (`rtl/debug/uart_debug.sv`); LED heartbeat | HBM sequential R/W test script (`hw/scripts/run_golden_tests.py`); Signal Tap setup for HBM AXI probes | Write PCIe DMA driver library (C, `c_ref/src/pcie_dma_lib.c`); BAR0/BAR2 mapping |
| W3 | `fp4_mac.sv` hardware test FSM: MMIO-based scale write, MAC test runner, result register | Run 15 MAC golden tests on hardware (via PCIe); per-bit Signal Tap comparison for failures | Python test orchestration script (`hw/scripts/run_golden_tests.py`); result logging |
| W4 | HBM bandwidth calibration: sequential pattern + MoE random access pattern; bank conflict analysis (layouts A/B/C) | HBM trace generator (`scripts/simulation/gen_hbm_trace.py`); Zipf-distributed access trace; dual-buffer overlap measurement | HBM bandwidth measurement and plotting tools |
| W5 | Synthesize `full_transformer_layer.sv` (bring-up dims: HIDDEN=8); weight preload via BAR0 | Run layer golden case C0 + C1 on hardware; per-stage latency measurement via Signal Tap | C reference model validation (`c_ref/src/fp4_ref.c`): verify C output matches hardware |
| W6 | Timing closure: fix violations in combinational softmax, MLA attention path | Full golden vector re-run; per-stage latency benchmark vs Icarus simulation | Driver stress test: sustained PCIe DMA at >= 28 GB/s |
| W7 | Buffer week: fix any remaining Phase 1 bugs | Buffer week: extend test coverage for edge cases | Buffer week: driver hardening |
| W8 | **Go/No-Go #1 review**; compile Phase 1 report; update TCO with hardware-validated numbers | **Go/No-Go #1 review**; report all three experiment results | **Go/No-Go #1 review**; start Phase 2 prep (order 8x chips, PCB schematic) |

#### Phase 1 Success Criteria

1. **Go/No-Go Gate #1 (MAC Precision):** 15/15 MAC golden tests pass on hardware; fp4 cosine similarity >= 0.995
2. **Go/No-Go Gate #2 (HBM Bandwidth):** MoE random access BW >= 550 GB/s; dual-buffer overlap >= 80%
3. **Go/No-Go Gate #3 (Layer E2E):** C0 golden case exact match; C1 within +/- 4; latency <= 2x Icarus simulation
4. PCIe DMA >= 28 GB/s sustained throughput
5. Quartus full compilation passes for `top_bringup.sv` with timing closed

---

### 3.2 Phase 2: Single-Node 8-Card (Month 3-4)

**Objective:** Scale from one chip to one full node (8 cards x 4 chips = 32 chips). Validate inter-chip communication, tensor-parallel all-reduce, MoE dispatch across chips. Achieve full 15-layer inference throughput benchmark.

**Target Platform:** 1x server with 8x custom 4-chip FPGA cards (32x AGM 039-F total).

#### Phase 2 Deliverables

| Deliverable | Description | Owner | Success Criterion |
|-------------|-------------|-------|-------------------|
| P2-D1 | F-Tile 200GbE link trained on all 8 cards; dual ToR switch configuration | RTL-4 | All 8 QSFP-DD links train; BER < 1e-15 |
| P2-D2 | RoCE v2 RDMA inter-chip communication: RDMA Write/Read verbs functional | RTL-4, SW-2 | Round-trip latency < 5 us; throughput >= 90% of line rate |
| P2-D3 | 8-card TP All-Reduce functional for attention projections | RTL-2, RTL-4 | All-reduce on 7168-element vector within 2 us |
| P2-D4 | MoE Dispatch + Reduce across 8 cards: router selects top-6 experts, dispatches to correct chip, reduces results | RTL-3, RTL-4 | Dispatch + Reduce per MoE layer < 250 ns (1-hit case); < 800 ns (0-hit case) |
| P2-D5 | Full 15-layer inference pipeline running on hardware | RTL-1..4 | End-to-end per-token latency <= 1.5 us (decode) |
| P2-D6 | Throughput benchmark: > 200 tok/s decode at B=8 | VRF-1, SW-1 | Sustained decode throughput >= 200 tok/s |

#### Phase 2 Work Assignment

| Week | RTL Team | Verification Team | Software Team |
|------|----------|-------------------|---------------|
| W9-10 | F-Tile SerDes instantiation on all 4 chips per card; C2C ring controller synthesis for production dimensions | C2C ring loopback testbench updates; dual-ring failover simulation | RoCE v2 driver stack (user-space verbs library for FPGA) |
| W11-12 | Dual ToR switch integration; RoCE RDMA packet format implementation in `rtl/chip/kv_dma_engine.sv` | Multi-chip RDMA testbench; packet loss + retransmission test | RDMA performance benchmark tools; MPI-style collectives over RDMA |
| W13-14 | TP All-Reduce RTL (ring-based); MoE Dispatch packet router | 8-card TP All-Reduce simulation in Python; dispatch trace validation | Weight distribution tool: compile per-chip weight layout for 32 chips |
| W15-16 | Multi-chip pipeline forward; full 15-layer integration | Full-cluster simulation with 15 layers across 32 chips | vLLM scheduler integration: bridge ModelRunner to hardware pipeline |
| W17 | Buffer week; timing closure on all multi-chip paths | Extended cluster simulation with varied batch sizes | End-to-end serving test with 10 concurrent requests |
| W18 | **Go/No-Go #2 review**; Phase 2 report | **Go/No-Go #2 review** | **Go/No-Go #2 review** |

#### Phase 2 Success Criteria

1. C2C dual ring BER < 1e-15 per link; single-hop latency < 50 ns
2. MoE dispatch/reduce for 0-hit case (all 6 experts remote) < 800 ns
3. TP All-Reduce for 7168-dim vector across 8 cards < 2 us
4. Full 15-layer decode >= 200 tok/s at B=8
5. Host-to-FPGA weight loading >= 28 GB/s per master chip

---

### 3.3 Phase 3: Dual-Node Interconnect (Month 5-6)

**Objective:** Scale from one node to two. Validate cross-node RoCE RDMA, ToR MLAG failover, and 30-layer pipeline. This is the first milestone where the system runs more layers than any single node can hold.

**Target Platform:** 2x servers (64x AGM 039-F chips). Dual ToR switches with MLAG.

#### Phase 3 Deliverables

| Deliverable | Description | Owner | Success Criterion |
|-------------|-------------|-------|-------------------|
| P3-D1 | Dual ToR MLAG configuration with RoCE multipath | RTL-4, SW-2 | Traffic flows across both ToRs; single ToR failure -> zero packet loss |
| P3-D2 | Cross-node RoCE RDMA: reliable transport across server boundary | RTL-4, SW-2 | Cross-node RDMA Write latency < 10 us; BW >= 80 Gbps |
| P3-D3 | ToR failover within 100 ms of link-down detection | RTL-4 | Automatic failover; < 1% packet loss during transition |
| P3-D4 | Cross-node MoE Dispatch + Combine: experts on node-2 correctly receive and return activations | RTL-3, RTL-4 | Cross-node dispatch < 5 us per expert message |
| P3-D5 | Dual-node full 30-layer pipeline (15 layers per node) | RTL-1..4 | End-to-end per-token latency <= 3 us (decode) |
| P3-D6 | Dual-node throughput benchmark >= 500 tok/s at B=16 | VRF-1, SW-1 | Sustained decode >= 500 tok/s |

#### Phase 3 Work Assignment

| Week | RTL Team | Verification Team | Software Team |
|------|----------|-------------------|---------------|
| W19-20 | Cross-node RDMA packet router updates; dual ToR address resolution | Dual-node C2C+RDMA simulation model | Cluster orchestration: node discovery, topology map, health monitoring |
| W21-22 | ToR failover FSM; MLAG multipath ECMP routing in RDMA stack | Failover injection tests; packet trace comparison pre/post failover | Weight layout compiler: split 30 layers across 2 nodes |
| W23-24 | Cross-node MoE dispatch/combine: multi-hop forwarding through C2C + RDMA | Multi-node MoE dispatch simulation; expert latency profiling | vLLM scheduler: multi-node request routing |
| W25-26 | 30-layer pipeline integration; inter-node pipeline forward | Full 30-layer cluster simulation | End-to-end serving test; latency profiling per layer |
| W27 | Buffer week; performance tuning | Extended soak test (1 hour continuous operation) | Metrics dashboard (Prometheus/Grafana) |
| W28 | Phase 3 review; gate check | Phase 3 report | Phase 3 report |

#### Phase 3 Success Criteria

1. Cross-node RDMA: latency < 10 us, BW >= 80 Gbps
2. ToR failover: recovery < 100 ms; no dropped tokens
3. Dual-node 30-layer decode >= 500 tok/s at B=16
4. MoE cross-node dispatch + combine <= 5 us per expert

---

### 3.4 Phase 4: Four-Node Full Cluster (Month 7-8)

**Objective:** Deploy the complete 4-node, 32-chip production cluster. Run the full 61-layer DeepSeek V4 Pro pipeline with MTP (Multi-Token Prediction). Validate long-context (128K tokens) and multi-session concurrency (5 to 20 concurrent requests).

**Target Platform:** 4x servers (128x AGM 039-F chips). 4-node RoCE fabric via dual ToR MLAG.

#### Phase 4 Deliverables

| Deliverable | Description | Owner | Success Criterion |
|-------------|-------------|-------|-------------------|
| P4-D1 | 32-chip full deployment: all 4 nodes operational, all layers mapped correctly | RTL-1..4, SW-1..3 | All 61 layers producing correct outputs; chip-to-layer mapping verified |
| P4-D2 | MTP (Multi-Token Prediction) pipeline: lm_head + MTP head on chip 31 | RTL-3 | MTP produces 2 draft tokens; acceptance rate >= 85% |
| P4-D3 | 128K context long-sequence test: KV cache spans full context window | RTL-2, VRF-1 | Correct attention over 128K tokens; KV cache hit rate >= 99% |
| P4-D4 | Multi-session concurrency test: 5, 10, 15, 20 concurrent sessions | SW-1, VRF-1 | Linear throughput scaling up to B=20; no cache thrashing |
| P4-D5 | System benchmark: > 500 tok/s aggregate decode at B=32 | VRF-1, SW-1 | Sustained decode throughput >= 500 tok/s (full system) |

#### Phase 4 Work Assignment

| Week | RTL Team | Verification Team | Software Team |
|------|----------|-------------------|---------------|
| W29-30 | 4-node network bring-up; full chip-to-layer mapping verification | 4-node cluster simulation; topology validation | Cluster management daemon; health checks; rolling restart |
| W31-32 | MTP head integration on chip 31; lm_head projection tensor parallelism | MTP acceptance rate benchmarking vs ground truth | PagedAttention KV cache: 128K context allocation; eviction policy |
| W33-34 | Long-context KV cache stress: 128K sliding window; cross-chip KV forwarding | Long-context golden validation (running attention over full 128K) | Continuous batching scheduler: multi-session optimization |
| W35-36 | Concurrency scaling: 5 -> 10 -> 15 -> 20; per-session metrics | Concurrency load tests; latency vs throughput curves | OpenAI-compatible API server (`scripts/vllm_serve/api_server.py`) integration |
| W37 | Buffer week; system stability testing | 24-hour soak test | Monitoring dashboard; alerting rules |
| W38 | Phase 4 review; benchmark report | Phase 4 report | Phase 4 report |

#### Phase 4 Success Criteria

1. Full 61-layer pipeline producing correct outputs on all 32 chips
2. MTP acceptance rate >= 85%; effective throughput improvement >= 1.7x
3. 128K context: attention correctness validated; KV cache operates within HBM budget
4. Multi-session concurrency: throughput scales to >= 80% of linear from 5 to 20 sessions
5. System benchmark: aggregate decode >= 500 tok/s at B=32

---

### 3.5 Phase 5: Optimization and Production (Month 9-10)

**Objective:** Push the system to production readiness. Extreme context lengths (512K-1M), hot expert multi-replica optimization, fault injection and recovery, power/thermal validation, and OpenAI API compatibility certification.

#### Phase 5 Deliverables

| Deliverable | Description | Owner | Success Criterion |
|-------------|-------------|-------|-------------------|
| P5-D1 | 512K-1M context extreme test: KV cache scalability and performance | RTL-2, VRF-1 | Correct attention at 512K; 1M with performance degradation < 30% |
| P5-D2 | Hot expert multi-replica optimization: Top-8 experts replicated across available HBM | RTL-3, SW-1 | Expert hit rate improvement >= 15%; C2C dispatch reduction >= 10% |
| P5-D3 | Fault injection + failover: chip failure, link failure, node failure scenarios | VRF-3, SW-2 | Graceful degradation; automatic recovery within 5 seconds |
| P5-D4 | Power optimization + thermal verification: per-chip power monitoring; thermal throttling | RTL-4, SW-3 | Total power <= 5.3 kW per node; chip junction temp < 85 C |
| P5-D5 | OpenAI API compatibility certification: `/v1/completions`, `/v1/chat/completions`, streaming | SW-1, SW-3 | All OpenAI test suite endpoints return correct responses |
| P5-D6 | Production deployment: documented runbook; monitoring; alerting; backup/restore | SW-1..3 | System operational with < 1 hour MTTR for common failures |

#### Phase 5 Work Assignment

| Week | RTL Team | Verification Team | Software Team |
|------|----------|-------------------|---------------|
| W39-40 | Extreme context: KV cache bank expansion; sliding window optimization | 512K-1M context simulation; attention correctness validation | OpenAI API endpoint finalization; streaming response support |
| W41-42 | Hot expert replication: HBM layout optimization; replica routing | Expert replication simulation in Python pipeline model | vLLM weight layout compiler: hot expert detection + replica placement |
| W43-44 | Fault injection FSM: chip heartbeat, link timeout, CRC error injection | Fault injection test suite; recovery time measurement | Cluster orchestration: automatic failover; degraded mode operation |
| W45-46 | Power optimization: clock gating for idle pipeline stages; DVFS exploration | Power measurement scripts; thermal camera validation | Power monitoring daemon; thermal throttling policy |
| W47-48 | API certification: run OpenAI Python client test suite against FPGA endpoint | End-to-end correctness: 1000 random prompts vs reference (GPU) output | Runbook documentation; SRE handoff; operational training |
| W49 | Buffer week; final performance profiling | Final validation report | Production deployment |
| W50 | **Project completion review** | Final report | Final report |

#### Phase 5 Success Criteria

1. 512K context: KV cache works correctly; 1M context: functional with < 30% throughput degradation
2. Hot expert replication improves hit rate by >= 15%; C2C dispatch traffic reduced by >= 10%
3. All injected faults recover automatically within 5 seconds
4. Power per node <= 5.3 kW; thermal within spec
5. All OpenAI API compatibility tests pass
6. Production runbook complete; SRE team trained

---

## 4. Module Ownership Matrix

### 4.1 Critical Path RTL Modules (by Effort)

Each module has an owner (design), test owner (verification), and reviewer (code review sign-off).

| Module | File Path(s) | PM Est. | Designer | Test Owner | Reviewer | Phase Due |
|--------|-------------|---------|----------|------------|----------|-----------|
| fp4 MAC (Scale-Aware) | `rtl/dsp/fp4_mac.sv` | 2 | RTL-1 | VRF-2 | RTL-2 | P1.W3 |
| fp4 Scale Reader | `rtl/dsp/fp4_scale_reader.sv` | 0.5 | RTL-1 | VRF-2 | RTL-2 | P1.W3 |
| fp4 Systolic Cell | `rtl/dsp/fp4_systolic_cell.sv` | 1 | RTL-1 | VRF-2 | RTL-2 | P1.W4 |
| fp4 Systolic 2D Array | `rtl/dsp/fp4_systolic_2d.sv` | 2 | RTL-1 | VRF-2 | RTL-2 | P1.W5 |
| fp4 GEMM Engine | `rtl/dsp/fp4_gemm_engine.sv` | 2.5 | RTL-1 | VRF-2 | RTL-2 | P1.W6 |
| fp4 Prefill Engine | `rtl/dsp/fp4_prefill_engine.sv` | 2 | RTL-1 | VRF-2 | RTL-3 | P2.W10 |
| **Systolic Array Subtotal** | | **10** | | | | |
| MLA QKV Projection | `rtl/attention/mla_qkv_proj.sv` | 3 | RTL-2 | VRF-1 | RTL-1 | P1.W5 |
| MLA RoPE | `rtl/attention/mla_rope.sv` | 2 | RTL-2 | VRF-1 | RTL-1 | P1.W5 |
| MLA KV Cache | `rtl/attention/mla_kv_cache.sv` | 2 | RTL-2 | VRF-1 | RTL-4 | P1.W6 |
| MLA Attention v2 | `rtl/attention/mla_attention_v2.sv` | 5 | RTL-2 | VRF-1 | RTL-1 | P1.W7 |
| **MLA Attention Subtotal** | | **12** | | | | |
| KV DMA Engine | `rtl/chip/kv_dma_engine.sv` | 3 | RTL-4 | VRF-3 | RTL-2 | P2.W11 |
| KV DMA Bridge | `rtl/chip/kv_dma_bridge.sv` | 1.5 | RTL-4 | VRF-3 | RTL-2 | P2.W11 |
| MLA KV Cache (shared) | `rtl/attention/mla_kv_cache.sv` | 1.5 | RTL-2 | VRF-1 | RTL-4 | P2.W12 |
| **KV Cache Manager Subtotal** | | **6** | | | | |
| MoE Router Top-K | `rtl/moe/router_topk.sv` | 1.5 | RTL-3 | VRF-3 | RTL-2 | P1.W5 |
| Expert FFN Engine (fp4 down) | `rtl/moe/expert_ffn_engine_fp4_down.sv` | 2 | RTL-3 | VRF-3 | RTL-1 | P1.W6 |
| Shared Expert (integrated) | (within `full_transformer_layer.sv`) | 0.5 | RTL-3 | VRF-1 | RTL-1 | P1.W6 |
| **MoE Router + Dispatch Subtotal** | | **4** | | | | |
| C2C Ring Controller | (PCIe+RoC v2 RDMA, external to `chip_top.sv`) | 3 | RTL-4 | VRF-3 | SW-2 | P2.W10 |
| **Chip2Chip Router Subtotal** | | **3** | | | | |
| Layer Compute Engine | `rtl/layer/layer_compute_engine.sv` | 1 | RTL-4 | VRF-1 | RTL-2 | P1.W6 |
| Full Transformer Layer | `rtl/layer/full_transformer_layer.sv` | 1 | RTL-4 | VRF-1 | RTL-1 | P1.W7 |
| **Inference Control FSM Subtotal** | | **2** | | | | |

### 4.2 Other RTL Modules (1-2 PM Each)

| Module | File Path(s) | PM | Designer | Test Owner | Reviewer | Phase Due |
|--------|-------------|-----|----------|------------|----------|-----------|
| Decoupled RoPE | (within `mla_rope.sv`) | 1 | RTL-2 | VRF-1 | RTL-1 | P1.W5 |
| Shared Expert FFN | (within `full_transformer_layer.sv`) | 1 | RTL-3 | VRF-1 | RTL-1 | P1.W6 |
| RMSNorm | `rtl/activation/rms_norm.sv` | 0.5 | RTL-3 | VRF-2 | RTL-2 | P1.W4 |
| SiLU LUT (Q12) | `rtl/activation/silu_q12_lut.sv` | 0.5 | RTL-3 | VRF-2 | RTL-2 | P1.W4 |
| Q12-to-FP8 Converter | `rtl/activation/q12_to_fp8_e4m3.sv` | 0.5 | RTL-1 | VRF-2 | RTL-2 | P1.W4 |
| Token Embedding LUT | `rtl/engram/lookup_engine.sv` | 1 | RTL-3 | VRF-3 | RTL-2 | P2.W13 |
| SRAM Cache | `rtl/engram/sram_cache.sv` | 0.5 | RTL-3 | VRF-3 | RTL-1 | P1.W5 |
| Hash Unit | `rtl/engram/hash_unit.sv` | 0.5 | RTL-3 | VRF-3 | RTL-1 | P1.W5 |
| MHC Mixer | `rtl/layer/mhc_mixer.sv` | 0.5 | RTL-3 | VRF-3 | RTL-2 | P2.W14 |
| MTP Head | `rtl/head/mtp_head.sv` | 1.5 | RTL-3 | VRF-3 | RTL-1 | P4.W31 |
| MTP Verify | `rtl/head/mtp_verify.sv` | 0.5 | RTL-3 | VRF-3 | RTL-1 | P4.W31 |
| PCIe DMA Wrapper | (in `top_master.sv`) | 1 | RTL-4 | VRF-3 | SW-2 | P1.W4 |
| ILA / Debug | `rtl/debug/uart_debug.sv` | 0.5 | RTL-4 | VRF-3 | RTL-1 | P1.W2 |
| DSP Stress Test | `rtl/debug/dsp_stress_test.sv` | 0.5 | RTL-1 | VRF-2 | RTL-2 | P1.W4 |
| HBM BW Test | `rtl/debug/hbm_bw_test.sv` | 0.5 | RTL-4 | VRF-2 | RTL-1 | P1.W2 |
| Integration | `hw/src/top_master.sv`, `hw/src/top_slave.sv`, `hw/src/full_stack/top_full_stack.sv`, `rtl/chip/chip_top.sv` | 8 | RTL-4 (lead), RTL-1..3 | VRF-1 | RTL-1 | P2-P4 |

### 4.3 Testbench Modules

| Testbench | File Path | PM | Owner | Validation Scope |
|-----------|----------|-----|-------|-----------------|
| fp4 MAC | `rtl/sim/tb_fp4_mac.sv` | 0.5 | VRF-2 | Single MAC: all fp4 x fp8 value pairs |
| fp4 Systolic 2D | `rtl/sim/tb_fp4_systolic_2d.sv` | 0.5 | VRF-2 | 2D systolic array: matrix multiply correctness |
| fp4 GEMM Engine | `rtl/sim/tb_fp4_gemm_engine.sv` | 1 | VRF-2 | GEMM engine: tiled matmul with HBM weight streaming |
| fp4 Prefill Engine | `rtl/sim/tb_fp4_prefill_engine.sv` | 0.5 | VRF-2 | Prefill engine: chunked prefill with sparse attention |
| fp4 Scale Reader | `rtl/sim/tb_fp4_scale_reader.sv` | 0.3 | VRF-2 | Scale lookup: all group_size=16 edge cases |
| Expert FFN Engine | `rtl/sim/tb_expert_ffn_engine.sv` | 1 | VRF-3 | Expert FFN: fp4 gate/up/down with golden comparison |
| Router Top-K | `rtl/sim/tb_router_topk.sv` | 0.5 | VRF-3 | Router: top-6 selection correctness; ties; overflow |
| MLA Attention v2 | `rtl/sim/tb_mla_attention_v2.sv` | 2 | VRF-1 | MLA attention: full Q/K/V/O path with KV cache |
| MLA QKV | `rtl/sim/tb_mla_qkv.sv` | 1 | VRF-1 | QKV projection: low-rank compress/decompress |
| KV DMA | `rtl/sim/tb_kv_dma.sv` | 1 | VRF-3 | KV DMA: HBM read/write for KV cache migration |
| RMSNorm | `rtl/sim/tb_rms_norm.sv` | 0.3 | VRF-2 | RMSNorm: numerical stability, fp8 conversion |
| SiLU LUT | `rtl/sim/tb_silu_q12_lut.sv` | 0.3 | VRF-2 | SiLU LUT: all Q12 value range |
| Full Transformer Layer | `rtl/sim/tb_full_transformer_layer.sv` | 2 | VRF-1 | Layer integration: all 10 stages end-to-end |
| C2C Ring | `rtl/sim/tb_c2c_ring.sv` | 1 | VRF-3 | C2C ring: 4-chip dual ring; failover; BER injection |
| Chip 12-Layer | `rtl/sim/tb_chip_12layer.sv` | 2 | VRF-1 | Chip-level: 12 layers with pipeline forward |
| Cluster 384-Layer | `rtl/sim/tb_cluster_384.sv` | 2 | VRF-1 | Full cluster: 32 chips, 384 layers, MoE dispatch |
| MHC Mixer | `rtl/sim/tb_mhc_mixer.sv` | 0.5 | VRF-3 | Multi-Head Concat Mixer |
| MTP Head | `rtl/sim/tb_mtp_head.sv` | 1 | VRF-3 | MTP: multi-token prediction + verify |
| Lookup Engine | `rtl/sim/tb_lookup_engine.sv` | 0.5 | VRF-3 | Token embedding LUT |

---

## 5. Software Work Breakdown

### 5.1 Python Simulation Stack (`scripts/`)

| Component | Directory / Key Files | Owner | Description |
|-----------|----------------------|-------|-------------|
| Hardware Config | `scripts/fpga_arch/config.py` | SW-1, SW-2 | Single source of truth: all hardware parameters, model dimensions, DSP/HBM specs, performance targets |
| FPGA Chip Model | `scripts/fpga_arch/chip.py` | SW-1 | Per-chip model: assigned layers, experts, HBM layout, DSP utilization |
| FPGA Cluster Model | `scripts/fpga_arch/cluster.py` | SW-1 | 32-chip cluster topology: chip-to-layer mapping, expert placement, C2C routing table |
| Pipeline Engine | `scripts/fpga_arch/pipeline.py` | SW-1 | 10-stage pipeline: per-stage timing, batch/prefill scaling, C2C contention, expert hit enumeration |
| Interconnect Model | `scripts/fpga_arch/interconnect.py` | SW-1 | C2C Dual Ring + PCIe P2P: link contention, parallel same-card, serial cross-card |
| Expert Popularity | `scripts/fpga_arch/expert_popularity.py` | SW-2 | Zipf model: expert access distribution, hot expert detection, replica benefit analysis |
| vLLM Scheduler | `scripts/vllm_serve/scheduler.py` | SW-1 | ContinuousBatchingScheduler: prefill/decode interleaving, request queuing, batch formation |
| KV Cache Manager | `scripts/vllm_serve/kv_cache.py` | SW-1 | PagedAttention: block allocation, eviction, multi-session isolation |
| Model Runner | `scripts/vllm_serve/model_runner.py` | SW-1 | Bridge from vLLM scheduler to FPGA pipeline: batch submission, result collection |
| API Server | `scripts/vllm_serve/api_server.py` | SW-3 | Poisson request generator + OpenAI-compatible endpoint (for simulation validation) |
| Weight Layout Compiler | `scripts/vllm_serve/weight_layout.py` | SW-2 | Per-chip weight assignment: expert partitioning, HBM address map, replica placement |
| Serving Sim (E2E) | `scripts/run_serving.py` | SW-1 | Primary entry point: event-driven end-to-end serving simulation with full metrics |
| CPU Prefill Coordinator | `scripts/prefill/coordinator.py` | SW-2 | CPU-FPGA hybrid prefill: task splitting, PCIe orchestration |
| CPU Prefill Scheduler | `scripts/prefill/scheduler.py` | SW-2 | Chunked prefill scheduling on CPU: AMX/AVX-512 dispatch |
| vLLM Prefill Adapter | `scripts/prefill/vllm_prefill.py` | SW-3 | vLLM model weights -> CPU prefill engine |

### 5.2 Numerical Simulation (`scripts/simulation/`)

| Component | Key Files | Owner | Description |
|-----------|----------|-------|-------------|
| FP4 Utilities | `fp4_utils.py` | SW-2 | fp4 quantization/dequantization, scale group management |
| MLA Attention | `mla_attention.py` | SW-2 | NumPy reference: low-rank Q/K/V, RoPE, sliding window attention |
| MoE Router | `moe_router.py` | SW-2 | NumPy reference: top-k gating, expert FFN |
| Transformer Layer | `transformer_layer.py` | SW-2 | NumPy reference: full transformer layer with golden output |
| Experiment 1 | `experiment_1_fp4_precision.py` | VRF-2 | fp4 precision: MAC error distribution vs PyTorch |
| Experiment 2 | `experiment_2_hbm_bandwidth.py` | VRF-2 | HBM bandwidth simulation: MoE random access patterns |
| Experiment 3 | `experiment_3_layer_latency.py` | VRF-2 | Layer latency: per-stage breakdown and bottleneck analysis |
| Golden Vector Gen | `gen_tb_vectors.py` | VRF-2 | 15 MAC golden tests: test vector generation |
| FFN Golden Gen | `gen_ffn_tb_vectors.py` | VRF-3 | Expert FFN golden cases |
| Layer Golden Gen | `gen_layer_golden.py` | VRF-1 | Full layer golden cases |
| Verify fp4 MAC | `verify_fp4_mac_stages.py` | VRF-2 | Per-stage MAC verification vs golden |
| Module Smoke | `scripts/run_module_smoke.py` | VRF-1 | Module-level smoke tests: all RTL modules |
| All Validations | `scripts/run_all_validations.py` | VRF-1 | Full validation suite runner |

### 5.3 C Reference Implementation (`c_ref/`)

| Component | File | Owner | Description |
|-----------|------|-------|-------------|
| FP4 Reference | `c_ref/src/fp4_ref.c` | SW-2 | fp4 quant/dequant/GEMM in C for hardware validation baseline |
| FP4 Test Suite | `c_ref/tests/test_fp4_ref.c` | SW-2 | Unit tests for fp4 reference implementation |
| CPU Prefill | `c_ref/prefill/cpu_prefill.c` | SW-3 | CPU prefill reference: AMX/AVX-512 accelerated attention |
| Weight Preloader | `c_ref/prefill/weight_preloader.c` | SW-2 | FPGA weight loading: PCIe DMA, HBM address programming |

### 5.4 Hardware Test Infrastructure (`hw/`)

| Component | File | Owner | Description |
|-----------|------|-------|-------------|
| Golden Test Runner | `hw/scripts/run_golden_tests.py` | VRF-2, SW-2 | Python script: reads golden vectors, programs FPGA via PCIe, reads back results |
| Quartus Project (bring-up) | `hw/quartus/bringup/` | RTL-1, RTL-4 | Quartus project for single-board bring-up with small dimensions |
| Quartus Project (master) | `hw/quartus/master/` | RTL-4 | Quartus project for PCIe master chip |
| Quartus Project (slave) | `hw/quartus/slave/` | RTL-4 | Quartus project for PCIe slave chips |
| Quartus Project (full stack) | `hw/quartus/full_stack/` | RTL-4 | Quartus project for full production stack |
| Quartus Project (HBM char) | `hw/quartus/hbm_char/` | RTL-4 | HBM characterization project |
| Quartus Project (DSP char) | `hw/quartus/dsp_char/` | RTL-1 | DSP characterization project |
| Quartus Project (PCIe test) | `hw/quartus/pcie_test/` | RTL-4 | PCIe test project |
| Quartus Project (C2C test) | `hw/quartus/c2c_test/` | RTL-4 | C2C ring test project |
| Common Modules QSF | `hw/quartus/common/common_modules.qsf` | RTL-1 | Common RTL file list shared across all Quartus projects |
| Signal Tap Config | `hw/scripts/debug.stp` | VRF-2 | Signal Tap configuration for key probe points |

---

## 6. Critical Path Analysis and Risk Register

### 6.1 Critical Path Schedule

The project's minimum duration is determined by the longest dependency chain:

```
fp4 MAC RTL (2 PM)
  -> fp4 Systolic Cell (1 PM)
    -> fp4 Systolic 2D (2 PM)
      -> fp4 GEMM Engine (2.5 PM)
        -> full_transformer_layer (1 PM)
          -> MLA Attention v2 (5 PM) -- PARALLEL with MoE Router (1.5 PM)
            -> chip_top integration (2 PM)
              -> top_master / top_slave (2 PM)
                -> full_stack integration (3 PM)
                  -> C2C ring bring-up (3 PM)
                    -> multi-chip pipeline (3 PM)
                      -> cluster deployment (3 PM)

Total critical path: ~31 PM on the RTL side
With 4 RTL engineers: ~7.75 months (matches the 8-month hardware delivery timeline)
```

The **fp4 Systolic Array** (10 PM), **MLA Attention Pipeline** (12 PM), and **Integration** (8 PM) are the three critical path blocks. Any delay on these directly pushes the final delivery date.

### 6.2 Risk Register

| Risk ID | Risk Description | Probability | Impact | Mitigation | Owner |
|---------|-----------------|-------------|--------|------------|-------|
| R1 | fp4 MAC accuracy does not match Python golden on hardware due to DSP rounding differences | Medium | High | Early hardware test (Phase 1 W3); fallback to fp8 weights (5.54 TMACS per chip instead of 11.07); documented as "known DSP behavior" if within 2 ULP | RTL-1, VRF-2 |
| R2 | HBM2e delivers < 550 GB/s in MoE random access pattern | Medium | High | Bank conflict analysis (Phase 1 W4); try interleaved/hashed layouts; if < 400 GB/s: reassess architecture (reduce expert count, increase per-chip HBM, accept lower perf) | RTL-4, VRF-2 |
| R3 | C2C ring BER > 1e-15 or link fails to train at target speed | Medium | High | Internal loopback first (PMA), then external; check refclk and F-Tile placement; fallback to lower line rate with FEC | RTL-4, VRF-3 |
| R4 | Quartus full compilation fails to close timing at 450 MHz DSP clock | Medium | High | Pipeline depth increase; register balancing; if still failing: reduce target to 350-400 MHz (throughput degrades proportionally) | RTL-1, RTL-4 |
| R5 | Cross-node RoCE RDMA latency exceeds 10 us budget | Medium | Medium | Buffer for additional C2C hop; tune RDMA completion queue polling; fallback: relax cross-node dispatch budget to 15 us (accepts ~5% throughput loss) | RTL-4, SW-2 |
| R6 | Single developer bottleneck: RTL-1 (fp4 systolic array) is on the critical path for all subsequent modules | Medium | High | RTL-2 (secondary: fp4 GEMM) cross-trained; if RTL-1 blocked by any reason, RTL-2 can take over systolic array while RTL-1 focuses on debugging | RTL-1, RTL-2 |
| R7 | Quartus compilation time (4-6 hours on c6i.16xlarge) limits iteration speed | High | Medium | Pre-synthesis simulation in Icarus (30s for bring-up dims); incremental compilation when possible; night-time compilation queue; reserve 2 cloud instances | RTL-1, RTL-4 |
| R8 | FPGA board hardware failure or shipping delay | Low | Critical | Order 2x boards initially (one primary, one cold spare); verify BSP Golden Top before custom RTL; 8-12 week lead time for 8x chip order in Phase 2 | RTL-1, SW-1 |
| R9 | PCIe 5.0 link fails to train at Gen5 x16 (cable, BIOS, or signal integrity issue) | Low | High | Validate with BSP Golden Top first; check MCIO cable (Amphenol HMC74-0631 sold separately); lspci verification; fallback to Gen4 x16 (28 GB/s still acceptable) | RTL-4, SW-2 |
| R10 | vLLM scheduler integration: Python simulation predicts throughput that hardware cannot achieve | Medium | Medium | Early calibration: `PipelineEngine.calibrate()` compares analytical model vs detailed sim (Phase 2); update K_PIPELINE constant from hardware measurements | SW-1, VRF-1 |
| R11 | Weight loading time dominates token latency at low batch sizes | Low | Medium | Double-buffered SRAM weights; weight pre-fetch pipeline; if still bottleneck: accept higher latency at B=1 (reasonable trade-off) | RTL-1, SW-2 |
| R12 | CPU Prefill (P2): AMX/AVX-512 throughput insufficient for target TTFT | Medium | Medium | CPU benchmark early (Phase 1 W3); if insufficient: upgrade CPU (Xeon GNR 6980P / EPYC Turin 9755); fallback to FPGA-only prefill (accepts higher TTFT) | SW-3 |
| R13 | MTP (Multi-Token Prediction) acceptance rate < 85%, reducing effective throughput gain | Medium | Low | Tune MTP verification threshold; if < 70%: disable MTP and accept single-token decode throughput | RTL-3, VRF-3 |
| R14 | Hot expert replication: HBM capacity insufficient for replica placement | Low | Low | Limit replication to top-4 experts; use SRAM for hottest 2; if still insufficient: skip replication (small throughput impact) | RTL-3, SW-1 |

### 6.3 Risk Heat Map

```
Impact
  High   | R9  | R1,R2,R3,R4,R6 |
         |     |                 |
  Medium |     | R5,R10,R12      | R7
         |     |                 |
  Low    |     | R11,R13,R14     |
         |_____|_________________|________
              Low    Medium       High
                    Probability
```

---

## 7. AI Tool Strategy per Role

### 7.1 Guiding Principles

Every engineer has access to AI coding tools (GitHub Copilot, Cursor, Claude, or equivalent). These tools are force multipliers but require role-specific usage patterns to maximize benefit and avoid common failure modes (hallucinated signals, incorrect timing assumptions, plausible-but-wrong SystemVerilog).

### 7.2 RTL Engineer AI Strategy

**What AI does well for RTL:**
- Generate repetitive boilerplate: state machine `case` blocks, register declarations, AXI handshake templates, pipeline stage registers
- Provide correct SystemVerilog syntax for common patterns: `always_ff`, `always_comb`, `generate`, parameterized width calculations
- Suggest signal naming conventions and port list organization
- Auto-complete testbench stimulus (for-loops, clock generation, reset sequences)
- Suggest timing constraint templates (`.sdc` files)

**What AI does poorly for RTL (human must verify):**
- Correct DSP block inference for Intel Agilex (AI often generates Xilinx-style DSP attributes)
- Timing-critical pipeline balancing (inserting the right number of register stages)
- Clock domain crossing logic (AI frequently misses proper synchronization)
- Reset domain design (async reset vs sync reset implications)
- Quartus-specific pragmas and synthesis attributes

**Per-task AI usage guidelines for RTL engineers:**

| Task | AI Usage | Human Must |
|------|----------|------------|
| Write new module (e.g., `fp4_systolic_cell.sv`) | Generate skeleton with parameterized ports, pipeline stage template | Verify DSP inference, pipeline depth, Quartus-specific attributes |
| Modify existing module | AI suggests diffs; human applies selectively | Check all downstream port connections; re-run module testbench |
| Debug failing testbench | AI can analyze waveform diff and suggest likely causes | Human must root-cause in Signal Tap / Icarus waveform viewer |
| Write timing constraints | AI generates `.sdc` template from port list | Human verifies all clock groups, false paths, multicycle exceptions |
| Pre-synthesis cleanup | AI can identify unused signals, width mismatches, sensitivity list issues | Human reviews each lint warning; Quartus might accept code Icarus rejects |
| Integration (chip_top, top_master) | AI can help with port mapping tables and instance arrays | Human must verify all connectivity, especially multi-chip C2C paths |

**Specific AI prompts for RTL workflow:**

```
# Template for new module creation
"Write a SystemVerilog module named [module_name] with the following ports:
 [port list]. Use a 3-stage pipeline. Target: Intel Agilex 7 M-Series (Quartus Prime Pro 24.3).
 Prefer synchronous reset. Include parameterized WIDTH and LANES."

# Template for timing analysis
"Analyze this timing report and suggest pipeline register insertion points to
 fix the failing paths. Target fmax: 450 MHz. Device: AGM 039-F."
```

### 7.3 Verification Engineer AI Strategy

**What AI does well for verification:**
- Generate Python golden model from SystemVerilog interface definition
- Create test vector generators covering edge cases (zero, max, min, NaN-equivalent, overflow)
- Write data comparison scripts (read FPGA result, compare to golden, report ULP difference)
- Generate Signal Tap trigger configurations from module port lists
- Auto-documentation of test coverage

**What AI does poorly for verification:**
- Understanding the numerical semantics of fp4 (microscaling block float)
- Identifying corner cases unique to FPGA hardware (HBM bank conflicts, C2C link contention)
- Generating correct Cocotb testbenches without hallucinating VPI calls
- Understanding Intel-specific test infrastructure (Signal Tap, System Console)

**Per-task AI usage guidelines for verification engineers:**

| Task | AI Usage | Human Must |
|------|----------|------------|
| Write testbench skeleton (`tb_*.sv`) | AI generates clock, reset, instantiation, basic stimulus | Human designs test cases and edge conditions |
| Generate golden test vectors | AI generates all-values-sweep Python script | Human validates fp4 encoding correctness; marks known-DSP-behavior edge cases |
| Write result comparison script | AI generates pass/fail logic with tolerance thresholds | Human sets correct ULP tolerance per test (1 for MAC, 4 for layer) |
| Analyze failing test wave dump | AI can parse VCD/FSDB and suggest first-divergence point | Human root-causes in context of pipeline timing |
| Coverage report generation | AI compiles verification metrics from test logs | Human assesses coverage gaps and writes additional tests |

### 7.4 C/Software Engineer AI Strategy

**What AI does well for software:**
- Python: numpy operations, dataclass definitions, type hints, argparse setup
- C: memory management patterns, PCIe BAR mapping boilerplate, DMA descriptor chains
- Performance analysis: matplotlib plotting, pandas data aggregation, throughput/latency calculations
- API server: FastAPI/Flask endpoint generation, OpenAPI schema
- Documentation: docstrings, README generation, API reference

**What AI does poorly for software:**
- PCIe driver code (hardware-specific register maps, DMA engine programming sequences)
- Intel FPGA-specific software infrastructure (OFS driver integration, MMIO register maps)
- vLLM internal APIs (scheduler internals, block manager, model runner hooks)
- Performance-critical CPU prefill code (AMX tile configuration, cache line alignment, NUMA awareness)

**Per-task AI usage guidelines for software engineers:**

| Task | AI Usage | Human Must |
|------|----------|------------|
| Python pipeline model (`pipeline.py`) | AI can refactor, add type hints, generate docstrings | Human defines the timing equations and physics models |
| vLLM scheduler integration | AI can suggest compatible API calls based on vLLM source | Human verifies against actual vLLM version; tests with real model weights |
| C reference implementation | AI can generate fp4 quant/dequant loops and test cases | Human verifies numerical correctness against Python golden |
| CPU prefill optimization | AI can suggest SIMD intrinsics (AVX-512, AMX) | Human benchmarks on actual hardware; tunes tile sizes and cache blocking |
| API server (`api_server.py`) | AI can generate OpenAI-compatible endpoint structure | Human handles request routing, error handling, and hardware bridge |

### 7.5 AI-Assisted Design Review Checklist

Before any code review, the author should run these AI-assisted checks:

1. **RTL:** Ask AI to identify missing `default` cases in `case` statements, unconnected ports, and width mismatches
2. **RTL:** Ask AI to compare signal naming against the project convention (`LPU_` prefix for package constants, lowercase `_n` suffix for active-low)
3. **Verification:** Ask AI to list all module ports and confirm each is driven/observed in testbench
4. **Software:** Ask AI to check for missing type hints, incorrect numpy dtype usage, and unhandled None returns
5. **All:** Ask AI to write a 3-sentence summary of what the module does -- if the AI gets it wrong, the code is insufficiently commented

---

## 8. Communication Plan

### 8.1 Meeting Cadence

The team is distributed across time zones. All meetings are designed for maximum time zone overlap (targeting a 4-hour window).

| Meeting | Frequency | Duration | Participants | Agenda |
|---------|-----------|----------|-------------|--------|
| **Daily Standup** | Daily | 15 min | All 10 | Async preferred (Slack/Teams thread); sync only if critical blocker |
| **RTL Sync** | Mon/Wed/Fri | 30 min | RTL-1..4, VRF-1 | Module integration issues, timing closure status, Quartus compilation results |
| **Verification Sync** | Tue/Thu | 30 min | VRF-1..3, RTL-1 (optional) | Test coverage, failing tests, golden model updates |
| **Software Sync** | Mon/Thu | 30 min | SW-1..3 | Serving stack progress, integration blockers, performance benchmarks |
| **Cross-Functional Sync** | Weekly (Fri) | 60 min | All 10 | RTL <-> Verification <-> Software alignment; dependency tracking; risk review |
| **Phase Gate Review** | End of each phase | 90 min | All 10 + stakeholders | Go/No-Go decision; phase report; next phase resource allocation |
| **Architecture Review** | Ad-hoc (before major RTL commits) | 60 min | RTL-1..4 | New module interface definition; cross-module timing contracts |
| **1-on-1** | Biweekly | 30 min | Lead + each engineer | Career development, blockers, well-being |

### 8.2 Communication Channels

| Channel | Tool | Purpose | Expected Response Time |
|---------|------|---------|----------------------|
| Daily engineering discussion | Slack/Teams `#fpga-lpu-eng` | Module-level questions, Compilation results, Test failures | < 4 hours (within working hours) |
| Urgent blocker | Slack/Teams `#fpga-lpu-urgent` + @channel | Hardware failure, Critical bug blocking multiple engineers, Quartus license issue | < 1 hour |
| Design decisions | GitHub Issues on project repo | Architecture decisions, Interface changes, Module API proposals | < 24 hours (with written proposal) |
| Code review | GitHub Pull Requests | All RTL/software changes | < 24 hours (reviewer must approve or request changes) |
| Documentation | `docs/` in repo | Design docs, phase reports, go/no-go decisions | Updated within 48 hours of decision |
| Simulation results | `hw/test_vectors/` + CI artifacts | Golden vector generation, test results, coverage reports | Updated per CI run |
| Meeting notes | Shared drive (Google Docs / Notion) | All sync meeting minutes | Within 4 hours of meeting |

### 8.3 Distributed Team Best Practices

1. **Async-first culture:** Default to written communication (Slack, GitHub Issues, design docs). Schedule synchronous meetings only when async discussion has stalled.
2. **Decision records:** Every architecture decision goes into a GitHub Issue with the tag `design-decision`. The issue template requires: context, options considered, decision, rationale, and reviewers.
3. **Overlap hours:** Identify a 4-hour window where all time zones overlap. Schedule all-hands meetings within this window. Individual team syncs can be more flexible.
4. **Handoff protocol:** When an engineer in TZ-A finishes their day, they write a brief "handoff" message in `#fpga-lpu-eng` with: what they completed, what's in progress, any blockers for the next TZ.
5. **No-meeting Wednesday:** Blocked for deep work. No recurring meetings scheduled on Wednesday.
6. **Recording policy:** All cross-functional syncs are recorded (with consent). Engineers in non-overlapping TZs watch async and comment in the meeting doc.

### 8.4 Documentation Standards

| Document Type | Location | Template | Update Frequency |
|---------------|----------|----------|-----------------|
| Module Design Doc | `docs/modules/MODULE_NAME.md` | Interface, block diagram, state machine, timing contract, test plan | Created before RTL coding; updated after significant changes |
| Phase Report | `docs/reports/phase_N_report.md` | Objectives, deliverables, test results, issues, go/no-go recommendation | End of each phase |
| Bug Report | GitHub Issues with `bug` label | Repro steps, expected vs actual, Signal Tap / wave trace, root cause hypothesis | As discovered |
| Performance Benchmark | `docs/benchmarks/YYYY-MM-DD_benchmark.md` | Configuration, metrics, comparison to baseline, analysis | Every major benchmark run |
| Runbook | `docs/runbook/` | Startup sequence, shutdown sequence, common failure modes, recovery procedures | Created in Phase 5; updated continuously |

---

## 9. Milestone Checklist

### 9.1 Phase 1 Milestones (Month 1-2)

- [ ] **M1.1** (W1): RTL code freeze (`phase1-freeze` git tag); Quartus project compiles without errors
- [ ] **M1.2** (W1): Python golden model frozen; all golden vectors committed to `hw/test_vectors/`
- [ ] **M1.3** (W1): BSP Golden Top design passes on development board (PCIe link up, HBM R/W, DDR5 access)
- [ ] **M1.4** (W2): UART TX sends "FPGA LPU boot" at 115200 baud; LED heartbeat blinks at 1 Hz
- [ ] **M1.5** (W2): HBM sequential write/read back correct; bandwidth measured
- [ ] **M1.6** (W3): fp4 MAC hardware test FSM operational: scale memory load, MAC test runner, result register
- [ ] **M1.7** (W4): **Go/No-Go #1 (MAC Precision):** 15/15 MAC golden tests pass; cosine >= 0.995
- [ ] **M1.8** (W4): Signal Tap configured with all key probe points for MAC pipeline
- [ ] **M1.9** (W5): `full_transformer_layer.sv` synthesizes and passes timing (bring-up dims)
- [ ] **M1.10** (W6): **Go/No-Go #2 (HBM Bandwidth):** MoE random BW >= 550 GB/s; dual-buffer overlap >= 80%
- [ ] **M1.11** (W7): **Go/No-Go #3 (Layer E2E):** C0 exact match, C1 within +/-4, latency <= 2x simulation
- [ ] **M1.12** (W8): Phase 1 report compiled; Go/No-Go decision meeting held
- [ ] **M1.13** (W8): 8x AGM 039-F chips ordered (for Phase 2); PCB schematic started

### 9.2 Phase 2 Milestones (Month 3-4)

- [ ] **M2.1** (W10): F-Tile SerDes trained on all 8 cards; BER < 1e-15 per link
- [ ] **M2.2** (W11): C2C ring 4-chip loopback functional; latency < 50 ns/hop
- [ ] **M2.3** (W12): RoCE v2 RDMA Write/Read functional between all 8 cards
- [ ] **M2.4** (W13): TP All-Reduce on 7168-dim vector < 2 us across 8 cards
- [ ] **M2.5** (W14): MoE Dispatch + Reduce functional: 0-hit case < 800 ns, 1-hit case < 250 ns
- [ ] **M2.6** (W15): Pipeline forward across all 32 chips: token flows chip 0 -> chip 31
- [ ] **M2.7** (W16): Full 15-layer inference producing correct outputs on all chips
- [ ] **M2.8** (W17): vLLM scheduler driving multi-chip pipeline with 10 concurrent requests
- [ ] **M2.9** (W18): **Go/No-Go #2 review:** decode >= 200 tok/s at B=8; PCIe DMA >= 28 GB/s

### 9.3 Phase 3 Milestones (Month 5-6)

- [ ] **M3.1** (W20): Dual ToR MLAG configured; traffic flows across both ToRs
- [ ] **M3.2** (W22): Cross-node RoCE RDMA: latency < 10 us; BW >= 80 Gbps
- [ ] **M3.3** (W23): ToR failover: recovery < 100 ms; < 1% packet loss during transition
- [ ] **M3.4** (W24): Cross-node MoE dispatch + combine: < 5 us per expert message
- [ ] **M3.5** (W26): Dual-node 30-layer pipeline end-to-end functional
- [ ] **M3.6** (W27): 1-hour continuous operation soak test passed
- [ ] **M3.7** (W28): Decode >= 500 tok/s at B=16; Prometheus/Grafana dashboard operational

### 9.4 Phase 4 Milestones (Month 7-8)

- [ ] **M4.1** (W30): All 4 nodes operational; chip-to-layer mapping verified for all 61 layers
- [ ] **M4.2** (W32): MTP head produces 2 draft tokens; acceptance rate benchmark
- [ ] **M4.3** (W34): 128K context: KV cache operates within HBM budget; attention correctness validated
- [ ] **M4.4** (W36): Multi-session concurrency: throughput scales to >= 80% of linear from 5-20 sessions
- [ ] **M4.5** (W37): 24-hour soak test passed; all monitoring alerts configured
- [ ] **M4.6** (W38): System benchmark: aggregate decode >= 500 tok/s at B=32

### 9.5 Phase 5 Milestones (Month 9-10)

- [ ] **M5.1** (W40): 512K context functional; 1M context with < 30% performance degradation
- [ ] **M5.2** (W42): Hot expert replication: hit rate improved by >= 15%; dispatch traffic reduced by >= 10%
- [ ] **M5.3** (W44): All injected faults recover automatically within 5 seconds
- [ ] **M5.4** (W46): Power per node <= 5.3 kW; chip junction temp < 85 C under full load
- [ ] **M5.5** (W48): All OpenAI API compatibility tests pass; streaming responses functional
- [ ] **M5.6** (W49): Production runbook complete; SRE team trained; handoff documentation finalized
- [ ] **M5.7** (W50): **Project completion review:** final performance report, TCO analysis, lessons learned

---

## 10. Go/No-Go Decision Gates

### 10.1 Go/No-Go #1 -- After Phase 1 (Month 2, Week 8)

**Purpose:** Validate that fp4 compute works correctly on real FPGA silicon. This is the single most important gate -- if fp4 MAC cannot match Python golden models, the entire architecture is invalid.

**Decision Criteria:**

| Gate | Condition | Verdict |
|------|-----------|---------|
| GG1-A: fp4 MAC Precision | 15/15 MAC golden tests PASS on hardware; fp4 cosine similarity >= 0.995 vs PyTorch | **GO** |
| GG1-A (WARN) | 13-14/15 pass; one test has rounding diff <= 2 ULP; root cause is DSP rounding artifact (not logic bug) | **WARN** -- Document as "known DSP behavior"; proceed with caution |
| GG1-A (STOP) | < 13/15 pass, or any mismatch > 2 ULP | **NO-GO** -- Debug with Signal Tap for 1 week; if unresolved, consider fp8 fallback or cancel |
| GG1-B: HBM Bandwidth | MoE random BW >= 550 GB/s; dual-buffer overlap >= 80% | **GO** |
| GG1-B (WARN) | 400-550 GB/s; overlap 60-80% | **WARN** -- Recalculate TCO with corrected numbers; still viable but with 20-30% lower throughput |
| GG1-B (STOP) | < 400 GB/s or overlap < 60% | **NO-GO** -- Architecture infeasible at current targets; reassess |
| GG1-C: Layer E2E | C0 golden case exact match; C1 within +/- 4 ULP; latency <= 2x Icarus simulation | **GO** |
| GG1-C (WARN) | C0 matches but C1 out of tolerance, or latency 2-3x simulation | **WARN** -- Extend Phase 1 by 1-2 weeks to debug |
| GG1-C (STOP) | C0 fails or latency > 3x simulation | **NO-GO** -- Hardware architecture flaw; re-evaluate |

**Overall Gate Decision:**

```
ALL GO:        Proceed to Phase 2 (order 8x chips, design 4-chip card PCB)
1-2 WARN:      Proceed with adjusted performance targets; update TCO model
ANY STOP:      Halt; re-evaluate architecture; consider fp8 fallback or project cancellation
```

**Budget Gate:** Phase 2 requires ~3M RMB for 8x AGM 039-F chips + PCB prototype. Only proceed if all three sub-gates are GO or WARN with acceptable performance impact.

---

### 10.2 Go/No-Go #2 -- After Phase 2 (Month 4, Week 18)

**Purpose:** Validate that the multi-chip architecture works at scale. This gate confirms that the C2C ring, TP All-Reduce, and MoE dispatch operate correctly across 32 chips before committing to the full 4-node deployment.

**Decision Criteria:**

| Gate | Condition | Verdict |
|------|-----------|---------|
| GG2-A: C2C Ring | BER < 1e-15 per link; single-hop latency < 50 ns; dual-ring failover functional | **GO** |
| GG2-A (WARN) | BER < 1e-12; latency < 100 ns; failover < 1 second | **WARN** -- Add FEC or retry for critical messages |
| GG2-A (STOP) | BER >= 1e-12 or link fails to train | **NO-GO** -- Architecture blocked |
| GG2-B: MoE Dispatch | 0-hit case < 800 ns; 1-hit case < 250 ns; 2+-hit: local bypass | **GO** |
| GG2-B (WARN) | 0-hit case < 1.5 us; 1-hit case < 500 ns | **WARN** -- Throughput 10-20% lower than modelled |
| GG2-B (STOP) | MoE dispatch produces incorrect expert results | **NO-GO** -- Debug routing logic |
| GG2-C: Throughput | Full 15-layer decode >= 200 tok/s at B=8 | **GO** |
| GG2-C (WARN) | 100-200 tok/s at B=8 | **WARN** -- Recalculate full-cluster throughput projection |
| GG2-C (STOP) | < 100 tok/s or pipeline hangs | **NO-GO** -- Fundamental bottleneck; investigate root cause |

**Overall Gate Decision:**

```
ALL GO:        Order remaining chips and servers for 4-node deployment (Phases 3-4)
1-2 WARN:      Proceed but adjust full-cluster throughput projection down proportionally
ANY STOP:      Pause; root-cause investigation before committing to cross-node scale-out
```

**Budget Gate:** Full cluster requires ~12M RMB for 24 servers + 96 chips + network infrastructure. Only proceed if Phase 2 throughput meets minimum threshold (> 100 tok/s at B=8) and projected full-cluster throughput meets business requirements.

---

### 10.3 Go/No-Go #3 -- After Phase 4 (Month 8, Week 38)

**Purpose:** Final performance validation before production hardening. This gate confirms the system meets its core performance target (~14,000 tok/s decode at high concurrency) and is ready for production optimization.

**Decision Criteria:**

| Gate | Condition | Verdict |
|------|-----------|---------|
| GG3-A: Full Pipeline | All 61 layers producing correct outputs; chip-to-layer mapping verified | **GO** |
| GG3-A (STOP) | Any layer produces incorrect output; chip assignment errors | **NO-GO** -- Must fix before proceeding |
| GG3-B: Throughput | Aggregate decode >= 500 tok/s at B=32 (full system) | **GO** |
| GG3-B (WARN) | 300-500 tok/s at B=32 | **WARN** -- Proceed with optimization focus in Phase 5 |
| GG3-B (STOP) | < 300 tok/s | **NO-GO** -- Significant architecture issue; full investigation |
| GG3-C: Stability | 24-hour continuous operation without crash, hang, or silent data corruption | **GO** |
| GG3-C (STOP) | Any crash, hang, or data corruption within 24 hours | **NO-GO** -- Must fix stability issues before production |

---

## Appendix A: Key File Path Reference

### RTL Source Files (`rtl/`)

```
rtl/
  include/
    lpu_config.svh              -- Global config: production vs bring-up parameters
  dsp/
    fp4_mac.sv                  -- fp4 x fp8 scale-aware MAC (3-stage pipeline)
    fp4_scale_reader.sv         -- Group_size=16 scale lookup
    fp4_systolic_cell.sv        -- Single systolic cell
    fp4_systolic_2d.sv          -- 2D systolic array (LANES x M_ROWS)
    fp4_gemm_engine.sv          -- Tiled GEMM with HBM weight streaming
    fp4_prefill_engine.sv       -- Chunked prefill with sparse attention
  attention/
    mla_attention_v2.sv         -- MLA attention: full Q/K/V/O path
    mla_qkv_proj.sv             -- Q/K/V low-rank projections
    mla_rope.sv                 -- Decoupled RoPE
    mla_kv_cache.sv             -- KV cache: banked SRAM with eviction
  moe/
    router_topk.sv              -- Top-K expert selection
    expert_ffn_engine_fp4_down.sv -- Expert FFN: gate/up in fp8, down in fp4
  activation/
    rms_norm.sv                 -- RMSNorm: fp32 accumulator, fp8 output
    silu_q12_lut.sv             -- SiLU LUT: Q12 signed quantization
    q12_to_fp8_e4m3.sv          -- Q12 -> FP8 E4M3 converter
  layer/
    full_transformer_layer.sv   -- Full layer: 10-stage pipeline
    layer_compute_engine.sv     -- Per-layer compute orchestration
    mhc_mixer.sv                -- Multi-Head Concat Mixer
  chip/
    chip_top.sv                 -- Single chip top: layers + C2C + PCIe
    kv_dma_engine.sv            -- KV DMA engine: HBM <-> SRAM migration
    kv_dma_bridge.sv            -- KV DMA bridge: PCIe <-> C2C routing
  head/
    mtp_head.sv                 -- Multi-Token Prediction head
    mtp_verify.sv               -- MTP acceptance verification
  engram/
    lookup_engine.sv            -- Token embedding lookup table
    sram_cache.sv               -- SRAM cache for hot embeddings
    hash_unit.sv                -- Hash-based embedding lookup
  debug/
    uart_debug.sv               -- UART debug console (115200 8N1)
    dsp_stress_test.sv          -- DSP stress test (sweep + max toggle)
    hbm_bw_test.sv              -- HBM bandwidth measurement
  sim/                          -- Testbenches (see Section 4.3)
    tb_*.sv                     -- Per-module testbenches
  legacy/                       -- Deprecated modules (reference only)
    mla_attention.sv
    fp4_systolic_array.sv
    fp4_linear_engine.sv
    fp4_scaled_tile.sv
    fp4_systolic_tile.sv
    c2c_node.sv
    expert_ffn_engine.sv
```

### Hardware Integration Files (`hw/`)

```
hw/
  src/
    top.sv                      -- Original board-level wrapper (Phase 1)
    top_master.sv               -- PCIe master chip top (Phase 2+)
    top_slave.sv                -- PCIe slave chip top (Phase 2+)
    full_stack/
      top_full_stack.sv         -- Full production stack (Phase 4+)
    bringup/
      top_bringup.sv            -- Bring-up wrapper for single-board test
    hbm_char/
      top_hbm_char.sv           -- HBM characterization wrapper
    dsp_char/
      top_dsp_char.sv           -- DSP characterization wrapper
    pcie_test/
      top_pcie_test.sv          -- PCIe test wrapper
    c2c_test/
      top_c2c_test.sv           -- C2C ring test wrapper
  quartus/
    fpga_lpu.qpf / .qsf         -- Phase 1 Quartus project
    common/
      common_modules.qsf        -- Shared RTL file list
    bringup/                    -- Bring-up project (small dims)
    master/                     -- Master chip project
    slave/                      -- Slave chip project
    full_stack/                 -- Full production project
    hbm_char/                   -- HBM characterization project
    dsp_char/                   -- DSP characterization project
    pcie_test/                  -- PCIe test project
    c2c_test/                   -- C2C test project
  scripts/
    run_golden_tests.py         -- Hardware golden test runner
    debug.stp                   -- Signal Tap configuration
  constraints/
    fpga_lpu.sdc                -- Timing constraints (100/156/390 MHz)
  ip/
    hbm_sys/                    -- HBM2e QSYS subsystem
    pcie_sys/                   -- PCIe 5.0 QSYS subsystem
  test_vectors/                 -- Golden test vectors (*.hex)
```

### Software Files (`scripts/`, `c_ref/`)

```
scripts/
  fpga_arch/                    -- Hardware simulation
    config.py                   -- Unified constants
    chip.py                     -- FPGAChip model
    cluster.py                  -- FPGACluster (32-chip assembly)
    pipeline.py                 -- 10-stage PipelineEngine
    interconnect.py             -- C2C Dual Ring + PCIe P2P
    expert_popularity.py        -- Zipf model
  vllm_serve/                   -- Serving stack simulation
    scheduler.py                -- ContinuousBatchingScheduler
    kv_cache.py                 -- PagedAttention
    model_runner.py             -- Bridge to FPGA pipeline
    weight_layout.py            -- WeightLayoutCompiler
    api_server.py               -- API server + Poisson request gen
    types.py                    -- Shared types
    config.py                   -- Serving config
  prefill/                      -- CPU-FPGA hybrid prefill
    coordinator.py              -- Task splitting + PCIe orchestration
    scheduler.py                -- CPU chunked prefill scheduling
    vllm_prefill.py             -- vLLM weight adapter
  simulation/                   -- NumPy functional sim
    fp4_utils.py
    mla_attention.py
    moe_router.py
    transformer_layer.py
  run_serving.py                -- End-to-end serving simulation (primary entry)
  run_module_smoke.py           -- Module smoke tests
  run_all_validations.py        -- Full validation suite
  run_e2e_validation.py         -- End-to-end validation
c_ref/
  src/
    fp4_ref.c                   -- fp4 quant/dequant/GEMM in C
  tests/
    test_fp4_ref.c              -- C unit tests
  prefill/
    cpu_prefill.c               -- CPU prefill reference (AMX)
    weight_preloader.c          -- FPGA weight preloader
```

### Documentation (`docs/`)

```
docs/
  eng/
    01_project_plan.md          -- This document
  fpga_inference_cluster_proposal.md  -- Architecture proposal
  fpga_inference_cluster_proposal_en.md -- English version
  chip_decomposition.md         -- 32-chip topology and layer mapping
  bringup_checklist.md          -- Phase 1 bring-up checklist
  bringup_go_nogo.md            -- Go/No-Go test criteria
  bringup_strategy.md           -- Bring-up strategy
  why_fpga_is_optimal.md        -- FPGA vs GPU economics
  tco_per_million_tokens.md     -- TCO analysis
  ds_v4_arch_gap_analysis.md    -- DeepSeek V4 architecture gap analysis
  30card_topology_feasibility.md -- 30-card topology feasibility
  p0_p2_implementation_plan.md  -- P0-P2 optimization implementation
  fp4_c_rtl_implementation_report.md -- fp4 C/RTL implementation report
  simulation_validation_summary.md -- Simulation validation summary
  module_smoke_report.md        -- Module smoke test results
```

---

## Appendix B: Weekly Status Report Template

Each engineer submits a brief status update in `#fpga-lpu-eng` every Friday by end of their workday:

```
[Name] Week [N] Status

Completed:
- [Item 1: brief description, link to PR/issue if applicable]
- [Item 2]

In Progress:
- [Item 1: % complete, expected finish date]
- [Item 2]

Blockers:
- [Blocker description, who can help, ticket link if filed]

Next Week:
- [Top 3 priorities for next week]

Quartus compilation status (RTL team only):
- Project: [bringup/master/slave/full_stack]
- Synthesis: PASS / FAIL (timing met: YES / NO)
- Key violations: [list worst paths]
```

---

## Appendix C: Cloud Infrastructure Requirements

| Resource | Purpose | Spec | Quantity | Monthly Est. Cost |
|----------|---------|------|----------|------------------|
| AWS c6i.16xlarge | Quartus compilation | 64 vCPU, 128 GB RAM | 2 instances (shared) | ~$2,000 |
| AWS c6i.8xlarge | Icarus simulation farm | 32 vCPU, 64 GB RAM | 1 instance | ~$500 |
| Development FPGA board | Phase 1 hardware | DK-DEV-AGM039EA | 2 (1 primary + 1 cold spare) | ~$20,000 (one-time) |
| Quartus Prime Pro license | Synthesis + P&R | 24.3+ with Agilex 7 support | 2 seats | ~$8,000/yr |
| GitHub Team / Enterprise | Source control + CI | Private repos + Actions minutes | 1 org | ~$500/month |

---

*End of document. For questions, contact the project lead. Next review: Phase 1 Go/No-Go (Month 2, Week 8).*
