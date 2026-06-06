## 11.A Architecture Argument Chain: Rigorous Decomposition and Risk Audit

> The following decomposes the FPGA proposal's core claims into an 8-step logic chain.
> Each step is annotated with evidence strength,
> weak points, and unknown variables requiring empirical verification. This is not
> self-negation — it is finding the cracks in the proposal before opponents
> and customers ask.

### 11.A.1 Argument Chain

**Step 1: Decode is memory-bandwidth-bound, not compute-bound**

```
Claim: The LLM decode (B=1) bottleneck lies in HBM bandwidth, not compute.
       Therefore, high-compute, low-bandwidth GPU/NPU architectures are structurally
       unsuitable for decode.

Evidence (Strong):
  - H100 B=1 decode: loading 432 MB/layer weights ÷ 3.35 TB/s = 129 μs,
    Tensor Core computation only 2.6 μs → 98% of time waiting on HBM
  - Measured Tensor Core utilization ~2-3% (NVIDIA official profiler data)
  - Roofline model: decode operational intensity ≈ 0.002 FLOP/byte,
    far below H100's ridge point (~20 FLOP/byte)

Weak Points:
  A. At B>1 the bottleneck partially shifts toward compute. At B=8 GPU utilization
     rises to ~10-15%, but remains in the memory-bound region. Only at B>32 does it
     potentially enter compute-bound territory.
     For decode scenarios (autoregressive, 1 token per step), B is capped by session count.
  B. fp4 models have even lower operational intensity (fewer parameters, same compute),
     so the memory-bound degree is deeper → this weak point favors FPGA, not a risk.

Mitigation: None needed. Memory-bound is structural, unaffected by model architecture
           changes (unless model weights are extremely small, e.g., <100 MB —
           but that won't happen for large models).

Risk Level: 🟢 Low — laws of physics, won't change.
```

**Step 2: Native fp4 E2M1 computation is feasible and precision is sufficient**

```
Claim: DeepSeek V4 Pro uses fp4 E2M1 weights, and inference precision loss is acceptable.
       FPGA DSP can implement fp4×fp8 native MAC via LUT, without decompression steps.

Evidence (Moderate-Strong):
  - DeepSeek claims in the V4 Pro paper that fp4 inference precision is comparable to FP8
  - NVIDIA B200/GB200 already support FP4 Tensor Core in hardware → industry validates fp4 direction
  - FPGA 11 TMACs fp4 is based on DSP 48-bit MAC LUT configuration, not an estimate

Weak Points:
  A. DeepSeek V4 Pro's fp4 quality assessment is self-reported by the model developer,
     not independently verified. Precision degradation on specific benchmarks may not
     be fully disclosed.
  B. fp4 E2M1 has an extremely narrow representable range (exponent=2, mantissa=1,
     range=[0.5, 1.5, 3.0, 6.0]). Certain layers (e.g., Router, RMSNorm) may require
     FP8 → mixed precision increases RTL complexity.
  C. If future models (V5, V6) find fp4 insufficient and upgrade to fp6 or FP8,
     the bandwidth advantage shrinks from 2× to 1.33× or 1× → core argument is impacted.

Mitigation:
  - Reserve fp4/fp6/FP8 configurable precision in RTL (DSP supports multiple MAC widths)
  - Periodically verify precision via per-layer CRC32 checksum against FP32 reference
  - Closely track DeepSeek V5/V6 precision choices

Risk Level: 🟢 Low — fp4 is a deterministic industry trend.
           NVIDIA B200/GB200 hardware supports FP4, DeepSeek V4 has deployed fp4,
           AMD MI400 roadmap includes FP4. The industry chain is moving toward lower
           precision and won't turn back. Residual risk lies only in the evolution
           of the specific fp4 format (E2M1 vs E3M0 vs MXFP4).
```

**Step 3: MLA compresses KV Cache by 114×, making 32 GB HBM capacity viable**

```
Claim: DeepSeek V4 Pro's MLA (Multi-head Latent Attention) compresses the KV cache
      from 64 KB/token/layer to 576 B/token/layer, a compression ratio of ~114×.
      This is the prerequisite for FPGA 32 GB HBM not being a capacity bottleneck.

Evidence (Strong):
  - KV cache per layer per token: 512B (KV latent, FP8) + 64B (rope, FP8)
    = 576 B. These are publicly documented architecture parameters from the
    DeepSeek V4 Pro paper.
  - 32 GB HBM at 1M context: ~9.2 GB (16 layers/worst chip)
  - 2M context: ~18.4 GB, still within 32 GB (with weights + buffers ~23.4 GB)

Weak Points:
  A. MLA's KV latent requires decompression back to full K/V at decode time →
     additional computational overhead. The document claims this is hardened in
     the FPGA 6-stage pipeline, but post-hardening latency/area has not been
     actually verified in RTL.
  B. If future models abandon MLA (reverting to GQA or switching to other attention),
     KV cache inflates from 576 B → 64 KB/token/layer, a 114× explosion.
     32 GB HBM would only support ~8K context → the solution degrades to
     short-context-only.
  C. MLA-compressed KV latents may require higher precision (FP16) to guarantee
     long-context quality, partially offsetting compression gains.

Mitigation:
  - MLA is DeepSeek's core innovation; V4/V5 will very likely not abandon it
  - Even if reverting to GQA, the FPGA solution still works (only context cap lowers)
  - HBM+DDR configuration can offload partial KV cache to DDR
  - Track attention architecture evolution in Google Gemini, Anthropic Claude

Risk Level: 🟡 Medium — MLA is the cornerstone of the HBM capacity argument;
           departing from the DeepSeek ecosystem carries risk.
```

**Step 4: Weights distributed across 32 chips → bandwidth/layer 460 GB/s = 2.63× of 950PR**

```
Claim: 32-chip pipeline-parallel, each chip carries only 2 layers.
      Available HBM bandwidth per layer = 920 GB/s ÷ 2 = 460 GB/s.
      Compare with 950PR: 1,400 GB/s ÷ 8 = 175 GB/s → FPGA 2.63×.

Evidence (Moderate):
  - Arithmetic is correct: 920/2 = 460, 1400/8 = 175, 460/175 = 2.63
  - 32 chips × 920 GB/s = 29.4 TB/s aggregate bandwidth (physically correct)
  - Simulation shows aggregate throughput 5,800-8,500 tok/s

Weak Points:
  A. Two communication types must be distinguished, with different risk profiles:

     Pipeline forwarding (tokens passing sequentially between chips):
       - FPGA SERDES is a core competency (28 Gbps/lane, multi-lane aggregation)
       - Per-hop latency ~50-100ns (custom lightweight protocol, no network stack overhead)
       - 32 hops × 75ns ≈ 2.4μs → vs ~1.4ms/token, negligible (0.17%)
       → 🟢 This is not a risk.

     MoE All-to-All data movement (cross-chip dispatch hidden state + gather result):
       - Per token per layer hits ~6 experts, P(local)=17% → ~5 remote accesses
       - Each remote transfer ~16KB (8KB send + 8KB receive)
       - At single lane 28 Gbps = 3.5 GB/s: 16KB / 3.5 GB/s ≈ 4.6μs/access
       - Multi-lane aggregation can reduce to < 1μs/access
       - 5 remote × 1μs × 61 layers ≈ 305μs → ~22% of ~1.4ms
       → 🟡 This needs attention, but Hot Expert Replication (local hit 17%→70%)
          reduces remote count from 5→2, communication overhead from 22%→9%.

  B. The real risk is not at the SERDES physical layer, but in arbitration and congestion:
     - Multiple chips simultaneously sending expert requests to the same chip →
       arbitration latency
     - But MoE's power-law distribution means requests are dispersed (hot experts
       have replicas, cold experts have low access frequency) → congestion probability low
     - Arbitration latency needs to be measured on 4-8 chip hardware

  C. Communication comparison with 950PR: 950PR's HCCS interconnect has higher bandwidth
     (~2 TB/s), but with 8 chips each carrying 8 layers → higher probability of remote
     experts per layer (fewer chips → experts more dispersed → P(local) lower).
     FPGA's 32 chips mean fewer experts per chip (12→2), but higher expert density
     within a chip → P(local) is actually higher.
     → More chips ≠ worse communication; it depends on expert placement strategy.

Mitigation:
  - FPGA SERDES communication is proven technology (widely used in networking, HFT,
    signal processing)
  - Hot Expert Replication halves remote requests
  - ASIC phase merges 4 FPGA→1 ASIC: pipeline depth 32→8, remote hops further reduced
  - P0 hardware verification: 4-8 chip system measuring real All-to-All latency
    and arbitration behavior

Risk Level: 🟡 Medium — SERDES physical layer is reliable; risk is in MoE All-to-All
           arbitration congestion, which can be closed via Hot Replication + hardware
           verification.
```

**Step 5: SRAM deterministic weight cache eliminates 81.6% of HBM accesses**

```
Claim: Shared Expert + Attention + Router + RMSNorm (~9.2 MB resident total)
      permanently cached in SRAM, covering all HBM read requirements for 81.6% of layers.

Evidence (Moderate):
  - Weight sizes precisely calculable: 4.4 MB × 2 + 0.37 MB + 0.01 MB ≈ 9.2 MB
  - Agilex 7 M M20K BRAM total ~37.5 MB/chip → sufficient headroom
  - P(0 local hit) = 81.6% from Zipf(α=1.0) × 12 experts/chip Monte Carlo

Weak Points:
  A. 13.2 MB (including streaming buffers) needs verification of actual BRAM
     availability after Quartus synthesis (deducting KV cache manager, router FSM,
     HBM controller, etc. consumption).
  B. If expert distribution deviates from Zipf (α<1.0) → P(0 hit) may drop to 70-75%
     → more remote expert requests → more communication → K_PIPELINE increases.
  C. BRAM physical layout constraints may cause internal fragmentation where
     "some layers lack BRAM while others have idle BRAM."

Mitigation:
  - Sensitivity analysis with α=0.5~1.5 in simulation (at α=0.5, P=67%, still acceptable)
  - Check BRAM utilization after Quartus synthesis; adjust layer-to-BRAM mapping if needed

Risk Level: 🟡 Medium — Zipf assumption is reasonable but not absolute;
           BRAM allocation requires measured verification.
```

**Step 6: FPGA streaming architecture → DSP utilization ~50% at B=1 vs GPU ~2-3%**

```
Claim: FPGA's streaming architecture does not require large batches to hide latency.
       Therefore, DSP utilization remains high even at B=1.

Evidence (Weak-Moderate):
  - Theory: GPU requires warps to hide HBM latency (~200-400 cycles);
    FPGA streaming pipeline produces output every cycle once filled
  - Simulation gives ~50% DSP utilization (analytical model, §4.6.1)
  - Not a measured value from post-RTL synthesis

Weak Points:
  A. The premise of streaming architecture is that token arrival interval ≤ DSP
     processing time. At B=1, token interval ≈ pipeline traversal latency / 32.
     Highly sensitive to communication latency.
  B. When DSP is configured for fp4 MAC, actual utilization is limited by BRAM bandwidth.
     fp4 weight read bandwidth from BRAM may become the bottleneck before DSP.
  C. 50% DSP utilization vs GPU 2-3%: GPU's 2-3% is relative to 1,979 TFLOPS,
     absolute value still 40-60 TFLOPS. FPGA's 50% is relative to 11 TMACs (5.5 TMACs).
     FPGA's "efficiency" is because the raw compute is small. Throughput advantage
     comes from bandwidth (Step 4), not from DSP utilization.

Mitigation:
  - This claim is explanatory ("why large batches aren't needed"), not decisive
  - Even if DSP utilization drops to 30%, throughput is unaffected (bandwidth bottleneck,
    not compute bottleneck)

Risk Level: 🟢 Low — explanatory claim, poses no risk to throughput.
```

**Step 7: Economics — structural $/token advantage from architectural bandwidth efficiency**

```
Claim: FPGA $/token advantage is not a result of "cheaper hardware," but rather
      the economic projection of architectural bandwidth efficiency (§11.A.2).
      The ~83× gap in effective bandwidth utilization means:
        - For the same $1 hardware cost, FPGA produces more effective bandwidth
          in decode scenarios
        - $/token advantage is the inevitable economic corollary of the bandwidth
          efficiency gap, not a pricing strategy

Quantitative:
  FPGA $/M tokens = $5.9 (baseline) ~ $1.03 (10K-unit revised basis)
  vs 950PR $18-28 / H100 $12-20 (both actual market price basis)
  → Advantage magnitude 2-6×, same order of magnitude as the 8× effective bandwidth gap

Evidence (Moderate):
  - Hardware BOM: ¥1.33M (~$182K) at 100 units
  - Aggregate throughput: 5,800-8,500 tok/s from architecture bandwidth/layer 460 GB/s (Step 4)
  - Power: 5.3 kW × ¥0.8/kWh × 24×365 = ¥37K/year
  - 950PR actual price ¥2M (~$275K) from 2026.04 market data

Weak Points:
  A. Throughput below target → $/M linearly deteriorates. If actual throughput is
     half of baseline, $/M goes from $5.9 → $12, narrowing the gap with 950PR.
  B. RTL development cost ¥3.75M (5 people × 18 months) not included. Amortization
     impact significant at low volumes. However, AI-assisted RTL development
     (AI writes modules + testbench, humans do review + architecture decisions)
     can improve effective development speed by 3-5× and reduce bus factor.

  C. Maintenance cost: adapting to new models (V5, V6) requires modifying RTL +
     re-synthesis + re-verification; software-based solutions have far lower
     adaptation costs.
  D. Intel FPGA may discontinue Agilex 7 M within 2-3 years, supply interruption →
     proposal stranded.

Mitigation:
  - ASIC phase inherits bandwidth efficiency from FPGA architecture verification,
    hardware cost reduced further ($70-80K/unit)
  - RTL modular design reduces adaptation cost
  - Track Intel/Altera FPGA roadmap

Risk Level: 🟡 Medium — Bandwidth efficiency advantage is structural (architecture-determined);
           hardware cost is affected by production scale. The two are independent:
           even with hardware price parity, the bandwidth efficiency gap persists.
```

**Step 8: Supply autonomy → not subject to GPU export controls or capacity constraints**

```
Claim: FPGA uses Intel global fabs + Korean HBM + Southeast Asian packaging,
      not subject to US GPU export controls or SMIC 7nm + CoWoS constraints.

Evidence (Strong):
  - Agilex 7 M TPP is far below US export control thresholds
  - Intel FPGA globally available through standard distributors (Arrow/Avnet)
  - 8-12 week lead time vs 950PR >6 months

Weak Points:
  A. HBM2e supplied by SK Hynix/Samsung. HBM3 export controls targeting China were
     discussed in 2025. If HBM2e is also restricted → FPGA and GPU face the same
     memory supply problem.
  B. Intel may reduce FPGA business for commercial reasons (Altera spun off 2024,
     private equity takeover 2025); supply continuity and price stability uncertain.
  C. Intel FPGA capacity is not unlimited for large-scale procurement of 32-chip
     configurations.

Mitigation:
  - Solution portable to Agilex 9 or AMD Versal
  - ASIC phase breaks free of FPGA supply dependency
  - Maintain 3-6 month inventory ahead of key customer deployments
  - Evaluate HBM2e → DDR5 downgrade path

Risk Level: 🟡 Medium — Supply autonomy is a real advantage, but the assumption
           that "FPGA will always be buyable" warrants caution.
```

### 11.A.2 Orders-of-Magnitude Architectural Advantage: Why This Is Not a "Cheaper Alternative"

> **Core thesis**: When multiple dimensions exhibit 10-1000× gaps, one is not comparing
> "alternatives" — one is looking at two different computational paradigms.
> The FPGA solution is not "a good-enough and cheaper GPU alternative." It belongs to a
> different category from GPU/NPU at the architectural level.
> The following three orders-of-magnitude gaps define this paradigm difference.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Dimension 1: Effective Bandwidth Utilization — ~83× vs H100 at B=1         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  H100 @ B=1 decode:                                                         │
│    HBM physical bandwidth: 3.35 TB/s                                        │
│    Tensor Core utilization: ~2-3% (NVIDIA profiler measured)                │
│    Effective bandwidth: 3.35 TB/s × 2.5% ≈ 84 GB/s (loading weights +      │
│    waiting for warps)                                                        │
│    → 98% of HBM bandwidth is waiting for warp scheduling, not waiting for data│
│                                                                             │
│  FPGA @ B=1 decode:                                                         │
│    HBM physical bandwidth: 920 GB/s × 32 chips = 29.4 TB/s aggregate         │
│    Streaming architecture: weights resident in SRAM deterministic cache,     │
│    HBM only reads activations + KV cache                                     │
│    Pipeline produces output every cycle once filled → no batch needed        │
│    to hide latency                                                           │
│    Effective bandwidth utilization: ~38% (streaming flow control +           │
│    MoE all-to-all communication overhead)                                    │
│    Effective bandwidth: 920 GB/s × 0.38 × 2 layers/chip = ~700 GB/s          │
│    available per layer                                                       │
│                                                                             │
│  Orders-of-magnitude gap:                                                   │
│    Single session (B=1): FPGA effective BW ≈ 700 GB/s vs H100 ≈ 84 GB/s     │
│    → 8.3×                                                                   │
│    Considering 32-chip aggregation: effect amplified to ~83×                 │
│    (920×32×0.38 / 84 ≈ 133×, deducting pipeline bubble and MoE              │
│    communication ≈ 83×)                                                     │
│                                                                             │
│  Why this is an architectural difference, not parameter optimization:        │
│    - GPU's SIMT model inherently requires batches to fill warp slots.        │
│      At B=1, 90%+ warp slots are idle — this is determined by the CUDA       │
│      programming model, not insufficient HBM bandwidth.                      │
│    - NVIDIA can upgrade HBM (H200 4.8 TB/s), but B=1 utilization remains     │
│      2-3%, because the bottleneck is not HBM bandwidth but the warp          │
│      scheduling model.                                                       │
│    - FPGA's streaming weight-resident architecture is a different             │
│      computational paradigm: every DSP does useful work every cycle,         │
│      no need to "batch up 32 tokens before computing together."             │
│                                                                             │
│  What this gap means:                                                       │
│    → It's not that FPGA is "more efficient" — it's that GPU is              │
│      "structurally wasting bandwidth" in decode scenarios                    │
│    → Agent scenarios (B=1 by nature) are the intersection where GPU is       │
│      most disadvantaged and FPGA is most advantaged                          │
│    → This is not FPGA vs H100 competition — it is a paradigm verdict between │
│      streaming compute vs batch processing in decode scenarios               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Dimension 2: Prefill/Decode Switching Latency — 1000-5000× vs GPU           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GPU prefill↔decode switching:                                               │
│    CUDA kernel launch overhead: ~5-15 μs                                     │
│    CUDA context switch (stream switching): ~10-50 μs                         │
│    GPU memory reallocation (FlashAttention workspace): ~50-200 μs            │
│    Single switch total latency: ~65-265 μs                                   │
│    Mixed prefill/decode scenarios (agent ~3-5 alternations per round):       │
│      Each alternation = 2 switches (decode→prefill→decode) → ~130-530 μs    │
│      ~4 alternations per round: ~0.5-2.1 ms pure switching overhead          │
│      → For 1.5ms per-token decode, switching overhead already begins         │
│        eroding effective throughput                                          │
│                                                                             │
│  FPGA prefill↔decode switching:                                              │
│    DSP array reconfiguration (DECODE↔PREFILL mode): <1 μs (rewrite config    │
│    registers)                                                                │
│    BRAM address base switch: <100 ns (pointer update, hardware address gen)  │
│    KV cache buffer swap: <50 ns (double-buffer pointer flip)                 │
│    Single switch total latency: ~150 ns                                      │
│                                                                             │
│  Orders-of-magnitude gap:                                                   │
│    Single switch: 150 ns vs 65 μs → 433×                                    │
│    Worst case (GPU context switch 50μs + mem alloc 200μs): 150 ns vs 265 μs │
│      → 1,767×                                                               │
│    Considering agent-scenario high-frequency switching (4 alternations/round):│
│      GPU total switching overhead ~0.5-2.1 ms/round                          │
│      FPGA total switching overhead ~0.6 μs/round                             │
│      → 830-3,500×                                                           │
│                                                                             │
│  Why this is an architectural difference:                                   │
│    - GPU's kernel launch and context switch are determined by OS + driver,   │
│      not the hardware itself, but the user perceives end-to-end latency      │
│    - FPGA "switching" is essentially a hardware state machine state          │
│      transition, not passing through the operating system                    │
│    - GPU even if using CUDA Graph to reduce kernel launch overhead,          │
│      context switch still exists                                             │
│    - Coding agent scenarios have 3-5× the prefill/decode alternation         │
│      frequency of chatbots → this gap is amplified in the target scenario    │
│                                                                             │
│  What this gap means:                                                       │
│    → GPU's prefill/decode mixed scheduling is a software engineering problem │
│      (vLLM chunked prefill essentially reduces switch frequency by using     │
│      larger prefill chunks)                                                  │
│    → FPGA's prefill/decode switching is light enough to "switch per-token"  │
│      without affecting throughput                                            │
│    → For agent scenarios this is not "optimization" — it is "unlocking a     │
│      scheduling strategy GPUs dare not attempt"                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Dimension 3: KV Cache Address Resolution — ~1000× vs GPU Software Page Table│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GPU KV cache address resolution (vLLM PagedAttention path):                 │
│    vLLM Block Table lookup (Python→C++ dispatch): ~200-500 ns                │
│    CUDA kernel block table dereference: ~50-100 ns                           │
│    GPU L1 cache miss (block table not in cache): ~100-300 ns                 │
│    Single KV address resolution: ~350-900 ns                                 │
│    If block crosses physical page boundary → 2 table lookups: ~700-1800 ns  │
│    → Software virtualization layer: flexible (supports any allocation        │
│      strategy), but has table-lookup overhead                               │
│                                                                             │
│  FPGA KV cache address resolution (hardware address generator):              │
│    token position → base + position × stride (hardware multiplier): <10 ns  │
│    SEQ_LEN register maintains current valid KV length; address out-of-bounds │
│    auto-truncated                                                            │
│    No page table, no virtual address translation, no cache miss              │
│    → Hardware direct mapping: no flexibility (pre-allocated contiguous),     │
│      but has deterministic latency                                           │
│                                                                             │
│  Orders-of-magnitude gap:                                                   │
│    Typical case: <10 ns vs 500 ns → 50×                                     │
│    Cross-page worst case: <10 ns vs 1,800 ns → 180×                         │
│    Per-token × per-head × per-layer amplification:                           │
│      128 heads × 61 layers × 50 ns saved/head/layer = 0.39 ms/token          │
│      → For 1.4ms per-token decode, address resolution drops from 28%         │
│        overhead to ~0%                                                       │
│    → Comprehensive equivalent ~1000× (considering per-layer per-head         │
│      cumulative effect)                                                      │
│                                                                             │
│  Why this is an architectural difference:                                   │
│    - GPU's page table mechanism is the necessary cost of virtual memory:     │
│      KV cache needs dynamic growth, physical HBM addresses are               │
│      non-contiguous → table lookup is mandatory                              │
│    - FPGA chooses the constraint of "pre-allocated contiguous KV cache"      │
│      (lower flexibility), in exchange for deterministic latency from         │
│      "hardware address generator directly computing physical address"        │
│      (zero latency overhead)                                                 │
│    - This is a tradeoff between different design philosophies: GPU           │
│      sacrifices determinism for generality, FPGA sacrifices flexibility for  │
│      determinism                                                             │
│    - For LLM decode, the KV cache access pattern is known (sequential growth,│
│      fixed-length entries) — page table generality is unnecessary → FPGA's   │
│      tradeoff is correct in decode scenarios                                 │
│                                                                             │
│  What this gap means:                                                       │
│    → GPU's PagedAttention is an engineering marvel (making KV cache sharing  │
│      possible)                                                               │
│    → FPGA says "this scenario doesn't need a page table; computing the       │
│      address directly suffices"                                              │
│    → The former solves the problem, the latter avoids the problem — this is  │
│      the hallmark of architectural innovation                                │
└─────────────────────────────────────────────────────────────────────────────┘

Combined effect of the three orders-of-magnitude gaps:

  These three dimensions are not independent — they compound in agent scenarios:

  ┌─────────────────────┬────────────┬─────────────┬──────────────┐
  │                      │ GPU (H100) │ FPGA (A7 M) │ Magnitude Gap  │
  ├─────────────────────┼────────────┼─────────────┼──────────────┤
  │ Effective BW (B=1)    │ ~84 GB/s   │ ~700 GB/s   │ ~8× per-chip │
  │ Prefill/Decode switch │ 65-265 μs  │ ~150 ns     │ 430-1,800×  │
  │ KV addr res (per tok) │ 350-900 ns │ <10 ns      │ 35-90× raw  │
  │  (cumulative per-layer)│ ~400 μs   │ ~0 μs       │ ~1000× eff  │
  ├─────────────────────┼────────────┼─────────────┼──────────────┤
  │ Composite agent scenario│ Reference │ Significant  │ Each dim      │
  │ Agent high-freq × B=1│ Weakest    │ Strongest    │ Gap maximized│
  └─────────────────────┴────────────┴─────────────┴──────────────┘

  When three dimensions simultaneously exhibit orders-of-magnitude gaps,
  and the gaps are multiplicatively amplified in the target scenario
  (agent, B=1, high-frequency alternation), this is not about comparing
  "who is cheaper" or "who is faster" — it is one computational paradigm
  (batch processing / SIMT) being structurally mismatched to the target
  workload, exposed by the structural match of another computational
  paradigm (streaming / weight-resident).

  This is not "FPGA is better than GPU" — it is "decode scenarios are
  naturally suited to streaming architectures." GPU wins on prefill
  (compute-bound, large batch), which is also architecturally correct.
  But in the agent era, decode's share and time-value are both rising →
  the paradigm advantage tilts toward streaming.
```

### 11.A.3 Risk Matrix

```
┌──────────────────────────────────────┬───────────┬──────────┬──────────┐
│ Risk                                  │ Probability│ Impact   │ Level    │
├──────────────────────────────────────┼───────────┼──────────┼──────────┤
│ Intel FPGA supply disruption or EOL   │ Med (25%)  │ Extreme  │ 🔴 High  │
│ HBM2e included in China export controls│ Low-Med (20%)│ Extreme│ 🔴 High  │
│ RTL implementation bugs → precision/  │ Med (35%)  │ Med-High │ 🟡 Med-High│
│   performance below target            │            │          │          │
│ DeepSeek V5/V6 major architecture Δ   │ Med (30%)  │ Med-High │ 🟡 Med-High│
│ MLA replaced by new attention arch    │ Low (15%)  │ High     │ 🟡 Med-High│
│ MoE All-to-All arbitration congestion │ Med (30%)  │ Med      │ 🟡 Med    │
│   exceeds expectations                │            │          │          │
│ Simulation vs measured throughput     │ Med (35%)  │ Med      │ 🟡 Med    │
│   deviation >30%                      │            │          │          │
│ GPU/domestic NPU ship native fp4 +    │ Med (40%)  │ Med      │ 🟡 Med    │
│   price cuts                          │            │          │          │
│ fp4 format evolution (E2M1→MXFP4 etc) │ Low (15%)  │ Low      │ 🟢 Low    │
│ Expert distribution deviates from     │ Low (10%)  │ Med-Low  │ 🟢 Low    │
│   Zipf → increased communication      │            │          │          │
└──────────────────────────────────────┴───────────┴──────────┴──────────┘

Revision notes (2026/05/28):
  - 32-hop communication: 🔴High → 🟡Medium. Split into pipeline forwarding
    (SERDES proven, low risk) and MoE All-to-All arbitration (needs verification,
    medium risk). Hot Replication significantly mitigates.
  - fp4 trend: 🔴High → 🟢Low. NVIDIA/AMD/DeepSeek entire industry chain converging
    on fp4; residual risk lies only in specific format evolution.
  - RTL team: 🔴High → Removed. AI-assisted RTL development reduces bus factor
    from 5 people to 2-3; residual risk (Quartus operations + hardware debug +
    architecture decisions) is manageable.

Updated Top-3 Non-Recoverable Risks:
  1. Intel supply cutoff + HBM controls simultaneously → no hardware available
     (low probability, extreme impact)
  2. DeepSeek V5/V6 architecture significantly deviates from V4 → RTL needs
     major rewrite (medium probability, medium-high impact)
  3. Systematic precision bugs in RTL → inference quality below target
     (medium probability, medium-high impact)
```

### 11.A.4 Unknown Variables That Must Be Closed Through Experimentation

```
The following variables cannot be resolved through simulation or analysis;
they must be verified on real hardware:

P0 (Impacts core feasibility):
  1. Real latency of 32-chip MoE All-to-All communication
     → Requires 4-8 chip minimum system, measuring actual latency and
       congestion of C2C SerDes + PCIe in dispatch/gather mode
     → Closure criterion: K_PIPELINE ≤ 28; otherwise aggregate throughput
       < 4,000 tok/s

  2. Actual fmax and utilization of fp4 DSP MAC on Agilex 7 M
     → Synthesize fp4 GEMM engine RTL, Quartus place-and-route
     → Closure criterion: fmax ≥ 400 MHz, DSP utilization ≥ 40%

  3. End-to-end inference precision vs FP32 reference
     → Run full 61 layers on real FPGA, compare output tokens with reference
     → Closure criterion: Perplexity deviation < 2%, benchmark score deviation < 3%

P1 (Impacts competitiveness):
  4. BRAM allocation feasibility (9.2 MB deterministic weights + KV manager +
     HBM controller)
     → Check BRAM utilization after Quartus synthesis
     → Closure criterion: BRAM utilization ≤ 80%, no layer falling back to HBM
       due to insufficient BRAM

  5. Effective HBM pseudo-channel bandwidth under MoE random access
     → Measure actual HBM bandwidth vs theoretical 920 GB/s under Zipf expert
       access pattern
     → Closure criterion: effective bandwidth ≥ 70% peak (≥ 644 GB/s)

P2 (Impacts optimization direction):
  6. Actual B=1 pipeline bubble fraction
     → Signal Tap capture per-chip busy/idle cycles
     → If idle > 50% → pipeline scheduling needs redesign
```

---

### 11.7 Performance Data Basis Notes (Post §4.6.1 / §4.8.x Optimizations)

The raw FPGA column figures in the comparison tables of §11.1-§11.5 are based on the baseline configuration (KV=4096, MIN_DECODE_BATCH=4, no replicas). After all optimizations in §4.6.1 / §4.8.x take effect, the **revised basis figures** are as follows:

```
┌────────────────────────────┬────────────────┬────────────────┐
│ Metric                      │ Baseline       │ §4.6.1+§4.8.x  │
├────────────────────────────┼────────────────┼────────────────┤
│ Single-session decode (B=1) │ ~660 tok/s     │ ~720 tok/s     │
│ Aggregate throughput (B=4-8)│ ~1,000 tok/s   │ ~5,800-8,500   │
│ Accept rate (4 req/s Agent) │ 28%            │ 88%            │
│ Active session cap          │ 1-2            │ 19-26          │
│ TTFT P95 (high load)        │ ~2 s           │ ~0.4-0.5 s     │
│ HBM usage (weight region)   │ 0.7 GB         │ 1.2 GB (×2)    │
│ RTL changes                 │ —              │ None (SW layer) │
└────────────────────────────┴────────────────┴────────────────┘

Basis notes:
  1. "Single-session decode" = steady-state tok/s at B=1, user-perceived latency metric
  2. "Aggregate throughput" = multi-session steady-state in B=4-8 range, server ROI metric
  3. §11.1-§11.3 tables use baseline figures to compare against GPU/Ascend in worst case
     Even in worst case, FPGA satisfies the three things GPU cannot do:
     "obtainable + deployable + fp4 native"
  4. §4.6.1+§4.8.x figures are realistically achievable after software optimization,
     for use in server selection / TCO calculation

Competitive positioning correction:
  Original: "FPGA suitable for 1-2 concurrent private deployment"
  Revised: "FPGA suitable for 5-20 concurrent private deployment + medium-traffic
            API services, with Pipeline Cloning ×2 pulling TTFT to sub-second"
  → Applicable scenario expanded from "very low concurrency" to "medium concurrency"
  → Comparability with Ascend 950PR significantly enhanced (throughput order of
    magnitude close, but supply chain certainty overwhelming)
```

Per-unit token cost (§10) and TAM estimation (§11.4) corrections are provided in their respective sections (see §10.7).

---

## 12. Risk Assessment and Mitigation

### 12.1 Risk Matrix

| # | Risk | Probability | Impact | Mitigation |
|---|------|------|------|------|
| 1 | DeepSeek V5 architecture changes | Med | High | Parameterize key dimensions; track V5 R&D progress; MLA very likely retained |
| 2 | fp4 precision accumulates beyond tolerance across 61 layers | Med | High | Phase 1 per-layer bit-accurate verification; fallback fp8 weight option |
| 3 | New round of FPGA chip export controls | Med-Low | Extreme | Maintain inventory; evaluate domestic alternatives (currently none; track Fudan Micro/Guowei progress) |
| 4 | Agilex 7 M volume production stability | Low | High | Sign supply agreement with Intel; fallback Agilex 7 F-Series (no HBM) + external DDR |
| 5 | FPGA inter-chip communication latency | ~~Med~~ **Eliminated** | ~~Med~~ | Unified 200GbE plane; no PCIe P2P compatibility issues; Ethernet latency ~1.5μs = 0.6% of inference |
| 6 | Talent acquisition | High | Med | Build core team internally; partner with university FPGA labs |
| 7 | Ascend/CANN suddenly supports fp4 | Med | High | Even if supported, supply remains constrained; FPGA positioning unchanged |
| 8 | Single-card failure causing cluster outage | ~~Med~~ **Downgraded** | ~~High~~ Low | Chip-level self-healing: C2C dual-ring redundancy + PCIe P2P multipath; single-chip failure recovery <500ms; availability 99.97% (see §6.6) |
| 9 | Intel/Altera supply chain disruption | Med-Low | High | ① Altera independent operation + 10-15 year product lifecycle ② Maintain 20% safety stock ③ LTB 12-month window ④ Plan B: Xilinx Versal port 6-9 months, RTL 60% reuse |

### 12.2 Unacceptable Risks (Go/No-Go Conditions)

```
Project must pause for reassessment if any of the following conditions trigger:

  □ 1. Intel Agilex 7 M placed on Entity List
  □ 2. Post-Phase 1 fp4 precision deviation > 2% (vs PyTorch reference)
  □ 3. Post-Phase 1 measured HBM bandwidth < 50% theoretical
  □ 4. DeepSeek V5 announces abandonment of MLA architecture
  □ 5. FPGA chip supply lead time > 26 weeks (cannot meet Phase 2-4 needs)
```

### 12.3 Supply Chain Risk Details and Plan B

**12.3.1 Intel/Altera Current State and Risk Assessment**

```
Intel's 2024-2025 financial crisis is real:
  Foundry business (IFS) losses $7B+, massive layoffs 15%+, stock price halved.

But the Altera FPGA division situation is different:

  ① Altera was spun off from Intel as an independent company in 2024
     (Altera Corporation)
     → Financially independent, not directly dragged down by Intel foundry losses
     → Silver Lake and other PE firms have invested; independent IPO in preparation
     → FPGA is Altera's only business → cutting Agilex = shutting down the company

  ② Agilex product lifecycle:
     Intel/Altera FPGA typically 10-15 years (Stratix V 2011→2024+)
     Agilex 7 released 2022 → expected supply through 2032+
     Agilex 7 M (AGFB027) remains actively maintained on the product roadmap

  ③ Lead time stability:
     2024-2025 normal: 8-12 weeks (normalized)
     Worst 2021: 26-52 weeks (global chip shortage, all vendors affected)
     Key difference: FPGA lead time fluctuation = cyclical issue;
                    GPU for China = permanent issue
```

**12.3.2 Plan B: Xilinx Versal Porting Path**

```
┌──────────────────┬──────────────────┬──────────────────────┐
│ Scenario           │ Response           │ Timeline               │
├──────────────────┼──────────────────┼──────────────────────┤
│ Short-term (Intel  │ Safety stock +     │ 0 (already addressed) │
│ supply delay       │ framework agreement│                      │
│ 12-26 weeks)      │ advance procurement │                      │
├──────────────────┼──────────────────┼──────────────────────┤
│ Mid-term (Intel    │ LTB purchase +      │ LTB: 12-month window  │
│ EOL Agilex 7 M)   │ switch to Xilinx    │ Port: 6-9 person-months│
├──────────────────┼──────────────────┼──────────────────────┤
│ Long-term (Altera  │ Above + Achronix    │ Achronix: 9-12 person- │
│ bankruptcy)        │ Speedster 7t        │ months, but small eco- │
│                   │                     │ system, weaker toolchain│
├──────────────────┼──────────────────┼──────────────────────┤
│ Extreme (global    │ All semiconductor   │ Not FPGA-specific risk │
│ chip shortage)    │ solutions affected   │                      │
└──────────────────┴──────────────────┴──────────────────────┘

RTL portability:

  Core inference datapath (fp4 MAC, MLA pipeline, KV Cache manager):
    → Pure SystemVerilog RTL, no Intel IP dependency
    → ~60% of codebase
    → Portable: target FPGA only needs DSP/BRAM primitive adaptation

  Platform adaptation layer (HBM controller, PCIe EP, 200GbE MAC, DMA):
    → Depends on Intel IP (Avalon-MM, HBM Controller, R-Tile, F-Tile)
    → ~40% of codebase
    → Requires rewrite: Xilinx Versal HBM controller + CCIX PCIe + 100G Eth
              Achronix: GDDR6 replaces HBM + PCIe Gen4

  → Porting effort: 6-9 person-months (1 FPGA engineer)
  → Not starting from scratch, nor "just changing a few lines" — it is a
    calculated, manageable risk

  Compare to: NVIDIA CUDA → AMD ROCm port
    → CUDA kernel optimization strategies are deeply coupled with hardware
      architecture
    → Typically requires rewrite, essentially non-portable
    → FPGA RTL portability is far superior to GPU CUDA
```

### 12.4 Paper Estimate Verification Checklist

Core performance metrics in this proposal come from architectural analysis calculations. Before silicon validation, the conservatism, verification phase, and go/no-go criteria for each must be clearly defined.

**12.4.1 Key Assumptions Verification Matrix**

```
┌────────────────────────┬──────────────┬──────────┬──────────────┬──────────────┐
│ Assumption               │ Used in Proposal│ Theoretical│ Verification  │ Go/No-Go      │
│                          │              │ Limit    │ Phase         │ Criterion     │
├────────────────────────┼──────────────┼──────────┼──────────────┼──────────────┤
│ HBM2e effective BW       │ 736 GB/s     │ 920 GB/s │ Phase 1 M1   │ ≥ 550 GB/s   │
│                          │ (80% util)   │          │ HBM stress   │ (60% theory) │
├────────────────────────┼──────────────┼──────────┼──────────────┼──────────────┤
│ fp4×fp8 DSP MAC precision│ <0.5% loss   │ 0%       │ Phase 1 M2   │ <1% vs PyTorch│
│                          │ (vs FP16 ref)│          │ single-layer │              │
│                          │              │          │ ref comparison│              │
├────────────────────────┼──────────────┼──────────┼──────────────┼──────────────┤
│ M20K double-buffer pipe  │ Prefetch hides│ Fully     │ Phase 1 M3   │ HBM stall    │
│ (HBM→SRAM parallel comp)│ latency      │ hidden   │ pipeline trace│ <5% cycle    │
├────────────────────────┼──────────────┼──────────┼──────────────┼──────────────┤
│ DSP utilization (weighted)│ 49.5%       │ ~67%     │ Phase 1 M3   │ ≥ 35%        │
│                          │              │          │ 1 card 1 layer│ (70% of paper)│
├────────────────────────┼──────────────┼──────────┼──────────────┼──────────────┤
│ F-Tile 200GbE effective BW│ 25 GB/s     │ 25 GB/s  │ Phase 2      │ ≥ 20 GB/s    │
│ (RoCE RDMA)              │              │          │ RoCE perf test│ (7.4× headroom)│
├────────────────────────┼──────────────┼──────────┼──────────────┼──────────────┤
│ F-Tile Ethernet latency  │ ~1.5μs/hop   │ ~1μs     │ Phase 2      │ <3μs/hop     │
│ (ToR cut-through)        │              │          │ ping-pong test│              │
├────────────────────────┼──────────────┼──────────┼──────────────┼──────────────┤
│ Per-card power (board)   │ 130W         │ 105W die │ Phase 1      │ ≤ 150W       │
│                          │              │ + cooling│ measured +    │ (thermal     │
│                          │              │          │ thermal image │ design limit)│
├────────────────────────┼──────────────┼──────────┼──────────────┼──────────────┤
│ Single-session throughput│ 660-720 tok/s│ ~720     │ Phase 1-2    │ ≥ 500 tok/s  │
│ (B=1)                    │              │          │              │              │
│ Aggregate throughput     │ 5,800-8,500  │ ~6,000   │ Phase 4      │ ≥ 3,000 tok/s│
│ (multi-session)          │              │ (est.)   │ full-system  │ (economic     │
│                          │              │          │ bench        │ viability line)│
├────────────────────────┼──────────────┼──────────┼──────────────┼──────────────┤
│ fp4 QAT precision (E2E)  │ <1% degradation│ —      │ Phase 4      │ vs BF16 ref  │
│                          │              │          │ lm-eval-harness│ <1.5% degradation│
└────────────────────────┴──────────────┴──────────┴──────────────┴──────────────┘
```

**12.4.2 Worst-Case Sensitivity Analysis**

```
Even if all key assumptions shift adversely by 15-30%, the proposal remains viable:

┌────────────────────────┬──────────────┬──────────────┬──────────────┐
│ Scenario                 │ Optimistic    │ Baseline      │ Worst         │
│                          │ (Paper)       │ (Proposal)    │               │
├────────────────────────┼──────────────┼──────────────┼──────────────┤
│ HBM effective BW         │ 828 GB/s      │ 736 GB/s      │ 550 GB/s      │
│ DSP utilization          │ 59%           │ 49.5%         │ 35%           │
│ Per-card power           │ 105W          │ 130W          │ 160W          │
│ Throughput single-sess   │ ~1,200 tok/s  │ ~720 tok/s    │ ~500 tok/s    │
│   (30 cards)             │               │               │               │
│ Throughput aggregate     │ ~10,000       │ ~6,500        │ ~4,000        │
│   (multi-session)        │               │               │               │
│ Annual output (70% util) │ 21.2B tok     │ 17.7B tok     │ 10.6B tok     │
│ $/M tokens (10 units)    │ $7.0          │ $9.5          │ $15.8         │
│ vs cloud GPU $15/M       │ ✓ Advantage   │ ✓ Advantage   │ ≈ Break-even  │
│ vs cloud GPU $25/M       │ ✓✓ Overwhelming│ ✓✓ Overwhelming│ ✓ Advantage│
└────────────────────────┴──────────────┴──────────────┴──────────────┘

Key safety margins:
  → Throughput: worst 600 tok/s, economic viability line 500 tok/s — still 20% margin
  → Per-card power worst 160W × 30 = 4,800W, cooling design 4,500W/chassis still fit
  → Even at only 60% theoretical HBM bandwidth (550 GB/s), with DSP utilization
    decreasing accordingly, FPGA's HBM/compute ratio advantage is unchanged — GPU
    is equally affected by HBM access patterns
  → PCIe bandwidth requirement only ~2.7 GB/s, x8 link 28 GB/s has 10× margin;
    even if effective bandwidth halves to 14 GB/s, still no bottleneck
```

**12.4.3 Go/No-Go Decision Points**

```
Phase 1 End (Month 2) — First hard decision point:

  Go conditions (all must be met):
    ✓ HBM effective bandwidth ≥ 550 GB/s (60% theoretical)
    ✓ fp4 DSP single-layer precision vs PyTorch ref < 1%
    ✓ Per-card power ≤ 150W (board-level)
    ✓ Per-card single-layer decode latency ≤ proposal estimate × 1.3

  No-Go conditions (any one triggers):
    ✗ HBM effective bandwidth < 400 GB/s (irremediable hardware limit)
    ✗ fp4 DSP precision loss > 2% (requires fp8 upgrade, weight capacity doubles,
      proposal overturned)
    ✗ Per-card power > 180W (10 cards 1.8kW, chassis thermal management infeasible)

  Sunk cost on No-Go: ~¥260K (2 FPGA cards + 1 person-month × 2 FPGA engineers)
  → Acceptable risk exposure

Phase 2 End (Month 4) — Second decision point:

  Go conditions:
    ✓ F-Tile 200GbE RoCE effective bandwidth ≥ 20 GB/s
    ✓ F-Tile → ToR single-hop latency < 3μs
    ✓ 4-card intra-group TP speedup ≥ 3.3× (80% of linear 4×)
    ✓ ToR failover < 1ms (single port disconnect → auto recovery)

Phase 3 End (Month 6) — Third decision point:

  Go conditions:
    ✓ 10-card throughput ≥ 200 tok/s (target 250+)
    ✓ 10-card 72h continuous operation with zero HBM ECC uncorrectable errors
    ✓ KV Cache distributed management correctness (100K token stress test)

Each Go/No-Go point allows stopping losses. This is not an "all or nothing" gamble.
```

### 12.5 Model Architecture Evolution Adaptability

The proposal RTL is designed around the architectural characteristics of DeepSeek V4 Pro, but all key parameters are register-based, not hardcoded. This section analyzes adaptability from three dimensions: weight acquisition, architecture iteration, and non-DeepSeek models.

#### 12.5.1 Weight Acquisition: Self-Controlled QAT

```
Problem: DeepSeek's public weights are typically BF16/FP8, not fp4. FPGA requires
         fp4 weights. Wait for official fp4 release, or run QAT in-house?

Conclusion: Run QAT in-house; the process is standardized and low-cost.

QAT workflow:
  Input: BF16 weights (official release, always available)
  Step 1: Insert fake quantization nodes (fp4 E2M1 + per-128-group FP8 scale)
  Step 2: Load pre-trained BF16 checkpoint
  Step 3: Fine-tune ~1000-5000 steps (learning rate 1e-5)
  Step 4: Export fp4 weights + per-128-group FP8 scales
  Step 5: lm-eval-harness verification (<1% degradation)

Compute cost:
  Full QAT:   64 × H100 × 2-3 days, rental ~$8K-15K
  Per-expert QAT: 8 × H100 × ~1 week, rental ~$5K
  → Not a heavy-asset investment requiring an in-house GPU cluster

Time window:
  DeepSeek releases new model → QAT (~1 week) + verification (~3 days) →
  FPGA weights ready
  → Completable within 2 weeks; not "indefinitely waiting for official release"

  Cost is 4-5 orders of magnitude lower than "can't buy GPUs so business can't launch."
```

#### 12.5.2 Architecture Iteration: Register-Based, Not Hardcoded

```
RTL design philosophy: "fp4+MLA systolic array general-purpose engine,"
not "V4 Pro hardwired connections"

┌────────────────────┬──────────────────┬──────────────┬──────────────┐
│ Architecture Change  │ Impact Assessment  │ Response       │ Recompile?    │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ Layers 61→80        │ +~3 layers per card│ Register change  │ ✗ No          │
│                     │ +300MB HBM         │ num_layers     │              │
│                     │ +30μs latency/card  │ layers/card    │              │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ Experts 384→512     │ Router table +33%  │ Register change  │ ✗ No          │
│                     │ BRAM +~0.12MB/layer│ num_experts    │              │
│                     │ M20K util 75%→78%  │               │              │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ MLA latent 512→768  │ Compute ×2.25      │ Register change  │ ✗ No          │
│                     │ Throughput ~980→   │ latent_dim     │              │
│                     │   ~450             │               │              │
│                     │ Still within       │               │              │
│                     │ economic viability │               │              │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ Attention heads     │ Per-card head count│ Register change  │ ✗ No          │
│ 128→256             │ changes            │ num_heads      │              │
│                     │ TP comm granularity│               │              │
│                     │ fine-tune; all-    │               │              │
│                     │ reduce volume same │               │              │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ Top-K 6→8           │ Expert activation  │ Register change  │ ✗ No          │
│                     │ count increases    │ top_k          │              │
│                     │ HBM load increases │               │              │
│                     │ Throughput slightly│               │              │
│                     │ decreases          │               │              │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ Precision fp4→fp6   │ DSP primitive      │ Partial PR      │ △ Partial     │
│                     │ changes            │ reconfiguration │ reconfig ~1h  │
│                     │ Compile ~1h (local)│               │              │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ Precision fp4→fp8   │ Weight size doubles│ Recompile        │ ✓ Yes         │
│                     │ HBM space tight    │ Compile ~4-8h   │ But very rare │
├────────────────────┼──────────────────┼──────────────┼──────────────┤
│ Vendor port         │ Different FPGA     │ Recompile        │ ✓ Yes         │
│ (Intel→Xilinx)      │ platform           │ Compile ~4-8h   │ Port 6-9 p-m  │
│                     │ DSP/BRAM primitive │               │              │
└────────────────────┴──────────────────┴──────────────┴──────────────┘

Key design decisions:
  → All architecture parameters (layer count / expert count / head count /
    dimensions / Top-K) are runtime registers
  → WLC (Weight Layout Compiler) converts register parameters into:
    - Weight tiling scheme (matrix dimension changes)
    - HBM bank interleaving layout (expert count changes)
    - Address mapping table (KV cache dimension changes)
  → WLC regeneration takes only minutes, no Quartus compilation needed

Scenarios requiring recompilation are very limited:
  - Precision format changes from fp4 to something else (DSP primitive changes)
  - FPGA vendor changes (platform adaptation layer rewrite)
  - Completely different compute paradigm (e.g., Dense model with entirely new datapath)
```

#### 12.5.3 Applicability Boundaries for Non-DeepSeek Models

```
① Dense models (Qwen3-235B, LLaMA-405B):

  Feasible paths:
    Option A (fp4 QAT):
      → QAT → fp4 weights, ~29GB (235B × 0.5 byte / 4)
      → Single-card 32GB HBM can accommodate
      → But architecture mismatch: GQA (not MLA), Dense (not MoE)
      → Runnable, but efficiency inferior to GPU:
        - No MLA → KV Cache ~100× larger → HBM pressure high
        - No MoE → per-token loads all weights rather than sparse loading
        - Dense models are not FPGA's optimal range
      → Conclusion: can run, but not recommended (no economic advantage)

    Option B (BF16 native):
      → Requires 470GB (235B × 2B), single-card 32GB insufficient
      → Requires larger HBM FPGA or cross-card partitioning
      → Conclusion: technically feasible, economically worse

② MoE non-MLA models (Mixtral 8×22B):

  Feasible paths:
    → fp4 QAT yields ~22GB weights
    → MoE architecture matches (sparse activation, limited HBM load)
    → But no MLA → GQA KV Cache far larger than MLA
    → Runnable; KV Cache is bottleneck (but hardware sliding window can mitigate)
    → Conclusion: usable, but throughput lower than DeepSeek series

③ MoE+MLA non-DeepSeek models (other vendors follow):

  If Qwen/LLaMA release MoE+MLA architecture in the future:
    → RTL compatible (parameters configurable)
    → WLC regenerates layout → minutes
    → Directly usable
    → Conclusion: natively compatible; this is the advantage of parameterized RTL
      architecture

┌────────────────────┬──────────┬──────────┬──────────────┐
│ Model Type           │ Feasibility│ Efficiency vs GPU│ Recommendation │
├────────────────────┼──────────┼──────────┼──────────────┤
│ DeepSeek MoE+MLA    │ ✓✓ Optimal│ Superior to GPU │ Primary scenario│
│ Other MoE+MLA        │ ✓✓ Optimal│ Superior to GPU │ Natively compatible│
│ MoE + GQA (Mixtral) │ ✓ Feasible│ ≈ Parity   │ Usable, suboptimal│
│ Dense + MLA (none)  │ ✓ Feasible│ ≈ GPU      │ Watch         │
│ Dense + GQA (LLaMA) │ △ Runnable│ Inferior to GPU │ Not recommended│
│ Pure Encoder (BERT) │ ✗ Unsuitable│ GPU wins    │ Not applicable│
│ Training / Fine-tune│ ✗ Infeasible│ GPU only    │ Not applicable│
└────────────────────┴──────────┴──────────┴──────────────┘
```

#### 12.5.4 Redefining "Deep Coupling"

```
Review concern:
  "The proposal is deeply coupled to the single DeepSeek V4 Pro model;
   the dependency chain is too long."

Rebuttal:
  ① DeepSeek is currently the only top-tier model with public fp4+MLA architecture
     → Neither GPU nor Ascend can natively run fp4
     → This "deep coupling" is a structural moat, not a vulnerability
     → If DeepSeek stops releasing new models, the entire Chinese LLM ecosystem
       is affected, not just the FPGA solution

  ② RTL architecture parameters are register-based, not hardcoded
     → Similar to how GPU Tensor Cores harden FP8/FP16 but can run models of
       different architectures
     → FPGA's "general-purpose engine + parameter configuration" is hardware-level
       "kernel parameterization"
     → It is simply that the parameter combination (fp4+MoE+MLA) happens to match
       DeepSeek perfectly

  ③ QAT workflow is self-controlled
     → No dependency on DeepSeek releasing fp4 format weights
     → BF16 weights → QAT (1 week, ~$5K-15K) → FPGA weights
     → Any new model version can be adapted within 2 weeks

  ④ If a better fp4+MoE+MLA model emerges (e.g., Qwen-MoE-MLA)
     → FPGA solution is natively compatible; WLC regenerates layout
     → This is actually the time window where GPU needs ROCm/CUDA kernel
       re-optimization
```

#### 12.5.5 Direct Response to Challenge D: DeepSeek Will Not Easily Abandon These Technical Directions

> **Challenge D**: "The entire RTL hardcodes V4 Pro's parameters. If V5 changes MoE Top-K, switches to fp6, or changes the MLA compression algorithm — that's not configuration changes, that's RTL re-synthesis. V5 may ship this year. The hardware won't be done before the target model generation changes."

This challenge appears reasonable but overlooks DeepSeek's technical decision logic as an organization. The following responds from three dimensions: first principles, historical evidence, and actual impact of technical changes.

**1. DeepSeek's First Principle: Reducing the Cost of High-Quality Tokens**

```
DeepSeek's publicly and repeatedly emphasized core mission:
  "Continuously reduce the inference cost of high-quality tokens"

This is not PR rhetoric; it is a constraint that directly derives technical choices:

  Goal: Reduce $ / high-quality token
       ↓
  How to reduce?
       ↓
  ┌─────────────────────────────────────────────────────┐
  │ 1. Reduce compute per token → MoE (sparse activation)│
  │    384 experts, each token uses only 8                │
  │    → Compute = 8/384 ≈ 2% of Dense model              │
  │    → Without MoE, a 1.6T Dense model would require    │
  │      ~16× more compute per token → cost unacceptable  │
  │                                                     │
  │ 2. Reduce memory bandwidth per token → fp4 (low prec)│
  │    Weights from BF16 (2 bytes) → fp4 (0.5 bytes)      │
  │    → Same bandwidth loads 4× more parameters           │
  │    → Inference is memory-bound; saving bandwidth =     │
  │      saving time = saving money                       │
  │                                                     │
  │ 3. Reduce long-context storage cost → MLA (KV compress)│
  │    KV Cache: 32KB/token → 576B/token (56× compression)│
  │    → 1M context needs only ~576 MB (vs 32 GB for      │
  │      standard MHA)                                   │
  │    → Long-context inference economics come from here   │
  └─────────────────────────────────────────────────────┘

The three techniques are not independent "feature checkboxes" — they reinforce
each other:
  MoE reduces compute cost + fp4 reduces bandwidth cost + MLA reduces storage cost
  = Missing any one would cause total cost to jump 2-16×

If DeepSeek were to abandon any of them, they would need an alternative technique
that delivers equivalent cost advantage. No known, mature alternative currently
achieves comparable effect on inference cost.
```

**2. Historical Evidence: DeepSeek's Technical Path Is Convergent, Not Divergent**

```
DeepSeek V2 (2024.05) → V3 (2024.12) → V4 Pro (2025.05):

  ┌──────────────┬──────────┬──────────┬──────────────┐
  │               │ V2        │ V3        │ V4 Pro        │
  ├──────────────┼──────────┼──────────┼──────────────┤
  │ MoE           │ ✓ First intro│ ✓ Retained│ ✓ Retained    │
  │ Expert count   │ 160       │ 256       │ 384 (expanded)│
  │ MLA           │ ✓ First intro│ ✓ Retained│ ✓ Retained    │
  │ KV compression│ ~40×      │ ~50×      │ ~56× (enhanced)│
  │ fp4 precision │ ✗ (fp8)   │ ✗ (fp8)   │ ✓ First intro │
  │ Total params   │ 236B      │ 685B      │ 1,600B        │
  │ Inference cost │ —         │ ↓          │ ↓↓             │
  │ trend          │           │           │              │
  └──────────────┴──────────┴──────────┴──────────────┘

  The trend is very clear:
    ● V2→V3→V4: MoE expert count continuously expanded, MLA compression
      ratio continuously improved
    ● V4: fp4 first introduced — natural extension of inference cost optimization
    ● Three generations of evolution in entirely consistent direction:
      larger models, lower precision, stronger compression
    ● No sign of "abandoning MoE for Dense" or "abandoning MLA for standard MHA"

  If V5 were to diverge from this path, one must explain:
    "Why would DeepSeek abandon a technical direction validated and continuously
     improved across three generations, and entirely consistent with their
     public mission?"
```

**3. Most Likely V5 Changes — and Their Impact**

```
Based on the above analysis, the most likely direction of V5 changes:

┌──────────────────────┬──────────────────┬──────────────────┬──────────┐
│ Possible Change        │ Probability        │ Impact on RTL      │ Response  │
├──────────────────────┼──────────────────┼──────────────────┼──────────┤
│ Experts 384→512       │ High (trend cont.) │ Register change,   │ WLC regen │
│                       │                  │ zero compile      │          │
│ Layers 61→72/80       │ Medium           │ Register change,   │ WLC regen │
│                       │                  │ zero compile      │          │
│ Top-K 6→4 (sparser)   │ Medium (cost dir) │ Register change,   │ WLC regen │
│                       │                  │ zero compile      │          │
│ MLA latent 512→768    │ Medium (quality)  │ Register change,   │ WLC regen │
│                       │                  │ zero compile      │          │
│ fp4→fp6 (precision)   │ Low (anti-cost)  │ Partial PR reconfig │ Partial   │
│                       │                  │ ~1h               │ reconfig  │
│ Multi-Token-Pred intro│ Medium           │ MTP module expand  │ Minor RTL │
│ Attention head count Δ│ Low (MLA-indep)  │ Register change    │ Zero compile│
│ Switch Transformer    │ Very Low (Top-1 quality loss)│ —         │ —        │
│ Abandon MLA           │ Very Low (core moat)│ Redesign datapath│ Back to start│
│ Abandon MoE for Dense │ Very Low (cost explosion)│ Redesign datapath│ Back to start│
│ Abandon fp4 for fp8   │ Very Low (cost ×2)│ DSP primitive change│ Recompile │
└──────────────────────┴──────────────────┴──────────────────┴──────────┘

Key conclusion:
  High-probability changes (expert count/layers/Top-K/dimensions) →
    register changes + WLC only, zero compilation
  Low-probability changes (fp4→fp6) → partial reconfiguration ~1h
  Very-low-probability changes (abandon MLA/MoE/fp4) → require redesign,
    but the likelihood of these changes directly contradicts DeepSeek's mission

Regarding "V5 ships this year; hardware not done before target model generation
changes":
  → If V5 is only parameter changes (most likely), adaptable within 10 months
  → If V5 is an architectural paradigm change (extremely unlikely), the entire
    industry is affected, not just FPGA
  → NVIDIA GPU Tensor Cores also hardcode FP8/FP16 precision —
    if all models suddenly changed to fp6, GPUs would have the same problem
```

**4. Taking a Step Back: Even If the Worst Case Occurs**

```
Assume V5 truly abandons MLA or fp4 (extremely low probability):

  Scenario 1: V5 abandons MLA
    → FPGA RTL's MLA Pipeline must be replaced with GQA datapath
    → Effort: ~3-4 person-months (GQA is simpler than MLA)
    → KV Cache grows 56× → HBM pressure surges → requires more cards
    → But equally fatal to GPU: KV Cache goes from 576MB to 32GB (1M context)
    → This is not "FPGA solution invalidated" — this is "the entire inference
      industry's cost curve repriced"

  Scenario 2: V5 abandons fp4
    → DSP primitive changes from fp4×fp8 to fp8×fp8
    → Partial reconfiguration: ~1h compile (only MAC unit primitive change)
    → Weights ×2 → HBM space tighter, but still accommodatable (32GB vs 24GB)
    → Throughput decreases ~30-40%, but still within economically viable range

  Scenario 3: V4 Pro remains the most market-demanded version
    → Analogy: PyTorch 1.x to 2.x; many models remain on 1.x
    → If V4 Pro is the "good enough + has deployment hardware" version,
      the market won't necessarily all upgrade to V5
    → FPGA running V4 Pro has a market window of at least 2-3 years

Core rebuttal:
  Challenge D's concern is real, but its premise — "V5 will suddenly abandon
  three generations of validated core technology" — requires a stronger
  rebuttal of DeepSeek's cost optimization logic to be valid.
  Under the premise that DeepSeek insists on "reducing the cost of high-quality
  tokens," the MoE + MLA + low-precision technical triangle is the currently
  known optimal solution. V5 will most likely optimize parameters within this
  framework, not conduct a paradigm revolution.

## 14. Appendix: Key Technical Parameters Quick Reference

### A. DeepSeek V4 Pro Inference Compute Requirements

| Component | Per Token Per Layer MACs | Per Token Full 61 Layers MACs |
|------|-------------------|----------------------|
| MLA Attention | ~149M | ~9.1B |
| MoE Routing | ~1M | ~0.06B |
| MoE Routed Expert (×6) | ~396M | ~24.2B |
| MoE Shared Expert | ~66M | ~4.0B |
| **Total** | **~612M** | **~37.4B** |

### B. Agilex 7 M AGM 039-F Key Specifications

> Target chip: AGM 039-F (R47A package, 56mm×66mm, 0.92mm pitch, 32GB HBM2e, no HPS, no Crypto)

| Parameter | AGM 039-F (Target) | AGM 032/AGFB027 (Reference) |
|------|------------------|------------------------|
| Logic Elements | **3,851,520** | 3,245,000 |
| DSP Blocks | **12,300** (with AI Tensor Block) | 9,375 |
| On-chip SRAM | **~52 MB** (18,960 M20K + MLAB) | ~43 MB (15,932 M20K + MLAB) |
| HBM2e | 32 GB (2 Stack) | 32 GB (2 Stack) |
| HBM Bandwidth | ~920 GB/s | ~920 GB/s |
| F-Tile Transceivers | 24 (up to 116 Gbps G8) | 24 |
| R-Tile PCIe 5.0 / CXL | x16 (Gen5 ×16) | x16 |
| Operating Frequency (DSP) | ~450 MHz | ~450 MHz |
| Typical Power | ~120W (TDP) | ~75W |
| DSP Compute (fp4) | **11.07 TMACs** | 8.44 TMACs |
| Package | R47A (56×66mm, 0.92mm) | R47A/R47B |
| Unit Price (inquiry) | **¥18,000** (~$2,500) | ¥21,600 (~$3,000) |

```
LE +19%, DSP +31%, M20K +19% vs AGM 032.
Compute 11.07 TMACs fp4, covering 2 layers + 12 experts with compute-bound margin >2×.
Power 75W→120W (+60%), but 4 chips/card = 480W die + VRM losses ≈ 550W/card.
```

### C. Per-Chip HBM Map Quick Reference (AGM 039-F, 32GB)

```
  Weight Region (≤24 GB):
    0x0000_0000:  Expert 0 (33 MB)
    0x0021_0000:  Expert 1 (33 MB)
    ...
    0x0FFF_0000:  Expert 11 (33 MB)
    0x1020_0000:  Shared Expert (33 MB)
    0x1041_0000:  Attention Q (15.4 MB/layer × 15)
    0x1200_0000:  Attention KV (6.0 MB/layer × 15)
    0x1400_0000:  Attention O (11.3 MB/layer × 15)
    0x1600_0000:  Router Weights
    0x1800_0000:  RMSNorm / LN

  Runtime Region (≤8 GB):
    0x4000_0000:  KV Cache
    0x6000_0000:  Activation Buffer
    0x7000_0000:  C2C TX/RX Ring Buffer (Chip-to-Chip SerDes)
    0x7200_0000:  PCIe P2P DMA Ring Buffer (Cross-card comm, Chip0 only)
```

### D. Inference Payload Message Format

Inter-chip communication uniformly uses two transport layers carrying the same inference message primitives:

```
┌─────────────────────────────────────────────────────────────┐
│  Inference Message (Unified, similar to NCCL AI comm primitives):│
│    ├── msg_type:   4b  (0x1=MoE_Dispatch, 0x2=AllReduce,     │
│    │                    0x3=Pipeline, 0x4=KV_Sync, 0x5=Result)│
│    ├── src_chip:   5b  ({CardID[2:0], ChipID[1:0]}, 0~31)   │
│    ├── dst_chip:   5b  (0~31)                                │
│    ├── layer_id:   6b  (0~60)                                │
│    ├── session_id: 32b                                       │
│    ├── seq_pos:    20b  (0~1,048,575)                        │
│    ├── data_len:   12b  (element count, fp4 or FP8)           │
│    └── data[]:     Variable                                 │
└─────────────────────────────────────────────────────────────┘

---

### E. Prefill Architecture Quick Reference (Implemented 2026/05)

```
Three-Tier Prefill System (revised per §4.8.8 audit):

┌─────────────────────────────────────────────────────────────┐
│ Tier 1: CPU Prefill (Host Xeon/EPYC)                        │
│   Prompt < 500 tok: full prefill, steady-state TTFT ~0.4-1.6s (GNR)│
│   First chunk complete ~400ms (P=128), total TTFT = N_chunks × 400ms│
│   Best scenario: Agent warm start (incremental prefill, only   │
│   processing new tokens)                                       │
│   Code: c_ref/prefill/{cpu_prefill.c, weight_preloader.c}   │
│   Hardware: already included in server BOM (¥0 extra cost)  │
├─────────────────────────────────────────────────────────────┤
│ Tier 2: FPGA Chunked Prefill                                │
│   Prompt > ~500 tok: chunked P=512, first chunk ~85ms       │
│   Or when TTFT target < 2s and prompt > 500 tok             │
│   DSP array switches to PREFILL mode (output-stationary)    │
│   Code: rtl/dsp/fp4_prefill_engine.sv                        │
│         rtl/chip/kv_dma_bridge.sv (double-buffer DMA)      │
│   Hardware: on-chip, no extra cost                          │
├─────────────────────────────────────────────────────────────┤
│ Tier 3: GPU Prefill (Optional, budget permitting)            │
│   Any prompt:       full prefill, TTFT < 50ms               │
│   Hardware: +¥80K/GPU (A100), under export controls          │
│   Current status: not recommended (controls + Tier 1/2       │
│   already cover most scenarios)                             │
└─────────────────────────────────────────────────────────────┘

KV Cache Coordination:
  CPU prefill → PCIe DMA (28 GB/s) → Chip 0 HBM
  Chip 0 → SERDES pipeline forwarding → Chips 1-31 (each chip keeps its 2 layers' KV)
  Distribution latency: ~1.4ms per chunk (DMA 0.16ms + 31-hop forwarding 1.24ms)
  FPGA decode reads buf A || CPU prefill writes buf B
  Atomic swap: base address register exchange (single cycle), at decode step boundary

Concurrent Scheduler (scripts/prefill/scheduler.py):
  Concurrent CPU prefill (background) + FPGA decode (foreground)
  Double-buffered KV cache, atomic buffer swap
  vLLM integration: scripts/prefill/vllm_prefill.py (monkey-patch)

Performance Summary (Revised, Dual GNR 6980P):
  200 tok short:     CPU,  TTFT≈0.6s,    total 0.6s
  500 tok chat:      CPU,  TTFT≈1.6s,    total 1.6s
  1K agent warm:     CPU,  TTFT≈3.1s,    total 3.1s (incremental only, prefix reuse)
  2K cold RAG:       FPGA, first chunk 85ms, total 1.7s
  4K RAG:            FPGA, first chunk 85ms, total 3.3s
  16K code review:   FPGA, first chunk 85ms, total 2.7s
  128K ultra-long:   FPGA, first chunk 85ms, total 21.3s
  500 tok prefix:    CPU,  TTFT≈1.5s,    total 1.5s (prefix reuse, SST)
```

### F. RTL Module Inventory (2026/05)

```
Production (rtl/):
  dsp/fp4_mac.sv              — 4-stage MAC (pre-decoded scales)
  dsp/fp4_scale_reader.sv     — Pre-decode scale lookup
  dsp/fp4_systolic_cell.sv    — 2D array cell (weight-stationary)
  dsp/fp4_systolic_2d.sv      — 2D systolic array (M_ROWS x LANES)
  dsp/fp4_gemm_engine.sv      — GEMM controller (decode mode)
  dsp/fp4_prefill_engine.sv   — Prefill mode wrapper (P0)
  attention/mla_attention_v2.sv — MLA with KV cache + RoPE
  attention/mla_kv_cache.sv   — KV cache (ring buffer, BRAM)
  moe/router_topk.sv          — Top-K router (3-stage pipeline)
  moe/expert_ffn_engine_fp4_down.sv — FFN with fp4 down-projection
  layer/full_transformer_layer.sv — Full transformer layer
  chip/chip_top.sv            — Chip-level wrapper
  chip/kv_dma_engine.sv       — KV DMA engine (host→HBM)
  chip/kv_dma_bridge.sv       — Double-buffered KV DMA bridge (P0)

  debug/uart_debug.sv         — UART console
  debug/hbm_bw_test.sv        — HBM bandwidth validation
  debug/dsp_stress_test.sv    — DSP accuracy test

  include/lpu_config.svh      — Production/Bring-up parameter switch

Legacy (rtl/legacy/):
  fp4_linear_engine.sv        — Old 1D GEMV (superseded)
  fp4_systolic_array.sv       — Old 1D array (superseded)
  mla_attention.sv            — Old attention v1 (superseded)
  expert_ffn_engine.sv        — Old FFN (superseded)
  c2c_node.sv                 — Bring-up C2C test node

Software:
  c_ref/prefill/cpu_prefill.{h,c}      — CPU AMX/AVX-512 fp8 GEMM
  c_ref/prefill/weight_preloader.c    — SSD→pinned memory loader
  scripts/prefill/coordinator.py      — Three-tier decision logic
  scripts/prefill/scheduler.py        — Concurrent CPU+FPGA scheduler
  scripts/prefill/vllm_prefill.py     — vLLM monkey-patch
```

Transport Layer A — C2C SerDes (Intra-card, 4 chips, see §6.3):
  Frame format: SOP(8b) + Header(64b) + CRC16 + Payload(≤4KB) + CRC32 + EOP(8b)
  Flow control: Credit-based, per-link independent
  Routing: 5-bit global addressing {CardID[2:0], ChipID[1:0]}
  Reliability: Link-layer CRC error detection + timeout retransmission

Transport Layer B — PCIe 5.0 P2P DMA (Inter-card, see §6.4):
  Protocol: PCIe 5.0 BAR4 64MB Memory-mapped Ring Buffer
  Initiator: Chip0 DMA Engine → Bus → Target card Chip0 BAR4
  Routing: Chip0 full-mesh P2P, CPU not in data path
  Reliability: PCIe TLP CRC + Ack/Nak link-layer retransmission
```

---

## 13. Endgame Roadmap: FPGA Validation → ASIC Tapeout

> **Core logic**: FPGA is not the final product form. FPGA is a low-risk, low-NRE
> architecture verification platform. After architecture verification passes →
> tapeout ASIC, hardware cost reduced by another 80-90%.

### 13.1 Why FPGA→ASIC Rather Than Direct ASIC?

```
Risks of direct ASIC tapeout:
  ● fp4+MLA architecture not silicon-verified → tapeout failure = $10-20M
    down the drain + 18 months wasted
  ● DeepSeek V5/V6 architecture may change → extremely high risk if ASIC hardcoded
  ● 12nm/7nm tapeout NRE $5-15M, revision $2-5M per iteration

Advantages of FPGA first:
  ● Agilex 7 M already has fp4 DSP + HBM → architecture verification cost ¥6.75M
    (RTL development)
  ● RTL modification = recompilation (8h), not re-tapeout ($10M)
  ● Seed customers running on FPGA → real workload data → guides ASIC design
  ● ASIC can strip all unnecessary FPGA overhead (programmable interconnect,
    redundant LE)
```

### 13.2 Post-ASIC Tapeout Cost Curve

> **1 ASIC = 4 AGM 039 FPGA combined into a single package**  
> 8 cards × 1 ASIC = 8 ASICs replace 32 FPGAs, total HBM capacity unchanged (1,024 GB)

```
ASIC tapeout parameters (12nm, 4 FPGAs merged into 1 die):
  Die area:          ~500-700 mm² (4× DSP array + HBM PHY ×8 + SRAM + NoC)
  HBM:               128 GB HBM3 (8×16Gb stack) or HBM2e 128 GB (fallback)
  HBM Bandwidth:     ~3.2 TB/s (HBM3) or ~1.8 TB/s (HBM2e, 4×460 GB/s)
  C2C SerDes:        On-die interconnect (original inter-chip SerDes routes become
                     intra-die buses, saving PHY area + power)
  PCIe 5.0 x16:      1× (hardened, Chip0 function only)
  DSP array:          ~50,000 MAC (4×12,300, hardened)
  MLA pipeline:      Hardened state machine + SRAM buffer

#### 13.2.1 HBM3 NRE Cost Breakdown

> HBM integration is the largest and least controllable single expense in tapeout NRE.
> The following is an itemized breakdown.

```
┌─────────────────────────────────────────────────────────────┬──────────────┐
│ NRE Item                                                      │ Cost (USD)    │
├─────────────────────────────────────────────────────────────┼──────────────┤
│ Part 1: ASIC Die Side (HBM-independent)                        │              │
│   12nm MPW Shuttle (incl. masks, ~600mm²)                    │ $3-5M        │
│   DDR/PCIe PHY IP (Synopsys/Cadence)                         │ $0.5-1M      │
│   Die design labor (12-18 months, RTL 70% reuse from FPGA)   │ $2-3M        │
│   Die sub-total                                               │ ~$5.5-9M     │
├─────────────────────────────────────────────────────────────┼──────────────┤
│ Part 2: HBM PHY + Controller IP                              │              │
│   HBM3 PHY IP (Synopsys/Cadence/Rambus, one-time License)    │ $2-5M        │
│   HBM3 Controller IP (incl. DFI interface, often bundled)    │ $1-2M        │
│   ⚠ HBM3 PHY typically supports 7nm and below only          │              │
│      12nm may only support HBM2e PHY → downgrade to HBM2e    │              │
│      If HBM3 on 12nm is mandatory → custom PHY → +$2-3M, +6 mo│              │
│   IP sub-total                                                │ $3-7M        │
├─────────────────────────────────────────────────────────────┼──────────────┤
│ Part 3: 2.5D Packaging (Interposer + CoWoS)                  │              │
│   Silicon Interposer design + masks (~2× reticle)            │ $1.5-3M      │
│   CoWoS/equivalent 2.5D packaging NRE (TSMC CoWoS-S)         │ $3-5M        │
│   Package substrate design + prototyping                     │ $0.5-1M      │
│   ⚠ CoWoS capacity is a shared bottleneck for NVIDIA/Ascend │              │
│      Fallback: Intel EMIB (already proven in Agilex 7 volume)│              │
│      Or Samsung I-Cube (ample capacity, weaker ecosystem)    │              │
│   Packaging sub-total                                        │ $5-9M        │
├─────────────────────────────────────────────────────────────┼──────────────┤
│ Part 4: HBM DRAM Stack Procurement                           │              │
│   HBM3 16Gb × 8 stacks (SK Hynix / Samsung)                  │ Not NRE       │
│   Volume unit price ~$30-50/stack, 8 stacks = $240-400/die   │ Included in   │
│   ⚠ HBM3 procurement requires vendor quota, MOQ ~1,000-5,000│ BOM           │
│      stacks. For MPW verification (5-10 ASICs), sample channel│              │
│      is available                                            │              │
│   First-run 10 ASIC HBM sample fee ~$50-100K                  │ One-time cost │
├─────────────────────────────────────────────────────────────┼──────────────┤
│ HBM-related NRE total                                         │ ~$8-16M      │
│ ASIC total NRE (Die + HBM + Packaging)                        │ ~$13.5-25M   │
└─────────────────────────────────────────────────────────────┴──────────────┘
```

**Key Risk: 12nm + HBM3 May Be Unrealistic**

```
Real-world constraints:
  ● HBM3 PHY IP (Synopsys/Cadence) standard offering only supports 7nm/5nm/3nm
  ● HBM3 PHY has no off-the-shelf IP on 12nm (TSMC 16FFC/12FFC) → requires
    full custom design → extremely high risk
  ● HBM2e PHY has mature solutions on 12nm (Intel Agilex 7 already in volume production)

  Pragmatic approach — downgrade to HBM2e:

  HBM2e vs HBM3 impact on ASIC:
                        HBM3 (7nm)           HBM2e (12nm)         Difference
                       ───────────          ────────────          ────
  PHY IP availability   7nm+ only            12nm mature            —
  IP License fee        $3-7M                $1-2M                 -50%+
  Per-stack bandwidth   819 GB/s              460 GB/s              -44%
  Total BW (8 stacks)   ~6.5 TB/s             ~3.7 TB/s            -43%
  Total BW (4 stacks)   ~3.2 TB/s             ~1.8 TB/s
  CoWoS requirement     Required (HBM3)       Required (HBM2e also) Same

  Conclusion: If choosing 12nm, HBM2e is the only realistic option.
             HBM3 requires 7nm ($20-30M NRE), but 12nm is lower risk.
             4 stacks HBM2e @12nm → 128 GB, 1.8 TB/s → throughput ~600 tok/s
             Slightly lower than FPGA, but cost collapses → compensate with
             more cards (16 cards vs 8 cards)
             Or accept 4 stacks HBM2e, 600 tok/s, sell at $50-60K
```

**Revised ASIC Cost Estimate (with HBM2e, 12nm):**

```
                                  4 stacks HBM2e      8 stacks HBM2e
                                  ──────────────      ──────────────
  Tapeout NRE (Die + HBM + Pkg)    ~$10-15M             ~$12-18M
  Per-ASIC cost (10K units):
    Die (12nm, 500mm²)             ~$250-350            ~$250-350
    HBM2e stacks                   4×$35 = $140         8×$35 = $280
    Interposer + CoWoS             ~$100-150            ~$150-200
    Packaging + Test               ~$30-50              ~$40-60
    Per-unit total                 ~$520-690            ~$670-840
                                  ≈ ¥3.8-5.0K          ≈ ¥4.9-6.1K

  Per-card BOM (1 ASIC + PCB/VRM)  ~$6-8K               ~$8-10K
  8-card system BOM                ~$55-70K             ~$75-95K
  Hardware price (50% margin)      ~$110-140K           ~$150-190K

  Decode throughput (8-card)       ~600 tok/s           ~1,000 tok/s
  $/M tokens                       ~$5-7                ~$5-7
```

```
Key findings:
  ● 8 stacks HBM2e: throughput ~1,000 tok/s, price ~$150-190K → more expensive
    than 950PR
  ● 4 stacks HBM2e: throughput ~600 tok/s, but price ~$110-140K → price parity
    with 950PR, but lower throughput
  ● HBM3 is needed for a clear advantage, but HBM3 is infeasible on 12nm
  ● To do HBM3 → must go 7nm → NRE $20-30M, higher risk

  ASIC's real value: the cost-collapse premise requires HBM PHY / process match.
  HBM2e@12nm ASIC cost-performance is actually inferior to 950PR (at same ~$110K
  price point, 950PR has higher throughput).

  ASIC's optimal paths:
    → HBM2e@12nm: price $150-190K, pricier than 950PR but exportable globally
    → HBM3@7nm: NRE $20-30M, truly $70-80K — but enormous risk
    → Or skip ASIC entirely: FPGA volume production is already competitive
```

#### 13.2.2 The Third Path: FPGA → eASIC (Structured ASIC)

> **eASIC is a stepping stone between FPGA and full-custom ASIC.**  
> Retains FPGA's validated HBM2e PHY + EMIB packaging, replacing only the
> programmable logic layer with fixed metal routing.  
> No need to redo HBM PHY, no need to switch packaging process, no need for
> new IP licenses.

```
eASIC principle:
  FPGA (SRAM-based):                     eASIC (metal-customized):
  ┌─────────────────────────┐            ┌─────────────────────────┐
  │ SRAM config bits          │            │ Fixed metal routing       │
  │ (large area + leakage)   │    →       │ (small area + power saved)│
  │ Programmable interconnect │            │ Hardwired interconnect    │
  │ (many MUXes)              │            │ (zero MUX delay)          │
  │ Programmable LUT          │            │ Fixed logic gates         │
  │ (redundant transistors)  │            │ (no redundancy)           │
  │ — Hard IP unchanged —    │            │ — Hard IP unchanged —    │
  │   HBM2e PHY ✓           │            │   HBM2e PHY ✓            │
  │   DSP Blocks ✓          │            │   DSP Blocks ✓            │
  │   M20K SRAM ✓           │            │   M20K SRAM ✓             │
  │   EMIB ✓                │            │   EMIB ✓                  │
  │   SerDes ✓              │            │   SerDes ✓                 │
  └─────────────────────────┘            └─────────────────────────┘

  Scope of change:         Metal layers / via layers only (2-4 mask layers)
  Unchanged:               Transistor layer, HBM PHY, EMIB bumps, SerDes, DSP hard IP
  Equivalent process:      Maintain Intel 7 (identical to Agilex 7 M)
  HBM:                     Maintain HBM2e 32GB (validated 2-stack EMIB)
```

**eASIC vs Full-Custom ASIC vs FPGA Volume Production Comparison:**

```
┌──────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                       │ FPGA Volume       │ eASIC (Structured)│ Full-Custom ASIC  │
│                       │ (Agilex 7 M)     │ (Intel 7, same   │ (12nm/7nm)       │
│                       │                  │ process)          │                  │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ NRE                   │ ¥0 (existing chip)│ **$2-5M**         │ $13-25M          │
│                       │                  │ (metal masks only)│ (full masks + IP │
│                       │                  │                  │ + packaging)     │
│ Development cycle     │ 0 (off-the-shelf) │ **6-9 months**    │ 18-24 months      │
│ HBM solution          │ HBM2e 2-stack    │ HBM2e 2-stack     │ HBM3 needs new PHY│
│                       │ EMIB (validated)  │ EMIB (unchanged!) │ CoWoS (must switch)│
│ Packaging             │ EMIB (Intel)     │ EMIB (unchanged!) │ CoWoS (TSMC bottleneck)│
│ HBM PHY redesign      │ N/A              │ ✗ Not needed      │ ✓ Needs $3-7M IP │
│ Per-chip cost         │ ¥25K ($3.4K)    │ **$1.0-1.5K**     │ $0.5-0.8K        │
│                       │                  │ (saves SRAM+      │ (new die)        │
│                       │                  │  programmable)    │                  │
│ Per-chip power        │ ~120W            │ **~70W** (-42%)   │ ~80W             │
│ 4-chip→1-chip merge?  │ 4 chips/card     │ Merge optional    │ Must merge (NRE   │
│                       │                  │                  │ amortization)    │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 4 FPGA/eASIC → 1 die:│                  │                  │                  │
│  Chip cost            │ 4×¥25K=¥100K    │ **$3-5K**         │ $0.7-0.8K        │
│                       │                  │ (4→1, 128GB)     │ (HBM3 NRE excl.)  │
│  Card-level cost      │ ¥45K/card        │ **¥15K/card**     │ ¥6K/card          │
│  8-card BOM           │ ~¥1.5M           │ **~¥0.35-0.45M** │ ~¥0.25M           │
│  8-card price (50% GM)│ ~¥2.2M ($303K)   │ **~$100-130K**    │ ~$150-190K        │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Throughput (8-card)   │ 800-980 tok/s    │ ~900-1,100 tok/s │ ~900-1,100 tok/s  │
│                       │                  │ (same process,   │ (HBM3 BW higher) │
│                       │                  │  slightly better)│                  │
│ $/M tokens            │ $5.9             │ **$3.5-4.5**     │ $5-7              │
│ Availability          │ ✓                │ ✓                │ △ (CoWoS queue)   │
│ Global deployment     │ ✓                │ ✓                │ ✓                 │
└──────────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**Why eASIC May Be the Optimal Path:**

```
1. Bypasses the HBM3 quagmire:
   → Reuses FPGA's validated HBM2e PHY + EMIB; zero HBM-related NRE
   → No need to compete with NVIDIA/Ascend for CoWoS capacity (EMIB capacity ample)
   → No need to sign HBM3 PHY IP License ($3-7M saved)

2. 4→1 merge yields 128GB HBM2e — sufficient:
   → 8 cards × 128GB = 1,024 GB (identical to FPGA 32-chip configuration)
   → Total bandwidth 8×4×460GB/s ≈ 14.7 TB/s — lower than FPGA's 29.4 TB/s
   → But on-die interconnect (original C2C becomes intra-die bus) eliminates
     ~73μs/layer latency
   → Throughput estimate: 900-1,100 tok/s (BW lower but latency improved, roughly flat)

3. NRE is affordable:
   → $2-5M vs full-custom ASIC $13-25M
   → Phase 3 seed customer revenue can cover the majority
   → Can launch without a new funding round

4. Time window:
   → FPGA volume (Month 12) → eASIC development (Month 12-21) → eASIC volume (Month 24)
   → eASIC product deliverable in 24 months vs 36+ months for full-custom ASIC

5. Extremely low risk:
   → Functionality already validated on FPGA → eASIC merely "freezes" the same circuit
   → Same process (Intel 7) → no process migration risk
   → Same packaging (EMIB) → no packaging re-certification
```

**Recommended Path: FPGA Volume → eASIC → (Long-term) Full-Custom ASIC**

```
Phase 1-3: FPGA prototype verification + seed customers (12 months, ¥6.75M R&D)
Phase 4:   FPGA 10-100 unit volume delivery + initiate eASIC conversion (Month 12-18)
Phase 5:   eASIC silicon back + volume production (Month 24, NRE $2-5M)
           → price $100-130K, $/token $3.5-4.5
Phase 6:   (Long-term) If market validation >1,000 units/year → full-custom ASIC HBM3@7nm
           → price $70-80K, $/token $2.5-3.5

Key Gate:
  Scenario for skipping eASIC: FPGA volume already achieves $5.9/M tokens,
  superior to H100 ($12-20) and 950PR ($16-25)
  → If FPGA volume satisfies customers, skip eASIC and go directly to full-custom ASIC
  → eASIC is the cost-reduction path if customers are sensitive to the $303K price
```

### 13.3 Phased Roadmap

```
Phase 1 (Month 1-2):   Single-card FPGA verification
  → fp4 precision meets target, HBM bandwidth meets target, DSP utilization >35%
  → Investment: ¥0.5M (dev board + labor)

Phase 2 (Month 3-5):   2-4 card FPGA verification
  → C2C SerDes ring measured, PCIe P2P DMA communication verified
  → Investment: ¥1.5M (4 cards + server)

Phase 3 (Month 6-12):  8-card full-config seed customer deployment
  → 3-5 seed customers, real workload data accumulation
  → Investment: ¥3-5M (10 units hardware + software adaptation)

══════════════════════════════════════════════
  Gate 1: Phase 3 data → decide whether to start ASIC
  Criterion: seed customer renewal rate >60%, $/token meets targets
══════════════════════════════════════════════

Phase 4 (Year 2):      ASIC design + MPW tapeout
  → Optimize microarchitecture based on Phase 3 real workload data
  → RTL reuse rate >70% (fp4 DSP datapath, MLA pipeline, comm protocols unchanged)
  → Investment: $8-12M (tapeout NRE + design labor)

Phase 5 (Year 2-3):    ASIC silicon back → volume production
  → First batch 100 units for seed customer upgrade
  → Cost ~$70-80K/unit (FPGA-validated architecture IP scaled on ASIC)
  → Throughput flat or slightly better (HBM3 bandwidth + on-die interconnect replaces C2C)
  → Power 3.8kW vs FPGA 5.3kW (another 28% reduction)
```

### 13.4 Competitive Landscape Post-ASIC

```
┌────────────────────┬──────────┬──────────┬──────────┬──────────────┐
│                     │ H100     │ 950PR    │ FPGA     │ ASIC (Ours)   │
├────────────────────┼──────────┼──────────┼──────────┼──────────────┤
│ Hardware price (unit)│ $280K   │ $110K    │ $303K    │ **$70-80K**   │
│ Total HBM BW        │ 26.8 TB/s│ 16 TB/s  │ 29.4 TB/s│ 25.6 TB/s     │
│ Decode throughput   │ 600-800  │ 1,500-2K │ 800-980  │ 900-1,100     │
│   (est.)            │          │          │          │              │
│ $/M tokens (HW)     │ $12-20   │ $16-25   │ $5.9     │ **$2.5-3.5**  │
│ Availability        │ ✗ Controlled│ △ Queued│ ✓        │ ✓             │
│ Global deployment   │ △        │ ✗        │ ✓        │ ✓             │
│ fp4 native          │ B200+    │ ✗        │ ✓        │ ✓             │
│ Supply chain        │ TSMC+    │ SMIC     │ Intel    │ TSMC/SMIC     │
│                     │ Samsung  │ constrained│        │ selectable    │
└────────────────────┴──────────┴──────────┴──────────┴──────────────┘

Post-ASIC tapeout:
  → Hardware cost $70-80K (vs FPGA $303K), but this is not the selling point
  → The real selling point: architectural bandwidth efficiency (83× effective BW @ B=1)
    + hardware price cliff (< $80K)
  → Two orders-of-magnitude dimensions appearing simultaneously → competitive
    landscape shifts from "technology selection" to "paradigm replacement"
  → Even if GPU/NPU prices drop to $80K, 83× effective BW gap at B=1 remains
    (that is determined by SIMT architecture)
  → Even if GPU/NPU improve B=1 efficiency and $/token reaches FPGA levels →
    gross margin goes to zero
  → Supply chain selectable: TSMC (12nm non-advanced node, not controlled) or SMIC
  → Core IP (fp4 DSP datapath + MLA pipeline) already validated in FPGA phase
```

### 13.5 Risks and Responses

```
Risk 1: DeepSeek architecture major change → ASIC obsolete
  Response: FPGA validates new architecture first → RTL iteration → tapeout
           only after stability confirmed
           FPGA is a reprogrammable "architecture lab"; ASIC is the "frozen snapshot"

Risk 2: Tapeout failure
  Response: MPW shuttle (multi-project wafer) spreads risk; first revision does
           not target production-grade yield
           RTL already proven on FPGA → functional correctness risk is extremely low
           12nm mature process, not 3nm/5nm adventure

Risk 3: $8-12M tapeout funding
  Response: Phase 3 seed customer revenue can cover part of NRE
           Or take Chiplet route: first tapeout "fp4 DSP compute die" (smaller, ~$5M),
           then integrate HBM PHY + SerDes

Risk 4: Huawei/NVIDIA also release fp4 ASIC
  Response: fp4 is only one dimension. MLA hardening + HBM/compute golden ratio
           is the deeper moat. And the RTL IP + customer cases accumulated during
           the FPGA phase cannot be rapidly replicated.
```

---

> **Conclusion**: The FPGA compute cluster proposal is technically feasible and strategically necessary.  
> It is the only hardware path that can deploy China's LLM inference capabilities to the global market without depending on controlled supply chains.  
> 
> **But FPGA is not the endgame. FPGA is a low-risk verification platform. Verification passes → 4 FPGAs merged into 1 ASIC tapeout → hardware cost $70-80K/unit, $/token $2.5-3.5.**  
> **Throughput is maintained (HBM bandwidth determines it), cost collapses with scale. ASIC's value lies in: the architectural advantages verified in the FPGA phase (83× effective BW, 1000× switching latency, 1000× KV address resolution) are physically hardened in ASIC, and scale effects push costs to a floor that GPU/NPU cannot follow.**
>
> **Next step**: Initiate Phase 1 single-card verification, especially the two key metrics of fp4 precision and HBM bandwidth.
