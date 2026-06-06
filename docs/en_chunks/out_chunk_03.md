### 4.8 Prefill Performance Analysis and Scheduling Strategy

Multiple sections of the proposal mention "Prefill dominates on GPU" and "Prefill accounts for only ~5% of inference time." A reasonable challenge from reviewers: TTFT (Time-To-First-Token) is the only phase of user-perceived latency — no matter how fast decode runs at 800 tok/s, a TTFT of 5 seconds is unacceptable. This section quantitatively analyzes the prefill performance boundaries of the FPGA cluster.

**4.8.1 TTFT Estimation**

```
Prefill is a compute-bound scenario:
  MAC per token per layer: ~611M (see §4.2)
  30 FPGA aggregate: 8.44 × 30 = 253 TMACs/s

Based on a 4-node pipeline (slowest node determines throughput):
  Node 0 (TP=7, 15 layers): P × 15 × 611M / 7  = P × 1.31T MACs
  Node 1 (TP=8, 15 layers): P × 15 × 611M / 8  = P × 1.15T MACs
  Node 2 (TP=8, 15 layers): P × 15 × 611M / 8  = P × 1.15T MACs
  Node 3 (TP=7, 16 layers): P × 16 × 611M / 7  = P × 1.40T MACs  ← bottleneck

  P = number of prompt tokens
  Bottleneck node time = P × 1.40T / 8.44T = P × 166 ms
  Pipeline fill ≈ 4 stages × 10 ms = 40 ms

  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │ Prompt Length │ Compute Time │ TTFT (w/ fill)│ vs H100*    │
  ├──────────────┼──────────────┼──────────────┼──────────────┤
  │ 200 tokens   │ 33 ms        │ ~70 ms       │ ~5 ms        │
  │ 512 tokens   │ 85 ms        │ ~125 ms      │ ~8 ms        │
  │ 1K tokens    │ 166 ms       │ ~210 ms      │ ~12 ms       │
  │ 4K tokens    │ 664 ms       │ ~800 ms      │ ~50 ms       │
  │ 8K tokens    │ 1.33 s       │ ~1.4 s       │ ~100 ms      │
  │ 16K tokens   │ 2.66 s       │ ~2.7 s       │ ~200 ms      │
  │ 32K tokens   │ 5.31 s       │ ~5.4 s       │ ~400 ms      │
  │ 128K tokens  │ 21.2 s       │ ~21.3 s      │ ~1.6 s       │
  └──────────────┴──────────────┴──────────────┴──────────────┘
  * H100 estimated as 8×990 TFLOPs / 2 (MoE sparsity ≈ 50% MAC utilization)
```

**4.8.2 Chunked Prefill: Why TTFT ≠ Full Prefill Latency**

```
Borrowing the Chunked Prefill strategy from vLLM/Sarathi:

  Long prompt → split into 512-token chunks
  decode_step → prefill_chunk → decode_step → prefill_chunk → ...

  Key effect:
    First chunk (512 tok) TTFT: ~125 ms  ← user-perceived latency
    Full prefill continues in the background, interleaved with decode

  128K prompt scenario:
    Full prefill theory: 21.3s
    Chunked: first token visible at 125ms
    256 chunks × 85ms = 21.8s completed in background
    User sees the first token begin generating within 125ms

  Applicability conditions:
    ✓ B=1 decode scenarios (agent, chatbot)
    ✗ Scenarios requiring full prefill completion before decode can begin (extremely rare)
```

**4.8.3 Applicability Boundaries of "Prefill accounts for 5%" — Tiered by Workload**

```
This figure only holds for short-prompt chatbots. It actually depends on context length:

  ┌──────────────────┬─────────┬──────────┬────────┬──────────────┐
  │ Workload          │ Prompt  │ Response │ TTFT   │ Prefill Share│
  ├──────────────────┼─────────┼──────────┼────────┼──────────────┤
  │ Short Q&A (ChatGPT)│ 200    │ 2000     │ 70ms   │ ~0.2%        │
  │ Customer Service  │ 1K      │ 500      │ 210ms  │ ~4%          │
  │ RAG (Retrieval)   │ 8K      │ 500      │ 1.4s   │ ~14%         │
  │ Code Review       │ 16K     │ 2000     │ 2.7s   │ ~7%          │
  │ Doc Summarization │ 32K     │ 1000     │ 5.4s   │ ~24%         │
  │ Long-form Writing │ 10K     │ 8000     │ 1.7s   │ ~1%          │
  │ Agent (multi-turn)│ 5-20K   │ 300/turn │ 0.8-3s │ ~15-30%      │
  └──────────────────┴─────────┴──────────┴────────┴──────────────┘

  "5%" only holds for the first row.
  The private deployment customers (finance/healthcare/government) that this proposal targets are primarily:
    Customer Service, RAG, Agent — medium prompts, TTFT requires attention.
```

**4.8.4 GPU Prefill Advantage: Frank Acknowledgment**

```
┌──────────────┬───────────────┬───────────────┬──────────────────┐
│ Prefill (4K)  │ 8×H100 (FP8) │ 8×B200 (FP4) │ 30 FPGA (fp4)    │
├──────────────┼───────────────┼───────────────┼──────────────────┤
│ TTFT          │ ~50ms         │ ~25ms         │ ~800ms           │
│ Cost          │ $240K (embargo)│ $320K (embargo)│ $321K (obtainable)│
│ Ratio vs FPGA │ 16× faster    │ 32× faster    │ 1×               │
│ China access  │ ✗            │ ✗             │ ✓                │
└──────────────┴───────────────┴───────────────┴──────────────────┘

GPU's absolute advantage in prefill comes from physics: high Tensor Core utilization
under large batch sizes, with compute density 3-9× that of FPGA. This is indisputable fact.

Three honest counterpoints:
  (a) H100/B200 are not obtainable for the China market. Comparing "who is faster"
      is less relevant than comparing "who is usable."
  (b) Chunked prefill makes first-token latency far lower than full prefill latency.
      Over 80% of commercial prompts are < 4K tokens, TTFT < 800ms is acceptable.
  (c) Among domestically obtainable hardware (Ascend 910C, etc.), prefill performance
      also does not approach H100. FPGA's prefill disadvantage is relative to
      "unobtainable GPUs," not relative to competitors.
```

**4.8.7 CPU Prefill Rebuttal: Memory Bandwidth & Storage**

A common challenge: "CPU memory bandwidth cannot keep up" + "weight storage is too large."
Each point is addressed below with concrete numbers.

```
Challenge 1: "61 layers of weights are too large, cannot fit in CPU memory"

  Weight per layer (fp8 uncompressed):  ~135 MB
  61 layers total (fp8):                ~8.2 GB
  If fp4 compressed:                    ~4.1 GB
  Typical server RAM:                   256 GB (Dual Xeon GNR)

  → 61 layers of weights occupy only 3.2% of RAM.
  → Even with simultaneous KV cache (128K tokens × 61 layers = 8 GB),
    total is 16 GB, only 6%.
  → "Cannot fit" does not hold.


Challenge 2: "DDR5 memory bandwidth cannot handle GEMM"

  Take W_Q [P=128, 7168] × [7168, 7168] as an example:
    Compute:   128 × 7168² = 6.6 GMACs
    Compute time: 6.6G / 10.5T = 0.63 ms
    Weight load: 7168² × 1B = 51.4 MB
    Memory time: 51.4 MB / 307 GB/s = 0.17 ms

    → Compute/memory ratio = 0.63/0.17 = 3.7x
    → This is a compute-bound operation, not memory-bound

  Full P=128 chunk, 61 layers:
    Total weights:    8.2 GB
    Memory time:      8.2 GB / 307 GB/s = 27 ms
    Compute time:     395 ms (calibrated)
    → Compute/memory ratio = 395/27 = 14.7x
    → Compute is 15× slower than memory!

  Why is memory not the bottleneck?
    Batch P=128: each weight byte is reused by 128 tokens.
    Effective memory demand: 51.4 MB / 128 = 0.4 MB/token weight bandwidth.
    DDR5 307 GB/s / 10.5 TFLOPS = 29 bytes/FLOP available.
    W_Q requires: 51.4 MB / 6.6 GMACs = 0.008 bytes/FLOP.
    Available/required = 29 / 0.008 = 3625× headroom.
    → "Bandwidth cannot keep up" does not hold.


Challenge 3: "KV cache too large for 128K ultra-long context"

  KV per token:  K_latent(512) + V_latent(512) = 1024B fp8
  128K tokens:   128K × 1024 = 131 MB per layer
  61 layers:     61 × 131 MB = 8.0 GB
  Plus weights:  8.2 + 8.0 = 16.2 GB

  → Only 6% of 256 GB RAM.
  → For chunked prefill: only need to store current chunk's KV (P=128 → 8 MB/layer)


The real bottleneck for CPU Prefill is compute, not memory:

  ┌──────────────────┬──────────┬──────────┬──────────────┐
  │ Constraint        │ Required │ Available│ Headroom      │
  ├──────────────────┼──────────┼──────────┼──────────────┤
  │ Weight storage    │ 8.2 GB   │ 256 GB   │ 31x           │
  │ KV Cache (128K)   │ 8.0 GB   │ 256 GB   │ 32x           │
  │ DDR5 bandwidth    │ 27 ms    │ 395 ms   │ 14.7x         │
  │ CPU compute (TF)  │ 395 ms   │ -        │ Bottleneck!   │
  └──────────────────┴──────────┴──────────┴──────────────┘
```



**4.8.6 2026 CPU Prefill Assessment — Hardware Has Caught Up**

> Update (2026/05): CPU prefill compute has jumped from ~1.7 TFLOPS (SPR) to 10+ TFLOPS (GNR/Turin).
> The gap has narrowed from 11× to 2×. CPU prefill can now cover 80% of commercial scenarios.

```
Prefill Performance of Purchasable CPUs in 2026 (P=128 chunk, DeepSeek V4 Pro, 61 layers):

┌──────────────────────────────┬──────────┬──────────┬──────────────┐
│ CPU                           │ Eff. TF   │ P=128 TTFT│ vs FPGA Decode│
├──────────────────────────────┼──────────┼──────────┼──────────────┤
│ Dual Xeon 6980P (GNR, 128c)  │ 10.5 TF   │  396 ms  │ 2.0x slower   │
│ Dual EPYC 9755 (Turin, 128c) │ 10.5 TF   │  396 ms  │ 2.0x slower   │
│ Dual EPYC 9965 (Turin, 192c) │  9.0 TF   │  462 ms  │ 2.4x slower   │
│ Quad Xeon 6980P (4-socket)   │ 18.2 TF   │  228 ms  │ 1.2x slower   │
│ Dual Xeon 8592+ (SPR, 2023)  │  1.7 TF   │ 2473 ms  │ 11x slower (ref)│
│ 1x A100 (GPU, fp16)           │ 187  TF   │   22 ms  │ 18x faster (embargo)│
└──────────────────────────────┴──────────┴──────────┴──────────────┘

Strategy by Scenario (2026):

┌──────────────────────┬───────────┬──────────────┬────────────────┐
│ Scenario              │ Prompt    │ Recommended  │ TTFT (GNR)     │
├──────────────────────┼───────────┼──────────────┼────────────────┤
│ Short Q&A / Chat      │ < 200     │ CPU full     │ 0.4-0.8s        │
│ Chat (short)          │ 200-500   │ CPU full     │ 0.8-1.6s        │
│ Agent warm (incremental)│ +500-2K │ CPU incr. ✅ │ 1.6-6.3s (incr.)│
│ RAG / Customer Srv    │ 1-2K      │ FPGA chunked │ first chunk 85ms│
│ Code Review           │ 10-20K    │ FPGA chunked │ first chunk 85ms│
│ Long doc / 128K ctx   │ 32-128K   │ FPGA chunked │ first chunk 85ms│
│ Ultra-low-latency TTFT│ Any       │ +GPU (A100)  │ < 50 ms         │
└──────────────────────┴───────────┴──────────────┴────────────────┘

> **Note**: TTFT values in the table above have been corrected per §4.8.8 audit results (v1.4).
> The original v1.3 version systematically underestimated CPU prefill TTFT (conflating "first chunk completion time" with "first token generation time").
> After correction, the practical range of CPU prefill shrinks from <4K to <500 tokens (or Agent warm start incremental mode).
> Medium-to-long prompts uniformly use FPGA Tier 2 chunked prefill.

BOM Impact:

┌──────────────────────┬───────────┬──────────────┬────────────────┐
│ Option                │ Added Cost│ Prefill Gain  │ Recommendation│
├──────────────────────┼───────────┼──────────────┼────────────────┤
│ Current SPR (existing)│ 0          │ Baseline      │ 2023 baseline  │
│ -> Upgrade GNR 6980P  │ +30K/CPU   │ x6 speedup    │ Recommended   │
│ -> Upgrade EPYC 9755  │ +25K/CPU   │ x6 speedup    │ Recommended   │
│ -> 4-Socket GNR       │ +60K+mobo  │ x11 speedup   │ Max perf       │
│ -> Add 1xA100 GPU     │ +80K       │ x30 speedup   │ Fastest, embargo│
└──────────────────────┴───────────┴──────────────┴────────────────┘
```


### 4.8.8 CPU Prefill + FPGA Decode: Full Feasibility Audit

> Core question: Is the CPU Prefill + FPGA Decode hybrid architecture truly viable?
> Short answer: Viable, but with strict prerequisites. The analysis direction of the current document (§4.8.6/§4.8.7/§14.E) is correct,
> but there are two gaps: **systematic underestimation of TTFT figures** and **incomplete description of key data paths**.
> Each is audited below.

**A. CPU Compute Feasibility — Verified ✅**

```
The compute analysis in §4.8.7 is correct and requires no revision:

  Dual Xeon GNR 6980P effective fp8 compute: ~10.5 TFLOPS (AMX BF16→fp8 converted)
  P=128 single chunk 61-layer compute:        ~4.1 GMACs × 61 = ~250 GMACs
  Compute time:                               ~395ms (including AMX tile config + data movement overhead)
  DDR5 bandwidth headroom:                    14.7× (307 GB/s vs 20.8 GB/s required)

The bottleneck for CPU prefill is compute, not memory bandwidth. For chunks with P≤512,
each weight byte is reused ≥128 times → compute-bound. This point is fully substantiated in §4.8.7.

However, there is an implicit assumption that needs explicit confirmation:

  Weight preloading (8.2 GB fp8) takes 27ms (§4.8.7 line 1670).
  This load happens only once at session startup (or on model switch).
  During steady-state operation, weights reside in CPU pinned memory and are not reloaded.
  → No impact on steady-state TTFT, but cold-start first-request TTFT = 395 + 27 = 422ms.
  → The document's 396ms is the correct steady-state figure, but should be labeled as "steady-state" rather than "first request."
```

**B. KV Cache DMA + 32-Chip Distribution Path — Viable but Incompletely Described ⚠️**

```
This is the largest architectural gap in the current document: §14.E only says "CPU prefill → PCIe DMA → FPGA HBM double-buffered,"
but does not describe how KV cache reaches Chips 1-31 from Chip 0.

Only Chip 0 has a PCIe connection. Chips 1-31 must obtain their respective KV caches
via SERDES pipeline forwarding. The following completes this path:

  ┌─────────────────────────────────────────────────────────────────┐
  │ CPU Prefill complete → KV latent (576B/token/layer, fp8)        │
  │                                                                  │
  │ Step 1: CPU → Chip 0 (PCIe 5.0 x16, ~28 GB/s)                   │
  │   P=128 chunk: 128 × 576B × 61 = 4.5 MB → DMA ~0.16ms           │
  │   128K full:    128K × 576B × 61 = 4.5 GB → DMA ~161ms          │
  │                                                                  │
  │ Step 2: Chip 0 retains KV for layers 0-1 (2/61 ≈ 148 KB)        │
  │         Forwards remaining 59/61 ≈ 4.35 MB → Chip 1 (SERDES 56 GB/s)│
  │                                                                  │
  │ Step 3: Chip k retains KV for layers 2k-2k+1, forwards remainder │
  │         Per-hop data volume decreases: 4.35→4.2→4.1→...→0 MB     │
  │         Per-hop latency: ~75ns (SERDES) + data/56GB/s            │
  │                                                                  │
  │ Step 4: Chip 31 receives KV for the last 2 layers (~148 KB)     │
  │                                                                  │
  │ Total pipeline distribution latency (P=128 chunk):               │
  │   DMA (CPU→Chip0):     0.16 ms                                   │
  │   31-hop forwarding:   ~1.24 ms (first hop carries most data,    │
  │                        last hop carries least)                   │
  │   Total:               ~1.4 ms ← negligible vs 395ms compute    │
  │                         (0.35%)                                  │
  │                                                                  │
  │ Total pipeline distribution latency (128K full, CPU completes    │
  │ full prefill then distributes):                                  │
  │   DMA (CPU→Chip0):     161 ms                                    │
  │   31-hop forwarding:   ~155 ms (4.5 GB / 56 GB/s × avg factor)  │
  │   Total:               ~316 ms ← significant! but must complete  │
  │                        before decode can begin                   │
  └─────────────────────────────────────────────────────────────────┘

Key findings:
  1. Chunked prefill (P=128): KV distribution overhead is only 1.4ms, completely negligible.
  2. Full prefill (128K): KV distribution overhead is 316ms, non-negligible.
     But 128K full prefill is itself impractical (compute would take ~395s),
     so chunked prefill is always used in practice.
  3. Pipeline forwarding can proceed in parallel with chunk N+1's CPU compute time,
     further hiding latency. However, the first chunk's forwarding is on the TTFT critical path.

Conclusion: The KV distribution path is viable with manageable overhead. The document should supplement this path description.
```

**C. End-to-End TTFT True Decomposition — Document Figures Need Revision 🔴**

```
This is the most serious gap. The scenario table in §4.8.6 lists figures like "TTFT ~395ms" and "TTFT 0.8-1.5s,"
but does not distinguish between "first chunk completion time" and "first token generation time."

Chunked prefill TTFT = prefill time for ALL chunks + first decode time.
It is NOT the first chunk completion time!

True decomposition (Dual GNR 6980P, P_chunk=128):

  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
  │ Phase                │ Latency  │ Share    │ Cumulative│ Notes   │
  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ Tokenize + Embed     │ 2-5 ms   │ -        │ 5 ms     │ CPU single-thread│
  │ Weight preload (cold) │ 27 ms   │ -        │ 32 ms    │ first request only│
  │ AMX GEMM chunk 1     │ 395 ms   │ 98.5%    │ 427 ms   │ P=128    │
  │ KV DMA → Chip 0      │ 0.16 ms  │ 0.04%    │ 427 ms   │ 4.5 MB   │
  │ KV forwarding 31-hop │ 1.24 ms  │ 0.3%     │ 428 ms   │ SERDES   │
  │ FPGA decode step 1   │ 1.4 ms   │ 0.3%     │ 430 ms   │ B=1      │
  │ Token → Host         │ <1 ms    │ -        │ ~430 ms  │ PCIe     │
  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘
  → P≤128 short prompt: TTFT ≈ 430ms (steady-state), close to the ~396ms claimed in document ✓

  But for longer prompts, multiple chunks are needed:

  ┌──────────────┬──────────┬─────────────────────┬──────────────────┐
  │ Prompt Length │ Chunks   │ True TTFT (computed) │ Document Claim   │
  ├──────────────┼──────────┼─────────────────────┼──────────────────┤
  │ 200 tokens   │ 2×P=128  │ ~0.8s               │ < 300 ms ✗       │
  │ 500 tokens   │ 4×P=128  │ ~1.6s               │ ~600 ms ✗        │
  │ 1,000 tokens │ 8×P=128  │ ~3.2s               │ -                │
  │ 2,000 tokens │ 16×P=128 │ ~6.3s               │ 0.8-1.5s ✗       │
  │ 4,000 tokens │ 32×P=128 │ ~12.6s              │ ~395ms ✗✗        │
  │ 128K tokens  │ 1000×P=128│ ~395s (6.6 min)     │ first chunk 125ms*│
  └──────────────┴──────────┴─────────────────────┴──────────────────┘

  *The document's "first chunk 125ms" is the FPGA Tier 2 figure, used for >4K scenarios.
   CPU Tier 1's "first chunk ~400ms" is 395ms compute + 5ms overhead.

  Corrected Scenario TTFT Table (CPU prefill, Dual GNR 6980P, P_chunk=128):

  ┌──────────────────────┬───────────┬──────────────┬──────────────────┐
  │ Scenario              │ Prompt    │ True TTFT     │ Recommended      │
  ├──────────────────────┼───────────┼──────────────┼──────────────────┤
  │ Short Q&A / Chat      │ < 200     │ 0.4-0.8s     │ CPU ✅            │
  │ RAG / Customer Srv    │ 200-500   │ 0.8-2.0s     │ CPU ✅ (borderline)│
  │ Agent incremental     │ 500-2K    │ 2.0-6.3s     │ CPU ⚠️ (marginal) │
  │ Multi-turn Agent (warm)│ 2K-5K inc│ 0.8-2.0s*    │ CPU ✅ (*incremental only)│
  │ Code Review           │ 5-20K     │ -            │ FPGA Tier 2 ✅    │
  │ Long doc / 128K       │ >4K       │ -            │ FPGA Tier 2 ✅    │
  └──────────────────────┴───────────┴──────────────┴──────────────────┘

  *Agent warm start: prefix KV cache reused, only prefill new tokens (typically 500-2K).

Key corrections:
  1. §4.8.6 "Short Q&A < 200: TTFT < 300 ms" → should be "0.4-0.8s"
  2. §4.8.6 "RAG 1-2K: TTFT 0.8-1.5s" → should be "3.1-6.3s"
     Such slow TTFT means RAG > 500 tokens should use FPGA Tier 2, not CPU
  3. §4.8.6 "Agent 2-5K: CPU chunked, TTFT 1.5-4s" → should be "6.2-15.4s"
     For a 5K prompt, CPU prefill would take ~15.4s, unacceptable.
     But Agent warm start only needs to prefill new tokens → practically acceptable
  4. The Tier 1/Tier 2 threshold should be lowered from 4K to ~500 tokens
     (500 tokens → 4 chunks × 395ms = 1.6s, already at the user-perception boundary)

Conclusion: CPU prefill is only suitable for short prompts (<500 tokens) and Agent warm start (incremental prefill).
           Medium-to-long prompts must use FPGA Tier 2. The document's scenario table needs substantial revision.
```

**D. CPU/FPGA Concurrent Scheduling Correctness — Viable but Insufficient Detail ⚠️**

```
Scenario: Request A is being decoded on FPGA, while Request B's prompt requires CPU prefill.
          The CPU must simultaneously handle prefill (AMX GEMM) + decode coordination (token dispatch/collection) + NIC traffic.

Resource allocation analysis:

  CPU core allocation (Dual GNR, 128C/256T):
    ┌─────────────────────┬──────────┬─────────────────────────────┐
    │ Task                 │ Cores    │ Notes                        │
    ├─────────────────────┼──────────┼─────────────────────────────┤
    │ AMX GEMM (prefill)   │ 64-96C   │ AMX one tile per core       │
    │ Decode coordination  │ 2-4C     │ Token dispatch, KV swap     │
    │ NIC interrupt/poll   │ 2-4C     │ Network I/O                 │
    │ OS + vLLM scheduler  │ 4-8C     │ Scheduling, memory mgmt     │
    │ Remaining (headroom) │ 16-56C   │ Burst handling              │
    └─────────────────────┴──────────┴─────────────────────────────┘

  DDR5 bandwidth allocation (307 GB/s total, 8-channel):
    ┌─────────────────────┬──────────┬─────────────────────────────┐
    │ Consumer             │ Bandwidth│ Share                        │
    ├─────────────────────┼──────────┼─────────────────────────────┤
    │ CPU prefill GEMM     │ ~21 GB/s │ 6.8% (weight streaming read)│
    │ KV cache DMA (PCIe)  │ ~0.1 GB/s│ 0.03% (chunked, avg)       │
    │ NIC TX/RX            │ ~5 GB/s  │ 1.6% (2×25GbE)              │
    │ OS + other           │ ~5 GB/s  │ 1.6%                        │
    │ Remaining            │ ~276 GB/s│ 90% ← ample                 │
    └─────────────────────┴──────────┴─────────────────────────────┘

  Correctness risks:
    1. AMX register state: AMX tile configuration must be saved/restored between prefill chunks.
       XSAVE/XRSTOR overhead is ~5-10μs → negligible if switching once per chunk.
       But if decode coordination requires frequent prefill interruption → switching overhead accumulates.

    2. KV cache double-buffer swap timing (§14.E):
       "Atomic swap: switch when B is ready and A is exhausted"
       Not clarified: does the swap happen between decode steps or can it interrupt an ongoing decode?
       If swap cannot happen mid-decode:
         → swap can only execute during decode step gaps (~1.4ms window)
         → swap itself takes < 10μs (PCIe write of a flag + FPGA interrupt)
         → negligible
       If decode is reading buf A while CPU finishes writing buf B:
         → CPU sets "B ready" flag
         → FPGA checks flag at the start of the next decode step
         → Atomically switches to buf B
         → No need to immediately interrupt decode

    3. KV cache isolation for concurrent sessions:
       Multiple sessions' CPU prefill produce their own KV caches.
       FPGA HBM requires partitioned management (per-session KV regions).
       vLLM already has PagedAttention block table management — this can be reused.

  Signal path for CPU prefill completion → FPGA notification:
    (a) CPU writes "prefill_done" flag to Chip 0's PCIe BAR
    (b) Chip 0 receives it, checks whether KV buf B is complete
    (c) At the start of the next decode step, swap buf A ↔ buf B
    (d) Begin decoding with the new KV cache

  Multi-session concurrency timing example:

    t=0     : Session A decoding on FPGA (step 50)
    t=0     : Session B request arrives, CPU begins prefill (AMX GEMM)
    t=395ms : Session B CPU prefill chunk 1 complete
    t=396ms : KV cache for B → PCIe DMA → Chip 0 → pipeline forward
    t=397ms : KV distribution complete, CPU sets "B ready" flag
    t=397ms : Session A decode step ends, FPGA checks flag, swap
    t=398ms : Session B first decode step begins
    → Session A perceived latency increase: 0ms (prefill in background, decode unaffected)
    → Session B TTFT: ~398ms ✓

  Key assumptions (need verification):
    - FPGA can accept KV cache swap interrupts between decode steps
    - Multi-session KV cache partitioning in HBM does not interfere
    - CPU core allocation strategy does not impact AMX throughput
```

**E. CPU→FPGA Prefill Handoff Threshold — Needs Quantitative Justification 🟡**

```
The current document uses 4K tokens as the CPU→FPGA handoff boundary (§14.E):
  "Prompt > 4K tok: FPGA chunked prefill"

But based on the true TTFT analysis in §C, this threshold should be based on TTFT user experience targets:

  User experience TTFT tolerance (industry heuristic):
    < 500ms  : real-time conversation, user imperceptible
    0.5-1.0s : slight delay, acceptable
    1.0-2.0s : noticeable delay, but acceptable for RAG/Agent scenarios
    > 2.0s   : unacceptable (user will refresh/retry)

  CPU prefill (GNR) P=128/chunk: ~3.1ms/token → TTFT ≈ prompt_tokens × 3.1ms
  FPGA prefill P=512/chunk:      ~0.66ms/token → TTFT ≈ prompt_tokens × 0.66ms

  Handoff threshold analysis:

    ┌──────────────┬──────────────────┬──────────────────┬──────────┐
    │ TTFT Target   │ CPU max prompt    │ FPGA max prompt   │ Recommend│
    ├──────────────┼──────────────────┼──────────────────┼──────────┤
    │ < 500ms      │ ~160 tokens      │ ~750 tokens      │ FPGA     │
    │ < 1.0s       │ ~320 tokens      │ ~1,500 tokens    │ CPU/FPGA │
    │ < 2.0s       │ ~640 tokens      │ ~3,000 tokens    │ CPU borderline│
    │ < 5.0s       │ ~1,600 tokens    │ ~7,500 tokens    │ CPU poor │
    └──────────────┴──────────────────┴──────────────────┴──────────┘

  Recommended handoff strategy (corrected):

    ┌──────────────────────┬───────────┬──────────────┬────────────────┐
    │ Prompt Length         │ Prefill Mode│ Typical TTFT│ Scenario       │
    ├──────────────────────┼───────────┼──────────────┼────────────────┤
    │ < 500 tokens         │ CPU full   │ 0.4-1.6s     │ Chat, short Q&A│
    │ 500-2K tokens        │ CPU chunked│ 1.6-6.3s     │ Agent warm only│
    │ 2K-128K tokens       │ FPGA chunked│ 85ms first chk│ General, RAG, long│
    │ Agent warm (any)     │ CPU incr.  │ incr.×3.1ms  │ Prefix reuse ✅│
    └──────────────────────┴───────────┴──────────────┴────────────────┘

  Differences from the document:
    - §14.E Tier 1 "Prompt < 4K: chunked P=128, TTFT ~395ms" → misleading figure
      Should be "first chunk completion 395ms, full TTFT = N_chunks × 395ms"
    - The practical range of CPU prefill is <500 tokens, not <4K
    - Agent warm start is CPU prefill's true killer scenario (incremental prefill is very lightweight)
    - The handoff threshold should be lowered from 4K to ~500-2000 tokens
```

**F. Double-Buffered KV Cache Atomic Swap — Mechanism Description Insufficient ⚠️**

```
The "atomic swap" mechanism described in §14.E needs to be supplemented:

  Current HBM allocation (per session, 32 GB/chip):

    ┌──────────────────────────────────────────────────────────┐
    │ KV buf A (active):           session's current decode use │
    │ KV buf B (shadow):           CPU/FPGA prefill writes to   │
    │ Weight cache (SRAM/HBM):     resident, unaffected by swap │
    │ Expert cache (HBM):          resident, unaffected by swap │
    └──────────────────────────────────────────────────────────┘

  Swap timing (FPGA side):

    At the end of each decode step:
      1. Check "B_ready" flag (from CPU via PCIe → Chip 0 → broadcast)
      2. If B_ready && decode_step_done:
          a. Hardware swaps A↔B base address registers (single cycle, no data copy!)
          b. Clear B_ready flag
          c. Next decode step reads KV from new A
      3. Otherwise continue using current A

    Key design decisions:
      - What is swapped is the address pointer, not data → zero-copy, single cycle
      - Swap only at decode step boundaries → guarantees KV read consistency
      - Worst case: decode step takes ~1.4ms, swap must wait for current step to finish
        → extra latency ≤1.4ms, negligible

  Multi-session extension:
    Each session has an independent A/B pair, KV regions do not overlap across sessions.
    Swap is triggered independently per session. Complexity O(S) managed in hardware address generator.

  Edge cases (must be handled in RTL):
    1. CPU prefill writes to buf B while FPGA starts reading buf B (premature swap)
       → Hardware lock: check "B_write_done" flag before swap (set by CPU after write completes)
    2. Multiple CPU prefills complete simultaneously, multiple Bs ready at once
       → Hardware arbitration: prioritize swaps by session_id queue
    3. Session teardown: reclaim buf A/B
       → Hardware KV manager marks region as free, analogous to PagedAttention block reclamation

  This mechanism is conceptually correct, but the document lacks:
    - RTL implementation description of base address register swap (should be in kv_dma_bridge.sv)
    - B_ready/B_write_done flag protocol (PCIe address mapping)
    - Multi-session arbitration logic
```

**G. Comprehensive Assessment and Risk Classification**

```
┌──────────────────────────────────────┬──────────┬──────────────────────┐
│ Dimension                              │ Verdict  │ Key Prerequisites    │
├──────────────────────────────────────┼──────────┼──────────────────────┤
│ A. CPU compute (10.5 TFLOPS AMX)      │ 🟢 Viable │ Fully verified       │
│ B. KV 32-chip distribution            │ 🟢 Viable │ SERDES forwarding path│
│                                       │          │ needs completion     │
│ C. TTFT (short prompt <500 tok)       │ 🟢 Acceptable│ ~0.4-1.6s        │
│ C. TTFT (medium prompt 500-2K tok)    │ 🟡 Marginal│ 1.6-6.3s, Agent only│
│ C. TTFT (long prompt >2K tok)         │ 🔴 Unacceptable│ Must use FPGA Tier 2│
│ D. Concurrent CPU prefill + FPGA dec.  │ 🟢 Viable │ Core partition + int.│
│ E. CPU→FPGA handoff threshold         │ 🟡 Needs fix│ Lower 4K → ~500 tok│
│ F. Double-buffer KV swap              │ 🟢 Viable │ Address swap, zero-copy│
│ G. Agent warm start (incr. prefill)   │ 🟢 Best case│ CPU prefill killer app│
└──────────────────────────────────────┴──────────┴──────────────────────┘

New CPU Prefill-specific risks (to be added to §11.A.3 risk matrix):

  ┌────────────────────────────────────┬───────────┬──────────┬──────────┐
  │ Risk                                │ Probability│ Impact   │ Level    │
  ├────────────────────────────────────┼───────────┼──────────┼──────────┤
  │ CPU prefill TTFT exceeds expectation│ High (60%) │ Medium   │ 🟡 Med-High│
  │ → user churn (current doc figures   │           │          │          │
  │   optimistic, reassess after fix)   │           │          │          │
  │ CPU AMX/DDR5 unexpected contention  │ Low (15%)  │ Medium   │ 🟢 Low    │
  │ under prefill+decode concurrency    │           │          │          │
  │ Intel AMX ISA future incompatibility│ Low (10%)  │ Medium   │ 🟢 Low    │
  │ (AMX is standard x86 extension,     │           │          │          │
  │  will not disappear)                │           │          │          │
  │ KV cache distribution path RTL bug  │ Med (30%)  │ Medium   │ 🟡 Medium │
  │ (SERDES forwarding logic error)     │           │          │          │
  └────────────────────────────────────┴───────────┴──────────┴──────────┘

New experimental closure variables to add (to §11.A.4):

  P0:
    7. End-to-end TTFT for concurrent CPU prefill + FPGA decode scenario
       → Run on real GNR server + 4-8 chip FPGA prototype:
         Session A decode (steady) + Session B CPU prefill (variable length)
       → Measure: Session B TTFT, whether Session A per-step latency is affected
       → Closure criterion: Session A decode latency increase < 5%, Session B TTFT matches analytical model

  P1:
    8. KV cache SERDES pipeline forwarding actual latency vs analytical
       → On 4-8 chip system, measure time for KV data broadcast from Chip 0 to all chips
       → Closure criterion: total distribution latency ≤ analytical value × 1.5

  P2:
    9. CPU prefill TTFT measured vs analytical
       → Run full 61-layer AMX GEMM on GNR server, P=128/256/512
       → Compare with analytical model: 395ms/790ms/1580ms
       → Closure criterion: measured ≤ analytical × 1.2
```

**H. Final Verdict**

```
Is CPU Prefill + FPGA Decode viable?

  Viable ✅, but the applicable scope is narrower than the current document claims:

  Best scenarios (where CPU prefill delivers maximum value):
    1. Short conversations (< 500 tokens prompt): TTFT 0.4-1.6s, zero incremental cost
    2. Agent warm start: only prefills incremental tokens, TTFT extremely low
    3. Low-traffic private deployments: Intel SPR → GNR upgrade yields 6× prefill speedup

  Unsuitable scenarios (must use FPGA Tier 2 or GPU Tier 3):
    1. Medium-to-long prompts (> 2K tokens): CPU TTFT > 6s, user-unacceptable
    2. High-concurrency API service: multiple concurrent CPU prefills contend for AMX units
    3. Ultra-low-latency TTFT (< 100ms): even GPU needs ≥1 A100

  Key figures in the document that need correction:
    1. §4.8.6 scenario table: TTFT from "0.3-4s" corrected to "0.4-15s"
    2. §14.E Tier 1 description: "TTFT ~395ms" → "first chunk completion 395ms, full TTFT = N_chunks × 395ms"
    3. CPU→FPGA handoff threshold: 4K → ~500 tokens
    4. Explicitly note Agent warm start as the best scenario for CPU prefill (incremental mode)

  Architectural judgment:
    This is a correct direction — CPU prefill solves the "short prompts need no extra hardware" problem.
    But it is not a silver bullet; medium-to-long prompts still require FPGA's chunked prefill capability.
    The three-tier system design (CPU Tier 1 → FPGA Tier 2 → GPU Tier 3) is correct,
    only the applicability boundaries of each tier need correction.
```


**4.8.5 Prefill Scheduling Strategy — Coexistence with Decode**

```
Recommended: Chunked Prefill (Phase 2 implementation)
  512 tokens/chunk, max 1 chunk between decode steps

  Multi-session scheduling:
    round_robin:
      session_A decode_step
      session_B prefill_chunk_1
      session_C decode_step
      session_A decode_step
      session_B prefill_chunk_2
      ...

  DSP allocation:
    prefill chunk (512 tok): DSP full speed, ~85ms
    decode step:            DSP weighted utilization ~50% per §4.4.1
    → during prefill chunk, decode is paused ~85ms
    → for agent scenarios (B=1, output < 500 tok per turn),
      the 85ms prefill pause does not affect user experience

  Alternative (Phase 3+): DSP partitioned scheduling
    → 70% DSP to decode (guarantee latency)
    → 30% DSP to background prefill
    → RTL must support DSP array partition isolation, additional work
```

---

### 4.8.x Chip 0 Prefill Entry Bottleneck Analysis

§4.8 gave the basic model for chunked prefill. After all concurrency optimizations from §4.6.1 took effect, both simulation and disaggregated-mode measurements exposed a consistent bottleneck: **Chip 0's admission rate caps the system's request acceptance rate**, rather than decode compute or HBM bandwidth.

This section quantifies this bottleneck and evaluates two architecture-level optimization paths.

**4.8.x.1 Why Chip 0 Is the Prefill Serialization Point**

```
Chip 0 hosts layers 0-1 and the Embedding lookup. Every new request must sequentially complete:
  1. Host CPU tokenize
  2. PCIe DMA sends prompt tokens to Chip 0
  3. Embedding lookup (single cycle on Chip 0)
  4. Run the first chunk through layers 0-1
  5. Pipeline forward the chunk to Chip 1; Chip 0 can only then accept the next chunk

Single-chunk turnaround time on Chip 0:
  per_layer_us @ chunked prefill (P=128, fp4+sparse) = 6,740 us
  Layers hosted on Chip 0                           = 2
  per_chunk_us                                      = 13,480 us
  chunks/s                                          = 74.2

At P=512 (4 chunks per request):
  admission_rate                                    = 18.5 req/s
```

This is precisely the root cause of the ~1.7 req/s measured in §4.6.1 under high-load disaggregated (4P+2D) mode — even with 4 prefill servers deployed, each server's Chip 0 can at most admit 18.5 req/s in aggregate, and Poisson burst traffic quickly queues up.

**4.8.x.2 Two Architecture-Level Optimization Paths — Analytical Model**

Quantified via the newly added `PipelineEngine.chip0_admission_rate()` method:

```
┌────┬──────────────────────────────────────┬───────────┬──────────┬────────┬────────┐
│ #  │ Configuration                         │ per_chunk │  req/s   │  tok/s │ gain   │
├────┼──────────────────────────────────────┼───────────┼──────────┼────────┼────────┤
│ A  │ Baseline (single chip 0, embedding on-chip)│ 13.48 ms │   18.5  │   9495 │   1.0× │
│ B  │ Embedding offload to host CPU         │  13.43 ms │   18.6  │   9531 │   1.0× │
│ C  │ Pipeline Cloning ×2 (16+16 chips)     │  13.48 ms │   37.1  │  18991 │   2.0× │
│ D  │ Pipeline Cloning ×2 + Embedding offload│ 13.43 ms │   37.2  │  19061 │   2.0× │
│ E  │ Pipeline Cloning ×4 (8+8+8+8 chips)   │  13.48 ms │   74.2  │  37981 │   4.0× │
│ F  │ Pipeline Cloning ×4 + Embedding offload│ 13.43 ms │   74.5  │  38123 │   4.0× │
└────┴──────────────────────────────────────┴───────────┴──────────┴────────┴────────┘
```

**4.8.x.3 End-to-End Simulation Validation (Agent 8 req/s, O=1024)**

Integrating Pipeline Cloning into `ServingSimulation` (`--pipeline-clone N`), comparing different clone counts:

```
                          clone=1    clone=2    clone=4
                          ────────   ────────   ────────
  Accept rate              52.7%      50.1%      54.0%
  Output TPS (tok/s)        8,526      7,752      8,389
  TTFT P50 (ms)               527        402        404    ← key improvement
  TTFT P95 (ms)             1,150        543        418    ← ×2.7 improvement
  Avg active session           23         35         36
  Avg batch size              7.1        4.8        2.9

Under high load (Agent 20 req/s, O=1024):
                          clone=1    clone=2    clone=4
                          ────────   ────────   ────────
  Accept rate              25.4%      18.4%      19.1%
  Output TPS (tok/s)        8,515      5,930      6,066
  TTFT P50 (ms)               550        435        390
  TTFT P95 (ms)             2,108        615        429    ← ×4.9 improvement
  Avg active session           17         43         66
```

**4.8.x.4 Measured vs Analytical Discrepancy — Interpretation**

```
Analytical model: Pipeline Cloning ×2 should double admission → accept rate should rise
Measured:         Pipeline Cloning ×2 instead slightly decreases Accept rate (52.7% → 50.1%)

Reason:           Pipeline Cloning splits 32 chips into two pipelines,
                  each pipeline's decode peak compute is halved (same total DSPs but divided across two pipelines).
                  While prefill admission rate doubles, each pipeline's decode processing capacity is also halved.

                  On Output TPS:
                    clone=1: one pipeline running 8526 tok/s (roughly 49% of 17,445 saturation)
                    clone=2: two pipelines each ~3876 tok/s (aggregate 7752, per-pipeline 44% saturation)
                    clone=4: four pipelines each ~2097 tok/s (aggregate 8389)

                  So the true value of Pipeline Cloning is:
                    ✓ Reduces TTFT P95 from 2.1s to 0.4s (×5 improvement)
                    ✓ Boosts serviceable concurrent sessions (17 → 66, ×4)
                    ✗ Does not significantly improve aggregate throughput (capped by decode peak)
```

**4.8.x.5 Practical Effect of Each Optimization**

```
Embedding Offload (B vs A):
  Embedding is an SRAM lookup, ~50 us per chunk.
  Saving 50 us out of 13,480 us = 0.4%.
  Conclusion: not the real bottleneck. Chip 0's bottleneck is the 2-layer MLA+MoE compute load,
               not the embedding/tokenize step.
               Embedding offload is not worth the extra PCIe round-trip + host coordination complexity.

Pipeline Cloning ×2 (C vs A):
  Splits 32 chips into two independent pipelines (16 chips each, 4 layers/chip),
  prefill admission rate doubles, while TTFT improves dramatically (×3 P95 improvement).

  Cost analysis:
    HBM per chip: 0.7 GB weights + 22 GB KV = 22.7 GB
                  → becomes ~1.2 GB weights + 21 GB KV (each pipeline holds full 384 experts,
                    per-chip weights double)
                  Still within 32 GB HBM physical limit — OK.
    Decode latency cost: per-token latency rises ~10-20%, because each chip hosts 4 layers instead of 2
                   (per-chip stage time increases), but pipeline depth halves (32→16 chips)
                   partially offsetting this.
    No hardware changes: purely a deployment/scheduling decision, no RTL changes needed.

Pipeline Cloning ×4 (E vs A):
  prefill admission rate ×4, at the cost of ~30% higher single-token decode latency.
  Each chip hosts 8 layers, compute density increases.
  Recommended for: scenarios with relaxed TTFT budgets and high peak traffic (API-type workloads).
```

**4.8.x.6 Stacking Effects with §4.6.1 Optimizations**

```
Single-server end-to-end throughput evolution path (Agent workloads):

  Stage 1: baseline configuration
    accept_rate    28%
    output_tps    1,000 tok/s
    bottleneck     chip 0 prefill (18 req/s) + decode scheduling floor

  Stage 2: §4.6.1 KV expansion + remove scheduling floor + Hot Expert Replication
    accept_rate    88%
    output_tps    5,800 tok/s
    bottleneck     chip 0 prefill (18 req/s)

  Stage 3: + Pipeline Cloning ×2 (§4.8.x)
    accept_rate    ~50% (under high arrival)
    output_tps    ~7,800 tok/s
    TTFT P95       from 1.15s to 0.54s (×2.1 improvement)
    bottleneck     decode peak DSP

  Stage 4: + Pipeline Cloning ×4
    accept_rate    similar
    output_tps     caps at ~8,400 tok/s
    TTFT P95       from 2.1s to 0.43s (×4.9 improvement, high-load scenario)
    bottleneck     hardware physical limit
```

**4.8.x.7 Deployment Recommendations**

For FPGA clusters as inference-serving platforms in production:

1. **Always enable the §4.6.1 optimization bundle** (KV expansion + remove scheduling floor + Hot Expert Replication). Zero hardware cost, ×6 throughput improvement.

2. **Default to Pipeline Cloning ×2**: recommended for any deployment serving 5+ concurrent agent sessions. **The primary benefit is a dramatic TTFT reduction (P95 improvement ×3-5)**; accept rate may dip slightly but user experience improves significantly. Pure deployment-phase decision (no RTL changes).

3. **Pipeline Cloning ×4 for high-traffic API scenarios**: enable when TTFT budgets are relaxed and concurrent sessions exceed 50. Cost is a 30% increase in single-token latency.

4. **Skip Embedding offload** — not the real bottleneck; complexity-to-benefit ratio is unfavorable.

Implementation cost: Pipeline Cloning requires the weight layout compiler (§5.3) to support outputting per-pipeline-split weight mappings. This is a low-risk software task, estimated at ~1 person-month.

---

### 4.9 Agent Scenario Adaptation Analysis

Agent workloads differ fundamentally from simple chatbots. These differences happen to amplify several of FPGA's architectural advantages while exposing the prefill shortfall's impact in specific sub-scenarios.

**4.9.1 Agent Workload Characteristics**

```
Typical Agent loop:

  System prompt (5-20K) + Tool definitions (2-5K)  ← first prefill
      ↓
  Turn 1: Full context → LLM → Tool call → Tool result (1-5K new tokens)
  Turn 2: Full history   → LLM → Tool call → Tool result
  ...
  Turn N: Full history   → LLM → Final answer

Key differences vs simple chatbot:

  ① Context grows monotonically with turns (can reach 32K-128K)
  ② Per-turn output is very short (<500 tokens, typically tool calls / brief reasoning)
  ③ Prefix is invariant → vast majority of KV Cache reusable (turns 2-N)
  ④ Naturally B=1 (agent reasons serially, cannot parallelize multiple plans)
  ⑤ Multi-turn interaction → long total session time (minutes)
```

**4.9.2 Three Structural Advantages of FPGA in Agent Scenarios**

**Advantage 1: KV Cache Prefix Reuse — Incremental Prefill Extremely Lightweight**

```
Turn 1 (cold start):
  → Full prefill of system prompt + user query (~10-30K tokens)
  → TTFT: 1.7-5.0s (full) / 125ms (first chunk)

Turns 2-N (warm, accounting for >90% of agent inference):
  → Only prefill new tokens (previous turn LLM output 200 + Tool output 2-5K)
  → Effective prefill ≈ 2-5K tokens → TTFT < 1s
  → Remaining prefix KV Cache directly reused by hardware, zero compute

  GPU (vLLM prefix caching):
    Also supports reuse, but requires software block table lookup/verification/copy.

  FPGA:
    Hardware KV address generator: {session_id, layer_id, seq_pos} → HBM physical address
    Prefix match = address offset, zero latency, zero CPU involvement.
    Multi-turn agent incremental prefill path is shorter than GPU's.
```

**Advantage 2: Agent Decode Naturally B=1, FPGA's Optimal Point**

```
Agent per-inference step:
  → Generates "whether to call tool + tool name + parameters" or "brief reasoning"
  → Typically <500 tokens
  → Naturally requires no batching

GPU at B=1: Tensor Core ~3% utilization, nearly fully idle
FPGA at B=1: DSP ~50% weighted utilization, every dollar is working

Agent scenario decode volume may not be large (short outputs, deduplicated),
but request frequency is high (multi-turn interaction), so the low per-token cost has a significant cumulative effect.
```

**Advantage 3: Long-Session KV Cache Hardware Management — GPU's Software Bottleneck**

```
Long agent session with 128K context:

  GPU (vLLM PagedAttention):
    → 128K / 16 blocks = 8K KV blocks
    → Each decode requires table-walking 8K blocks
    → Software block table management grows linearly with session count
    → CPU allocation/deallocation of blocks is a non-negligible load

  FPGA:
    → Hardware address generator, combinational logic
    → Per-token KV address resolution < 10ns
    → Sliding window (128 positions) automatic hardware eviction
    → Zero CPU, zero software, zero incremental overhead as context grows

  Under long sessions, FPGA's KV management advantage shifts from "negligible" to "measurable."
```

**4.9.3 Two Disadvantages of FPGA in Agent Scenarios**

**Disadvantage 1: First Prefill Cold-Start Latency**

```
First agent request (system prompt + tools + history):
  Short (5K):  TTFT ~1s, acceptable
  Medium (20K): TTFT ~3.3s (full) / 125ms (first chunk)
  Long (128K): TTFT ~21s (full) / 125ms (first chunk)

  Chunked prefill mitigates:
    → First token still visible at 125ms
    → Agent can start reasoning/tool calling upon receiving first token
    → No need to wait for full prefill completion

  Agents typically enter high-frequency interaction from turn 2 onward,
  so first-prefill cold-start latency has limited impact on overall experience.
```

**Disadvantage 2: Massively Concurrent Agents Are Limited**

```
1 FPGA = 1-2 concurrent sessions × 30 FPGA = 30-60 concurrent agents

  ┌────────────────────┬──────────────┬──────────────┐
  │ Scenario             │ FPGA (30 cards)│ H200 (8 cards)│
  ├────────────────────┼──────────────┼──────────────┤
  │ Enterprise 10 agents │ More than enough│ More than enough│
  │ Dept. 50 agents      │ Just enough   │ More than enough│
  │ SaaS 1000 agents     │ ✗ insufficient│ Feasible      │
  │ Per-agent isolation  │ ✓ natural     │ △ needs MIG   │
  └────────────────────┴──────────────┴──────────────┘

  FPGA's 30-60 concurrency is sufficient for enterprise private deployment agent scenarios.
  Scaling to multiple clusters uses physical replication rather than batching — natural tenant isolation.
  SaaS platforms with high-concurrency multi-tenancy are GPU/Groq territory.
```

**4.9.4 Agent Scenario Decision Matrix**

```
┌──────────────────────────────┬──────┬──────┬──────────────────────┐
│ Agent Dimension               │ GPU  │ FPGA │ Notes                 │
├──────────────────────────────┼──────┼──────┼──────────────────────┤
│ First long-context prefill   │ ★★★★ │ ★★   │ Chunked first token OK│
│ Incremental prefill (prefix) │ ★★★  │ ★★★★ │ FPGA KV Cache zero SW │
│ B=1 decode (agent serial)   │ ★    │ ★★★★ │ FPGA arch decisive win │
│ Long-session KV mgmt         │ ★★   │ ★★★★ │ HW > SW, long context  │
│ Multi-agent concurrency (>100)│ ★★★★ │ ★    │ GPU absolute advantage │
│ Low agent concurrency (<50)  │ ★★★  │ ★★★★ │ FPGA optimal zone      │
│ Per-turn latency (<500 tok)  │ ★★★  │ ★★★  │ Comparable             │
│ Data isolation (fin/med)     │ ★★   │ ★★★★ │ FPGA physical isolation│
│ Multimodal agent             │ ★★★★ │ ★    │ Needs external NPU     │
└──────────────────────────────┴──────┴──────┴──────────────────────┘

Conclusion:
  Text-only Agent × Enterprise private deploy × Long session × <50 concurrent → FPGA advantage
  Multimodal Agent × SaaS multi-tenant × Short session × >100 concurrent → GPU advantage
```

**4.9.5 Business Narrative for Agent Scenarios**

```
Chinese enterprise agent deployments face a triple constraint:

  ① Data must not leave premises: financial transaction records, medical records, government documents
     → Public APIs (DeepSeek/GPT) = unusable
     → Must be privately deployed hardware

  ② High-end GPUs are unobtainable: H100/B200 embargoed, Ascend backordered
     → Private GPU clusters = cannot purchase
     → FPGA = obtainable

  ③ Long-session multi-turn interaction: agents are not single Q&A
     → KV Cache management becomes a bottleneck beyond compute
     → FPGA hardware KV management = no CPU, no software

  Agents are the ultimate form of enterprise AI. FPGA's structural advantages in agent scenarios
  (B=1 decode, KV prefix reuse, hardware cache management)
  are more pronounced than in chatbot scenarios — not merely a "fallback when GPUs are unavailable,"
  but rather "for this specific agent workload, FPGA is architecturally superior."
```


### 4.9.6 Coding Agent: FPGA's Killer Scenario

**Why is a coding agent fundamentally different from a general agent?**

A general agent's tool calls are sparse — occasionally search, occasionally query a database. A coding agent's
tool calls are extremely dense — generate code → compile → read errors → fix → recompile → read test results,
potentially 3-5 prefill/decode alternations per turn.

```
General agent per-turn pattern:
  decode (decide what to do, ~100 tok) → tool execution (seconds of waiting) → prefill (result, 1-5K tok)
  Long intervals, one decode, one prefill, user latency-insensitive

Coding agent per-turn pattern:
  decode (generate function, ~200 tok) → execute (LSP/compile/test, ms-seconds)
  → prefill (error message, ~500B-2K tok)
  → decode (fix code, ~100 tok)
  → execute → prefill (test pass/fail, ~500B)
  → decode (continue writing next function)
  ...
  3-5 prefill/decode alternations per turn, short intervals between each
```

This difference amplifies three of FPGA's structural advantages:

**Advantage 1: High-frequency prefill/decode switching — GPU scheduling latency is repeatedly penalized**

```
Per switch:
  FPGA: DSP register reconfiguration → < 1μs (combinational logic writes a config word)
  GPU:  CPU scheduler → CUDA kernel launch → SM context switch → millisecond-scale

Coding agent 3-5 switches per turn:
  FPGA cumulative switch overhead:  5 × 1μs = 5μs       (negligible)
  GPU cumulative switch overhead:   5 × 1-5ms = 5-25ms   (accumulates to user-perceptible)

High-frequency tool calls (MCP, LSP, compiler) will become mainstream for agents in 2026-2027.
FPGA's zero-switch-overhead is not a "nice to have" — it is a fundamental requirement for coding agents.
```

**Advantage 2: Code context KV cache is extremely stable — the watershed between hardware and software management**

```
Coding agent context composition:
  ├── System prompt (role + rules):        5-10K,  invariant across session
  ├── Project context (file tree, types):  20-50K,  partial update on file switch
  ├── Conversation history:                growing, appended on each prefill
  └── Tool outputs (compile errors, LSP):  <2K,     old results discarded each time

Prefix stability: ~80-90% of KV cache unchanged across the entire session.

GPU (vLLM PagedAttention):
  → block table must still traverse all KV blocks (including invariant prefix)
  → 30K prefix = ~2000 blocks, per-step table walk ~50μs
  → coding agent 3-5 decodes per turn → cumulative table walk 150-250μs

FPGA:
  → Hardware KV address = base + layer * stride + seq_pos * kv_bytes
  → Prefix match = address offset, zero software, zero traversal
  → Per-token KV address resolution < 10ns, independent of context length
```

**Advantage 3: IDE latency budget is extremely tight — deterministic latency > average latency**

```
User latency expectations in IDE scenarios:
  < 200ms:  "instant" — code completion level
  < 500ms:  "smooth" — agent single-step response
  < 2s:     "waiting" — agent multi-step reasoning complete
  > 2s:     "stutter" — user begins suspecting a bug

GPU non-determinism sources:
  → KV block fragmentation triggering GC pause:    10-50ms, random
  → CUDA kernel scheduling queuing:                1-5ms,  varies with GPU load
  → vLLM continuous batching reorganization:        5-20ms, more frequent with more requests

FPGA determinism sources:
  → Hardware KV address generation:    < 10ns (combinational logic, no queuing)
  → Streaming pipeline:                 1.4ms/token (deterministic, no stalls)
  → No GC, no block table, no kernel launch
```

**What shortfalls does the coding agent help FPGA avoid?**

```
FPGA's three main shortfalls are naturally mitigated in coding agent scenarios:

  1. Multi-agent concurrency ceiling:
     AI IDE scenario: one developer = at most 1-2 parallel agent sessions
     (one writing backend, one writing frontend, extreme case)
     → FPGA's 30-60 concurrency is never challenged

  2. Multimodal requirements:
     Coding agents are text-only interaction (code + compile errors + LSP + git diff)
     → No ViT/CLIP/vision encoder needed

  3. Cold-start prefill latency:
     IDE can warm up in background on project open:
     → On project open, FPGA background-prefills system prompt + project context KV cache
     → By the time user starts writing first prompt, KV cache is already ready
     → User-perceived TTFT ≈ incremental prefill time (< 1s)
     → Cold start effectively resolved by "warm start" strategy
```

**4.9.6.1 Coding Agent Business Model: "AI IDE Box"**

```
Positioning: not selling FPGA cards, but selling "coding agent-dedicated inference nodes."

Primary hardware configuration (HBM-Only, 32 chips maxing out decode bandwidth):

  ┌─────────────────────────────────────────────────────────┐
  │ FPGA Coding Agent Node                                  │
  │                                                         │
  │  FPGA: 32 Agilex 7 M chips, 8 cards × 4 chips/card     │
  │        HBM-Only configuration (32 GB HBM2e per chip)    │
  │        All weights in HBM, pipeline-parallel 32-chip    │
  │        distribution                                     │
  │        Bandwidth/layer: 920 GB/s ÷ 2 layers/chip        │
  │                        = 460 GB/s/layer                 │
  │        Chip BOM: 32 chips × ¥18K = ¥576K                 │
  │        Card-level BOM + server BOM: ~¥1.33M              │
  │                                                         │
  │  Server: Dual Xeon GNR 6980P (included in BOM)         │
  │          256 GB DDR5, CPU prefill capable (Tier 1)      │
  │                                                         │
  │  Aggregate throughput: 5,800-8,500 tok/s (B≥4,          │
  │                        post-§4.6.1 optimizations)       │
  │  B=1 throughput: ~720 tok/s (single-session decode)    │
  │                                                         │
  │  Service capacity:                                      │
  │    Concurrency: 30-60 coding agent sessions (with        │
  │                Pipeline Cloning ×2 can extend to 50+)   │
  │    Latency: per-token 1.4ms (deterministic, streaming)  │
  │    Context: supports 128K context, KV cache HW-managed  │
  │    Security: code never leaves enterprise network,       │
  │              physical isolation                         │
  │                                                         │
  │  vs 950PR 8-card (¥2M):                                  │
  │    BOM:       ¥1.33M vs ¥2M                              │
  │    Eff. BW:   29.4 TB/s vs ~11 TB/s → 2.7× at B=1       │
  │    Throughput: 5,800-8,500 vs 2,500-4,000 tok/s →       │
  │               2.1-2.3×                                   │
  │    B=1:       ~720 vs ~200-300 tok/s                     │
  │    → Competing on architectural bandwidth efficiency,    │
  │      not on "cheaper"                                   │
  └─────────────────────────────────────────────────────────┘

  Downgrade option (HBM+DDR economy config, §2.7):
    For small teams (5-10 people), optional 5-8 HBM+DDR chips:
      → Chip BOM ¥175K, DDR stores weights, HBM runs KV cache
      → Throughput 800-1,500 tok/s, serving 10-20 agent sessions
      → DDR cost reduction is FPGA-architecture-unique flexibility,
        no such path for GPU/NPU
    DDR is a cost-reduction path, not affecting the main architecture's bandwidth argument.

Target customers and decision chain:
  ┌──────────────────────┬──────────────────┬──────────────────────┐
  │ Customer Type         │ Pain Point        │ Decision Maker       │
  ├──────────────────────┼──────────────────┼──────────────────────┤
  │ Fintech               │ Code cannot go    │ CTO + Security       │
  │                       │ to public cloud   │                      │
  │ Defense/Gov IT        │ Air-gapped network│ IT Procurement +     │
  │                       │ domestic hw req.  │ Security Approval    │
  │ Internet (mid-size)   │ GPU queue/embargo │ Engineering VP       │
  │ Outsourcing/SW vendor │ Client requires   │ Project Delivery Lead│
  │                       │ data locality     │                      │
  │ University/Research   │ Budget-limited,   │ Lab Director         │
  │                       │ need private      │                      │
  └──────────────────────┴──────────────────┴──────────────────────┘

China market benchmarks:
  2025-2026 domestic coding agents have entered rapid growth:
    - Tongyi Lingma (Alibaba): enterprise private deployment option
    - CodeBuddy (Tencent): large-scale internal use
    - SenseTime Raccoon: code generation + review
    - Various DeepSeek V3/V4-based private coding agents

  All of these face the same backend problem:
    → Using public API: code security unacceptable
    → Using H100/H200: cannot purchase
    → Using Ascend: backordered, no native fp4 support, poor price-performance
    → FPGA coding agent node = the only solution simultaneously satisfying
      "obtainable + fp4 native + private deployment"
```

**4.9.6.2 Addressing Anticipated Challenges**

```
Challenge 1: "Cursor/Copilot all use cloud GPUs; users don't mind uploading code"

  Response:
    a) Cursor/Copilot's individual users and paying enterprise users are two distinct groups.
       Enterprise users (especially finance/defense/outsourcing) have explicit compliance requirements;
       "code must not leave the enterprise network" is a hard constraint — this is not about
       whether users mind, but whether compliance can be passed.

    b) GitHub Copilot launched "Copilot Enterprise with data residency" in 2025 precisely
       because enterprise customers demanded data localization. This proves that the
       "code not uploaded" market demand is real.

    c) China market specifics: enterprises using Cursor/Copilot inherently face data export risks.
       Simultaneously, domestic GPU supply is extremely constrained. The FPGA coding agent node
       simultaneously solves two inescapable problems in China: data security and hardware availability.

Challenge 2: "Coding agents require high TTFT; CPU prefill is too slow"

  Response:
    a) Warm start is the primary strategy: when IDE opens a project, background-warm the
       system prompt + project context KV cache. By the time the user starts interacting,
       the prefix is ready, only incremental prefill needed (< 1s TTFT).

    b) Even on cold start, coding agent first responses have a "progress bar" mental model —
       users are accustomed to waits like "indexing project..." — not instant like chatbots.

    c) Tier 2 FPGA chunked prefill (P=512, ~85ms/chunk) gives ~3.4s first chunk or
       ~3.4s full TTFT for a 20K system prompt — acceptable against "project open" expectations.

Challenge 3: "Model coding capability matters more than inference hardware; DeepSeek V4's coding
           ability lags behind Claude/GPT"

  Response:
    a) The model gap is narrowing: DeepSeek V4's coding benchmarks already approach GPT-4o levels.
       DeepSeek V5 is expected late 2025-2026, with coding capability likely further closing the gap.

    b) FPGA's architecture is not tied to a specific model: any fp4 MoE + MLA architecture model
       can be deployed. DeepSeek V5/V6, Qwen 3 MoE, or any future open-source coding model works.

    c) "Good enough" threshold: coding agents don't need models that solve IMO-level math.
       What's needed: understand project context → generate reasonable code → understand compile errors → fix.
       This task has far lower model capability requirements than "winning a programming contest."

Challenge 4: "Per-token 1.4ms, generating a 200-token function = 280ms, too slow"

  Response:
    a) Coding agent interaction mode is streaming: users start reading upon seeing the first line,
       no need to wait for the entire function to generate. 1.4ms means ~714 tok/s —
       far faster than human reading speed.

    b) Actual latency perception: 200 tokens × 1.4ms = 280ms, plus incremental prefill ~500ms,
       total < 800ms — within the IDE "smooth" perception range.

    c) For comparison: human thinking + typing time is typically seconds to tens of seconds.
       The agent bottleneck is reasoning quality (is the generated code correct), not per-token latency.
```

**4.9.6.3 Endgame Assessment for Coding Agent**

```
Among all agent categories, coding agent is the most favorable for FPGA architecture:

  ┌──────────────────────────┬─────────┬───────────┬─────────────┐
  │                          │ Chatbot │ Gen. Agent │ Coding Agent │
  ├──────────────────────────┼─────────┼───────────┼─────────────┤
  │ Prefill/decode alt. freq.│ 1:1     │ 1:1~1:2   │ 1:3~1:5 🔥  │
  │ KV cache prefix stability│ Low     │ Medium     │ Extreme 🔥   │
  │ Concurrent sessions/user │ 1       │ 1-2       │ 1-2 🔥       │
  │ Latency sensitivity      │ Medium  │ Low-Med    │ High 🔥      │
  │ Multimodal requirement   │ Low     │ Med-High   │ Low 🔥       │
  │ Data privacy requirement │ Medium  │ High       │ Extreme 🔥   │
  │ Model capability bar     │ Medium  │ High       │ Med-High     │
  │ ─────────────────────── │ ─────── │ ───────── │ ─────────── │
  │ FPGA fitness             │ ★★★     │ ★★★★      │ ★★★★★       │
  └──────────────────────────┴─────────┴───────────┴─────────────┘

  Endgame assessment:
    "Private deployment + Coding Agent" is not the fallback after FPGA fails to find a GPU market —
    it is the implied design target scenario of the FPGA inference architecture from the start.

    In this scenario, FPGA is not a "good-enough and cheaper alternative,"
    but an architecturally superior solution on key dimensions (switching frequency, KV determinism,
    hardware isolation).

    If it can simultaneously capture "finance/defense code security compliance" +
    "internet company GPU supply shortage incremental demand,"
    the TAM estimate for coding agent inference hardware in the China market is:
      - 100K developers × 30% AI agent penetration = 30K concurrent agents
      - Each HBM-Only 32-chip (8-card) set serves 30-60 sessions → 1,000 sets
      - 1,000 sets × ¥1.33M/set (chips + cards + server) = ¥1.33B (coding agent alone)
      - Small teams can choose HBM+DDR downgrade (5 chips ¥90K), expanding addressable market
      - Extending to general agents + customer service + RAG → layered market expansion
```
