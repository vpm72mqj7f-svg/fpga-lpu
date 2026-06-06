## 11. Competitive Analysis

### 11.1 Benchmarking Matrix (Two Phases)

```
Phase 1 — FPGA Prototype Validation Period (Now-18 months):

┌──────────────┬──────────┬──────────┬──────────┬──────────────┐
│              │NVIDIA H100│Ascend 950PR│Domestic GPU│Our FPGA      │
│              │/H200/B200│          │(Camb/Hy/  │8-card×4-chip │
│              │          │          │ Biren)    │AGM039        │
├──────────────┼──────────┼──────────┼──────────┼──────────────┤
│ Availability │ ✗ Sanctions│ △ 6-18mo queue│ △ Uncertain│ ✓ 8-12wk lead │
│ Global Deploy│ △ Partial│ ✗ Near-zero│ ✗ Near-zero│ ✓ Std equipment│
│ HW Price/set │ ~$280K   │ ~$110K   │ ~$100-150K│ ~$303K       │
│ $/M token    │ $12-20   │ $16-25   │ $15-30   │ $5.9         │
│ fp4 Native   │ ✓ B200   │ ✗ None   │ ✗ None   │ ✓ Custom     │
│ MLA HW Accel │ ✗ Software│ ✗ CANN sched│ ✗ Software│ ✓ Hardened   │
│ SW Ecosystem │ ★★★★★   │ ★★★★    │ ★★~★★★  │ ★★          │
│ Deploy Flex  │ ★★       │ ★★       │ ★★       │ ★★★★★       │
│ Positioning  │ Embargo BM│ Best domestic│ Fallback│ Arch Valid Plat│
└──────────────┴──────────┴──────────┴──────────┴──────────────┘

Phase 2 — ASIC Tape-out Mass Production Period (18-36 months):

┌──────────────┬──────────┬──────────┬──────────┬──────────────┐
│              │NVIDIA H100│Ascend 950PR│Domestic GPU│Our ASIC      │
│              │/H200/B200│          │(Camb/Hy/  │12nm custom   │
│              │          │          │ Biren)    │chip          │
├──────────────┼──────────┼──────────┼──────────┼──────────────┤
│ Availability │ ✗ Sanctions│ △ Queue  │ △ Uncertain│ ✓ Self-controlled│
│ Global Deploy│ △ Partial│ ✗ Near-zero│ ✗ Near-zero│ ✓ Own chip    │
│ HW Price/set │ ~$280K   │ ~$110K   │ ~$100K   │ **~$70-80K**  │
│ $/M token    │ $12-20   │ $16-25   │ $15-30   │ **$2.5-3.5**  │
│ fp4 Native   │ ✓ B200+  │ ✗        │ ✗        │ ✓ Hardened    │
│ MLA HW Accel │ ✗ Software│ ✗        │ ✗        │ ✓ Hardened    │
│ Supply Stab  │ ✗ Cut off │ △ SMIC   │ △        │ ✓ TSMC/SMIC  │
│ Positioning  │ Embargo  │Domestic lim│ Fallback│ **Arch Dominance**│
└──────────────┴──────────┴──────────┴──────────┴──────────────┘

Key differences:
  Phase 1 (FPGA): Availability and global deployment — the only two perfect scores; architectural bandwidth efficiency already validated (effective bandwidth ~83× @ B=1)
  Phase 2 (ASIC): Architectural advantage physically hardened + manufacturing cost collapse → the only solution simultaneously delivering two orders-of-magnitude advantages
```

### 11.2 Uniqueness Argument

```
Dimension 1: Supply Autonomy

  NVIDIA:   Subject to US export controls, zero allocation of high-end models to China
  Ascend:   Constrained by SMIC 7nm capacity + CoWoS packaging sanctions
            Huawei wafer allocation limited, priority to Huawei Cloud and major clients
  Domestic GPU: Biren/Moore Threads equally affected by Entity List
            Cambricon/Hygon supply volumes limited
  FPGA:     Intel global fab network (US/Ireland/Israel)
            HBM from Korea (SK Hynix/Samsung)
            Packaging in Southeast Asia (Malaysia/Vietnam)
            → Not dependent on any single jurisdiction
            → Not subject to GPU compute sanctions (TPP far below threshold)

Dimension 2: Deployment Autonomy

  Target: Chinese LLM going global → deployment in SEA/ME/LATAM/Africa

  If the foundation is Ascend:
    → Huawei export license + Huawei local support system
    → Huawei's relationships with certain countries/regions may carry policy risk

  If the foundation is Intel FPGA:
    → Globally standard IT equipment, no special export license required
    → Local Dell/Supermicro/HP distributors can procure servers
    → FPGA cards enter as standard PCIe devices
    → Not subject to GPU export control restrictions

Dimension 3: Technology Moat

  fp4 + MLA hardware acceleration = a dimension absent from all other solutions
  - NVIDIA B200/GB200 already supports FP4 Tensor Core, but subject to export controls + astronomical pricing
  - Ascend has no fp4 — Huawei has not announced fp4 support plans for next generation
  - Cambricon/Hygon/Biren all lack fp4
  - Among hardware obtainable in China, only a custom FPGA can perform native fp4 inference
```

### 11.3 Ascend 910C In-Depth Comparative Analysis

Review feedback noted: the real choice for Chinese customers is not FPGA vs H100, but FPGA vs Ascend 910C. Huawei Ascend is the default domestic alternative, with a complete CANN software stack and strong government backing. This section provides a comprehensive six-dimensional comparison.

**11.3.1 Hardware Architecture: fp4 and MLA Are Ascend's Collective Blind Spots**

```
Da Vinci Core (Ascend 910B/C) supported precisions:
  ✓ INT8, INT4 (quantized inference only)
  ✓ FP16, BF16
  △ FP8 (910C reportedly supported, not publicly confirmed)
  ✗ fp4 (E2M1) — no silicon-level support, no known roadmap

DeepSeek V4 Pro's fp4 weights on Ascend:

  fp4 weights (HBM)
    → load → decompress to FP8 (additional Vector Unit overhead)
    → feed to Cube Unit FP8 MAC
    → 3 steps, decompression consumes ~10-15% extra latency and ALU resources
    → faces exactly the same structural problem as GPUs

  MLA Kernel Launch overhead:
    CANN task scheduling latency ~10-30μs (heavier than CUDA ~5μs)
    6 attention kernels × 30μs = 180μs launch overhead / layer
    61 layers: 11ms pure scheduling latency (vs FPGA zero)

┌──────────────────────┬────────────┬────────────┬──────────────┐
│                       │ Ascend 910C│ H100 (sanctions)│ FPGA (our approach)│
├──────────────────────┼────────────┼────────────┼──────────────┤
│ fp4 Native Support    │ ✗          │ △ B200+    │ ✓ Native fp4 │
│ fp4 Inference Path    │ Decomp→FP8 │ B200+ native│ LUT→DSP fp4 │
│ MLA Hardware Accel    │ ✗ (CANN)  │ ✗ (CUDA)   │ ✓ 6-stage pipe│
│ KV Cache HW Mgmt      │ ✗ Software │ ✗ Software │ ✓ Hardware   │
│ Decode B=1 Compute Util│ ~5-8%     │ ~2-3%      │ ~50%         │
└──────────────────────┴────────────┴────────────┴──────────────┘
```

**11.3.2 Supply Availability: Is Ascend Really Not Supply-Constrained?**

```
Ascend 910C manufacturing constraints:
  SMIC 7nm (N+2):      Capacity contested by Huawei phone SoCs, 5G base stations, and Ascend
  CoWoS-class advanced packaging: JCET/TFME capacity limited,
                       HBM-to-die interconnect yield still ramping
  2024-2025 actual shipments: Estimated ~500K-800K units/year (incl. 910B+910C)
                       vs market demand >2M units

Huawei internal allocation priority:
  Tier 0: Huawei Cloud internal (Pangu LLM + Ascend Cloud services)
  Tier 1: National projects (defense, meteorology, research supercomputing)
  Tier 2: Strategic partners (Baidu, iFlytek, telecom carriers)
  Tier 3: Large enterprise clients (finance, energy) → queue 6-18 months
  Tier 4: SMEs → essentially unobtainable

Signed contracts with payment made, waiting 12 months for delivery is the norm per customer feedback.

Contrast with FPGA:
  Agilex 7 M: Intel 10nm SuperFin, mature process, no capacity shortage
  Directly purchasable on the open market in 2024, advance order lead time 8-12 weeks
  32-unit order volume is "routine customer" tier for Intel distributors
  Supply chain depends on no sanctioned entities (chips from Intel global fab,
  HBM from Korea, packaging in Southeast Asia)

The key difference is not "FPGA is faster than Ascend,"
but "FPGA lead time 12 weeks, Ascend queue 12 months" —
predictability itself is a competitive barrier.
```

**11.3.3 Overseas Deployment: Ascend Cannot Go Global — A Structural Fatal Flaw**

```
Chinese LLM overseas deployment:

  Ascend 910C:
    Huawei on US Entity List → cannot transact with any semiconductor containing US technology
    Also subject to China's technology export restrictions → advanced AI chips restricted from export
    Double lockdown → overseas deployment nearly impossible
    (Limited exceptions: some SEA/Africa via special channels, extremely low volume)

  FPGA (our approach):
    Intel chip, standard PCIe device, globally universal
    Not subject to GPU compute sanctions (TPP far below threshold)
    Not affected by Entity List (Intel is a multinational)
    Deployable in any country

  This difference is structural and does not change as Ascend capacity improves.
  If your customers are Chinese companies going global (TikTok, Temu, Shein-level
  overseas AI inference demand), Ascend is simply unavailable; FPGA is the only option.
```

**11.3.4 Software Ecosystem: CANN Is More Mature Than Us — But That Doesn't Make It the Right Tool**

```
CANN (Ascend):
  ✓ 5+ years of development, relatively feature-complete
  ✓ PyTorch adaptation (torch_npu), supports mainstream models
  ✓ MindSpore native integration
  ✗ Closed-source, Huawei-controlled
  ✗ DeepSeek V4 Pro's unique fp4+MLA requires custom operators
  ✗ Custom operator development has a high barrier (TBE/TIK DSL, incomplete documentation)
  ✗ Bugs require reliance on Huawei FAE support (queued)
  ✗ CANN version upgrades may require deployed model re-adaptation

Our toolchain (§5.3):
  ✗ Built in-house, maturity ★★★
  ✓ Minimal — WLC only needs to generate weight layout for a single fixed hardware datapath
  ✓ Full-stack self-controlled — no dependency on third-party SDK version iterations
  ✓ DeepSeek V4 Pro-specific optimizations hardened at the RTL level
  ✓ Configure once, run stably, no need to chase versions

Key difference:
  CANN is a general-purpose framework → problems wait for Huawei scheduling → uncontrollable
  WLC is a purpose-built tool → problems fixed in-house → controllable

  For the specific model DeepSeek V4 Pro,
  the maintenance complexity of a specialized solution is actually lower than adapting a general framework.
  Ascend's software advantage is real for general-purpose model training,
  but for inference deployment running only a single fp4+MLA model,
  this advantage is significantly diluted.
```

**11.3.5 Cost Comparison**

```
┌────────────────────────┬──────────────────┬──────────────────┐
│                         │ 8×Ascend 910C     │ 30 FPGA (our approach)│
├────────────────────────┼──────────────────┼──────────────────┤
│ Per-card price (est)   │ ¥80-120K          │ ¥18-21K (10 sets)│
│ Full cluster            │ ¥800K-1.2M        │ ¥1.46M (10 sets) │
│                         │ (incl. Huawei Atlas│ ¥1.53M (100 sets)│
│                         │  chassis)         │                  │
│ SW stack license        │ CANN free         │ In-house, ¥0     │
│ R&D investment          │ Low (CANN mature) │ High (RTL+WLC)   │
│ DeepSeek V4 Decode tput │                   │                  │
│  - Single session (B=1) │ ~400-600 tok/s (est)│ ~660-720 tok/s │
│  - Aggregate (multi-sess)│ ~1,500-2,000 (est)│ ~5,800-8,500    │
│                         │ (fp4→fp8 decomp + │ (fp4 native,     │
│                         │  CANN sched overhead)│ §4.6.1 optimizations on)│
│ $/M token (3yr TCO)    │ ~$12-18 (est)      │ ~$7-9 (10 sets)  │
├────────────────────────┼──────────────────┼──────────────────┤
│ Availability (China)    │ △ Queue 6-18mo    │ ✓ Advance 8-12wk│
│ Deployability (overseas)│ ✗ Nearly impossible│ ✓ Global        │
│ Supply chain certainty │ ★★                │ ★★★★            │
└────────────────────────┴──────────────────┴──────────────────┘

Ascend's per-card price range is wide because Huawei prices differently for different customers,
and it fluctuates significantly with capacity constraints. At 10K-unit volume, FPGA unit cost
drops below ¥10K/chip; Ascend has no corresponding high-volume discount path.

Root cause of throughput gap: fp4→fp8 decompression ≈ 10-15% extra latency,
CANN scheduling ≈ 5-10% overhead, MLA software implementation ≈ additional overhead.
These three factors combined mean Ascend's actual B=1 decode throughput
is lower than its paper specs. FPGA's native fp4 +
hardware MLA acceleration avoids overhead on all three points.
```

**11.3.6 Ascend Comparison Core Conclusion**

```
Common perception of the competitive landscape:
  NVIDIA (best) > Ascend (domestic alternative) > FPGA (niche compromise)

Actual competitive landscape for DeepSeek V4 Pro inference:

  Technical fit (fp4 + MLA, B=1 decode):
    FPGA > NVIDIA B200 (>$30K, sanctioned) > Ascend (no fp4 support)

  Obtainable in mainland China:
    Ascend ≈ FPGA > smuggled NVIDIA > legitimate NVIDIA (=0)

  Deployable overseas:
    FPGA > downgraded H20 > Ascend (=0)

  Software maturity:
    NVIDIA > Ascend > FPGA

  $/M token (at scale):
    FPGA ~$5-7 ≈ Ascend estimated ~$5-8 > NVIDIA ~$9-12

FPGA is not "the backup that can't match Ascend."
For the specific workload of DeepSeek V4 Pro inference:
  ① fp4 + MLA silicon support: FPGA unique, Ascend unsupported
  ② China supply certainty: FPGA 12 weeks, Ascend 12 months
  ③ Overseas deployment permission: FPGA global, Ascend zero
  ④ Toolchain self-control: FPGA full-stack in-house, Ascend depends on Huawei

Ascend's advantages in general model training, software ecosystem, and Huawei brand trust —
but these three points cannot cover its silicon-level architectural disadvantage
in the "fp4 + MLA + overseas deployment" scenario. For the target customers defined in this document
(Chinese overseas enterprises needing private deployment of DeepSeek V4 Pro inference),
FPGA is a substantively superior solution to Ascend.
```

### 11.4 Total Addressable Market (TAM) Estimation

**Review challenge: "Is compute demand clearly established? Plainly speaking, can the cards actually be sold?"**

This is the most fundamental business question for the entire proposal. If demand does not exist, all technical arguments are castles in the air. Below we first confront the demand-reality question head-on, then proceed to quantitative TAM estimation.

**11.4.0 Demand Reality: Why This Market Is Not Imaginary**

**I. Demand Is Policy-Created, Not Market-Hyped**

```
GPU export controls are not temporary market fluctuations, but structural, irreversible geopolitical reality:

  Controls have continuously tightened since 2024:
    ● H100/B200 → globally embargoed to China (3A090 rule)
    ● H20 → added to control list in 2025
    ● AMD MI300X → equally controlled
    ● Geographic expansion: China → Middle East → some SEA countries

  Result: a massive "pent-up demand pool":
    Global high-end GPU inference server annual shipments ~80K-120K units
    Of which demand suppressed by controls ~30K-50K units/year
    → This demand has not disappeared; it is merely waiting for obtainable alternatives

This is not a question of "can FPGA create a new market,"
but rather "existing GPU demand has had its supply cut off by controls — can FPGA fill the gap."
```

**II. Target Customers' Real Predicament — We Are Not Seeking Demand; Demand Is Seeking a Path**

```
The objective situation of three customer categories:

A. Chinese AI companies going global (highest certainty):
   ● Scenario: Own models need inference deployment in SEA/ME
   ● Status quo: Cannot rent GPUs in overseas data centers (sanctions); Ascend cannot go global (Huawei sanctions)
   ● Choice: FPGA or abandon overseas business
   ● Demand rigidity: High — overseas users exist, revenue exists, not doing it means losing market
   ● Case reference: ByteDance overseas AI inference demand grew >300% YoY in 2024,
              but GPU supply grew near zero, all barely sustained by domestic H20 inventory

B. State-owned enterprise (SOE) overseas institutions:
   ● Scenario: Bank overseas branch AI customer service/risk control, carrier overseas AI value-added services
   ● Status quo: Data cannot leave internal network (compliance), public cloud API unavailable
   ● Choice: FPGA private deployment or abandon AI capability
   ● Demand rigidity: Medium-high — budgets exist, mandates exist, procurement processes exist
   ● Key feature: Procurement decisions consider not just $/token, but "can it be deployed"

C. Overseas local enterprises (SEA/ME/LATAM):
   ● Scenario: Local finance/government needs AI but cannot/will not buy Chinese cloud APIs
   ● Status quo: Cannot buy GPUs locally either; Ascend has no ecosystem locally
   ● Choice: FPGA or wait (no end in sight)
   ● Demand rigidity: Medium — market education takes time, but structural shortage exists
```

**III. Proof by Contradiction: If "cards cannot be sold," which assumption would fail?**

```
For demand to go to zero, at least one of the following must be true:

  ✗ GPU controls lifted → extremely unlikely (this is structural policy, not reversible)
  ✗ Chinese models no longer need overseas deployment → contrary to current trends (TikTok, Temu,
    Shein, gaming going global are all accelerating)
  ✗ Ascend can be freely exported → Huawei equally sanctioned, and SMIC capacity bottlenecked
  ✗ Customers would rather abandon AI than buy FPGA → possible (some customers), but out of 200
    potential customers, only 5-10 need to say "yes" for the 10-set validation target to be met
  ✗ Competitors emerge → good news, proves market exists. FPGA's fp4 native + exportable
    nature is a structural differentiator

Core thesis: The demand pool is known to exist (suppressed GPU inference demand ~30K-50K units/year).
FPGA does not need to create new demand; it only needs to capture 0.5-2% of this 30K-50K unit/year gap.
This is not "selling ice to Eskimos," but "selling legal alternative beverages during Prohibition."
```

**IV. Phased Demand Validation Path — No Need to Bet Everything at Once**

```
Demand validation itself is what Phases 1-3 are designed to accomplish:

  Phase 1 (Now-12 months): Not selling cards — validating whether demand exists
    → Not waiting for orders before building, but building to get orders
    → 2 dev boards validate technology → take benchmark data to talk to customers
    → Goal: In-depth technical discussions with 3-5 potential customers
    → Success criterion: At least 1 customer signs MOU/LOI (payment not required)

  Phase 2 (12-24 months): Seed customers validate business closure
    → 10-cluster deployment to 1-2 real customer scenarios
    → Goal: Validate "customers willing to pay" + "FPGA can be operated"
    → Success criterion: At least 1 customer repurchases or expands
    → If zero customers willing to pay at this point → cut losses, total investment ~¥20M, manageable

  Phase 3 (24-36 months): Commercial scaling
    → Based on seed customer cases, expand to 100 sets
    → At this point demand is no longer "imagined" but "on the order book"

Key principle: 10 sets is a market validation investment, not a capacity investment.
        If 10 sets cannot find a customer, it proves demand truly does not exist — cut losses promptly.
        But without even doing 10 sets, we will never know whether demand is real.
```

**V. Candid "Demand = 0" Scenario Analysis**

```
Assuming the worst case — zero commercial orders in 3 years:

  Sunk cost:
    Hardware: ¥2.3M × N (N≤10, unsold prototype hardware can be disassembled)
    R&D: ¥6.75M (RTL IP can be retained, usable for other acceleration scenarios)
    Operations: ¥0.43M × N years

  Worst-case total loss: ~¥10-15M (Phase 1 stop-loss)

  Comparative reference:
    Equivalent-scale GPU company annual GPU depreciation: ¥50-200M
    Huawei Ascend annual R&D investment: ¥10B+

  This is not a "bet the company" wager.

  Moreover, "3 years zero orders" in the current supply-demand landscape requires
  nearly all external conditions to deteriorate simultaneously:
    Controls relax + models stop going global + Ascend export ban lifted + customers refuse to try
    → extremely low probability
```

Below we present quantitative TAM estimation from both Bottom-Up and Top-Down perspectives:

**Bottom-Up: Breakdown by Customer Profile**

```
┌──────────────────────────────────┬──────────┬──────────┬──────────┐
│ Customer Profile                  │ Near 1-2yr│ Mid 3-5yr│ Long 5-10yr│
│                                  │ (10-50 sets)│(100-500 sets)│(1K-5K sets)│
├──────────────────────────────────┼──────────┼──────────┼──────────┤
│ A. Chinese Tech Going Global     │          │          │          │
│   TikTok/ByteDance (SEA/ME AI)   │ 30-50    │ 80-150   │ 300-500  │
│   Alibaba Cloud Intl (AI Region) │ 10-20    │ 50-100   │ 200-400  │
│   Tencent/Baidu/Kuaishou overseas│ 10-20    │ 40-80    │ 150-300  │
│   Subtotal                        │ 50-90    │ 170-330  │ 650-1200 │
├──────────────────────────────────┼──────────┼──────────┼──────────┤
│ B. SOE Overseas Institutions     │          │          │          │
│   Big-4 bank overseas (AI CS/risk)│ 20-40   │ 60-120   │ 200-400  │
│   Top-3 carrier overseas (AI VAS)│ 10-20    │ 30-60    │ 100-200  │
│   Belt & Road projects (infra AI)│ 10-15    │ 30-50    │ 80-150   │
│   Subtotal                        │ 40-75    │ 120-230  │ 380-750  │
├──────────────────────────────────┼──────────┼──────────┼──────────┤
│ C. Target Market Local Enterprises│         │          │          │
│   SEA finance/e-commerce          │ 10-20    │ 40-80    │ 150-300  │
│   ME oil/gov/finance              │ 10-20    │ 40-80    │ 150-300  │
│   LATAM telecom/finance           │ 5-10     │ 20-40    │ 80-150   │
│   Africa gov digitalization       │ 5-10     │ 20-40    │ 80-150   │
│   Subtotal                        │ 30-60    │ 120-240  │ 460-900  │
├──────────────────────────────────┼──────────┼──────────┼──────────┤
│ D. Global Regulated (model-neutral)│         │          │          │
│   Medical AI (private imaging/diag)│ 10-20   │ 40-80    │ 150-300  │
│   Financial compliance (AML/risk) │ 10-20    │ 40-80    │ 150-300  │
│   Gov/defense (friendly nations)  │ 5-10     │ 20-50    │ 80-200   │
│   Subtotal                        │ 25-50    │ 100-210  │ 380-800  │
├──────────────────────────────────┼──────────┼──────────┼──────────┤
│ Total (FPGA cluster sets)        │ 145-275  │ 510-1010 │ 1870-3650│
│ Median estimate                   │ ~200     │ ~700     │ ~2,500   │
└──────────────────────────────────┴──────────┴──────────┴──────────┘
```

**Feasibility Check: Benchmarking Against Known Market Data**

```
Global GPU inference server shipments (2025, estimated): ~500K units/year
Of which high-end inference (H100/B200 class):          ~80K-120K units/year
Of which sanctioned markets (China+specific countries):  ~30K-50K units/year (suppressed demand from controls)

FPGA clusters are not going after the existing GPU market, but serving "demand suppressed by GPU controls."
China's GPU inference demand gap alone is approximately 30K-50K units/year.
Even if FPGA captures 5-10%, that is 1,500-5,000 sets/year.

Adding overseas local demand (SEA/ME/LATAM/Africa), long-term 2,500 sets is reachable.
10,000 sets requires Chinese models to hold 15-20% share of global inference market → needs 5-10 years.
```

**Top-Down Cross-Validation:**

```
Global LLM inference market (2028, conservative estimate): $50B
Private deployment share:                                   20% = $10B
Chinese model share of private deployment:                  15% = $1.5B
FPGA-capturable hardware share (non-GPU zone):              30% = $450M
Per-set FPGA cluster annual TCO:                           ~$130K (100-set tier)
Supportable deployed sets:                                 $450M / $130K ≈ 3,500 sets

Consistent order-of-magnitude with Bottom-Up mid-term (~700) and long-term (~2,500) spanning 3-5 years.
```

**Three-Tier Business Milestones:**

```
10 sets (Near-term 12-18 months):
  → 1-2 seed customers (e.g., a bank overseas branch + an SOE overseas project)
  → Validate "FPGA can be deployed, operated, and delivered"
  → Customer willingness-to-pay validated → price anchoring
  → Milestone: First commercial contract signed

100 sets (Mid-term 2-4 years):
  → 5-10 industry customers
  → Typical: ByteDance overseas 30 sets + Alibaba Cloud Intl 15 sets + 3 banks 10 sets each + others
  → FPGA volume supply chain established, cost enters $7/M token range
  → Milestone: Single customer >10 sets repeat purchase

10,000 sets (Long-term 7-10 years):
  → Carrier/cloud-provider scale procurement (hundreds of customers)
  → Chinese models become one of the global mainstream options, FPGA becomes standard inference hardware
  → Requires: DeepSeek/Chinese models sustaining leadership + FPGA path validated by the market
  → Milestone: Single contract >100 sets
```

**Candid Uncertainties:**

The largest variables in the above estimates:
1. Whether DeepSeek can sustain model competitiveness (if surpassed, TAM goes to zero)
2. Control trends (if relaxed, FPGA's premium over GPU is compressed; if intensified, FPGA TAM expands)
3. Whether customers accept the operational model of "non-CUDA hardware"

**Conclusion**: A clearly identifiable target market exists — 200 sets (near-term) → 700 sets (mid-term) → 2,500 sets (long-term). 10,000 sets is the North Star, requiring Chinese models to dominate global inference market share. The market is large enough; the question is execution, not TAM itself.

```

### 11.5 Panoramic Comparison of Five Mainstream Domestic Compute Cards

> April 2026 real market data. All domestic cards lack native fp4 support.
> **Actual market price is approximately 5× the official list price** (supply-demand imbalance + capacity constraints + channel markup).

**11.5.1 Core Specification Comparison (with Official List Price and Actual Market Price)**

```
┌──────────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬──────────────┐
│                    │ Huawei Ascend│ Hygon DCU  │ Kunlunxin 3 │ Moore Threads│ Cambricon   │ Our FPGA     │
│                    │ 950PR       │ Z100        │ P800        │ MTT S5000   │ MLU370-X8   │ AGM 039-F    │
│                    │ (Atlas 350) │             │             │             │ (dual-chip) │ (single chip)│
├──────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼──────────────┤
│ Architecture       │ Da Vinci(custom)│ GPGPU+ROCm│ XPU-P/R     │ Pinghu(MUSA)│ MLUarch03   │ FPGA streaming│
│ Process            │ equiv 5nm(N+3)│ —          │ —           │ 7nm         │ —           │ Intel 7(10nm)│
│ FP8 Compute        │ —           │ 512 TFLOPS  │ 320 TFLOPS  │ 1000 TFLOPS │ 192 TFLOPS  │ — (non-GPU)  │
│ INT8 Compute       │ 4096 TOPS   │ 1024 TOPS   │ 1280 TOPS   │ 2048 TOPS   │ 256 TOPS    │ —            │
│ fp4 E2M1 Native    │ ✗ (FP4@decomp)│ ✗          │ ✗           │ ✗           │ ✗           │ ✓ 11 TMACs   │
│ Memory             │ 112 GB HBM  │ 64 GB HBM2e │ 64 GB GDDR6 │ 64 GB GDDR6 │ 48 GB LPDDR5│ 32 GB HBM2e  │
│ Bandwidth          │ 1.4 TB/s    │ 933 GB/s    │ 768 GB/s    │ 819 GB/s    │ 614 GB/s    │ 920 GB/s     │
│ Power              │ 600W        │ 350W        │ 300W        │ 400W        │ 250W        │ 120W         │
├──────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼──────────────┤
│ Official MSRP (2026.4)│ ~¥50K     │ ~¥28K       │ ~¥32K       │ ~¥35K       │ ~¥22K       │ ~¥18K        │
│ Actual market (×5) │ ~¥250K      │ ~¥140K      │ ~¥160K      │ ~¥175K      │ ~¥110K      │ ≈ MSRP (in stock)│
│ 8-card system actual│ ~¥2.0M     │ ~¥1.12M     │ ~¥1.28M     │ ~¥1.40M     │ ~¥880K      │ ~¥1.33M      │
│                    │             │             │             │             │             │ (32 chips×4/card)│
├──────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼──────────────┤
│ Core positioning   │ LLM inference│ General compute│ Internet infer│ Train+Infer │ Inference-focused│ fp4 decode  │
│                    │ Prefill+Rec │ CUDA migration│ Finance     │ LLM adaptation│ Small/med train│ Specialized │
└──────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴──────────────┘

Note: 950PR "FP4 1.56 PFLOPS" is Huawei's official marketing figure, but the Da Vinci architecture
      lacks native fp4 MAC units; it is actually fp4→FP8 decompress-then-compute, not true native fp4
      inference. See §11.6.2 for details.
```

**11.5.2 Key Findings**

```
I. fp4 Native: FPGA's Uniqueness

  All five domestic cards + NVIDIA H100 (non-B200) + Ascend entire lineup → none support native fp4 E2M1.
  FPGA is currently the only chip obtainable in China capable of native fp4 inference.

  This means DeepSeek V4 Pro's fp4 weights all require "decompress→FP8→compute" on domestic cards:
    Weight load volume unchanged (fp4 6.1 GB), but the decompression step consumes ALU + power + latency.
    FPGA takes "fp4→BRAM→DSP" two steps; domestic GPU/NPU take three steps.

  950PR's advertised "FP4 1.56 PFLOPs" is a marketing number — the Da Vinci Cube Unit can only do FP8 MAC;
  fp4→FP8 decompression is done by the Vector Unit, reducing actual effective throughput by 10-20%.

II. Memory Capacity vs Bandwidth: The Decode Scenario Mismatch

  All domestic cards have 48-112 GB of memory, far exceeding the actual decode single-session requirement (~6 GB).
  But the decode bottleneck is bandwidth, not capacity:

  Bandwidth-to-Compute Ratio (MBW, GB/s per TFLOP — higher is better for decode):
    Ascend 950PR:  1.4 TB/s / 1,560 TFLOPS(FP4) ≈ 0.9 GB/T
    Hygon Z100:    933 GB/s / 512 TFLOPS(FP8)  ≈ 1.8 GB/T
    Kunlunxin P800: 768 GB/s / 320 TFLOPS(FP8) ≈ 2.4 GB/T
    Moore S5000:   819 GB/s / 1,000 TFLOPS(FP8) ≈ 0.8 GB/T
    Cambricon X8:  614 GB/s / 192 TFLOPS(FP8)  ≈ 3.2 GB/T
    FPGA A7 M:     920 GB/s / 11 TMACs(fp4)    ≈ 110 GB/T  ← 23-122× advantage

  GPU/NPU are designed for compute-bound scenarios (training, prefill) — surplus compute, insufficient bandwidth.
  FPGA is designed for memory-bound scenarios (decode) — compute just right, bandwidth abundant.

  This is the quantified expression of "using a GPU for decode is like using a sledgehammer to crack a nut":
    Cambricon 192 TFLOPS compute, but decode B=1 uses only ~2% → 96% compute idle
    FPGA 11 TMACs compute, decode B=1 uses ~50% → compute matched to bandwidth

III. Actual Price 5×: Quantified Evidence of GPU Scarcity

  Official MSRP 5× actual transaction price = a signal of supply-demand imbalance:
    - SMIC 7nm capacity contested by phone SoCs / base stations / NPUs
    - CoWoS advanced packaging capacity concentrated at TSMC (sanctioned) → domestic capacity scarce
    - Domestic GPU annual shipments ~500K-800K units vs demand >2M units

  FPGA is not dependent on these bottlenecks:
    - Intel global fab (US/Ireland/Israel)
    - HBM from Korea (SK Hynix/Samsung)
    - Standard packaging (not dependent on CoWoS)
    - Not subject to GPU compute sanctions (TPP far below threshold)
    → Actual price = official price (no premium)

IV. PD Disaggregation Cannot Solve the Domestic GPU Decode Dilemma

  PD Disaggregation (Prefill/Decode Disaggregation) is a software-level optimization;
  all domestic GPUs can implement it through their respective software stacks (CANN/ROCm/MUSA/etc.).

  But after PD disaggregation, the decode node's hardware bottleneck remains unchanged:
    - Compute idle problem worsens (decode B=1 Tensor Core utilization ~2-8%)
    - Large memory advantage cannot translate to decode throughput (bottleneck is bandwidth, not capacity)
    - fp4 decompression overhead unchanged (no domestic GPU has native fp4)

  PD disaggregation essentially "prevents idle compute from being even more idle" — moving prefill away,
  decode cards remain bottlenecked by HBM bandwidth; memory capacity offers no help.

  See §11.5.3 "Context Length Advantage" for quantitative analysis of decode nodes after PD disaggregation.
```

**11.5.3 Domestic Card Decode Scenario Quick Ranking**

```
DeepSeek V4 Pro Decode single session (B=1) estimated throughput (ranked by HBM bandwidth):

  ┌──────────────────┬──────────────┬──────────────┬──────────────┐
  │ Chip              │ HBM Bandwidth│ Single sess  │ Bottleneck    │
  │                  │              │ decode est   │               │
  ├──────────────────┼──────────────┼──────────────┼──────────────┤
  │ Ascend 950PR     │ 1.4 TB/s     │ ~250-350     │ fp4 decomp+BW │
  │ Moore S5000      │ 819 GB/s     │ ~180-250     │ fp4 decomp+BW │
  │ Hygon Z100       │ 933 GB/s     │ ~200-280     │ fp4 decomp+BW │
  │ Kunlunxin P800   │ 768 GB/s     │ ~170-240     │ fp4 decomp+BW │
  │ Cambricon X8     │ 614 GB/s     │ ~140-200     │ fp4 decomp+BW │
  │ FPGA A7 M (single)│ 920 GB/s    │ ~660-720     │ BW near-sat   │
  └──────────────────┴──────────────┴──────────────┴──────────────┘

  FPGA single-chip decode throughput is 2-5× domestic GPU, reasons:
    1. fp4 native (no decompression, zero ALU waste)
    2. Bandwidth/compute ratio 110 GB/T (domestic GPU 0.8-3.2 GB/T, 34-122× worse)
    3. Streaming architecture (no kernel launch overhead; domestic GPU: CANN/ROCm scheduling 10-30μs/kernel)
    4. MLA 6-stage hardware pipeline (domestic GPU: software implementation, 6 attention kernels × 30μs ≈ 180μs/layer)

  System-level (8-card cluster, TP=8):
    Ascend 950PR 8-card: ~2,000-2,800 tok/s (aggregate, limited by MoE All-to-All communication)
    FPGA 32-chip:       ~5,800-8,500 tok/s (aggregate, §4.6.1 optimizations on)

  Note: GPU's advantage lies in prefill (large batch, high compute utilization).
  But in decode-only or agent (B=1) scenarios, FPGA is the structurally superior solution.
```

---

### 11.6 Ascend 950PR In-Depth Comparative Analysis

> Huawei Ascend 950PR is the latest mass-production model in the domestic AI chip lineup. Note: 950PR's claimed
> "FP4 1.56 PFLOPS" is the fp4→FP8 decompress-equivalent compute, not native fp4 MAC.
> Below uses actual market specifications (112GB HBM, 1.4 TB/s, 600W, ¥250K/card actual price).

**11.6.1 Full Hardware Specification Comparison**

> Single chip/card → Single inference cluster (8-card node) → DeepSeek V4 Pro inference measured estimates.

**I. Chip-Level Comparison**

```
┌────────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Parameter               │ NVIDIA H100 SXM   │ Ascend 950PR     │ AGM 039-F (FPGA) │ Custom ASIC (target)│
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Process                 │ TSMC 4nm (N4)    │ equiv 5nm (N+3)  │ Intel 7 (10nm)   │ TSMC 12nm        │
│ Die area (est.)          │ ~814 mm²         │ ~600 mm² (est.)  │ ~800 mm² (est.)  │ ~500-700 mm²     │
│ Transistors (est.)       │ ~80B             │ ~40B (est.)      │ ~25B (est.)      │ ~30-40B          │
│ Architecture             │ 1 GPU die        │ 1 Da Vinci die   │ 1 FPGA           │ 4 FPGA merged 1  │
│                          │ + 5×HBM          │ + 4×HiBL         │ + 2×HBM2e        │ + 8×HBM3         │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Compute precision & peak:│                  │                  │                  │                  │
│  FP16/BF16              │ 989 TFLOPS       │ ~500 TFLOPS      │ — (non-GPU paradigm)│ —              │
│  FP8                    │ 1,979 TFLOPS     │ ~1,000 TFLOPS    │ —                │ ~500 TFLOPS (est)│
│  fp4 E2M1 (native)       │ ✗ (B200+ only)   │ ✗ (decomp→FP8 req)│ ✓ 11.07 TMACs   │ ✓ hardened ~44 TMACs│
│  INT8                   │ 1,979 TOPS       │ ~1,000 TOPS      │ —                │ ~500 TOPS (est)  │
│  Sparse compute          │ 2× (structured)  │ None             │ None             │ None             │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Memory:                  │                  │                  │                  │                  │
│  Capacity                │ 80 GB HBM3       │ 112 GB HBM       │ 32 GB HBM2e      │ 128 GB HBM3      │
│  Bandwidth               │ 3.35 TB/s        │ ~1.4 TB/s        │ 920 GB/s         │ ~3.2 TB/s (4× stack)│
│  HBM stack count         │ 5× HBM3 (6-high) │ —                │ 2× HBM2e         │ 8× HBM3 (or 4×)  │
│  Total HBM cap (single set)│ 640 GB          │ 896 GB (8 chips) │ 1,024 GB (32 chips)│ 1,024 GB (8 chips)│
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Power:                   │                  │                  │                  │                  │
│  TDP (single chip/card)  │ 700W (SXM)       │ 600W             │ ~120W (per chip) │ ~350W (est.)     │
│  Card-level power (incl VRM)│ 700W           │ 600W             │ ~550W (4-chip/card)│ ~400W (single chip/card)│
│  System power (8-card, incl server)│ ~6.0 kW │ ~5.3 kW          │ ~5.3 kW          │ ~3.8 kW          │
│  Annual electricity (¥0.8/kWh)│ ~¥40K        │ ~¥35K            │ ~¥35K            │ ~¥26K            │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Inter-card interconnect: │                  │                  │                  │                  │
│  Interconnect protocol   │ NVLink 4.0       │ HCCS             │ PCIe 5.0 (cross-card)│ PCIe 5.0      │
│                              + InfiniBand NDR  │ + custom interconnect│ + C2C SerDes(chip-to-chip)│ (on-chip merged)│
│  Inter-card bandwidth    │ 900 GB/s (NVLink)│ ~2.0 TB/s        │ 28 GB/s (PCIe)   │ 28 GB/s          │
│  Cross-node interconnect │ 400 GB/s (IB)    │ ~400 GB/s        │ N/A (single node)│ N/A (single node)│
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Price (single chip/card):│                  │                  │                  │                  │
│  Official MSRP           │ ~$30,000         │ ~¥50K            │ ¥18,000 (~$2,500)│ ~$600-800 (est.) │
│  Actual market (×5)      │ N/A (embargoed)  │ ~¥250K (~$34K)   │ ≈ official (in-stock)│ per chip      │
│  Gross margin (est.)     │ ~65-70%          │ ~40-50%          │ N/A (FPGA spot)  │ ~50% (custom)    │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Supply & Deployment:     │                  │                  │                  │                  │
│  Availability            │ ✗ Sanctions (3A090)│ △ Queue 6-18mo │ ✓ 8-12 weeks     │ ✓ Self-controlled│
│  Global deployment       │ △ Partially limited│ ✗ Huawei sanctioned│ ✓ Std equipment│ ✓ Own chip       │
│  Lead time               │ N/A (embargoed)  │ >6 months        │ 8-12 weeks       │ 16-20 weeks (MPW)│
│  Supply stability        │ ✗ Cut off        │ △ SMIC capacity constrained│ ✓ Intel global fab│ ✓ Multi-foundry│
└────────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**II. Single Inference Cluster Comparison (8-Card Node, DeepSeek V4 Pro Decode)**

```
┌────────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Parameter               │ 8×H100 SXM       │ 8×Ascend 950PR   │ 8-card×4-chip FPGA│ 8×ASIC (4-in-1) │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Chip count              │ 8 GPU            │ 8 Da Vinci       │ 32 FPGA          │ 8 ASIC           │
│                          │                  │                  │ (4 chips/card)   │ (4 FPGA→1 ASIC)  │
│ System total compute (FP8)│ 15.8 PFLOPs    │ ~8 PFLOPs        │ — (fp4 paradigm) │ ~4 PFLOPs        │
│ System total compute (fp4)│ ✗               │ ✗                │ 354 TMACs (32 chips)│ ~354 TMACs (8 chips)│
│ Total memory             │ 640 GB           │ 896 GB (8 chips) │ 1,024 GB (32 chips)│ 1,024 GB (8 chips)│
│ Total HBM bandwidth      │ 26.8 TB/s        │ ~11.2 TB/s       │ 29.4 TB/s (32 chips)│ ~25.6 TB/s (8 chips)│
│ BW/layer (61 layers avg) │ 439 GB/s/layer   │ 184 GB/s/layer   │ 482 GB/s/layer   │ 420 GB/s/layer   │
│ Per-chip BW/layers hosted│ 419 GB/s/layer   │ 175 GB/s/layer   │ 460 GB/s/layer   │ —                │
│ System power (incl server)│ ~6.0 kW         │ ~5.3 kW          │ ~5.3 kW          │ ~3.8 kW          │
│ Cooling                  │ Liquid (recomm.) │ Liquid (recomm.) │ Air (4U)         │ Air (2U)         │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Hardware BOM             │ ~$240K (est.)    │ ~$90K (est.)     │ ~¥1.94M (~$267K) │ ~$35-45K (est.)  │
│ Hardware selling price (incl margin)│ ~$280K │ ~$275K (actual) │ ~$303K (100 sets)│ **~$60-80K**     │
│ Gross margin             │ 65-70% (NVIDIA)  │ 40-50% (Huawei)  │ 45% (FPGA)       │ 50% (custom)     │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ DeepSeek V4 Pro Inference:│                 │                  │                  │                  │
│  Single token decode latency│ ~6-10 ms (est.)│ ~5-7 ms (est.)  │ ~10 ms (est.)    │ ~8-9 ms (est.)   │
│  Decode single-sess (B=1)│ ~600-800 tok/s  │ ~1,200-1,600    │ ~660-720 tok/s   │ ~900-1,100       │
│  Decode aggregate (multi-sess)│ ~2,000-3,000│ ~2,500-4,000    │ ~5,800-8,500     │ ~6,000-9,000     │
│                         │                  │   tok/s (needs decomp)│ (fp4 native) │   tok/s (est.)   │
│  Prefill capability      │ ★★★★★ (strong) │ ★★★★ (strong)   │ ★★ (weak, non-target)│ ★★ (weak)    │
│  Batch=1 compute util    │ ~2-3%            │ ~5-8%            │ ~50% (DSP pinned)│ ~50% (hardened)  │
├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ $/M token (HW depreciation)│ $12-20         │ $16-25           │ $5.9             │ **$2.5-3.5**     │
│  (70% util, 3yr deprec) │                  │                  │                  │                  │
│ Annual electricity (HW)  │ ~$5.5K           │ ~$5.9K           │ ~$4.9K           │ ~$3.5K           │
│ Annual elec ($/M token)  │ ~$0.3            │ ~$0.3            │ ~$0.3            │ ~$0.2            │
└────────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘

ASIC throughput derivation:
  Decode bottleneck = Total HBM bandwidth / per-token weight load
  FPGA: 29.4 TB/s → 980 tok/s
  ASIC: 25.6 TB/s → 980 × (25.6/29.4) ≈ 850 tok/s (HBM bandwidth slightly lower)
  But 4-chip C2C inter-chip communication becomes on-chip bus → saves ~1.2μs/hop × 61 layers ≈ 73μs/layer
  → Actual latency slightly better, throughput ≈ 900-1,100 tok/s

  Core change: Throughput roughly unchanged (dominated by HBM bandwidth); hardware selling price from $303K → $60-80K (~1/4).
  ASIC's value is not "faster" — it is the FPGA-validated architectural advantage physically hardened at 1/4 the hardware cost —
  two orders-of-magnitude dimensions (effective bandwidth + cost discontinuity) simultaneously present in a single product.
```

**II.5: Bandwidth/Layer Is the Root Cause of Decode Performance — Why FPGA Aggregate Throughput Crushes 950PR**

```
Decode bottleneck = HBM bandwidth / per-token weight load. But fair comparison must normalize to "per layer":

  ┌──────────────────┬──────────────┬──────────────┬──────────────┐
  │                   │ 8×H100       │ 8×950PR      │ 32×FPGA      │
  ├──────────────────┼──────────────┼──────────────┼──────────────┤
  │ Total BW          │ 26.8 TB/s    │ 11.2 TB/s    │ 29.4 TB/s    │
  │ Layers per chip   │ ~8 layers    │ ~8 layers    │ ~2 layers    │
  │ Per-chip BW/layer │ 419 GB/s/layer│ 175 GB/s/layer│ 460 GB/s/layer│
  │ Relative to FPGA  │ 0.91×       │ 0.38×       │ 1.00× (baseline)│
  └──────────────────┴──────────────┴──────────────┴──────────────┘

  Conclusion: FPGA's BW/layer is 2.63× that of 950PR, and 1.10× that of H100.
        950PR's 112 GB/chip appears to offer large capacity, but 8 layers share 1.4 TB/s →
        only 175 GB/s per layer, less than 40% of FPGA (460 GB/s).

Why doesn't single-session show the 2.63× advantage?

  At B=1, the bottleneck shifts from bandwidth to communication:
    - FPGA 32 chips × ~0.04ms per C2C hop → pipeline traversal overhead is significant
    - 950PR 8 chips × ~0.02ms per HCCS hop → shallower 8-hop depth, lower communication overhead
    - MoE All-to-All across 32 chips has ~4× the dispatch/gather hop count vs 8 chips
    → At B=1, communication overhead share is high, partially offsetting bandwidth advantage

  But at B≥4, communication overhead is amortized by multi-token concurrency:
    - Multiple tokens' All-to-All can be merged → per-token communication overhead drops sharply
    - Bandwidth/layer advantage fully unleashed → 2.1-2.3× aggregate throughput advantage

Why can ASIC single-session reach 900-1,100 tok/s?

  ASIC = 4 FPGA merged into 1 chip → 8 chips cover 61 layers → pipeline depth from 32→8:
    - Each chip hosts ~8 layers (same as 950PR), but on-chip interconnect replaces C2C SerDes
    - BW/layer: 3.2 TB/s / 8 layers = 400 GB/s/layer (still > 950PR's 175)
    - Communication overhead drops from 32 hops to 8 → B=1 performance improves substantially
    → ASIC single-session 900-1,100 tok/s vs 950PR 1,200-1,600 tok/s
      (BW/layer 2.3× but 950PR HCCS latency is lower, narrowing the gap)

Core insight:

  32-chip distribution is not a disadvantage — it buys 2.63× BW/layer.
  The cost is higher communication overhead at B=1 (exactly what the ASIC phase addresses).
  But in real deployments (multi-user concurrent, Agent/Chat mixed), aggregate throughput is the billable metric;
  FPGA's 5,800-8,500 tok/s vs 950PR's 2,500-4,000 tok/s = 2.1-2.3× advantage,
  a direct reflection of the 2.63× BW/layer.
```

**II.7: Two FPGA Deployment Configurations — HBM-Only vs HBM+DDR (Vendor Performance Model Validation)**

```
FPGA's 32-chip high-bandwidth configuration is not the only option. FPGA vendors offer two memory configurations:

  ┌──────────────────────────┬──────────────────────┬──────────────────────┐
  │                           │ HBM-Only (32 GB)     │ HBM+DDR (32+128 GB)  │
  ├──────────────────────────┼──────────────────────┼──────────────────────┤
  │ FPGA count (storage-constrained)│ >25 chips       │ >=5 chips            │
  │ Per-chip total memory      │ 32 GB HBM2e          │ 32 GB HBM2e + 128 GB │
  │                          │                      │         DDR           │
  │ Weight storage strategy   │ All in HBM           │ DDR stores weights,   │
  │                          │                      │ HBM runs KV Cache +    │
  │                          │                      │ active layers          │
  ├──────────────────────────┼──────────────────────┼──────────────────────┤
  │ B=1 BW tok/s/chip         │ 24.3 ~ 25.1          │ 29.0 ~ 29.9          │
  │ B=1 compute tok/s/chip (ceiling)│ 898 (88T INT8) │ 898 (88T INT8)       │
  │ B=32 compute tok/s/chip/batch│ 28.1 (≈898/32)   │ 28.1                 │
  ├──────────────────────────┼──────────────────────┼──────────────────────┤
  │ System aggregate tput (B≥4)│ ~5,800-8,500 tok/s  │ ~800-1,500 tok/s     │
  │                          │ (32 chips, all HBM, hi-tput)│ (5-8 chips, HBM+DDR, econ)│
  │ Target scenario           │ High-concurrency API / Agent│ Private deploy / single-user│
  │ Relative to 950PR tput advantage│ 2.1-2.3×      │ 0.3-0.6× (cost-oriented)│
  └──────────────────────────┴──────────────────────┴──────────────────────┘

Key findings (vendor model validated):

  ✓ Compute ceiling 898 tok/s/chip vs bandwidth floor 24-30 tok/s/chip → 37:1 gap
    → Compute is never the bottleneck. Even at B=32, compute ceiling 28.1 batches/s × 32 tok/batch = 898 tok/s
    → Identical to B=1 compute ceiling → compute ceiling is independent of batch size
    → Fundamentally validates the thesis that "bandwidth/compute ratio determines decode performance"

  ✓ DDR's core value is not acceleration but cost reduction:
    - 5 HBM+DDR FPGAs can hold the entire model weights → chip BOM from 32→5 (6.4×)
    - Cost is total bandwidth from 29.4 TB/s → 4.6 TB/s → throughput scales proportionally
    - Applicable scenarios: single-user private deployment, edge inference, cost-sensitive scenarios
    - At this point per-chip throughput 29 tok/s × 5 = 145 tok/s (B=1), still adequate for personal use

  ✓ Two configurations cover the full spectrum:
    High-throughput config (32 HBM):  vs 950PR 8-card → 2.1-2.3× aggregate throughput
    Economy config (5 HBM+DDR): vs private deployment → chip BOM ¥175K, 950PR 8-card ¥2.0M
    → FPGA can "downgrade" via DDR to extreme cost efficiency; GPU/NPU have no such cost-reduction path
    (950PR's 112 GB HBM cannot be downgraded — that is the chip's physical specification)

  ✓ Comparison with 950PR:
    Economy config (5 FPGA + DDR): BV=1 effective BW ~460 GB/s / ¥175K = 26 GB/s/10K-yuan
    950PR 8-card actual price:     BV=1 effective BW ~175 GB/s / ¥2.0M = 0.88 GB/s/10K-yuan
    → Effective bandwidth/$ is ~30× that of 950PR; lower chip BOM is the result of bandwidth architecture choices
```


**III. Key Differences at a Glance**

```
Compute dimension:
  H100:     FP8 king (1,979 TFLOPS), no native fp4 → model weights 2× waste
  950PR:    FP8 domestic best (1,000 TFLOPS), fp4 requires decompression → ~15-20% efficiency loss
  FPGA:     fp4 native (11 TMACs/chip × 32 chips), no FP8 → purpose-optimized for fp4 inference
  ASIC:     4 FPGA merged into 1 chip, fp4 hardened ~44 TMACs/chip → on-chip interconnect replaces C2C SerDes

Memory dimension:
  H100:     80 GB HBM3, 3.35 TB/s → highest per-card capacity
  950PR:    112 GB HBM, 1.4 TB/s → among largest domestic memory capacities
  FPGA:     32 GB HBM2e × 32 chips = 1,024 GB, 29.4 TB/s aggregate bandwidth
  ASIC:     128 GB HBM3 × 8 chips = 1,024 GB, 25.6 TB/s → capacity unchanged, bandwidth slightly lower

Power dimension:
  H100:     700W/card → system 6.0 kW, liquid cooling required
  950PR:    600W/card → system 5.3 kW, liquid cooling required
  FPGA:     550W/card (4 chips) → system 5.3 kW, air cooling feasible (4U)
  ASIC:     ~400W/card (single chip) → system 3.8 kW, air cooling easy (2U), 28% lower than FPGA

Price dimension:
  H100:     $30K/card → 8-card $280K (unobtainable)
  950PR:    Official ¥50K/card → actual ¥250K/card (5× premium) → 8-card ¥2.0M (~$275K)
  FPGA:     $26K/card (4 FPGA chips) → 8-card $303K (8-12 week lead time)
  ASIC:     ~$8-10K/card (1 chip) → 8-card **$60-80K** (self-controlled, industry lowest)

$/token dimension (DeepSeek V4 Pro, 70% util, pure hardware depreciation):
  H100:     $12-20/M  — but unobtainable; discussion moot
  950PR:    $18-28/M  — domestic best, but fp4 decompression drags efficiency, actual price inflates depreciation
  FPGA:     $5.9/M    — fp4 native + aggregate bandwidth 29.4 TB/s compensates for per-chip bandwidth disadvantage
  ASIC:     $2.5-3.5/M — architectural efficiency hardened + manufacturing cost advantage compounded; $/token ~40-60% of FPGA

Throughput dimension (DeepSeek V4 Pro Decode, single set):

  Key premise: BW/layer is the root cause of decode throughput
    FPGA:  460 GB/s/layer (920 GB/s ÷ 2 layers/chip)  ← baseline
    950PR: 175 GB/s/layer (1,400 GB/s ÷ 8 layers/chip) ← 38% of FPGA
    H100:  419 GB/s/layer (3,350 GB/s ÷ 8 layers/chip) ← 91% of FPGA

  Single-session decode (B=1, single-user perceived throughput):
    H100:     600-800 tok/s — at B=1 Tensor Core utilization ~2%
    950PR:    1,200-1,600 tok/s — 8-chip pipeline, HCCS low-latency communication
    FPGA:     660-720 tok/s — 32-chip pipeline, C2C communication overhead dominates B=1
                              (2.63× BW/layer advantage offset by 4× pipeline depth communication)
    ASIC:     900-1,100 tok/s — 8-chip pipeline, on-chip interconnect, BW/layer 400 GB/s

  Aggregate decode (multi-session steady-state, B=4-8):
    H100:     ~2,500 tok/s (B=8, but vLLM actual MoE utilization only ~3%)
    950PR:    ~2,500-4,000 tok/s (BW/layer 175 GB/s → still bandwidth-constrained after communication amortized)
    FPGA:     5,800-8,500 tok/s (BW/layer 460 GB/s → bandwidth advantage fully unleashed after communication amortized,
              ─ 2.63× BW/layer ≈ 2.1-2.3× aggregate throughput ✓ consistent)
              ─ §4.6.1 optimizations on: KV expansion + micro-batch + Hot Replication
              ─ Agent 4 req/s: 5,800 tok/s, accept 88%
              ─ Agent 8 req/s: 8,500 tok/s, accept 53%
              ─ + Pipeline Cloning ×2 (§4.8.x): TTFT P95 from 1.15s down to 0.54s
    ASIC:     6,000-9,000 tok/s (est., BW/layer 400 GB/s + shallow pipeline)

Power dimension:
  H100:     700W/card → system 6.0 kW, liquid cooling required
  950PR:    600W/card → system 5.3 kW, liquid cooling required
  FPGA:     550W/card (incl. 4 chips) → system 5.3 kW, air cooling feasible (4U)
  ASIC:     ~120W/card → system 1.8 kW, air cooling easy (2U)

Price dimension:
  H100:     $30K/card → 8-card $280K (unobtainable)
  950PR:    Official ¥50K/card → actual ¥250K/card (5×) → 8-card ¥2.0M (~$275K)
  FPGA:     $26K/card (4 chips) → 8-card $303K (8-12 week lead time)
  ASIC:     ~$20-24K/card (HBM2e@12nm) → 8-card $150-190K (self-controlled)

$/token dimension (DeepSeek V4 Pro, 70% util, pure hardware depreciation):
  H100:     $12-20/M  — but unobtainable; discussion moot
  950PR:    $18-28/M  — domestic best, but fp4 decompression drags efficiency + actual price inflates depreciation
  FPGA:     $5.9/M    — fp4 native + total bandwidth 29.4 TB/s compensates for per-chip bandwidth disadvantage
  ASIC:     $5-7/M (HBM2e@12nm) or $2.5-3.5/M (HBM3@7nm)
```

950PR path (FP8 Tensor Core):
  fp4 weights (HBM, ~6.1 GB)
    → load HBM (6.1 / 2,000 = 3.05 ms)
    → decompress fp4→FP8 (wastes ALU, adds latency)
    → FP8 Tensor Core MAC
  → 3 steps, decompression step consumes compute and power

FPGA path (DSP fp4 native):
  fp4 weights (HBM, ~6.1 GB)
    → load HBM (6.1 / 920 = 6.63 ms)
    → BRAM lookup (does not consume DSP)
    → DSP fp4×fp8 MAC (native)
  → 2 steps, decompression completed in BRAM

**11.6.2 The Most Critical Difference: fp4 Native vs Decompress-then-Compute**

```
The core bottleneck of DeepSeek V4 Pro inference is not compute, but the fp4 processing path:

950PR path (FP8 Tensor Core):
  fp4 weights (HBM, ~6.1 GB)
    → load HBM (6.1 / 1,400 = 4.36 ms)
    → decompress fp4→FP8 (wastes ALU, adds latency)
    → FP8 Tensor Core MAC
  → 3 steps, decompression step consumes compute and power

FPGA path (DSP fp4 native):
  fp4 weights (HBM, ~6.1 GB)
    → load HBM (6.1 / 920 = 6.63 ms)
    → BRAM lookup (does not consume DSP)
    → DSP fp4×fp8 MAC (native)
  → 2 steps, decompression completed in BRAM

Key point: Even though 950PR's HBM bandwidth of 1.4 TB/s > FPGA's 920 GB/s,
      the additional overhead of decompressing fp4→FP8 partially offsets that bandwidth advantage.
      FPGA's fp4 native is an architectural advantage, not something bandwidth numbers can capture.
```

**11.6.3 Context Length Advantage: fp4 Lets HBM Serve KV Cache Rather Than Weights**

> 950PR's single-chip 112 GB HBM appears to crush FPGA's 32 GB, but 950PR hosts 8 layers/chip (14 GB/layer)
> vs FPGA 2 layers/chip (16 GB/layer) — FPGA's actual HBM/layer is 14% higher.
> With fp4 weight halving + actual market price 5× premium, FPGA's context accessibility far exceeds its paper specs.

**I. Single-Chip HBM Actual Allocation (1M context, single session)**

```
┌────────────────────────────┬──────────────────┬──────────────────┐
│                             │ Ascend 950PR     │ FPGA Agilex 7 M   │
├────────────────────────────┼──────────────────┼──────────────────┤
│ Single-chip HBM             │ 112 GB           │ 32 GB            │
│ Layers hosted               │ ~8 layers        │ ~2 layers        │
│ HBM / layer (structural limit)│ 14 GB/layer    │ 16 GB/layer      │
├────────────────────────────┼──────────────────┼──────────────────┤
│ Weights (fp4 vs FP8)        │ ~600 MB (FP8)    │ ~75 MB (fp4)     │
│ KV Cache (1M ctx)           │ ~4.6 GB          │ ~1.15 GB         │
│ Activation/buffer           │ ~1.0 GB          │ ~0.5 GB          │
├────────────────────────────┼──────────────────┼──────────────────┤
│ Actual usage                │ ~6.2 GB          │ ~1.7 GB          │
│ HBM utilization             │ 5.5%             │ 5.4%             │
│ Remaining headroom          │ ~105.8 GB        │ ~30.3 GB         │
│ Single-chip theoretical max context│ ~23M tokens│ ~26M tokens      │
└────────────────────────────┴──────────────────┴──────────────────┘

Key findings:

  ✓ FPGA's HBM/layer (16 GB) **exceeds** 950PR (14 GB/layer) — 14% higher.
    Single-session decode context ceiling is determined by HBM/layer;
    FPGA theoretical max context (~26M) > 950PR (~23M), 13% higher.

  ✓ In the 1M context real-world scenario:
    950PR has 105.8 GB idle (94% HBM wasted)
    FPGA has 30.3 GB idle (95% HBM wasted)
    → Both have ample headroom, but 950PR paid a much higher price for idle HBM (actual price ¥250K/card vs FPGA ¥18K/chip)

  ✓ The value of fp4 weight compression + small-chip architecture:
    - FPGA achieves larger context ceiling with 1/3.5 the HBM capacity
    - System-level total weight footprint: FPGA ~5 GB (fp4) vs 950PR ~38 GB (FP8)
    - System total HBM: FPGA 1,024 GB vs 950PR 896 GB
    → FPGA system-level KV Cache available space is ~161 GB more (supports ~17M more tokens)
```

**II. Concurrency Under Large Context**

```
Single system (FPGA 32 chips vs 950PR 8 chips), 1M context:

┌────────────────────────────┬──────────────────┬──────────────────┐
│                             │ Ascend 950PR     │ FPGA Agilex 7 M   │
├────────────────────────────┼──────────────────┼──────────────────┤
│ System total HBM            │ 896 GB (8 chips) │ 1,024 GB (32 chips)│
│ Weight total footprint (system-level)│ ~38 GB (FP8)│ ~5 GB (fp4)    │
│ Single session KV (1M ctx)  │ ~37 GB           │ ~37 GB           │
│ Single session total        │ ~75 GB           │ ~42 GB           │
│ Remaining for concurrency/larger ctx│ ~821 GB  │ ~982 GB          │
│ 1M ctx max concurrent sessions│ ~11             │ ~23              │
└────────────────────────────┴──────────────────┴──────────────────┘

  ✓ FPGA system total HBM is 128 GB more (14%); headroom is 161 GB more (20%)
  ✓ fp4 weight compression saves ~33 GB → supports ~3 additional 1M ctx concurrent sessions
  ✓ For private deployment (1-2 concurrent), both are more than sufficient, but FPGA's headroom
    can all be invested in Hot Expert Replication (boosting decode throughput) rather than wasted on weights
```

**III. Context-per-Watt: The Hidden Threshold for Large-Context Deployment**

```
Power consumption of a single chip supporting 1M context:

  950PR:  600W → at 1M ctx only ~5% HBM in use, but 600W full power running
          → Effective context-per-watt: 1M / 600W = 1,667 tokens/W

  FPGA:   130W → similarly ~5% HBM in use, 130W running
          → Effective context-per-watt: 1M / 130W = 7,692 tokens/W

  → FPGA's context-per-watt is 4.6× that of 950PR

This means:
  - Under the same power budget, FPGA can support 5.8× the context capacity
  - Edge machine rooms (≤3 kW power) can deploy FPGA large-context inference; 950PR requires liquid-cooled data centers
  - For Agent + long-document analysis and other large-context scenarios, FPGA's deployment threshold is significantly lower
```

**IV. Honest Conclusion**

```
Looking solely at "single-session max context":
  The two are comparable (~26M tokens), because HBM/layer is ~16 GB for both.
  fp4 weight compression (8× per-layer weight savings vs FP8 on 950PR)
  has limited impact on context ceiling in single-session scenarios — KV Cache dominates HBM usage;
  weight share is too small (~1-5%).

Looking solely at "system-level context capacity":
  FPGA is slightly better (~33 GB extra KV space ≈ +3.6M tokens or +3 concurrent sessions),
  but the gap is not decisive enough to be a key selling point.

But looking at "context deployment accessibility":
  ✅ FPGA achieves larger context ceiling than 950PR's 112 GB HBM with only 32 GB HBM2e (26M vs 23M)
  ✅ FPGA achieves 1M context at 130W vs 950PR's 600W — 4.6× context-per-watt
  ✅ fp4 means FPGA does not need to "stack large memory" — small chip + low power = large context
     deployable at the edge rather than requiring data centers
  ✅ This is a victory of architectural efficiency: "FPGA supports larger context with 1/3.5 the HBM + 1/4.6 the power"
  ✅ At actual market price (5× premium), FPGA's context-per-yuan is ~7× that of 950PR
```

**11.6.4 Hardware Pricing Comparison (Pure Hardware Margin, Excluding IP/R&D Amortization)**

> Comparison principle: All three parties compared at hardware selling price (BOM + manufacturing + margin), excluding any R&D/IP amortization.
> NVIDIA does not amortize CUDA R&D into H100 pricing, Huawei does not amortize CANN R&D into 950PR pricing,
> the FPGA solution similarly does not amortize RTL IP into hardware pricing.

```
Benchmarked against a single inference cluster:

┌────────────────────┬──────────────────┬──────────────────┬──────────────────────┐
│                    │ 8×H100 SXM       │ 8×Ascend 950PR   │ FPGA 8-card×4-chip AGM 039│
├────────────────────┼──────────────────┼──────────────────┼──────────────────────┤
│ HW selling price (to customer)│ ~$280K│ ~$110K           │ ~$303K (100 sets)     │
│                    │ (H100 $30K×8      │ (950PR $13.7K×8  │ (~¥2.20M, 45% margin) │
│                    │  + server+IB)     │  + server+HC)    │ ~$202K (10K sets, 50%)│
├────────────────────┼──────────────────┼──────────────────┼──────────────────────┤
│ GPU/chip gross margin│ ~65-70%         │ ~40-50%          │ 35-50% (scale-dependent)│
│                    │ (NVIDIA monopoly premium)│ (domestic sub premium)│ (IT HW standard margin)│
├────────────────────┼──────────────────┼──────────────────┼──────────────────────┤
│ DeepSeek V4 Pro     │ ~600-800         │ ~1,500-2,000     │ ~800-980              │
│  Decode tput (est.) │ tok/s            │ tok/s (needs decomp)│ tok/s (fp4 native)  │
├────────────────────┼──────────────────┼──────────────────┼──────────────────────┤
│ $/M token (HW)     │ $12-20           │ $16-25           │ $5.0-7.2              │
│  (70% util, 3yr)   │ (single set, unobtainable)│ (single set, capacity-limited)│ (100-10K set volume)│
├────────────────────┼──────────────────┼──────────────────┼──────────────────────┤
│ Availability       │ ✗ Sanctions      │ △ Queue 6-18mo   │ ✓ 8-12 week lead time │
│ Global deployment  │ △ Partially limited│ ✗ Huawei sanctioned│ ✓ Standard PCIe device│
└────────────────────┴──────────────────┴──────────────────┴──────────────────────┘
```

```
Key interpretations:

  1. Hardware price comparison alone is meaningless — H100/950PR prices only hold under the premise of "obtainable."
     The real competitive dimension is "effective bandwidth/$" (see §11.A.2 Dimension 1):
       FPGA: ~350 GB/s effective/chip ÷ ¥18K ≈ 194 GB/s/10K-yuan
       950PR: ~175 GB/s effective/card ÷ ¥250K ≈ 7 GB/s/10K-yuan
       → This is an architectural gap (~28×), not a pricing gap

  2. Hardware gross margin:
     H100:  NVIDIA monopoly premium 65-70% → unobtainable, premium is meaningless
     950PR: Huawei domestic-substitution premium 40-50% → 12-month queue, premium = waiting cost
     FPGA:  IT hardware standard margin 35-50% → obtainable, deliverable

  3. $/M token comparison (pure hardware depreciation):
     H100:  $12-20/M  (but unobtainable)
     950PR: $16-25/M  (fp4 decompression efficiency loss)
     FPGA:  $5.0-7.2/M (100-10K set volume, direct projection of architectural bandwidth efficiency)

  4. FPGA's $/token advantage over 950PR is rooted in architectural bandwidth efficiency:
     Effective bandwidth utilization ~38% (streaming weight-resident) vs GPU 2-3% (SIMT warp scheduling)
     This gap is determined by the compute paradigm, not by process, frequency, or pricing.
     Even if 950PR physical bandwidth doubled, B=1 effective utilization would remain 2-3% → gap maintained.

  5. If 950PR later supports native fp4, its $/token could drop to $10-15/M,
     but the structural problem of B=1 effective bandwidth utilization (SIMT batch processing model) would not change with data type.
```

```
950PR throughput estimates are based on:
  → HBM bandwidth 1.4 TB/s, loading 6.1 GB weights ≈ 4.36 ms
  → fp4→FP8 decompression additional ~0.3-0.5 ms
  → Actual per-token decode latency ~4.7-4.9 ms
  → 8-card parallel (TP=8): ~1,600-1,700 tok/s (theoretical)
  → Deducting MoE All-to-All communication + utilization loss → ~1,200-1,600 tok/s

If 950PR later supports native fp4 inference via firmware (similar to B200),
throughput could further improve to ~2,000-2,500 tok/s; this requires ongoing monitoring.
```

**11.6.5 Scenario Applicability Matrix**

```
┌────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Scenario            │ Ascend 950PR     │ FPGA (Agilex 7 M)│ Conclusion       │
├────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Public cloud API (high concur)│ ✅ Best    │ ❌ Concur cap 1-2│ 950PR wins      │
│ Domestic private deploy│ △ Queue for card│ ✅ 8-12wk lead  │ Whoever arrives first│
│ Overseas deployment │ ❌ Huawei restricted│ ✅ Std equipment│ FPGA wins        │
│ fp4 native inference│ ❌ Needs decomp   │ ✅ DSP native    │ FPGA wins        │
│ Prefill (large batch)│ ✅ Tensor Core   │ ❌ Not strong    │ 950PR wins       │
│ Agent scenario (B=1)│ △ Tensor Core idle│ ✅ DSP ~50% util│ FPGA wins        │
│ Multi-model fast switch│ Second-level   │ <1s (hot reload) │ Comparable       │
│ Software ecosystem  │ CANN optimizing   │ In-house, no eco-dep│ Different constraints│
└────────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**11.6.6 Why Is Hardware More Expensive Than 950PR? — An Honest Answer**

> This is a must-ask question from investors/customers. It requires a direct response, not avoidance.

```
Hardware selling price comparison (single inference cluster):

  8×H100 SXM:      ~$280K  (≈ ¥2.0M)  ← unobtainable
  8×Ascend 950PR:  ~$110K  (≈ ¥800K)  ← 12-month queue
  FPGA 8-card×4-chip: ~$303K  (≈ ¥2.22M, 100 sets)  ← why are we the most expensive?

Answer: 32 FPGA chips vs 8 Ascend chips. It is not that our chips are expensive; we need 4× the chip count.
```

**Root Cause 1: Per-Chip Capacity Gap**

```
                  Per-chip HBM    Per-chip compute    Layers per chip
                  ─────────       ──────────────      ──────────────
AGM 039-F          32 GB          12,300 DSP           ~2 layers / chip
Ascend 950PR       112 GB         1,000 TFLOPS         ~8 layers / chip
NVIDIA H100        80 GB          1,979 TFLOPS         ~8 layers / chip

→ AGM 039's per-chip capacity is only 1/4 of 950PR
→ Covering 61 layers of DeepSeek V4 Pro requires 32 FPGAs vs 8 950PRs
→ Even if each FPGA is ¥25K (far below 950PR's ¥100K), 32 chips × ¥25K = ¥800K
   vs 8 chips × ¥100K = ¥800K — chip cost breaks even, but adds 24 chips' worth of BOM/PCB/assembly
→ Plus 8 PCB carrier boards (4 chips/board) vs 8 standard GPU cards, PCB cost is higher
```

**Root Cause 2: Hardware Architecture Trade-off**

```
FPGA assembles large compute from small chips:
  ✅ Benefits: Chip-level redundancy (single-chip failure does not affect entire system), flexible scaling, no advanced packaging constraint
  ❌ Costs: More chips → PCB/connector/assembly cost ×4, power distribution more dispersed

950PR uses a large chip:
  ✅ Benefits: Large per-chip capacity, lower hardware cost, simpler system
  ❌ Costs: Depends on advanced packaging (CoWoS), advanced process (SMIC 7nm), concentrated yield risk
```

**Root Cause 3: Can This Price Gap Be Narrowed?**

```
Room on the chip side:
  AGM 039 ¥25K → ¥18K (10K sets):  -28%
  AGM 039 ¥18K → ¥12K (if Intel gives high-volume pricing): -33%
  → Extreme case: 32 chips × ¥12K = ¥384K, BOM can drop to ~¥1.2M

  FPGA achievable floor (10K sets + deep discount):
    Chips: ¥12K × 32 = ¥384K
    Card-level BOM: ¥18K × 8 = ¥144K
    Server: ¥120K
    Assembly + spares: ¥100K
    Full cost: ~¥748K, add 40% margin → ~¥1.05M (≈ $144K)

  vs 950PR @$110K: gap narrows from 2.7× to 1.3×
  vs H100 @$280K:  but price is not the dimension — effective bandwidth/$ is (see §11.A.2)

Conclusion: The price gap is essentially a "small chip vs large chip" architectural choice;
      it cannot be fully erased, but volume production + deep discounts can significantly narrow it.
      The final gap is ~30% rather than 3×.
```

**Honest Conclusion:**

```
Is FPGA hardware more expensive than 950PR? It depends on which price you compare:

  Official MSRP dimension: 950PR ¥50K/card → 8-card ¥400K (~$55K), FPGA appears 5.5× more expensive
  Actual market price:     950PR ¥250K/card (5× premium) → 8-card ¥2.0M (~$275K)
                           FPGA ¥18K/chip × 32 = ¥576K + card-level BOM ≈ ¥1.33M (~$182K)
                           → Actual price difference is only about 10%!

  Volume (10K sets): FPGA ~$144K, 950PR actual price ~$275K → effective bandwidth/$ advantage ~10×

  Key insight: 950PR's "¥50K official price" essentially does not exist in the real market.
           The root cause of the 5× actual transaction premium is the dual constraint of SMIC 7nm + CoWoS capacity.
           FPGA is not subject to these constraints → list price equals actual price.

So why would a customer choose FPGA instead of queuing for 950PR?

  → Actual price difference is only about 10%, but FPGA aggregate throughput is 2.1-2.3×
  → BW/layer 2.63× advantage → $/token superior (FPGA $5.9 vs 950PR $18-28)
  → 12-month queue vs 8-12 week lead time
  → 950PR cannot go overseas vs FPGA global deployment
  → If the customer can wait 12 months + does not need overseas + does not need high throughput → 950PR is an option
  → If the customer needs delivery certainty + overseas + high throughput → FPGA wins

FPGA competes against "unobtainable-at-reasonable-price 950PR" and "embargoed H100."
In a world where 950PR is available at ¥50K off the shelf anytime, that would be a different competitive landscape.
But that world does not exist.

However, **the endgame is not FPGA.**
After FPGA validation passes → 4 FPGA merged into 1 ASIC tape-out → hardware cost drops to ~$70-80K/set (see §13).
At that point ASIC hardware price ~$70-80K, approximately 25-29% of 950PR actual price (~$275K).
Throughput is roughly unchanged (HBM bandwidth slightly lower: 25.6 vs 29.4 TB/s). At the ASIC stage: architectural bandwidth efficiency (already validated) + manufacturing cost collapse — two orders-of-magnitude dimensions simultaneously present.
```

**11.6.7 Comprehensive Assessment**

```
950PR's advantages:
  ✅ Highest brand recognition among domestic GPUs (Huawei ecosystem + CANN)
  ✅ Best domestic choice for public cloud high-concurrency API scenarios (large-batch prefill)
  ✅ Abundant FP8 compute (1,000 TFLOPS) → strong prefill capability
  ✅ Single-chip 112 GB HBM → ample KV Cache capacity for multi-session concurrency

950PR's limitations:
  ❌ No native fp4 (requires decompression, ~15-20% efficiency loss)
  ❌ BW/layer only 175 GB/s → 38% of FPGA → decode throughput structurally constrained
  ❌ Overseas deployment restricted (Huawei = sanctioned entity)
  ❌ Actual market price 5× premium (¥50K→¥250K) → paper cost-effectiveness is not real
  ❌ Supply volume uncertain (SMIC 7nm + CoWoS constraints) → lead time >6 months
  ❌ Per-card power 600W > FPGA 130W (electricity cost 4.6×)

FPGA vs 950PR core differences:

  950PR seeks the optimal solution within "GPU solutions obtainable in China"
    → GPU architecture domestic substitution, constrained by SMIC + CoWoS capacity
    → Official MSRP competitive, actual market price 5× premium

  FPGA seeks the optimal solution within a "fundamentally different compute paradigm"
    → Architecture match: fp4 native + BW/layer 460 GB/s = structurally optimal for decode
    → Small chips × 32 = 2.63× BW/layer advantage → 2.1-2.3× aggregate throughput
    → Actual price = list price (no capacity-constraint premium)
    → Deeper advantages: effective bandwidth utilization, switching latency, KV address resolution —
      three dimensions with 10-1000× order-of-magnitude gaps (detailed in §11.A.2)

The two are different compute paradigms with scenario-based division of labor:
  → Public cloud API (high-concurrency prefill, compute-bound) → GPU/NPU
  → Decode-heavy scenarios (Agent/Chat/long-document, memory-bound) → FPGA (natural architectural match)
  → Overseas deployment → FPGA (only deployable option)
  → Private + compliance + fast lead time → FPGA (8-12 weeks vs >6 months)
  → GPU's prefill advantage and FPGA's decode advantage are two manifestations of the same physical law,
    not one's "defect" — but in the agent era, the rising share of decode → paradigm advantage tilts toward streaming
```

```
Overall verdict:

  Hardware choice for DeepSeek V4 Pro Decode scenario:

  🥇 FPGA Cluster (Agilex 7 M) — best architectural match + actually obtainable
      BW/layer 460 GB/s (2.63× 950PR) → aggregate throughput 2.1-2.3×
      fp4 native + zero decompression + overseas deployable + 8-12 week lead time
      ¥18K/chip (list price = actual price, no capacity premium)
      Limitations: B=1 communication overhead, requires in-house RTL

  🥈 Ascend 950PR — strong prefill, but decode constrained by BW/layer
      BW/layer 175 GB/s (38% of FPGA)
      Official ¥50K/card attractive, but actual market price ¥250K/card (5×)
      Advantages: abundant prefill compute, Huawei ecosystem, high-concurrency public cloud
      Limitations: no native fp4, overseas sales banned, lead time >6 months, actual price weakens cost-effectiveness

  🥉 H100/B200 — strongest performance but unobtainable
      BW/layer 419 GB/s (91% of FPGA)
      Irreplaceable CUDA ecosystem + extreme compute
      Limitations: sanctioned embargo, actually unobtainable → discussion moot
```

---


---
