# FPGA LPU -- Tech Lead Onboarding Guide

**Role:** Technical Lead, 10-person FPGA LPU project
**Team:** 4 RTL + 3 Verification + 3 Software
**Goal:** 32-chip Intel Agilex 7 M-Series inference cluster for DeepSeek V4 Pro
**Your first job:** Set up the project infrastructure and get the team moving.

---

## Pre-Team: Infrastructure Setup (Before Anyone Joins)

### 1. GitHub Configuration Checklist

- [ ] **Create 4 GitHub teams** (Settings > Access > Teams):
  - `@fpga-lpu-leads` -- tech lead, RTL lead, SW lead (Admin access)
  - `@fpga-lpu-rtl` -- 4 RTL engineers (Write access)
  - `@fpga-lpu-verif` -- 3 verification engineers (Write access)
  - `@fpga-lpu-sw` -- 3 software engineers (Write access)

- [ ] **Add members** to each team. Every member gets read access to all repos (HW/SW codesign requires cross-domain visibility). Merge gate enforcement is via CODEOWNERS, not read restrictions.

- [ ] **Enable branch protection on `master`** (Settings > Rules > Rulesets):
  - Require a pull request before merging (1 approval minimum)
  - Require review from Code Owners
  - Require status checks to pass: `Icarus RTL Simulation`, `Python Simulation Tests`
  - Require branches to be up to date before merging
  - Block force pushes
  - Restrict deletions
  - Bypass list: `@fpga-lpu-leads` (emergency hotfixes only)

- [ ] **Create GitHub Project board** named `FPGA LPU Tracker` (Projects > New Project, type: "Team"):
  - Custom fields: Team (RTL/Verif/SW/All), Phase (1-5), Priority (P0-Critical through P3-Low), Effort PM (0-12), Module (RTL module names)
  - Create views: Kanban (All), My Tasks, Critical Path, Phase 1, By Team
  - Full field definitions in `docs/eng/06_github_setup_guide.md` section 3

- [ ] **Create milestones** in GitHub Issues (see `docs/eng/06_github_setup_guide.md` section 4):
  - Phase 1: Single-Card (Month 2 end)
  - Gate 1: Go/No-Go (Month 2 + 1 week)
  - Phase 2: 8-Card Node (Month 4 end)
  - Gate 2: Go/No-Go (Month 4 + 1 week)
  - Phase 3: Dual-Node (Month 6 end)
  - Phase 4: Full Cluster (Month 8 end)
  - Gate 3: Final Go/No-Go (Month 8 + 1 week)
  - Phase 5: Production (Month 10 end)

- [ ] **Create labels** (full list in `docs/eng/06_github_setup_guide.md` section 3):
  - Domain labels: `rtl`, `rtl-dsp`, `rtl-attention`, `rtl-moe`, `rtl-layer`, `rtl-chip`, `rtl-interfaces`, `verif`, `verif-unit`, `verif-integration`, `verif-onboard`, `sw`, `sw-driver`, `sw-scheduler`, `sw-api`, `sw-compiler`
  - Meta labels: `docs`, `ci`, `infra`, `bug`, `enhancement`, `question`, `blocked`, `good-first-issue`
  - Phase labels: `phase-1` through `phase-5`

- [ ] **Set up Git LFS** for bitstreams and Quartus output files:
  ```bash
  git lfs install
  git lfs track "*.sof" "*.pof" "*.qar" "*.rbf"
  git lfs track "hw/quartus/**/output_files/**"
  git add .gitattributes
  git commit -m "Configure Git LFS for Quartus output files"
  ```

### 2. Verify CI Passes

Push a test commit (e.g., a whitespace change in a tracked RTL file) to a branch and open a draft PR. Confirm both workflows run and pass:

- **`Icarus RTL Simulation`** (`.github/workflows/icarus-sim.yml`): Runs 12 testbenches -- unit (fp4 MAC, GEMM, systolic 2D, RMSNorm, SiLU), module (MLA QKV, router top-K, expert FFN), and integration (layer compute engine, full transformer layer, chip 12-layer). Target: under 5 minutes.
- **`Python Simulation Tests`** (`.github/workflows/python-tests.yml`): Runs module smoke tests, functional experiments, and a 10-second serving simulation smoke. Target: under 3 minutes.

If either fails on `master`, fix before the team starts. The branch protection rules you set in step 1 will block all PR merges until these pass.

### 3. Set Up Self-Hosted Runner for Quartus Nightly Builds

Deploy on AWS `c6i.16xlarge` (64 vCPU, 128 GB RAM, ~$2K/month) or equivalent:

```bash
# On the runner instance
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64.tar.gz \
  https://github.com/actions/runner/releases/download/v2.322.0/actions-runner-linux-x64-2.322.0.tar.gz
tar xzf actions-runner-linux-x64.tar.gz
./config.sh --url https://github.com/<org>/fpga-lpu --token <token> --labels quartus,self-hosted
./run.sh
```

Verify registration: Settings > Actions > Runners should show the new runner as "Idle". Tag it with `quartus` and `self-hosted` labels. Quartus compilation workflows (to be added) will target `runs-on: [self-hosted, quartus]`.

### 4. Seed the Issue Tracker with Initial Tasks

Create one issue per module, per Phase 1. Use the module ownership matrix from `docs/eng/01_project_plan.md` section 4.

Minimum initial set:

- [ ] RTL-1: `fp4_mac.sv` (2 PM, P1.W3), `fp4_systolic_cell.sv` (1 PM, P1.W4), `fp4_systolic_2d.sv` (2 PM, P1.W5)
- [ ] RTL-2: `mla_qkv_proj.sv` (3 PM, P1.W5), `mla_rope.sv` (2 PM, P1.W5), `mla_kv_cache.sv` (2 PM, P1.W6)
- [ ] RTL-3: `router_topk.sv` (1.5 PM, P1.W5), `rms_norm.sv` (0.5 PM, P1.W4), `silu_q12_lut.sv` (0.5 PM, P1.W4)
- [ ] RTL-4: `top_bringup.sv`, HBM example design (P1.W2), `uart_debug.sv` (0.5 PM, P1.W2), PCIe DMA wrapper (1 PM, P1.W4)
- [ ] VRF-1: Golden model freeze (`gen_layer_golden.py`), `tb_mla_attention_v2.sv`, `tb_full_transformer_layer.sv`
- [ ] VRF-2: 15 MAC golden tests (`gen_tb_vectors.py`), `tb_fp4_mac.sv`, `tb_fp4_systolic_2d.sv`
- [ ] VRF-3: `tb_router_topk.sv`, `tb_expert_ffn_engine_fp4_down.sv`, HBM trace generator
- [ ] SW-1: `config.py` finalize, `scheduler.py` skeleton, `pipeline.py` calibration
- [ ] SW-2: `fp4_ref.c`, PCIe DMA driver library, weight layout compiler skeleton
- [ ] SW-3: `api_server.py` skeleton, `cpu_prefill.c` stub, performance benchmarking tools

Each issue: assign to primary owner, add Phase 1 milestone, add phase-1 label, set priority (P0 for critical path: systolic array, MLA pipeline), set estimated PM.

---

## Week 1: Team Onboarding

### 1. Send Each Team Member (Day 0, before kickoff)

Send an email or Slack DM with:

- **Repo URL:** `https://github.com/<org>/fpga-lpu`
- **Their role file:** `.claude/roles/<role>-engineer.md` -- they should copy this to `CLAUDE.md` in their working directory. See `.claude/roles/README.md` for instructions.
- **Their onboarding guide:** Customize per role (create these if they do not exist yet):
  - RTL: `docs/eng/onboarding/rtl-onboarding.md`
  - Verification: `docs/eng/onboarding/verif-onboarding.md`
  - Software: `docs/eng/onboarding/sw-onboarding.md`

### 2. Day 1: All-Hands Kickoff (30 minutes)

Agenda:

1. **Project vision** (5 min): 32-chip FPGA inference cluster. 14,000 tok/s decode at high concurrency, ~660 tok/s at batch-1. Target: DeepSeek V4 Pro on Intel Agilex 7 M-Series. Why FPGA: GPU decode is memory-bound (6 MACs per byte from HBM), FPGA HBM2e matches bandwidth per dollar without stranded compute silicon.

2. **Team structure** (5 min): 4 RTL (RTL-1 leads systolic array and GEMM, RTL-2 leads MLA attention and KV cache, RTL-3 leads MoE router and expert engines, RTL-4 leads chip2chip, PCIe, and integration). 3 Verification (VRF-1 leads test architecture and golden models, VRF-2 leads DSP/MAC testbenches and precision, VRF-3 leads full-layer/cluster testbenches). 3 Software (SW-1 leads vLLM scheduler and API, SW-2 leads FPGA driver and DMA runtime, SW-3 leads CPU prefill and performance analysis). Cross-training matrix in `docs/eng/01_project_plan.md` section 2.3.

3. **Who owns what** (5 min): Review the CODEOWNERS file (`.github/CODEOWNERS`). RTL team owns `rtl/` and `hw/`. Verification team owns `rtl/sim/` and `scripts/simulation/`. Software team owns `scripts/fpga_arch/`, `scripts/vllm_serve/`, and `c_ref/`. Leads own `.github/`, `docs/eng/01_project_plan.md`, `docs/eng/02_architecture_overview.md`, `Makefile`. Everyone has read access to everything.

4. **Communication channels and meeting cadence** (10 min): Review the ongoing meeting cadence below. Point to `#fpga-lpu-eng` for daily discussion, `#fpga-lpu-urgent` for blockers. GitHub Issues for design decisions. PRs require one CODEOWNER approval and passing CI.

5. **Q&A** (5 min).

### 3. Day 1-2: Individual Onboarding

Each member follows their role-specific onboarding guide independently. Your job during these two days:
- Be available on Slack for questions.
- Monitor that everyone has cloned the repo and can run the CI checks locally.
- Verify the CLAUDE.md role files are working (each member's AI assistant should context-load correctly).

### 4. Day 3: First Daily Standup (15 minutes)

Goal: verify everyone's environment works end-to-end.

Checklist per role:
- **RTL engineers:** Can run `make SIM=iverilog run` in `rtl/sim/` and see PASS. Can open Quartus project `hw/quartus/fpga_lpu.qpf` without license errors.
- **Verification engineers:** Can run `python run_module_smoke.py` in `scripts/` and see all tests pass. Golden model generates correct vectors for fp4 MAC.
- **Software engineers:** Can run `python run_serving.py --duration 10 --arrival-rate 2` and see meaningful TPS numbers. C reference compiles and passes unit tests.

### 5. Day 5: First Weekly Review (30 minutes, Friday)

Goal: verify everyone has run their first sim/test on real project code.

Checklist:
- [ ] Every RTL engineer has simulated at least one module testbench.
- [ ] Every verification engineer has run the module smoke suite and viewed results.
- [ ] Every software engineer has run the end-to-end serving simulation.
- [ ] CI is green on `master`.
- [ ] GitHub Project board has all initial tasks in "Ready" or "In Progress".

---

## Ongoing: Meeting Cadence

All meetings target the 4-hour time zone overlap window. No recurring meetings on Wednesday (deep work day).

```
Daily (15 min):       Standup -- what I did, what I'm doing, blockers
                       Async-first: Slack thread update is acceptable.
                       Sync standup only if there's an active blocker.

Tuesday (30 min):     RTL Sync -- interface changes, timing issues,
                       code review backlog, Quartus compilation status.
                       Attendees: RTL-1..4, VRF-1.

Wednesday (30 min):   Verification Sync -- test failures, coverage gaps,
                       golden model updates, CI health.
                       Attendees: VRF-1..3, RTL-1 (optional).
                       NOTE: This is the exception to no-meeting Wednesday
                       if the team prefers to keep deep work on other days.

Thursday (30 min):    Software Sync -- scheduler changes, API design,
                       performance regressions, driver progress.
                       Attendees: SW-1..3.

Friday (30 min):      All-Hands -- demos, cross-team issues, milestone
                       progress, risk review. Record for async viewing.
                       Attendees: All 10.
```

**Additional meetings:**
- **Architecture Review:** Ad-hoc, before major RTL commits. 60 min. RTL-1..4.
- **Phase Gate Review:** End of each phase. 90 min. All 10 + stakeholders.
- **1-on-1:** Biweekly, 30 min with each engineer. Career development, blockers, well-being.

**Communication channels:**
- `#fpga-lpu-eng` (Slack): Daily engineering discussion. Expected response: under 4 hours during working hours.
- `#fpga-lpu-urgent` (Slack): Blockers. Use `@channel`. Expected response: under 1 hour.
- GitHub Issues: Design decisions (tag `design-decision`), bug reports (tag `bug`). Expected response: under 24 hours with written proposal.
- GitHub Pull Requests: Code review. Expected response: under 24 hours (approve or request changes).

---

## Ongoing: Weekly Lead Checklist

Run through this every Friday before the all-hands sync. Takes about 20 minutes.

- [ ] **Review GitHub Project board.** Filter "Critical Path" view first. Are any P0 items stuck in the same column for over 3 days? If yes, find the owner and unblock.
- [ ] **Check CI status.** Open `https://github.com/<org>/fpga-lpu/actions`. Any red workflows on `master`? If yes, file an issue and assign. CI failures on `master` block all PRs.
- [ ] **Review open PRs over 2 days old.** Why is each one blocked? Waiting for review? Waiting for CI? Merge conflict? Tag the reviewer or escalate.
- [ ] **Check critical path items.** These are the two longest-lead-time chains:
  - **Systolic Array** (RTL-1, RTL-2 backup): `fp4_mac` -> `fp4_systolic_cell` -> `fp4_systolic_2d` -> `fp4_gemm_engine` -> `full_transformer_layer`. Total: 10 PM.
  - **MLA Pipeline** (RTL-2, RTL-1 backup): `mla_qkv_proj` -> `mla_rope` -> `mla_kv_cache` -> `mla_attention_v2`. Total: 12 PM.
  Any delay on these directly pushes the final delivery date. If either is behind schedule by over 1 week, escalate to stakeholders immediately.
- [ ] **Update risk register.** Review the 14 risks in `docs/eng/01_project_plan.md` section 6.2. Any new risks? Any probability or impact changed? Update the document and flag in the all-hands.
- [ ] **Check milestone progress.** Open GitHub Milestones. Is the current phase on track? If not, what is the projected slip?
- [ ] **Send weekly status to stakeholders.** Use the template below. Send by Friday EOD.

---

## Go/No-Go Gate Preparation

There are three formal Go/No-Go gates. Criteria are in `docs/eng/01_project_plan.md` section 10. Each gate has sub-gates with GO / WARN / STOP verdicts.

### Gate Timeline

| Gate | When | Phase End | Key Decision |
|------|------|-----------|-------------|
| Gate 1 | Month 2, Week 8 | Phase 1: Single-Card | fp4 precision validated on silicon? HBM bandwidth acceptable? |
| Gate 2 | Month 4, Week 18 | Phase 2: 8-Card Node | Multi-chip C2C/MoE/TP works? Throughput at B=8 acceptable? |
| Gate 3 | Month 8, Week 38 | Phase 4: Full Cluster | Full 61-layer correct? Stability under 24-hour soak? |

### Preparation Schedule (for each gate)

1. **Two weeks before gate date:**
   - Pre-check each sub-gate criterion against current data.
   - Identify any criterion at risk of WARN or STOP.
   - Assign owners to close gaps. Escalate to stakeholders if a STOP verdict is possible.

2. **One week before gate date:**
   - Dry-run the gate review with RTL/SW leads.
   - Compile data package: test results, throughput numbers, timing reports, open issues.
   - Draft the formal recommendation (GO / WARN with remediation plan / STOP with pivot options).

3. **Gate day:**
   - Formal 90-minute review with all stakeholders.
   - Present data for each sub-gate criterion.
   - Make the formal verdict.

4. **Output:**
   - GO: Proceed to next phase. Trigger budget gates (chip orders, infrastructure purchases).
   - WARN: Proceed with adjusted performance targets. Document the remediation plan with owner and deadline.
   - STOP: Halt. Re-evaluate architecture. Present pivot options with cost/schedule impact.

### Gate 1 Specifics (First One You Will Run)

Three sub-gates:
- **GG1-A (fp4 MAC Precision):** 15/15 golden tests pass on hardware, cosine similarity >= 0.995 vs PyTorch.
- **GG1-B (HBM Bandwidth):** MoE random access BW >= 550 GB/s, dual-buffer overlap >= 80%.
- **GG1-C (Layer E2E):** C0 golden case exact match, C1 within +/- 4 ULP, latency <= 2x Icarus simulation.

Budget gate: Phase 2 requires approximately 3M RMB for 8x AGM 039-F chips and PCB prototype. Only proceed if all three sub-gates are GO or WARN with acceptable performance impact.

---

## Key Documents to Know

| Document | What It Contains | Why You Need It |
|----------|-----------------|-----------------|
| `docs/eng/01_project_plan.md` | Staffing, phases, module ownership, risk register, gate criteria | Primary reference. Read it cover-to-cover before Day 1. |
| `docs/eng/02_architecture_overview.md` | System design, data flow, hardware platform, RTL hierarchy | Onboarding reading for every team member. |
| `docs/eng/06_github_setup_guide.md` | GitHub configuration (teams, rulesets, project board, Git LFS) | YOUR job to execute. Covers everything in the Pre-Team checklist above. |
| `.github/CODEOWNERS` | Merge gate enforcement per directory | Determines who must approve every PR. You are the fallback owner (`* @fpga-lpu-leads`). |
| `scripts/fpga_arch/config.py` | Hardware truth: DSP count, HBM specs, model dimensions, pipeline performance, cost economics | Single source of truth for all software models. Approve ALL changes here. |
| `rtl/include/lpu_config.svh` | RTL truth: production vs bring-up parameters, clock frequencies, array sizing | Single source of truth for all RTL. Approve ALL changes here. |
| `.github/workflows/icarus-sim.yml` | CI gate: 12 testbenches on every RTL PR | Must stay green. Blocked PRs kill momentum. |
| `.github/workflows/python-tests.yml` | CI gate: module smoke, functional experiments, serving smoke | Must stay green. Blocked PRs kill momentum. |
| `.claude/roles/README.md` | AI role file instructions | Send to each team member with their role file. |
| `docs/eng/onboarding/` | Role-specific onboarding guides | Create these for each role (rtl, verif, sw) before the team starts. |

---

## Status Report Template

Send this by Friday EOD to stakeholders (email or shared doc). Keep it under one page.

```
Week [N] -- FPGA LPU Status

Completed:
- [list with PR/issue links]

In Progress:
- [list with % complete and expected finish]

Blockers:
- [list with owner and ETA for resolution]

Risks:
- [any new or escalated risks from the risk register]

Next Week:
- [top 3-5 planned items]

Milestone: Phase [X] -- [Y]% complete -- [on track / at risk / behind]

Key metrics:
- CI: [green / red, list failures]
- Open PRs: [count, oldest age]
- Critical path: [on track / behind by N days]
- Test pass rate: RTL [N/12], Python smoke [pass/fail]
```

---

## Quick Reference: Phase Cadence

```
Month 1-2:  Phase 1 (Single-Card)    -- fp4 MAC, HBM BW, single-layer E2E
Month 2 end: Gate 1 Go/No-Go         -- fp4 precision + HBM + layer validation
Month 3-4:  Phase 2 (8-Card Node)    -- C2C ring, MoE dispatch, 15-layer throughput
Month 4 end: Gate 2 Go/No-Go         -- multi-chip validation
Month 5-6:  Phase 3 (Dual-Node)      -- cross-node RDMA, 30-layer pipeline
Month 7-8:  Phase 4 (Full Cluster)   -- 61-layer, 128K context, MTP
Month 8 end: Gate 3 Go/No-Go         -- final performance + stability
Month 9-10: Phase 5 (Production)     -- 1M context, failover, API certification
Month 10 end: Project completion
```
