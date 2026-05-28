# GitHub Repository Configuration Guide

This document covers the manual configuration steps that cannot be automated via files in `.github/`. Execute these once during initial setup.

---

## 1. GitHub Teams (Settings → Access → Teams)

Create these 4 teams and assign members:

| Team | Members | Access |
|------|---------|--------|
| `@fpga-lpu-leads` | Tech lead + RTL lead + SW lead | Admin |
| `@fpga-lpu-rtl` | 4 RTL engineers | Write |
| `@fpga-lpu-verif` | 3 verification engineers | Write |
| `@fpga-lpu-sw` | 3 software engineers | Write |

**Note**: On a free GitHub org plan, all members get Read access to all repos. This is intentional — HW/SW co-design requires cross-domain visibility. Access control is enforced at the **merge gate** (CODEOWNERS), not the read gate.

---

## 2. Branch Protection Rules (Settings → Rules → Rulesets)

Create a ruleset targeting the `master` branch:

```
Name: "master protection"
Target: Include default branch (master)

Rules:
  [x] Require a pull request before merging
      [x] Require approvals: 1
      [x] Dismiss stale pull request approvals when new commits are pushed
  [x] Require review from Code Owners
  [x] Require status checks to pass before merging
      [x] Require branches to be up to date before merging
      Status checks to require:
      - "Icarus RTL Simulation"
      - "Python Simulation Tests"
  [x] Block force pushes
  [x] Restrict deletions

Bypass list: @fpga-lpu-leads (for emergency hotfixes)
```

---

## 3. GitHub Project Board (Projects → New Project)

Create a "Team" type project, name it `FPGA LPU Tracker`.

### Custom Fields

| Field | Type | Values |
|-------|------|--------|
| Team | Single select | `RTL`, `Verif`, `SW`, `All` |
| Phase | Single select | `Phase 1`, `Phase 2`, `Phase 3`, `Phase 4`, `Phase 5` |
| Priority | Single select | `P0-Critical`, `P1-High`, `P2-Medium`, `P3-Low` |
| Effort (PM) | Number | 0-12 |
| Module | Single select | RTL module names from the module list |

### Views

| View | Type | Filter | Purpose |
|------|------|--------|---------|
| **Kanban (All)** | Board | Status grouped | Full task board, daily standup |
| **My Tasks** | Board | `assignee:@me` | Personal view |
| **Critical Path** | Table | `priority:P0-Critical` | Weekly lead review |
| **Phase 1** | Board | `phase:"Phase 1"` | Current phase focus |
| **By Team** | Board | `team:RTL / Verif / SW` tabs | Per-team views |

### Labels

```
rtl, rtl-dsp, rtl-attention, rtl-moe, rtl-layer, rtl-chip, rtl-interfaces
verif, verif-unit, verif-integration, verif-onboard
sw, sw-driver, sw-scheduler, sw-api, sw-compiler
docs, ci, infra
phase-1, phase-2, phase-3, phase-4, phase-5
bug, enhancement, question, blocked
good-first-issue
```

---

## 4. Milestones

| Milestone | Due | Description |
|-----------|-----|-------------|
| Phase 1: Single-Card | Month 2 end | PCIe, HBM, fp4 core, single-layer benchmark |
| Gate 1: Go/No-Go | Month 2 + 1 week | fp4 precision, HBM BW, single-layer E2E |
| Phase 2: 8-Card Node | Month 4 end | RoCE v2, MoE dispatch, 15-layer throughput |
| Gate 2: Go/No-Go | Month 4 + 1 week | C2C ring, MoE correctness, B=8 benchmark |
| Phase 3: Dual-Node | Month 6 end | Cross-node RDMA, 30-layer pipeline |
| Phase 4: Full Cluster | Month 8 end | 61-layer, 128K context, multi-session |
| Gate 3: Final Go/No-Go | Month 8 + 1 week | Full correctness, stability, throughput |
| Phase 5: Production | Month 10 end | 1M context, failover, API certification |

---

## 5. Issue Templates

Recommended issue types (create via Settings → General → Set up templates):

- **Bug Report** — RTL bug, simulation mismatch, software crash
- **Feature Request** — New module, optimization, tooling
- **Task** — Standard work item with phase/label/assignee
- **Design Review** — Architecture decision requiring cross-team discussion

---

## 6. Git LFS Setup (for bitstreams and large binaries)

```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.sof" "*.pof" "*.qar" "*.rbf"
git lfs track "hw/quartus/**/output_files/**"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for Quartus output files"
```

---

## 7. Self-Hosted Runner for Quartus (Nightly Builds)

Deploy on an AWS `c6i.16xlarge` (or similar: 64 vCPU, 128 GB RAM):

```bash
# On the runner instance
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64.tar.gz \
  https://github.com/actions/runner/releases/download/v2.322.0/actions-runner-linux-x64-2.322.0.tar.gz
tar xzf actions-runner-linux-x64.tar.gz
./config.sh --url https://github.com/<org>/fpga-lpu --token <token> --labels quartus,self-hosted
./run.sh
```

The nightly Quartus workflow (to be added later) will target `runs-on: [self-hosted, quartus]`.

---

## 8. Communication Integration (Optional)

| Tool | GitHub Integration | Purpose |
|------|-------------------|---------|
| Slack | `/github subscribe org/repo` | PR/Issue notifications to team channels |
| Discord | GitHub webhook → Discord | Alternative for overseas team |

---

## Summary: Who Can Do What

```
                    Read RTL   Read SW   Merge RTL   Merge SW   Push to master
@fpga-lpu-leads      Yes       Yes       Yes         Yes        Yes (bypass)
@fpga-lpu-rtl        Yes       Yes       Yes         No         No (PR only)
@fpga-lpu-verif      Yes       Yes       Yes*        No         No (PR only)
@fpga-lpu-sw         Yes       Yes       No          Yes        No (PR only)

* rtl/sim/ testbenches only
```

**Rationale for full read access**: RTL engineers need the Python simulation (it IS the executable spec). SW engineers need the RTL interfaces (they write the driver). Verification engineers need everything. Restricting reads in a 10-person HW/SW co-design project causes more integration bugs than it prevents.
