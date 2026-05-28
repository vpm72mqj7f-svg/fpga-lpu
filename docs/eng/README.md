# FPGA LPU — Engineering Documentation

32-chip FPGA inference cluster for DeepSeek V4 Pro LLM serving.
Team: 10 engineers (4 RTL / 3 Verification / 3 Software). All docs in English for overseas team.

## Quick Start

| Role | Read This First | Then |
|------|----------------|------|
| **Everyone** | [02 Architecture Overview](02_architecture_overview.md) | [01 Project Plan](01_project_plan.md) |
| **RTL Engineers (4)** | [03 RTL Developer Guide](03_rtl_developer_guide.md) | `rtl/include/lpu_config.svh`, run `make tb_fp4_mac` in `rtl/sim/` |
| **Verification Engineers (3)** | [04 Verification Guide](04_verification_guide.md) | `scripts/simulation/run_all.py`, `rtl/sim/Makefile` |
| **Software Engineers (3)** | [05 Software & Simulation Guide](05_software_guide.md) | `python scripts/run_serving.py --duration 30` |

## Document Index

| # | Document | Lines | Description |
|---|----------|-------|-------------|
| 01 | [Project Plan](01_project_plan.md) | 972 | 5-phase / 10-month plan, staffing, module ownership, critical path, risk register, AI tool strategy, Go/No-Go gates |
| 02 | [Architecture Overview](02_architecture_overview.md) | 1,158 | System design, 10-stage pipeline, memory hierarchy, interconnect, RTL hierarchy, key design decisions |
| 03 | [RTL Developer Guide](03_rtl_developer_guide.md) | 1,412 | Codebase map, coding conventions, bring-up vs production, interfaces, Icarus/Quartus flows, design patterns |
| 04 | [Verification Guide](04_verification_guide.md) | 1,526 | Golden model methodology, 24+ testbench inventory, unit/integration/system test strategy, on-board validation, CI |
| 05 | [Software & Simulation Guide](05_software_guide.md) | 1,301 | Python sim stack (fpga_arch + vllm_serve), C runtime, APIs/protocols, weight compiler, development workflow |

## Key Reference Files

| What | Path |
|------|------|
| Hardware constants (single source of truth) | `scripts/fpga_arch/config.py` |
| Central RTL config (bring-up vs production) | `rtl/include/lpu_config.svh` |
| Architecture diagram (ASCII) | `scripts/ARCHITECTURE.txt` |
| Full proposal (English) | `docs/fpga_inference_cluster_proposal_en.md` |
| RTL common file list | `hw/quartus/common/common_modules.qsf` |
| Simulation Makefile | `rtl/sim/Makefile` |
| Top-level build | `Makefile` |

## Running the Simulation

```bash
# Quick serving simulation (30 seconds)
python scripts/run_serving.py --duration 30 --arrival-rate 5

# All functional experiments (fp4 precision, HBM bandwidth, layer latency)
python scripts/simulation/run_all.py

# Module smoke tests
python scripts/run_module_smoke.py

# Full validation suite (18 configs x 3 workloads)
python scripts/run_all_validations.py
```

## RTL Quick Commands

```bash
# Run a single testbench (Icarus)
cd rtl/sim && make tb_fp4_mac

# Run all testbenches
cd rtl/sim && make all

# Generate golden test vectors from Python reference
python scripts/simulation/gen_tb_vectors.py
python scripts/simulation/gen_ffn_tb_vectors.py
```

## Project Timeline

```
M1-M2   Phase 1: Single-card verification
M3-M4   Phase 2: Single-node 8-card
M5-M6   Phase 3: Dual-node interconnect
M7-M8   Phase 4: Four-node full cluster (32 chips, 61 layers)
M9-M10  Phase 5: Optimization & production
```

Gate 1 (post-Phase 1): fp4 MAC precision, HBM bandwidth, single-layer E2E
Gate 2 (post-Phase 2): C2C ring, MoE dispatch, B=8 throughput
Gate 3 (post-Phase 4): Full pipeline correctness, 24h stability, throughput target
