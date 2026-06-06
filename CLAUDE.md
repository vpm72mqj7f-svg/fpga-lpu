# FPGA LPU — Shared Project Context

32-chip Agilex 7 M-Series FPGA inference cluster for DeepSeek V4 Pro LLM serving.
Team: 4 RTL / 3 Verification / 3 Software engineers.

## Project Structure

```
rtl/           — SystemVerilog modules (dsp, activation, attention, moe, layer, chip, engram, head)
rtl/sim/       — Testbenches, golden models, Makefile (Questa/Icarus/Verilator)
scripts/       — Python simulation + experiments
  fpga_arch/   — Hardware architecture model (config, pipeline, cluster, chip)
  vllm_serve/  — vLLM serving stack (scheduler, KV cache, model runner)
  simulation/  — Numerical experiments (fp4 precision, HBM BW, layer latency)
  prefill/     — CPU/FPGA hybrid prefill coordinator
c_ref/         — C runtime (fp4 ref, CPU prefill, weight preloader)
docs/eng/      — Engineering documentation (6 guides + onboarding)
```

## Key Files

- `scripts/fpga_arch/config.py` — Hardware constants (single source of truth)
- `rtl/include/lpu_config.svh` — RTL config (bring-up sim vs production)
- `docs/eng/02_architecture_overview.md` — System architecture
- `docs/eng/03_rtl_developer_guide.md` — RTL conventions
- `docs/eng/04_verification_guide.md` — Test methodology

## Coordination Protocol

Each team member has a role file at `.claude/roles/{name}.md`.
The shared task board is `TASKS.md` at the repo root.
When you complete a task, update TASKS.md with status and any blocking dependencies.
Read other roles' task updates to understand dependencies impacting your work.

## Communication Channels (simulated)

- `TASKS.md` — Status updates (everyone reads/writes)
- Role files — Individual work instructions (read-only, your session)
- Git commits — Code changes with role attribution
- PR reviews — Cross-role coordination (CODEOWNERS enforces)

## Phase 1 Goal (Current)

Single-card verification: all 36 RTL modules passing testbenches,
Python simulation matching golden models, fp4 precision validated.
