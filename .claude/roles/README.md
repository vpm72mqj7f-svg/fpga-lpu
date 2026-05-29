# AI Role Configuration Guide

Each team member gets an AI assistant pre-configured with their role's context, codebase knowledge, and task assignments.

## Quick Start (Per Engineer)

**Step 1**: Copy your role file to your working directory:
```bash
# RTL engineers (4 people)
copy .claude\roles\rtl-engineer.md CLAUDE.md

# Verification engineers (3 people)
copy .claude\roles\verification-engineer.md CLAUDE.md

# Software engineers (3 people)
copy .claude\roles\software-engineer.md CLAUDE.md

# Tech leads
copy .claude\roles\tech-lead.md CLAUDE.md
```

**Step 2**: Start Claude Code. It auto-loads CLAUDE.md.

**Step 3**: The AI now knows:
- What project this is
- Your specific role and responsibilities
- The full codebase map relevant to you
- Your coding standards and workflows
- Your current phase tasks
- How to run simulations, tests, and builds

## Role Summary

| Role | File | AI Knows |
|------|------|----------|
| **RTL Engineer** | `rtl-engineer.md` | ~50 SystemVerilog modules, Quartus/Icarus flows, coding conventions, DSP timing rules, module specs |
| **Verification Engineer** | `verification-engineer.md` | 24+ testbenches, golden model methodology, Python reference models, accuracy tolerances, Signal Tap, CI |
| **Software Engineer** | `software-engineer.md` | Python simulation stack (fpga_arch + vllm_serve), C runtime, key formulas, scheduler logic, API design |
| **Tech Lead** | `tech-lead.md` | Architecture decisions, cross-team coordination, Go/No-Go gates, risk register, code review, project tracking |

## Directory-Level Configs (Optional)

For deeper integration, place additional CLAUDE.md files in specific directories:

```
rtl/CLAUDE.md            → Auto-loads when working in rtl/
rtl/dsp/CLAUDE.md        → Additional DSP-specific context
scripts/fpga_arch/CLAUDE.md → Python sim context
```

The role file + directory file combine automatically.

## Example: What "RTL Engineer AI" Can Do

```
User: "Write the fp4_systolic_2d module with 128 lanes, 32 rows, Avalon-ST interface"

AI: [Generates SystemVerilog matching:
     - Signal naming conventions (i_*, o_*, r_*, w_*)
     - Parameterized LANES and M_ROWS from lpu_config.svh
     - DSP inference attributes for 450 MHz
     - Valid/ready handshake from avalon_stream.svh
     - Synthesizable 2-always-block FSM]

User: "This module fails timing at 450MHz, critical path is in the accumulator chain"

AI: [Suggests:
     - Pipeline register insertion points
     - ALLOW_RETIMING attribute placement
     - Register balancing recommendations
     - Cites specific examples from existing modules]
```

## Example: What "Verification Engineer AI" Can Do

```
User: "Golden comparison fails for fp4 GEMM at test case 42"

AI: [Traces the computation:
     - Reads golden vector from tb_golden_pkg.sv
     - Reads fp4_utils.py reference for expected output
     - Compares RTL stage-by-stage against verify_fp4_mac_stages.py
     - Identifies: Q12 accumulation overflow at stage 3 when >256 terms
     - Suggests: widen accumulator or add saturation]
```

## Example: What "Software Engineer AI" Can Do

```
User: "TPS drops 30% when I enable pipeline-clone 2"

AI: [Analyzes:
     - Checks throughput_model() in pipeline.py
     - Examines chip0_admission_rate() 
     - Identifies: KV cache fragmentation from double allocation
     - Suggests: adjust KV block allocation strategy for cloned pipelines
     - Generates the code change]
```
