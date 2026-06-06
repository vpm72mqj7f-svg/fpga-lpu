# FPGA LPU — AI Role Configuration

10 individual role files, one per team member.
Each session runs in its own terminal with role-specific context.

## Quick Start: 10 Terminal Tabs

Open 10 terminals, all in `D:\workspace\fpgalpu`. In each, the first prompt is:

```
Read .claude/roles/<role-file>.md and follow its instructions.
Start working on the first pending task assigned to you in TASKS.md.
```

| Tab | Role | File | Focus Area |
|-----|------|------|------------|
| 1 | RTL-ENG1 | `rtl-eng1.md` | DSP datapath (fp4 MAC, systolic, GEMM) |
| 2 | RTL-ENG2 | `rtl-eng2.md` | MLA / Attention pipeline |
| 3 | RTL-ENG3 | `rtl-eng3.md` | Layer / MoE / Chip integration |
| 4 | RTL-ENG4 | `rtl-eng4.md` | Activation / Head / Engram + testbench gaps |
| 5 | VERIF-ENG1 | `verif-eng1.md` | DSP + Activation golden model verification |
| 6 | VERIF-ENG2 | `verif-eng2.md` | MLA + MoE verification |
| 7 | VERIF-ENG3 | `verif-eng3.md` | Layer / Chip / Cluster integration tests |
| 8 | SW-ENG1 | `sw-eng1.md` | FPGA architecture model & pipeline simulation |
| 9 | SW-ENG2 | `sw-eng2.md` | Serving stack & scheduler |
| 10 | SW-ENG3 | `sw-eng3.md` | Simulation experiments & validation |

## Alternative: Subdirectory CLAUDE.md

For RTL engineers working primarily in one directory, copy the role file:

```bash
cp .claude/roles/rtl-eng1.md rtl/dsp/CLAUDE.md    # Auto-loads when working in rtl/dsp/
cp .claude/roles/sw-eng1.md scripts/CLAUDE.md       # Auto-loads when working in scripts/
```

## Coordination

- **Task board**: `TASKS.md` at repo root — everyone reads/writes
- **Shared context**: `CLAUDE.md` at repo root — loaded by all sessions
- **Commits**: Prefix with role tag, e.g. `[RTL-ENG1] fix fp4_mac overflow`
- **PR reviews**: CODEOWNERS enforces review hierarchy
- **Legacy role files**: `rtl-engineer.md`, `verification-engineer.md`, `software-engineer.md`, `tech-lead.md` are the old 4-template versions — use the individual files instead
