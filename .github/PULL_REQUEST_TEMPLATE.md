## Summary

<!-- What does this PR do? 1-2 sentences. -->

## Type

- [ ] RTL (new module / bug fix / optimization)
- [ ] Testbench (new test / golden update)
- [ ] Software (Python sim / scheduler / API)
- [ ] Documentation
- [ ] CI / Infrastructure

## Checklist

### RTL Changes
- [ ] Icarus simulation passed (`cd rtl/sim && make tb_<module>`)
- [ ] Golden vector comparison passed (if applicable)
- [ ] Quartus synthesis passed (bring-up config, if DSP/datapath touched)
- [ ] Timing closure checked (if production config changed)

### Software Changes
- [ ] `python scripts/run_module_smoke.py` passed
- [ ] `python scripts/simulation/run_all.py` passed
- [ ] `python scripts/run_serving.py --duration 10` passed
- [ ] No regression in TPS/TTFT metrics (attach before/after if behavioral change)

### Documentation Changes
- [ ] Spelling/grammar checked
- [ ] File paths verified
- [ ] Cross-references updated

### All PRs
- [ ] Branch is up-to-date with `master`
- [ ] No unrelated changes mixed in
- [ ] PR title describes the change (not the ticket number)

## Reviewer Assignment

<!-- CODEOWNERS will auto-assign. Delete if manual override needed. -->

| Directory | Required Reviewer |
|-----------|-------------------|
| `rtl/` | @fpga-lpu-rtl |
| `rtl/sim/` | @fpga-lpu-rtl + @fpga-lpu-verif |
| `scripts/fpga_arch/`, `scripts/vllm_serve/` | @fpga-lpu-sw |
| `scripts/simulation/` | @fpga-lpu-verif |
| `docs/eng/` | Per-document owner (see CODEOWNERS) |
