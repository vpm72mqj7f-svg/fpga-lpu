#!/usr/bin/env python3
"""Run all available simulation suites and collect a concise report."""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'

CASES = [
    ('module_smoke', [sys.executable, 'scripts/run_module_smoke.py']),
    ('functional_suite', [sys.executable, 'scripts/simulation/run_all.py']),
    ('architecture_integration', [sys.executable, '-m', 'scripts.architecture.integration']),
    ('serving_agent_short', [
        sys.executable, 'scripts/run_serving.py',
        '--duration', '30', '--arrival-rate', '4', '--agent',
        '--agent-output-per-turn', '256', '--microbatch',
        '--expert-replication', 'hot', '--pipeline-clone', '2',
    ]),
]


def run_case(name, cmd):
    print(f"\n=== {name} ===", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=600)
    out_path = DOCS / f'sim_{name}.log'
    out_path.write_text(proc.stdout + '\n--- STDERR ---\n' + proc.stderr,
                        encoding='utf-8')
    print(f"exit={proc.returncode}, log={out_path}", flush=True)
    # short tail
    tail = (proc.stdout + proc.stderr).splitlines()[-10:]
    for line in tail:
        safe = line[:160].encode('ascii', errors='replace').decode('ascii')
        print(safe, flush=True)
    return {
        'name': name,
        'cmd': ' '.join(cmd),
        'exit_code': proc.returncode,
        'log': str(out_path.relative_to(ROOT)),
        'tail': tail,
    }


def main():
    results = []
    for name, cmd in CASES:
        results.append(run_case(name, cmd))

    summary = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'passed': sum(1 for r in results if r['exit_code'] == 0),
        'total': len(results),
        'results': results,
    }
    (DOCS / 'simulation_validation_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    md = ['# FPGA LPU Simulation Validation Summary', '']
    md.append(f"Generated: {summary['generated_at']}")
    md.append(f"Passed: {summary['passed']}/{summary['total']}")
    md.append('')
    md.append('| Suite | Exit | Log |')
    md.append('|---|---:|---|')
    for r in results:
        md.append(f"| {r['name']} | {r['exit_code']} | `{r['log']}` |")
    md.append('')
    md.append('## Notes')
    md.append('- `module_smoke`: direct unit smoke for fpga_arch/vllm_serve modules including WLC.')
    md.append('- `functional_suite`: NumPy fp4/HBM/layer experiments.')
    md.append('- `architecture_integration`: legacy layered architecture demo.')
    md.append('- `serving_agent_short`: end-to-end serving simulation with D+C+A+Pipeline Cloning.')
    (DOCS / 'simulation_validation_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

    print('\n=== SUMMARY ===')
    print(f"Passed: {summary['passed']}/{summary['total']}")
    print('Report: docs/simulation_validation_summary.md')
    if summary['passed'] != summary['total']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
