#!/usr/bin/env python3
"""
Unified Regression Script — FPGA LPU Simulation Validation Suite.

Runs ALL tests in one command:
  1. All 10 module smoke tests (imported from run_module_smoke.py)
  2. All 3 functional experiments (imported from simulation/run_all.py)
  3. A quick serving simulation (run_serving.py --duration 10 --arrival-rate 2)

Reports pass/fail with wall-clock timing for each section.
Exit code 0 if all pass, 1 if any fail.

Usage:
  cd D:/workspace/fpgalpu
  python scripts/run_regression.py                    # full suite
  python scripts/run_regression.py --skip-serving      # skip serving sim
  python scripts/run_regression.py --skip-smoke         # skip smoke tests
  python scripts/run_regression.py --skip-experiments   # skip experiments
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


# ============================================================================
# Result tracking
# ============================================================================

class RegressionResult:
    """Holds results for one section of the regression suite."""

    def __init__(self, name: str):
        self.name = name
        self.status = 'PENDING'
        self.wall_seconds = 0.0
        self.details: list = []
        self.error: str | None = None
        self.traceback: str | None = None

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'status': self.status,
            'wall_seconds': round(self.wall_seconds, 2),
            'details': self.details,
            'error': self.error,
        }


def timed_section(name: str, func, *args, **kwargs) -> RegressionResult:
    """Run a section with wall-clock timing and exception capture."""
    result = RegressionResult(name)
    t0 = time.perf_counter()
    try:
        ret = func(*args, **kwargs)
        result.wall_seconds = time.perf_counter() - t0
        if isinstance(ret, dict) and 'status' in ret:
            result.status = ret['status']
            if 'details' in ret:
                result.details = ret['details']
        elif isinstance(ret, bool):
            result.status = 'PASS' if ret else 'FAIL'
        else:
            result.status = 'PASS'
            if ret is not None:
                result.details.append(str(ret))
    except Exception as exc:
        result.wall_seconds = time.perf_counter() - t0
        result.status = 'FAIL'
        result.error = str(exc)
        result.traceback = traceback.format_exc(limit=6)
    return result


# ============================================================================
# Section 1: Module Smoke Tests (10 tests)
# ============================================================================

def run_smoke_tests() -> dict:
    """Run all 10 module smoke tests via direct import."""
    from run_module_smoke import TESTS, run_case

    results = []
    n_pass = 0
    for name, fn in TESTS:
        r = run_case(name, fn)
        results.append(r)
        if r['status'] == 'PASS':
            n_pass += 1

    # Build a compact detail line for each test
    detail_lines = []
    for r in results:
        val = r.get('value', r.get('error', ''))
        if isinstance(val, dict):
            # pick a representative key
            keys = list(val.keys())
            sample_key = keys[0] if keys else ''
            sample_val = val[sample_key] if sample_key else ''
            val_str = f"{sample_key}={sample_val}, ..." if len(keys) > 1 else str(val)
        else:
            val_str = str(val)
        # truncate long values
        if len(val_str) > 60:
            val_str = val_str[:57] + '...'
        detail_lines.append(f"  {r['name']}: {r['status']} | {val_str}")

    overall = n_pass == len(results)
    return {
        'status': 'PASS' if overall else 'FAIL',
        'details': [
            f"Passed: {n_pass}/{len(results)}",
            *detail_lines,
        ],
    }


# ============================================================================
# Section 2: Functional Experiments (3 experiments)
# ============================================================================

def run_functional_experiments() -> dict:
    """Run all 3 functional experiments via direct import."""
    # run_all.main() returns bool and prints tables — we capture return for pass/fail
    from simulation.run_all import run_ffn_experiment, run_experiment_2, run_experiment_3
    from simulation.fp4_utils import fp4_e2m1_info

    details = []

    # Experiment 1: fp4 precision
    t0 = time.perf_counter()
    r1 = run_ffn_experiment(hidden_size=7168, intermediate_size=3072,
                            num_tokens=200, seed=42)
    t1 = time.perf_counter() - t0

    cs = r1['mean_cosine']
    p1 = cs >= 0.995
    details.append(f"  Exp1 fp4 precision: {'PASS' if p1 else 'FAIL'} | "
                   f"cosine={cs:.5f} (target>=0.995) | wall={t1:.1f}s")

    # Experiment 2: HBM bandwidth
    t0 = time.perf_counter()
    r2 = run_experiment_2(num_tokens=2000)
    t2 = time.perf_counter() - t0

    bw = r2['effective_bw_gbps']
    p2 = bw >= 920 * 0.60
    details.append(f"  Exp2 HBM bandwidth: {'PASS' if p2 else 'FAIL'} | "
                   f"bw={bw:.0f} GB/s (target>=552 GB/s) | wall={t2:.1f}s")

    # Experiment 3: layer latency
    t0 = time.perf_counter()
    r3 = run_experiment_3()
    t3 = time.perf_counter() - t0

    lat = r3['weighted_latency_us']
    p3 = lat <= 15.0
    details.append(f"  Exp3 layer latency: {'PASS' if p3 else 'FAIL'} | "
                   f"lat={lat:.1f} us (target<=15 us) | wall={t3:.1f}s")

    overall = p1 and p2 and p3
    return {
        'status': 'PASS' if overall else 'FAIL',
        'details': details,
    }


# ============================================================================
# Section 3: Quick Serving Simulation
# ============================================================================

def run_serving_quick() -> dict:
    """Run a quick serving simulation via subprocess."""
    cmd = [
        sys.executable,
        str(ROOT / 'scripts' / 'run_serving.py'),
        '--duration', '10',
        '--arrival-rate', '2',
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(ROOT), timeout=300)

    details = []
    status = 'PASS' if proc.returncode == 0 else 'FAIL'

    # Extract key metrics from stdout for the summary
    output = proc.stdout
    for line in output.splitlines():
        stripped = line.strip()
        if any(k in stripped for k in [
            'Finished:', 'Accept rate:', 'Output TPS:',
            'TTFT P50:', 'TTFT P95:', 'TPOT P50:', 'TPOT P95:',
        ]):
            details.append(f"  {stripped}")

    if proc.returncode != 0:
        details.append(f"  STDERR: {proc.stderr[:500]}")

    return {
        'status': status,
        'details': details,
    }


# ============================================================================
# Main
# ============================================================================

SECTION_FNS = {
    'smoke': ('Module Smoke Tests (10)', run_smoke_tests),
    'experiments': ('Functional Experiments (3)', run_functional_experiments),
    'serving': ('Quick Serving Simulation (10s)', run_serving_quick),
}


def main():
    parser = argparse.ArgumentParser(
        description='FPGA LPU Unified Regression Suite'
    )
    parser.add_argument('--skip-smoke', action='store_true',
                        help='Skip module smoke tests')
    parser.add_argument('--skip-experiments', action='store_true',
                        help='Skip functional experiments')
    parser.add_argument('--skip-serving', action='store_true',
                        help='Skip quick serving simulation')
    parser.add_argument('--json', type=str, default=None,
                        help='Export results as JSON')
    args = parser.parse_args()

    skip = set()
    if args.skip_smoke:
        skip.add('smoke')
    if args.skip_experiments:
        skip.add('experiments')
    if args.skip_serving:
        skip.add('serving')

    print()
    print("=" * 72)
    print("  FPGA LPU — Unified Regression Suite")
    print("=" * 72)
    print(f"  Root: {ROOT}")
    print(f"  Python: {sys.executable}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    suite_start = time.perf_counter()
    section_results: list[RegressionResult] = []

    for key, (label, fn) in SECTION_FNS.items():
        if key in skip:
            print(f"[SKIP] {label}")
            r = RegressionResult(label)
            r.status = 'SKIPPED'
            section_results.append(r)
            continue

        print(f"[RUN ] {label}", flush=True)
        ret = timed_section(label, fn)
        section_results.append(ret)

        status_tag = f"[{ret.status}]"
        wall_str = f"({ret.wall_seconds:.1f}s)"
        print(f"{status_tag} {label} {wall_str}")
        for d in ret.details:
            print(d)
        if ret.error:
            print(f"  ERROR: {ret.error}")
        print()

    suite_wall = time.perf_counter() - suite_start

    # ── Summary ──
    print()
    print("=" * 72)
    print("  REGRESSION SUMMARY")
    print("=" * 72)
    print()

    n_pass = 0
    n_fail = 0
    n_skip = 0
    total_wall = 0.0

    for r in section_results:
        total_wall += r.wall_seconds
        if r.status == 'PASS':
            n_pass += 1
        elif r.status == 'FAIL':
            n_fail += 1
        elif r.status == 'SKIPPED':
            n_skip += 1

    # Table
    print(f"  {'Section':<40s} {'Status':>8s}  {'Wall Time':>10s}")
    print(f"  {'-'*40} {'-'*8}  {'-'*10}")
    for r in section_results:
        s = r.status if r.status in ('PASS', 'FAIL', 'SKIPPED') else r.status[:8]
        print(f"  {r.name:<40s} {s:>8s}  {r.wall_seconds:>9.1f}s")

    print(f"  {'-'*40} {'-'*8}  {'-'*10}")
    print(f"  {'TOTAL':<40s} {'':>8s}  {total_wall:>9.1f}s")
    print()

    total_active = n_pass + n_fail
    print(f"  Sections: {n_pass} pass, {n_fail} fail, {n_skip} skipped")
    print(f"  Suite wall-clock: {suite_wall:.1f}s")
    print()

    if n_fail > 0:
        print("  OVERALL: [FAIL] — one or more sections failed.")
        print("=" * 72)
        print()
        if args.json:
            _export_json(args.json, section_results, suite_wall)
        raise SystemExit(1)
    else:
        print("  OVERALL: [PASS] — all sections passed.")
        print("=" * 72)
        print()
        if args.json:
            _export_json(args.json, section_results, suite_wall)


def _export_json(filepath: str, section_results: list, suite_wall_seconds: float):
    """Export regression results as JSON."""
    data = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'suite_wall_seconds': round(suite_wall_seconds, 2),
        'sections': [r.to_dict() for r in section_results],
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Results exported to {filepath}")


if __name__ == '__main__':
    main()
