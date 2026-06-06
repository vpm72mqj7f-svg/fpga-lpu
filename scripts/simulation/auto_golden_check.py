#!/usr/bin/env python3
"""
auto_golden_check.py — Automate golden vector regeneration after RTL changes.

Checks git diff for changed RTL files and regenerates the corresponding golden
test vector packages. Reports which files need updating and diffs the changes.

Can be used as:
  - Pre-commit hook:  python scripts/simulation/auto_golden_check.py --check
  - CI check:         python scripts/simulation/auto_golden_check.py --ci
  - Interactive regen: python scripts/simulation/auto_golden_check.py --regen
  - Dry-run:          python scripts/simulation/auto_golden_check.py --dry-run

Mapping:
  rtl/dsp/*.sv (fp4_mac, systolic_cell, scale_reader)
    → regenerates rtl/sim/tb_golden_pkg.sv via gen_tb_vectors.py

  rtl/moe/*.sv (expert_ffn_engine_fp4_down, router_topk)
    → regenerates rtl/sim/tb_ffn_golden_pkg.sv via gen_ffn_tb_vectors.py

  rtl/layer/*.sv (layer_compute_engine, full_transformer_layer)
    → regenerates rtl/sim/tb_layer_golden_pkg.sv via gen_layer_golden.py
"""

import argparse
import difflib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RTL_DIR = ROOT / 'rtl'
SIM_DIR = RTL_DIR / 'sim'
SCRIPT_DIR = ROOT / 'scripts' / 'simulation'

# ── Golden file registry ──
# Each entry: (golden_file_name, generator_script, trigger_glob_patterns)
GOLDEN_REGISTRY = [
    {
        'name': 'tb_golden_pkg.sv',
        'generator': 'gen_tb_vectors.py',
        'description': 'DSP golden vectors (fp4_mac)',
        'triggers': [
            'rtl/dsp/fp4_mac.sv',
            'rtl/dsp/fp4_systolic_cell.sv',
            'rtl/dsp/fp4_scale_reader.sv',
            'rtl/dsp/fp4_systolic_2d.sv',
        ],
        'dir_triggers': ['rtl/dsp'],
    },
    {
        'name': 'tb_ffn_golden_pkg.sv',
        'generator': 'gen_ffn_tb_vectors.py',
        'description': 'FFN golden vectors (expert_ffn_engine_fp4_down)',
        'triggers': [
            'rtl/moe/expert_ffn_engine_fp4_down.sv',
            'rtl/moe/router_topk.sv',
        ],
        'dir_triggers': ['rtl/moe'],
    },
    {
        'name': 'tb_layer_golden_pkg.sv',
        'generator': 'gen_layer_golden.py',
        'description': 'Layer golden vectors (layer_compute_engine)',
        'triggers': [
            'rtl/layer/layer_compute_engine.sv',
            'rtl/layer/full_transformer_layer.sv',
        ],
        'dir_triggers': ['rtl/layer'],
    },
]

# Additional cross-cutting triggers: any include/SVH file change may affect all
COMMON_TRIGGERS = [
    'rtl/include',
]


def get_changed_rtl_files(staged_only: bool = False,
                          base_ref: str = 'HEAD') -> list[str]:
    """Return list of changed RTL files (.sv, .svh) from git diff.

    Excludes files in rtl/sim/ (testbenches, golden files themselves)
    since they don't affect golden vector computation.

    Args:
        staged_only: if True, only check staged changes (pre-commit use).
        base_ref: git ref to diff against (default: HEAD for unstaged,
                  HEAD~1 for pre-push).
    """
    changed = []

    # Staged changes
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR'],
            capture_output=True, text=True, cwd=str(ROOT), timeout=10,
        )
        if result.returncode == 0:
            staged = set(result.stdout.strip().split('\n'))
            if staged_only:
                changed.extend(f for f in staged if _is_rtl_source(f))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Unstaged changes (working tree)
    if not staged_only:
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', '--diff-filter=ACMR'],
                capture_output=True, text=True, cwd=str(ROOT), timeout=10,
            )
            if result.returncode == 0:
                unstaged = set(result.stdout.strip().split('\n'))
                changed.extend(f for f in unstaged if _is_rtl_source(f))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Untracked RTL files
        try:
            result = subprocess.run(
                ['git', 'ls-files', '--others', '--exclude-standard'],
                capture_output=True, text=True, cwd=str(ROOT), timeout=10,
            )
            if result.returncode == 0:
                untracked = set(result.stdout.strip().split('\n'))
                changed.extend(f for f in untracked if _is_rtl_source(f))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Deduplicate and strip empty strings
    return sorted(set(f for f in changed if f))


def _is_rtl_source(filepath: str) -> bool:
    """Check if a filepath is an RTL source file (not a testbench or golden file).

    Excludes:
      - rtl/sim/  (testbenches, golden packages, debug files)
      - Files ending with _golden_pkg.sv (already in sim/, double-check)
    """
    if not filepath.endswith(('.sv', '.svh')):
        return False
    # Exclude sim directory (testbenches and golden files)
    if filepath.startswith('rtl/sim/') or filepath.startswith('rtl\\sim\\'):
        return False
    # Exclude golden package files
    if '_golden_pkg.sv' in filepath:
        return False
    return True


def classify_changes(changed_files: list[str]) -> dict[str, list[str]]:
    """Map changed files to which golden generators they trigger.

    Returns: {golden_name: [triggering_files], ...}
    """
    triggered = {}

    for entry in GOLDEN_REGISTRY:
        name = entry['name']
        triggers = set()

        for f in changed_files:
            # Exact trigger match
            for pat in entry['triggers']:
                if f == pat or f.startswith(pat.replace('.sv', '')):
                    triggers.add(f)
                    break
            # Directory trigger
            for d in entry['dir_triggers']:
                if f.startswith(d + '/') or f == d:
                    triggers.add(f)
                    break
            # Common cross-cutting triggers (SVH includes)
            for ct in COMMON_TRIGGERS:
                if f.startswith(ct + '/'):
                    triggers.add(f)
                    break

        if triggers:
            triggered[name] = sorted(triggers)

    return triggered


def read_golden_file(name: str) -> str | None:
    """Read a golden package file from rtl/sim/. Returns content or None."""
    path = SIM_DIR / name
    if path.exists():
        return path.read_text(encoding='utf-8')
    return None


def regenerate_golden(name: str) -> tuple[bool, str]:
    """Regenerate a golden file by running its generator script.

    Returns: (success, message)
    """
    entry = next((e for e in GOLDEN_REGISTRY if e['name'] == name), None)
    if entry is None:
        return False, f"No registry entry for {name}"

    gen_script = SCRIPT_DIR / entry['generator']
    if not gen_script.exists():
        return False, f"Generator not found: {gen_script}"

    try:
        result = subprocess.run(
            [sys.executable, str(gen_script)],
            capture_output=True, text=True,
            cwd=str(SCRIPT_DIR), timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, f"Generator failed (exit={result.returncode}):\n{result.stderr[:1000]}"
    except subprocess.TimeoutExpired:
        return False, f"Generator timeout: {gen_script}"
    except Exception as e:
        return False, f"Error running {gen_script}: {e}"


def diff_golden(entry: dict, old_content: str, new_content: str) -> str:
    """Produce a unified diff between old and new golden file contents."""
    if old_content == new_content:
        return "(no changes)"
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{entry['name']}",
        tofile=f"b/{entry['name']}",
        n=3,
    )
    return ''.join(diff)


def run_check(changed_files: list[str], regen: bool = False,
              dry_run: bool = False) -> dict:
    """Main check logic.

    Returns: { 'status': 'ok'|'stale'|'error',
               'triggered': {name: [files]},
               'results': {name: {'regen_ok': bool, 'changed': bool, 'diff': str}} }
    """
    triggered = classify_changes(changed_files)

    if not triggered:
        return {
            'status': 'ok',
            'message': 'No golden files affected by current RTL changes.',
            'triggered': {},
            'results': {},
        }

    results = {}
    any_changed = False
    any_error = False

    for name, triggering_files in triggered.items():
        entry = next(e for e in GOLDEN_REGISTRY if e['name'] == name)
        old_content = read_golden_file(name) or ''

        if dry_run:
            results[name] = {
                'regen_ok': True,
                'changed': False,
                'diff': '(dry run — would regenerate)',
                'triggering_files': triggering_files,
            }
            continue

        if regen:
            ok, msg = regenerate_golden(name)
            if not ok:
                results[name] = {
                    'regen_ok': False,
                    'changed': False,
                    'diff': '',
                    'error': msg,
                    'triggering_files': triggering_files,
                }
                any_error = True
                continue

            new_content = read_golden_file(name) or ''
            changed = (old_content != new_content)
            if changed:
                any_changed = True
            diff = diff_golden(entry, old_content, new_content)

            results[name] = {
                'regen_ok': True,
                'changed': changed,
                'diff': diff,
                'triggering_files': triggering_files,
            }
        else:
            # Check-only mode: compare with what would be regenerated
            ok, _ = regenerate_golden(name)
            if not ok:
                results[name] = {
                    'regen_ok': False,
                    'changed': False,
                    'diff': '',
                    'error': 'Failed to run generator for comparison',
                    'triggering_files': triggering_files,
                }
                any_error = True
                continue

            new_content = read_golden_file(name) or ''
            changed = (old_content != new_content)
            if changed:
                any_changed = True
            diff = diff_golden(entry, old_content, new_content)

            # Restore original (generator overwrote the file)
            if old_content:
                (SIM_DIR / name).write_text(old_content, encoding='utf-8')

            results[name] = {
                'regen_ok': True,
                'changed': changed,
                'diff': diff if changed else '',
                'triggering_files': triggering_files,
            }

    if any_error:
        status = 'error'
    elif any_changed:
        status = 'stale'
    else:
        status = 'ok'

    return {
        'status': status,
        'message': '',
        'triggered': triggered,
        'results': results,
    }


def print_report(result: dict, verbose: bool = False):
    """Print a human-readable report."""
    print()
    print("=" * 72)
    print("  Golden Vector Auto-Check Report")
    print("=" * 72)
    print()

    if result['status'] == 'ok':
        print("  Status:  OK — golden vectors are up to date.")
        print(result.get('message', ''))
        print()
        return

    for name, info in result['results'].items():
        entry = next(e for e in GOLDEN_REGISTRY if e['name'] == name)
        files_str = ', '.join(info.get('triggering_files', []))

        if not info.get('regen_ok'):
            tag = '[ERROR]'
            print(f"  {tag} {name} ({entry['description']})")
            print(f"    Triggered by: {files_str}")
            print(f"    Error: {info.get('error', 'unknown')}")
            print()
            continue

        if info.get('changed'):
            tag = '[STALE]'
        else:
            tag = '[OK]   '

        print(f"  {tag} {name} ({entry['description']})")
        print(f"    Triggered by: {files_str}")

        if info.get('changed') and info.get('diff'):
            if verbose:
                print(f"    Diff ({len(info['diff'].splitlines())} lines):")
                for line in info['diff'].splitlines()[:40]:
                    print(f"      {line}")
                if len(info['diff'].splitlines()) > 40:
                    print(f"      ... ({len(info['diff'].splitlines()) - 40} more lines)")
            else:
                diff_lines = [l for l in info['diff'].splitlines()
                             if l.startswith(('+', '-')) and not l.startswith(('+++', '---', '@@'))]
                added = sum(1 for l in diff_lines if l.startswith('+'))
                removed = sum(1 for l in diff_lines if l.startswith('-'))
                print(f"    Changes: +{added} lines, -{removed} lines")
        print()

    # Summary
    stale_count = sum(1 for i in result['results'].values() if i.get('changed'))
    error_count = sum(1 for i in result['results'].values() if not i.get('regen_ok'))

    if result['status'] == 'stale':
        print(f"  [ACTION REQUIRED] {stale_count} golden file(s) need regeneration.")
        print(f"    Run: python scripts/simulation/auto_golden_check.py --regen")
        print()
    elif result['status'] == 'error':
        print(f"  [ERROR] {error_count} golden generator(s) failed.")
        print()


def install_precommit_hook():
    """Install as a git pre-commit hook."""
    hook_path = ROOT / '.git' / 'hooks' / 'pre-commit'
    hook_script = f'''#!/bin/bash
# Auto-generated by auto_golden_check.py
# Checks golden vectors are up-to-date before committing RTL changes.
python {SCRIPT_DIR / 'auto_golden_check.py'} --check
exit $?
'''
    hook_path.write_text(hook_script)
    if os.name != 'nt':
        os.chmod(hook_path, 0o755)
    print(f"Pre-commit hook installed at {hook_path}")
    print("Golden vector check will run before each commit.")


def main():
    parser = argparse.ArgumentParser(
        description='Auto Golden Vector Regeneration Checker'
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', default=True,
                      help='Check if golden files need regeneration (default)')
    mode.add_argument('--regen', action='store_true',
                      help='Regenerate stale golden files')
    mode.add_argument('--dry-run', action='store_true',
                      help='Show what would be regenerated without doing it')
    mode.add_argument('--ci', action='store_true',
                      help='CI mode: fail if golden files are stale')
    mode.add_argument('--install-hook', action='store_true',
                      help='Install as git pre-commit hook')
    parser.add_argument('--staged-only', action='store_true',
                        help='Only check staged changes (for pre-commit hook)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show full diffs')
    parser.add_argument('--all', action='store_true',
                        help='Regenerate ALL golden files regardless of changes')

    args = parser.parse_args()

    if args.install_hook:
        install_precommit_hook()
        return

    # Collect changed RTL files
    if args.all:
        # Regenerate all golden files
        changed = []
        for entry in GOLDEN_REGISTRY:
            for pat in entry['triggers']:
                changed.append(pat)
    else:
        changed = get_changed_rtl_files(staged_only=args.staged_only)
        if not changed:
            print("No RTL files changed. Golden vectors are up to date.")
            return

    print(f"Found {len(changed)} changed RTL file(s):")
    for f in changed[:20]:
        print(f"  {f}")
    if len(changed) > 20:
        print(f"  ... and {len(changed) - 20} more")

    # Dry-run: classify and report only
    if args.dry_run:
        triggered = classify_changes(changed)
        if not triggered:
            print("\nNo golden files affected by these changes.")
            return
        print("\nThe following golden files would be regenerated:")
        for name, files in triggered.items():
            entry = next(e for e in GOLDEN_REGISTRY if e['name'] == name)
            print(f"  {name} ({entry['description']})")
            print(f"    Triggered by: {', '.join(files)}")
        return

    # Run check
    result = run_check(
        changed,
        regen=args.regen,
        dry_run=False,
    )

    print_report(result, verbose=args.verbose)

    if args.ci and result['status'] in ('stale', 'error'):
        raise SystemExit(1)
    elif args.regen and result['status'] == 'ok':
        print("All golden vectors are already up to date.")
    elif not args.regen and result['status'] == 'stale':
        print("Use --regen to regenerate stale files, or --dry-run to preview.")
        if args.ci:
            raise SystemExit(1)
    elif result['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
