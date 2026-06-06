"""
run_dsp_regression.py — Focused regression script for DSP + activation testbenches.

Compiles and runs all DSP+activation testbenches using Icarus Verilog.
Reports pass/fail per testbench, returns exit code 0 if all pass.

Usage:
    cd rtl/sim
    python run_dsp_regression.py          # use default iverilog
    python run_dsp_regression.py --verbose # show full output
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Testbench definitions: (name, dut_src, tb_src, inc_dir)
# dut_src is relative to RTL_ROOT, tb_src is relative to SIM_DIR
# ---------------------------------------------------------------------------
RTL_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = RTL_ROOT / "sim"
INC_DIR = RTL_ROOT / "include"

# Altera IP behavioral wrappers — required for all testbenches that use
# altera_mult_add, altera_syncram, altera_scfifo instances
SIM_IP = [
    SIM_DIR / "altera_mult_add.sv",
    SIM_DIR / "altera_syncram.sv",
    SIM_DIR / "altera_scfifo.sv",
]

TESTBENCHES = [
    # DSP testbenches (with full dependency chain for iverilog)
    {
        "name": "tb_fp4_mac",
        "srcs": [
            RTL_ROOT / "dsp" / "fp4_mac.sv",
            SIM_DIR / "tb_fp4_mac.sv",
        ],
    },
    {
        "name": "tb_fp4_scale_reader",
        "srcs": [
            RTL_ROOT / "dsp" / "fp4_mac.sv",           # dependency
            RTL_ROOT / "dsp" / "fp4_scale_reader.sv",   # DUT
            SIM_DIR / "tb_fp4_scale_reader.sv",
        ],
    },
    {
        "name": "tb_cell_mini",
        "srcs": [
            RTL_ROOT / "dsp" / "fp4_mac.sv",            # dependency
            RTL_ROOT / "dsp" / "fp4_systolic_cell.sv",  # DUT
            SIM_DIR / "tb_cell_mini.sv",
        ],
    },
    {
        "name": "tb_fp4_systolic_2d",
        "srcs": [
            RTL_ROOT / "dsp" / "fp4_mac.sv",            # leaf
            RTL_ROOT / "dsp" / "fp4_systolic_cell.sv",  # dependency
            RTL_ROOT / "dsp" / "fp4_systolic_2d.sv",    # DUT
            SIM_DIR / "tb_fp4_systolic_2d.sv",
        ],
    },
    {
        "name": "tb_fp4_gemm_engine",
        "srcs": [
            RTL_ROOT / "dsp" / "fp4_mac.sv",            # leaf
            RTL_ROOT / "dsp" / "fp4_systolic_cell.sv",  # leaf
            RTL_ROOT / "dsp" / "fp4_systolic_2d.sv",    # dependency
            RTL_ROOT / "dsp" / "fp4_gemm_engine.sv",    # DUT
            SIM_DIR / "tb_fp4_gemm_engine.sv",
        ],
    },
    # Activation testbenches (standalone)
    {
        "name": "tb_rms_norm",
        "srcs": [
            RTL_ROOT / "activation" / "rms_norm.sv",
            SIM_DIR / "tb_rms_norm.sv",
        ],
    },
    {
        "name": "tb_silu_q12_lut",
        "srcs": [
            RTL_ROOT / "activation" / "silu_q12_lut.sv",
            SIM_DIR / "tb_silu_q12_lut.sv",
        ],
    },
    # Prefill engine (multi-module dependency chain)
    {
        "name": "tb_fp4_prefill_engine",
        "srcs": [
            RTL_ROOT / "dsp" / "fp4_mac.sv",
            RTL_ROOT / "dsp" / "fp4_systolic_cell.sv",
            RTL_ROOT / "dsp" / "fp4_systolic_2d.sv",
            RTL_ROOT / "dsp" / "fp4_prefill_engine.sv",
            SIM_DIR / "tb_fp4_prefill_engine.sv",
        ],
    },
]


def find_iverilog():
    """Find iverilog executable."""
    import shutil
    iverilog = shutil.which("iverilog")
    if iverilog is None:
        print("ERROR: iverilog not found in PATH")
        sys.exit(1)
    return iverilog


def find_vvp():
    """Find vvp executable."""
    import shutil
    vvp = shutil.which("vvp")
    if vvp is None:
        print("ERROR: vvp not found in PATH")
        sys.exit(1)
    return vvp


def run_testbench(tb_def, iverilog_exe, vvp_exe, verbose=False):
    """
    Compile and run a single testbench.
    Returns (name, passed: bool, output: str).
    """
    name = tb_def["name"]
    srcs = tb_def["srcs"]
    vvp_path = SIM_DIR / f"{name}.vvp"

    # Check source files exist
    for p in srcs:
        if not p.exists():
            return name, False, f"  ERROR: Source file not found: {p}"

    # Compile (SIM_IP wrappers included for all testbenches)
    cmd_compile = [
        iverilog_exe, "-g2012",
        f"-I{INC_DIR}",
        "-o", str(vvp_path),
    ] + [str(p) for p in SIM_IP] + [str(p) for p in srcs]

    try:
        result = subprocess.run(
            cmd_compile,
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return name, False, "  TIMEOUT: Compilation exceeded 60s"

    if result.returncode != 0:
        stderr_str = result.stderr.decode('utf-8', errors='replace')
        return name, False, f"  COMPILE ERROR:\n{stderr_str}"

    # Run
    cmd_run = [vvp_exe, str(vvp_path)]
    try:
        result = subprocess.run(
            cmd_run,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return name, False, "  TIMEOUT: Simulation exceeded 120s"

    stdout_str = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
    stderr_str = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
    output = stdout_str + stderr_str
    passed = "PASS" in output and "FAIL" not in output

    if verbose or not passed:
        return name, passed, output
    else:
        # Extract just the summary line
        lines = output.strip().split("\n")
        summary = [l for l in lines if "PASS" in l or "FAIL" in l]
        summary_output = "\n".join(summary) if summary else output[-500:]
        return name, passed, summary_output


def main():
    parser = argparse.ArgumentParser(
        description="DSP + Activation testbench regression"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full output for all testbenches"
    )
    parser.add_argument(
        "--iverilog", type=str, default=None,
        help="Path to iverilog executable"
    )
    parser.add_argument(
        "--vvp", type=str, default=None,
        help="Path to vvp executable"
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only run testbenches matching this substring"
    )
    args = parser.parse_args()

    iverilog_exe = args.iverilog or find_iverilog()
    vvp_exe = args.vvp or find_vvp()

    # Filter testbenches if requested
    tbs = TESTBENCHES
    if args.filter:
        tbs = [t for t in tbs if args.filter in t["name"]]

    print("=" * 70)
    print(" DSP + Activation Regression")
    print(f" iverilog: {iverilog_exe}")
    print(f" vvp:      {vvp_exe}")
    print(f" Tests:    {len(tbs)}")
    print("=" * 70)
    print()

    results = []
    for tb in tbs:
        sys.stdout.write(f"  {tb['name']:30s} ... ")
        sys.stdout.flush()
        name, passed, output = run_testbench(tb, iverilog_exe, vvp_exe, args.verbose)
        results.append((name, passed, output))

        if passed:
            print("PASS")
        else:
            print("FAIL")

        # Clean up vvp file
        vvp_path = SIM_DIR / f"{name}.vvp"
        if vvp_path.exists():
            vvp_path.unlink()

    # Summary
    print()
    print("=" * 70)
    print(" Results Summary")
    print("=" * 70)

    pass_count = sum(1 for _, p, _ in results if p)
    fail_count = len(results) - pass_count

    for name, passed, output in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print()
    print(f"  Total:  {len(results)}")
    print(f"  Passed: {pass_count}")
    print(f"  Failed: {fail_count}")
    print("=" * 70)

    # Print failures with details
    if fail_count > 0:
        print()
        print(" Failure Details:")
        print("-" * 70)
        for name, passed, output in results:
            if not passed:
                print(f"\n--- {name} ---")
                # Show last 40 lines of output
                lines = output.strip().split("\n")
                for line in lines[-40:]:
                    print(f"  {line}")
        print()

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
