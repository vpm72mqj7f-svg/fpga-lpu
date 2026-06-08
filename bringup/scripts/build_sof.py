#!/usr/bin/env python3
"""Build .sof for v2_lite and v4_flash on Quartus server"""
import subprocess, sys, os

LICENSE = "/home/ic-server31/license_31_171.dat"
QDIR = "/opt/intelFPGA_pro/26.1/quartus/bin"
PROJS = ["v2_lite", "v4_flash"]
BASE = "/home/ic-server31/bringup"

for proj in PROJS:
    print(f"===== {proj} =====")
    proj_dir = f"{BASE}/{proj}"
    env = {"LM_LICENSE_FILE": LICENSE}

    for stage in ["syn", "fit", "asm"]:
        cmd = [f"{QDIR}/quartus_{stage}", proj]
        print(f"  {stage}...", end=" ", flush=True)
        r = subprocess.run(cmd, cwd=proj_dir, env={**os.environ, **env},
                          capture_output=True, text=True, timeout=600)
        # Check result
        if r.returncode != 0:
            for line in r.stdout.split('\n') + r.stderr.split('\n'):
                if 'Error (' in line:
                    print(f"FAIL: {line.strip()}")
            break
        else:
            print("OK")
    else:
        # All stages passed — check for .sof
        sof = f"{proj_dir}/{proj}.sof"
        if os.path.exists(sof):
            size = os.path.getsize(sof)
            print(f"  .sof: {size/1024/1024:.0f} MB -> {sof}")
        else:
            print(f"  .sof not found, searching...")
            for root, dirs, files in os.walk(proj_dir):
                for f in files:
                    if f.endswith('.sof'):
                        print(f"  found: {os.path.join(root, f)}")
    print()
