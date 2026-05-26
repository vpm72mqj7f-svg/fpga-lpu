#!/bin/bash
# ============================================================================
# build_project.sh — Build a single FPGA project with Quartus
#
# Usage: build_project.sh <project_dir> <qpf_name>
#   build_project.sh master fpga_lpu_master
# ============================================================================
set -euo pipefail

PROJ_DIR="${1:?Usage: build_project.sh <project_dir> <qpf_name>}"
PROJ_QPF="${2:?}"
PROJ_ROOT="/workspace"
QUARTUS_BIN="${QUARTUS_HOME}/bin"
PROJ_PATH="${PROJ_ROOT}/hw/quartus/${PROJ_DIR}"

if [ ! -d "${PROJ_PATH}" ]; then
    echo "ERROR: Project directory not found: ${PROJ_PATH}"
    exit 1
fi

cd "${PROJ_PATH}"

QPF_FILE="${PROJ_QPF}.qpf"
if [ ! -f "${QPF_FILE}" ]; then
    echo "ERROR: QPF not found: ${QPF_FILE}"
    exit 1
fi

THREADS=$(nproc)

# ---------------------------------------------------------------------------
# Step 1: Analysis & Synthesis
# ---------------------------------------------------------------------------
echo "  [1/4] Analysis & Synthesis..."
"${QUARTUS_BIN}/quartus_map" \
    "${PROJ_QPF}" \
    --parallel=${THREADS} \
    --64bit \
    --read_settings_files=on \
    --write_settings_files=off \
    2>&1 | tee "${PROJ_QPF}_map.log"

if ! grep -q "Quartus Prime Analysis & Synthesis was successful" "${PROJ_QPF}_map.log"; then
    echo "  ERROR: Synthesis failed"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Fitter (Place & Route)
# ---------------------------------------------------------------------------
echo "  [2/4] Fitter (Place & Route)..."
"${QUARTUS_BIN}/quartus_fit" \
    "${PROJ_QPF}" \
    --parallel=${THREADS} \
    --64bit \
    2>&1 | tee "${PROJ_QPF}_fit.log"

if ! grep -q "Quartus Prime Fitter was successful" "${PROJ_QPF}_fit.log"; then
    echo "  ERROR: Fitter failed"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Timing Analysis
# ---------------------------------------------------------------------------
echo "  [3/4] Timing Analysis..."
"${QUARTUS_BIN}/quartus_sta" \
    "${PROJ_QPF}" \
    --parallel=${THREADS} \
    --64bit \
    2>&1 | tee "${PROJ_QPF}_sta.log"

# Check timing closure
if grep -q "Timing constraints are not met" "${PROJ_QPF}_sta.log"; then
    echo "  WARNING: Timing constraints NOT met!"
    # Generate detailed timing report
    "${QUARTUS_BIN}/quartus_sta" "${PROJ_QPF}" \
        --do_report_timing \
        --report_script=timing_report.tcl \
        2>/dev/null || true
else
    echo "  OK: Timing constraints met."
fi

# ---------------------------------------------------------------------------
# Step 4: Assembler (Generate SOF)
# ---------------------------------------------------------------------------
echo "  [4/4] Assembler (SOF generation)..."
"${QUARTUS_BIN}/quartus_asm" \
    "${PROJ_QPF}" \
    --64bit \
    2>&1 | tee "${PROJ_QPF}_asm.log"

if ! grep -q "Quartus Prime Assembler was successful" "${PROJ_QPF}_asm.log"; then
    echo "  ERROR: Assembler failed"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resource Summary
# ---------------------------------------------------------------------------
echo ""
echo "  --- Resource Utilization ---"
grep -A 20 "Fitter Resource Usage Summary" "${PROJ_QPF}_fit.log" | head -20 || true
echo ""
echo "  --- Clock Fmax ---"
grep -i "fmax\|maximum frequency" "${PROJ_QPF}_sta.log" | head -5 || true

echo ""
echo "  Build complete: ${PROJ_QPF}"
echo "  SOF: output_files/${PROJ_QPF}.sof"
exit 0
