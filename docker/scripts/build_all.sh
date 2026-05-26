#!/bin/bash
# ============================================================================
# build_all.sh — Build all 8 FPGA LPU projects
#
# Projects built in order:
#   1. bringup     — Go/No-Go validation sequencer
#   2. hbm_char    — HBM2e bandwidth characterization
#   3. dsp_char    — DSP array accuracy + timing
#   4. pcie_test   — PCIe 5.0 DMA throughput
#   5. c2c_test    — C2C ring link test
#   6. full_stack  — Full pipeline integration
#   7. master      — Production master bitstream
#   8. slave       — Production slave bitstream
#
# Output: hw/quartus/<project>/output_files/*.sof
# ============================================================================
set -euo pipefail

PROJECT_ROOT="/workspace"
QUARTUS_DIR="${PROJECT_ROOT}/hw/quartus"
RESULTS_DIR="${PROJECT_ROOT}/build_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Project list in build order
PROJECTS=(
    "bringup:fpga_lpu_bringup"
    "hbm_char:fpga_lpu_hbm_char"
    "dsp_char:fpga_lpu_dsp_char"
    "pcie_test:fpga_lpu_pcie_test"
    "c2c_test:fpga_lpu_c2c_test"
    "full_stack:fpga_lpu_full_stack"
    "master:fpga_lpu_master"
    "slave:fpga_lpu_slave"
)

mkdir -p "${RESULTS_DIR}/${TIMESTAMP}"

echo "========================================================================"
echo " FPGA LPU — Build All Projects"
echo "========================================================================"
echo " Timestamp:  ${TIMESTAMP}"
echo " Projects:   ${#PROJECTS[@]}"
echo " Results:    ${RESULTS_DIR}/${TIMESTAMP}"
echo " Threads:    $(nproc)"
echo "========================================================================"
echo ""

TOTAL_START=$(date +%s)
PASS_COUNT=0
FAIL_COUNT=0
declare -A BUILD_TIMES

for entry in "${PROJECTS[@]}"; do
    PROJ_NAME="${entry%%:*}"
    PROJ_QPF="${entry##*:}"

    echo ""
    echo "--------------------------------------------------------------------"
    echo " BUILD: ${PROJ_NAME} (${PROJ_QPF})"
    echo "--------------------------------------------------------------------"

    PROJ_START=$(date +%s)

    if build_project.sh "${PROJ_NAME}" "${PROJ_QPF}"; then
        PROJ_END=$(date +%s)
        PROJ_TIME=$((PROJ_END - PROJ_START))
        BUILD_TIMES[${PROJ_NAME}]=${PROJ_TIME}
        PASS_COUNT=$((PASS_COUNT + 1))

        # Copy results
        PROJ_OUT="${QUARTUS_DIR}/${PROJ_NAME}/output_files"
        if [ -d "${PROJ_OUT}" ]; then
            cp -r "${PROJ_OUT}" "${RESULTS_DIR}/${TIMESTAMP}/${PROJ_NAME}/"
        fi

        echo "  => PASS (${PROJ_TIME}s)"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "  => FAIL"
    fi
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

# ---------------------------------------------------------------------------
# Build Report
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo " BUILD SUMMARY"
echo "========================================================================"
echo " Total time:  ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s)"
echo " Passed:      ${PASS_COUNT}/${#PROJECTS[@]}"
echo " Failed:      ${FAIL_COUNT}/${#PROJECTS[@]}"
echo ""

for entry in "${PROJECTS[@]}"; do
    PROJ_NAME="${entry%%:*}"
    PROJ_TIME="${BUILD_TIMES[${PROJ_NAME}]:-N/A}"
    PROJ_SOF="${RESULTS_DIR}/${TIMESTAMP}/${PROJ_NAME}/$(echo ${entry##*:} | sed 's/fpga_lpu_//').sof"
    if [ -f "${PROJ_SOF}" ]; then
        SOF_SIZE=$(ls -lh "${PROJ_SOF}" | awk '{print $5}')
        echo "  [PASS] ${PROJ_NAME}: ${PROJ_TIME}s, SOF=${SOF_SIZE}"
    else
        echo "  [FAIL] ${PROJ_NAME}"
    fi
done

echo ""
echo " Results: ${RESULTS_DIR}/${TIMESTAMP}/"
echo "========================================================================"

# Generate JSON report
cat > "${RESULTS_DIR}/${TIMESTAMP}/build_report.json" << JSONEOF
{
  "timestamp": "${TIMESTAMP}",
  "total_time_s": ${TOTAL_TIME},
  "passed": ${PASS_COUNT},
  "failed": ${FAIL_COUNT},
  "projects": {}
}
JSONEOF

exit $((FAIL_COUNT > 0 ? 1 : 0))
