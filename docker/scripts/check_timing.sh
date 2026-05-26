#!/bin/bash
# ============================================================================
# check_timing.sh — Post-build timing analysis across all projects
# ============================================================================
set -euo pipefail
PROJ_ROOT="/workspace"
RESULTS="${PROJ_ROOT}/build_results"

echo "========================================================================"
echo " Timing Analysis — All Projects"
echo "========================================================================"

for proj_dir in "${PROJ_ROOT}"/hw/quartus/*/; do
    proj_name=$(basename "$proj_dir")
    [ "$proj_name" = "common" ] && continue

    sta_log="${proj_dir}/*_sta.log"
    fit_log="${proj_dir}/*_fit.log"

    echo ""
    echo "--- ${proj_name} ---"

    # Fmax
    if ls ${sta_log} 2>/dev/null; then
        grep -h "Fmax\|Maximum Frequency\|; Slow 1100mV" ${sta_log} 2>/dev/null | head -5 || echo "  No Fmax data"
    fi

    # Resource utilization
    if ls ${fit_log} 2>/dev/null; then
        echo "  Resources:"
        grep -h "Logic utilization\|ALM\|DSP\|M20K\|MLAB" ${fit_log} 2>/dev/null | head -8 || echo "  No resource data"
    fi

    # Critical paths
    if ls ${sta_log} 2>/dev/null; then
        echo "  Worst slack:"
        grep -h "Worst-case slack\|Setup slack\|Hold slack" ${sta_log} 2>/dev/null | head -3 || echo "  No slack data"
    fi
done
