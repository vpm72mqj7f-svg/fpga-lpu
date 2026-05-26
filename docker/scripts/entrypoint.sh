#!/bin/bash
# ============================================================================
# entrypoint.sh — Quartus Build Container Entry Point
# ============================================================================
set -euo pipefail

QUARTUS_BIN="${QUARTUS_HOME}/bin"

# Verify Quartus is available
if [ ! -f "${QUARTUS_BIN}/quartus_sh" ]; then
    echo "============================================================"
    echo " ERROR: Quartus Prime Pro not found at ${QUARTUS_HOME}"
    echo "============================================================"
    echo ""
    echo " Mount Quartus installation:"
    echo "   -v /host/path/intelFPGA_pro:/opt/intelFPGA_pro"
    echo ""
    echo " Or set QUARTUS_HOME to an alternative path."
    echo "============================================================"
    exit 1
fi

QUARTUS_VERSION=$("${QUARTUS_BIN}/quartus_sh" --version 2>/dev/null | head -1 || echo "unknown")
echo "Quartus: ${QUARTUS_VERSION}"
echo "License: ${LM_LICENSE_FILE}"
echo "Threads: $(nproc)"
echo "Memory:  $(free -h | awk '/^Mem:/{print $2}')"
echo "============================================================"

# Route to sub-command
CMD="${1:-build-all}"
shift || true

case "$CMD" in
    build-all)    build_all.sh "$@";;
    build)        build_project.sh "$@";;
    shell)        exec /bin/bash;;
    check-timing) check_timing.sh "$@";;
    *)            echo "Usage: $0 {build-all|build <project>|check-timing|shell}"; exit 1;;
esac
