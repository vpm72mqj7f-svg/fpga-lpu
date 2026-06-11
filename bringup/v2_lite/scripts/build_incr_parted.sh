#!/bin/bash
# =============================================================================
# V2-Lite Full — Incremental compilation (FFN-only rebuild)
# Usage: bash build_incr_parted.sh
#
# Requires: partitions/{u_hbm,u_pcie,u_isp}.qdb from a previous full build.
#
# Imports preserved partitions (HBM2, PCIe, ISP) and only re-synthesizes
# the FFN partition. Reduces build time from ~58 min to ~10-15 min.
# =============================================================================
set -e
PROJ="v2_lite_full"
BDIR="/home/ic-server31/bringup/${PROJ}"
QDIR="/opt/intelFPGA_pro/26.1/quartus/bin"

cd "$BDIR"

echo "=== [1/4] Verify QDB files exist ==="
MISSING=""
for part in u_hbm u_pcie u_isp; do
    if [ -f "partitions/${part}.qdb" ]; then
        echo "  OK: partitions/${part}.qdb"
    else
        echo "  MISSING: partitions/${part}.qdb"
        MISSING="$MISSING $part"
    fi
done
if [ -n "$MISSING" ]; then
    echo "ERROR: Run build_full_parted.sh first to generate QDB files"
    exit 1
fi

echo "=== [2/4] Import preserved partitions ==="
for part in u_hbm u_pcie u_isp; do
    $QDIR/quartus_cdb $PROJ \
        --import_block "$part" \
        --file partitions/${part}.qdb
    echo "  $part: imported"
done

echo "=== [3/4] Incremental compilation ==="
if ! $QDIR/quartus_sh --flow compile $PROJ 2>&1 | tee incr_build.log; then
    echo "Incremental build FAILED"
    echo "If partition mismatch, run build_full_parted.sh to regenerate QDBs"
    grep "^Error" incr_build.log | head -20
    exit 1
fi

echo "=== [4/4] Result ==="
grep -c "Full Compilation was successful" incr_build.log
grep -i "DSP block" output_files/${PROJ}.map.rpt 2>/dev/null | head -2
echo ""
echo "=== INCREMENTAL BUILD COMPLETE ==="
