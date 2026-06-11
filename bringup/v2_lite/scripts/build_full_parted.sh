#!/bin/bash
# =============================================================================
# V2-Lite Full — Partition-aware full compilation + QDB export
# Usage: bash build_full_parted.sh
#
# Runs a complete compilation with partition flow enabled, then exports
# the stable partitions (HBM2, PCIe, ISP) as QDB files for future
# incremental builds.
#
# QDB export allows subsequent builds to skip re-synthesis of HBM2/PCIe
# partitions — only the FFN partition is rebuilt.
# =============================================================================
set -e
PROJ="v2_lite_full"
BDIR="/home/ic-server31/bringup/${PROJ}"
QDIR="/opt/intelFPGA_pro/26.1/quartus/bin"

cd "$BDIR"

echo "=== [1/4] Clean build artifacts ==="
rm -rf db incremental_db qdb .qsys_edit 2>/dev/null
pkill -9 -f quartus 2>/dev/null || true
sleep 2

echo "=== [2/4] Full compilation (syn + fit + asm) ==="
$QDIR/quartus_sh --flow compile $PROJ 2>&1 | tee full_build.log
if ! grep -q "Full Compilation was successful" full_build.log; then
    echo "BUILD FAILED — see full_build.log"
    grep "^Error" full_build.log | head -20
    exit 1
fi
echo "Compilation PASSED"

echo "=== [3/4] Export partition databases (QDB) ==="
mkdir -p partitions
$QDIR/quartus_cdb --update_mif $PROJ

for part in u_hbm u_pcie u_isp; do
    echo "  Exporting $part ..."
    $QDIR/quartus_cdb $PROJ \
        --export_block "$part" \
        --snapshot final \
        --file partitions/${part}.qdb
    echo "    → partitions/${part}.qdb"
done

echo "=== [4/4] DSP + Resource summary ==="
grep -i "DSP block\|Total comb\|Total registers\|Total block\|Total DSP" \
    output_files/${PROJ}.map.rpt 2>/dev/null | head -10
grep "Slack" output_files/${PROJ}.sta.rpt 2>/dev/null | head -5

echo ""
echo "=== FULL BUILD COMPLETE ==="
echo "Next incremental build: bash build_incr_parted.sh"
