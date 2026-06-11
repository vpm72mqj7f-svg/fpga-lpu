#!/bin/bash
# ============================================================================
# V2-Lite Full — Synthesis script for ic31
# Usage: bash synth_v2_full.sh
# ============================================================================
set -e
PROJ="v2_lite_full"
BDIR="/home/ic-server31/bringup/${PROJ}"
QDIR="/opt/intelFPGA_pro/26.1/quartus/bin"

cd "$BDIR"

echo "=== [1/5] Clean previous build artifacts ==="
rm -rf db incremental_db qdb output_files/*.map.* 2>/dev/null
pkill -9 -f quartus 2>/dev/null || true
sleep 1

echo "=== [2/5] Verify QSF source files ==="
grep "TOP_LEVEL_ENTITY" ${PROJ}.qsf
grep "SYSTEMVERILOG_FILE\|VERILOG_FILE" ${PROJ}.qsf | grep -v "^#"

echo "=== [3/5] Verify source files exist ==="
for f in \
  v2_lite_full_top.sv \
  v2_lite_isp_debug.v \
  v2_lite_ffn_engine.sv \
  systolic_array.sv \
  hbm2_weight_reader.sv \
  silu_activation.sv \
  random_start.v
do
  if [ -f "$f" ]; then echo "  OK: $f"; else echo "  MISSING: $f"; fi
done

echo "=== [4/5] Run Synthesis (quartus_syn) ==="
$QDIR/quartus_syn --64bit $PROJ -c $PROJ 2>&1 | tee ${PROJ}_syn.log
if ! grep -q "successful" ${PROJ}_syn.log; then
    echo "SYNTHESIS FAILED — see ${PROJ}_syn.log"
    grep "Error" ${PROJ}_syn.log | head -20
    exit 1
fi
echo "Synthesis PASSED"

echo "=== [5/5] Check DSP count ==="
grep -i "DSP block" output_files/${PROJ}.map.rpt 2>/dev/null || true
grep -A5 "Resource Usage" output_files/${PROJ}.map.summary 2>/dev/null || true

echo ""
echo "=== SYNTHESIS COMPLETE ==="
echo "Next: quartus_fit ${PROJ} -c ${PROJ}"
echo "      quartus_asm ${PROJ} -c ${PROJ}"
