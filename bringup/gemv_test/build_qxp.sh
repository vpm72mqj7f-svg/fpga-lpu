#!/bin/bash
# build_qxp.sh — Standard GEMV QXP build flow
# Usage: bash build_qxp.sh
set -e
cd "$(dirname "$0")"
QDIR=/opt/intelFPGA_pro/26.1/quartus/bin

echo "=== Step 1: Setup partition ==="
$QDIR/quartus_sh -t setup_partition.tcl

echo "=== Step 2: Synthesis ==="
rm -rf db incremental_db output_files dni qdb
$QDIR/quartus_syn --64bit gemv_test -c gemv_test

echo "=== Step 3: Export QXP ==="
$QDIR/quartus_cdb gemv_test -c gemv_test --export_partition top --qxp gemv_dsp.qxp

echo "=== Done: gemv_dsp.qxp ==="
ls -la gemv_dsp.qxp
grep "Total DSP" gemv_test.syn.rpt
