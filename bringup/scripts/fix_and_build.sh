#!/bin/bash
# Fix duplicate altera_attribute in Intel IP files, then build v2_lite_hbm
cd ~/bringup/v2_lite_hbm

# Fix duplicate attributes
echo "Fixing altera_attribute..."
find ip/ed_synth/ -name "*.sv" -exec sed -i 's/(\* altera_attribute.*//' {} \; 2>/dev/null
find ip/ed_synth/ -name "*.v" -exec sed -i 's/(\* altera_attribute.*//' {} \; 2>/dev/null

export LM_LICENSE_FILE=/home/ic-server31/license_31_171.dat
Q=/opt/intelFPGA_pro/26.1/quartus/bin

echo "=== Synth ==="
rm -rf db incremental_db
$Q/quartus_syn v2_lite_hbm 2>&1 | tail -3

echo "=== Fit ==="
$Q/quartus_fit v2_lite_hbm 2>&1 | tail -3

echo "=== ASM ==="
$Q/quartus_asm v2_lite_hbm 2>&1 | grep -E "Critical|generated|Successful|Error \("

echo "=== SOF ==="
find . -name "*.sof" -ls 2>/dev/null
