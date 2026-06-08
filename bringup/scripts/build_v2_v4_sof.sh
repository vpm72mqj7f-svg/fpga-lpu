#!/bin/bash
ssh ic-server31@172.18.10.31 bash << 'INNER'
export LM_LICENSE_FILE=/home/ic-server31/license_31_171.dat
Q=/opt/intelFPGA_pro/26.1/quartus/bin

for proj in v2_lite v4_flash; do
  echo "===== $proj ====="
  cd ~/bringup/$proj
  echo "--- Synth ---"
  $Q/quartus_syn $proj 2>&1 | grep -E "Error|Successfully synthesized" | tail -2
  echo "--- Fit ---"
  $Q/quartus_fit $proj 2>&1 | grep -E "Error|Successfully committed" | tail -2
  echo "--- ASM ---"
  $Q/quartus_asm $proj 2>&1 | grep -E "Error|generated|25207|Successful" | tail -3
  ls -lh ${proj}.sof 2>/dev/null
  find . -name "*.sof" -ls 2>/dev/null | head -3
  echo ""
done
echo "DONE"
INNER
