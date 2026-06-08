#!/bin/bash
# Build .sof files for all three projects on Quartus server
ssh ic-server31@172.18.10.31 bash << 'INNER'
export LM_LICENSE_FILE=/home/ic-server31/license_31_171.dat
Q=/opt/intelFPGA_pro/26.1/quartus/bin

for proj in bringup v2_lite v4_flash; do
  echo ""
  echo "============================================"
  echo " Building: $proj"
  echo "============================================"
  cd ~/bringup/$proj

  echo "--- Synthesis ---"
  $Q/quartus_syn $proj 2>&1 | grep -E "Error|Successful|Warning.*entity" | tail -3

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "SYNTH FAILED for $proj — skipping fitter"
    continue
  fi

  echo "--- Fitter ---"
  $Q/quartus_fit $proj 2>&1 | grep -E "Error|Successful|utilization" | tail -3

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "FITTER FAILED for $proj — skipping assembler"
    continue
  fi

  echo "--- Assembler ---"
  $Q/quartus_asm $proj 2>&1 | grep -E "Error|Successful" | tail -3

  if [ -f ${proj}.sof ]; then
    ls -lh ${proj}.sof
  else
    find . -name "*.sof" -ls 2>/dev/null
  fi
done

echo ""
echo "============================================"
echo " DONE"
echo "============================================"
INNER
