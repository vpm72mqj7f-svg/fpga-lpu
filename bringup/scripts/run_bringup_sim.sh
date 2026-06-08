#!/bin/bash
# Run bringup Questa simulation on Quartus server (via ARM)
ssh ic-server31@172.18.10.31 bash << 'INNER'
Q=/opt/intelFPGA_pro/26.1/questa_fse/linux_x86_64
S=/home/ic-server31/LR-170370_License.dat
export SALT_LICENSE_SERVER=$S
cd ~/bringup/sim
rm -rf work
$Q/vlib work
for f in ../rtl/pll_controller.sv ../rtl/reset_controller.sv ../rtl/led_controller.sv ../rtl/s10_ffn_engine.sv ../rtl/bringup_top.sv tb_bringup_top.sv; do
  $Q/vlog -sv $f 2>&1 | grep Error | grep -v Warning | head -1
done
echo "=== Bringup Sim ==="
$Q/vsim -c tb_bringup_top -do "run -all; quit" 2>&1 | tail -20
INNER
