#!/bin/bash
# Run V4-Flash Questa simulation on Quartus server (via ARM)
ssh ic-server31@172.18.10.31 bash << 'INNER'
Q=/opt/intelFPGA_pro/26.1/questa_fse/linux_x86_64
S=/home/ic-server31/LR-170370_License.dat
export SALT_LICENSE_SERVER=$S

# Copy clean modules from v2_lite
cp ~/bringup/v2_lite/rtl/fp8_mac.sv ~/bringup/v4_flash/rtl/
cp ~/bringup/v2_lite/rtl/silu_activation.sv ~/bringup/v4_flash/rtl/
cp ~/bringup/v2_lite/rtl/systolic_array.sv ~/bringup/v4_flash/rtl/
cp ~/bringup/v2_lite/rtl/hbm2_weight_reader.sv ~/bringup/v4_flash/rtl/

# Fix weight_ready multi-driver in v4_flash ffn_engine
cd ~/bringup/v4_flash/rtl
sed -i '206s/.*weight_ready.*/        .weight_ready    (),/' v4_flash_ffn_engine.sv 2>/dev/null
sed -i '250s/.*weight_ready.*/        .weight_ready    (),/' v4_flash_ffn_engine.sv 2>/dev/null

# Compile
cd ~/bringup/v4_flash/sim
rm -rf work
$Q/vlib work
for f in ../../rtl/pll_controller.sv ../../rtl/reset_controller.sv ../../rtl/led_controller.sv \
         ../rtl/fp8_mac.sv ../rtl/silu_activation.sv \
         ../rtl/systolic_array.sv ../rtl/hbm2_weight_reader.sv \
         ../rtl/v4_flash_ffn_engine.sv ../rtl/v4_flash_top.sv \
         tb_v4_flash_top.sv; do
  err=$($Q/vlog -sv $f 2>&1 | grep "Error" | grep -v Warning | head -1)
  [ -n "$err" ] && echo "  $f: $err"
done

echo "=== V4-Flash Sim ==="
$Q/vsim -c tb_v4_flash_top -do "run -all; quit" 2>&1 | tail -20
INNER
