# =============================================================================
# sim.do — V4-Flash FFN ModelSim / QuestaSim do-file
#
# Usage: vsim -do sim.do
# =============================================================================

if {[file exists work]} { vdel -all }
vlib work

# Compile support modules
vlog -sv ../rtl/pll_controller.sv
vlog -sv ../rtl/reset_controller.sv
vlog -sv ../rtl/led_controller.sv

# Compile V4-Flash design
vlog -sv ../rtl/v4_flash_ffn_engine.sv
vlog -sv ../rtl/v4_flash_top.sv

# Compile testbench
vlog -sv tb_v4_flash_top.sv

# Load and run
vsim -voptargs=+acc tb_v4_flash_top

# Add waves
add wave -divider "Clock & Reset"
add wave /tb_v4_flash_top/clk_sys_100m_p
add wave /tb_v4_flash_top/cpu_reset_n

add wave -divider "LEDs"
add wave -hex /tb_v4_flash_top/led

add wave -divider "FFN Engine"
add wave /tb_v4_flash_top/dut/u_ffn/state
add wave /tb_v4_flash_top/dut/u_ffn/busy
add wave /tb_v4_flash_top/dut/u_ffn/done
add wave /tb_v4_flash_top/dut/u_ffn/expert_cnt
add wave /tb_v4_flash_top/dut/u_ffn/inter_row

add wave -divider "Bringup FSM"
add wave /tb_v4_flash_top/dut/b_state

run -all
