# =============================================================================
# sim.do — V2-Lite FFN ModelSim / QuestaSim do-file
#
# Usage: vsim -do sim.do
# =============================================================================

# Create work library
if {[file exists work]} { vdel -all }
vlib work

# Compile support modules from bringup
vlog -sv ../rtl/pll_controller.sv
vlog -sv ../rtl/reset_controller.sv
vlog -sv ../rtl/led_controller.sv

# Compile V2-Lite design
vlog -sv ../rtl/v2_lite_ffn_engine.sv
vlog -sv ../rtl/v2_lite_top.sv

# Compile testbench
vlog -sv tb_v2_lite_top.sv

# Load and run
vsim -voptargs=+acc tb_v2_lite_top

# Add waves
add wave -divider "Clock & Reset"
add wave /tb_v2_lite_top/clk_sys_100m_p
add wave /tb_v2_lite_top/cpu_reset_n

add wave -divider "LEDs"
add wave -hex /tb_v2_lite_top/led

add wave -divider "FFN Engine"
add wave /tb_v2_lite_top/dut/u_ffn/state
add wave /tb_v2_lite_top/dut/u_ffn/busy
add wave /tb_v2_lite_top/dut/u_ffn/done
add wave /tb_v2_lite_top/dut/u_ffn/expert_idx

add wave -divider "Bringup FSM"
add wave /tb_v2_lite_top/dut/b_state

# Run
run -all
