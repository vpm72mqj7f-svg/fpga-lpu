# V2-Lite FFN Engine — QuestaSim Production Simulation
# Target: Questa FSE 26.1 (ic31: /opt/intelFPGA_pro/26.1/questa_fse/)
# Usage: vsim -do sim_ffn_prod.do

# ---- Cleanup ----
quit -sim
onerror { quit -f -code 1 }

# ---- Questa FSE path (ic31) ----
set QDIR /opt/intelFPGA_pro/26.1/questa_fse

# ---- Create Altera LPM library (for altera_mult_add) ----
vlib lpm
vlog +acc -sv -work lpm $QDIR/intel/verilog/src/220model.v

# ---- Compile behavioral model for altera_mult_add (standalone simulation wrapper) ----
vlog +acc -sv -work work altera_mult_add_sim.sv

# ---- Compile Production FFN RTL ----
vlog +acc -sv -work work \
    fp8_mac.sv \
    silu_activation.sv \
    hbm2_weight_reader.sv \
    systolic_array.sv \
    v2_lite_ffn_engine.sv

# ---- Compile testbench (SIM_SMALL for fast sim) ----
vlog +acc -sv -work work +define+SIM_SMALL tb_ffn_engine_sv.sv

# ---- Load and run ----
vsim -L lpm -voptargs=+acc work.tb_ffn_engine_sv

# Enable waveform dump
add wave -r /*
add wave -divider "FFN Debug Ports"
add wave -radix hex sim:/tb_ffn_engine_sv/dut/dbg_*
add wave -radix unsigned sim:/tb_ffn_engine_sv/dut/perf_*
add wave -divider "FSM"
add wave -radix symbolic sim:/tb_ffn_engine_sv/dut/state

# Run
run -all

# Result check
if { [exa -boolean tb_ffn_engine_sv/errors] && [exa tb_ffn_engine_sv/errors == 0] } {
    echo "=== ALL TESTS PASSED ==="
    quit -f -code 0
} else {
    echo "=== TESTS FAILED ==="
    quit -f -code 1
}
