#=============================================================================
# rtl/sim/Makefile — Simulation for fp4_mac and downstream RTL modules
# Target tools: Questa Intel Starter / ModelSim, optional Verilator
#=============================================================================

# ---- Tool selection ----
# Set SIM=questa (default), SIM=verilator, or SIM=iverilog
SIM ?= questa

# ---- Paths ----
RTL_ROOT  := $(abspath ..)
INC_DIR   := $(RTL_ROOT)/include
DSP_DIR   := $(RTL_ROOT)/dsp
SIM_DIR   := $(RTL_ROOT)/sim

# ---- Source files ----
SV_INC    := $(INC_DIR)/fp4_types.svh

# ---- Test configuration ----
# Override TOP, DUT_SRC, TB_SRC on command line for different modules.
# Default: fp4_mac
TOP       ?= tb_fp4_mac
DUT_SRC   ?= $(DSP_DIR)/fp4_mac.sv
TB_SRC    ?= $(SIM_DIR)/tb_fp4_mac.sv

#=============================================================================
# Questa / ModelSim
#=============================================================================
ifeq ($(SIM),questa)

QUESTA ?= vsim
VLOG   ?= vlog

# Intel Questa Starter Edition may be questa-2024.x or vsim
QUESTA_EXE := $(shell which $(QUESTA) 2>/dev/null || which questa 2>/dev/null || which vsim 2>/dev/null)
VLOG_EXE   := $(shell which $(VLOG) 2>/dev/null || which vlog 2>/dev/null)

work:
	$(shell which vlib 2>/dev/null || echo vlib) work

compile: work
	$(VLOG_EXE) +define+SIM +incdir+$(INC_DIR) $(DUT_SRC)
	$(VLOG_EXE) +define+SIM +incdir+$(INC_DIR) $(TB_SRC)

run: compile
	$(QUESTA_EXE) -c -do "run -all; quit" $(TOP)

gui: compile
	$(QUESTA_EXE) -do "run -all" $(TOP) &

clean:
	rm -rf work transcript *.wlf *.log

#=============================================================================
# Verilator (lint + simulation)
#=============================================================================
else ifeq ($(SIM),verilator)

VERILATOR ?= verilator

# Verilator simulation flow
# Requires creating a C++ harness; for now use lint-only mode
lint:
	$(VERILATOR) --lint-only -Wall +incdir+$(INC_DIR) $(DUT_SRC)

compile: lint
	@echo "Verilator full sim requires C++ harness — lint-only for now"

run: compile

clean:
	rm -rf obj_dir

#=============================================================================
# Icarus Verilog (open-source, limited SV support)
#=============================================================================
else ifeq ($(SIM),iverilog)

IVERILOG  ?= iverilog
VVP       ?= vvp

compile:
	$(IVERILOG) -g2012 -I$(INC_DIR) -o $(TOP).vvp $(DUT_SRC) $(TB_SRC)

run: compile
	$(VVP) $(TOP).vvp

clean:
	rm -f $(TOP).vvp

else
$(error Unknown SIM setting: $(SIM). Use SIM=questa, SIM=verilator, or SIM=iverilog)
endif

#=============================================================================
# Common
#=============================================================================
.PHONY: work compile run gui lint clean

#=============================================================================
# Convenience targets: make tb_fp4_mac [SIM=iverilog]
# Maps testbench name → DUT source file automatically.
#=============================================================================
TB_NAME := $(patsubst tb_%,%,$(MAKECMDGOALS))

# Common module → DUT path mapping (add as needed)
tb_fp4_mac:                  DUT = $(DSP_DIR)/fp4_mac.sv
tb_fp4_scale_reader:         DUT = $(DSP_DIR)/fp4_scale_reader.sv
tb_fp4_systolic_cell:        DUT = $(DSP_DIR)/fp4_systolic_cell.sv
tb_fp4_systolic_2d:          DUT = $(DSP_DIR)/fp4_systolic_2d.sv
tb_fp4_gemm_engine:          DUT = $(DSP_DIR)/fp4_gemm_engine.sv
tb_fp4_prefill_engine:       DUT = $(DSP_DIR)/fp4_prefill_engine.sv
tb_fp4_systolic_array:       DUT = $(DSP_DIR)/fp4_systolic_array.sv
tb_fp4_linear_engine:        DUT = $(DSP_DIR)/fp4_linear_engine.sv
tb_fp4_systolic_tile:        DUT = $(DSP_DIR)/fp4_systolic_tile.sv
tb_fp4_scaled_tile:          DUT = $(DSP_DIR)/fp4_scaled_tile.sv
tb_cell_mini:                DUT = $(DSP_DIR)/fp4_systolic_cell.sv
tb_2d_mini:                  DUT = $(DSP_DIR)/fp4_systolic_2d.sv
tb_2d_1x4:                   DUT = $(DSP_DIR)/fp4_systolic_2d.sv
tb_2x2:                      DUT = $(DSP_DIR)/fp4_systolic_2d.sv
tb_2d_4x4:                   DUT = $(DSP_DIR)/fp4_systolic_2d.sv
tb_rms_norm:                 DUT = $(RTL_ROOT)/activation/rms_norm.sv
tb_silu_q12_lut:             DUT = $(RTL_ROOT)/activation/silu_q12_lut.sv
tb_q12_to_fp8_e4m3:          DUT = $(RTL_ROOT)/activation/q12_to_fp8_e4m3.sv
tb_mla_qkv:                  DUT = $(RTL_ROOT)/attention/mla_qkv_proj.sv
tb_mla_attention:            DUT = $(RTL_ROOT)/attention/mla_attention.sv
tb_mla_attention_v2:         DUT = $(RTL_ROOT)/attention/mla_attention_v2.sv
tb_mla_kv_cache:             DUT = $(RTL_ROOT)/attention/mla_kv_cache.sv
tb_router_topk:              DUT = $(RTL_ROOT)/moe/router_topk.sv
tb_expert_ffn_engine:        DUT = $(RTL_ROOT)/moe/expert_ffn_engine.sv
tb_expert_ffn_engine_fp4_down:         DUT = $(RTL_ROOT)/moe/expert_ffn_engine_fp4_down.sv
tb_expert_ffn_engine_fp4_down_golden:  DUT = $(RTL_ROOT)/moe/expert_ffn_engine_fp4_down.sv
tb_mhc_mixer:                DUT = $(RTL_ROOT)/layer/mhc_mixer.sv
tb_layer_compute_engine:     DUT = $(RTL_ROOT)/layer/layer_compute_engine.sv
tb_layer_compute_engine_golden: DUT = $(RTL_ROOT)/layer/layer_compute_engine.sv
tb_full_transformer_layer:   DUT = $(RTL_ROOT)/layer/full_transformer_layer.sv
tb_chip_12layer:             DUT = $(RTL_ROOT)/chip/chip_top.sv
tb_cluster_384:              DUT = $(RTL_ROOT)/chip/chip_top.sv
tb_kv_dma:                   DUT = $(RTL_ROOT)/chip/kv_dma_engine.sv
tb_c2c_ring:                 DUT = $(RTL_ROOT)/legacy/c2c_node.sv
tb_lookup_engine:            DUT = $(RTL_ROOT)/engram/lookup_engine.sv
tb_mtp_head:                 DUT = $(RTL_ROOT)/head/mtp_head.sv

tb_%:
	$(MAKE) TOP=$@ DUT_SRC="$(DUT)" TB_SRC="$(SIM_DIR)/$@.sv" run
