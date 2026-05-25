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
DUT_SRC   := $(DSP_DIR)/fp4_mac.sv
TB_SRC    := $(SIM_DIR)/tb_fp4_mac.sv

# ---- Test configuration ----
TOP       := tb_fp4_mac

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
