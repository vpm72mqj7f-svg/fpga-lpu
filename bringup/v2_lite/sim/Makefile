# V2-Lite FFN Simulation Makefile
# Icarus Verilog + Questa fallback

TOP      = tb_v2_lite_top
IVL      = iverilog
VVP      = vvp
IVLFLAGS = -g2012 -Wall

SRC = ../../rtl/pll_controller.sv \
      ../../rtl/reset_controller.sv \
      ../../rtl/led_controller.sv \
      ../rtl/v2_lite_ffn_engine.sv \
      ../rtl/v2_lite_top.sv \
      tb_v2_lite_top.sv

all: sim

sim: $(TOP).vvp
	$(VVP) -n $(TOP).vvp

$(TOP).vvp: $(SRC)
	$(IVL) $(IVLFLAGS) -o $@ $(SRC)

wave:
	gtkwave tb_v2_lite_top.vcd &

clean:
	rm -rf $(TOP).vvp *.vcd
