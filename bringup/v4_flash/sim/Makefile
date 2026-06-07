# V4-Flash FFN Simulation Makefile
# Icarus Verilog + Questa fallback

TOP      = tb_v4_flash_top
IVL      = iverilog
VVP      = vvp
IVLFLAGS = -g2012 -Wall

SRC = ../../rtl/pll_controller.sv \
      ../../rtl/reset_controller.sv \
      ../../rtl/led_controller.sv \
      ../rtl/v4_flash_ffn_engine.sv \
      ../rtl/v4_flash_top.sv \
      tb_v4_flash_top.sv

all: sim

sim: $(TOP).vvp
	$(VVP) -n $(TOP).vvp

$(TOP).vvp: $(SRC)
	$(IVL) $(IVLFLAGS) -o $@ $(SRC)

wave:
	gtkwave tb_v4_flash_top.vcd &

clean:
	rm -rf $(TOP).vvp *.vcd
