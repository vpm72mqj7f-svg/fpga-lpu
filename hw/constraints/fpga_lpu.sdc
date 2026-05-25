# ========================================================================== #
# FPGA LPU — Timing Constraints (SDC)
# Target: Intel Agilex 7 M (AGMF039R47A1E2V, DK-DEV-AGM039EA)
#
# Clock frequencies per ref: Intel doc 782461 (Dev Kit User Guide)
#   CLK_100M_PCIE:   100.00 MHz  (PCIe reference)
#   CLK_156_25MHZ:    156.25 MHz  (F-Tile / QSFP-DD)
#   CLK_245_76MHZ:    245.76 MHz  (QSFP-DD)
#   CLK_312_50MHZ:    312.50 MHz  (Sample clock)
#   CLK_390_625MHZ:   390.625 MHz (DSP target — derived from PLL)
#   DDR5:             5600 Mbps   (board has Micron MTC10F1084S1RC56BG1 x1 DIMM)
#   HBM2e:            920 GB/s    (2048-bit Avalon-MM @ 450 MHz effective)
# ========================================================================== #

# ── Clock Definitions ──────────────────────────────────────────────────────

# PCIe reference (from onboard 100 MHz oscillator via R-Tile)
create_clock -name clk_pcie_ref -period 10.000 [get_ports pcie_refclk_p]

# Board system clock (from Si5332 clock generator, 156.25 MHz default)
# create_clock -name clk_sys_156 -period 6.400 [get_ports clk_156_p]

# DSP core clock: target 450 MHz (period = 2222 ps)
# Derive from CLK_390_625MHZ via PLL (390.625 × 23/20 ≈ 449.2 MHz)
# create_generated_clock -name clk_dsp -source [get_ports clk_390_p] \
#     -multiply_by 23 -divide_by 20 [get_pins pll_dsp|outclk_0]

# HBM reference clock (from UIB dedicated pins)
# create_clock -name clk_hbm_ref -period 2.222 [get_ports hbm_refclk_p]

# ── Clock Groups ───────────────────────────────────────────────────────────

# set_clock_groups -asynchronous \
#     -group [get_clocks clk_dsp] \
#     -group [get_clocks clk_pcie_ref] \
#     -group [get_clocks clk_hbm_ref]

# ── False Paths ────────────────────────────────────────────────────────────

# Reset is asynchronous
# set_false_path -from [get_ports cpu_reset_n]

# JTAG / Signal Tap debug paths
# set_false_path -to [get_pins sld_signaltap:*]

# ── DSP Multicycle Paths ───────────────────────────────────────────────────

# fp4_mac: 3-stage pipeline, accumulator toggles on valid only
# set_multicycle_path -setup 2 \
#     -to [get_registers *u_mac*accumulator*]
# set_multicycle_path -hold  1 \
#     -to [get_registers *u_mac*accumulator*]

# ── Clock Uncertainty ──────────────────────────────────────────────────────

# set_clock_uncertainty -setup 0.050 [get_clocks clk_dsp]
# set_clock_uncertainty -hold  0.020 [get_clocks clk_dsp]

# derive_pll_clocks -create_base_clocks
# derive_clock_uncertainty
