# ========================================================================== #
# FPGA LPU — Timing Constraints (SDC)
# Target: Intel Agilex 7 M (AGMF039R47A1E2VR0, DK-DEV-AGM039EA)
#
# Reference: Intel doc 782461 (Agilex 7 M-Series Dev Kit User Guide)
#            https://github.com/altera-fpga/agilex7f-ed-gsrd (GSRD format)
#
# Clock domains:
#   clk_board_100m:    100.00 MHz  (onboard oscillator, sys/control)
#   clk_dsp:           390.625 MHz (DSP systolic array, from PLL)
#   clk_pcie:          250.00 MHz  (PCIe 5.0 R-Tile reference)
#   clk_hbm:           450.00 MHz  (HBM2e UIB, from dedicated HBM refclk)
# ========================================================================== #

# ── Board Clock (100 MHz oscillator) ─────────────────────────────────────
create_clock -name clk_board_100m -period 10.000 [get_ports clk_board_100m]

# ── Reset ────────────────────────────────────────────────────────────────
# Asynchronous reset (false path from reset port)
set_false_path -from [get_ports cpu_reset_n]

# ── Generated Clocks (from PLL) ──────────────────────────────────────────
# [TODO] Uncomment after PLL instantiation in top-level:
#
# DSP clock: 100 MHz × 39/10 = 390 MHz → 100 MHz × 125/32 = 390.625 MHz
# create_generated_clock -name clk_dsp -source [get_ports clk_board_100m] \
#     -multiply_by 125 -divide_by 32 [get_pins u_pll|outclk_dsp]
#
# PCIe reference: 100 MHz (pass-through to R-Tile)
# create_generated_clock -name clk_pcie -source [get_ports clk_board_100m] \
#     -divide_by 1 [get_pins u_pll|outclk_pcie]
#
# HBM reference: from dedicated UIB refclk pins
# create_generated_clock -name clk_hbm -source [get_ports hbm_refclk_p] \
#     -divide_by 1 [get_pins u_pll|outclk_hbm]

# ── Clock Groups (asynchronous domains) ──────────────────────────────────
# set_clock_groups -asynchronous \
#     -group [get_clocks clk_board_100m] \
#     -group [get_clocks clk_dsp] \
#     -group [get_clocks clk_pcie] \
#     -group [get_clocks clk_hbm]

# ── I/O Constraints ──────────────────────────────────────────────────────
# Board LED outputs (slow, no tight timing)
set_output_delay -clock clk_board_100m -max 5.000 [get_ports debug_led*]
set_output_delay -clock clk_board_100m -min 0.000 [get_ports debug_led*]

# UART TX (115200 baud → ~8.68 us per bit, relaxed timing)
set_output_delay -clock clk_board_100m -max 20.000 [get_ports uart_tx]
set_output_delay -clock clk_board_100m -min 0.000  [get_ports uart_tx]

# ── DSP Multicycle Paths ─────────────────────────────────────────────────
# fp4_mac: 3-stage pipeline, accumulator toggles on valid only
# Relax timing on accumulator path
# set_multicycle_path -setup 2 -to [get_registers *u_mac*accumulator*]
# set_multicycle_path -hold  1 -to [get_registers *u_mac*accumulator*]

# ── Systolic Array Timing ────────────────────────────────────────────────
# HBM → DSP data path: 1-cycle to cross clock domain
# set_max_delay -from [get_registers *hbm*] -to [get_registers *u_array*] 2.500
# set_min_delay -from [get_registers *hbm*] -to [get_registers *u_array*] 0.000

# ── False Paths ──────────────────────────────────────────────────────────
# JTAG / Signal Tap debug
set_false_path -to [get_pins sld_signaltap:*]

# Cross-clock-domain synchronizers (2-FF chains)
# set_false_path -from [get_registers *sync_reg*] -to [get_registers *sync_reg*]

# ── Clock Uncertainty ────────────────────────────────────────────────────
# derive_pll_clocks -create_base_clocks
# derive_clock_uncertainty

# ── Report Timing (post-fit) ─────────────────────────────────────────────
# After fitting, run:
#   report_timing -setup -npaths 100 -detail full_path -file timing_setup.rpt
#   report_timing -hold  -npaths 100 -detail full_path -file timing_hold.rpt
#   report_clock_transfer -file clock_transfer.rpt
