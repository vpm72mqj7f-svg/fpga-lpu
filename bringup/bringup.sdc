# =============================================================================
# bringup.sdc — Synopsys Design Constraints
# Target: Intel Stratix 10 MX (1SM21BHU2F53E1VG)
# Board: DK-DEV-1SMX-H-A
# =============================================================================

# =============================================================================
# Clock Definitions
# =============================================================================

# ---- System Clock: 100 MHz LVDS (Si5341A U16, CLK_SYS_100M) ----
create_clock -name clk_sys_100m -period 10.000 [get_ports {clk_sys_100m_p}]

# ---- 50 MHz System Clock (Si5338A U18, CLK_SYS_50M) ----
create_clock -name clk_sys_50m -period 20.000 [get_ports {clk_sys_50m_p}]

# ---- PCIe Reference Clock: 100 MHz HCSL (Si5341A U16, REFCLK_PCIE_EP) ----
create_clock -name clk_pcie_ref -period 10.000 [get_ports {pcie_ep_refclk_p}]

# ---- Configuration Clock: 125 MHz LVCMOS (Si510 U17, S10_OSC_CLK_1) ----
create_clock -name clk_osc_125m -period 8.000 [get_ports {clk_osc_125m}]

# ---- HBM2 UIB Clocks: 100 MHz LVDS ----
# UIB0: Si5341A U16, pins AR26/AP26
# UIB1: Si5341A U16, pins P27/R27
# These are consumed by the HBM2 IP; define here for completeness
create_clock -name clk_uib0 -period 10.000 [get_ports {hbm2_uib0_refclk_p}]
create_clock -name clk_uib1 -period 10.000 [get_ports {hbm2_uib1_refclk_p}]

# ---- HBM2 ESRAM Clocks: 100 MHz LVDS ----
# ESRAM0: Si5341A U16, pins AU31/AU32
# ESRAM1: Si5341A U16, pins V31/U31
create_clock -name clk_esram0 -period 10.000 [get_ports {hbm2_esram0_refclk_p}]
create_clock -name clk_esram1 -period 10.000 [get_ports {hbm2_esram1_refclk_p}]

# ---- Core Backup Clock: 100 MHz LVDS (Si5338A U18, CLK_CORE_BAK) ----
create_clock -name clk_core_bak -period 10.000 [get_ports {clk_core_bak_p}]

# ---- PCIe Transceiver Clock: 100 MHz LVDS (Si5338A U18, REFCLK_PCIE_EP1) ----
create_clock -name clk_pcie_xcvr -period 10.000 [get_ports {pcie_xcvr_refclk_p}]

# ---- DDR4 Memory Clocks: 133.333 MHz LVDS (Si5338B U19) ----
create_clock -name clk_ddr4_comp -period 7.500 [get_ports {ddr4_comp_clk_p}]
create_clock -name clk_ddr4_dimm -period 7.500 [get_ports {ddr4_dimm_clk_p}]

# ---- HiLo Memory Clock: 133.333 MHz LVDS (Si5338B U19) ----
create_clock -name clk_hilo -period 7.500 [get_ports {hilo_clk_p}]

# =============================================================================
# Clock Groups (Asynchronous relationships)
# =============================================================================

# PCIe reference clock is asynchronous to system clock
set_clock_groups -asynchronous \
    -group {clk_sys_100m clk_sys_50m clk_osc_125m clk_core_bak} \
    -group {clk_pcie_ref clk_pcie_xcvr}

# HBM2 clocks are asynchronous to system clocks
set_clock_groups -asynchronous \
    -group {clk_sys_100m} \
    -group {clk_uib0 clk_uib1 clk_esram0 clk_esram1}

# DDR4/HiLo memory clocks are asynchronous to system
set_clock_groups -asynchronous \
    -group {clk_sys_100m} \
    -group {clk_ddr4_comp clk_ddr4_dimm clk_hilo}

# =============================================================================
# Generated Clocks — PLL Outputs
# =============================================================================

# Internal PLL: 100 MHz → 500 MHz (core fabric) + 250 MHz (DSP)
# These will be derived from the PLL instantiation in pll_controller.sv
# Define approximate constraints; tighten after PLL configuration is finalized

# ---- Estimated Core Clock: 500 MHz (from PLL) ----
# create_generated_clock -name clk_core_500m -source [get_ports {clk_sys_100m_p}] \
#     -divide_by 1 -multiply_by 5 [get_pins {pll_inst|iopll_inst|outclk0}]

# ---- Estimated DSP Clock: 250 MHz (from PLL) ----
# create_generated_clock -name clk_dsp_250m -source [get_ports {clk_sys_100m_p}] \
#     -divide_by 2 -multiply_by 5 [get_pins {pll_inst|iopll_inst|outclk1}]

# =============================================================================
# I/O Constraints
# =============================================================================

# ---- LED outputs: no tight timing needed ----
set_output_delay -clock clk_sys_100m -max 5.000 [get_ports {led[*]}]
set_output_delay -clock clk_sys_100m -min 1.000 [get_ports {led[*]}]

# ---- Reset input: false path (async reset, internally synchronized) ----
set_input_delay -clock clk_sys_100m -max 5.000 [get_ports {cpu_reset_n}]
set_input_delay -clock clk_sys_100m -min 0.000 [get_ports {cpu_reset_n}]

# ---- PCIe PERST: async, synchronized internally ----
set_false_path -to [get_registers *pcie_perst_sync*]

# =============================================================================
# Relaxation for Bring-Up
# =============================================================================
# During initial bring-up, relax timing to speed compilation
# Remove or tighten for production builds

# Allow multi-cycle paths for large memories
# set_multicycle_path -setup 2 -to [get_registers {*ffn_weight_buf*}]
# set_multicycle_path -hold  1 -to [get_registers {*ffn_weight_buf*}]

# =============================================================================
# Global Settings
# =============================================================================
set_global_assignment -name OPTIMIZATION_MODE "HIGH PERFORMANCE EFFORT"
