#==============================================================================
# pin_assignment_master.tcl — Master FPGA Pin Assignments
#
# Board: DK-DEV-AGM039EA (Intel Agilex 7 M-Series HBM2e Development Kit)
# Device: AGMF039R47A1E2VR0 (ES) / AGMF039R47A1E1VC (Production)
#
# Reference: Altera GSRD pin_assignment_table.tcl format
#            https://github.com/altera-fpga/agilex7f-ed-gsrd
#
# IMPORTANT: Pin locations marked [TODO-BSP] require the AGM039F board BSP.
#            Obtain from:
#            - Intel doc 782461: Agilex 7 M-Series HBM2e Dev Kit User Guide
#            - DK-DEV-AGM039EA Board Schematic
#            - Quartus Pin Planner (after loading BSP device DB)
#==============================================================================

#==============================================================================
# Board-Level I/O: Clock, Reset, LEDs, UART
#==============================================================================

# -- 100 MHz Board Oscillator --
# [TODO-BSP] Replace CK18 with actual AGM039EA pin from BSP
# IO Standard: "TRUE DIFFERENTIAL SIGNALING" or "LVDS"
set_location_assignment PIN_CK18 -to clk_board_100m
set_instance_assignment -name IO_STANDARD "TRUE DIFFERENTIAL SIGNALING" -to clk_board_100m

# -- CPU Reset (active-low pushbutton) --
# [TODO-BSP] Replace A45 with actual AGM039EA pin from BSP
set_location_assignment PIN_A45 -to cpu_reset_n
set_instance_assignment -name IO_STANDARD "1.2 V" -to cpu_reset_n

# -- Debug LEDs (4 user LEDs) --
# [TODO-BSP] Replace with actual AGM039EA LED pins from BSP
set_location_assignment PIN_B50 -to debug_led[0]
set_instance_assignment -name IO_STANDARD "1.2 V" -to debug_led[0]
set_location_assignment PIN_A49 -to debug_led[1]
set_instance_assignment -name IO_STANDARD "1.2 V" -to debug_led[1]
set_location_assignment PIN_D48 -to debug_led[2]
set_instance_assignment -name IO_STANDARD "1.2 V" -to debug_led[2]
set_location_assignment PIN_E47 -to debug_led[3]
set_instance_assignment -name IO_STANDARD "1.2 V" -to debug_led[3]

# -- UART TX (debug console) --
# [TODO-BSP] Replace with actual AGM039EA UART pin from BSP
# set_location_assignment PIN_XX -to uart_tx
# set_instance_assignment -name IO_STANDARD "1.2 V" -to uart_tx

#==============================================================================
# PCIe 5.0 R-Tile (x16, Master Only)
#==============================================================================
# Reference: GSRD board_devkit_fm86/fm87 PCIe pin assignments
#            AGM039F R-Tile is at different bank than AGF023FA
# [TODO-BSP] All PCIe pins require AGM039EA BSP
#
# PCIe TX pairs (16 lanes):
# set_location_assignment PIN_BP55 -to pcie_tx_p[0]
# ...
# set_location_assignment PIN_AE52 -to pcie_tx_p[15]
#
# PCIe RX pairs (16 lanes):
# set_location_assignment PIN_BP61 -to pcie_rx_p[0]
# ...
# set_location_assignment PIN_AE58 -to pcie_rx_p[15]
#
# PCIe Reference Clocks:
# set_location_assignment PIN_AJ48 -to pcie_refclk0
# set_instance_assignment -name IO_STANDARD "HCSL" -to pcie_refclk0
# set_location_assignment PIN_AE48 -to pcie_refclk1
# set_instance_assignment -name IO_STANDARD "HCSL" -to pcie_refclk1
#
# PCIe Reset:
# set_location_assignment PIN_BU58 -to pcie_perst_n
# set_instance_assignment -name IO_STANDARD "1.8 V" -to pcie_perst_n

#==============================================================================
# HBM2e UIB (32 GB, Integrated)
#==============================================================================
# HBM2e is integrated on the AGM039F M-series die. Its UIB (Universal
# Interface Bus) connects internally — NO external pin assignments needed.
# The HBM2e controller IP is instantiated in Platform Designer (QSYS).
#
# QSF assignments for HBM:
# set_global_assignment -name HBM_STACK_HEIGHT 4
# set_global_assignment -name HBM_DATA_RATE "1800 MHz"
# set_global_assignment -name HBM_AXI_DATA_WIDTH 256

#==============================================================================
# C2C (Chip-to-Chip) F-Tile Transceivers
#==============================================================================
# [TODO-BSP] C2C uses F-Tile transceivers. Pin assignments depend on which
#            F-Tile channels are available on the AGM039EA board.
#
# Ring A (Clockwise, 4 lanes):
# set_location_assignment PIN_XX -to c2c_tx_a_p[0]
# set_location_assignment PIN_XX -to c2c_tx_a_n[0]
# ...
#
# Ring B (Counter-clockwise, 4 lanes):
# set_location_assignment PIN_XX -to c2c_tx_b_p[0]
# set_location_assignment PIN_XX -to c2c_tx_b_n[0]
# ...
#
# IO Standard for C2C transceivers: "HIGH SPEED DIFFERENTIAL I/O"

#==============================================================================
# QSFP-DD (F-Tile, Phase 2 — cross-card interconnect)
#==============================================================================
# [TODO-BSP] QSFP-DD uses F-Tile transceivers. For 8-card system:
#           4 lanes × 25G = 100G per QSFP-DD port
# set_location_assignment PIN_XX -to qsfp_tx_p[0]
# ... (4-8 lanes per direction)

#==============================================================================
# Configuration Pins (SDM — Secure Device Manager)
#==============================================================================
# Reference: GSRD config_sdmio
# These are typically fixed by the board design and may not need assignment
# in the .qsf. Configure in Quartus Device & Pin Options instead.

# set_global_assignment -name USE_HPS_COLD_RESET SDM_IO11
# set_global_assignment -name USE_CONF_DONE SDM_IO16

#==============================================================================
# Power Management (PMBus)
#==============================================================================
# Reference: GSRD config_pwrmgt (LTC3888 or ED8401)
# [TODO-BSP] Check AGM039EA schematic for power management IC

# set_global_assignment -name VID_OPERATION_MODE "PMBUS MASTER"
# set_global_assignment -name USE_PWRMGT_SCL SDM_IO0
# set_global_assignment -name USE_PWRMGT_SDA SDM_IO11

#==============================================================================
# Unused Pins
#==============================================================================
# Set unused pins to tri-state with weak pull-up (saves power, prevents noise)
set_global_assignment -name RESERVE_ALL_UNUSED_PINS "AS INPUT TRI-STATED WITH WEAK PULL-UP"

#==============================================================================
# How to fill [TODO-BSP] pin locations:
#==============================================================================
# 1. Open DK-DEV-AGM039EA BSP in Quartus Prime Pro 24.3
# 2. Pin Planner → View board pin locations
# 3. Or: use Board Test System (BTS) GUI: Tools → Board Test System
# 4. Or: reference Intel doc 782461 Appendix A (Pin Tables)
# 5. Update the set_location_assignment lines above
# 6. Run: Analysis & Synthesis → I/O Assignment Analysis
#==============================================================================
