#==============================================================================
# pin_assignment_slave.tcl — Slave FPGA Pin Assignments
#
# Board: DK-DEV-AGM039EA (Intel Agilex 7 M-Series HBM2e Development Kit)
# Device: AGMF039R47A1E2VR0 (ES) / AGMF039R47A1E1VC (Production)
#
# Key difference from Master: NO PCIe R-Tile pins.
# C2C F-Tile transceivers may use different channels (more available).
# Saves ~15% pins vs Master, simplified board routing.
#==============================================================================

#==============================================================================
# Board-Level I/O: Clock, Reset, LEDs, UART
#==============================================================================
# Same board as Master — identical pin locations

set_location_assignment PIN_CK18 -to clk_board_100m ;# [TODO-BSP]
set_instance_assignment -name IO_STANDARD "TRUE DIFFERENTIAL SIGNALING" -to clk_board_100m

set_location_assignment PIN_A45 -to cpu_reset_n ;# [TODO-BSP]
set_instance_assignment -name IO_STANDARD "1.2 V" -to cpu_reset_n

set_location_assignment PIN_B50 -to debug_led[0] ;# [TODO-BSP]
set_instance_assignment -name IO_STANDARD "1.2 V" -to debug_led[0]
set_location_assignment PIN_A49 -to debug_led[1] ;# [TODO-BSP]
set_instance_assignment -name IO_STANDARD "1.2 V" -to debug_led[1]
set_location_assignment PIN_D48 -to debug_led[2] ;# [TODO-BSP]
set_instance_assignment -name IO_STANDARD "1.2 V" -to debug_led[2]
set_location_assignment PIN_E47 -to debug_led[3] ;# [TODO-BSP]
set_instance_assignment -name IO_STANDARD "1.2 V" -to debug_led[3]

#==============================================================================
# HBM2e UIB (32 GB, Integrated)
#==============================================================================
# Same as Master — HBM2e is on-die, no external pin assignments.
# set_global_assignment -name HBM_STACK_HEIGHT 4
# set_global_assignment -name HBM_DATA_RATE "1800 MHz"
# set_global_assignment -name HBM_AXI_DATA_WIDTH 256

#==============================================================================
# C2C (Chip-to-Chip) F-Tile Transceivers — Slave Role
#==============================================================================
# [TODO-BSP] Slave uses dual F-Tile channels for C2C ring forwarding.
#            Since no PCIe pins are used, more F-Tile channels are available
#            vs the Master. Pin assignments depend on board BSP.
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

#==============================================================================
# QSFP-DD (F-Tile, Phase 2 — cross-card interconnect)
#==============================================================================
# Same as Master — QSFP-DD for multi-card mesh
# set_location_assignment PIN_XX -to qsfp_tx_p[0] ;# [TODO-BSP]

#==============================================================================
# Configuration & Power
#==============================================================================
set_global_assignment -name RESERVE_ALL_UNUSED_PINS "AS INPUT TRI-STATED WITH WEAK PULL-UP"

#==============================================================================
# Slave-Specific Notes
#==============================================================================
# 1. NO PCIe R-Tile pins → saves ~68 pins (16 TX + 16 RX + 2 REFCLK + PERST)
# 2. NO KV DMA engine instantiated (IS_PCIE_MASTER=0 gates PCIe logic)
# 3. C2C uses 8 TX + 8 RX transceiver channels (dual 4-lane rings)
# 4. DIP switch can set CHIP_ID (1-31) — connect to GPIO if available
#    set_location_assignment PIN_XX -to dip_chip_id[0] ;# [TODO-BSP]
#==============================================================================
