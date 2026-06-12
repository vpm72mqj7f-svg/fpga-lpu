# gen_pcie_ep_v2.tcl — Generate PCIe Gen3 x16 Endpoint Qsys via CLI
# Usage: qsys-script --script=gen_pcie_ep_v2.tcl
package require qsys

set SYS_NAME "pcie_ep_system"
set INST_NAME "pcie_ep"

# Step 1: Create system
create_system $SYS_NAME
puts "System: $SYS_NAME"

# Step 2: Add PCIe S10 Hard IP with AVMM bridge
set pcie [add_instance $INST_NAME altera_pcie_s10_hip_avmm_bridge]
puts "PCIe instance added"

# Step 3: Configure PCIe EP — Gen3 x16, 256-bit, 250 MHz
# ==========================================================================
# Mode: x16 link width
set_instance_parameter_value $INST_NAME pf0_link_capable               pf0_conn_x16
# Gen3 speed (via wrala_hwtcl: 14 = Gen3x16, 256-bit, 250 MHz)
set_instance_parameter_value $INST_NAME wrala_hwtcl                    "Gen3x16, Interface - 256 bit, 250 MHz"
set_instance_parameter_value $INST_NAME app_interface_freq_hwtcl        "250 MHz"
set_instance_parameter_value $INST_NAME app_interface_width_hwtcl       "256-bit"

# BAR Configuration
# BAR0: 4KB non-prefetchable (12-bit address) → Register Map
set_instance_parameter_value $INST_NAME pf0_bar0_address_width_hwtcl    12
set_instance_parameter_value $INST_NAME pf0_bar0_type_hwtcl             "32-bit Non-Prefetchable Memory"
# BAR2: 4GB prefetchable (32-bit address) → HBM2 Window
set_instance_parameter_value $INST_NAME pf0_bar2_address_width_hwtcl    32
set_instance_parameter_value $INST_NAME pf0_bar2_type_hwtcl             "64-bit Prefetchable Memory"

# Device/Vendor IDs (use Intel default for now)
set_instance_parameter_value $INST_NAME pf0_vendor_id_hwtcl             0x1172
set_instance_parameter_value $INST_NAME pf0_device_id_hwtcl             0x0000
set_instance_parameter_value $INST_NAME pf0_revision_id_hwtcl           0x01
set_instance_parameter_value $INST_NAME pf0_subsystem_vendor_id_hwtcl   0x1172
set_instance_parameter_value $INST_NAME pf0_subsystem_device_id_hwtcl   0x0000

# Class Code: 0x058000 = Memory Controller
set_instance_parameter_value $INST_NAME pf0_class_code_hwtcl            0x058000

# Step 4: Add clock and reset sources
set clk_100 [add_instance clk_100 clock_source]
set_instance_parameter_value clk_100 clockFrequency 100000000
set rst_in [add_instance rst_in reset_source]

# Step 5: Add AXI4 bridge for BAR2 → HBM2
set axi_bar2 [add_instance axi_bar2 altera_axi_bridge]
set_instance_parameter_value axi_bar2 USE_MASTER "1"
set_instance_parameter_value axi_bar2 DATA_WIDTH 256
set_instance_parameter_value axi_bar2 ADDR_WIDTH 28

# Step 6: Connections
# Clock: clk_100 → PCIe refclk
add_connection clk_100 clk $INST_NAME refclk
add_connection clk_100 clk $INST_NAME pld_clk
add_connection clk_100 clk $INST_NAME coreclkout_hip

# Reset: rst_in → PCIe npor + pin_perst
add_connection rst_in reset $INST_NAME npor
add_connection rst_in reset $INST_NAME pin_perst

# PCIe BAR2 AVMM master → AXI bridge slave
add_connection $INST_NAME bar2 axi_bar2 s0

# Export interfaces for top-level
# PCIe serial lanes (RX input to HIP)
export_interface $INST_NAME hip_serial_rx_in0  pcie_ep_rx_in0
export_interface $INST_NAME hip_serial_tx_out0 pcie_ep_tx_out0

# Refclk
export_interface $INST_NAME refclk pcie_ep_refclk

# PERST#
export_interface $INST_NAME npor pcie_perst_n

# AXI4 master output (from BAR2 bridge)
export_interface axi_bar2 m0 bar2_axi_master

# AXI4-Lite master for BAR0 (register access)
export_interface $INST_NAME cra_slave bar0_axi_lite_slave

# Save and generate
save_system $SYS_NAME.qsys
puts "Saved: ${SYS_NAME}.qsys"
puts ""
puts "=== Generating synthesis files ==="
puts "Run: qsys-generate ${SYS_NAME}.qsys --synthesis=VERILOG"
