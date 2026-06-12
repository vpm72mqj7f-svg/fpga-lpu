# gen_pcie_ep_v3.tcl — PCIe Gen3 x16 Endpoint Qsys (corrected API)
# Usage: qsys-script --script=gen_pcie_ep_v3.tcl
package require qsys

set SYS_NAME "pcie_ep_system"

# Create system
create_system $SYS_NAME
puts "=== Creating $SYS_NAME ==="

# PCIe S10 Hard IP with AVMM bridge
set pcie [add_instance pcie_ep altera_pcie_s10_hip_avmm_bridge]

# === Configure: Gen3 x16, 256-bit, 250 MHz ===
set_instance_parameter_value pcie_ep wrala_hwtcl                    "Gen3x16, Interface - 256 bit, 250 MHz"
set_instance_parameter_value pcie_ep app_interface_freq_hwtcl        "250 MHz"
set_instance_parameter_value pcie_ep pf0_link_capable               pf0_conn_x16

# === BAR Configuration ===
# BAR0: 4KB non-prefetchable memory (12-bit addr) → register map
set_instance_parameter_value pcie_ep pf0_bar0_address_width_hwtcl    12
set_instance_parameter_value pcie_ep pf0_bar0_type_hwtcl             "32-bit Non-Prefetchable Memory"
# BAR2: 4GB prefetchable memory (32-bit addr) → HBM2 window
set_instance_parameter_value pcie_ep pf0_bar2_address_width_hwtcl    32
set_instance_parameter_value pcie_ep pf0_bar2_type_hwtcl             "64-bit Prefetchable Memory"

# === Device Identification ===
set_instance_parameter_value pcie_ep pf0_pci_type0_vendor_id_hwtcl     0x1172
set_instance_parameter_value pcie_ep pf0_pci_type0_device_id_hwtcl     0xE001
set_instance_parameter_value pcie_ep pf0_revision_id_hwtcl              0x01
set_instance_parameter_value pcie_ep pf0_subsys_vendor_id_hwtcl         0x1172
set_instance_parameter_value pcie_ep pf0_subsys_dev_id_hwtcl            0x0001
set_instance_parameter_value pcie_ep pf0_class_code_hwtcl               0x058000
set_instance_parameter_value pcie_ep pf0_base_class_code_hwtcl          0x05
set_instance_parameter_value pcie_ep pf0_subclass_code_hwtcl            0x80

# === Clock source ===
set clk [add_instance clk_100 clock_source]
set_instance_parameter_value clk_100 clockFrequency 100000000

# === Reset source ===
set rst [add_instance rst_in reset_source]

# === AXI4 bridge for BAR2 → HBM2 weight download ===
set axi [add_instance axi_bar2 altera_axi_bridge]
# 256-bit data, 28-bit address (256M words for HBM2)
set_instance_parameter_value axi_bar2 DATA_WIDTH 256
set_instance_parameter_value axi_bar2 ADDR_WIDTH 28

# === Connections ===
# Clock
add_connection clk_100 clk pcie_ep refclk

# Reset
add_connection rst_in reset pcie_ep npor
add_connection rst_in reset pcie_ep pin_perst

# BAR2 AVMM master → AXI bridge → exported as AXI4 master
# Note: BAR2 interface name is rxm_bar2 (needs BAR2 width > 0 to exist)
# If BAR2 not exposed as separate AVMM, use the TX slave interface
catch {add_connection pcie_ep rxm_bar2 axi_bar2 s0} err2
puts "BAR2-axi connect: $err2"

# Clock/reset for AXI bridge
add_connection clk_100 clk axi_bar2 clk
add_connection rst_in reset axi_bar2 reset

# === Export interfaces for top-level ===
puts "=== Exporting interfaces ==="
# AXI4 master from BAR2 bridge → connect to HBM2 writer in top-level
catch {export axi_bar2 m0 bar2_axi_master} err_exp
puts "export axi: $err_exp"

# Save system
save_system ${SYS_NAME}.qsys
puts "=== SAVED: ${SYS_NAME}.qsys ==="
puts ""
puts "Run: qsys-generate ${SYS_NAME}.qsys --synthesis=VERILOG"
