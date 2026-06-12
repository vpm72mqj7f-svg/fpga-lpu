# gen_pcie_ep_v4.tcl — Minimal PCIe Gen3 x16 EP Qsys
# Usage: qsys-script --script=gen_pcie_ep_v4.tcl
package require qsys

set SYS "pcie_ep"
create_system $SYS

# === PCIe S10 Hard IP (AVMM bridge) ===
add_instance pcie_ep altera_pcie_s10_hip_avmm_bridge

# Gen3 x16, 256-bit, 250 MHz
set_instance_parameter_value pcie_ep wrala_hwtcl                  "Gen3x16, Interface - 256 bit, 250 MHz"
set_instance_parameter_value pcie_ep app_interface_freq_hwtcl      "250 MHz"
set_instance_parameter_value pcie_ep pf0_link_capable             pf0_conn_x16

# BAR0: 4KB → register map
set_instance_parameter_value pcie_ep pf0_bar0_address_width_hwtcl  12
set_instance_parameter_value pcie_ep pf0_bar0_type_hwtcl           "32-bit Non-Prefetchable Memory"

# BAR2: 4GB → HBM2 window
set_instance_parameter_value pcie_ep pf0_bar2_address_width_hwtcl  32
set_instance_parameter_value pcie_ep pf0_bar2_type_hwtcl           "64-bit Prefetchable Memory"

# Device IDs
set_instance_parameter_value pcie_ep pf0_pci_type0_vendor_id_hwtcl   0x1172
set_instance_parameter_value pcie_ep pf0_pci_type0_device_id_hwtcl   0xE001
set_instance_parameter_value pcie_ep pf0_revision_id_hwtcl           0x01
set_instance_parameter_value pcie_ep pf0_subsys_vendor_id_hwtcl      0x1172
set_instance_parameter_value pcie_ep pf0_subsys_dev_id_hwtcl         0x0001
set_instance_parameter_value pcie_ep pf0_class_code_hwtcl            0x058000

# === Clock ===
add_instance clk clock_source
set_instance_parameter_value clk clockFrequency 100000000

# Connect clock to PCIe refclk
add_connection clk clk pcie_ep refclk

# Save
save_system ${SYS}.qsys
puts "SAVED: ${SYS}.qsys"
puts "Run: qsys-generate ${SYS}.qsys --synthesis=VERILOG"
puts ""
puts "Unconnected interfaces become top-level ports:"
puts "  rxm_bar0  → BAR0 AVMM master (register access)"
puts "  rxm_bar2  → BAR2 AVMM master (HBM2 window)"
puts "  hip_serial → PCIe serial lanes (RX/TX)"
puts "  npor/pin_perst → reset inputs"
