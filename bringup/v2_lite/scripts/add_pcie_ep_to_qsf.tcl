# add_pcie_ep_to_qsf.tcl — Replace old pcie_xcvr_system with new pcie_ep
# Run: quartus_sh -t scripts/add_pcie_ep_to_qsf.tcl
# After Qsys generation: qsys-script --script=scripts/gen_pcie_ep_v4.tcl && qsys-generate pcie_ep.qsys --synthesis=VERILOG

set PROJ v2_lite_full

project_open $PROJ

# === REMOVE old XCVR-only Qsys systems ===
set old_qsys {pcie_xcvr_system pcie_xcvr_test xcvr_test_system}
foreach q $old_qsys {
    set_global_assignment -name QSYS_FILE -remove ${q}.qsys
}

# === REMOVE old IP files ===
set old_ips [get_global_assignments -name IP_FILE]
foreach ip $old_ips {
    if {[string match "*pcie_xcvr*" $ip] || [string match "*xcvr_test*" $ip]} {
        set_global_assignment -name IP_FILE -remove $ip
    }
}

# === ADD new PCIe EP Qsys ===
set_global_assignment -name QSYS_FILE pcie_ep.qsys

# === ADD pcie_ep IP files ===
set ip_dir ip/pcie_ep
if {[file exists $ip_dir]} {
    foreach ip_file [glob -nocomplain $ip_dir/**/*.ip $ip_dir/*.ip] {
        set_global_assignment -name IP_FILE $ip_file
        puts "  + IP: $ip_file"
    }
}

# === UPDATE SDC ===
# Remove old qts_pcie_ep.sdc if empty
set old_sdc [get_global_assignments -name SDC_FILE]
foreach sdc $old_sdc {
    if {$sdc eq "qts_pcie_ep.sdc"} {
        if {![file exists $sdc] || [file size $sdc] == 0} {
            set_global_assignment -name SDC_FILE -remove $sdc
            puts "  - removed empty SDC: $sdc"
        }
    }
}
# Add new SDC
set_global_assignment -name SDC_FILE v2_lite_full.sdc
puts "  + SDC: v2_lite_full.sdc"

project_close
puts "QSF updated for pcie_ep"
