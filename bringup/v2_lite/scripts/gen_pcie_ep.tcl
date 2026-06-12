# =============================================================================
# gen_pcie_ep.tcl — Generate PCIe Gen3 x16 Endpoint Qsys system
# Usage: qsys-script --script=gen_pcie_ep.tcl --cmd="run"
# Target: Stratix 10 MX, DK-DEV-1SMX-H-A, PCIe banks 1C/1D/1E
# =============================================================================

proc run {} {
    # Create new system
    set sys_name "pcie_ep_system"

    # Remove existing if present
    catch { file delete -force $sys_name }

    # Create new Qsys project
    set project [qsys::create_system $sys_name]
    qsys::set_system_description $project "PCIe Gen3 x16 Endpoint for V2-Lite"

    # ==========================================================================
    # Clock source (100 MHz refclk from board Si5341A U16)
    # ==========================================================================
    set clk_ref [qsys::add_instance $project clk_100 clock_source]
    qsys::set_parameter $clk_ref clockFrequency 100000000

    # ==========================================================================
    # Reset source
    # ==========================================================================
    set rst_in [qsys::add_instance $project reset_in reset_source]

    # ==========================================================================
    # PCIe Hard IP — Stratix 10, Gen3 x16, Endpoint mode
    # ==========================================================================
    set pcie [qsys::add_instance $project pcie_s10_ep altera_pcie_s10_hip_avmm_bridge]

    # Core configuration
    qsys::set_parameter $pcie device_family "Stratix 10"
    qsys::set_parameter $pcie pcie_mode 0;              # 0 = Endpoint
    qsys::set_parameter $pcie pcie_gen_sel 2;            # 0=Gen1, 1=Gen2, 2=Gen3
    qsys::set_parameter $pcie pcie_link_width 4;         # 4 = x16
    qsys::set_parameter $pcie port_type 0;               # 0 = Native Endpoint
    qsys::set_parameter $pcie pcie_spec_version 3;       # 3 = PCIe 3.0

    # Avalon-MM bridge configuration
    qsys::set_parameter $pcie bar0_size "4 KBytes - 12 bits"
    qsys::set_parameter $pcie bar0_type "32-bit Non-Prefetchable Memory"
    qsys::set_parameter $pcie bar2_size "4 GBytes - 32 bits"
    qsys::set_parameter $pcie bar2_type "64-bit Prefetchable Memory"

    # Reference clock
    qsys::set_parameter $pcie refclk_frequency "100 MHz"

    # ==========================================================================
    # ATX PLL — required for PCIe (H-Tile Bank 1C)
    # ==========================================================================
    set atx [qsys::add_instance $project atx_pll_1c altera_xcvr_atx_pll_s10_htile]
    qsys::set_parameter $atx reference_clock_frequency "100.0 MHz"
    qsys::set_parameter $atx output_data_rate "8000 Mbps";  # Gen3 = 8 GT/s
    qsys::set_parameter $atx pll_type "CMU"
    qsys::set_parameter $atx number_of_channels 16

    # ==========================================================================
    # XCVR Reset Controller
    # ==========================================================================
    set xcvr_rst [qsys::add_instance $project xcvr_reset altera_xcvr_reset_control_s10]
    qsys::set_parameter $xcvr_rst number_of_channels 16

    # ==========================================================================
    # AXI4-Lite Bridge (BAR0 → 32-bit slave for register access)
    # ==========================================================================
    set mm_bridge [qsys::add_instance $project mm_bridge altera_avalon_mm_bridge]
    qsys::set_parameter $mm_bridge data_width 32
    qsys::set_parameter $mm_bridge address_width 12

    # ==========================================================================
    # AXI4 Bridge (BAR2 → 256-bit master for HBM2 weight download)
    # ==========================================================================
    set axi_bridge [qsys::add_instance $project axi_bridge altera_axi_bridge]

    # ==========================================================================
    # Connections
    # ==========================================================================
    # Clock
    qsys::connect $clk_ref clk $pcie refclk
    qsys::connect $clk_ref clk $atx pll_refclk0
    qsys::connect $clk_ref clk $xcvr_rst clock
    qsys::connect $clk_ref clk $mm_bridge clk
    qsys::connect $clk_ref clk $axi_bridge clk

    # Reset
    qsys::connect $rst_in reset $pcie npor
    qsys::connect $rst_in reset $pcie pin_perst
    qsys::connect $rst_in reset $xcvr_rst reset
    qsys::connect $rst_in reset $mm_bridge reset
    qsys::connect $rst_in reset $axi_bridge reset

    # ATX PLL → PCIe
    qsys::connect $atx tx_serial_clk $pcie tx_serial_clk
    qsys::connect $atx pll_locked $pcie pll_locked
    qsys::connect $atx pll_powerdown $pcie pll_powerdown

    # XCVR Reset → PCIe
    qsys::connect $xcvr_rst tx_ready $pcie tx_ready
    qsys::connect $xcvr_rst rx_ready $pcie rx_ready
    qsys::connect $xcvr_rst tx_cal_busy $pcie tx_cal_busy
    qsys::connect $xcvr_rst rx_cal_busy $pcie rx_cal_busy
    qsys::connect $xcvr_rst pll_locked $pcie pll_locked
    qsys::connect $xcvr_rst pll_cal_busy $pcie pll_cal_busy

    # PCIe AVMM → MM Bridge (BAR0)
    qsys::connect $pcie bar0 $mm_bridge s0

    # PCIe AVMM → AXI Bridge (BAR2)
    qsys::connect $pcie bar2 $axi_bridge s0

    # ==========================================================================
    # Export Interfaces
    # ==========================================================================
    # PCIe serial lanes
    qsys::export_port $pcie rx_in0  pcie_ep_rx_in0
    qsys::export_port $pcie tx_out0 pcie_ep_tx_out0
    # ... 16 lanes — will generate individual lanes in Qsys

    # BAR0 AXI4-Lite Master (for register access)
    qsys::export_port $mm_bridge m0 bar0_axi_master

    # BAR2 AXI4 Master (for HBM2 weight download)
    qsys::export_port $axi_bridge m0 bar2_axi_master

    # Refclk input
    qsys::export_port $pcie refclk refclk_pcie_ep

    # PERST#
    qsys::export_port $pcie npor pcie_perst_n

    # ==========================================================================
    # Save and generate
    # ==========================================================================
    qsys::save_system $project "${sys_name}.qsys"
    qsys::generate_system $project -synthesis VERILOG -simulation VERILOG

    puts "PCIe EP Qsys generated: ${sys_name}.qsys"
    puts "Next: qsys-generate ${sys_name}.qsys --synthesis=VERILOG"
}

# Run if executed directly
run
