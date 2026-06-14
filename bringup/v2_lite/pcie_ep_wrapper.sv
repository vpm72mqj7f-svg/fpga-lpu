// pcie_ep.v — Wrapper for PCIe HIP (regenerated) with BAR0 ports
module pcie_ep (
    input  wire        clk_clk,              // clk.clk
    input  wire        reset_reset_n,        // reset.reset_n
    input  wire        npor_npor,            // npor.npor
    input  wire        pin_perst_n,          // pin_perst
    input  wire        ninit_done_in,        // ninit_done
    output wire        coreclkout_hip,       // coreclkout_hip
    output wire [63:0] bar0_address,         // rxm_bar0.address
    output wire [3:0]  bar0_byteenable,      // .byteenable
    input  wire [31:0] bar0_readdata,        // .readdata
    output wire [31:0] bar0_writedata,       // .writedata
    output wire        bar0_read,            // .read
    output wire        bar0_write,           // .write
    input  wire        bar0_readdatavalid,   // .readdatavalid
    input  wire        bar0_waitrequest      // .waitrequest
);
    wire clk_clk_clk;
    wire [15:0] pcie_rxm_irq_irq;
    wire pcie_app_nreset_status_reset;

    pcie_ep_clk clk_inst (
        .in_clk(clk_clk),
        .reset_n(reset_reset_n),
        .clk_out(clk_clk_clk),
        .reset_n_out()
    );

    pcie_ep_pcie pcie_inst (
        .refclk(clk_clk_clk),
        .coreclkout_hip(coreclkout_hip),
        .npor(npor_npor),
        .pin_perst(pin_perst_n),
        .app_nreset_status(pcie_app_nreset_status_reset),
        .ninit_done(ninit_done_in),
        .rxm_bar0_address_o(bar0_address),
        .rxm_bar0_byteenable_o(bar0_byteenable),
        .rxm_bar0_readdata_i(bar0_readdata),
        .rxm_bar0_writedata_o(bar0_writedata),
        .rxm_bar0_read_o(bar0_read),
        .rxm_bar0_write_o(bar0_write),
        .rxm_bar0_readdatavalid_i(bar0_readdatavalid),
        .rxm_bar0_waitrequest_i(bar0_waitrequest),
        .rxm_irq_i(pcie_rxm_irq_irq),
        .cra_chipselect_i(1'b0),
        .cra_address_i(15'd0),
        .cra_byteenable_i(4'hF),
        .cra_read_i(1'b0),
        .cra_readdata_o(),
        .cra_write_i(1'b0),
        .cra_writedata_i(32'd0),
        .cra_waitrequest_o(),
        .cra_readdatavalid_o(),
        .cra_irq_o(),
        .simu_mode_pipe(1'b0),
        .test_in(67'd0),
        .sim_pipe_pclk_in(1'b0),
        .sim_pipe_rate(),
        .sim_ltssmstate(),
        .txdata0(), .txdata1(), .txdata2(), .txdata3(),
        .txdata4(), .txdata5(), .txdata6(), .txdata7(),
        .txdatak0(), .txdatak1(), .txdatak2(), .txdatak3(),
        .txdatak4(), .txdatak5(), .txdatak6(), .txdatak7(),
        .txcompl0(), .txcompl1(), .txcompl2(), .txcompl3(),
        .txcompl4(), .txcompl5(), .txcompl6(), .txcompl7(),
        .txelecidle0(), .txelecidle1(), .txelecidle2(), .txelecidle3(),
        .txelecidle4(), .txelecidle5(), .txelecidle6(), .txelecidle7(),
        .txdetectrx0(), .txdetectrx1(), .txdetectrx2(), .txdetectrx3(),
        .txdetectrx4(), .txdetectrx5(), .txdetectrx6(), .txdetectrx7(),
        .powerdown0(), .powerdown1(), .powerdown2(), .powerdown3(),
        .powerdown4(), .powerdown5(), .powerdown6(), .powerdown7(),
        .rxpolarity0(1'b0), .rxpolarity1(1'b0), .rxpolarity2(1'b0), .rxpolarity3(1'b0),
        .rxpolarity4(1'b0), .rxpolarity5(1'b0), .rxpolarity6(1'b0), .rxpolarity7(1'b0),
        .currentspeed(),
        .lane_act(),
        .derr_cor_ext_rcv(),
        .derr_cor_ext_rpl(),
        .derr_rpl(),
        .dl_ltssm(),
        .dlup_exit(),
        .ev128ns_done(),
        .ev1us_done(),
        .hotrst_exit(),
        .int_status(),
        .l2_exit(),
        .lane_width_code(),
        .rx_in0(1'b0), .rx_in1(1'b0), .rx_in2(1'b0), .rx_in3(1'b0),
        .rx_in4(1'b0), .rx_in5(1'b0), .rx_in6(1'b0), .rx_in7(1'b0),
        .tx_out0(), .tx_out1(), .tx_out2(), .tx_out3(),
        .tx_out4(), .tx_out5(), .tx_out6(), .tx_out7(),
        .pipe_hclk_in(1'b0),
        .pipe_hclk_out(),
        .rateswitch_out(),
        .txsw_done(),
        .txelecidle_delay(1'b0),
        .txdeemph_delay(1'b0),
        .pld_pcie_hip_control_lock(1'b0),
        .reconfig_clk(1'b0),
        .reconfig_reset(1'b0)
    );
endmodule
