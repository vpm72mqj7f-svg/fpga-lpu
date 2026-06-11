////////////////////////////////////////////////////////////////////////////////
// v2_lite_full_top.v — PCIe Gen3x16 + HBM2 + FFN (Triple Merge)
// Generated: 2026-06-10, Quartus Prime Pro 26.1
////////////////////////////////////////////////////////////////////////////////
`timescale 1 ps / 1 ps
module v2_lite_full
   (core_clk_iopll_ref_clk_clk, hbm_0_example_design_pll_ref_clk_clk, clk_50m, cpu_resetn, led,
    m2u_bridge_cattrip, m2u_bridge_temp, m2u_bridge_wso,
    m2u_bridge_reset_n, m2u_bridge_wrst_n, m2u_bridge_wrck,
    m2u_bridge_shiftwr, m2u_bridge_capturewr, m2u_bridge_updatewr,
    m2u_bridge_selectwir, m2u_bridge_wsi,
    refclk_pcie_ep_p, refclk_pcie_ep_edge_p, refclk_pcie_ep1_p,
    pcie_ep_rx_p, pcie_ep_tx_p,
    s10_pcie_perstn0, s10_pcie_perstn1, pcie_ep_waken,
    pcie_ep_i2c_scl, pcie_ep_i2c_sda
    );

   // Shared clocks and reset
   input core_clk_iopll_ref_clk_clk;           // 100MHz core clock (shared)
   input hbm_0_example_design_pll_ref_clk_clk; // HBM2 PLL refclk
   input clk_50m;                               // 50MHz (PCIe)
   input cpu_resetn;
   output [3:0] led;

   // HBM2 m2u_bridge
   input m2u_bridge_cattrip;
   input [2:0] m2u_bridge_temp;
   input [7:0] m2u_bridge_wso;
   output m2u_bridge_reset_n, m2u_bridge_wrst_n, m2u_bridge_wrck;
   output m2u_bridge_shiftwr, m2u_bridge_capturewr, m2u_bridge_updatewr;
   output m2u_bridge_selectwir, m2u_bridge_wsi;

   // PCIe
   input refclk_pcie_ep_p, refclk_pcie_ep_edge_p, refclk_pcie_ep1_p;
   output [15:0] pcie_ep_rx_p;
   input [15:0] pcie_ep_tx_p;
   input s10_pcie_perstn0, s10_pcie_perstn1, pcie_ep_waken;
   input pcie_ep_i2c_scl;
   inout pcie_ep_i2c_sda;

   // ========================================================================
   // HBM2 Qsys (ed_synth)
   // ========================================================================
   wire tg0_0_pass, tg0_0_fail, tg0_0_timeout, tg0_1_pass, tg0_1_fail, tg0_1_timeout;
   wire tg1_0_pass, tg1_0_fail, tg1_0_timeout, tg1_1_pass, tg1_1_fail, tg1_1_timeout;
   wire tg2_0_pass, tg2_0_fail, tg2_0_timeout, tg2_1_pass, tg2_1_fail, tg2_1_timeout;
   wire tg3_0_pass, tg3_0_fail, tg3_0_timeout, tg3_1_pass, tg3_1_fail, tg3_1_timeout;
   wire tg4_0_pass, tg4_0_fail, tg4_0_timeout, tg4_1_pass, tg4_1_fail, tg4_1_timeout;
   wire tg5_0_pass, tg5_0_fail, tg5_0_timeout, tg5_1_pass, tg5_1_fail, tg5_1_timeout;
   wire tg6_0_pass, tg6_0_fail, tg6_0_timeout, tg6_1_pass, tg6_1_fail, tg6_1_timeout;
   wire tg7_0_pass, tg7_0_fail, tg7_0_timeout, tg7_1_pass, tg7_1_fail, tg7_1_timeout;

   ed_synth u_hbm (
       .core_clk_iopll_ref_clk_clk(core_clk_iopll_ref_clk_clk),
       .core_clk_iopll_reset_reset(~cpu_resetn),
       .hbm_0_example_design_pll_ref_clk_clk(hbm_0_example_design_pll_ref_clk_clk),
       .hbm_0_example_design_wmcrst_n_in_reset_n(cpu_resetn),
       .hbm_only_reset_in_reset(~cpu_resetn),
       .m2u_bridge_cattrip(m2u_bridge_cattrip),
       .m2u_bridge_temp(m2u_bridge_temp), .m2u_bridge_wso(m2u_bridge_wso),
       .m2u_bridge_reset_n(m2u_bridge_reset_n), .m2u_bridge_wrst_n(m2u_bridge_wrst_n),
       .m2u_bridge_wrck(m2u_bridge_wrck), .m2u_bridge_shiftwr(m2u_bridge_shiftwr),
       .m2u_bridge_capturewr(m2u_bridge_capturewr), .m2u_bridge_updatewr(m2u_bridge_updatewr),
       .m2u_bridge_selectwir(m2u_bridge_selectwir), .m2u_bridge_wsi(m2u_bridge_wsi),
       .tg0_0_status_traffic_gen_pass(tg0_0_pass), .tg0_0_status_traffic_gen_fail(tg0_0_fail),
       .tg0_0_status_traffic_gen_timeout(tg0_0_timeout),
       .tg0_1_status_traffic_gen_pass(tg0_1_pass), .tg0_1_status_traffic_gen_fail(tg0_1_fail),
       .tg0_1_status_traffic_gen_timeout(tg0_1_timeout),
       .tg1_0_status_traffic_gen_pass(tg1_0_pass), .tg1_0_status_traffic_gen_fail(tg1_0_fail),
       .tg1_0_status_traffic_gen_timeout(tg1_0_timeout),
       .tg1_1_status_traffic_gen_pass(tg1_1_pass), .tg1_1_status_traffic_gen_fail(tg1_1_fail),
       .tg1_1_status_traffic_gen_timeout(tg1_1_timeout),
       .tg2_0_status_traffic_gen_pass(tg2_0_pass), .tg2_0_status_traffic_gen_fail(tg2_0_fail),
       .tg2_0_status_traffic_gen_timeout(tg2_0_timeout),
       .tg2_1_status_traffic_gen_pass(tg2_1_pass), .tg2_1_status_traffic_gen_fail(tg2_1_fail),
       .tg2_1_status_traffic_gen_timeout(tg2_1_timeout),
       .tg3_0_status_traffic_gen_pass(tg3_0_pass), .tg3_0_status_traffic_gen_fail(tg3_0_fail),
       .tg3_0_status_traffic_gen_timeout(tg3_0_timeout),
       .tg3_1_status_traffic_gen_pass(tg3_1_pass), .tg3_1_status_traffic_gen_fail(tg3_1_fail),
       .tg3_1_status_traffic_gen_timeout(tg3_1_timeout),
       .tg4_0_status_traffic_gen_pass(tg4_0_pass), .tg4_0_status_traffic_gen_fail(tg4_0_fail),
       .tg4_0_status_traffic_gen_timeout(tg4_0_timeout),
       .tg4_1_status_traffic_gen_pass(tg4_1_pass), .tg4_1_status_traffic_gen_fail(tg4_1_fail),
       .tg4_1_status_traffic_gen_timeout(tg4_1_timeout),
       .tg5_0_status_traffic_gen_pass(tg5_0_pass), .tg5_0_status_traffic_gen_fail(tg5_0_fail),
       .tg5_0_status_traffic_gen_timeout(tg5_0_timeout),
       .tg5_1_status_traffic_gen_pass(tg5_1_pass), .tg5_1_status_traffic_gen_fail(tg5_1_fail),
       .tg5_1_status_traffic_gen_timeout(tg5_1_timeout),
       .tg6_0_status_traffic_gen_pass(tg6_0_pass), .tg6_0_status_traffic_gen_fail(tg6_0_fail),
       .tg6_0_status_traffic_gen_timeout(tg6_0_timeout),
       .tg6_1_status_traffic_gen_pass(tg6_1_pass), .tg6_1_status_traffic_gen_fail(tg6_1_fail),
       .tg6_1_status_traffic_gen_timeout(tg6_1_timeout),
       .tg7_0_status_traffic_gen_pass(tg7_0_pass), .tg7_0_status_traffic_gen_fail(tg7_0_fail),
       .tg7_0_status_traffic_gen_timeout(tg7_0_timeout),
       .tg7_1_status_traffic_gen_pass(tg7_1_pass), .tg7_1_status_traffic_gen_fail(tg7_1_fail),
       .tg7_1_status_traffic_gen_timeout(tg7_1_timeout),
       // FFN AXI read channel — connected to HBM2 axi_0_0 port
       .ffn_axi_arid(9'd0), .ffn_axi_araddr(ffn_araddr[27:0]), .ffn_axi_arlen(ffn_arlen),
       .ffn_axi_arsize(ffn_arsize), .ffn_axi_arburst(2'b01), .ffn_axi_arprot(3'b000),
       .ffn_axi_arqos(4'd0), .ffn_axi_aruser(1'b0), .ffn_axi_arvalid(ffn_arvalid),
       .ffn_axi_arready(ffn_arready),
       .ffn_axi_rid(), .ffn_axi_rdata(ffn_rdata), .ffn_axi_rresp(ffn_rresp),
       .ffn_axi_rlast(ffn_rlast), .ffn_axi_rvalid(ffn_rvalid), .ffn_axi_rready(ffn_rready),
       // Write channel tied inactive (FFN is read-only)
       .ffn_axi_awid(9'd0), .ffn_axi_awaddr(28'd0), .ffn_axi_awlen(8'd0), .ffn_axi_awsize(3'd0),
       .ffn_axi_awburst(2'b01), .ffn_axi_awprot(3'd0), .ffn_axi_awqos(4'd0), .ffn_axi_awuser(1'b0),
       .ffn_axi_awvalid(1'b0), .ffn_axi_awready(),
       .ffn_axi_wdata(256'd0), .ffn_axi_wstrb(32'd0), .ffn_axi_wlast(1'b0),
       .ffn_axi_wvalid(1'b0), .ffn_axi_wready(),
       .ffn_axi_bid(), .ffn_axi_bresp(), .ffn_axi_bvalid(), .ffn_axi_bready(1'b0)
   );

   // HBM2 GPIO (aggregates TG status)
   q_sys_gpio u_gpio (
       .clk_clk(core_clk_iopll_ref_clk_clk),
       .pio_0_external_connection_export({tg0_0_pass,tg0_1_pass,tg1_0_pass,tg1_1_pass,
                                          tg2_0_pass,tg2_1_pass,tg3_0_pass,tg3_1_pass,
                                          tg4_0_pass,tg4_1_pass,tg5_0_pass,tg5_1_pass,
                                          tg6_0_pass,tg6_1_pass,tg7_0_pass,tg7_1_pass,
                                          tg0_0_fail,tg0_1_fail,tg1_0_fail,tg1_1_fail,
                                          tg2_0_fail,tg2_1_fail,tg3_0_fail,tg3_1_fail,
                                          tg4_0_fail,tg4_1_fail,tg5_0_fail,tg5_1_fail,
                                          tg6_0_fail,tg6_1_fail,tg7_0_fail,tg7_1_fail}),
       .reset_reset_n(cpu_resetn)
   );

   wire hbm_all_pass = &{tg0_0_pass,tg0_1_pass,tg1_0_pass,tg1_1_pass,
                          tg2_0_pass,tg2_1_pass,tg3_0_pass,tg3_1_pass,
                          tg4_0_pass,tg4_1_pass,tg5_0_pass,tg5_1_pass,
                          tg6_0_pass,tg6_1_pass,tg7_0_pass,tg7_1_pass};

   // ========================================================================
   // PCIe Qsys (pcie_xcvr_system)
   // ========================================================================
   wire pcie_atx_pll_locked;
   wire pcie_pll_locked_a, pcie_pll_locked_b, pcie_pll_locked_c, pcie_pll_locked_d;
   wire pcie_pll_locked_e, pcie_pll_locked_f, pcie_pll_locked_g, pcie_pll_locked_h;
   wire pcie_pll_locked_i, pcie_pll_locked_j, pcie_pll_locked_k, pcie_pll_locked_l;
   wire pcie_pll_locked_m, pcie_pll_locked_n, pcie_pll_locked_o, pcie_pll_locked_p;
   wire int_reset_n;

   random_start u_rand (.clock(clk_50m), .r_start(), .int_reset_n(int_reset_n));

   pcie_xcvr_system u_pcie (
       .atx_pll_1c_refclk_in_clk_clk(refclk_pcie_ep_p),
       .clk_100_clk(core_clk_iopll_ref_clk_clk),
       .clk_100_reset_reset_n(cpu_resetn & int_reset_n),
       .clk_50_clk(clk_50m),
       .clk_50_reset_reset_n(cpu_resetn & int_reset_n),
       .pcie_xcvr_system_bank_1c_0_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[0]),
       .pcie_xcvr_system_bank_1c_0_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[0]),
       .pcie_xcvr_system_bank_1c_1_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[1]),
       .pcie_xcvr_system_bank_1c_1_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[1]),
       .pcie_xcvr_system_bank_1c_2_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[2]),
       .pcie_xcvr_system_bank_1c_2_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[2]),
       .pcie_xcvr_system_bank_1c_3_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[3]),
       .pcie_xcvr_system_bank_1c_3_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[3]),
       .pcie_xcvr_system_bank_1c_4_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[4]),
       .pcie_xcvr_system_bank_1c_4_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[4]),
       .pcie_xcvr_system_bank_1c_5_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[5]),
       .pcie_xcvr_system_bank_1c_5_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[5]),
       .pcie_xcvr_system_bank_1d_0_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[6]),
       .pcie_xcvr_system_bank_1d_0_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[6]),
       .pcie_xcvr_system_bank_1d_1_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[7]),
       .pcie_xcvr_system_bank_1d_1_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[7]),
       .pcie_xcvr_system_bank_1d_2_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[8]),
       .pcie_xcvr_system_bank_1d_2_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[8]),
       .pcie_xcvr_system_bank_1d_3_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[9]),
       .pcie_xcvr_system_bank_1d_3_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[9]),
       .pcie_xcvr_system_bank_1d_4_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[10]),
       .pcie_xcvr_system_bank_1d_4_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[10]),
       .pcie_xcvr_system_bank_1d_5_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[11]),
       .pcie_xcvr_system_bank_1d_5_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[11]),
       .pcie_xcvr_system_bank_1e_0_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[12]),
       .pcie_xcvr_system_bank_1e_0_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[12]),
       .pcie_xcvr_system_bank_1e_1_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[13]),
       .pcie_xcvr_system_bank_1e_1_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[13]),
       .pcie_xcvr_system_bank_1e_2_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[14]),
       .pcie_xcvr_system_bank_1e_2_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[14]),
       .pcie_xcvr_system_bank_1e_3_xcvr_native_s10_0_rx_serial_data_rx_serial_data(pcie_ep_tx_p[15]),
       .pcie_xcvr_system_bank_1e_3_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[15]),
       .pcie_xcvr_system_pll_status_pll_locked_output_pll_locked(pcie_atx_pll_locked),
       .dbg_pll_locked_a(pcie_pll_locked_a),
       .dbg_pll_locked_b(pcie_pll_locked_b),
       .dbg_pll_locked_c(pcie_pll_locked_c),
       .dbg_pll_locked_d(pcie_pll_locked_d),
       .dbg_pll_locked_e(pcie_pll_locked_e),
       .dbg_pll_locked_f(pcie_pll_locked_f),
       .dbg_pll_locked_g(pcie_pll_locked_g),
       .dbg_pll_locked_h(pcie_pll_locked_h),
       .dbg_pll_locked_i(pcie_pll_locked_i),
       .dbg_pll_locked_j(pcie_pll_locked_j),
       .dbg_pll_locked_k(pcie_pll_locked_k),
       .dbg_pll_locked_l(pcie_pll_locked_l),
       .dbg_pll_locked_m(pcie_pll_locked_m),
       .dbg_pll_locked_n(pcie_pll_locked_n),
       .dbg_pll_locked_o(pcie_pll_locked_o),
       .dbg_pll_locked_p(pcie_pll_locked_p)
   );

   assign pcie_ep_i2c_sda = 1'bz;

   // ========================================================================
   // FFN Engine — Production DSP-based SystemVerilog module
   // ========================================================================
   reg [7:0] rc; wire rn = (rc == 8'd255);
   always @(posedge core_clk_iopll_ref_clk_clk or negedge cpu_resetn)
     if (!cpu_resetn) rc <= 8'd0; else if (rc < 8'd255) rc <= rc + 8'd1;

   // FFN-to-HBM2 AXI read channel wires
   wire [31:0] ffn_araddr;
   wire [7:0]  ffn_arlen;
   wire [2:0]  ffn_arsize;
   wire        ffn_arvalid, ffn_arready;
   wire [255:0] ffn_rdata;
   wire [1:0]  ffn_rresp;
   wire        ffn_rvalid, ffn_rready, ffn_rlast;

   reg fv, fr, fp, fs; reg [16383:0] fd;
   wire rdy, tv, busy, done; wire [16383:0] td;

   // Expert ID: 6 entries of 7-bit each (unpacked array for SV module)
   wire [6:0] ffn_expert_id [0:5];
   assign ffn_expert_id[0] = 7'd0;
   assign ffn_expert_id[1] = 7'd1;
   assign ffn_expert_id[2] = 7'd2;
   assign ffn_expert_id[3] = 7'd3;
   assign ffn_expert_id[4] = 7'd4;
   assign ffn_expert_id[5] = 7'd5;

   // FFN production debug wires
   wire [3:0]  ffn_dbg_fsm;
   wire [2:0]  ffn_dbg_expert_cnt;
   wire        ffn_dbg_gate_done, ffn_dbg_up_done, ffn_dbg_down_done;
   wire        ffn_dbg_silu_active, ffn_dbg_merge_active;
   wire        ffn_dbg_hbm2_busy, ffn_dbg_sa_active;
   wire [2:0]  ffn_dbg_hbm2r_fsm, ffn_dbg_hbm2r_wr_wm, ffn_dbg_hbm2r_rd_wm;
   wire [31:0] ffn_perf_token, ffn_perf_cycle, ffn_perf_expert, ffn_perf_axi_rbeat;
   wire        ffn_err_merge_ovf, ffn_err_silu_ovf, ffn_err_axi_resp;

   v2_lite_ffn_engine #(
       .HIDDEN   (2048),
       .INTER    (1408),
       .NUM_EXPERTS (66),
       .TOP_K    (6),
       .DATA_W   (8),
       .ACCUM_W  (24),
       .DSP_LANES (64),
       .VERSION  (32'h0B061A01)
   ) u_ffn (
       .clk               (core_clk_iopll_ref_clk_clk),
       .rst_n             (rn),
       .pcie_rx_valid     (fv),
       .pcie_rx_data      (fd),
       .pcie_rx_ready     (rdy),
       .pcie_tx_valid     (tv),
       .pcie_tx_data      (td),
       .pcie_tx_ready     (fr),
       .m_axi_araddr      (ffn_araddr),
       .m_axi_arlen       (ffn_arlen),
       .m_axi_arsize      (ffn_arsize),
       .m_axi_arvalid     (ffn_arvalid),
       .m_axi_arready     (ffn_arready),
       .m_axi_rdata       (ffn_rdata),
       .m_axi_rresp       (ffn_rresp),
       .m_axi_rvalid      (ffn_rvalid),
       .m_axi_rready      (ffn_rready),
       .m_axi_rlast       (ffn_rlast),
       .expert_id         (ffn_expert_id),
       .busy              (busy),
       .done              (done),
       .dbg_fsm_state     (ffn_dbg_fsm),
       .dbg_expert_cnt    (ffn_dbg_expert_cnt),
       .dbg_gate_done     (ffn_dbg_gate_done),
       .dbg_up_done       (ffn_dbg_up_done),
       .dbg_down_done     (ffn_dbg_down_done),
       .dbg_silu_active   (ffn_dbg_silu_active),
       .dbg_merge_active  (ffn_dbg_merge_active),
       .dbg_hbm2_busy     (ffn_dbg_hbm2_busy),
       .dbg_sa_active     (ffn_dbg_sa_active),
       .dbg_hbm2r_fsm     (ffn_dbg_hbm2r_fsm),
       .dbg_hbm2r_wr_watermark (ffn_dbg_hbm2r_wr_wm),
       .dbg_hbm2r_rd_watermark (ffn_dbg_hbm2r_rd_wm),
       .perf_token_cnt    (ffn_perf_token),
       .perf_cycle_cnt    (ffn_perf_cycle),
       .perf_expert_cnt   (ffn_perf_expert),
       .perf_axi_rbeat    (ffn_perf_axi_rbeat),
       .err_merge_overflow(ffn_err_merge_ovf),
       .err_silu_overflow (ffn_err_silu_ovf),
       .err_axi_resp_err  (ffn_err_axi_resp)
   );

   // FFN self-test FSM (retained for bring-up verification)
   parameter [3:0] BI=0, BS=1, BB=2, BP=5;
   reg [3:0] bs;
   always @(posedge core_clk_iopll_ref_clk_clk or negedge rn) begin
      if (!rn) begin bs<=BI; fv<=0; fr<=0; fp<=0; fs<=0; end
      else case(bs)
        BI: bs<=BS;
        BS: begin if(!fs) begin fv<=1; fd<=16384'd0;
           if(rdy) begin fv<=0; fs<=1; bs<=BB; end
        end else bs<=BB; end
        BB: if(done) begin fr<=1; bs<=BP; end
        BP: fp<=1;
      endcase
   end

   // ========================================================================
   // Signal Tap Debug Bus — preserved from optimization
   // ========================================================================
   (* keep *) wire [3:0]  dbg_ffn_state = bs;
   (* keep *) wire        dbg_ffn_busy  = busy;
   (* keep *) wire        dbg_ffn_done  = done;
   (* keep *) wire        dbg_ffn_pass  = fp;
   (* keep *) wire        dbg_ffn_rx_v  = fv;
   (* keep *) wire        dbg_ffn_rx_r  = rdy;
   (* keep *) wire        dbg_ffn_tx_v  = tv;
   (* keep *) wire        dbg_ffn_tx_r  = fr;
   (* keep *) wire [7:0]  dbg_ffn_td0   = td[7:0];
   (* keep *) wire [7:0]  dbg_ffn_td1   = td[15:8];
   (* keep *) wire        dbg_hbm_tg_pass = hbm_all_pass;
   (* keep *) wire        dbg_pcie_pll   = pcie_atx_pll_locked;
   (* keep *) wire [15:0] dbg_pcie_pll_bank = {pcie_pll_locked_p, pcie_pll_locked_o, pcie_pll_locked_n, pcie_pll_locked_m, pcie_pll_locked_l, pcie_pll_locked_k, pcie_pll_locked_j, pcie_pll_locked_i, pcie_pll_locked_h, pcie_pll_locked_g, pcie_pll_locked_f, pcie_pll_locked_e, pcie_pll_locked_d, pcie_pll_locked_c, pcie_pll_locked_b, pcie_pll_locked_a};
   (* keep *) wire        dbg_ffn_arvalid = ffn_arvalid;
   (* keep *) wire        dbg_ffn_arready = ffn_arready;
   // Production debug (from DSP .sv engine)
   (* keep *) wire [3:0]  dbg_ffn_fsm_prod = ffn_dbg_fsm;
   (* keep *) wire [2:0]  dbg_ffn_expert_prod = ffn_dbg_expert_cnt;
   (* keep *) wire        dbg_tok_cnt_bit0 = ffn_perf_token[0];
   (* keep *) wire        dbg_cyc_cnt_bit0 = ffn_perf_cycle[0];
   (* keep *) wire        dbg_err_any = ffn_err_merge_ovf | ffn_err_silu_ovf | ffn_err_axi_resp;

   // ========================================================================
   // LEDs: Status from all 3 subsystems
   //   led[0] = HBM2 traffic gen all-pass
   //   led[1] = PCIe PLL not-locked (off = good)
   //   led[2] = FFN busy/done
   //   led[3] = Heartbeat
   // ========================================================================
   reg [26:0] hb;
   always @(posedge core_clk_iopll_ref_clk_clk or negedge cpu_resetn)
     if(!cpu_resetn) hb<=0; else hb<=hb+1;

   assign led[0] = hbm_all_pass;
   assign led[1] = ~pcie_atx_pll_locked;  // OFF when PCIe PLL locked (good)
   assign led[2] = fp ? hb[25] : done;    // FFN pass=blink, else done status
   assign led[3] = hb[26];                // Heartbeat


   // ========================================================================
   // ========================================================================
   // In-System Source/Probe — Multi-instance Debug Register Map
   // Instances: PCIE, HBM2, FFN, SYS (see ISP_DEBUG_REGISTER_MAP.md)
   // ========================================================================
   v2_lite_isp_debug #(.HEARTBEAT_WIDTH(27)) u_isp (
       .clk               (core_clk_iopll_ref_clk_clk),
       .rst_n             (cpu_resetn),
       .led               (led),
       .pcie_atx_pll_locked (pcie_atx_pll_locked),
       .pcie_pll_locked_bank({pcie_pll_locked_p, pcie_pll_locked_o, pcie_pll_locked_n, pcie_pll_locked_m,
                              pcie_pll_locked_l, pcie_pll_locked_k, pcie_pll_locked_j, pcie_pll_locked_i,
                              pcie_pll_locked_h, pcie_pll_locked_g, pcie_pll_locked_f, pcie_pll_locked_e,
                              pcie_pll_locked_d, pcie_pll_locked_c, pcie_pll_locked_b, pcie_pll_locked_a}),
       .tg0_0_pass  (tg0_0_pass),  .tg0_0_fail  (tg0_0_fail),  .tg0_0_timeout  (tg0_0_timeout),
       .tg0_1_pass  (tg0_1_pass),  .tg0_1_fail  (tg0_1_fail),  .tg0_1_timeout  (tg0_1_timeout),
       .tg1_0_pass  (tg1_0_pass),  .tg1_0_fail  (tg1_0_fail),  .tg1_0_timeout  (tg1_0_timeout),
       .tg1_1_pass  (tg1_1_pass),  .tg1_1_fail  (tg1_1_fail),  .tg1_1_timeout  (tg1_1_timeout),
       .tg2_0_pass  (tg2_0_pass),  .tg2_0_fail  (tg2_0_fail),  .tg2_0_timeout  (tg2_0_timeout),
       .tg2_1_pass  (tg2_1_pass),  .tg2_1_fail  (tg2_1_fail),  .tg2_1_timeout  (tg2_1_timeout),
       .tg3_0_pass  (tg3_0_pass),  .tg3_0_fail  (tg3_0_fail),  .tg3_0_timeout  (tg3_0_timeout),
       .tg3_1_pass  (tg3_1_pass),  .tg3_1_fail  (tg3_1_fail),  .tg3_1_timeout  (tg3_1_timeout),
       .tg4_0_pass  (tg4_0_pass),  .tg4_0_fail  (tg4_0_fail),  .tg4_0_timeout  (tg4_0_timeout),
       .tg4_1_pass  (tg4_1_pass),  .tg4_1_fail  (tg4_1_fail),  .tg4_1_timeout  (tg4_1_timeout),
       .tg5_0_pass  (tg5_0_pass),  .tg5_0_fail  (tg5_0_fail),  .tg5_0_timeout  (tg5_0_timeout),
       .tg5_1_pass  (tg5_1_pass),  .tg5_1_fail  (tg5_1_fail),  .tg5_1_timeout  (tg5_1_timeout),
       .tg6_0_pass  (tg6_0_pass),  .tg6_0_fail  (tg6_0_fail),  .tg6_0_timeout  (tg6_0_timeout),
       .tg6_1_pass  (tg6_1_pass),  .tg6_1_fail  (tg6_1_fail),  .tg6_1_timeout  (tg6_1_timeout),
       .tg7_0_pass  (tg7_0_pass),  .tg7_0_fail  (tg7_0_fail),  .tg7_0_timeout  (tg7_0_timeout),
       .tg7_1_pass  (tg7_1_pass),  .tg7_1_fail  (tg7_1_fail),  .tg7_1_timeout  (tg7_1_timeout),
       .ffn_state         (dbg_ffn_state),
       .ffn_busy          (dbg_ffn_busy),
       .ffn_done          (dbg_ffn_done),
       .ffn_pass          (dbg_ffn_pass),
       .ffn_tdata_lo      (dbg_ffn_td0),
       .ffn_tdata_hi      (dbg_ffn_td1),
       .ffn_arvalid       (dbg_ffn_arvalid),
       .ffn_arready       (dbg_ffn_arready),
       // Production FFN engine debug (direct from DSP .sv module)
       .ffn_dbg_fsm       (ffn_dbg_fsm),
       .ffn_dbg_expert_cnt(ffn_dbg_expert_cnt),
       .ffn_dbg_gate_done (ffn_dbg_gate_done),
       .ffn_dbg_up_done   (ffn_dbg_up_done),
       .ffn_dbg_down_done (ffn_dbg_down_done),
       .ffn_dbg_silu_active(ffn_dbg_silu_active),
       .ffn_dbg_merge_active(ffn_dbg_merge_active),
       .ffn_dbg_hbm2_busy (ffn_dbg_hbm2_busy),
       .ffn_dbg_sa_active (ffn_dbg_sa_active),
       .ffn_dbg_hbm2r_fsm (ffn_dbg_hbm2r_fsm),
       .ffn_dbg_hbm2r_wr_wm(ffn_dbg_hbm2r_wr_wm),
       .ffn_dbg_hbm2r_rd_wm(ffn_dbg_hbm2r_rd_wm),
       .ffn_perf_token    (ffn_perf_token),
       .ffn_perf_cycle    (ffn_perf_cycle),
       .ffn_perf_expert   (ffn_perf_expert),
       .ffn_perf_axi_rbeat(ffn_perf_axi_rbeat),
       .ffn_err_merge_ovf (ffn_err_merge_ovf),
       .ffn_err_silu_ovf  (ffn_err_silu_ovf),
       .ffn_err_axi_resp  (ffn_err_axi_resp)
   );


endmodule
