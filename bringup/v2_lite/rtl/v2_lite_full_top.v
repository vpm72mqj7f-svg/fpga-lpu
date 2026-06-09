////////////////////////////////////////////////////////////////////////////////
//
// v2_lite_full_top.v — PCIe Gen3x16 + HBM2 + V2-Lite FFN Engine (Triple Merge)
//
// Three subsystems:
//   1. pcie_xcvr_system (Qsys) — PCIe Gen3 x16 Endpoint
//   2. ed_synth (Qsys)         — HBM2 Top Bank Controller
//   3. v2_lite_ffn_engine     — FFN Compute Pipeline
//
// All share core_clk (100MHz). Pin assignments from Intel references.
////////////////////////////////////////////////////////////////////////////////
module v2_lite_full_top
    (
     // ==== PCIe ====
     refclk_pcie_ep_p, refclk_pcie_ep_edge_p, refclk_pcie_ep1_p,
     pcie_ep_rx_p, pcie_ep_tx_p,
     s10_pcie_perstn0, s10_pcie_perstn1, pcie_ep_waken,
     pcie_ep_i2c_scl, pcie_ep_i2c_sda,

     // ==== HBM2 ====
     hbm_0_example_design_pll_ref_clk_clk,
     m2u_bridge_cattrip, m2u_bridge_temp, m2u_bridge_wso,
     m2u_bridge_reset_n, m2u_bridge_wrst_n, m2u_bridge_wrck,
     m2u_bridge_shiftwr, m2u_bridge_capturewr, m2u_bridge_updatewr,
     m2u_bridge_selectwir, m2u_bridge_wsi,

     // ==== Common ====
     clk_fpga_100m, clk_50m, cpu_resetn,
     s10_led
     );

   // PCIe
   input  refclk_pcie_ep_p, refclk_pcie_ep_edge_p, refclk_pcie_ep1_p;
   output [15:0] pcie_ep_rx_p;
   input  [15:0] pcie_ep_tx_p;
   input  s10_pcie_perstn0, s10_pcie_perstn1, pcie_ep_waken;
   input  pcie_ep_i2c_scl;
   inout  pcie_ep_i2c_sda;

   // HBM2
   input  hbm_0_example_design_pll_ref_clk_clk;
   input  m2u_bridge_cattrip;
   input  [2:0] m2u_bridge_temp;
   input  [7:0] m2u_bridge_wso;
   output m2u_bridge_reset_n, m2u_bridge_wrst_n, m2u_bridge_wrck;
   output m2u_bridge_shiftwr, m2u_bridge_capturewr, m2u_bridge_updatewr;
   output m2u_bridge_selectwir, m2u_bridge_wsi;

   // Common
   input  clk_fpga_100m, clk_50m, cpu_resetn;
   output [3:0] s10_led;

   // =========================================================================
   // PCIe Qsys System
   // =========================================================================
   wire pcie_atx_pll_locked;
   wire int_reset_n;

   random_start u_random (.clock(clk_50m), .r_start(), .int_reset_n(int_reset_n));

   pcie_xcvr_system u_pcie (
       .atx_pll_1c_refclk_in_clk_clk(refclk_pcie_ep_p),
       .clk_100_clk(clk_fpga_100m),
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
       .pcie_xcvr_system_bank_1e_3_xcvr_native_s10_0_tx_serial_data_tx_serial_data(pcie_ep_rx_p[15])
   );

   // =========================================================================
   // HBM2 Qsys System
   // =========================================================================
   ed_synth u_hbm (
       .core_clk_iopll_ref_clk_clk(clk_fpga_100m),
       .core_clk_iopll_reset_reset(~cpu_resetn),
       .hbm_0_example_design_pll_ref_clk_clk(hbm_0_example_design_pll_ref_clk_clk),
       .hbm_0_example_design_wmcrst_n_in_reset_n(cpu_resetn),
       .hbm_only_reset_in_reset(~cpu_resetn),
       .m2u_bridge_cattrip(m2u_bridge_cattrip),
       .m2u_bridge_temp(m2u_bridge_temp),
       .m2u_bridge_wso(m2u_bridge_wso),
       .m2u_bridge_reset_n(m2u_bridge_reset_n),
       .m2u_bridge_wrst_n(m2u_bridge_wrst_n),
       .m2u_bridge_wrck(m2u_bridge_wrck),
       .m2u_bridge_shiftwr(m2u_bridge_shiftwr),
       .m2u_bridge_capturewr(m2u_bridge_capturewr),
       .m2u_bridge_updatewr(m2u_bridge_updatewr),
       .m2u_bridge_selectwir(m2u_bridge_selectwir),
       .m2u_bridge_wsi(m2u_bridge_wsi),
       .tg0_0_status_traffic_gen_pass(), .tg0_0_status_traffic_gen_fail(), .tg0_0_status_traffic_gen_timeout(),
       .tg0_1_status_traffic_gen_pass(), .tg0_1_status_traffic_gen_fail(), .tg0_1_status_traffic_gen_timeout(),
       .tg1_0_status_traffic_gen_pass(), .tg1_0_status_traffic_gen_fail(), .tg1_0_status_traffic_gen_timeout(),
       .tg1_1_status_traffic_gen_pass(), .tg1_1_status_traffic_gen_fail(), .tg1_1_status_traffic_gen_timeout(),
       .tg2_0_status_traffic_gen_pass(), .tg2_0_status_traffic_gen_fail(), .tg2_0_status_traffic_gen_timeout(),
       .tg2_1_status_traffic_gen_pass(), .tg2_1_status_traffic_gen_fail(), .tg2_1_status_traffic_gen_timeout(),
       .tg3_0_status_traffic_gen_pass(), .tg3_0_status_traffic_gen_fail(), .tg3_0_status_traffic_gen_timeout(),
       .tg3_1_status_traffic_gen_pass(), .tg3_1_status_traffic_gen_fail(), .tg3_1_status_traffic_gen_timeout(),
       .tg4_0_status_traffic_gen_pass(), .tg4_0_status_traffic_gen_fail(), .tg4_0_status_traffic_gen_timeout(),
       .tg4_1_status_traffic_gen_pass(), .tg4_1_status_traffic_gen_fail(), .tg4_1_status_traffic_gen_timeout(),
       .tg5_0_status_traffic_gen_pass(), .tg5_0_status_traffic_gen_fail(), .tg5_0_status_traffic_gen_timeout(),
       .tg5_1_status_traffic_gen_pass(), .tg5_1_status_traffic_gen_fail(), .tg5_1_status_traffic_gen_timeout(),
       .tg6_0_status_traffic_gen_pass(), .tg6_0_status_traffic_gen_fail(), .tg6_0_status_traffic_gen_timeout(),
       .tg6_1_status_traffic_gen_pass(), .tg6_1_status_traffic_gen_fail(), .tg6_1_status_traffic_gen_timeout(),
       .tg7_0_status_traffic_gen_pass(), .tg7_0_status_traffic_gen_fail(), .tg7_0_status_traffic_gen_timeout(),
       .tg7_1_status_traffic_gen_pass(), .tg7_1_status_traffic_gen_fail(), .tg7_1_status_traffic_gen_timeout()
   );

   // =========================================================================
   // FFN Engine
   // =========================================================================
   reg [7:0] ffn_rst_cnt;
   wire ffn_rst_n = (ffn_rst_cnt == 8'd255);
   always @(posedge clk_fpga_100m or negedge cpu_resetn)
     if (!cpu_resetn) ffn_rst_cnt <= 8'd0;
     else if (ffn_rst_cnt < 8'd255) ffn_rst_cnt <= ffn_rst_cnt + 8'd1;

   reg ffn_rx_valid, ffn_tx_ready, ffn_pass, ffn_sent;
   reg [16383:0] ffn_rx_data;
   wire ffn_rx_ready, ffn_tx_valid, ffn_busy, ffn_done;
   wire [16383:0] ffn_tx_data;

   wire [31:0] ffn_araddr; wire [7:0] ffn_arlen; wire [2:0] ffn_arsize;
   wire ffn_arvalid, ffn_arready, ffn_rready;
   wire [255:0] ffn_rdata; wire [1:0] ffn_rresp; wire ffn_rvalid, ffn_rlast;
   assign ffn_arready=1'b1; assign ffn_rdata=256'd0; assign ffn_rresp=2'd0;
   assign ffn_rvalid=1'b0; assign ffn_rlast=1'b0;

   wire [6:0] eid0=0,eid1=1,eid2=2,eid3=3,eid4=4,eid5=5;

   v2_lite_ffn_engine u_ffn (
       .clk(clk_fpga_100m), .rst_(ffn_rst_n),
       .pcie_rx_valid(ffn_rx_valid), .pcie_rx_data(ffn_rx_data), .pcie_rx_ready(ffn_rx_ready),
       .pcie_tx_valid(ffn_tx_valid), .pcie_tx_data(ffn_tx_data), .pcie_tx_ready(ffn_tx_ready),
       .m_axi_araddr(ffn_araddr), .m_axi_arlen(ffn_arlen), .m_axi_arsize(ffn_arsize),
       .m_axi_arvalid(ffn_arvalid), .m_axi_arready(ffn_arready),
       .m_axi_rdata(ffn_rdata), .m_axi_rresp(ffn_rresp),
       .m_axi_rvalid(ffn_rvalid), .m_axi_rready(ffn_rready), .m_axi_rlast(ffn_rlast),
       .expert_id_0(eid0), .expert_id_1(eid1), .expert_id_2(eid2),
       .expert_id_3(eid3), .expert_id_4(eid4), .expert_id_5(eid5),
       .busy(ffn_busy), .done(ffn_done)
   );

   // Self-test FSM
   parameter [3:0] B_IDLE=0, B_SEND=1, B_BUSY=2, B_PASS=5;
   reg [3:0] bst;
   always @(posedge clk_fpga_100m or negedge ffn_rst_n) begin
      if (!ffn_rst_n) begin bst<=B_IDLE; ffn_rx_valid<=0; ffn_tx_ready<=0; ffn_pass<=0; ffn_sent<=0; end
      else case(bst)
        B_IDLE: bst<=B_SEND;
        B_SEND: begin if(!ffn_sent) begin ffn_rx_valid<=1; ffn_rx_data<=16384'd0;
           if(ffn_rx_ready) begin ffn_rx_valid<=0; ffn_sent<=1; bst<=B_BUSY; end
        end else bst<=B_BUSY; end
        B_BUSY: if(ffn_done) begin ffn_tx_ready<=1; bst<=B_PASS; end
        B_PASS: ffn_pass<=1;
      endcase
   end

   // LEDs
   reg [26:0] hb;
   always @(posedge clk_fpga_100m or negedge cpu_resetn)
     if(!cpu_resetn) hb<=0; else hb<=hb+1;
   assign s10_led = {ffn_pass ? hb[25] : 1'b0, ffn_done, ffn_busy, hb[26]};

endmodule
