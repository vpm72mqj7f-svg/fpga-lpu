////////////////////////////////////////////////////////////////////////////////
// v2_lite_full_top.v — PCIe Gen3x16 + HBM2 + FFN (Triple Merge)
////////////////////////////////////////////////////////////////////////////////
module v2_lite_full
    (core_clk_iopll_ref_clk_clk, hbm_0_example_design_pll_ref_clk_clk, clk_50m, cpu_resetn, led,
     m2u_bridge_cattrip, m2u_bridge_temp, m2u_bridge_wso,
     m2u_bridge_reset_n, m2u_bridge_wrst_n, m2u_bridge_wrck,
     m2u_bridge_shiftwr, m2u_bridge_capturewr, m2u_bridge_updatewr,
     m2u_bridge_selectwir, m2u_bridge_wsi,
     refclk_pcie_ep_p, pcie_ep_rx_p, pcie_ep_rx_n, pcie_ep_tx_p, pcie_ep_tx_n,
     s10_pcie_perstn0, s10_pcie_perstn1, pcie_ep_waken);

   // HBM2
   input core_clk_iopll_ref_clk_clk, hbm_0_example_design_pll_ref_clk_clk;
   input clk_50m, cpu_resetn;
   output [3:0] led;
   input m2u_bridge_cattrip; input [2:0] m2u_bridge_temp; input [7:0] m2u_bridge_wso;
   output m2u_bridge_reset_n, m2u_bridge_wrst_n, m2u_bridge_wrck;
   output m2u_bridge_shiftwr, m2u_bridge_capturewr, m2u_bridge_updatewr;
   output m2u_bridge_selectwir, m2u_bridge_wsi;
   // PCIe
   input refclk_pcie_ep_p;
   output [15:0] pcie_ep_rx_p, pcie_ep_rx_n, pcie_ep_tx_p, pcie_ep_tx_n;
   input s10_pcie_perstn0, s10_pcie_perstn1, pcie_ep_waken;

   // === HBM2 Qsys ===
   ed_synth u_hbm (
       .core_clk_iopll_ref_clk_clk(core_clk_iopll_ref_clk_clk), .core_clk_iopll_reset_reset(~cpu_resetn),
       .hbm_0_example_design_pll_ref_clk_clk(hbm_0_example_design_pll_ref_clk_clk),
       .hbm_0_example_design_wmcrst_n_in_reset_n(cpu_resetn), .hbm_only_reset_in_reset(~cpu_resetn),
       .m2u_bridge_cattrip(m2u_bridge_cattrip), .m2u_bridge_temp(m2u_bridge_temp), .m2u_bridge_wso(m2u_bridge_wso),
       .m2u_bridge_reset_n(m2u_bridge_reset_n), .m2u_bridge_wrst_n(m2u_bridge_wrst_n),
       .m2u_bridge_wrck(m2u_bridge_wrck), .m2u_bridge_shiftwr(m2u_bridge_shiftwr),
       .m2u_bridge_capturewr(m2u_bridge_capturewr), .m2u_bridge_updatewr(m2u_bridge_updatewr),
       .m2u_bridge_selectwir(m2u_bridge_selectwir), .m2u_bridge_wsi(m2u_bridge_wsi),
       .tg0_0_status_traffic_gen_pass(),.tg0_0_status_traffic_gen_fail(),.tg0_0_status_traffic_gen_timeout(),
       .tg0_1_status_traffic_gen_pass(),.tg0_1_status_traffic_gen_fail(),.tg0_1_status_traffic_gen_timeout(),
       .tg1_0_status_traffic_gen_pass(),.tg1_0_status_traffic_gen_fail(),.tg1_0_status_traffic_gen_timeout(),
       .tg1_1_status_traffic_gen_pass(),.tg1_1_status_traffic_gen_fail(),.tg1_1_status_traffic_gen_timeout(),
       .tg2_0_status_traffic_gen_pass(),.tg2_0_status_traffic_gen_fail(),.tg2_0_status_traffic_gen_timeout(),
       .tg2_1_status_traffic_gen_pass(),.tg2_1_status_traffic_gen_fail(),.tg2_1_status_traffic_gen_timeout(),
       .tg3_0_status_traffic_gen_pass(),.tg3_0_status_traffic_gen_fail(),.tg3_0_status_traffic_gen_timeout(),
       .tg3_1_status_traffic_gen_pass(),.tg3_1_status_traffic_gen_fail(),.tg3_1_status_traffic_gen_timeout(),
       .tg4_0_status_traffic_gen_pass(),.tg4_0_status_traffic_gen_fail(),.tg4_0_status_traffic_gen_timeout(),
       .tg4_1_status_traffic_gen_pass(),.tg4_1_status_traffic_gen_fail(),.tg4_1_status_traffic_gen_timeout(),
       .tg5_0_status_traffic_gen_pass(),.tg5_0_status_traffic_gen_fail(),.tg5_0_status_traffic_gen_timeout(),
       .tg5_1_status_traffic_gen_pass(),.tg5_1_status_traffic_gen_fail(),.tg5_1_status_traffic_gen_timeout(),
       .tg6_0_status_traffic_gen_pass(),.tg6_0_status_traffic_gen_fail(),.tg6_0_status_traffic_gen_timeout(),
       .tg6_1_status_traffic_gen_pass(),.tg6_1_status_traffic_gen_fail(),.tg6_1_status_traffic_gen_timeout(),
       .tg7_0_status_traffic_gen_pass(),.tg7_0_status_traffic_gen_fail(),.tg7_0_status_traffic_gen_timeout(),
       .tg7_1_status_traffic_gen_pass(),.tg7_1_status_traffic_gen_fail(),.tg7_1_status_traffic_gen_timeout()
   );

   // === PCIe Qsys ===
   wire int_reset_n;
   random_start u_rand (.clock(clk_50m), .r_start(), .int_reset_n(int_reset_n));

   pcie_xcvr_system u_pcie (
       .atx_pll_1c_refclk_in_clk_clk(refclk_pcie_ep_p),
       .clk_100_clk(core_clk_iopll_ref_clk_clk), .clk_100_reset_reset_n(cpu_resetn & int_reset_n),
       .clk_50_clk(clk_50m), .clk_50_reset_reset_n(cpu_resetn & int_reset_n),
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
   assign pcie_ep_rx_n = {16{1'b0}};
   assign pcie_ep_tx_n = {16{1'b0}};

   // === FFN Engine ===
   reg [7:0] rc; wire rn = (rc == 8'd255);
   always @(posedge core_clk_iopll_ref_clk_clk or negedge cpu_resetn)
     if (!cpu_resetn) rc <= 8'd0; else if (rc < 8'd255) rc <= rc + 8'd1;

   reg fv, fr, fp, fs; reg [16383:0] fd;
   wire rdy, tv, busy, done; wire [16383:0] td;
   wire [31:0] a; wire [7:0] b; wire [2:0] c; wire d,ff; wire [255:0] g; wire [1:0] h; wire i,j;
   assign d=1'b1; assign g=256'd0; assign h=2'd0; assign i=1'b0; assign j=1'b0;
   wire [6:0] e0=0,e1=1,e2=2,e3=3,e4=4,e5=5;

   v2_lite_ffn_engine u_ffn (
       .clk(core_clk_iopll_ref_clk_clk), .rst_(rn),
       .pcie_rx_valid(fv), .pcie_rx_data(fd), .pcie_rx_ready(rdy),
       .pcie_tx_valid(tv), .pcie_tx_data(td), .pcie_tx_ready(fr),
       .m_axi_araddr(a), .m_axi_arlen(b), .m_axi_arsize(c),
       .m_axi_arvalid(d), .m_axi_arready(ff), .m_axi_rdata(g), .m_axi_rresp(h),
       .m_axi_rvalid(i), .m_axi_rready(ff), .m_axi_rlast(j),
       .expert_id_0(e0),.expert_id_1(e1),.expert_id_2(e2),.expert_id_3(e3),.expert_id_4(e4),.expert_id_5(e5),
       .busy(busy), .done(done)
   );

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

   wire [7:0] fb0 = td[7:0], fb1 = td[15:8], fb2 = td[23:16];
   reg [26:0] hb;
   always @(posedge core_clk_iopll_ref_clk_clk or negedge cpu_resetn)
     if(!cpu_resetn) hb<=0; else hb<=hb+1;
   assign led = {fp ? hb[25] : 1'b0, |fb2[7:4], |fb1[3:0], hb[26]};
endmodule
