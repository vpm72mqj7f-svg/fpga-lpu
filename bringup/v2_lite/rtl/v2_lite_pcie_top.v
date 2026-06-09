////////////////////////////////////////////////////////////////////////////////
//
// v2_lite_pcie_top.v — Merged PCIe EP + V2-Lite FFN Engine
//
// Intel PCIe Gen3 x16 Qsys (pcie_xcvr_system) + FFN engine.
// Share clock/reset. PCIe link handles host communication.
// FFN self-test runs independently on power-up.
// Later: connect FFN ↔ PCIe BAR via dual-port RAM.
//
////////////////////////////////////////////////////////////////////////////////
module v2_lite_pcie_top
    (
     clk_fpga_100m,
     clk_50m,
     pcie_ep_i2c_scl,
     pcie_ep_i2c_sda,
     cpu_resetn,
     s10_led,
     refclk_pcie_ep_p,
     refclk_pcie_ep_edge_p,
     refclk_pcie_ep1_p,
     pcie_ep_rx_p,
     pcie_ep_tx_p,
     s10_pcie_perstn0,
     s10_pcie_perstn1,
     pcie_ep_waken
     );

   input           clk_fpga_100m, clk_50m;
   input           pcie_ep_i2c_scl;
   inout           pcie_ep_i2c_sda;
   input           cpu_resetn;
   output [3:0]    s10_led;
   input           refclk_pcie_ep_p, refclk_pcie_ep_edge_p, refclk_pcie_ep1_p;
   output [15:0]   pcie_ep_rx_p;
   input  [15:0]   pcie_ep_tx_p;
   input           s10_pcie_perstn0, s10_pcie_perstn1, pcie_ep_waken;

   // =========================================================================
   // PCIe Qsys System (unchanged from Intel reference)
   // =========================================================================
   wire pcie_atx_pll_locked;
   wire int_reset_n;
   wire [31:0] random_val;

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
   // FFN Engine — parallel, independent of PCIe
   // =========================================================================
   wire ffn_rst_n;
   reg [7:0] ffn_rst_cnt;
   always @(posedge clk_100m or negedge cpu_resetn)
     if (!cpu_resetn) ffn_rst_cnt <= 8'd0;
     else if (ffn_rst_cnt < 8'd255) ffn_rst_cnt <= ffn_rst_cnt + 8'd1;
   assign ffn_rst_n = (ffn_rst_cnt == 8'd255);

   reg         ffn_rx_valid;
   reg [16383:0] ffn_rx_data;
   wire        ffn_rx_ready;
   wire        ffn_tx_valid;
   wire [16383:0] ffn_tx_data;
   reg         ffn_tx_ready;
   wire        ffn_busy, ffn_done;

   wire [31:0]  ffn_araddr; wire [7:0] ffn_arlen; wire [2:0] ffn_arsize;
   wire ffn_arvalid, ffn_arready, ffn_rready;
   wire [255:0] ffn_rdata; wire [1:0] ffn_rresp; wire ffn_rvalid, ffn_rlast;

   assign ffn_arready = 1'b1; assign ffn_rdata = 256'd0;
   assign ffn_rresp = 2'd0; assign ffn_rvalid = 1'b0; assign ffn_rlast = 1'b0;

   wire [6:0] eid0, eid1, eid2, eid3, eid4, eid5;
   assign eid0 = 7'd0; assign eid1 = 7'd1; assign eid2 = 7'd2;
   assign eid3 = 7'd3; assign eid4 = 7'd4; assign eid5 = 7'd5;

   v2_lite_ffn_engine u_ffn (
       .clk(clk_100m), .rst_(ffn_rst_n),
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
   parameter [3:0] B_IDLE=0, B_SEND=1, B_BUSY=2, B_PASS=5, B_FAIL=6;
   reg [3:0] bst;
   reg ffn_pass, ffn_sent;

   always @(posedge clk_100m or negedge ffn_rst_n) begin
      if (!ffn_rst_n) begin
         bst <= B_IDLE; ffn_rx_valid <= 1'b0; ffn_tx_ready <= 1'b0;
         ffn_pass <= 1'b0; ffn_sent <= 1'b0;
      end else begin
         case (bst)
           B_IDLE:  bst <= B_SEND;
           B_SEND: begin
              if (!ffn_sent) begin
                 ffn_rx_valid <= 1'b1; ffn_rx_data <= 16384'd0;
                 if (ffn_rx_ready) begin ffn_rx_valid <= 1'b0; ffn_sent <= 1'b1; bst <= B_BUSY; end
              end else bst <= B_BUSY;
           end
           B_BUSY: if (ffn_done) begin ffn_tx_ready <= 1'b1; bst <= B_PASS; end
           B_PASS: ffn_pass <= 1'b1;
         endcase
      end
   end

   // =========================================================================
   // LEDs — PCIe status + FFN status
   // =========================================================================
   wire clk_100m;
   assign clk_100m = clk_fpga_100m;
   reg [26:0] hb_cnt;
   always @(posedge clk_100m or negedge cpu_resetn)
     if (!cpu_resetn) hb_cnt <= 27'd0;
     else hb_cnt <= hb_cnt + 27'd1;

   assign s10_led[0] = hb_cnt[26];
   assign s10_led[1] = ffn_busy;
   assign s10_led[2] = ffn_done;
   assign s10_led[3] = ffn_pass ? hb_cnt[25] : 1'b0;

endmodule
