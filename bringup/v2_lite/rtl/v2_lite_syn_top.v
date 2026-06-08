////////////////////////////////////////////////////////////////////////////////
//
// v2_lite_syn_top.v — V2-Lite FFN Synthesis Top (Intel reference pinout)
//
////////////////////////////////////////////////////////////////////////////////
module v2_lite_syn_top
    (
     core_clk_iopll_ref_clk_clk,
     cpu_resetn,
     led
     );

   input        core_clk_iopll_ref_clk_clk;
   input        cpu_resetn;
   output [3:0] led;

   // Clock & Reset
   reg [7:0] rst_cnt;
   wire rst_n;
   assign rst_n = (rst_cnt == 8'd255);
   always @(posedge core_clk_iopll_ref_clk_clk or negedge cpu_resetn)
     if (!cpu_resetn) rst_cnt <= 8'd0;
     else if (rst_cnt < 8'd255) rst_cnt <= rst_cnt + 8'd1;

   // FFN Engine
   reg         ffn_rx_valid;
   reg  [16383:0] ffn_rx_data;
   wire        ffn_rx_ready;
   wire        ffn_tx_valid;
   wire [16383:0] ffn_tx_data;
   reg         ffn_tx_ready;
   wire        ffn_busy, ffn_done;

   wire [31:0]  ffn_araddr;
   wire [7:0]   ffn_arlen;
   wire [2:0]   ffn_arsize;
   wire         ffn_arvalid, ffn_arready;
   wire [255:0] ffn_rdata;
   wire [1:0]   ffn_rresp;
   wire         ffn_rvalid, ffn_rready, ffn_rlast;

   assign ffn_arready = 1'b1;
   assign ffn_rdata  = 256'd0;
   assign ffn_rresp  = 2'd0;
   assign ffn_rvalid = 1'b0;
   assign ffn_rlast  = 1'b0;

   wire [6:0] ffn_expert_id_0, ffn_expert_id_1, ffn_expert_id_2, ffn_expert_id_3, ffn_expert_id_4, ffn_expert_id_5;
   assign ffn_expert_id_0 = 7'd0;
   assign ffn_expert_id_1 = 7'd1;
   assign ffn_expert_id_2 = 7'd2;
   assign ffn_expert_id_3 = 7'd3;
   assign ffn_expert_id_4 = 7'd4;
   assign ffn_expert_id_5 = 7'd5;

   v2_lite_ffn_engine u_ffn (
       .clk(core_clk_iopll_ref_clk_clk), .rst_(rst_n),
       .pcie_rx_valid(ffn_rx_valid), .pcie_rx_data(ffn_rx_data), .pcie_rx_ready(ffn_rx_ready),
       .pcie_tx_valid(ffn_tx_valid), .pcie_tx_data(ffn_tx_data), .pcie_tx_ready(ffn_tx_ready),
       .m_axi_araddr(ffn_araddr), .m_axi_arlen(ffn_arlen), .m_axi_arsize(ffn_arsize),
       .m_axi_arvalid(ffn_arvalid), .m_axi_arready(ffn_arready),
       .m_axi_rdata(ffn_rdata), .m_axi_rresp(ffn_rresp),
       .m_axi_rvalid(ffn_rvalid), .m_axi_rready(ffn_rready), .m_axi_rlast(ffn_rlast),
       .expert_id_0(ffn_expert_id_0), .expert_id_1(ffn_expert_id_1), .expert_id_2(ffn_expert_id_2),
       .expert_id_3(ffn_expert_id_3), .expert_id_4(ffn_expert_id_4), .expert_id_5(ffn_expert_id_5),
       .busy(ffn_busy), .done(ffn_done)
   );

   // Self-test FSM
   parameter [3:0] B_IDLE=0, B_WAIT=1, B_SEND=2, B_BUSY=3, B_PASS=5, B_FAIL=6;
   reg [3:0] bst;
   reg       ffn_pass;
   reg       ffn_rx_sent;

   always @(posedge core_clk_iopll_ref_clk_clk or negedge rst_n) begin
      if (!rst_n) begin
         bst <= B_IDLE; ffn_rx_valid <= 1'b0; ffn_tx_ready <= 1'b0;
         ffn_pass <= 1'b0; ffn_rx_sent <= 1'b0;
      end else begin
         case (bst)
           B_IDLE:  bst <= B_WAIT;
           B_WAIT:  bst <= B_SEND;
           B_SEND: begin
              if (!ffn_rx_sent) begin
                 ffn_rx_valid <= 1'b1;
                 ffn_rx_data <= 16384'd0;  // placeholder data
                 if (ffn_rx_ready) begin ffn_rx_valid <= 1'b0; ffn_rx_sent <= 1'b1; bst <= B_BUSY; end
              end else bst <= B_BUSY;
           end
           B_BUSY: begin
              if (ffn_done) begin ffn_tx_ready <= 1'b1; bst <= ffn_pass ? B_PASS : B_FAIL; end
           end
           B_PASS, B_FAIL: ;
         endcase
      end
   end

   // LEDs — basic status
   reg [26:0] hb_cnt;
   always @(posedge core_clk_iopll_ref_clk_clk or negedge cpu_resetn)
     if (!cpu_resetn) hb_cnt <= 27'd0;
     else hb_cnt <= hb_cnt + 27'd1;

   assign led[0] = hb_cnt[26];           // heartbeat
   assign led[1] = ffn_busy;              // FFN active
   assign led[2] = ffn_done;              // FFN done pulse
   assign led[3] = ffn_pass ? hb_cnt[25] : 1'b0;  // PASS blink / FAIL off

endmodule
