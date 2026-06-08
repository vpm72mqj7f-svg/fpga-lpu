////////////////////////////////////////////////////////////////////////////////
//
// FPGA LPU Bringup
//
// Filename     : tb_ffn_engine.v
// Description  : V2-Lite FFN Engine Testbench (plain Verilog, Icarus/Questa)
//
////////////////////////////////////////////////////////////////////////////////

`timescale 1ns/1ps
module tb_ffn_engine;

   // =========================================================================
   // Parameters
   // =========================================================================
   parameter HIDDEN      = 2048;
   parameter INTER       = 1408;
   parameter NUM_EXPERTS = 66;
   parameter TOP_K       = 6;
   parameter DATA_W      = 8;

   // =========================================================================
   // DUT Signals — Clock & Reset
   // =========================================================================
   reg        clk;
   reg        rst_;

   // =========================================================================
   // DUT Signals — PCIe Streaming
   // =========================================================================
   reg                     pcie_rx_valid;
   reg  [HIDDEN*DATA_W-1:0] pcie_rx_data;
   wire                    pcie_rx_ready;
   wire                    pcie_tx_valid;
   wire [HIDDEN*DATA_W-1:0] pcie_tx_data;
   reg                     pcie_tx_ready;

   // =========================================================================
   // DUT Signals — AXI4 Read Master
   // =========================================================================
   wire [31:0]  m_axi_araddr;
   wire [7:0]   m_axi_arlen;
   wire [2:0]   m_axi_arsize;
   wire         m_axi_arvalid;
   wire         m_axi_arready;
   reg  [255:0] m_axi_rdata;
   reg  [1:0]   m_axi_rresp;
   reg          m_axi_rvalid;
   wire         m_axi_rready;
   reg          m_axi_rlast;

   // =========================================================================
   // DUT Signals — Expert Selection
   // =========================================================================
   wire [6:0] expert_id_0, expert_id_1, expert_id_2, expert_id_3, expert_id_4, expert_id_5;

   // =========================================================================
   // DUT Signals — Status
   // =========================================================================
   wire busy;
   wire done;

   // =========================================================================
   // Clock generation — 100 MHz = 10 ns period
   // =========================================================================
   initial clk = 0;
   always #5 clk = ~clk;

   // =========================================================================
   // DUT instantiation
   // =========================================================================
   v2_lite_ffn_engine #(
       .HIDDEN(HIDDEN),
       .INTER(INTER),
       .NUM_EXPERTS(NUM_EXPERTS),
       .TOP_K(TOP_K),
       .DATA_W(DATA_W)
   ) dut (
       .clk(clk),
       .rst_(rst_),
       .pcie_rx_valid(pcie_rx_valid),
       .pcie_rx_data(pcie_rx_data),
       .pcie_rx_ready(pcie_rx_ready),
       .pcie_tx_valid(pcie_tx_valid),
       .pcie_tx_data(pcie_tx_data),
       .pcie_tx_ready(pcie_tx_ready),
       .m_axi_araddr(m_axi_araddr),
       .m_axi_arlen(m_axi_arlen),
       .m_axi_arsize(m_axi_arsize),
       .m_axi_arvalid(m_axi_arvalid),
       .m_axi_arready(m_axi_arready),
       .m_axi_rdata(m_axi_rdata),
       .m_axi_rresp(m_axi_rresp),
       .m_axi_rvalid(m_axi_rvalid),
       .m_axi_rready(m_axi_rready),
       .m_axi_rlast(m_axi_rlast),
       .expert_id_0(expert_id_0),
       .expert_id_1(expert_id_1),
       .expert_id_2(expert_id_2),
       .expert_id_3(expert_id_3),
       .expert_id_4(expert_id_4),
       .expert_id_5(expert_id_5),
       .busy(busy),
       .done(done)
   );

   assign expert_id_0 = 7'd0;
   assign expert_id_1 = 7'd1;
   assign expert_id_2 = 7'd2;
   assign expert_id_3 = 7'd3;
   assign expert_id_4 = 7'd4;
   assign expert_id_5 = 7'd5;

   // =========================================================================
   // AXI4 SRAM behavioral model (simple: respond with ramp data)
   // =========================================================================
   reg [255:0] sram [0:63];  // 64 beats × 256-bit = 2KB
   reg [7:0]   sram_beat_cnt;
   reg         sram_ar_accepted;
   reg [2:0]   sram_latency_cnt;
   integer     i;

   initial begin
      for (i = 0; i < 64; i = i + 1) begin
         sram[i][  7:  0] = i % 256;
         sram[i][ 15:  8] = (i+1) % 256;
         sram[i][ 23: 16] = (i+2) % 256;
         sram[i][ 31: 24] = (i+3) % 256;
         sram[i][ 39: 32] = (i+4) % 256;
         sram[i][ 47: 40] = (i+5) % 256;
         sram[i][ 55: 48] = (i+6) % 256;
         sram[i][ 63: 56] = (i+7) % 256;
         sram[i][ 71: 64] = (i+8) % 256;
         sram[i][ 79: 72] = (i+9) % 256;
         sram[i][ 87: 80] = (i+10) % 256;
         sram[i][ 95: 88] = (i+11) % 256;
         sram[i][103: 96] = (i+12) % 256;
         sram[i][111:104] = (i+13) % 256;
         sram[i][119:112] = (i+14) % 256;
         sram[i][127:120] = (i+15) % 256;
         sram[i][135:128] = (i+16) % 256;
         sram[i][143:136] = (i+17) % 256;
         sram[i][151:144] = (i+18) % 256;
         sram[i][159:152] = (i+19) % 256;
         sram[i][167:160] = (i+20) % 256;
         sram[i][175:168] = (i+21) % 256;
         sram[i][183:176] = (i+22) % 256;
         sram[i][191:184] = (i+23) % 256;
         sram[i][199:192] = (i+24) % 256;
         sram[i][207:200] = (i+25) % 256;
         sram[i][215:208] = (i+26) % 256;
         sram[i][223:216] = (i+27) % 256;
         sram[i][231:224] = (i+28) % 256;
         sram[i][239:232] = (i+29) % 256;
         sram[i][247:240] = (i+30) % 256;
         sram[i][255:248] = (i+31) % 256;
      end
   end

   // Zero-latency SRAM model (verified standalone)
   reg [2:0] rbeat;

   always @(posedge clk or negedge rst_) begin
      if (!rst_) begin
         m_axi_rdata  <= 256'd0; m_axi_rresp <= 2'd0; m_axi_rvalid <= 1'b0;
         m_axi_rlast  <= 1'b0; rbeat <= 3'd0;
      end else begin
         // Respond to arvalid
         if (m_axi_arvalid && !m_axi_rvalid) begin
            rbeat  <= 3'd0;
            m_axi_rdata  <= sram[0];
            m_axi_rvalid <= 1'b1;
            m_axi_rlast  <= 1'b0;
         end

         // Stream beats on handshake
         if (m_axi_rvalid && m_axi_rready) begin
            if (rbeat == 3'd7) begin
               m_axi_rvalid <= 1'b0;
               m_axi_rlast  <= 1'b0;
            end else begin
               rbeat  <= rbeat + 3'd1;
               m_axi_rdata  <= sram[rbeat + 3'd1];
               m_axi_rvalid <= 1'b1;
            end
         end
      end
   end
   assign m_axi_arready = 1'b1;

   // =========================================================================
   // Test sequence
   // =========================================================================
   reg [31:0] errors;
   integer j;

   initial begin
      errors = 0;
      $display("============================================================");
      $display(" V2-Lite FFN Engine Testbench");
      $display(" HIDDEN=%0d INTER=%0d EXPERTS=%0d TOP_K=%0d", HIDDEN, INTER, NUM_EXPERTS, TOP_K);
      $display("============================================================");

      // ---- Init ----
      rst_ = 1'b0;
      pcie_rx_valid = 1'b0;
      pcie_rx_data  = 16384'b0;
      pcie_tx_ready = 1'b0;
      repeat(10) @(posedge clk);

      // ---- Test 1: Reset release ----
      $display("\n[Test 1] Reset release...");
      rst_ = 1'b1;
      repeat(10) @(posedge clk);
      $display("  rst_ released, state should be S_IDLE");
      if (busy == 1'b0) $display("  PASS: busy=0");
      else begin $display("  FAIL: busy=1"); errors = errors + 1; end

      // ---- Test 2: Activation load ----
      $display("\n[Test 2] Load activation (ramp pattern)...");
      pcie_rx_valid = 1'b1;
      for (j = 0; j < HIDDEN; j = j + 1) begin
         pcie_rx_data[j*DATA_W +: DATA_W] <= (j % 256);
      end
      // Wait for FFN engine to accept data (busy goes high)
      @(posedge clk);
      @(posedge clk);
      wait(busy == 1'b1);
      pcie_rx_valid = 1'b0;
      $display("  Activation loaded, state should advance");

      // ---- Test 3: Wait for FFN pipeline to complete ----
      $display("\n[Test 3] Wait for FFN pipeline...");
      pcie_tx_ready = 1'b1;
      wait(done);
      $display("  FFN done asserted!");

      // ---- Test 4: Check output ----
      $display("\n[Test 4] Check output...");
      if (pcie_tx_valid) begin
         $display("  pcie_tx_valid=1, first byte=0x%02X", pcie_tx_data[7:0]);
         $display("  PASS");
      end else begin
         $display("  FAIL: pcie_tx_valid not asserted");
         errors = errors + 1;
      end

      // ---- Summary ----
      $display("\n============================================================");
      if (errors == 0) $display(" ALL TESTS PASSED");
      else             $display(" %0d TESTS FAILED", errors);
      $display("============================================================");
      $finish;
   end

   // =========================================================================
   // Timeout watchdog
   // =========================================================================
   initial begin
      #500000000;  // 500 ms
      $display("TIMEOUT: Simulation exceeded 500ms");
      $finish;
   end

   // =========================================================================
   // Waveform dump
   // =========================================================================
   initial begin
      $dumpfile("tb_ffn_engine.vcd");
      $dumpvars(0, tb_ffn_engine);
   end

endmodule
