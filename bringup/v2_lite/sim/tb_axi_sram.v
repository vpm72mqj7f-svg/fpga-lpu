////////////////////////////////////////////////////////////////////////////////
//
// FPGA LPU Bringup
//
// Filename     : tb_axi_sram.v
// Description  : Standalone AXI4 SRAM model verification
//
////////////////////////////////////////////////////////////////////////////////

`timescale 1ns/1ps
module tb_axi_sram;

   reg clk, rst_;

   // AXI Read Master side (simulated DUT)
   reg  [31:0]  araddr;
   reg  [7:0]   arlen;
   reg  [2:0]   arsize;
   reg          arvalid;
   wire         arready;
   reg  [255:0] rdata;
   reg  [1:0]   rresp;
   reg          rvalid;
   reg          rready;
   reg          rlast;

   // Clock
   initial clk = 0;
   always #5 clk = ~clk;

   // Simple AXI master FSM
   reg [7:0] beat_cnt;
   reg [2:0] state;  // 0=idle, 1=req, 2=data
   reg [31:0] errors;

   always @(posedge clk or negedge rst_) begin
      if (!rst_) begin
         state    <= 3'd0;
         arvalid  <= 1'b0;
         arlen    <= 8'd7;
         arsize   <= 3'd5;
         araddr   <= 32'd0;
         rready   <= 1'b0;
         beat_cnt <= 8'd0;
      end else begin
         case (state)
           3'd0: begin  // IDLE — issue read request
              arvalid <= 1'b1;
              state <= 3'd1;
           end
           3'd1: begin  // Wait for arready
              if (arready) begin
                 arvalid <= 1'b0;
                 rready  <= 1'b1;
                 state   <= 3'd2;
              end
           end
           3'd2: begin  // Receive data
              if (rvalid && rready) begin
                 if (beat_cnt == 8'd7) begin  // last beat
                    rready <= 1'b0;
                    state  <= 3'd0;
                 end else begin
                    beat_cnt <= beat_cnt + 8'd1;
                 end
              end
           end
         endcase
      end
   end

   // AXI SRAM (zero-latency)
   reg [255:0] sram [0:63];
   reg [2:0]   rbeat;
   integer i;

   initial begin
      for (i = 0; i < 64; i = i + 1) sram[i] = {256{1'b0}};
   end

   always @(posedge clk or negedge rst_) begin
      if (!rst_) begin
         rdata  <= 256'd0; rresp <= 2'd0; rvalid <= 1'b0; rlast <= 1'b0;
         rbeat  <= 3'd0;
      end else begin
         // Start: respond to arvalid
         if (arvalid && !rvalid) begin
            rbeat  <= 3'd0;
            rdata  <= sram[0];
            rvalid <= 1'b1;
            rlast  <= 1'b0;
         end

         // Stream beats
         if (rvalid && rready) begin
            if (rbeat == 3'd7) begin  // last beat done
               rvalid <= 1'b0;
               rlast  <= 1'b0;
            end else begin
               rbeat  <= rbeat + 3'd1;
               rdata  <= sram[rbeat + 3'd1];
               rvalid <= 1'b1;
               rlast  <= (rbeat == 3'd6);  // second-to-last sets rlast
            end
         end
      end
   end
   assign arready = 1'b1;

   // Test sequence
   initial begin
      errors = 0;
      $display("=== AXI SRAM Standalone Test ===");
      rst_ = 1'b0;
      repeat(5) @(posedge clk);
      rst_ = 1'b1;
      repeat(5) @(posedge clk);

      // Wait for read to start
      wait(arvalid);
      $display("[1] arvalid asserted, addr=0x%08X", araddr);

      // Wait for first data
      wait(rvalid);
      $display("[2] rvalid asserted, data[7:0]=0x%02X (expect 0x00)", rdata[7:0]);
      if (rdata[7:0] == 8'h00) $display("    PASS"); else begin $display("    FAIL"); errors=errors+1; end

      // Wait for last beat
      repeat(20) @(posedge clk);  // wait for all 8 beats
      $display("[3] Total beats received: %0d", beat_cnt);
      if (beat_cnt == 8'd7) $display("    PASS: 8 beats"); else begin $display("    FAIL: %0d beats", beat_cnt); errors=errors+1; end

      repeat(10) @(posedge clk);

      if (errors == 0) $display("\nALL TESTS PASSED");
      else $display("\n%0d TESTS FAILED", errors);
      $finish;
   end

   initial begin
      $dumpfile("tb_axi_sram.vcd");
      $dumpvars(0, tb_axi_sram);
   end

   initial #100000 $finish;  // 100us timeout

endmodule
