`timescale 1ns/1ps
`include "fp4_types.svh"

module _dbg_sys;
    logic clk, rst_n;
    fp4_mac_input_t mac_in;
    fp4_mac_output_t mac_out;
    logic accum_clr;

    fp4_mac #(.ACCUM_WIDTH(32), .VEC_LANES(1)) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 0; accum_clr = 0; mac_in = '0;
        repeat (5) @(posedge clk); rst_n = 1;
        repeat (2) @(posedge clk);

        $display("After reset:");
        $display("  mac_out.result = 0x%08h (%0d)", mac_out.result, mac_out.result);
        $display("  mac_out.valid  = %b", mac_out.valid);

        // Drive a simple MAC: fp4=+1.0, fp8=+1.0, scale=256 (pre-decoded 1.0)
        mac_in.weight = 4'h4;
        mac_in.scale  = 12'd256;
        mac_in.activ  = 8'h38;
        mac_in.valid  = 1'b1;
        @(posedge clk);
        mac_in.valid  = 1'b0;

        // Drain pipeline (6 cycles)
        repeat (6) @(posedge clk);

        $display("After MAC op:");
        $display("  mac_out.result = 0x%08h (%0d)", mac_out.result, mac_out.result);
        $display("  mac_out.valid  = %b", mac_out.valid);

        if (mac_out.result === 32'hx) $display("FAIL: result is X");
        else if (mac_out.result > 3500 && mac_out.result < 4700) $display("PASS: ~4096 as expected");
        else $display("FAIL: unexpected result %0d", mac_out.result);

        $finish;
    end

    initial begin #500000; $error("TIMEOUT"); $finish; end
endmodule
