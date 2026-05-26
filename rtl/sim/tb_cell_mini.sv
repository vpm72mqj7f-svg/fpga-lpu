`timescale 1ns/1ps
`include "fp4_types.svh"

module tb_cell_mini;
    logic clk, rst_n;
    logic wt_wr_en;
    logic [3:0]  wt_wr_data;
    logic [11:0] sc_wr_data;
    logic [7:0]  activ_in;
    logic        valid_in;
    logic        accum_clr;
    logic [31:0] accum_out;

    fp4_systolic_cell u_cell (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 0; wt_wr_en = 0; valid_in = 0; accum_clr = 0;
        repeat (5) @(posedge clk); rst_n = 1;
        repeat (2) @(posedge clk);

        $display("=== Minimal Cell Test ===");

        // Load weight=+1.0(fp4=0x4), scale=256 (pre-decoded fp8 1.0)
        @(posedge clk); #1;
        wt_wr_en = 1; wt_wr_data = 4'h4; sc_wr_data = 12'd256;
        @(posedge clk); #1;
        wt_wr_en = 0;

        // Feed one activation
        @(posedge clk); #1;
        valid_in = 1; activ_in = 8'h38;  // fp8 1.0
        @(posedge clk); #1;
        valid_in = 0;

        // Wait for pipeline drain (6 cycles)
        repeat (8) @(posedge clk);

        $display("accum_out = %0d (0x%08h)", $signed(accum_out), accum_out);
        if (accum_out > 3500 && accum_out < 4700)
            $display("PASS — expected ~4096");
        else if (^(accum_out) === 1'bx)
            $display("FAIL — got X (uninitialized weight or scale?)");
        else
            $display("FAIL — got %0d", $signed(accum_out));

        $finish;
    end
endmodule
