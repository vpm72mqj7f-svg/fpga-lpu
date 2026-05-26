`timescale 1ns/1ps
`include "fp4_types.svh"

module tb_2d_mini;
    localparam int LANES = 1;
    localparam int M_ROWS = 1;

    logic clk, rst_n;
    logic wt_wr_en;
    logic [0:0] wt_wr_row, wt_wr_col;
    logic [3:0]  wt_wr_data;
    logic [11:0] sc_wr_data;
    logic valid_in;
    logic [7:0] activ_flat;
    logic accum_clr, reduce_start, reduce_done;
    logic [31:0] result_flat;

    fp4_systolic_2d #(.LANES(LANES), .M_ROWS(M_ROWS)) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 0; wt_wr_en = 0; valid_in = 0; accum_clr = 0; reduce_start = 0;
        repeat (5) @(posedge clk); rst_n = 1;
        repeat (2) @(posedge clk);

        $display("=== 2D Array: 1x1 Minimal Test ===");

        // Load weight
        @(posedge clk); #1;
        wt_wr_en=1; wt_wr_row=0; wt_wr_col=0; wt_wr_data=4'h4; sc_wr_data=12'd256;
        @(posedge clk); #1;
        wt_wr_en=0;

        // Feed activation
        @(posedge clk); #1;
        valid_in=1; activ_flat=8'h38;
        @(posedge clk); #1;
        valid_in=0;

        // Drain
        repeat(8) @(posedge clk);

        // Reduce
        @(posedge clk); #1; reduce_start=1;
        @(posedge clk); #1; reduce_start=0;

        while (!reduce_done) @(posedge clk);

        $display("result_flat[0] = %0d", $signed(result_flat[31:0]));
        if ($signed(result_flat[31:0]) > 3500)
            $display("PASS");
        else if (^(result_flat[31:0]) === 1'bx)
            $display("FAIL — X detected");
        else
            $display("FAIL — value=%0d", $signed(result_flat[31:0]));

        $finish;
    end
endmodule
