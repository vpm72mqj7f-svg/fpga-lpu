`timescale 1ns/1ps
`include "fp4_types.svh"

module tb_2d_4x4;
    localparam int L = 4;
    localparam int MR = 4;

    logic clk, rst_n;
    logic wt_wr_en, valid_in, accum_clr, reduce_start, reduce_done;
    logic [1:0] wt_wr_row, wt_wr_col;
    logic [3:0]  wt_wr_data;
    logic [11:0] sc_wr_data;
    logic [L*8-1:0] activ_flat;
    logic [MR*32-1:0] result_flat;

    fp4_systolic_2d #(.LANES(L), .M_ROWS(MR)) dut (.*);

    integer pass, fail;
    integer r;
    logic signed [31:0] v;

    initial clk = 0; always #5 clk = ~clk;

    task load(input int r, c, input [3:0] w);
        @(posedge clk); #1; wt_wr_en=1; wt_wr_row=r; wt_wr_col=c;
        wt_wr_data=w; sc_wr_data=12'd256;
        @(posedge clk); #1; wt_wr_en=0;
    endtask

    initial begin
        rst_n=0; wt_wr_en=0; valid_in=0; accum_clr=0; reduce_start=0;
        repeat(5) @(posedge clk); rst_n=1; repeat(2) @(posedge clk);

        $display("=== 4x4 Array: Identity Weights ===");

        // Load 4x4 identity
        for (int r = 0; r < MR; r++)
            for (int c = 0; c < L; c++)
                load(r, c, (r == c) ? 4'h4 : 4'h0);

        // Feed activation (all 1.0), single cycle
        @(posedge clk); #1;
        valid_in = 1; activ_flat = {L{8'h38}};
        @(posedge clk); #1;
        valid_in = 0;

        // Drain pipeline
        repeat(8) @(posedge clk);

        // Reduce
        @(posedge clk); #1; reduce_start = 1;
        @(posedge clk); #1; reduce_start = 0;
        while (!reduce_done) @(posedge clk);

        // Check
        pass = 0; fail = 0;
        for (r = 0; r < MR; r++) begin
            v = result_flat[r*32 +: 32];
            $display("  row %0d: %0d", r, v);
            if (v > 3500 && v < 4700) begin $display("    [ OK ]"); pass++; end
            else begin $display("    [FAIL]"); fail++; end
        end

        $display("%0d PASS, %0d FAIL", pass, fail);
        if (fail > 0) $fatal(1, "FAIL");
        $finish;
    end
endmodule
