`timescale 1ns/1ps
`include "fp4_types.svh"
module tb_2x2;
    localparam L=2, MR=2;
    logic clk,rst_n,wt_wr_en,valid_in,accum_clr,reduce_start,reduce_done;
    logic [0:0] wt_wr_row,wt_wr_col;
    logic [3:0] wt_wr_data;
    logic [11:0] sc_wr_data;
    logic [L*8-1:0] activ_flat;
    logic [MR*32-1:0] result_flat;
    fp4_systolic_2d #(.LANES(L),.M_ROWS(MR)) dut(.*);
    initial clk=0; always #5 clk=~clk;
    initial begin
        rst_n=0; wt_wr_en=0; valid_in=0; accum_clr=0; reduce_start=0;
        repeat(5) @(posedge clk); rst_n=1; repeat(2) @(posedge clk);
        $display("=== 2x2 Array ===");
        for(int r=0;r<MR;r++) for(int c=0;c<L;c++) begin
            @(posedge clk); #1; wt_wr_en=1; wt_wr_row=r; wt_wr_col=c;
            wt_wr_data=(r==c)?4'h4:4'h0; sc_wr_data=12'd256;
            @(posedge clk); #1; wt_wr_en=0;
        end
        @(posedge clk); #1; valid_in=1; activ_flat={L{8'h38}};
        @(posedge clk); #1; valid_in=0;
        repeat(8) @(posedge clk);
        @(posedge clk); #1; reduce_start=1; @(posedge clk); #1; reduce_start=0;
        while(!reduce_done) @(posedge clk);
        for(int r=0;r<MR;r++) begin
            $display("row%0d=%0d", r, $signed(result_flat[r*32+:32]));
            if($signed(result_flat[r*32+:32])===32'bx) $display("  X!");
        end
        $finish;
    end
endmodule
