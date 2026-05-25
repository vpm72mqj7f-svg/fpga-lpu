`timescale 1ns/1ps

module tb_mla_attention;
    logic clk, rst_n, score_wr_en, v_wr_en, valid_in, valid_out;
    logic [5:0] score_wr_idx, v_wr_idx;
    logic signed [31:0] score_wr_data, v_wr_data;
    logic signed [31:0] y0,y1,y2,y3,y4,y5,y6,y7;

    mla_attention dut (.*);
    initial clk=0; always #5 clk=~clk;

    task wr_s(input [5:0] idx, input [31:0] d); @(posedge clk); score_wr_en=1; score_wr_idx=idx; score_wr_data=d; @(posedge clk); score_wr_en=0; endtask
    task wr_v(input [5:0] idx, input [31:0] d); @(posedge clk); v_wr_en=1; v_wr_idx=idx; v_wr_data=d; @(posedge clk); v_wr_en=0; endtask

    initial begin
        rst_n=0; score_wr_en=0; v_wr_en=0; valid_in=0;
        repeat(4) @(posedge clk); rst_n=1;

        // All scores = 4096 (uniform attention)
        for (int i=0; i<64; i++) wr_s(i, 4096);
        for (int r=0; r<8; r++) for (int c=0; c<8; c++)
            wr_v(r*8+c, (r==c) ? 4096 : 0);

        @(posedge clk); #1; valid_in=1;
        @(posedge clk); #1; valid_in=0;
        @(posedge clk); #1;
        if (valid_out) begin
            // Expected: all 64 (uniform softmax 1/64 × identity V = 4096/64)
            if (y0!=64 || y1!=64 || y2!=64 || y3!=64 ||
                y4!=64 || y5!=64 || y6!=64 || y7!=64) begin
                $error("got %0d %0d %0d %0d %0d %0d %0d %0d", y0,y1,y2,y3,y4,y5,y6,y7);
                $fatal;
            end
            $display("PASS tb_mla_attention");
            $finish;
        end
        $error("timeout");
        $fatal;
    end
endmodule
