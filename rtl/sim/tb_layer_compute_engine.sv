`timescale 1ns/1ps

module tb_layer_compute_engine;
    logic clk, rst_n;
    logic gate_w_wr_en, up_w_wr_en, down_w_wr_en;
    logic [1:0] gate_w_wr_row, up_w_wr_row;
    logic [2:0] down_w_wr_row;
    logic [0:0] gate_w_wr_beat, up_w_wr_beat, down_w_wr_beat;
    logic [15:0] gate_w_wr_data, up_w_wr_data, down_w_wr_data;
    logic gamma_wr_en;
    logic [2:0] gamma_wr_idx;
    logic signed [31:0] gamma_wr_data;
    logic scale_wr_en;
    logic [1:0] scale_wr_addr;
    logic [7:0] scale_wr_data;
    logic valid_in;
    logic signed [31:0] a0,a1,a2,a3,a4,a5,a6,a7;
    logic valid_out;
    logic router_ok;
    logic signed [31:0] y0,y1,y2,y3,y4,y5,y6,y7;

    // Router weight preload
    logic rtr_w_wr_en;
    logic [1:0] rtr_w_wr_expert;
    logic [2:0] rtr_w_wr_idx;
    logic signed [31:0] rtr_w_wr_data;

    layer_compute_engine dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task ws(input [1:0] a, input [7:0] d); @(posedge clk); scale_wr_en=1; scale_wr_addr=a; scale_wr_data=d; @(posedge clk); scale_wr_en=0; endtask
    task wg(input [1:0] r, input [0:0] b, input [15:0] d); @(posedge clk); gate_w_wr_en=1; gate_w_wr_row=r; gate_w_wr_beat=b; gate_w_wr_data=d; @(posedge clk); gate_w_wr_en=0; endtask
    task wu(input [1:0] r, input [0:0] b, input [15:0] d); @(posedge clk); up_w_wr_en=1; up_w_wr_row=r; up_w_wr_beat=b; up_w_wr_data=d; @(posedge clk); up_w_wr_en=0; endtask
    task wd(input [2:0] r, input [15:0] d); @(posedge clk); down_w_wr_en=1; down_w_wr_row=r; down_w_wr_beat=0; down_w_wr_data=d; @(posedge clk); down_w_wr_en=0; endtask
    task wgamma(input [2:0] i, input signed [31:0] d); @(posedge clk); gamma_wr_en=1; gamma_wr_idx=i; gamma_wr_data=d; @(posedge clk); gamma_wr_en=0; endtask
    task wrtr(input [1:0] e, input [2:0] i, input signed [31:0] d); @(posedge clk); rtr_w_wr_en=1; rtr_w_wr_expert=e; rtr_w_wr_idx=i; rtr_w_wr_data=d; @(posedge clk); rtr_w_wr_en=0; endtask

    initial begin
        rst_n=0; gate_w_wr_en=0; up_w_wr_en=0; down_w_wr_en=0; gamma_wr_en=0;
        rtr_w_wr_en=0;
        scale_wr_en=0; valid_in=0;
        {a0,a1,a2,a3,a4,a5,a6,a7}='0;
        repeat(4) @(posedge clk); rst_n=1;

        // Preload scales
        ws(0,8'h38); ws(1,8'h38);

        // Preload gamma (identity)
        for (int i=0; i<8; i++) wgamma(i[2:0], 4096);

        // Preload router weights (diagonal)
        for (int e = 0; e < 4; e++) begin
            for (int i = 0; i < 8; i++) begin
                wrtr(e[1:0], i[2:0], (i == e) ? 4096 : 0);
            end
        end

        // FFN: gate/up all rows = 1.0 (fp4 +1.0 = 0x4), down identity
        for (int r=0; r<4; r++) begin
            wg(r[1:0], 0, {4{4'h1}}); wg(r[1:0], 1, {4{4'h0}}); // gate=1.0
            wu(r[1:0], 0, {4{4'h1}}); wu(r[1:0], 1, {4{4'h0}}); // up=1.0
        end
        wd(0, {4'h0,4'h0,4'h0,4'h4});
        wd(1, {4'h0,4'h0,4'h4,4'h0});
        wd(2, {4'h0,4'h4,4'h0,4'h0});
        wd(3, {4'h4,4'h0,4'h0,4'h0});
        for (int r=4; r<8; r++) wd(r[2:0], {4{4'h0}});

        // Input activation: all 1.0
        a0=4096; a1=4096; a2=4096; a3=4096; a4=4096; a5=4096; a6=4096; a7=4096;
        @(posedge clk); #1; valid_in=1;
        @(posedge clk); #1; valid_in=0;

        for (int cyc=0; cyc<500; cyc++) begin
            @(posedge clk);
            if (valid_out) begin
                #1;
                $display("Layer output: %0d %0d %0d %0d %0d %0d %0d %0d",
                         y0, y1, y2, y3, y4, y5, y6, y7);
                // RMSNorm1 output all 4096 → FFN identity → RMSNorm2 scales
                // Expected: nonzero ~5790-5800, zeros = 0
                if (y0 < 5700 || y0 > 5900) $error("y0=%0d out of range", y0);
                if (y3 < 5700 || y3 > 5900) $error("y3=%0d out of range", y3);
                if (y4 != 0) $error("y4 expected 0 got %0d", y4);
                if (!router_ok) $error("router_ok not set (expert 0 not selected)");
                $display("PASS tb_layer_compute_engine (router_ok=%0d)", router_ok);
                $finish;
            end
        end
        $error("timeout");
        $fatal;
    end
endmodule
