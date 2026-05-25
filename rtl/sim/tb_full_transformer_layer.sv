`timescale 1ns/1ps

module tb_full_transformer_layer;
    logic clk, rst_n, gamma_wr_en, attn_score_wr_en, attn_v_wr_en;
    logic [5:0] attn_score_wr_idx, attn_v_wr_idx;
    logic signed [31:0] attn_score_wr_data, attn_v_wr_data;
    logic rtr_w_wr_en; logic [1:0] rtr_w_wr_expert; logic [2:0] rtr_w_wr_idx;
    logic signed [31:0] rtr_w_wr_data, gamma_wr_data;
    logic [2:0] gamma_wr_idx;
    logic gate_w_wr_en, up_w_wr_en, down_w_wr_en;
    logic [1:0] gate_w_wr_row, up_w_wr_row; logic [2:0] down_w_wr_row;
    logic [0:0] gate_w_wr_beat, up_w_wr_beat, down_w_wr_beat;
    logic [15:0] gate_w_wr_data, up_w_wr_data, down_w_wr_data;
    logic scale_wr_en; logic [1:0] scale_wr_addr; logic [7:0] scale_wr_data;
    logic valid_in, valid_out, router_ok;
    logic signed [31:0] a0,a1,a2,a3,a4,a5,a6,a7;
    logic signed [31:0] y0,y1,y2,y3,y4,y5,y6,y7;

    full_transformer_layer dut (.*);
    initial clk=0; always #5 clk=~clk;

    task ws(input [1:0] a, input [7:0] d); @(posedge clk); scale_wr_en=1; scale_wr_addr=a; scale_wr_data=d; @(posedge clk); scale_wr_en=0; endtask
    task wg(input [1:0] r, input [0:0] b, input [15:0] d); @(posedge clk); gate_w_wr_en=1; gate_w_wr_row=r; gate_w_wr_beat=b; gate_w_wr_data=d; @(posedge clk); gate_w_wr_en=0; endtask
    task wu(input [1:0] r, input [0:0] b, input [15:0] d); @(posedge clk); up_w_wr_en=1; up_w_wr_row=r; up_w_wr_beat=b; up_w_wr_data=d; @(posedge clk); up_w_wr_en=0; endtask
    task wd(input [2:0] r, input [15:0] d); @(posedge clk); down_w_wr_en=1; down_w_wr_row=r; down_w_wr_beat=0; down_w_wr_data=d; @(posedge clk); down_w_wr_en=0; endtask
    task wgamma(input[2:0] i, input[31:0] d); @(posedge clk); gamma_wr_en=1; gamma_wr_idx=i; gamma_wr_data=d; @(posedge clk); gamma_wr_en=0; endtask
    task wrtr(input[1:0] e, input[2:0] i, input[31:0] d); @(posedge clk); rtr_w_wr_en=1; rtr_w_wr_expert=e; rtr_w_wr_idx=i; rtr_w_wr_data=d; @(posedge clk); rtr_w_wr_en=0; endtask
    task was(input[5:0] i, input[31:0] d); @(posedge clk); attn_score_wr_en=1; attn_score_wr_idx=i; attn_score_wr_data=d; @(posedge clk); attn_score_wr_en=0; endtask
    task wav(input[5:0] i, input[31:0] d); @(posedge clk); attn_v_wr_en=1; attn_v_wr_idx=i; attn_v_wr_data=d; @(posedge clk); attn_v_wr_en=0; endtask

    initial begin
        rst_n=0; gamma_wr_en=0; attn_score_wr_en=0; attn_v_wr_en=0; rtr_w_wr_en=0;
        gate_w_wr_en=0; up_w_wr_en=0; down_w_wr_en=0; scale_wr_en=0; valid_in=0;
        repeat(4) @(posedge clk); rst_n=1;

        ws(0,8'h38); ws(1,8'h38);
        for (int i=0; i<8; i++) wgamma(i, 4096);
        for (int i=0; i<64; i++) was(i, 4096);  // uniform attn scores
        for (int r=0; r<8; r++) for (int c=0; c<8; c++)
            wav((r*8+c), (r==c) ? 4096 : 0); // V identity
        for (int e=0; e<4; e++) for (int i=0; i<8; i++)
            wrtr(e, i, (i==e) ? 4096 : 0); // router diagonal
        for (int r=0; r<4; r++) begin
            wg(r,0,{4'h4,4'h0,4'h0,4'h0}); wg(r,1,{4{4'h0}}); // gate +1.0
            wu(r,0,{4'h4,4'h0,4'h0,4'h0}); wu(r,1,{4{4'h0}}); // up +1.0
        end
        wd(0,{4'h0,4'h0,4'h0,4'h4}); wd(1,{4'h0,4'h0,4'h4,4'h0});
        wd(2,{4'h0,4'h4,4'h0,4'h0}); wd(3,{4'h4,4'h0,4'h0,4'h0});
        for (int r=4; r<8; r++) wd(r, {4{4'h0}});

        a0=4096;a1=4096;a2=4096;a3=4096;a4=4096;a5=4096;a6=4096;a7=4096;
        @(posedge clk); #1; valid_in=1; @(posedge clk); #1; valid_in=0;

        for (int cyc=0; cyc<500; cyc++) begin
            @(posedge clk);
            if (valid_out) begin #1;
                $display("Layer out: %0d %0d %0d %0d %0d %0d %0d %0d", y0,y1,y2,y3,y4,y5,y6,y7);
                if (y0==0 && y1==0) $error("output all zero"); else begin
                    $display("PASS tb_full_transformer_layer (router_ok=%0d)", router_ok);
                    $finish;
                end
            end
        end
        $error("timeout"); $fatal;
    end
endmodule
