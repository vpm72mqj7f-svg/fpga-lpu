`timescale 1ns/1ps

module tb_expert_ffn_engine_fp4_down;
    localparam int HIDDEN = 8;
    localparam int INTER = 4;
    localparam int LANES = 4;
    localparam int K_BEATS = 2;
    logic clk, rst_n;
    logic activ_wr_en;
    logic [$clog2(K_BEATS)-1:0] activ_wr_beat;
    logic [LANES*8-1:0] activ_wr_data;
    logic scale_wr_en;
    logic [1:0] scale_wr_addr;
    logic [7:0] scale_wr_data;
    logic gate_w_wr_en, up_w_wr_en, down_w_wr_en;
    logic [$clog2(INTER)-1:0] gate_w_wr_row, up_w_wr_row;
    logic [$clog2(K_BEATS)-1:0] gate_w_wr_beat, up_w_wr_beat;
    logic [LANES*4-1:0] gate_w_wr_data, up_w_wr_data;
    logic [$clog2(HIDDEN)-1:0] down_w_wr_row;
    logic [0:0] down_w_wr_beat;
    logic [LANES*4-1:0] down_w_wr_data;
    logic start, busy, done, result_valid;
    logic [$clog2(HIDDEN)-1:0] result_row;
    logic [31:0] result_data;
    logic [31:0] results [HIDDEN];
    int seen;

    expert_ffn_engine_fp4_down dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task ws(input [1:0] a, input [7:0] d); begin @(posedge clk); scale_wr_en<=1; scale_wr_addr<=a; scale_wr_data<=d; @(posedge clk); scale_wr_en<=0; end endtask
    task wa(input int b, input [LANES*8-1:0] d); begin @(posedge clk); activ_wr_en<=1; activ_wr_beat<=b[$clog2(K_BEATS)-1:0]; activ_wr_data<=d; @(posedge clk); activ_wr_en<=0; end endtask
    task wg(input int r,input int b,input [LANES*4-1:0] d); begin @(posedge clk); gate_w_wr_en<=1; gate_w_wr_row<=r[$clog2(INTER)-1:0]; gate_w_wr_beat<=b[$clog2(K_BEATS)-1:0]; gate_w_wr_data<=d; @(posedge clk); gate_w_wr_en<=0; end endtask
    task wu(input int r,input int b,input [LANES*4-1:0] d); begin @(posedge clk); up_w_wr_en<=1; up_w_wr_row<=r[$clog2(INTER)-1:0]; up_w_wr_beat<=b[$clog2(K_BEATS)-1:0]; up_w_wr_data<=d; @(posedge clk); up_w_wr_en<=0; end endtask
    task wd(input int r,input [LANES*4-1:0] d); begin @(posedge clk); down_w_wr_en<=1; down_w_wr_row<=r[$clog2(HIDDEN)-1:0]; down_w_wr_beat<=0; down_w_wr_data<=d; @(posedge clk); down_w_wr_en<=0; end endtask

    initial begin
        rst_n=0; activ_wr_en=0; scale_wr_en=0; gate_w_wr_en=0; up_w_wr_en=0; down_w_wr_en=0;
        start=0; seen=0; activ_wr_beat=0; activ_wr_data='0; scale_wr_addr=0; scale_wr_data=0;
        gate_w_wr_row=0; gate_w_wr_beat=0; gate_w_wr_data='0; up_w_wr_row=0; up_w_wr_beat=0; up_w_wr_data='0; down_w_wr_row=0; down_w_wr_beat=0; down_w_wr_data='0;
        for (int i=0;i<HIDDEN;i++) results[i]=0;
        repeat(4) @(posedge clk); rst_n=1;

        ws(0,8'h38); ws(1,8'h38);
        wa(0,{4{8'h38}}); wa(1,{4{8'h38}});
        for (int r=0;r<INTER;r++) begin
            wg(r,0,{4{4'h1}}); wg(r,1,{4{4'h0}}); // gate=1.0
            wu(r,0,{4{4'h1}}); wu(r,1,{4{4'h0}}); // up=1.0
        end
        // down identity for rows 0..3 over 4 intermediate columns
        wd(0,{4'h0,4'h0,4'h0,4'h4});
        wd(1,{4'h0,4'h0,4'h4,4'h0});
        wd(2,{4'h0,4'h4,4'h0,4'h0});
        wd(3,{4'h4,4'h0,4'h0,4'h0});
        for (int r=4;r<HIDDEN;r++) wd(r,{4{4'h0}});

        @(posedge clk); start<=1; @(posedge clk); start<=0;
        for (int cyc=0;cyc<800;cyc++) begin
            @(posedge clk);
            if (result_valid) begin results[result_row]=result_data; seen++; $display("row %0d result=0x%08h", result_row, result_data); end
            if (done) begin
                #1;
                if (seen != HIDDEN) begin $error("expected %0d rows got %0d", HIDDEN, seen); $fatal; end
                for (int r=0;r<HIDDEN;r++) begin
                    if (r < INTER) begin
                        if (results[r] !== 32'h00000c00) begin $error("row %0d expected 0x00000c00 got 0x%08h", r, results[r]); $fatal; end
                    end else if (results[r] !== 32'h00000000) begin $error("row %0d expected 0 got 0x%08h", r, results[r]); $fatal; end
                end
                $display("PASS tb_expert_ffn_engine_fp4_down");
                $finish;
            end
        end
        $error("timeout waiting for done"); $fatal;
    end
endmodule
