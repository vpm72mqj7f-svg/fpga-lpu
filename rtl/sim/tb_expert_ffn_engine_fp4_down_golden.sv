`timescale 1ns/1ps
`include "tb_ffn_golden_pkg.sv"

module tb_expert_ffn_engine_fp4_down_golden;
    localparam int HIDDEN = tb_ffn_golden_pkg::HIDDEN;
    localparam int INTER = tb_ffn_golden_pkg::INTER;
    localparam int LANES = tb_ffn_golden_pkg::LANES;
    localparam int K_BEATS = tb_ffn_golden_pkg::K_BEATS;

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

    task reset_io;
        begin
            activ_wr_en=0; scale_wr_en=0; gate_w_wr_en=0; up_w_wr_en=0; down_w_wr_en=0;
            activ_wr_beat=0; activ_wr_data='0; scale_wr_addr=0; scale_wr_data=0;
            gate_w_wr_row=0; gate_w_wr_beat=0; gate_w_wr_data='0;
            up_w_wr_row=0; up_w_wr_beat=0; up_w_wr_data='0;
            down_w_wr_row=0; down_w_wr_beat=0; down_w_wr_data='0;
            start=0; seen=0;
            for (int i=0;i<HIDDEN;i++) results[i]=0;
        end
    endtask

    task ws(input [1:0] a, input [7:0] d); begin @(posedge clk); scale_wr_en<=1; scale_wr_addr<=a; scale_wr_data<=d; @(posedge clk); scale_wr_en<=0; end endtask
    task wa(input int b, input [LANES*8-1:0] d); begin @(posedge clk); activ_wr_en<=1; activ_wr_beat<=b[$clog2(K_BEATS)-1:0]; activ_wr_data<=d; @(posedge clk); activ_wr_en<=0; end endtask
    task wg(input int r,input int b,input [LANES*4-1:0] d); begin @(posedge clk); gate_w_wr_en<=1; gate_w_wr_row<=r[$clog2(INTER)-1:0]; gate_w_wr_beat<=b[$clog2(K_BEATS)-1:0]; gate_w_wr_data<=d; @(posedge clk); gate_w_wr_en<=0; end endtask
    task wu(input int r,input int b,input [LANES*4-1:0] d); begin @(posedge clk); up_w_wr_en<=1; up_w_wr_row<=r[$clog2(INTER)-1:0]; up_w_wr_beat<=b[$clog2(K_BEATS)-1:0]; up_w_wr_data<=d; @(posedge clk); up_w_wr_en<=0; end endtask
    task wd(input int r,input [LANES*4-1:0] d); begin @(posedge clk); down_w_wr_en<=1; down_w_wr_row<=r[$clog2(HIDDEN)-1:0]; down_w_wr_beat<=0; down_w_wr_data<=d; @(posedge clk); down_w_wr_en<=0; end endtask

    task load_case(
        input logic [K_BEATS*LANES*8-1:0] act_pack,
        input logic [INTER*K_BEATS*LANES*4-1:0] gate_pack,
        input logic [INTER*K_BEATS*LANES*4-1:0] up_pack,
        input logic [HIDDEN*LANES*4-1:0] down_pack
    );
        begin
            ws(0,8'h38); ws(1,8'h38); ws(2,8'h38); ws(3,8'h38);
            for (int b=0;b<K_BEATS;b++) begin
                wa(b, act_pack[b*LANES*8 +: LANES*8]);
            end
            for (int r=0;r<INTER;r++) begin
                for (int b=0;b<K_BEATS;b++) begin
                    wg(r,b,gate_pack[(r*K_BEATS+b)*LANES*4 +: LANES*4]);
                    wu(r,b,up_pack[(r*K_BEATS+b)*LANES*4 +: LANES*4]);
                end
            end
            for (int r=0;r<HIDDEN;r++) begin
                wd(r, down_pack[r*LANES*4 +: LANES*4]);
            end
        end
    endtask

    task run_case(
        input string name,
        input logic [K_BEATS*LANES*8-1:0] act_pack,
        input logic [INTER*K_BEATS*LANES*4-1:0] gate_pack,
        input logic [INTER*K_BEATS*LANES*4-1:0] up_pack,
        input logic [HIDDEN*LANES*4-1:0] down_pack,
        input logic [HIDDEN*32-1:0] expected_pack
    );
        bit found_done;
        begin
            $display("--- %s ---", name);
            seen = 0;
            for (int i=0;i<HIDDEN;i++) results[i]=0;
            load_case(act_pack, gate_pack, up_pack, down_pack);
            @(posedge clk); start<=1; @(posedge clk); start<=0;
            found_done = 0;
            for (int cyc=0;cyc<1000;cyc++) begin
                @(posedge clk);
                if (result_valid) begin
                    results[result_row]=result_data;
                    seen++;
                end
                if (done && !found_done) begin
                    found_done = 1;
                    #1;
                    if (seen != HIDDEN) begin $error("%s expected %0d rows got %0d", name, HIDDEN, seen); $fatal; end
                    for (int r=0;r<HIDDEN;r++) begin
                        if (results[r] !== expected_pack[r*32 +: 32]) begin
                            $error("%s row %0d expected 0x%08h got 0x%08h", name, r, expected_pack[r*32 +: 32], results[r]);
                            $fatal;
                        end
                    end
                    $display("[ OK ] %s", name);
                    start <= 0;
                end
            end
            if (!found_done) begin
                $error("%s timeout", name);
                $fatal;
            end
            repeat (5) @(posedge clk);
        end
    endtask

    initial begin
        rst_n=0;
        reset_io();
        repeat(4) @(posedge clk); rst_n=1;
        run_case("C0", tb_ffn_golden_pkg::C0_ACT_PACK, tb_ffn_golden_pkg::C0_GATE_W_PACK,
                 tb_ffn_golden_pkg::C0_UP_W_PACK, tb_ffn_golden_pkg::C0_DOWN_W_PACK,
                 tb_ffn_golden_pkg::C0_EXPECTED_PACK);
        run_case("C1", tb_ffn_golden_pkg::C1_ACT_PACK, tb_ffn_golden_pkg::C1_GATE_W_PACK,
                 tb_ffn_golden_pkg::C1_UP_W_PACK, tb_ffn_golden_pkg::C1_DOWN_W_PACK,
                 tb_ffn_golden_pkg::C1_EXPECTED_PACK);
        $display("PASS tb_expert_ffn_engine_fp4_down_golden");
        $finish;
    end
endmodule
