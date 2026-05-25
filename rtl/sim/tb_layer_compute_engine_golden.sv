`timescale 1ns/1ps
`include "tb_layer_golden_pkg.sv"

module tb_layer_compute_engine_golden;
    localparam int HIDDEN = tb_layer_golden_pkg::HIDDEN;
    localparam int INTER  = tb_layer_golden_pkg::INTER;
    localparam int LANES  = tb_layer_golden_pkg::LANES;
    localparam int K_BEATS = tb_layer_golden_pkg::K_BEATS;

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
    logic rtr_w_wr_en;
    logic [1:0] rtr_w_wr_expert;
    logic [2:0] rtr_w_wr_idx;
    logic signed [31:0] rtr_w_wr_data;
    logic valid_in, valid_out, router_ok;
    logic signed [31:0] a0,a1,a2,a3,a4,a5,a6,a7;
    logic signed [31:0] y0,y1,y2,y3,y4,y5,y6,y7;
    bit case_done;

    layer_compute_engine dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task ws(input [1:0] a, input [7:0] d); @(posedge clk); scale_wr_en=1; scale_wr_addr=a; scale_wr_data=d; @(posedge clk); scale_wr_en=0; endtask
    task wg(input [1:0] r, input [0:0] b, input [15:0] d); @(posedge clk); gate_w_wr_en=1; gate_w_wr_row=r; gate_w_wr_beat=b; gate_w_wr_data=d; @(posedge clk); gate_w_wr_en=0; endtask
    task wu(input [1:0] r, input [0:0] b, input [15:0] d); @(posedge clk); up_w_wr_en=1; up_w_wr_row=r; up_w_wr_beat=b; up_w_wr_data=d; @(posedge clk); up_w_wr_en=0; endtask
    task wd(input [2:0] r, input [15:0] d); @(posedge clk); down_w_wr_en=1; down_w_wr_row=r; down_w_wr_beat=0; down_w_wr_data=d; @(posedge clk); down_w_wr_en=0; endtask
    task wgamma(input [2:0] i, input signed [31:0] d); @(posedge clk); gamma_wr_en=1; gamma_wr_idx=i; gamma_wr_data=d; @(posedge clk); gamma_wr_en=0; endtask
    task wrtr(input [1:0] e, input [2:0] i, input signed [31:0] d); @(posedge clk); rtr_w_wr_en=1; rtr_w_wr_expert=e; rtr_w_wr_idx=i; rtr_w_wr_data=d; @(posedge clk); rtr_w_wr_en=0; endtask

    task preload_all(
        input logic [INTER*K_BEATS*LANES*4-1:0] gate_pack,
        input logic [INTER*K_BEATS*LANES*4-1:0] up_pack,
        input logic [HIDDEN*LANES*4-1:0] down_pack,
        input logic [32*32-1:0] rtr_pack
    );
        begin
            ws(0,8'h38); ws(1,8'h38);
            for (int i=0; i<8; i++) wgamma(i[2:0], 4096);
            for (int r=0; r<INTER; r++) begin
                for (int b=0; b<K_BEATS; b++) begin
                    wg(r[1:0], b[0:0], gate_pack[(r*K_BEATS+b)*LANES*4 +: LANES*4]);
                    wu(r[1:0], b[0:0], up_pack[(r*K_BEATS+b)*LANES*4 +: LANES*4]);
                end
            end
            for (int r=0; r<HIDDEN; r++) wd(r[2:0], down_pack[r*LANES*4 +: LANES*4]);
            for (int e=0; e<4; e++) for (int i=0; i<8; i++)
                wrtr(e[1:0], i[2:0], rtr_pack[(e*8+i)*32 +: 32]);
        end
    endtask

    task run_case(
        input string name,
        input logic [HIDDEN*32-1:0] in_pack,
        input logic [INTER*K_BEATS*LANES*4-1:0] gate_pack,
        input logic [INTER*K_BEATS*LANES*4-1:0] up_pack,
        input logic [HIDDEN*LANES*4-1:0] down_pack,
        input logic [32*32-1:0] rtr_pack,
        input logic [HIDDEN*32-1:0] expected_pack
    );
        begin
            case_done = 0;
            $display("--- %s ---", name);
            preload_all(gate_pack, up_pack, down_pack, rtr_pack);
            a0 = in_pack[0*32+:32]; a1 = in_pack[1*32+:32]; a2 = in_pack[2*32+:32]; a3 = in_pack[3*32+:32];
            a4 = in_pack[4*32+:32]; a5 = in_pack[5*32+:32]; a6 = in_pack[6*32+:32]; a7 = in_pack[7*32+:32];
            @(posedge clk); #1; valid_in = 1;
            @(posedge clk); #1; valid_in = 0;
            for (int cyc=0; cyc<500 && !case_done; cyc++) begin
                @(posedge clk);
                if (valid_out) begin
                    #1;
                    if (y0 < expected_pack[0*32+:32]-4 || y0 > expected_pack[0*32+:32]+4 ||
                        y1 < expected_pack[1*32+:32]-4 || y1 > expected_pack[1*32+:32]+4 ||
                        y2 < expected_pack[2*32+:32]-4 || y2 > expected_pack[2*32+:32]+4 ||
                        y3 < expected_pack[3*32+:32]-4 || y3 > expected_pack[3*32+:32]+4 ||
                        y4 != expected_pack[4*32+:32] || y5 != expected_pack[5*32+:32] ||
                        y6 != expected_pack[6*32+:32] || y7 != expected_pack[7*32+:32]) begin
                        $error("%s mismatch", name);
                        $display("got: %0d %0d %0d %0d %0d %0d %0d %0d", y0,y1,y2,y3,y4,y5,y6,y7);
                    if (!router_ok) begin $error("%s router_ok not set", name); $fatal; end
                    $display("[ OK ] %s", name);
                    case_done = 1;
                end
            end
            if (!case_done) begin
                $error("%s timeout", name);
                $fatal;
            end
            repeat (10) @(posedge clk);
        end
    endtask

    initial begin
        rst_n=0; valid_in=0; gate_w_wr_en=0; up_w_wr_en=0; down_w_wr_en=0;
        gamma_wr_en=0; scale_wr_en=0; rtr_w_wr_en=0;
        repeat(4) @(posedge clk); rst_n=1;

        run_case("C0", tb_layer_golden_pkg::C0_IN_PACK,
                 tb_layer_golden_pkg::C0_GATE_PACK, tb_layer_golden_pkg::C0_UP_PACK,
                 tb_layer_golden_pkg::C0_DOWN_PACK, tb_layer_golden_pkg::C0_RTR_PACK,
                 tb_layer_golden_pkg::C0_EXPECTED_PACK);
        run_case("C1", tb_layer_golden_pkg::C1_IN_PACK,
                 tb_layer_golden_pkg::C1_GATE_PACK, tb_layer_golden_pkg::C1_UP_PACK,
                 tb_layer_golden_pkg::C1_DOWN_PACK, tb_layer_golden_pkg::C1_RTR_PACK,
                 tb_layer_golden_pkg::C1_EXPECTED_PACK);

        $display("PASS tb_layer_compute_engine_golden");
        $finish;
    end
endmodule
