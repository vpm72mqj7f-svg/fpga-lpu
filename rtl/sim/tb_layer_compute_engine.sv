`timescale 1ns/1ps

module tb_layer_compute_engine;
    localparam int HIDDEN = 8;
    localparam int INTER  = 4;

    logic clk, rst_n;
    logic gate_w_wr_en, up_w_wr_en, down_w_wr_en;
    logic [$clog2(INTER)-1:0] gate_w_wr_row, up_w_wr_row;
    logic [$clog2(HIDDEN)-1:0] down_w_wr_row;
    logic [0:0] gate_w_wr_beat, up_w_wr_beat, down_w_wr_beat;
    logic [15:0] gate_w_wr_data, up_w_wr_data, down_w_wr_data;
    logic gamma_wr_en;
    logic [$clog2(HIDDEN)-1:0] gamma_wr_idx;
    logic signed [31:0] gamma_wr_data;
    logic scale_wr_en;
    logic [1:0] scale_wr_addr;
    logic [7:0] scale_wr_data;
    logic valid_in;
    logic [HIDDEN*32-1:0] a_flat;
    logic valid_out;
    logic router_ok;
    logic [HIDDEN*32-1:0] y_flat;

    // Convenience aliases
    wire signed [31:0] y0 = y_flat[0*32+:32];
    wire signed [31:0] y1 = y_flat[1*32+:32];
    wire signed [31:0] y2 = y_flat[2*32+:32];
    wire signed [31:0] y3 = y_flat[3*32+:32];
    wire signed [31:0] y4 = y_flat[4*32+:32];
    wire signed [31:0] y5 = y_flat[5*32+:32];
    wire signed [31:0] y6 = y_flat[6*32+:32];
    wire signed [31:0] y7 = y_flat[7*32+:32];

    // Router weight preload
    logic rtr_w_wr_en;
    logic [1:0] rtr_w_wr_expert;
    logic [$clog2(HIDDEN)-1:0] rtr_w_wr_idx;
    logic signed [31:0] rtr_w_wr_data;

    layer_compute_engine #(.HIDDEN(HIDDEN), .INTER(INTER)) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task ws(input [1:0] a, input [7:0] d); @(posedge clk); scale_wr_en<=1; scale_wr_addr<=a; scale_wr_data<=d; @(posedge clk); scale_wr_en<=0; endtask
    task wg(input [$clog2(INTER)-1:0] r, input [0:0] b, input [15:0] d); @(posedge clk); gate_w_wr_en<=1; gate_w_wr_row<=r; gate_w_wr_beat<=b; gate_w_wr_data<=d; @(posedge clk); gate_w_wr_en<=0; endtask
    task wu(input [$clog2(INTER)-1:0] r, input [0:0] b, input [15:0] d); @(posedge clk); up_w_wr_en<=1; up_w_wr_row<=r; up_w_wr_beat<=b; up_w_wr_data<=d; @(posedge clk); up_w_wr_en<=0; endtask
    task wd(input [$clog2(HIDDEN)-1:0] r, input [15:0] d); @(posedge clk); down_w_wr_en<=1; down_w_wr_row<=r; down_w_wr_beat<=0; down_w_wr_data<=d; @(posedge clk); down_w_wr_en<=0; endtask
    task wgamma(input [$clog2(HIDDEN)-1:0] i, input signed [31:0] d); @(posedge clk); gamma_wr_en<=1; gamma_wr_idx<=i; gamma_wr_data<=d; @(posedge clk); gamma_wr_en<=0; endtask
    task wrtr(input [1:0] e, input [$clog2(HIDDEN)-1:0] i, input signed [31:0] d); @(posedge clk); rtr_w_wr_en<=1; rtr_w_wr_expert<=e; rtr_w_wr_idx<=i; rtr_w_wr_data<=d; @(posedge clk); rtr_w_wr_en<=0; endtask

    initial begin
        rst_n=0; gate_w_wr_en=0; up_w_wr_en=0; down_w_wr_en=0; gamma_wr_en=0;
        rtr_w_wr_en=0;
        scale_wr_en=0; valid_in=0;
        a_flat='0;
        repeat(4) @(posedge clk); rst_n=1;

        // Preload scales
        ws(1'b0, 8'h38); ws(1'b1, 8'h38);

        // Preload gamma (identity)
        for (int i=0; i<8; i++) wgamma(i[$clog2(HIDDEN)-1:0], 4096);

        // Preload router weights (diagonal)
        for (int e = 0; e < 4; e++) begin
            for (int i = 0; i < 8; i++) begin
                wrtr(e[1:0], i[$clog2(HIDDEN)-1:0], (i == e) ? 4096 : 0);
            end
        end

        // FFN: gate/up all rows = 1.0 (fp4 +1.0 = 0x4), down identity
        for (int r=0; r<4; r++) begin
            wg(r[$clog2(INTER)-1:0], 1'b0, {4{4'h1}}); wg(r[$clog2(INTER)-1:0], 1'b1, {4{4'h0}});
            wu(r[$clog2(INTER)-1:0], 1'b0, {4{4'h1}}); wu(r[$clog2(INTER)-1:0], 1'b1, {4{4'h0}});
        end
        wd(3'd0, {4'h0,4'h0,4'h0,4'h4});
        wd(3'd1, {4'h0,4'h0,4'h4,4'h0});
        wd(3'd2, {4'h0,4'h4,4'h0,4'h0});
        wd(3'd3, {4'h4,4'h0,4'h0,4'h0});
        for (int r=4; r<8; r++) wd(r[$clog2(HIDDEN)-1:0], {4{4'h0}});

        // Input activation: all 1.0
        for (int i = 0; i < HIDDEN; i++) a_flat[i*32+:32] <= 4096;
        @(posedge clk); #1; valid_in<=1;
        @(posedge clk); #1; valid_in<=0;

        for (int cyc=0; cyc<500; cyc++) begin
            @(posedge clk);
            if (valid_out) begin
                #1;
                $display("Layer output: %0d %0d %0d %0d %0d %0d %0d %0d",
                         y0, y1, y2, y3, y4, y5, y6, y7);
                // Verify pipeline functional: non-zero outputs, no X/hang
                if (y0 == 0 && y1 == 0 && y2 == 0 && y3 == 0 &&
                    y4 == 0 && y5 == 0 && y6 == 0 && y7 == 0)
                    $error("All outputs zero");
                if ($isunknown(y0) || $isunknown(y1) || $isunknown(y2) || $isunknown(y3) ||
                    $isunknown(y4) || $isunknown(y5) || $isunknown(y6) || $isunknown(y7))
                    $error("Output contains X");
                if (!router_ok) $error("router_ok not set (expert 0 not selected)");
                $display("PASS tb_layer_compute_engine (router_ok=%0d)", router_ok);
                $finish;
            end
        end
        $error("timeout");
        $fatal;
    end
endmodule
