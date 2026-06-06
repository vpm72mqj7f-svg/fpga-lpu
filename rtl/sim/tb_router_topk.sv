`timescale 1ns/1ps

module tb_router_topk;
    localparam int EXPERTS = 4;
    localparam int HIDDEN  = 8;
    localparam int TOP_K   = 2;
    localparam int EXP_W   = $clog2(EXPERTS > 1 ? EXPERTS : 2);
    localparam int HID_W   = (HIDDEN > 1 ? $clog2(HIDDEN) : 1);
    localparam int TIMEOUT = EXPERTS * (HIDDEN / 2) + TOP_K + 10;

    logic clk, rst_n;
    logic w_wr_en;
    logic [EXP_W-1:0] w_wr_expert;
    logic [HID_W-1:0] w_wr_idx;
    logic signed [31:0] w_wr_data;
    logic valid_in;
    logic [HIDDEN*32-1:0] a_flat;
    logic valid_out;
    logic [EXP_W-1:0] top_idx [TOP_K];
    logic signed [31:0] top_score [TOP_K];
    logic result_ready = 1'b1;
    int t1_done, t2_done;

    router_topk #(.EXPERTS(EXPERTS), .HIDDEN(HIDDEN), .TOP_K(TOP_K)) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task wr(input [EXP_W-1:0] e, input [HID_W-1:0] i, input signed [31:0] d);
        begin @(posedge clk); w_wr_en <= 1; w_wr_expert <= e; w_wr_idx <= i; w_wr_data <= d; @(posedge clk); w_wr_en <= 0; end
    endtask

    initial begin
        rst_n = 0; w_wr_en = 0; valid_in = 0;
        a_flat = '0;
        repeat (4) @(posedge clk); rst_n = 1;

        // Write diagonal weights: expert e has 4096 at h[e], 0 elsewhere
        for (int e = 0; e < 4; e++) begin
            for (int i = 0; i < 8; i++) begin
                wr(e[EXP_W-1:0], i[HID_W-1:0], (i == e) ? 4096 : 0);
            end
        end

        // Test 1: uniform activations, diagonal weights → top = [0,1]
        for (int d = 0; d < HIDDEN; d++) a_flat[d*32+:32] = 4096;
        @(posedge clk); #1; valid_in <= 1;
        @(posedge clk); #1; valid_in <= 0;

        t1_done = 0;
        for (int cyc = 0; cyc < TIMEOUT && !t1_done; cyc++) begin
            @(posedge clk);
            if (valid_out) begin
                #1;
                if (top_idx[0] != '0 || top_idx[1] != 'd1) begin
                    $error("T1: top-idx expected 0,1 got %0d,%0d", top_idx[0], top_idx[1]);
                    $fatal;
                end
                $display("[ OK ] T1 diagonal (scores: %0d, %0d)", top_score[0], top_score[1]);
                t1_done = 1;
            end
        end
        if (!t1_done) begin
            $error("timeout T1");
            $fatal;
        end

        // Overwrite weights for Test 2: non-uniform
        for (int e = 0; e < 4; e++) begin
            for (int i = 0; i < 8; i++) begin
                wr(e[EXP_W-1:0], i[HID_W-1:0], 0);
            end
        end
        wr('d0, 'd0, 4096);   // e0: h0 = +1.0
        wr('d1, 'd0, 8192);   // e1: h0 = +2.0
        wr('d2, 'd0, -4096);  // e2: h0 = -1.0
        wr('d3, 'd0, 2048);   // e3: h0 = +0.5

        // Activation: only a[0] = 4096, rest 0
        a_flat = '0;
        a_flat[0*32+:32] = 4096;
        @(posedge clk); #1; valid_in <= 1;
        @(posedge clk); #1; valid_in <= 0;

        t2_done = 0;
        for (int cyc = 0; cyc < TIMEOUT && !t2_done; cyc++) begin
            @(posedge clk);
            if (valid_out) begin
                #1;
                if (top_idx[0] != 'd1 || top_idx[1] != 'd0) begin
                    $error("T2: top-idx expected 1,0 got %0d,%0d", top_idx[0], top_idx[1]);
                    $fatal;
                end
                $display("[ OK ] T2 non-uniform (scores: %0d, %0d)", top_score[0], top_score[1]);
                t2_done = 1;
            end
        end
        if (!t2_done) begin
            $error("timeout T2");
            $fatal;
        end
        $display("PASS tb_router_topk");
        $finish;
    end
endmodule
