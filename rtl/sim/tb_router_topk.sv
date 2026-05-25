`timescale 1ns/1ps

module tb_router_topk;
    logic clk, rst_n;
    logic w_wr_en;
    logic [1:0] w_wr_expert;
    logic [2:0] w_wr_idx;
    logic signed [31:0] w_wr_data;
    logic valid_in;
    logic signed [31:0] a0,a1,a2,a3,a4,a5,a6,a7;
    logic valid_out;
    logic [1:0] top0_idx, top1_idx;
    logic signed [31:0] top0_score, top1_score;
    logic result_ready = 1'b1;
    int t1_done, t2_done;

    router_topk dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task wr(input [1:0] e, input [2:0] i, input signed [31:0] d);
        begin @(posedge clk); w_wr_en = 1; w_wr_expert = e; w_wr_idx = i; w_wr_data = d; @(posedge clk); w_wr_en = 0; end
    endtask

    initial begin
        rst_n = 0; w_wr_en = 0; valid_in = 0;
        {a0,a1,a2,a3,a4,a5,a6,a7} = '0;
        repeat (4) @(posedge clk); rst_n = 1;

        // Write diagonal weights: expert e has 4096 at h[e], 0 elsewhere
        for (int e = 0; e < 4; e++) begin
            for (int i = 0; i < 8; i++) begin
                wr(e[1:0], i[2:0], (i == e) ? 4096 : 0);
            end
        end

        // Test 1: uniform activations, diagonal weights → top = [0,1]
        a0 = 4096; a1 = 4096; a2 = 4096; a3 = 4096;
        a4 = 4096; a5 = 4096; a6 = 4096; a7 = 4096;
        @(posedge clk); #1; valid_in = 1;
        @(posedge clk); #1; valid_in = 0;

        t1_done = 0;
        for (int cyc = 0; cyc < 10 && !t1_done; cyc++) begin
            @(posedge clk);
            if (valid_out) begin
                #1;
                if (top0_idx != 2'd0 || top1_idx != 2'd1) begin
                    $error("T1: top-idx expected 0,1 got %0d,%0d", top0_idx, top1_idx);
                    $fatal;
                end
                $display("[ OK ] T1 diagonal (scores: %0d, %0d)", top0_score, top1_score);
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
                wr(e[1:0], i[2:0], 0);
            end
        end
        wr(2'd0, 3'd0, 4096);   // e0: h0 = +1.0
        wr(2'd1, 3'd0, 8192);   // e1: h0 = +2.0
        wr(2'd2, 3'd0, -4096);  // e2: h0 = -1.0
        wr(2'd3, 3'd0, 2048);   // e3: h0 = +0.5

        // Activation: only a0 = 4096, rest 0
        a0 = 4096; a1 = 0; a2 = 0; a3 = 0; a4 = 0; a5 = 0; a6 = 0; a7 = 0;
        @(posedge clk); #1; valid_in = 1;
        @(posedge clk); #1; valid_in = 0;

        t2_done = 0;
        for (int cyc = 0; cyc < 10 && !t2_done; cyc++) begin
            @(posedge clk);
            if (valid_out) begin
                #1;
                if (top0_idx != 2'd1 || top1_idx != 2'd0) begin
                    $error("T2: top-idx expected 1,0 got %0d,%0d", top0_idx, top1_idx);
                    $fatal;
                end
                $display("[ OK ] T2 non-uniform (scores: %0d, %0d)", top0_score, top1_score);
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
