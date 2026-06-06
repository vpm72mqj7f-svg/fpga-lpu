`timescale 1ns/1ps

// Quick M_OUT=4 test to reproduce gate/up last-row-zero bug
// Exact same params as expert_ffn_engine_fp4_down's gate engine
module tb_le_m4_quick;
    localparam int M_OUT = 4;
    localparam int K_TOTAL = 8;
    localparam int LANES = 4;
    localparam int GROUP_SIZE = 4;
    localparam int NUM_GROUPS = 4;
    localparam int K_BEATS = (K_TOTAL + LANES - 1) / LANES;
    localparam int ADDR_W = $clog2(NUM_GROUPS > 1 ? NUM_GROUPS : 2);
    localparam int BEAT_W = $clog2(K_BEATS > 1 ? K_BEATS : 2);

    logic clk, rst_n;
    logic weight_wr_en;
    logic [$clog2(M_OUT)-1:0] weight_wr_row;
    logic [BEAT_W-1:0] weight_wr_beat;
    logic [LANES*4-1:0] weight_wr_data;
    logic activ_wr_en;
    logic [BEAT_W-1:0] activ_wr_beat;
    logic [LANES*8-1:0] activ_wr_data;
    logic scale_wr_en;
    logic [ADDR_W-1:0] scale_wr_addr;
    logic [7:0] scale_wr_data;
    logic start;
    logic busy, done;
    logic result_valid;
    logic [$clog2(M_OUT)-1:0] result_row;
    logic [31:0] result_data;
    logic result_ready = 1'b1;

    int seen_rows;
    int fail;
    int i, cyc;
    logic [31:0] results [M_OUT];

    fp4_linear_engine #(
        .M_OUT(M_OUT), .K_TOTAL(K_TOTAL), .LANES(LANES),
        .GROUP_SIZE(GROUP_SIZE), .NUM_GROUPS(NUM_GROUPS), .ADDR_WIDTH(ADDR_W)
    ) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 0;
        weight_wr_en = 0; weight_wr_row = 0; weight_wr_beat = 0; weight_wr_data = '0;
        activ_wr_en = 0; activ_wr_beat = 0; activ_wr_data = '0;
        scale_wr_en = 0; scale_wr_addr = 0; scale_wr_data = 0;
        start = 0; seen_rows = 0; fail = 0;
        for (i = 0; i < M_OUT; i = i + 1) results[i] = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // scale=1.0 (FP8=0x38) for all 4 groups
        for (i = 0; i < NUM_GROUPS; i = i + 1) begin
            @(posedge clk); scale_wr_en=1; scale_wr_addr=i[ADDR_W-1:0]; scale_wr_data=8'h38; @(posedge clk); scale_wr_en=0;
        end

        // activ: all 1.0 (FP8=0x38) in both beats
        @(posedge clk); activ_wr_en=1; activ_wr_beat=0; activ_wr_data={4{8'h38}}; @(posedge clk); activ_wr_en=0;
        @(posedge clk); activ_wr_en=1; activ_wr_beat=1; activ_wr_data={4{8'h38}}; @(posedge clk); activ_wr_en=0;

        // Identity-like weights: beat0 all 1.0, beat1 all 0 (like e2e test)
        // All 4 rows have same weights
        for (i = 0; i < M_OUT; i = i + 1) begin
            @(posedge clk); weight_wr_en=1; weight_wr_row=i[1:0]; weight_wr_beat=0;
            weight_wr_data={4{4'h4}}; @(posedge clk); weight_wr_en=0;
            @(posedge clk); weight_wr_en=1; weight_wr_row=i[1:0]; weight_wr_beat=1;
            weight_wr_data=16'h0; @(posedge clk); weight_wr_en=0;
        end

        // Start compute
        @(posedge clk); start <= 1;
        @(posedge clk); start <= 0;

        // Wait for results
        for (cyc = 0; cyc < 500; cyc = cyc + 1) begin
            @(posedge clk);
            if (result_valid) begin
                results[result_row] = result_data;
                seen_rows = seen_rows + 1;
                $display("row %0d result=0x%08h (%0d)", result_row, result_data, result_data);
            end
            if (done) begin
                #1;
                $display("Done: seen_rows=%0d", seen_rows);
                for (i = 0; i < M_OUT; i = i + 1)
                    $display("  results[%0d] = 0x%08h (%0d)", i, results[i], results[i]);
                fail = 0;
                for (i = 0; i < M_OUT; i = i + 1) begin
                    if (results[i] == 0) begin
                        $error("FAIL: row %0d is zero!", i);
                        fail = fail + 1;
                    end
                end
                if (fail == 0) $display("PASS: all %0d rows non-zero", M_OUT);
                else $display("FAIL: %0d rows are zero", fail);
                $finish;
            end
        end
        $error("timeout waiting for done");
        $finish;
    end
endmodule
