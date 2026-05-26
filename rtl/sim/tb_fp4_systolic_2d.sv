`timescale 1ns/1ps

`include "fp4_types.svh"

module tb_fp4_systolic_2d;
    localparam int LANES = 4;
    localparam int M_ROWS = 4;
    localparam int ACCUM_WIDTH = 32;

    logic clk, rst_n;

    logic wt_wr_en;
    logic [$clog2(M_ROWS)-1:0] wt_wr_row;
    logic [$clog2(LANES)-1:0]  wt_wr_col;
    logic [3:0]  wt_wr_data;
    logic [11:0] sc_wr_data;

    logic valid_in;
    logic [LANES*8-1:0] activ_flat;
    logic accum_clr, reduce_start, reduce_done;
    logic [M_ROWS*ACCUM_WIDTH-1:0] result_flat;

    fp4_systolic_2d #(.LANES(LANES), .M_ROWS(M_ROWS)) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task load_cell(input int r, c, input [3:0] w, input [11:0] s);
        @(posedge clk); #1;
        wt_wr_en=1; wt_wr_row=r; wt_wr_col=c; wt_wr_data=w; sc_wr_data=s;
        @(posedge clk); #1;
        wt_wr_en=0;
    endtask

    integer pass, fail;

    initial begin
        pass = 0; fail = 0;
        rst_n = 0; wt_wr_en = 0; valid_in = 0;
        accum_clr = 0; reduce_start = 0;

        repeat (5) @(posedge clk); rst_n = 1;
        repeat (2) @(posedge clk);

        $display("============================================================");
        $display(" tb_fp4_systolic_2d — Direct Array Test");
        $display(" LANES=%0d M_ROWS=%0d", LANES, M_ROWS);
        $display("============================================================");

        //-----------------------------------------------------------------
        // T1: Identity test
        //   4×4 identity weights (fp4 +1.0 = 0x4)
        //   scale = 1.0 → pre-decoded: fp8_to_scaled12(8'h38) ≈ 256
        //   activation = [1.0, 1.0, 1.0, 1.0] × 1 beat
        //   expected: each row = 1.0*1.0*1.0 × 256 >> 8 = 1.0 = 4096
        //-----------------------------------------------------------------
        $display("");
        $display("--- T1: 4x4 Identity ---");

        // Load identity weights with pre-decoded scale=256 (fp8 1.0)
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                load_cell(r, c, (r==c) ? 4'h4 : 4'h0, 12'd256);

        // Feed activation: all 1.0 (hold for 2 cycles for pipeline)
        @(posedge clk); #1;
        valid_in = 1; activ_flat = {8'h38, 8'h38, 8'h38, 8'h38};
        @(posedge clk); #1;
        // hold
        @(posedge clk); #1;
        valid_in = 0;

        // Wait for MAC pipeline (4 stages + 1 activation pipeline = 5 cycles)
        repeat (6) @(posedge clk);

        // Reduce
        @(posedge clk); #1;
        reduce_start = 1;
        @(posedge clk); #1;
        reduce_start = 0;

        // Wait for reduce_done
        while (!reduce_done) @(posedge clk);

        // Check results
        $display("  Results:");
        for (int r = 0; r < M_ROWS; r++) begin
            logic [31:0] val = result_flat[r*32 +: 32];
            $display("    row %0d: %0d (0x%08h)", r, $signed(val), val);
            if (val > 3500 && val < 4700) begin pass++; $display("      [ OK ]"); end
            else begin fail++; $display("      [FAIL] expected ~4096"); end
        end

        //-----------------------------------------------------------------
        // T2: All-ones, 2 beats (K=8 via K_BEATS=2)
        //   All weights = +1.0, scale=1.0
        //   Beat 0: [1,1,1,1], Beat 1: [1,1,1,1]
        //   Total: 8 × 1.0 × 1.0 × 1.0 = 8.0 = 32768
        //-----------------------------------------------------------------
        $display("");
        $display("--- T2: All-Ones, 2 Beats (K=8 equivalent) ---");

        // Load all-ones weights
        for (int r = 0; r < M_ROWS; r++)
            for (int c = 0; c < LANES; c++)
                load_cell(r, c, 4'h4, 12'd256);

        // Clear accumulators
        @(posedge clk); #1; accum_clr = 1;
        @(posedge clk); #1; accum_clr = 0;

        // Beat 0
        @(posedge clk); #1;
        valid_in = 1; activ_flat = {8'h38, 8'h38, 8'h38, 8'h38};
        @(posedge clk); #1;
        // Beat 1
        activ_flat = {8'h38, 8'h38, 8'h38, 8'h38};
        @(posedge clk); #1;
        valid_in = 0;

        // Drain
        repeat (7) @(posedge clk);

        // Reduce
        @(posedge clk); #1;
        reduce_start = 1;
        @(posedge clk); #1;
        reduce_start = 0;
        while (!reduce_done) @(posedge clk);

        // Check: 8 × 4096 = 32768
        for (int r = 0; r < M_ROWS; r++) begin
            logic [31:0] val = result_flat[r*32 +: 32];
            $display("    row %0d: %0d", r, $signed(val));
            if (val > 32000 && val < 33600) begin pass++; $display("      [ OK ]"); end
            else begin fail++; $display("      [FAIL] expected ~32768"); end
        end

        //-----------------------------------------------------------------
        // Results
        //-----------------------------------------------------------------
        $display("");
        $display("============================================================");
        if (fail == 0)
            $display(" ALL %0d TESTS PASSED", pass);
        else
            $display(" %0d PASSED, %0d FAILED", pass, fail);
        $display("============================================================");
        if (fail > 0) $fatal(1, "FAIL");
        $finish;
    end

    initial begin #5000000; $error("TIMEOUT"); $finish; end
endmodule
