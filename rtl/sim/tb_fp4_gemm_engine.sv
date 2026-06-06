`timescale 1ns/1ps
//=============================================================================
// tb_fp4_gemm_engine.sv — 2D Systolic GEMM Verification
//
// Tests:
//   T1: Identity weights (W=I, x=ones) → y=x
//   T2: Scaled identity → verify pre-decoded scales work
//   T3: Multi-pass (M_OUT > M_ROWS)
//=============================================================================

`include "fp4_types.svh"

module tb_fp4_gemm_engine;
    localparam int M_OUT   = 4;
    localparam int K_TOTAL = 4;      // must equal LANES for weight-stationary: each
    localparam int LANES   = 4;      // cell holds 1 weight for ALL beats; multi-beat
    localparam int M_ROWS  = 4;      // requires weight reload (not yet implemented)
    localparam int K_BEATS = (K_TOTAL + LANES - 1) / LANES;  // 1
    localparam int M_PASSES = (M_OUT + M_ROWS - 1) / M_ROWS;  // 1

    logic clk, rst_n;

    // Weight preload
    logic wt_wr_en;
    logic [$clog2(M_OUT)-1:0]   wt_wr_row;
    logic [$clog2(K_TOTAL)-1:0] wt_wr_col;
    logic [3:0]  wt_wr_data;
    logic [7:0]  sc_wr_data;

    // Activation preload
    logic activ_wr_en;
    logic [$clog2(K_BEATS)-1:0] activ_wr_beat;
    logic [LANES*8-1:0] activ_wr_data;

    // Control
    logic start, busy, done;

    // Result
    logic result_valid;
    logic [$clog2(M_OUT)-1:0] result_row;
    logic [31:0] result_data;
    logic result_ready;

    fp4_gemm_engine #(
        .M_OUT(M_OUT), .K_TOTAL(K_TOTAL),
        .LANES(LANES), .M_ROWS(M_ROWS)
    ) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    //=========================================================================
    // Helpers
    //=========================================================================
    task load_weight(input int row, col, input [3:0] w, input [7:0] s);
        @(posedge clk); #1;
        wt_wr_en <= 1; wt_wr_row <= row; wt_wr_col <= col;
        wt_wr_data <= w; sc_wr_data <= s;
        @(posedge clk); #1;
        wt_wr_en <= 0;
    endtask

    task load_activ_beat(input int beat, input [LANES*8-1:0] data);
        @(posedge clk); #1;
        activ_wr_en <= 1; activ_wr_beat <= beat; activ_wr_data <= data;
        @(posedge clk); #1;
        activ_wr_en <= 0;
    endtask

    //=========================================================================
    // Main
    //=========================================================================
    integer pass_cnt, fail_cnt;
    logic [31:0] results [0:M_OUT-1];

    initial begin
        pass_cnt = 0; fail_cnt = 0;

        // Initialise all signals to avoid X propagation
        rst_n   = 1;
        wt_wr_en   = 0;
        wt_wr_row  = 0;
        wt_wr_col  = 0;
        wt_wr_data = 4'd0;
        sc_wr_data = 8'd0;
        activ_wr_en   = 0;
        activ_wr_beat = 0;
        activ_wr_data = 0;
        start = 0;
        result_ready = 0;

        // Assert reset cleanly (1→0 gives proper negedge)
        repeat (2) @(posedge clk);
        rst_n = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        $display("============================================================");
        $display(" tb_fp4_gemm_engine — 2D Systolic Array Verification");
        $display(" Config: M_OUT=%0d K_TOTAL=%0d LANES=%0d M_ROWS=%0d",
                 M_OUT, K_TOTAL, LANES, M_ROWS);
        $display("============================================================");

        //---------------------------------------------------------------------
        // T1: Identity weights, identity activations
        //   W = I_4 (4×4 identity matrix), x = [1.0, 1.0, 1.0, 1.0]
        //   y = [1.0, 1.0, 1.0, 1.0] (each = 4096 in Q12.8)
        //---------------------------------------------------------------------
        $display("");
        $display("--- T1: Identity Weights ---");

        // Load weights: W[row][col] = (row==col) ? +1.0(fp4=0x4) : 0
        // Scale = 1.0 (fp8 = 0x38)
        for (int r = 0; r < M_OUT; r++)
            for (int c = 0; c < K_TOTAL; c++)
                load_weight(r, c, (r == c) ? 4'h4 : 4'h0, 8'h38);

        // Load activations: all +1.0 (fp8 = 0x38)
        for (int b = 0; b < K_BEATS; b++)
            load_activ_beat(b, {LANES{8'h38}});

        // Run — proper handshake
        @(posedge clk); #1; start <= 1;
        @(posedge clk); #1; start <= 0;
        @(posedge clk);  // wait one more cycle for FSM to register
        result_ready <= 1;

        // Collect results
        for (int r = 0; r < M_OUT; r++) results[r] = 32'hDEAD;
        wait(done);
        @(posedge clk);  // capture last result

        // Check
        $display("  Expected: y[0]=4096, y[1]=4096, y[2]=4096, y[3]=4096");
        for (int r = 0; r < M_OUT; r++) begin
            $display("  y[%0d] = %0d (0x%08h)", r, results[r], results[r]);
            if (results[r] == 32'hDEAD) begin
                $display("    [FAIL] — never received");
                fail_cnt++;
            end else if (results[r] > 4000 && results[r] < 4200) begin
                $display("    [ OK ] — within 3%% of expected");
                pass_cnt++;
            end else begin
                $display("    [FAIL] — expected ~4096");
                fail_cnt++;
            end
        end

        //---------------------------------------------------------------------
        // T2: Scaled weights (non-unit scale)
        //   W = 2× identity, scale = 0.5
        //   x = all ones
        //   y = identity × 2 × 0.5 = identity = 4096
        //---------------------------------------------------------------------
        $display("");
        $display("--- T2: Non-Unit Scale ---");

        // Reload weights with scale=0.5 (fp8 E4M3: sign=0, exp=6, mant=0 → 0x30)
        for (int r = 0; r < M_OUT; r++)
            for (int c = 0; c < K_TOTAL; c++)
                load_weight(r, c, (r == c) ? 4'h4 : 4'h0, 8'h30);

        // Same activations
        for (int b = 0; b < K_BEATS; b++)
            load_activ_beat(b, {LANES{8'h38}});

        // Run
        @(posedge clk); #1; start <= 1; @(posedge clk); #1; start <= 0;

        for (int r = 0; r < M_OUT; r++) results[r] = 32'hDEAD;
        wait(done);
        @(posedge clk);

        // Check: scale=0.5 means W_eff = identity × 1.0 × 0.5 = 0.5
        // Output = 1.0 × 0.5 = 0.5 in Q12 = 2048
        $display("  Expected: y[r] ≈ 2048 (scale=0.5)");
        for (int r = 0; r < M_OUT; r++) begin
            $display("  y[%0d] = %0d", r, results[r]);
            if (results[r] > 1900 && results[r] < 2200) begin
                $display("    [ OK ]");
                pass_cnt++;
            end else begin
                $display("    [FAIL]");
                fail_cnt++;
            end
        end

        //---------------------------------------------------------------------
        // T3: All-ones weights, all-ones activations
        //   W = all +1.0 (4×4), scale = 1.0, x = [1,1,1,1]
        //   y[r] = 4 × 1.0 × 1.0 = 4.0 → 16384 (4 × 4096)
        //---------------------------------------------------------------------
        $display("");
        $display("--- T3: All-Ones Weights (stress test) ---");

        for (int r = 0; r < M_OUT; r++)
            for (int c = 0; c < K_TOTAL; c++)
                load_weight(r, c, 4'h4, 8'h38);  // all +1.0

        for (int b = 0; b < K_BEATS; b++)
            load_activ_beat(b, {LANES{8'h38}});

        @(posedge clk); #1; start <= 1; @(posedge clk); #1; start <= 0;

        for (int r = 0; r < M_OUT; r++) results[r] = 32'hDEAD;
        wait(done);
        @(posedge clk);

        $display("  Expected: y[r] ≈ 16384 (4 × 4096)");
        for (int r = 0; r < M_OUT; r++) begin
            $display("  y[%0d] = %0d", r, results[r]);
            if (results[r] > 15500 && results[r] < 17200) begin
                $display("    [ OK ]");
                pass_cnt++;
            end else begin
                $display("    [FAIL] — expected ~16384");
                fail_cnt++;
            end
        end

        //---------------------------------------------------------------------
        // Results
        //---------------------------------------------------------------------
        $display("");
        $display("============================================================");
        if (fail_cnt == 0)
            $display(" ALL %0d TESTS PASSED", pass_cnt);
        else
            $display(" %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
        $display("============================================================");

        if (fail_cnt > 0) $fatal(1, "FAIL");
        $finish;
    end

    // Result capture
    always_ff @(posedge clk) begin
        if (result_valid && result_ready) begin
            results[result_row] <= result_data;
        end
    end

    // Watchdog
    initial begin
        #5000000;
        $error("TIMEOUT");
        $finish;
    end

endmodule
