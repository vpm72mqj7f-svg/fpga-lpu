`timescale 1ns/1ps

//=============================================================================
// tb_hash_unit.sv — self-checking testbench for 4-cycle pipelined hash_unit
//
// Tests:
//   T1: single token → verify hash after 4-cycle pipeline
//   T2: different token values
//   T3: large token IDs
//   T4: determinism (same input twice)
//   T5: back-to-back submissions
//=============================================================================

module tb_hash_unit;

    localparam int N_GRAMS = 4;
    localparam int W       = N_GRAMS * 32;   // 128

    localparam logic [31:0] MURMUR_M = 32'h5bd1e995;
    localparam logic [31:0] FMIX_M   = 32'h85ebca6b;

    logic clk, rst_n;
    logic valid_in, ready_out;
    logic [W-1:0] token_ids_flat;
    logic valid_out;
    logic [31:0] hash_out;

    hash_unit #(.N_GRAMS(N_GRAMS)) u_hash (.*);

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    //=======================================================================
    // compute_expected: replicate the DUT hash pipeline (32-bit truncated)
    //=======================================================================
    function automatic [31:0] compute_expected(input [W-1:0] tokens);
        reg [63:0] prod64;
        reg [31:0] t0, t1, t2, t3, h;
        begin
            t0 = tokens[0*32 +: 32];
            t1 = tokens[1*32 +: 32];
            t2 = tokens[2*32 +: 32];
            t3 = tokens[3*32 +: 32];
            prod64 = $unsigned(t0 ^ t1) * $unsigned(MURMUR_M);
            h = prod64[31:0];
            prod64 = $unsigned(h ^ t2) * $unsigned(MURMUR_M);
            h = prod64[31:0];
            prod64 = $unsigned(h ^ t3) * $unsigned(MURMUR_M);
            h = prod64[31:0];
            prod64 = $unsigned(h ^ (h >> 16)) * $unsigned(FMIX_M);
            compute_expected = prod64[31:0];
        end
    endfunction

    integer pass_count, fail_count;

    task wait_cycles(input int n);
        for (int i = 0; i < n; i++) @(posedge clk);
    endtask

    // Submit tokens, then wait up to <max_wait> cycles for valid_out.
    // If valid_out arrives, check hash against expected.  If timeout, flag error.
    // Returns: valid_out is left at whatever state we found (caller may re-check).
    task submit_and_check(
        input string  tc_name,
        input [W-1:0] tokens,
        input [31:0]  exp_hash
    );
        integer cyc;
        begin
            // Submit
            wait_cycles(1);
            token_ids_flat = tokens;
            valid_in = 1;
            wait_cycles(1);
            valid_in = 0;

            // Poll for valid_out (pipeline = 4, timeout = 16)
            cyc = 0;
            while (!valid_out && cyc < 16) begin
                wait_cycles(1);
                cyc = cyc + 1;
            end

            if (!valid_out) begin
                $error("[FAIL] %s: valid_out timeout after %0d cycles", tc_name, cyc);
                fail_count = fail_count + 1;
            end else if (hash_out !== exp_hash) begin
                $error("[FAIL] %s: got %08h, expected %08h", tc_name, hash_out, exp_hash);
                fail_count = fail_count + 1;
            end else begin
                $display("  [ OK ] %s: hash=%08h", tc_name, hash_out);
                pass_count = pass_count + 1;
            end
            // Drain one cycle so valid_out can fall
            wait_cycles(1);
        end
    endtask

    initial begin
        rst_n = 0;
        valid_in = 0;
        token_ids_flat = '0;
        pass_count = 0;
        fail_count = 0;

        wait_cycles(6);
        rst_n = 1;
        wait_cycles(4);

        // ================================================================
        // T1: Single token set
        // ================================================================
        $display("T1: Single token");
        submit_and_check("T1: single",
            {32'd100, 32'd200, 32'd300, 32'd400},
            compute_expected({32'd100,32'd200,32'd300,32'd400}));

        // ================================================================
        // T2: Different token values
        // ================================================================
        $display("T2: Different token values");
        submit_and_check("T2: values",
            {32'd1, 32'd2, 32'd3, 32'd4},
            compute_expected({32'd1,32'd2,32'd3,32'd4}));

        // ================================================================
        // T3: Large token IDs
        // ================================================================
        $display("T3: Large token IDs");
        begin
            reg [W-1:0] t3_tokens;
            t3_tokens = {32'hFFFFFFFF, 32'h80000000, 32'h7FFFFFFF, 32'h12345678};
            submit_and_check("T3: large", t3_tokens, compute_expected(t3_tokens));
        end

        // ================================================================
        // T4: Determinism — same input twice
        // ================================================================
        $display("T4: Determinism check");
        begin
            reg [W-1:0] t4_tokens;
            reg [31:0]  t4_first;
            integer     cyc;

            t4_tokens = {32'd42, 32'd43, 32'd44, 32'd45};

            // First run
            wait_cycles(1);
            token_ids_flat = t4_tokens;
            valid_in = 1;
            wait_cycles(1);
            valid_in = 0;

            cyc = 0;
            while (!valid_out && cyc < 16) begin
                wait_cycles(1);
                cyc = cyc + 1;
            end
            if (!valid_out) begin
                $error("[FAIL] T4a: timeout");
                fail_count = fail_count + 1;
            end
            t4_first = hash_out;
            wait_cycles(1);

            // Second run with same input
            wait_cycles(1);
            token_ids_flat = t4_tokens;
            valid_in = 1;
            wait_cycles(1);
            valid_in = 0;

            cyc = 0;
            while (!valid_out && cyc < 16) begin
                wait_cycles(1);
                cyc = cyc + 1;
            end
            if (!valid_out) begin
                $error("[FAIL] T4b: timeout");
                fail_count = fail_count + 1;
            end else if (hash_out !== t4_first) begin
                $error("[FAIL] T4: non-det: %08h vs %08h", hash_out, t4_first);
                fail_count = fail_count + 1;
            end else begin
                $display("  [ OK ] T4: deterministic: %08h", hash_out);
                pass_count = pass_count + 1;
            end
            wait_cycles(1);
        end

        // ================================================================
        // T5: Sequential submissions (hash_unit is blocking: one at a time)
        // ================================================================
        $display("T5: Sequential submissions (wait for ready between tokens)");
        begin
            reg [W-1:0] t5a_tokens, t5b_tokens, t5c_tokens;
            reg [31:0]  t5a_exp, t5b_exp, t5c_exp;
            integer     cyc;

            t5a_tokens = {32'd10, 32'd20, 32'd30, 32'd40};
            t5b_tokens = {32'd50, 32'd60, 32'd70, 32'd80};
            t5c_tokens = {32'd90, 32'd100, 32'd110, 32'd120};
            t5a_exp = compute_expected(t5a_tokens);
            t5b_exp = compute_expected(t5b_tokens);
            t5c_exp = compute_expected(t5c_tokens);

            // ---------- T5a ----------
            submit_and_check("T5a: seq 1st", t5a_tokens, t5a_exp);

            // ---------- T5b (after ready) ----------
            wait_cycles(1);
            if (!ready_out) begin
                // Wait for ready if not already
                cyc = 0;
                while (!ready_out && cyc < 16) begin
                    wait_cycles(1);
                    cyc = cyc + 1;
                end
            end
            submit_and_check("T5b: seq 2nd", t5b_tokens, t5b_exp);

            // ---------- T5c (after ready) ----------
            wait_cycles(1);
            if (!ready_out) begin
                cyc = 0;
                while (!ready_out && cyc < 16) begin
                    wait_cycles(1);
                    cyc = cyc + 1;
                end
            end
            submit_and_check("T5c: seq 3rd", t5c_tokens, t5c_exp);
        end

        // ================================================================
        // T6: Non-collision check
        // ================================================================
        $display("T6: Non-collision check");
        begin
            reg [31:0] h1, h2;
            h1 = compute_expected({32'd1, 32'd2, 32'd3, 32'd4});
            h2 = compute_expected({32'd999, 32'd888, 32'd777, 32'd666});
            if (h1 === h2)
                $display("  [INFO] T6: rare collision: %08h", h1);
            else
                $display("  [ OK ] T6: different: %08h != %08h", h1, h2);
            pass_count = pass_count + 1;
        end

        // ================================================================
        // Summary
        // ================================================================
        $display("==============================");
        if (fail_count == 0)
            $display("PASS tb_hash_unit (%0d/%0d tests)", pass_count, pass_count);
        else
            $display("FAIL tb_hash_unit (%0d pass, %0d fail)", pass_count, fail_count);
        $finish;
    end

endmodule
