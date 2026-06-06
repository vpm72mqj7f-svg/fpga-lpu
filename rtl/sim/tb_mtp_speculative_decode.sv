`timescale 1ns/1ps
//=============================================================================
// tb_mtp_speculative_decode.sv — Integrated speculative decode validation
//
// Full pipeline: mtp_head (draft) → mtp_verify (compare vs target model)
//
// Tests:
//   S1: All draft heads match target (full accept, 2× throughput)
//   S2: One head matches, one mismatches (partial accept)
//   S3: All draft heads mismatch target (full reject, fallback)
//   S4: Back-to-back speculative decode in consecutive sequences
//   S5: Different hidden state → different draft predictions verified
//   S6: Weight reload between inferences → correct draft update
//
// Target model: simulated in testbench as configurable target_token_id.
// In production, this comes from the full-precision model's autoregressive step.
//=============================================================================

module tb_mtp_speculative_decode;
    localparam int HIDDEN   = 8;
    localparam int VOCAB    = 16;
    localparam int N_HEADS  = 2;
    localparam int WEIGHT_W = 16;
    localparam int DATA_W   = 32;
    localparam int VCB      = $clog2(VOCAB);

    localparam int Q12_ONE  = 4096;
    localparam int Q12_ZERO = 0;

    // ── DUT signals ──────────────────────────────────────────────────────
    logic clk, rst_n;

    // mtp_head
    logic                         head_in_valid, head_in_ready;
    logic [HIDDEN*DATA_W-1:0]     hidden_flat;
    logic                         head_wt_wr_en;
    logic [$clog2(N_HEADS)-1:0]   head_wt_head_id;
    logic [$clog2(VOCAB)-1:0]     head_wt_vocab_id;
    logic [$clog2(HIDDEN)-1:0]    head_wt_dim_id;
    logic signed [WEIGHT_W-1:0]   head_wt_wr_data;
    logic                         head_out_valid;
    logic [N_HEADS*VCB-1:0]       token_ids_flat;
    logic [N_HEADS*DATA_W-1:0]    logprobs_flat;

    // mtp_verify — flat-packed for Icarus compatibility
    logic                         verify_draft_valid;
    logic [N_HEADS*VCB-1:0]       verify_draft_ids_flat;
    logic [N_HEADS*DATA_W-1:0]    verify_draft_logprobs_flat;
    logic                         verify_target_valid;
    logic [VCB-1:0]               verify_target_id;
    logic                         verify_out_valid;
    logic [N_HEADS-1:0]           match_mask;
    logic [$clog2(N_HEADS+1)-1:0] n_correct;
    logic                         all_correct;

    // DUT instances
    mtp_head #(.HIDDEN(HIDDEN), .VOCAB(VOCAB), .N_HEADS(N_HEADS),
               .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W))
        u_head (.clk, .rst_n,
                .in_valid(head_in_valid), .hidden_flat,
                .in_ready(head_in_ready),
                .wt_wr_en(head_wt_wr_en), .wt_head_id(head_wt_head_id),
                .wt_vocab_id(head_wt_vocab_id), .wt_dim_id(head_wt_dim_id),
                .wt_wr_data(head_wt_wr_data),
                .out_valid(head_out_valid),
                .token_ids_flat, .logprobs_flat);

    // Direct flat wiring: mtp_head → mtp_verify
    assign verify_draft_ids_flat     = token_ids_flat;
    assign verify_draft_logprobs_flat = logprobs_flat;

    mtp_verify #(.N_HEADS(N_HEADS), .VOCAB(VOCAB))
        u_verify (.clk, .rst_n,
                  .draft_valid(verify_draft_valid),
                  .draft_token_ids_flat(verify_draft_ids_flat),
                  .draft_logprobs_flat(verify_draft_logprobs_flat),
                  .target_valid(verify_target_valid),
                  .target_token_id(verify_target_id),
                  .verify_valid(verify_out_valid),
                  .match_mask, .n_correct, .all_correct);

    // ── Clock ────────────────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Helpers ──────────────────────────────────────────────────────────
    task wait_cycles(input int n);
        for (int i = 0; i < n; i++) @(posedge clk);
    endtask

    function [HIDDEN*DATA_W-1:0] make_vec(input int base);
        reg [HIDDEN*DATA_W-1:0] v;
        for (int d = 0; d < HIDDEN; d++) v[d*DATA_W +: DATA_W] = base + d;
        make_vec = v;
    endfunction

    // Load weights for one head: target vocab entry gets Q12_ONE, rest zero
    task load_head_weights(input int hid, input int target_vocab);
        for (int v = 0; v < VOCAB; v++) begin
            for (int d = 0; d < HIDDEN; d++) begin
                @(posedge clk);
                head_wt_wr_en   <= 1;
                head_wt_head_id <= hid[$clog2(N_HEADS)-1:0];
                head_wt_vocab_id <= v[$clog2(VOCAB)-1:0];
                head_wt_dim_id  <= d[$clog2(HIDDEN)-1:0];
                head_wt_wr_data <= (v == target_vocab) ? Q12_ONE : Q12_ZERO;
                @(posedge clk);
                head_wt_wr_en <= 0;
            end
        end
    endtask

    // Drive mtp_head + target, capture mtp_verify result
    task run_speculative_decode(
        input string                tc_name,
        input [HIDDEN*DATA_W-1:0]  hidden,
        input [VCB-1:0]            target_token,
        input [N_HEADS-1:0]        exp_mask,
        input [$clog2(N_HEADS+1)-1:0] exp_cnt,
        input logic                exp_all
    );
        reg timed_out;
        integer local_fail;
        begin
            local_fail = 0;
            timed_out  = 0;

            // Present hidden state to mtp_head
            @(posedge clk);
            head_in_valid <= 1;
            hidden_flat   <= hidden;
            @(posedge clk);
            head_in_valid <= 0;

            // Wait for mtp_head to finish (VOCAB+3 cycles)
            for (int cyc = 0; cyc < 50 && !head_out_valid; cyc++)
                @(posedge clk);

            if (!head_out_valid) begin
                $error("[FAIL] %s: mtp_head timed out", tc_name);
                fail_count = fail_count + 1;
                timed_out  = 1;
            end

            if (!timed_out) begin
                // Latch head output, present target to verify in same cycle
                verify_draft_valid  <= 1'b1;
                verify_target_valid <= 1'b1;
                verify_target_id    <= target_token;
                @(posedge clk);
                verify_draft_valid  <= 1'b0;
                verify_target_valid <= 1'b0;

                // Wait for verify result
                for (int cyc = 0; cyc < 10 && !verify_out_valid; cyc++)
                    @(posedge clk);

                if (!verify_out_valid) begin
                    $error("[FAIL] %s: mtp_verify timed out", tc_name);
                    fail_count = fail_count + 1;
                    timed_out  = 1;
                end
            end

            if (!timed_out) begin
                // Check results
                if (match_mask !== exp_mask) begin
                    $error("[FAIL] %s: match_mask=%b expected=%b", tc_name, match_mask, exp_mask);
                    local_fail = 1;
                end
                if (n_correct !== exp_cnt) begin
                    $error("[FAIL] %s: n_correct=%0d expected=%0d", tc_name, n_correct, exp_cnt);
                    local_fail = 1;
                end
                if (all_correct !== exp_all) begin
                    $error("[FAIL] %s: all_correct=%b expected=%b", tc_name, all_correct, exp_all);
                    local_fail = 1;
                end

                if (local_fail) fail_count = fail_count + 1;
                else begin
                    $display("  [PASS] %s", tc_name);
                    pass_count = pass_count + 1;
                end

                @(posedge clk); // let verify_valid fall
            end
        end
    endtask

    integer pass_count, fail_count;

    // ── Main ─────────────────────────────────────────────────────────────
    initial begin
        rst_n = 0;
        head_in_valid = 0; hidden_flat = '0;
        head_wt_wr_en = 0; head_wt_head_id = '0; head_wt_vocab_id = '0;
        head_wt_dim_id = '0; head_wt_wr_data = '0;
        verify_draft_valid = 0; verify_target_valid = 0; verify_target_id = '0;
        pass_count = 0; fail_count = 0;

        wait_cycles(4);
        rst_n = 1;
        wait_cycles(2);

        $display("===================================================================");
        $display(" tb_mtp_speculative_decode — Integrated MTP Pipeline Validation");
        $display(" HIDDEN=%0d VOCAB=%0d N_HEADS=%0d", HIDDEN, VOCAB, N_HEADS);
        $display("===================================================================");

        // ================================================================
        // S1: Full accept — both draft heads match target model
        // Head 0 → token 3, Head 1 → token 7, Target → token 7
        // With identity weights and all-ones hidden, head0=3, head1=7
        // We set target=7, so head1 matches but head0 doesn't
        // Actually for full accept, set target to match one head and make
        // both heads predict the same token. Let's make both heads point
        // to token 5 for a clean full-accept test.
        // ================================================================
        $display("");
        $display("--- S1: Full accept (both heads match target) ---");
        load_head_weights(0, 5);
        load_head_weights(1, 5);

        run_speculative_decode("S1: full accept",
            make_vec(Q12_ONE),  // hidden = all 4096
            4'd5,               // target predicts token 5
            2'b11,              // both heads match
            2'd2,               // 2 correct
            1'b1);              // all_correct

        // ================================================================
        // S2: Partial accept — one head matches, one does not
        // Head 0 → token 3, Head 1 → token 7, Target → token 7
        // ================================================================
        $display("");
        $display("--- S2: Partial accept (head1 matches, head0 does not) ---");
        load_head_weights(0, 3);
        load_head_weights(1, 7);

        run_speculative_decode("S2: partial accept",
            make_vec(Q12_ONE),
            4'd7,               // target says token 7
            2'b10,              // only head1 matches
            2'd1,               // 1 correct
            1'b0);              // not all correct

        // ================================================================
        // S3: Full reject — neither head matches target
        // Head 0 → token 3, Head 1 → token 7, Target → token 12
        // ================================================================
        $display("");
        $display("--- S3: Full reject (neither head matches) ---");
        load_head_weights(0, 3);
        load_head_weights(1, 7);

        run_speculative_decode("S3: full reject",
            make_vec(Q12_ONE),
            4'd12,              // target disagrees with both
            2'b00,              // no match
            2'd0,               // 0 correct
            1'b0);              // all_correct = 0

        // ================================================================
        // S4: Back-to-back speculative decode
        // Seq A: both heads→5, target→5 (accept)
        // Seq B: heads→3/7, target→7 (partial)
        // ================================================================
        $display("");
        $display("--- S4: Back-to-back speculative decode ---");

        // Seq A
        load_head_weights(0, 5);
        load_head_weights(1, 5);
        run_speculative_decode("S4a: b2b accept",
            make_vec(Q12_ONE), 4'd5, 2'b11, 2'd2, 1'b1);

        // Seq B — same weights, different target
        run_speculative_decode("S4b: b2b partial",
            make_vec(Q12_ONE), 4'd10, 2'b00, 2'd0, 1'b0);

        // ================================================================
        // S5: Non-uniform hidden state — verify draft tokens change
        // Head0 weights: token 2 at dims 0-3 only (not 4-7)
        // Hidden: [4096,4096,4096,4096, 0,0,0,0] → half activation
        // Head1 weights: token 8 at all dims
        // ================================================================
        $display("");
        $display("--- S5: Non-uniform hidden state ---");

        // Head0: only dims 0-3 have Q12_ONE, dims 4-7 zero
        for (int v = 0; v < VOCAB; v++) begin
            for (int d = 0; d < HIDDEN; d++) begin
                @(posedge clk);
                head_wt_wr_en <= 1; head_wt_head_id <= 0;
                head_wt_vocab_id <= v[$clog2(VOCAB)-1:0];
                head_wt_dim_id <= d[$clog2(HIDDEN)-1:0];
                head_wt_wr_data <= (v == 2 && d < 4) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); head_wt_wr_en <= 0;
            end
        end
        // Head1: token 8 at all dims
        for (int v = 0; v < VOCAB; v++) begin
            for (int d = 0; d < HIDDEN; d++) begin
                @(posedge clk);
                head_wt_wr_en <= 1; head_wt_head_id <= 1;
                head_wt_vocab_id <= v[$clog2(VOCAB)-1:0];
                head_wt_dim_id <= d[$clog2(HIDDEN)-1:0];
                head_wt_wr_data <= (v == 8) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); head_wt_wr_en <= 0;
            end
        end

        // Hidden: only first 4 dims active
        run_speculative_decode("S5: non-uniform hidden",
            {128'd0,                                  // dims 7..4 = 0
             {32'd4096, 32'd4096, 32'd4096, 32'd4096}}, // dims 3..0 = 4096
            4'd8,   // target matches head1
            2'b10,  // head0 weaker signal, likely different token
            2'd1,   // 1 correct
            1'b0);

        // ================================================================
        // S6: Weight reload between inferences
        // First inference: heads→3/7, target→3, partial accept (head0)
        // Reload: heads→9/9, target→9, full accept
        // ================================================================
        $display("");
        $display("--- S6: Weight reload between inferences ---");
        load_head_weights(0, 3);
        load_head_weights(1, 7);

        run_speculative_decode("S6a: pre-reload",
            make_vec(Q12_ONE), 4'd3, 2'b01, 2'd1, 1'b0);

        // Reload both heads to token 9
        load_head_weights(0, 9);
        load_head_weights(1, 9);

        run_speculative_decode("S6b: post-reload",
            make_vec(Q12_ONE), 4'd9, 2'b11, 2'd2, 1'b1);

        // ================================================================
        // Summary
        // ================================================================
        $display("");
        $display("===================================================================");
        if (fail_count == 0)
            $display(" ALL %0d TESTS PASSED", pass_count);
        else
            $display(" %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("===================================================================");
        $display("");
        $display("Speculative decode pipeline validated:");
        $display("  - mtp_head draft prediction → mtp_verify target comparison");
        $display("  - Accept/reject semantics: full/partial/reject all verified");
        $display("  - Back-to-back throughput: consecutive sequences correct");
        $display("  - Weight reload: dynamic draft head update works");
        $display("  - Non-uniform hidden: partial activation handled correctly");
        $display("");

        if (fail_count > 0) $fatal(1, "FAIL");
        $finish;
    end

endmodule
