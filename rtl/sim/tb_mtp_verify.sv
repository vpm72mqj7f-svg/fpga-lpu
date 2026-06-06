`timescale 1ns/1ps

//=============================================================================
// tb_mtp_verify.sv — self-checking testbench for mtp_verify speculative decode
//
// Tests:
//   T1: all heads match target (accept)
//   T2: first head mismatch, second matches (partial reject)
//   T3: all heads mismatch (reject)
//   T4: draft valid alone without target — should not fire
//   T5: target valid alone without draft — should not fire
//   T6: back-to-back verification in consecutive cycles
//
// Checking on negedge clk avoids NBA read races with the DUT's always_ff.
//=============================================================================

module tb_mtp_verify;

    localparam int N_HEADS = 2;
    localparam int VOCAB   = 16;
    localparam int VCB     = $clog2(VOCAB);

    logic clk, rst_n;
    logic draft_valid, target_valid;
    logic [N_HEADS*VCB-1:0]  draft_token_ids_flat;
    logic [N_HEADS*32-1:0]   draft_logprobs_flat;
    logic [VCB-1:0]          target_token_id;
    logic verify_valid;
    logic [N_HEADS-1:0] match_mask;
    logic [$clog2(N_HEADS+1)-1:0] n_correct;
    logic all_correct;

    mtp_verify #(.N_HEADS(N_HEADS), .VOCAB(VOCAB))
        u_verify (.*);

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count, fail_count;

    task check_result(
        input string tc_name,
        input logic [N_HEADS-1:0] exp_match,
        input logic [$clog2(N_HEADS+1)-1:0] exp_correct,
        input logic exp_all
    );
        integer local_fail;
        begin
            local_fail = 0;
            if (verify_valid !== 1'b1) begin
                $error("[FAIL] %s: verify_valid not asserted", tc_name);
                local_fail = 1;
            end else begin
                if (match_mask !== exp_match) begin
                    $error("[FAIL] %s: match_mask = %b, expected %b",
                           tc_name, match_mask, exp_match);
                    local_fail = 1;
                end
                if (n_correct !== exp_correct) begin
                    $error("[FAIL] %s: n_correct = %0d, expected %0d",
                           tc_name, n_correct, exp_correct);
                    local_fail = 1;
                end
                if (all_correct !== exp_all) begin
                    $error("[FAIL] %s: all_correct = %b, expected %b",
                           tc_name, all_correct, exp_all);
                    local_fail = 1;
                end
            end
            if (local_fail)
                fail_count = fail_count + 1;
            else begin
                $display("  [ OK ] %s", tc_name);
                pass_count = pass_count + 1;
            end
        end
    endtask

    // Pack two token IDs into flat format
    function [N_HEADS*VCB-1:0] pack_ids(input [VCB-1:0] d0, d1);
        pack_ids = {d1[VCB-1:0], d0[VCB-1:0]};
    endfunction

    //=======================================================================
    // drive_one_shot: present tokens, capture result on next negedge, check
    //
    // Timing (N = negedge, P = posedge):
    //   N0: set tokens, assert valids
    //   P1: DUT samples inputs  →  verify_valid scheduled
    //   N1: result stable, CHECK HERE, then deassert valids
    //   P2: DUT samples valids=0 →  verify_valid cleared
    //   N2: verify_valid now 0, safe to proceed
    //=======================================================================
    task drive_one_shot(
        input string                   tc_name,
        input logic [VCB-1:0]         d0, d1,
        input logic [VCB-1:0]         tgt,
        input logic [N_HEADS-1:0]     exp_mask,
        input logic [$clog2(N_HEADS+1)-1:0] exp_cnt,
        input logic                   exp_all,
        input logic                   expect_fire   // 1 = expect verify_valid
    );
        begin
            draft_token_ids_flat = pack_ids(d0, d1);
            target_token_id      = tgt;
            @(negedge clk);
            draft_valid  = 1;
            target_valid = 1;
            @(negedge clk);  // result stable (NBA settled after preceding posedge)
            if (expect_fire) begin
                check_result(tc_name, exp_mask, exp_cnt, exp_all);
            end else begin
                if (verify_valid) begin
                    $error("[FAIL] %s: verify_valid asserted unexpectedly", tc_name);
                    fail_count = fail_count + 1;
                end else begin
                    $display("  [ OK ] %s", tc_name);
                    pass_count = pass_count + 1;
                end
            end
            draft_valid  = 0;
            target_valid = 0;
            @(negedge clk);  // let verify_valid fall
        end
    endtask

    initial begin
        rst_n = 0;
        draft_valid = 0;
        target_valid = 0;
        draft_token_ids_flat = '0;
        draft_logprobs_flat  = '0;
        target_token_id = '0;
        pass_count = 0;
        fail_count = 0;

        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // ================================================================
        // T1: All heads match (accept)
        // ================================================================
        $display("T1: All heads match (accept)");
        drive_one_shot("T1: all match", 4'd5, 4'd5, 4'd5,
                       2'b11, 2'd2, 1'b1, 1'b1);

        // ================================================================
        // T2: Head 0 mismatch, Head 1 match
        // ================================================================
        $display("T2: Head 0 mismatch, Head 1 match");
        drive_one_shot("T2: partial", 4'd3, 4'd7, 4'd7,
                       2'b10, 2'd1, 1'b0, 1'b1);

        // ================================================================
        // T3: All heads mismatch (reject)
        // ================================================================
        $display("T3: All heads mismatch (reject)");
        drive_one_shot("T3: all mismatch", 4'd2, 4'd2, 4'd9,
                       2'b00, 2'd0, 1'b0, 1'b1);

        // ================================================================
        // T4: Draft valid alone — should not fire
        // ================================================================
        $display("T4: Draft valid alone (no target)");
        draft_token_ids_flat = pack_ids(4'd1, 4'd1);
        target_token_id      = 4'd1;
        @(negedge clk);
        draft_valid  = 1;
        target_valid = 0;
        @(negedge clk);
        draft_valid  = 0;
        @(negedge clk);  // check: should not fire
        if (verify_valid) begin
            $error("[FAIL] T4: fired with only draft_valid");
            fail_count = fail_count + 1;
        end else begin
            $display("  [ OK ] T4: correctly gated");
            pass_count = pass_count + 1;
        end
        @(negedge clk);

        // ================================================================
        // T5: Target valid alone — should not fire
        // ================================================================
        $display("T5: Target valid alone (no draft)");
        draft_token_ids_flat = pack_ids(4'd1, 4'd1);
        target_token_id      = 4'd1;
        @(negedge clk);
        draft_valid  = 0;
        target_valid = 1;
        @(negedge clk);
        target_valid = 0;
        @(negedge clk);
        if (verify_valid) begin
            $error("[FAIL] T5: fired with only target_valid");
            fail_count = fail_count + 1;
        end else begin
            $display("  [ OK ] T5: correctly gated");
            pass_count = pass_count + 1;
        end
        @(negedge clk);

        // ================================================================
        // T6: Back-to-back (two results in consecutive cycles)
        //===============================================================
        $display("T6: Back-to-back verification");

        // First shot
        draft_token_ids_flat = pack_ids(4'd10, 4'd10);
        target_token_id      = 4'd10;
        @(negedge clk);
        draft_valid  = 1;
        target_valid = 1;
        @(negedge clk);  // DUT sampled T6a inputs on posedge; keep valids high
        // T6a result stable. Check it.
        check_result("T6a: b2b first", 2'b11, 2'd2, 1'b1);

        // Second shot — valids still high, change tokens only
        draft_token_ids_flat = pack_ids(4'd15, 4'd0);
        target_token_id      = 4'd0;
        @(negedge clk);  // DUT sampled T6b inputs; result stable
        check_result("T6b: b2b second", 2'b10, 2'd1, 1'b0);

        draft_valid  = 0;
        target_valid = 0;
        @(negedge clk);

        // ================================================================
        // Summary
        // ================================================================
        $display("==============================");
        if (fail_count == 0)
            $display("PASS tb_mtp_verify (%0d/%0d tests)", pass_count, pass_count);
        else
            $display("FAIL tb_mtp_verify (%0d pass, %0d fail)", pass_count, fail_count);
        $finish;
    end

endmodule
