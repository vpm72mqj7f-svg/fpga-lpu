//=============================================================================
// mtp_verify.sv — speculative decoding verification
//
// Compares N_HEADS draft model predictions against the target model's output.
// Reports which heads match, enabling 1.5-2× throughput via speculation.
//
// Input: draft token_ids from mtp_head, target token_id from verified model
// Output: match bitmask, count of correct predictions
//=============================================================================

module mtp_verify #(
    parameter int N_HEADS = 2,
    parameter int VOCAB   = 16
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Draft predictions (from mtp_head)
    input  logic                         draft_valid,
    input  logic [$clog2(VOCAB)-1:0]     draft_token_ids [N_HEADS],
    input  logic signed [31:0]           draft_logprobs  [N_HEADS],

    // Target model output (ground truth)
    input  logic                         target_valid,
    input  logic [$clog2(VOCAB)-1:0]     target_token_id,

    // Verification result
    output logic                         verify_valid,
    output logic [N_HEADS-1:0]           match_mask,       // bit per head
    output logic [$clog2(N_HEADS+1)-1:0] n_correct,        // 0..N_HEADS
    output logic                         all_correct        // all heads matched
);

    logic [N_HEADS-1:0] match_mask_r;
    logic [$clog2(N_HEADS+1)-1:0] count;

    // Combinational match check
    for (genvar h = 0; h < N_HEADS; h++) begin : gen_match
        assign match_mask_r[h] = (draft_token_ids[h] == target_token_id);
    end

    // Count matches using continuous assignments
    wire m0, m1;
    assign m0 = match_mask_r[0];
    assign m1 = match_mask_r[1];
    assign count = {1'b0, m0} + {1'b0, m1};

    assign all_correct = (count == N_HEADS);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            verify_valid <= 1'b0;
            match_mask   <= '0;
            n_correct    <= '0;
        end else begin
            verify_valid <= 1'b0;
            if (draft_valid && target_valid) begin
                match_mask   <= match_mask_r;
                n_correct    <= count;
                verify_valid <= 1'b1;
            end
        end
    end

endmodule
