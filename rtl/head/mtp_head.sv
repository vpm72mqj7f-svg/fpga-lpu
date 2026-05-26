//=============================================================================
// mtp_head.sv — Multi-Token Prediction head (parallel lm_head projections)
//
// N_HEADS independent projection matrices share the same hidden state input.
// Each head: hidden_state × W_head → logits → argmax → (token_id, logprob)
//
// Multicycle dot-product: VOCAB cycles per inference.
// DSP: one 18×19 multiply per hidden dim per head per cycle (Agilex 7 M-Series).
// Weight storage: M20K BRAM (N_HEADS × VOCAB × HIDDEN entries × 16-bit).
//=============================================================================

(* altera_attribute = "-name DSP_BLOCK_BALANCING AUTO" *)
module mtp_head #(
    parameter int HIDDEN    = 8,
    parameter int VOCAB     = 16,
    parameter int N_HEADS   = 2,
    parameter int WEIGHT_W  = 16,    // Q12 signed
    parameter int DATA_W    = 32     // Q12 signed hidden state
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Hidden state input (flattened: HIDDEN × DATA_W bits)
    input  logic                         in_valid,
    input  logic [HIDDEN*DATA_W-1:0]     hidden_flat,
    output logic                         in_ready,

    // Weight load port
    input  logic                         wt_wr_en,
    input  logic [$clog2(N_HEADS)-1:0]   wt_head_id,
    input  logic [$clog2(VOCAB)-1:0]     wt_vocab_id,
    input  logic [$clog2(HIDDEN)-1:0]    wt_dim_id,
    input  logic signed [WEIGHT_W-1:0]   wt_wr_data,

    // Prediction output (flat packed for Icarus compatibility)
    output logic                         out_valid,
    output logic [N_HEADS*$clog2(VOCAB)-1:0] token_ids_flat,
    output logic [N_HEADS*DATA_W-1:0]        logprobs_flat
);

    localparam int VOCAB_BITS = $clog2(VOCAB);
    localparam int HEAD_BITS  = $clog2(N_HEADS);
    localparam int DIM_BITS   = $clog2(HIDDEN);

    // Weight storage
    logic signed [WEIGHT_W-1:0] weights [N_HEADS][VOCAB][HIDDEN];

    // Pipeline state
    typedef enum logic [1:0] { S_IDLE, S_COMPUTE, S_DONE } state_t;
    state_t state;

    // Registered inputs
    logic signed [DATA_W-1:0] hidden_r [HIDDEN];

    // Compute iteration
    logic [VOCAB_BITS-1:0]    vocab_idx;

    // Dot product results per head (combinational)
    logic signed [DATA_W-1:0] dot_q12 [N_HEADS];

    // Argmax tracking per head
    logic signed [DATA_W-1:0]  best_logprob [N_HEADS];
    logic [VOCAB_BITS-1:0]     best_token   [N_HEADS];

    assign in_ready = (state == S_IDLE);

    // Combinational dot product for current vocab_idx
    always_comb begin
        for (int h = 0; h < N_HEADS; h++) begin
            dot_q12[h] =
                ($signed(hidden_r[0]) * weights[h][vocab_idx][0] >>> 12) +
                ($signed(hidden_r[1]) * weights[h][vocab_idx][1] >>> 12) +
                ($signed(hidden_r[2]) * weights[h][vocab_idx][2] >>> 12) +
                ($signed(hidden_r[3]) * weights[h][vocab_idx][3] >>> 12) +
                ($signed(hidden_r[4]) * weights[h][vocab_idx][4] >>> 12) +
                ($signed(hidden_r[5]) * weights[h][vocab_idx][5] >>> 12) +
                ($signed(hidden_r[6]) * weights[h][vocab_idx][6] >>> 12) +
                ($signed(hidden_r[7]) * weights[h][vocab_idx][7] >>> 12);
        end
    end

    // Weight write
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int h = 0; h < N_HEADS; h++)
                for (int v = 0; v < VOCAB; v++)
                    for (int d = 0; d < HIDDEN; d++)
                        weights[h][v][d] <= '0;
        end else if (wt_wr_en) begin
            weights[wt_head_id][wt_vocab_id][wt_dim_id] <= wt_wr_data;
        end
    end

    // Main FSM + datapath
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            vocab_idx   <= '0;
            out_valid   <= 1'b0;
            token_ids_flat <= '0;
            logprobs_flat  <= '0;
            for (int h = 0; h < N_HEADS; h++) begin
                hidden_r[h]     <= '0;
                best_logprob[h] <= 32'sh80000000;
                best_token[h]   <= '0;
            end
        end else begin
            out_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (in_valid) begin
                        for (int d = 0; d < HIDDEN; d++)
                            hidden_r[d] <= $signed(hidden_flat[d*DATA_W +: DATA_W]);
                        vocab_idx <= '0;
                        for (int h = 0; h < N_HEADS; h++) begin
                            best_logprob[h] <= 32'sh80000000;
                            best_token[h]   <= '0;
                        end
                        state <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    for (int h = 0; h < N_HEADS; h++) begin
                        if (dot_q12[h] > best_logprob[h]) begin
                            best_logprob[h] <= dot_q12[h];
                            best_token[h]   <= vocab_idx;
                        end
                    end

                    if (vocab_idx == (VOCAB - 1)) begin
                        // Write results to flat output
                        for (int h = 0; h < N_HEADS; h++) begin
                            token_ids_flat[h*VOCAB_BITS +: VOCAB_BITS] <= best_token[h];
                            logprobs_flat[h*DATA_W     +: DATA_W]      <= best_logprob[h];
                        end
                        out_valid <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        vocab_idx <= vocab_idx + 1'b1;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
