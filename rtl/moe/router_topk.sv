//=============================================================================
// router_topk.sv — MoE Router Top-K (EXPERTS=4, HIDDEN=8 bring-up)
//
// Pipeline: 3 stages
//   Stage 1: Latch activations, 32 parallel 32b×32b multiplies → pairwise sums
//   Stage 2: Adder-tree reduction → 4 expert scores
//   Stage 3: Top-2 search → output
//
// Weight storage: 2D array w[EXPERTS][HIDDEN] (replaces flat register set).
//=============================================================================

module router_topk #(
    parameter int EXPERTS = 4,
    parameter int HIDDEN  = 8
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         w_wr_en,
    input  logic [$clog2(EXPERTS)-1:0]   w_wr_expert,
    input  logic [$clog2(HIDDEN)-1:0]    w_wr_idx,
    input  logic signed [31:0]           w_wr_data,

    input  logic                         valid_in,
    input  logic signed [31:0]           a0, a1, a2, a3, a4, a5, a6, a7,

    output logic                         valid_out,
    input  logic                         result_ready,
    output logic [$clog2(EXPERTS)-1:0]   top0_idx, top1_idx,
    output logic signed [31:0]           top0_score, top1_score
);

    //=========================================================================
    // Weight storage — 2D array replacing 32 individual registers
    //=========================================================================
    logic signed [31:0] w [EXPERTS-1:0][HIDDEN-1:0];

    always_ff @(posedge clk) begin
        if (w_wr_en) w[w_wr_expert][w_wr_idx] <= w_wr_data;
    end

    //=========================================================================
    // Pipeline registers
    //=========================================================================
    typedef enum logic [1:0] { S_IDLE, S_COMPUTE, S_REDUCE, S_OUTPUT } state_t;
    state_t state;

    // Stage 1: latched activations + partial products
    logic signed [31:0]       s1_a [HIDDEN-1:0];
    logic signed [63:0]       s1_pair [EXPERTS-1:0][(HIDDEN/2)-1:0];
    logic                     s1_active;

    // Stage 2: reduced scores per expert
    logic signed [63:0]       s2_score [EXPERTS-1:0];
    logic                     s2_active;

    //=========================================================================
    // FSM + Pipeline
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            valid_out  <= 1'b0;
            s1_active  <= 1'b0;
            s2_active  <= 1'b0;
            top0_idx   <= '0;
            top1_idx   <= '0;
            top0_score <= '0;
            top1_score <= '0;
            for (int i = 0; i < HIDDEN; i++) s1_a[i] <= '0;
            for (int e = 0; e < EXPERTS; e++) begin
                s2_score[e] <= '0;
                for (int p = 0; p < HIDDEN/2; p++) s1_pair[e][p] <= '0;
            end
        end else begin
            valid_out <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (valid_in) begin
                        s1_a[0] <= a0; s1_a[1] <= a1; s1_a[2] <= a2; s1_a[3] <= a3;
                        s1_a[4] <= a4; s1_a[5] <= a5; s1_a[6] <= a6; s1_a[7] <= a7;
                        s1_active <= 1'b1;
                        state <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    // Stage 1→2: pairwise products → partial sums
                    // 32 DSP multiplies, then pairwise additions
                    for (int e = 0; e < EXPERTS; e++) begin
                        for (int p = 0; p < HIDDEN/2; p++) begin
                            s1_pair[e][p] <=
                                $signed(s1_a[2*p]) * $signed(w[e][2*p]) +
                                $signed(s1_a[2*p+1]) * $signed(w[e][2*p+1]);
                        end
                    end
                    s1_active <= 1'b0;
                    s2_active <= 1'b1;
                    state <= S_REDUCE;
                end

                S_REDUCE: begin
                    // Stage 2→3: adder-tree reduction to per-expert scores
                    for (int e = 0; e < EXPERTS; e++) begin
                        s2_score[e] <=
                            (s1_pair[e][0] + s1_pair[e][1]) +
                            (s1_pair[e][2] + s1_pair[e][3]);
                    end
                    s2_active <= 1'b0;
                    state <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    // Top-2 search over EXPERTS scores
                    logic signed [63:0] best, second;
                    logic [$clog2(EXPERTS)-1:0] bi, si;

                    best   = s2_score[0];
                    bi     = '0;
                    for (int e = 1; e < EXPERTS; e++) begin
                        if (s2_score[e] > best) begin
                            best = s2_score[e];
                            bi   = e[$clog2(EXPERTS)-1:0];
                        end
                    end

                    second = {1'b1, {63{1'b0}}};  // min signed 64b
                    si     = '0;
                    for (int e = 0; e < EXPERTS; e++) begin
                        if (e[$clog2(EXPERTS)-1:0] != bi) begin
                            if (s2_score[e] > second) begin
                                second = s2_score[e];
                                si     = e[$clog2(EXPERTS)-1:0];
                            end
                        end
                    end

                    top0_idx   <= bi;
                    top1_idx   <= si;
                    top0_score <= best[31:0];
                    top1_score <= second[31:0];
                    valid_out  <= 1'b1;
                    state      <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
