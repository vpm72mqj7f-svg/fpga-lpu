//=============================================================================
// router_topk.sv — MoE Router Top-K
//
// Pipeline: time-multiplexed dot-product with altera_mult_add DSP IP.
// Iterates over expert × dimension-pair combinations sequentially.
// Top-K selection is multi-cycle (K iterations, each sweeping all experts).
// Bring-up: EXPERTS=4, HIDDEN=8 → 16+2=18 cycles.
// Production: EXPERTS=384, HIDDEN=7168 → ~1.38M+6=1.38M cycles.
//   NOTE: Production comparison-tree timing needs pipeline (TBD).
//=============================================================================

`include "lpu_config.svh"

module router_topk #(
    parameter int EXPERTS = lpu_config_pkg::LPU_NUM_EXPERTS,
    parameter int HIDDEN  = lpu_config_pkg::LPU_HIDDEN,
    parameter int TOP_K   = lpu_config_pkg::LPU_TOP_K
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         w_wr_en,
    input  logic [$clog2(EXPERTS)-1:0]   w_wr_expert,
    input  logic [$clog2(HIDDEN)-1:0]    w_wr_idx,
    input  logic signed [31:0]           w_wr_data,

    input  logic                         valid_in,
    input  logic [HIDDEN*32-1:0]         a_flat,

    output logic                         valid_out,
    input  logic                         result_ready,
    output logic [$clog2(EXPERTS)-1:0]   top_idx  [TOP_K],
    output logic signed [31:0]           top_score [TOP_K]
);

    localparam int PAIRS     = HIDDEN / 2;
    localparam int PAIR_BITS = $clog2(PAIRS > 1 ? PAIRS : 2);
    localparam int EXP_BITS  = $clog2(EXPERTS > 1 ? EXPERTS : 2);
    localparam int K_BITS    = $clog2(TOP_K > 1 ? TOP_K : 2);

    //=========================================================================
    // Weight storage — flip-flop array for bring-up
    // Production: replace with altera_syncram BRAM (EXPERTS × HIDDEN entries)
    //=========================================================================
    logic signed [31:0] w [EXPERTS-1:0][HIDDEN-1:0];

    always_ff @(posedge clk) begin
        if (w_wr_en) w[w_wr_expert][w_wr_idx] <= w_wr_data;
    end

    //=========================================================================
    // FSM
    //=========================================================================
    typedef enum logic [2:0] { S_IDLE, S_LATCH, S_COMPUTE, S_REDUCE, S_SELECT, S_OUTPUT } state_t;
    state_t state;

    // Latched activations
    logic signed [31:0] s1_a [HIDDEN-1:0];

    // Iteration counters
    logic [EXP_BITS-1:0]  expert_idx;
    logic [PAIR_BITS-1:0] pair_idx;

    // Accumulated scores
    logic signed [63:0] s2_score [EXPERTS-1:0];

    // Top-K selection state
    logic [K_BITS-1:0]          sel_round;   // 0..TOP_K-1
    logic [EXPERTS-1:0]         taken;       // bitmap of already-selected experts

    //=========================================================================
    // DSP: 2 altera_mult_add instances for even/odd pair multiply
    //=========================================================================
    logic signed [31:0] dsp_a_even, dsp_a_odd, dsp_b_even, dsp_b_odd;
    wire  signed [63:0] dsp_prod_even, dsp_prod_odd;

    altera_mult_add #(.A_WIDTH(32), .B_WIDTH(32), .PIPE_STAGES(0))
    u_dsp_even (.clock(clk), .a(dsp_a_even), .b(dsp_b_even), .result(dsp_prod_even));

    altera_mult_add #(.A_WIDTH(32), .B_WIDTH(32), .PIPE_STAGES(0))
    u_dsp_odd (.clock(clk), .a(dsp_a_odd), .b(dsp_b_odd), .result(dsp_prod_odd));

    // Combinational DSP input drive — updates immediately with counter changes
    always_comb begin
        dsp_a_even = s1_a[2*pair_idx];
        dsp_a_odd  = s1_a[2*pair_idx+1];
        dsp_b_even = w[expert_idx][2*pair_idx];
        dsp_b_odd  = w[expert_idx][2*pair_idx+1];
    end

    // Combinational top-1 search among untaken experts (for S_SELECT)
    logic [EXP_BITS-1:0]  sel_best_idx;
    logic signed [63:0]   sel_best_score;

    always_comb begin
        sel_best_idx   = '0;
        sel_best_score = {1'b1, {63{1'b0}}};  // min signed 64b
        for (int e = 0; e < EXPERTS; e++) begin
            if (!taken[e] && s2_score[e] > sel_best_score) begin
                sel_best_score = s2_score[e];
                sel_best_idx   = e[EXP_BITS-1:0];
            end
        end
    end

    assign valid_out = (state == S_OUTPUT);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            expert_idx <= '0;
            pair_idx   <= '0;
            sel_round  <= '0;
            taken      <= '0;
            for (int k = 0; k < TOP_K; k++) begin
                top_idx[k]   <= '0;
                top_score[k] <= '0;
            end
            for (int i = 0; i < HIDDEN; i++) s1_a[i] <= '0;
            for (int e = 0; e < EXPERTS; e++) s2_score[e] <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (valid_in) begin
                        for (int d = 0; d < HIDDEN; d++)
                            s1_a[d] <= $signed(a_flat[d*32+:32]);
                        state <= S_LATCH;
                    end
                end

                S_LATCH: begin
                    expert_idx <= '0;
                    pair_idx   <= '0;
                    for (int e = 0; e < EXPERTS; e++) s2_score[e] <= '0;
                    state <= S_COMPUTE;
                end

                S_COMPUTE: begin
                    s2_score[expert_idx] <= s2_score[expert_idx] + dsp_prod_even + dsp_prod_odd;

                    if (pair_idx == PAIRS - 1) begin
                        pair_idx <= '0;
                        if (expert_idx == EXPERTS - 1) begin
                            expert_idx <= '0;
                            sel_round  <= '0;
                            taken      <= '0;
                            state <= S_SELECT;
                        end else begin
                            expert_idx <= expert_idx + 1'b1;
                        end
                    end else begin
                        pair_idx <= pair_idx + 1'b1;
                    end
                end

                S_SELECT: begin
                    // One round: find max score among untaken experts (combinational)
                    // Result registered at end of cycle
                    top_idx[sel_round]   <= sel_best_idx;
                    top_score[sel_round] <= sel_best_score[31:0];
                    taken[sel_best_idx]  <= 1'b1;

                    if (sel_round == TOP_K - 1) begin
                        state <= S_OUTPUT;
                    end else begin
                        sel_round <= sel_round + 1'b1;
                    end
                end

                S_OUTPUT: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
