// =============================================================================
// ffn_router_topk.sv — MoE Router: 128-MAC GEMV + Top-K selector
// V2-Lite SPEC §2: 2048×66 router weight, DSP_LANES=128, top-K=6
// =============================================================================

module ffn_router_topk #(
    parameter int HIDDEN      = 2048,
    parameter int NUM_EXPERTS = 66,
    parameter int TOP_K       = 6,
    parameter int DSP_LANES   = 128,
    parameter int DATA_W      = 8,
    parameter int SCORE_W     = 16
) (
    input  logic                         clk, rst_n,
    input  logic                         start,
    output logic                         busy, done,

    input  logic [HIDDEN*DATA_W-1:0]     activ_data,
    input  logic                         activ_valid,
    output logic                         activ_ready,

    output logic [$clog2(NUM_EXPERTS*HIDDEN):0] wt_addr,
    input  logic [DSP_LANES*DATA_W-1:0]       wt_data,
    output logic                               wt_read,

    output logic [$clog2(NUM_EXPERTS)-1:0]     topk_expert [TOP_K],
    output logic [SCORE_W-1:0]                  topk_score  [TOP_K],
    output logic                                topk_valid,

    output logic [3:0]                          dbg_fsm
);

    typedef enum logic [2:0] { S_IDLE, S_COMPUTE, S_TOPK_SORT, S_OUTPUT, S_DONE } st_t;
    st_t st;
    assign busy = (st != S_IDLE && st != S_DONE);
    assign activ_ready = (st == S_IDLE);
    assign dbg_fsm = st;

    localparam int CYCLES_PER_EXPERT = HIDDEN / DSP_LANES;  // 2048/128 = 16

    // =========================================================================
    // Scores: 66 × 16-bit, one expert per 16 cycles
    // =========================================================================
    logic [SCORE_W-1:0]            scores [NUM_EXPERTS];
    logic [$clog2(NUM_EXPERTS):0]  expert_cnt;
    logic [5:0]                     cyc_cnt;  // 0..15

    generate
        for (genvar gz = 0; gz < NUM_EXPERTS; gz++) begin : g_scores
            always_ff @(posedge clk or negedge rst_n)
                if (!rst_n) scores[gz] <= 0;
        end
    endgenerate

    // =========================================================================
    // MAC lanes: 128 × ($signed(activ) * $signed(weight))
    // =========================================================================
    logic [DSP_LANES-1:0][SCORE_W-1:0] partials;

    generate
        for (genvar gl = 0; gl < DSP_LANES; gl++) begin : g_mac
            always_ff @(posedge clk) begin
                partials[gl] <= $signed(activ_data[(cyc_cnt*DSP_LANES + gl)*DATA_W +: DATA_W])
                               * $signed(wt_data[gl*DATA_W +: DATA_W]);
            end
        end
    endgenerate

    // =========================================================================
    // Serial accumulator: sum 128 partials over 128 sub-cycles
    // =========================================================================
    logic [7:0]                     partial_idx;
    logic [SCORE_W-1:0]             accum_sum;

    generate
        // Initialize accum_sum at start of each expert
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) accum_sum <= 0;
            else if (st == S_COMPUTE && cyc_cnt == 0 && partial_idx == 0)
                accum_sum <= 0;
            else if (st == S_COMPUTE)
                accum_sum <= accum_sum + partials[partial_idx];
        end
    endgenerate

    // =========================================================================
    // Top-K: 66 experts sequential insertion sort
    // =========================================================================
    logic [SCORE_W-1:0] tk_scores [TOP_K];
    logic [$clog2(NUM_EXPERTS)-1:0] tk_ids [TOP_K];
    logic sort_done;
    logic [$clog2(NUM_EXPERTS):0]   sort_idx;

    generate
        for (genvar gk = 0; gk < TOP_K; gk++) begin : g_tk
            always_ff @(posedge clk or negedge rst_n)
                if (!rst_n) begin tk_scores[gk] <= 0; tk_ids[gk] <= 0; end
        end
    endgenerate

    wire [SCORE_W-1:0] current_score = scores[sort_idx];

    // Sequential Top-K insertion (one expert per cycle)
    // Comparator chain: check against each of TOP_K slots
    wire [TOP_K-1:0] tk_cmp_gt;
    generate
        for (genvar gk = 0; gk < TOP_K; gk++) begin : g_cmp
            assign tk_cmp_gt[gk] = (current_score > tk_scores[gk]);
        end
    endgenerate

    // =========================================================================
    // FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; expert_cnt <= 0; cyc_cnt <= 0; partial_idx <= 0;
            wt_read <= 0; done <= 0; sort_idx <= 0; sort_done <= 0;
            topk_valid <= 0;
        end else begin
            done <= 0; topk_valid <= 0;

            case (st)
                S_IDLE: begin
                    if (start) begin expert_cnt <= 0; cyc_cnt <= 0; partial_idx <= 0; st <= S_COMPUTE; end
                end

                S_COMPUTE: begin
                    wt_read <= 1;
                    partial_idx <= partial_idx + 1;
                    if (partial_idx == DSP_LANES - 1) begin
                        partial_idx <= 0; cyc_cnt <= cyc_cnt + 1;
                        // Save score after all sub-cycles
                        if (cyc_cnt == CYCLES_PER_EXPERT - 1) begin
                            scores[expert_cnt] <= accum_sum;
                            cyc_cnt <= 0; expert_cnt <= expert_cnt + 1;
                        end
                    end
                    if (expert_cnt == NUM_EXPERTS) begin
                        wt_read <= 0; sort_idx <= 0; sort_done <= 0; st <= S_TOPK_SORT;
                    end
                end

                S_TOPK_SORT: begin
                    sort_idx <= sort_idx + 1;
                    if (sort_idx < NUM_EXPERTS && !sort_done) begin
                        // Insert current_score into top-K
                        if (tk_cmp_gt[0]) begin tk_scores[0] <= current_score; tk_ids[0] <= sort_idx; end
                        else if (tk_cmp_gt[1]) begin tk_scores[1] <= current_score; tk_ids[1] <= sort_idx; end
                        else if (tk_cmp_gt[2]) begin tk_scores[2] <= current_score; tk_ids[2] <= sort_idx; end
                        else if (tk_cmp_gt[3]) begin tk_scores[3] <= current_score; tk_ids[3] <= sort_idx; end
                        else if (tk_cmp_gt[4]) begin tk_scores[4] <= current_score; tk_ids[4] <= sort_idx; end
                        else if (tk_cmp_gt[5]) begin tk_scores[5] <= current_score; tk_ids[5] <= sort_idx; end
                    end
                    if (sort_idx == NUM_EXPERTS) begin sort_done <= 1; st <= S_OUTPUT; end
                end

                S_OUTPUT: begin
                    for (int gk = 0; gk < TOP_K; gk++) begin
                        topk_expert[gk] <= tk_ids[gk];
                        topk_score[gk]  <= tk_scores[gk];
                    end
                    topk_valid <= 1; st <= S_DONE;
                end

                S_DONE: begin done <= 1; if (!start) st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule
