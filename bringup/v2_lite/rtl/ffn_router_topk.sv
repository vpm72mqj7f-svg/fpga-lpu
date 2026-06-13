// =============================================================================
// ffn_router_topk.sv — MoE Router: 128-MAC GEMV + Top-K selector
// V2-Lite SPEC §2: 2048×66 router weight, 128 MAC, top-K=6 selection
//
// Operation:
//   1. Load router weight (2048×66, ~135KB, resident in SRAM)
//   2. Compute activ × weight → 66 scores
//   3. Select top-K experts by score
// =============================================================================

module ffn_router_topk #(
    parameter int HIDDEN      = 2048,
    parameter int NUM_EXPERTS = 66,
    parameter int TOP_K       = 6,
    parameter int ROUTER_LANES= 128,
    parameter int DATA_W      = 8,
    parameter int SCORE_W     = 16
) (
    input  logic                         clk, rst_n,
    input  logic                         start,
    output logic                         busy, done,

    // Activation input
    input  logic [HIDDEN*DATA_W-1:0]     activ_data,
    input  logic                         activ_valid,
    output logic                         activ_ready,

    // Router weight SRAM interface
    output logic [$clog2(NUM_EXPERTS*HIDDEN):0] wt_addr,
    input  logic [ROUTER_LANES*DATA_W-1:0]     wt_data,
    output logic                                wt_read,

    // Top-K output
    output logic [$clog2(NUM_EXPERTS)-1:0]      topk_expert [TOP_K],
    output logic [SCORE_W-1:0]                   topk_score  [TOP_K],
    output logic                                 topk_valid,

    output logic [3:0]                           dbg_fsm
);

    typedef enum logic [2:0] { S_IDLE, S_COMPUTE, S_TOPK_SORT, S_OUTPUT, S_DONE } st_t;
    st_t st;

    assign busy = (st != S_IDLE && st != S_DONE);
    assign activ_ready = (st == S_IDLE);
    assign dbg_fsm = st;

    // =========================================================================
    // Router MAC: 128 lanes, time-multiplexed over 2048/128=16 cycles per expert
    // Scores accumulated in fabric registers (66 experts × 16-bit)
    // =========================================================================
    localparam int CYCLES_PER_EXPERT = HIDDEN / ROUTER_LANES;  // 2048/128 = 16
    localparam int TOTAL_CYCLES = NUM_EXPERTS * CYCLES_PER_EXPERT; // 66*16 = 1056

    logic [$clog2(NUM_EXPERTS):0]   expert_cnt;
    logic [$clog2(CYCLES_PER_EXPERT):0] cyc_cnt;
    logic [SCORE_W-1:0]            scores [NUM_EXPERTS];

    // MAC lanes: simple multiply-accumulate
    logic [ROUTER_LANES-1:0][SCORE_W-1:0] router_partial;

    genvar li;
    generate
        for (li = 0; li < ROUTER_LANES; li++) begin : g_rmac
            logic signed [DATA_W-1:0] a, b;
            always_ff @(posedge clk) begin
                a <= $signed(activ_data[(cyc_cnt*ROUTER_LANES + li)*DATA_W +: DATA_W]);
                b <= $signed(wt_data[li*DATA_W +: DATA_W]);
                router_partial[li] <= a * b;
            end
        end
    endgenerate

    // Score accumulation + expert cycling
    logic [$clog2(CYCLES_PER_EXPERT):0] acc_cyc;
    genvar ei;
    generate for (ei = 0; ei < NUM_EXPERTS; ei++) begin : g_score_rst
        always_ff @(posedge clk or negedge rst_n)
            if (!rst_n) scores[ei] <= 0;
    end endgenerate

    always_ff @(posedge clk) begin
        if (st == S_COMPUTE) begin
            if (acc_cyc == 0) scores[expert_cnt] <= 0;
            else begin
                logic [SCORE_W-1:0] sum;
                sum = scores[expert_cnt];
                for (int i = 0; i < ROUTER_LANES; i++) sum = sum + router_partial[i];
                scores[expert_cnt] <= sum;
            end
            acc_cyc <= acc_cyc + 1;
            if (acc_cyc == CYCLES_PER_EXPERT - 1) begin acc_cyc <= 0; expert_cnt <= expert_cnt + 1; end
        end
    end

    // =========================================================================
    // Top-K: sequential comparison (66 cycles for 66 experts)
    // =========================================================================
    logic [SCORE_W-1:0] topk_scores [TOP_K];
    logic [$clog2(NUM_EXPERTS)-1:0] topk_ids [TOP_K];
    logic sort_done;

    genvar kk;
    generate for (kk = 0; kk < TOP_K; kk++) begin : g_tk
        always_ff @(posedge clk or negedge rst_n)
            if (!rst_n) begin topk_scores[kk] <= 0; topk_ids[kk] <= 0; end
    end endgenerate

    logic [$clog2(NUM_EXPERTS):0] sort_idx;
    always_ff @(posedge clk) begin
        if (st == S_TOPK_SORT && !sort_done) begin
            // Insert scores[sort_idx] into sorted top-K
            for (int k = 0; k < TOP_K; k++) begin
                if (scores[sort_idx] > topk_scores[k]) begin
                    for (int j = TOP_K-1; j > k; j--) begin
                        topk_scores[j] <= topk_scores[j-1];
                        topk_ids[j] <= topk_ids[j-1];
                    end
                    topk_scores[k] <= scores[sort_idx];
                    topk_ids[k] <= sort_idx;
                    break;
                end
            end
            sort_idx <= sort_idx + 1;
            if (sort_idx == NUM_EXPERTS - 1) sort_done <= 1;
        end
    end
        if (st == S_TOPK_SORT) begin
            topk_scores <= '{default:0}; topk_ids <= '{default:0};
            for (int e = 0; e < NUM_EXPERTS; e++) begin
                for (int k = 0; k < TOP_K; k++) begin
                    if (scores[e] > topk_scores[k]) begin
                        for (int j = TOP_K-1; j > k; j--) begin
                            topk_scores[j] <= topk_scores[j-1]; topk_ids[j] <= topk_ids[j-1];
                        end
                        topk_scores[k] <= scores[e]; topk_ids[k] <= e[$clog2(NUM_EXPERTS)-1:0];
                        break;
                    end
                end
            end
            sort_done <= 1;
        end
    end

    // Output
    always_ff @(posedge clk) begin
        if (st == S_OUTPUT) begin
            for (int k = 0; k < TOP_K; k++) begin
                topk_expert[k] <= topk_ids[k];
                topk_score[k]  <= topk_scores[k];
            end
            topk_valid <= 1;
        end else begin
            topk_valid <= 0;
        end
    end

    // =========================================================================
    // FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; expert_cnt <= 0; cyc_cnt <= 0; acc_cyc <= 0; done <= 0;
        end else begin
            done <= 0;
            case (st)
                S_IDLE: if (start) begin expert_cnt <= 0; acc_cyc <= 0; st <= S_COMPUTE; end
                S_COMPUTE: begin
                    wt_read <= 1;
                    if (expert_cnt == NUM_EXPERTS) begin
                        wt_read <= 0; sort_done <= 0; st <= S_TOPK_SORT;
                    end
                end
                S_TOPK_SORT: if (sort_done) st <= S_OUTPUT;
                S_OUTPUT: begin st <= S_DONE; end
                S_DONE: begin done <= 1; if (!start) st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule
