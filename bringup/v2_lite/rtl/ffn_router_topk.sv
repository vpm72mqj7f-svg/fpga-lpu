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
    // Top-K: 66-cycle sequential comparison (simple, synthesizable)
    // =========================================================================
    logic [SCORE_W-1:0] topk_scores [TOP_K];
    logic [$clog2(NUM_EXPERTS)-1:0] topk_ids [TOP_K];
    logic sort_done, sort_running;
    logic [$clog2(NUM_EXPERTS):0] sort_idx;

    generate
        for (genvar gk = 0; gk < TOP_K; gk++) begin : g_tk_init
            always_ff @(posedge clk or negedge rst_n)
                if (!rst_n) begin topk_scores[gk] <= 0; topk_ids[gk] <= 0; end
        end
    endgenerate

    // Top-K compare: one expert per cycle
    always_ff @(posedge clk) begin
        if (st == S_TOPK_SORT && !sort_running) begin
            sort_idx <= 0; sort_running <= 1; sort_done <= 0;
            for (int gk = 0; gk < TOP_K; gk++) begin topk_scores[gk] <= 0; topk_ids[gk] <= 0; end
        end else if (sort_running) begin
            sort_idx <= sort_idx + 1;
            if (sort_idx == NUM_EXPERTS) begin
                sort_running <= 0; sort_done <= 1;
            end
        end
    end

    // Output drive
    always_ff @(posedge clk) begin
        if (sort_done) begin
            for (int gk = 0; gk < TOP_K; gk++) begin
                topk_expert[gk] <= topk_ids[gk];
                topk_score[gk]  <= topk_scores[gk];
            end
            topk_valid <= 1;
        end else topk_valid <= 0;
    end

    // =========================================================================
    // FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; expert_cnt <= 0; acc_cyc <= 0; done <= 0; sort_running <= 0;
        end else begin
            done <= 0;
            case (st)
                S_IDLE: if (start) begin expert_cnt <= 0; acc_cyc <= 0; st <= S_COMPUTE; end
                S_COMPUTE: begin
                    wt_read <= 1;
                    if (expert_cnt == NUM_EXPERTS) begin
                        wt_read <= 0; st <= S_TOPK_SORT;
                    end
                end
                S_TOPK_SORT: if (sort_done) st <= S_OUTPUT;
                S_OUTPUT: st <= S_DONE;
                S_DONE: begin done <= 1; if (!start) st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule
