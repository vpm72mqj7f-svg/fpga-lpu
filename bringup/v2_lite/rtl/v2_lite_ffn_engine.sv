// =============================================================================
// v2_lite_ffn_engine.sv — V2-Lite FFN Top Engine (SPEC aligned)
//
// Submodules:
//   ffn_router_topk     — 128-MAC Router + Top-K selector
//   ffn_shared_expert   — Shared expert (SRAM weights, 9MB M20K)
//   ffn_routed_expert   — Routed expert (HBM2 weights, AXI4 read)
//   ffn_gemv_array      — 512-MAC GEMV core (shared by all experts)
//   silu_activation     — SiLU LUT + DSP interp
//   hbm2_weight_reader  — AXI4 read master for HBM2 weight streaming
//
// FSM: IDLE → ROUTER → SHARED → ROUTED(×6) → OUTPUT
// Mode: MODE=0 Decode (1 token, 512 MAC), MODE=1 Prefill (N tokens, split MAC)
// =============================================================================

module v2_lite_ffn_engine #(
    parameter int HIDDEN      = 2048,
    parameter int INTER       = 1408,
    parameter int NUM_EXPERTS = 66,
    parameter int TOP_K       = 6,
    parameter int DATA_W      = 8,
    parameter int ACCUM_W     = 24,
    parameter int DSP_LANES   = 512,
    parameter int ROUTER_LANES= 128
) (
    input  logic                         clk, rst_n,
    input  logic                         mode_prefill,   // 0=Decode, 1=Prefill

    // PCIe RX (activation from CPU)
    input  logic                         pcie_rx_valid,
    input  logic [HIDDEN*DATA_W-1:0]     pcie_rx_data,
    output logic                         pcie_rx_ready,

    // PCIe TX (FFN result to CPU)
    output logic                         pcie_tx_valid,
    output logic [HIDDEN*DATA_W-1:0]     pcie_tx_data,
    input  logic                         pcie_tx_ready,

    // HBM2 AXI4 Read (for routed expert weights)
    output logic [31:0]                  m_axi_araddr,
    output logic [7:0]                   m_axi_arlen,
    output logic [2:0]                   m_axi_arsize,
    output logic                         m_axi_arvalid,
    input  logic                         m_axi_arready,
    input  logic [255:0]                 m_axi_rdata,
    input  logic [1:0]                   m_axi_rresp,
    input  logic                         m_axi_rvalid,
    output logic                         m_axi_rready,
    input  logic                         m_axi_rlast,

    // BAR0 status readbacks
    output logic                         busy, done,
    output logic [3:0]                   dbg_fsm,
    output logic [2:0]                   dbg_expert_cnt,
    output logic [5:0]                   dbg_expert_ids [TOP_K],
    output logic [31:0]                  perf_token_cnt,
    output logic [31:0]                  perf_cycle_cnt,
    output logic [63:0]                  pr_debug,
    output logic                         err_merge_overflow,
    output logic                         err_silu_overflow,
    output logic                         err_axi_resp_err,

    // Debug
    output logic [3:0]                   dbg_sub_fsm,
    output logic                         dbg_gemv_busy,
    output logic                         dbg_hbm2_busy
);

    typedef enum logic [3:0] { S_IDLE, S_ROUTER, S_SHARED, S_ROUTED, S_OUTPUT, S_DONE } st_t;
    st_t st;

    assign busy = (st != S_IDLE && st != S_DONE);
    assign dbg_fsm = st;
    assign pcie_rx_ready = (st == S_IDLE);

    // =========================================================================
    // Activation buffer (M20K inferred)
    // =========================================================================
    (* ramstyle = "M20K" *) logic [DATA_W-1:0] activ_buf [HIDDEN];
    logic [$clog2(HIDDEN):0] activ_idx;

    always_ff @(posedge clk) begin
        if (st == S_IDLE && pcie_rx_valid)
            for (int i = 0; i < HIDDEN; i++)
                activ_buf[i] <= pcie_rx_data[i*DATA_W +: DATA_W];
    end

    // =========================================================================
    // Router: 128-MAC, 66 experts → top-6 IDs
    // =========================================================================
    logic                               router_start, router_busy, router_done;
    logic [$clog2(NUM_EXPERTS)-1:0]     router_topk_id [TOP_K];
    logic [15:0]                         router_topk_score [TOP_K];
    logic                               router_topk_valid;

    ffn_router_topk u_router (
        .clk, .rst_n, .start(router_start), .busy(router_busy), .done(router_done),
        .activ_data({HIDDEN*DATA_W{1'b0}}), .activ_valid(1'b1), // TODO: wire to activ_buf
        .topk_expert(router_topk_id), .topk_score(router_topk_score),
        .topk_valid(router_topk_valid)
    );

    // =========================================================================
    // GEMV Array: 512-MAC, shared by all expert controllers
    // =========================================================================
    logic                               gemv_start, gemv_busy, gemv_done;
    logic [$clog2(HIDDEN):0]            gemv_rows;
    logic [DSP_LANES*DATA_W-1:0]        gemv_activ, gemv_weight;
    logic                               gemv_activ_valid, gemv_activ_ready;
    logic                               gemv_weight_rd;

    ffn_gemv_array #(.DSP_LANES(DSP_LANES)) u_gemv (
        .clk, .rst_n, .start(gemv_start), .busy(gemv_busy), .done(gemv_done),
        .activ_valid(gemv_activ_valid), .activ_ready(gemv_activ_ready),
        .activ_data(gemv_activ), .weight_valid(1'b1), .weight_ready(),
        .weight_data(gemv_weight), .wt_preload_req(), .wt_preload_row(),
        .wt_preload_ack(1'b1), .result_valid(), .result_ready(1'b1),
        .result_data(), .result_row(), .result_last(),
        .mode_prefill, .prefill_tokens(6'd1)
    );

    // =========================================================================
    // Shared Expert: SRAM weights, GEMV gate/up/down
    // =========================================================================
    logic                               shared_start, shared_busy, shared_done;
    logic [HIDDEN*DATA_W-1:0]           shared_out;
    logic                               shared_valid;

    ffn_shared_expert u_shared (
        .clk, .rst_n, .start(shared_start), .busy(shared_busy), .done(shared_done),
        .activ_in({HIDDEN*DATA_W{1'b0}}), .activ_valid(1'b1), .activ_ready(),
        .ffn_out(shared_out), .ffn_valid(shared_valid),
        .sram_read(), .silu_valid(),
        .gemv_start(), .gemv_rows(gemv_rows),
        .gemv_busy(gemv_busy), .gemv_done(gemv_done),
        .gemv_activ(gemv_activ), .gemv_activ_valid(gemv_activ_valid),
        .gemv_activ_ready(gemv_activ_ready),
        .gemv_weight(gemv_weight), .gemv_weight_rd(gemv_weight_rd)
    );

    // =========================================================================
    // Routed Expert: HBM2 weights, GEMV gate/up/down, TOP_K iterations
    // =========================================================================
    logic                               routed_start, routed_busy, routed_done;
    logic [HIDDEN*DATA_W-1:0]           routed_out;
    logic                               routed_valid;
    logic [$clog2(TOP_K):0]             expert_iter;
    logic [$clog2(NUM_EXPERTS):0]       current_expert;

    ffn_routed_expert u_routed (
        .clk, .rst_n, .start(routed_start), .expert_id(current_expert),
        .busy(routed_busy), .done(routed_done),
        .activ_in({HIDDEN*DATA_W{1'b0}}), .ffn_out(routed_out), .ffn_valid(routed_valid),
        .m_axi_araddr, .m_axi_arlen, .m_axi_arsize,
        .m_axi_arvalid, .m_axi_arready,
        .m_axi_rdata, .m_axi_rvalid, .m_axi_rready, .m_axi_rlast,
        .gemv_start(gemv_start), .gemv_rows(gemv_rows),
        .gemv_busy(gemv_busy), .gemv_done(gemv_done),
        .gemv_weight(gemv_weight)
    );

    assign dbg_gemv_busy = gemv_busy;
    assign dbg_hbm2_busy = m_axi_rvalid;
    assign dbg_sub_fsm = st;
    assign dbg_expert_cnt = expert_iter;

    // =========================================================================
    // Main FSM
    // =========================================================================
    logic [ACCUM_W-1:0] ffn_accum [HIDDEN];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; router_start <= 0; shared_start <= 0; routed_start <= 0;
            expert_iter <= 0; current_expert <= 0;
            perf_token_cnt <= 0; perf_cycle_cnt <= 0;
            pcie_tx_valid <= 0; done <= 0;
        end else begin
            done <= 0; router_start <= 0; shared_start <= 0; routed_start <= 0;
            pcie_tx_valid <= 0;
            perf_cycle_cnt <= perf_cycle_cnt + 1;

            case (st)
                S_IDLE: begin
                    if (pcie_rx_valid) begin
                        for (int i = 0; i < HIDDEN; i++) ffn_accum[i] <= 0;
                        router_start <= 1; st <= S_ROUTER;
                    end
                end

                S_ROUTER: begin
                    if (router_done) begin
                        expert_iter <= 0; shared_start <= 1; st <= S_SHARED;
                    end
                end

                S_SHARED: begin
                    if (shared_done) begin
                        // Accumulate shared expert result
                        for (int i = 0; i < HIDDEN; i++)
                            ffn_accum[i] <= ffn_accum[i] + shared_out[i*DATA_W +: DATA_W];
                        current_expert <= router_topk_id[0];
                        routed_start <= 1; st <= S_ROUTED;
                    end
                end

                S_ROUTED: begin
                    if (routed_done) begin
                        for (int i = 0; i < HIDDEN; i++)
                            ffn_accum[i] <= ffn_accum[i] + routed_out[i*DATA_W +: DATA_W];
                        expert_iter <= expert_iter + 1;
                        if (expert_iter < TOP_K - 1) begin
                            current_expert <= router_topk_id[expert_iter + 1];
                            routed_start <= 1;
                        end else st <= S_OUTPUT;
                    end
                end

                S_OUTPUT: begin
                    pcie_tx_valid <= 1;
                    for (int i = 0; i < HIDDEN; i++)
                        pcie_tx_data[i*DATA_W +: DATA_W] <= ffn_accum[i];
                    if (pcie_tx_ready) begin
                        perf_token_cnt <= perf_token_cnt + 1;
                        st <= S_DONE;
                    end
                end

                S_DONE: begin done <= 1; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end

    assign pr_debug = {perf_cycle_cnt[31:0], perf_token_cnt};
    assign err_merge_overflow = 1'b0; assign err_silu_overflow = 1'b0; assign err_axi_resp_err = 1'b0;

endmodule
