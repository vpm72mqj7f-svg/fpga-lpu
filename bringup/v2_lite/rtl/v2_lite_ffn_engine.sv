// =============================================================================
// v2_lite_ffn_engine.sv — V2-Lite FFN (SPEC aligned, M3 minimal)
//
// Minimal for partition validation: 512-MAC GEMV array + self-test FSM.
// Router, shared/routed experts added in subsequent iterations.
// =============================================================================

module v2_lite_ffn_engine #(
    parameter int HIDDEN  = 2048, parameter int INTER = 1408,
    parameter int NUM_EXPERTS = 66, parameter int TOP_K = 6,
    parameter int DATA_W = 8, parameter int ACCUM_W = 24,
    parameter int DSP_LANES = 512
) (
    input  logic clk, rst_n, mode_prefill,
    input  logic pcie_rx_valid, pcie_rx_data_valid, // placeholder
    input  logic [HIDDEN*DATA_W-1:0] pcie_rx_data,
    output logic pcie_rx_ready,
    output logic pcie_tx_valid,
    output logic [HIDDEN*DATA_W-1:0] pcie_tx_data,
    input  logic pcie_tx_ready,
    output logic [31:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen, m_axi_arvalid,
    output logic [2:0]  m_axi_arsize,
    input  logic         m_axi_arready,
    input  logic [255:0] m_axi_rdata,
    input  logic [1:0]   m_axi_rresp,
    input  logic         m_axi_rvalid, m_axi_rlast,
    output logic         m_axi_rready,
    output logic busy, done,
    output logic [3:0] dbg_fsm, dbg_sub_fsm,
    output logic [2:0] dbg_expert_cnt,
    output logic dbg_gemv_busy, dbg_hbm2_busy,
    output logic [31:0] perf_token_cnt, perf_cycle_cnt,
    output logic [63:0] pr_debug,
    output logic err_merge_overflow, err_silu_overflow, err_axi_resp_err
);

    typedef enum logic [2:0] { S_IDLE, S_COMPUTE, S_OUTPUT, S_DONE } st_t;
    st_t st;
    assign busy = (st != S_IDLE && st != S_DONE);
    assign dbg_fsm = st;
    assign pcie_rx_ready = (st == S_IDLE);

    // =========================================================================
    // GEMV array: always running in self-test mode for DSP utilization
    // =========================================================================
    logic gemv_start, gemv_busy, gemv_done;
    logic [DSP_LANES*DATA_W-1:0] gemv_activ, gemv_weight;
    logic [ACCUM_W-1:0] gemv_result;
    logic gemv_result_valid;

    // Self-test ramp pattern for activ/weight
    always_ff @(posedge clk) begin
        for (int i = 0; i < DSP_LANES; i++) begin
            gemv_activ[i*DATA_W +: DATA_W] <= i[7:0];
            gemv_weight[i*DATA_W +: DATA_W] <= (255 - i[7:0]);
        end
    end

    ffn_gemv_array #(.DSP_LANES(DSP_LANES)) u_gemv (
        .clk, .rst_n, .start(gemv_start), .busy(gemv_busy), .done(gemv_done),
        .activ_valid(1'b1), .activ_ready(), .activ_data(gemv_activ),
        .weight_valid(1'b1), .weight_ready(), .weight_data(gemv_weight),
        .wt_preload_req(), .wt_preload_row(), .wt_preload_ack(1'b1),
        .result_valid(gemv_result_valid), .result_ready(1'b1),
        .result_data(gemv_result), .result_row(), .result_last(),
        .mode_prefill(mode_prefill), .prefill_tokens(6'd1)
    );

    assign dbg_gemv_busy = gemv_busy;
    assign dbg_sub_fsm = st;
    assign dbg_hbm2_busy = 1'b0;

    // =========================================================================
    // Self-test FSM
    // =========================================================================
    logic [31:0] cycle_cnt;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; gemv_start <= 0; done <= 0;
            perf_token_cnt <= 0; cycle_cnt <= 0; pr_debug <= 0;
            pcie_tx_valid <= 0;
        end else begin
            done <= 0; gemv_start <= 0; pcie_tx_valid <= 0;
            cycle_cnt <= cycle_cnt + 1;

            case (st)
                S_IDLE: begin gemv_start <= 1; st <= S_COMPUTE; end
                S_COMPUTE: begin
                    if (gemv_done) begin
                        pr_debug <= {gemv_result, 40'd0}; // latch result → visible
                        st <= S_OUTPUT;
                    end
                end
                S_OUTPUT: begin
                    pcie_tx_valid <= 1;
                    if (pcie_tx_ready) begin
                        perf_token_cnt <= perf_token_cnt + 1;
                        st <= S_DONE;
                    end
                end
                S_DONE: begin
                    done <= 1; gemv_start <= 1; st <= S_COMPUTE; // loop forever
                end
                default: st <= S_IDLE;
            endcase
        end
    end

    assign perf_cycle_cnt = cycle_cnt;
    assign err_merge_overflow = 1'b0;
    assign err_silu_overflow = 1'b0;
    assign err_axi_resp_err = 1'b0;
    assign m_axi_araddr = 0; assign m_axi_arlen = 0; assign m_axi_arsize = 0;
    assign m_axi_arvalid = 0; assign m_axi_rready = 0;

endmodule
