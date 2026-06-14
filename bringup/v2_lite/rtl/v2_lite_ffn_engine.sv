// =============================================================================
// v2_lite_ffn_engine.sv — V2-Lite FFN Engine (Phase 1 Bring-Up)
// GEMV array with lpm_mult DSP, directly wired to prevent optimization
// =============================================================================
module v2_lite_ffn_engine #(
    parameter int HIDDEN      = 2048, parameter int INTER = 1408,
    parameter int NUM_EXPERTS = 66,   parameter int TOP_K = 6,
    parameter int DATA_W = 8,         parameter int ACCUM_W = 24,
    parameter int DSP_LANES = 512
) (
    input  logic clk, rst_n, mode_prefill,
    input  logic pcie_rx_valid,
    input  logic [HIDDEN*DATA_W-1:0] pcie_rx_data,
    output logic pcie_rx_ready,
    output logic pcie_tx_valid,
    output logic [HIDDEN*DATA_W-1:0] pcie_tx_data,
    input  logic pcie_tx_ready,
    output logic [31:0] m_axi_araddr, m_axi_arlen, m_axi_arsize, m_axi_arvalid,
    input  logic m_axi_arready,
    input  logic [255:0] m_axi_rdata,
    input  logic [1:0] m_axi_rresp,
    input  logic m_axi_rvalid, m_axi_rlast,
    output logic m_axi_rready,
    output logic busy, done,
    output logic [3:0] dbg_fsm, dbg_sub_fsm,
    output logic [2:0] dbg_expert_cnt,
    output logic dbg_gemv_busy, dbg_hbm2_busy,
    output logic [31:0] perf_token_cnt, perf_cycle_cnt,
    output logic [63:0] pr_debug,
    output logic err_merge_overflow, err_silu_overflow, err_axi_resp_err
);
    // GEMV interface
    logic gemv_busy, gemv_done, gemv_result_valid;
    logic [DSP_LANES*DATA_W-1:0] gemv_activ, gemv_weight;
    logic [ACCUM_W-1:0] gemv_result;
    logic [$clog2(INTER+1)-1:0] gemv_result_row;
    logic gemv_result_last;
    logic [3:0] gemv_fsm;
    logic [9:0] gemv_cycle;
    logic [ACCUM_W-1:0] gemv_dbg_reduced;

    // Ramp data pattern for activation. Weight from AXI read data.
    logic [7:0] ramp;
    always_ff @(posedge clk) ramp <= ramp + 1;
    assign gemv_activ = {DSP_LANES{ramp}};
    // Feed AXI read data into weight path — creates real HBM2→GEMV signal chain
    assign gemv_weight = {DSP_LANES{m_axi_rdata[7:0]}};

    ffn_gemv_array #(.DSP_LANES(DSP_LANES),.INPUT_DIM(HIDDEN),.OUTPUT_DIM(INTER)) u_gemv(
        .clk,.rst_n,.start(1'b1),.busy(gemv_busy),.done(gemv_done),
        .activ_valid(1'b1),.activ_ready(),.activ_data(gemv_activ),
        .weight_valid(1'b1),.weight_ready(),.weight_data(gemv_weight),
        .wt_preload_req(),.wt_preload_row(),.wt_preload_ack(1'b1),
        .result_valid(gemv_result_valid),.result_ready(1'b1),
        .result_data(gemv_result),.result_row(gemv_result_row),
        .result_last(gemv_result_last),
        .dbg_fsm(gemv_fsm),.dbg_cycle(gemv_cycle),
        .dbg_reduced_out(gemv_dbg_reduced),
        .mode_prefill(1'b0),.prefill_tokens(6'd1)
    );

    // Wire outputs directly to prevent Quartus optimization
    assign dbg_fsm = gemv_fsm;
    assign dbg_gemv_busy = gemv_busy;
    assign dbg_expert_cnt = gemv_result_row[2:0];
    assign done = gemv_done;
    assign busy = gemv_busy;
    assign pcie_rx_ready = 1'b0;

    // XOR-based output anchor — every result bit must reach here
    always_ff @(posedge clk) begin
        pcie_tx_valid <= gemv_result_valid;
        pcie_tx_data <= { {(HIDDEN*DATA_W-ACCUM_W){1'b0}}, gemv_result };
        pr_debug <= { gemv_dbg_reduced, gemv_result };
    end

    // =========================================================================
    // AXI4 Read State Machine — HBM2 weight fetch (Phase 1 bring-up)
    // Keeps HBM2→FFN→PCIe data path alive to prevent DSP optimization
    // =========================================================================
    typedef enum logic [1:0] { AR_IDLE, AR_REQ, AR_DATA, AR_DONE } ar_st_t;
    ar_st_t ar_st;
    logic [31:0] ar_cnt;
    assign dbg_hbm2_busy = (ar_st != AR_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_st <= AR_IDLE;
            m_axi_arvalid <= 0; m_axi_araddr <= 0;
            m_axi_arlen <= 0; m_axi_arsize <= 3'd5; // 32-byte beats
            m_axi_rready <= 0;
            ar_cnt <= 0;
        end else begin
            case (ar_st)
                AR_IDLE: begin
                    m_axi_arvalid <= 1;
                    m_axi_araddr <= 32'h0000_0000; // HBM2 base
                    m_axi_arlen <= 8'd3;           // 4 beats = 128 bytes
                    m_axi_rready <= 1;
                    ar_cnt <= 0;
                    ar_st <= AR_REQ;
                end
                AR_REQ: begin
                    if (m_axi_arvalid && m_axi_arready) begin
                        m_axi_arvalid <= 0;
                        ar_st <= AR_DATA;
                    end
                end
                AR_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        ar_cnt <= ar_cnt + 1;
                        if (m_axi_rlast) ar_st <= AR_DONE;
                    end
                end
                AR_DONE: begin
                    m_axi_rready <= 0;
                    ar_st <= AR_IDLE;
                end
                default: ar_st <= AR_IDLE;
            endcase
        end
    end

    assign perf_token_cnt = ar_cnt;
    assign perf_cycle_cnt = 32'd0;
    assign dbg_sub_fsm = {1'b0, ar_st};
    assign err_merge_overflow = 1'b0;
    assign err_silu_overflow = 1'b0;
    assign err_axi_resp_err = 1'b0;
endmodule
