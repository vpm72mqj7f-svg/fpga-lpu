// =============================================================================
// ffn_routed_expert.sv — Routed Expert Controller (weights from HBM2)
// V2-Lite SPEC: same GEMV as shared expert but reads from HBM2 via AXI4
//
// Reuses the same ffn_gemv_array for gate/up/down projections.
// Reads weights sequentially from HBM2 per expert.
// =============================================================================

module ffn_routed_expert #(
    parameter int HIDDEN      = 2048,
    parameter int INTER       = 1408,
    parameter int DSP_LANES   = 512,
    parameter int DATA_W      = 8,
    parameter int ACCUM_W     = 24,
    parameter int AXI_DATA_W  = 256
) (
    input  logic                         clk, rst_n,
    input  logic                         start,
    input  logic [$clog2(66):0]          expert_id,
    output logic                         busy, done,

    // Activation
    input  logic [HIDDEN*DATA_W-1:0]     activ_in,
    output logic [HIDDEN*DATA_W-1:0]     ffn_out,
    output logic                         ffn_valid,

    // HBM2 AXI4 Read (reuses hbm2_weight_reader)
    output logic [31:0]                  m_axi_araddr,
    output logic [7:0]                   m_axi_arlen,
    output logic [2:0]                   m_axi_arsize,
    output logic                         m_axi_arvalid,
    input  logic                         m_axi_arready,
    input  logic [AXI_DATA_W-1:0]       m_axi_rdata,
    input  logic                         m_axi_rvalid,
    output logic                         m_axi_rready,
    input  logic                         m_axi_rlast,

    // GEMV control
    output logic                         gemv_start,
    output logic [$clog2(HIDDEN):0]      gemv_rows,
    input  logic                         gemv_busy, gemv_done,
    input  logic [DSP_LANES*DATA_W-1:0]  gemv_weight,

    output logic [3:0]                   dbg_fsm
);
    typedef enum logic [3:0] { S_IDLE, S_GATE, S_SILU, S_UP, S_MERGE, S_DOWN, S_ACCUM, S_DONE } st_t;
    st_t st;
    assign busy = (st != S_IDLE && st != S_DONE);
    assign dbg_fsm = st;

    localparam int EXPERT_SIZE = 9 * 1024 * 1024;  // 9MB per expert

    logic [31:0] hbm_addr;
    logic        gate_done, up_done, down_done;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; hbm_addr <= 0; gate_done <= 0; up_done <= 0; down_done <= 0;
            gemv_start <= 0; m_axi_arvalid <= 0; m_axi_rready <= 0;
            ffn_valid <= 0; done <= 0;
        end else begin
            done <= 0; gemv_start <= 0; ffn_valid <= 0;

            case (st)
                S_IDLE: begin
                    if (start) begin
                        hbm_addr <= expert_id * EXPERT_SIZE;
                        gemv_rows <= INTER;
                        gemv_start <= 1;
                        st <= S_GATE;
                    end
                end

                S_GATE: begin
                    m_axi_rready <= 1;
                    if (gemv_done) begin gate_done <= 1; st <= S_SILU; end
                end

                S_SILU: begin
                    // SiLU placeholder
                    if (gate_done) begin st <= S_UP; gemv_rows <= INTER; gemv_start <= 1; end
                end

                S_UP: begin
                    m_axi_rready <= 1;
                    if (gemv_done) begin up_done <= 1; st <= S_MERGE; end
                end

                S_MERGE: begin
                    // Element-wise multiply placeholder
                    if (up_done) begin st <= S_DOWN; gemv_rows <= HIDDEN; gemv_start <= 1; end
                end

                S_DOWN: begin
                    m_axi_rready <= 1;
                    if (gemv_done) begin down_done <= 1; st <= S_ACCUM; end
                end

                S_ACCUM: begin
                    if (down_done) begin
                        ffn_valid <= 1; st <= S_DONE;
                    end
                end

                S_DONE: begin done <= 1; if (!start) st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule
