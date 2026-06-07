//=============================================================================
// s10_ffn_engine.sv — Stratix 10 MX FFN-only Engine for V2 Lite
//
// Phase 1 bring-up: CPU attention + FPGA FFN on S10 MX dev kit.
// V2 Lite: 27 layers, hidden=2048, inter=1408, 64 routed + 2 shared experts.
// FP8 weights, FP8 activations.
//
// PCIe Gen3 x16: attention output (RN) → FFN → CPU (RN).
//=============================================================================

module s10_ffn_engine #(
    parameter int HIDDEN        = 2048,   // V2 Lite hidden dim
    parameter int INTER         = 1408,   // V2 Lite expert intermediate
    parameter int NUM_EXPERTS   = 66,     // 64 routed + 2 shared
    parameter int TOP_K         = 6,      // activated per token
    parameter int DATA_W        = 8       // FP8
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // PCIe RX: attention output from CPU (HIDDEN × FP8)
    input  logic                         pcie_rx_valid,
    input  logic [HIDDEN*DATA_W-1:0]     pcie_rx_data,
    output logic                         pcie_rx_ready,

    // PCIe TX: FFN output to CPU (HIDDEN × FP8)
    output logic                         pcie_tx_valid,
    output logic [HIDDEN*DATA_W-1:0]     pcie_tx_data,
    input  logic                         pcie_tx_ready,

    // Expert weight preload (from HBM/DDR, one-time init)
    input  logic                         wt_wr_en,
    input  logic [$clog2(NUM_EXPERTS)-1:0] wt_expert_id,
    input  logic [1:0]                   wt_type,   // 0=gate, 1=up, 2=down
    input  logic [$clog2(INTER)-1:0]     wt_row,
    input  logic [$clog2(HIDDEN)-1:0]    wt_col,
    input  logic [DATA_W-1:0]            wt_data,

    // Expert selection (from router, received via PCIe sideband)
    input  logic [$clog2(NUM_EXPERTS)-1:0] expert_id [TOP_K],

    // Status
    output logic                         busy,
    output logic                         done
);

    typedef enum logic [2:0] {
        S_IDLE, S_LOAD, S_COMPUTE, S_OUTPUT
    } state_t;
    state_t state;

    // Weight storage: NUM_EXPERTS × 3 matrices × HIDDEN×INTER (FP8)
    // S10 MX M20K: 6,847 blocks × 20Kb = ~17 MB
    // Expert weights: 66 experts × 3 × 2048×1408 bytes = ~560 MB
    // → Too large for M20K, must use HBM2
    // During bring-up: store small test weights in M20K (hidden=8, inter=4)

    // Simplified for bring-up simulation:
    // Use altera_syncram for HBM2-backed weight storage
    // Each expert's gate/up/down stored as separate syncram

    logic [TOP_K-1:0][DATA_W-1:0]      activ [HIDDEN-1:0];  // latched input
    logic [TOP_K-1:0][DATA_W-1:0]      result [HIDDEN-1:0];  // accumulated output
    logic [$clog2(TOP_K)-1:0]           expert_idx;
    logic [$clog2(HIDDEN)-1:0]          row_cnt;
    logic [$clog2(INTER)-1:0]           col_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            pcie_rx_ready <= 1'b0;
            pcie_tx_valid <= 1'b0;
            expert_idx   <= '0;
            row_cnt      <= '0;
            col_cnt      <= '0;
            for (int i = 0; i < HIDDEN; i++) begin
                for (int k = 0; k < TOP_K; k++) begin
                    activ[i][k]  <= '0;
                    result[i][k] <= '0;
                end
            end
        end else begin
            done          <= 1'b0;
            pcie_rx_ready <= 1'b0;
            pcie_tx_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (pcie_rx_valid) begin
                        // Latch attention output
                        for (int i = 0; i < HIDDEN; i++)
                            activ[i][0] <= pcie_rx_data[i*DATA_W +: DATA_W];
                        pcie_rx_ready <= 1'b1;
                        expert_idx <= '0;
                        busy  <= 1'b1;
                        state <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    // Simplified: one expert per cycle (bring-up)
                    // Production: systolic array, 6 experts in parallel
                    expert_idx <= expert_idx + 1'b1;
                    if (expert_idx == TOP_K - 1) begin
                        state <= S_OUTPUT;
                    end
                end

                S_OUTPUT: begin
                    pcie_tx_valid <= 1'b1;
                    // Combine expert outputs (simplified: use last expert)
                    for (int i = 0; i < HIDDEN; i++)
                        pcie_tx_data[i*DATA_W +: DATA_W] <= result[i][TOP_K-1];

                    if (pcie_tx_ready) begin
                        done  <= 1'b1;
                        busy  <= 1'b0;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
