// =============================================================================
// v2_lite_ffn_engine.sv — DeepSeek V2-Lite FFN Compute Engine
//
// V2-Lite: hidden=2048, inter=1408, 66 experts (64+2), TOP_K=6, FP8
//
// Architecture:
//   Stage 1 (Gate):  hidden[] × gate_w[inter][hidden] → gate_out[inter]
//   Stage 2 (Up):    hidden[] × up_w[inter][hidden]   → up_out[inter]
//   Stage 3 (Gate):  gate_out ⊙ up_out               → combined[inter]
//   Stage 4 (Down):  combined[] × down_w[hidden][inter] → ffn_out[hidden]
//
// DSP Array: T parallel MAC units, time-multiplexed across INTER dimension.
//   T = 64 (bring-up), scales to 256 (production).
//   Gate+Up merged into one dot-product unit, shared across experts.
//
// Weights stored in HBM2 (8 GB). During bringup: small M20K weight buffer.
// =============================================================================

module v2_lite_ffn_engine #(
    parameter int HIDDEN      = 2048,
    parameter int INTER       = 1408,
    parameter int NUM_EXPERTS = 66,
    parameter int TOP_K       = 6,
    parameter int DATA_W      = 8,      // FP8
    parameter int DSP_LANES   = 64      // parallel MAC units
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         pcie_rx_valid,
    input  logic [HIDDEN*DATA_W-1:0]     pcie_rx_data,
    output logic                         pcie_rx_ready,
    output logic                         pcie_tx_valid,
    output logic [HIDDEN*DATA_W-1:0]     pcie_tx_data,
    input  logic                         pcie_tx_ready,
    input  logic                         wt_wr_en,
    input  logic [$clog2(NUM_EXPERTS)-1:0] wt_expert_id,
    input  logic [1:0]                   wt_type,
    input  logic [$clog2(INTER)-1:0]     wt_row,
    input  logic [$clog2(HIDDEN)-1:0]    wt_col,
    input  logic [DATA_W-1:0]            wt_data,
    input  logic [$clog2(NUM_EXPERTS)-1:0] expert_id [TOP_K],
    output logic                         busy,
    output logic                         done
);

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [3:0] {
        S_IDLE, S_LOAD_ACTIV, S_GATE_UP, S_ACTIVATE, S_DOWN, S_ACCUM, S_OUTPUT
    } state_t;
    state_t state, next_state;

    // =========================================================================
    // Activation buffer (latched attention output)
    // =========================================================================
    logic [DATA_W-1:0] activ [HIDDEN-1:0];
    logic [DATA_W-1:0] ffn_out [HIDDEN-1:0];

    // =========================================================================
    // DSP dot-product unit
    // =========================================================================
    logic [$clog2(INTER)-1:0]  dp_row;       // current output row
    logic [$clog2(HIDDEN)-1:0] dp_col;       // current input col within row
    logic [DATA_W-1:0]         dp_a [DSP_LANES-1:0];  // activation inputs
    logic [DATA_W-1:0]         dp_b [DSP_LANES-1:0];  // weight inputs
    logic [2*DATA_W+$clog2(DSP_LANES)-1:0] dp_sum;   // accumulated sum

    // Per-expert accumulators (down projection output, HIDDEN-wide)
    logic [2*DATA_W+$clog2(INTER)-1:0] expert_accum [HIDDEN-1:0];

    // Expert counter (0..TOP_K-1)
    logic [$clog2(TOP_K):0] expert_idx;

    // =========================================================================
    // Weight Buffer (M20K: small test weights for bring-up)
    // Production: HBM2-backed via AXI4 interface
    // =========================================================================
    localparam int WT_DEPTH = 256;  // bring-up: 256×FP8 test weights
    logic [DATA_W-1:0] gate_w [WT_DEPTH-1:0];
    logic [DATA_W-1:0] up_w   [WT_DEPTH-1:0];
    logic [DATA_W-1:0] down_w [WT_DEPTH-1:0];

    // =========================================================================
    // Main FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            busy        <= 1'b0;
            done        <= 1'b0;
            pcie_rx_ready <= 1'b0;
            pcie_tx_valid <= 1'b0;
            dp_row      <= '0;
            dp_col      <= '0;
            dp_sum      <= '0;
            expert_idx  <= '0;
            for (int i = 0; i < HIDDEN; i++) begin
                activ[i]   <= '0;
                ffn_out[i] <= '0;
                expert_accum[i] <= '0;
            end
        end else begin
            done        <= 1'b0;
            pcie_rx_ready <= 1'b0;
            pcie_tx_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (pcie_rx_valid) begin
                        for (int i = 0; i < HIDDEN; i++)
                            activ[i] <= pcie_rx_data[i*DATA_W +: DATA_W];
                        pcie_rx_ready <= 1'b1;
                        busy      <= 1'b1;
                        expert_idx <= '0;
                        state     <= S_LOAD_ACTIV;
                    end
                end

                S_LOAD_ACTIV: begin
                    // Load activation into DSP lanes
                    for (int k = 0; k < DSP_LANES; k++)
                        dp_a[k] <= activ[dp_col * DSP_LANES + k];
                    state <= S_GATE_UP;
                end

                S_GATE_UP: begin
                    // Compute gate(activ) ⊙ up(activ) for one row
                    // Gate: sum(activ[j] * gate_w[row][j])
                    // Up:   sum(activ[j] * up_w[row][j])
                    // Simplified bring-up: load from weight buffer
                    for (int k = 0; k < DSP_LANES; k++) begin
                        dp_b[k] <= gate_w[(dp_row * HIDDEN / DSP_LANES + dp_col) % WT_DEPTH];
                    end
                    dp_col <= dp_col + 1;
                    if (dp_col == (HIDDEN / DSP_LANES) - 1) begin
                        dp_col <= '0;
                        dp_row <= dp_row + 1;
                        state  <= S_ACTIVATE;
                    end
                end

                S_ACTIVATE: begin
                    // Apply SiLU (gate) and multiply with Up
                    // Simplified: pass-through for bringup simulation
                    state <= S_DOWN;
                end

                S_DOWN: begin
                    // Down projection: combined[] × down_w[hidden][inter]
                    // Simplified: accumulate dummy value for bringup
                    state <= S_ACCUM;
                end

                S_ACCUM: begin
                    // Accumulate per-expert result
                    expert_idx <= expert_idx + 1;
                    if (expert_idx == TOP_K - 1)
                        state <= S_OUTPUT;
                    else
                        state <= S_LOAD_ACTIV;
                end

                S_OUTPUT: begin
                    pcie_tx_valid <= 1'b1;
                    for (int i = 0; i < HIDDEN; i++)
                        pcie_tx_data[i*DATA_W +: DATA_W] <= ffn_out[i];
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

    // =========================================================================
    // Weight preload (bring-up: store in M20K buffer)
    // =========================================================================
    always_ff @(posedge clk) begin
        if (wt_wr_en) begin
            case (wt_type)
                2'd0: gate_w[wt_row] <= wt_data;
                2'd1: up_w[wt_row]   <= wt_data;
                2'd2: down_w[wt_row] <= wt_data;
            endcase
        end
    end

endmodule
