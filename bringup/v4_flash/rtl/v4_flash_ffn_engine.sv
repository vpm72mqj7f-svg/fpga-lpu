// =============================================================================
// v4_flash_ffn_engine.sv — DeepSeek V4-Flash FFN Compute Engine
//
// V4-Flash: hidden=7168, inter=3072, 385 experts (384+1), TOP_K=6, FP8
//
// Architecture: Systolic Array with HBM2 weight streaming
//
//   ┌──────────┐    ┌──────────────┐    ┌──────────┐
//   │ HBM2     │───▶│ Systolic     │───▶│ HBM2     │
//   │ (weights)│    │ Array (DSP)  │    │ (results)│
//   └──────────┘    │ 128 lanes    │    └──────────┘
//                   │ fp8×fp8→fp16 │
//                   └──────────────┘
//
// Pipeline (per expert):
//   1. Load activation from PCIe → HBM2 activation buffer
//   2. Stream gate_w, up_w rows from HBM2 → systolic array
//   3. Gate(activ) ⊙ SiLU(Gate(activ)) → combined[inter]
//   4. Stream combined[] and down_w from HBM2 → systolic array
//   5. Accumulate down results → ffn_out[hidden]
//   6. Merge TOP_K expert outputs (weighted sum)
//
// DSP Budget (S10 MX 1SM21BHU2F53): 3,960 DSP blocks
//   - Systolic array: 128 lanes × 1 MAC × 2 (fp8×fp8) = 256 DSP
//   - Time-multiplexed: 8 cycles per INTER row → effective 1024 MACs/cycle
//   - Gate+Up merge: 2 rows per cycle (gate and up simultaneous)
//
// Weight Storage (HBM2, 8 GB):
//   - Expert cache: 6 experts × 66 MB = 396 MB (active set, streamed on-demand)
//   - Activation buffer: 7,168 × FP8 = 7 KB per token
//   - Output buffer: 7,168 × FP8 × 2 = 14 KB (double-buffered)
//
// Performance (V4-Flash, per token):
//   - Gate+Up: 7,168×3,072×2 MACs / (128 DSP × 2 MACs × 500 MHz) ≈ 345 μs per expert
//   - Down:    3,072×7,168×1 MACs / (128 DSP × 2 MACs × 500 MHz) ≈ 345 μs per expert
//   - 6 experts: ~4.1 ms → ~240 tok/s
//   - With DSP scaling: 256 lanes → ~480 tok/s
// =============================================================================

module v4_flash_ffn_engine #(
    parameter int HIDDEN      = 7168,
    parameter int INTER       = 3072,
    parameter int NUM_EXPERTS = 385,
    parameter int TOP_K       = 6,
    parameter int DATA_W      = 8,       // FP8
    parameter int DSP_LANES   = 128,     // systolic array width
    parameter int ACCUM_W     = 24       // fp16 accumulator + headroom
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
        S_IDLE, S_LOAD_TOKEN, S_GATE_UP_ROW, S_ACTIVATE,
        S_DOWN_ROW, S_NEXT_EXPERT, S_OUTPUT
    } state_t;
    state_t state;

    // =========================================================================
    // Activation Buffer (double-buffered: 7,168 × FP8)
    // =========================================================================
    logic [DATA_W-1:0] activ_buf [HIDDEN-1:0];

    // =========================================================================
    // Systolic Array inputs
    // =========================================================================
    logic [DATA_W-1:0]            sa_a [DSP_LANES-1:0];  // activation lane
    logic [DATA_W-1:0]            sa_b [DSP_LANES-1:0];  // weight lane
    logic [ACCUM_W-1:0]           sa_sum [DSP_LANES-1:0]; // per-lane accum
    logic                         sa_valid;

    // Gate output buffer (per INTER element)
    logic [ACCUM_W-1:0]           gate_buf [INTER-1:0];
    logic [ACCUM_W-1:0]           up_buf   [INTER-1:0];
    logic [ACCUM_W-1:0]           combined [INTER-1:0];

    // Down projection: accumulate back to HIDDEN
    logic [ACCUM_W-1:0]           down_accum [HIDDEN-1:0];

    // Expert output (fp8, HIDDEN-wide)
    logic [DATA_W-1:0]            ffn_out [HIDDEN-1:0];

    // =========================================================================
    // Counters
    // =========================================================================
    logic [$clog2(INTER):0]       inter_row;     // current GATE/UP row
    logic [$clog2(HIDDEN):0]      hidden_col;    // current DOWN column
    logic [$clog2(TOP_K):0]       expert_cnt;
    logic [$clog2(HIDDEN/DSP_LANES):0] sa_cycle; // DSP cycle within a row

    // =========================================================================
    // Main Pipeline FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            pcie_rx_ready <= 1'b0;
            pcie_tx_valid <= 1'b0;
            inter_row  <= '0;
            hidden_col <= '0;
            sa_cycle   <= '0;
            expert_cnt <= '0;
            sa_valid   <= 1'b0;
            // Chunked loop: HIDDEN=7168 > Quartus 5000 limit
            for (int _ch = 0; _ch < HIDDEN; _ch += 2048)
                for (int i = _ch; i < _ch + 2048 && i < HIDDEN; i++) begin
                activ_buf[i]  <= '0;
                down_accum[i] <= '0;
                ffn_out[i]    <= '0;
            end
            // synthesis loop_limit 10000
            for (int j = 0; j < INTER; j++) begin
                gate_buf[j] <= '0;
                up_buf[j]   <= '0;
                combined[j] <= '0;
            end
        end else begin
            done        <= 1'b0;
            pcie_rx_ready <= 1'b0;
            pcie_tx_valid <= 1'b0;
            sa_valid     <= 1'b0;

            case (state)
                // --------------------------------------------------------
                // IDLE: wait for PCIe attention output from CPU
                // --------------------------------------------------------
                S_IDLE: begin
                    if (pcie_rx_valid) begin
                        for (int _c = 0; _c < HIDDEN; _c += 2048)
                            for (int i = _c; i < _c + 2048 && i < HIDDEN; i++)
                            activ_buf[i] <= pcie_rx_data[i*DATA_W +: DATA_W];
                        pcie_rx_ready <= 1'b1;
                        busy      <= 1'b1;
                        expert_cnt <= '0;
                        state     <= S_LOAD_TOKEN;
                    end
                end

                // --------------------------------------------------------
                // LOAD_TOKEN: prepare systolic array for next expert
                // --------------------------------------------------------
                S_LOAD_TOKEN: begin
                    inter_row  <= '0;
                    sa_cycle   <= '0;
                    // Clear gate/up buffers
                    for (int j = 0; j < INTER; j++) begin
                        gate_buf[j] <= '0;
                        up_buf[j]   <= '0;
                    end
                    state <= S_GATE_UP_ROW;
                end

                // --------------------------------------------------------
                // GATE_UP_ROW: compute gate[i] and up[i] via systolic array
                // --------------------------------------------------------
                S_GATE_UP_ROW: begin
                    sa_valid <= 1'b1;
                    // Load activation lanes and weight lanes
                    for (int k = 0; k < DSP_LANES; k++) begin
                        sa_a[k] <= activ_buf[sa_cycle * DSP_LANES + k];
                        // Weight from HBM2 stream (simplified: zero for bringup)
                        sa_b[k] <= '0;
                    end
                    sa_cycle <= sa_cycle + 1;
                    if (sa_cycle == (HIDDEN / DSP_LANES) - 1) begin
                        sa_cycle  <= '0;
                        inter_row <= inter_row + 1;
                        // Latch systolic outputs into gate/up buffers
                        // gate_buf[inter_row] <= sa_sum[...];
                        // up_buf[inter_row]   <= sa_sum[...];
                        if (inter_row == INTER - 1)
                            state <= S_ACTIVATE;
                    end
                end

                // --------------------------------------------------------
                // ACTIVATE: SiLU(gate) × up → combined
                // --------------------------------------------------------
                S_ACTIVATE: begin
                    // Apply activation: combined[i] = SiLU(gate[i]) * up[i]
                    // Simplified for bringup: pass-through
                    for (int j = 0; j < INTER; j++)
                        combined[j] <= up_buf[j];  // placeholder
                    inter_row  <= '0;
                    hidden_col <= '0;
                    sa_cycle   <= '0;
                    state <= S_DOWN_ROW;
                end

                // --------------------------------------------------------
                // DOWN_ROW: combined[] × down_w → down_accum[hidden]
                // --------------------------------------------------------
                S_DOWN_ROW: begin
                    sa_valid <= 1'b1;
                    // Stream combined[j] and down_w[h][j] into systolic array
                    // Simplified bringup: no actual compute
                    sa_cycle <= sa_cycle + 1;
                    if (sa_cycle == (INTER / DSP_LANES) - 1) begin
                        sa_cycle   <= '0;
                        hidden_col <= hidden_col + 1;
                        if (hidden_col == HIDDEN - 1)
                            state <= S_NEXT_EXPERT;
                    end
                end

                // --------------------------------------------------------
                // NEXT_EXPERT: accumulate into output, loop to next expert
                // --------------------------------------------------------
                S_NEXT_EXPERT: begin
                    // Merge: ffn_out += expert_weight × down_accum
                    // Simplified: just accumulate
                    for (int _c = 0; _c < HIDDEN; _c += 2048)
                        for (int i = _c; i < _c + 2048 && i < HIDDEN; i++)
                        down_accum[i] <= down_accum[i] + ffn_out[i];  // placeholder
                    expert_cnt <= expert_cnt + 1;
                    if (expert_cnt == TOP_K - 1)
                        state <= S_OUTPUT;
                    else
                        state <= S_LOAD_TOKEN;
                end

                // --------------------------------------------------------
                // OUTPUT: drive FFN result on PCIe TX
                // --------------------------------------------------------
                S_OUTPUT: begin
                    pcie_tx_valid <= 1'b1;
                    for (int _c = 0; _c < HIDDEN; _c += 2048)
                        for (int i = _c; i < _c + 2048 && i < HIDDEN; i++)
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

endmodule
