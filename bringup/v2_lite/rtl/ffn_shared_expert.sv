// =============================================================================
// ffn_shared_expert.sv — Shared Expert Controller (weights in SRAM)
// V2-Lite SPEC §5: 9MB shared weight in M20K, GEMV gate/up/down
//
// SRAM Layout (per shared expert):
//   gate_w: 2048×1408×1B = 2.9MB @ offset 0
//   up_w:   2048×1408×1B = 2.9MB @ offset 2.9MB
//   down_w: 1408×2048×1B = 2.9MB @ offset 5.8MB
//
// Cycles @ 250MHz, 512-MAC:
//   Gate:  4c/row × 1408 rows = 5,632c = 22.5μs
//   SiLU:  1408/64 = 22c = 0.1μs
//   Up:    5,632c = 22.5μs
//   Merge: 22c = 0.1μs
//   Down:  3c/row × 2048 rows = 6,144c = 24.6μs
//   Total: ~17.5K cycles = 70μs
// =============================================================================

module ffn_shared_expert #(
    parameter int HIDDEN      = 2048,
    parameter int INTER       = 1408,
    parameter int DSP_LANES   = 512,
    parameter int DATA_W      = 8,
    parameter int ACCUM_W     = 24
) (
    input  logic                         clk, rst_n,
    input  logic                         start,
    output logic                         busy, done,

    // Activation input
    input  logic [HIDDEN*DATA_W-1:0]     activ_in,
    input  logic                         activ_valid,
    output logic                         activ_ready,

    // FFN output
    output logic [HIDDEN*DATA_W-1:0]     ffn_out,
    output logic                         ffn_valid,

    // SRAM weight interface
    output logic [$clog2(INTER*HIDDEN):0] sram_addr,
    input  logic [DSP_LANES*DATA_W-1:0]  sram_rdata,
    output logic                          sram_read,

    // SiLU activation
    output logic [INTER-1:0][15:0]        silu_in,
    input  logic [INTER-1:0][15:0]        silu_out,
    output logic                           silu_valid,

    // GEMV array control
    output logic                         gemv_start,
    output logic [$clog2(HIDDEN):0]      gemv_rows,
    input  logic                         gemv_busy, gemv_done,
    output logic [DSP_LANES*DATA_W-1:0]  gemv_activ,
    output logic                         gemv_activ_valid,
    input  logic                         gemv_activ_ready,
    input  logic [DSP_LANES*DATA_W-1:0]  gemv_weight,
    output logic                         gemv_weight_rd,

    output logic [3:0]                   dbg_fsm
);
    typedef enum logic [3:0] { S_IDLE, S_GATE, S_SILU, S_UP, S_MERGE, S_DOWN, S_OUTPUT, S_DONE } st_t;
    st_t st;
    assign busy = (st != S_IDLE && st != S_DONE);
    assign dbg_fsm = st;
    assign activ_ready = (st == S_IDLE);

    // SRAM addressing: gate@0, up@HIDDEN*INTER, down@2*HIDDEN*INTER
    localparam int GATE_BASE  = 0;
    localparam int UP_BASE    = HIDDEN * INTER;
    localparam int DOWN_BASE  = 2 * HIDDEN * INTER;

    logic [$clog2(INTER*HIDDEN):0] row_addr;
    logic [2:0]  pipeline_stage; // gate=0, silu=1, up=2, merge=3, down=4
    logic [15:0] gate_buf [INTER];
    logic [15:0] up_buf   [INTER];
    logic [15:0] combined [INTER];
    logic [DATA_W-1:0]    ffn_accum [HIDDEN];
    logic [$clog2(INTER):0] merge_idx;

    // SiLU: gate → LUT → gate (in-place via silu_activation)
    // Hardcoded: 256-entry sigmoid LUT, 64-wide processing (see silu_activation.sv)

    // GEMV control
    assign gemv_activ = activ_in[DSP_LANES*DATA_W-1:0]; // feed first chunk
    assign gemv_weight_rd = (st != S_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; row_addr <= 0; pipeline_stage <= 0; merge_idx <= 0;
            gemv_start <= 0; sram_read <= 0; silu_valid <= 0; ffn_valid <= 0; done <= 0;
        end else begin
            done <= 0; gemv_start <= 0; sram_read <= 0; silu_valid <= 0; ffn_valid <= 0;

            case (st)
                S_IDLE: begin
                    if (start && activ_valid) begin
                        row_addr <= GATE_BASE; pipeline_stage <= 0;
                        gemv_rows <= INTER;
                        gemv_start <= 1; sram_read <= 1;
                        st <= S_GATE;
                    end
                end

                S_GATE: begin
                    sram_read <= 1;
                    if (gemv_done) begin
                        // SiLU on gate outputs
                        st <= S_SILU;
                    end
                end

                S_SILU: begin
                    silu_valid <= 1;
                    // SiLU processes gate_buf → gate_buf (64-wide, pipelined)
                    if (merge_idx < INTER) begin
                        silu_in[merge_idx] <= gate_buf[merge_idx];
                        gate_buf[merge_idx] <= silu_out[merge_idx];
                        merge_idx <= merge_idx + 1;
                    end else begin
                        merge_idx <= 0; silu_valid <= 0;
                        // Start Up projection
                        row_addr <= UP_BASE; gemv_rows <= INTER;
                        gemv_start <= 1; sram_read <= 1;
                        st <= S_UP;
                    end
                end

                S_UP: begin
                    sram_read <= 1;
                    if (gemv_done) st <= S_MERGE;
                end

                S_MERGE: begin
                    // Element-wise: combined[i] = SiLU(gate[i]) × up[i]
                    if (merge_idx < INTER) begin
                        combined[merge_idx] <= gate_buf[merge_idx]; // FP16 multiply placeholder
                        merge_idx <= merge_idx + 1;
                    end else begin
                        merge_idx <= 0;
                        row_addr <= DOWN_BASE; gemv_rows <= HIDDEN;
                        gemv_start <= 1; sram_read <= 1;
                        st <= S_DOWN;
                    end
                end

                S_DOWN: begin
                    sram_read <= 1;
                    if (gemv_done) st <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    ffn_valid <= 1;
                    for (int i = 0; i < HIDDEN; i++)
                        ffn_out[i*DATA_W +: DATA_W] <= ffn_accum[i];
                    st <= S_DONE;
                end

                S_DONE: begin done <= 1; if (!start) st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule
