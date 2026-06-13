// =============================================================================
// ffn_gemv_array.sv — 512-MAC GEMV Systolic Array
// V2-Lite SPEC §2: DSP_LANES=512, activation-stationary, weight streaming
//
// Modes:
//   DECODE (single token): 512 MAC all serve 1 activation vector
//     Gate/Up:  2048/512 = 4 cycles/row × 1408 rows = 5,632 cycles
//     Down:     1408/512 ≈ 3 cycles/row × 2048 rows = 6,144 cycles
//
//   PREFILL (multi-token): 512 MAC split into 64 tiles × 8 MAC/token
//     Each tile: 2048/8 = 256 cycles per output element
//
// FSM: IDLE → PRELOAD → STREAM → DRAIN → REDUCE → STORE → DONE
// =============================================================================

module ffn_gemv_array #(
    parameter int DSP_LANES  = 512,      // MAC parallelism
    parameter int INPUT_DIM  = 2048,     // V2-Lite hidden dim
    parameter int OUTPUT_DIM = 1408,     // V2-Lite intermediate dim
    parameter int DATA_W     = 8,        // FP8 E4M3
    parameter int ACCUM_W    = 24,       // accumulator width
    parameter int PIPE_STAGES = 3        // pipeline depth for 250MHz
) (
    input  logic                         clk, rst_n,
    input  logic                         start,
    output logic                         busy, done,

    // Activation: flat packed DSP_LANES × FP8 per beat
    input  logic                         activ_valid,
    output logic                         activ_ready,
    input  logic [DSP_LANES*DATA_W-1:0]  activ_data,

    // Weight: flat packed DSP_LANES × FP8 per beat
    input  logic                         weight_valid,
    output logic                         weight_ready,
    input  logic [DSP_LANES*DATA_W-1:0]  weight_data,

    // Weight preload handshake
    output logic                         wt_preload_req,
    output logic [$clog2(OUTPUT_DIM):0]  wt_preload_row,
    input  logic                         wt_preload_ack,

    // Result output
    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [ACCUM_W-1:0]           result_data,
    output logic [$clog2(OUTPUT_DIM):0]  result_row,
    output logic                         result_last,

    // Mode
    input  logic                         mode_prefill,    // 0=Decode, 1=Prefill
    input  logic [5:0]                   prefill_tokens,  // number of tokens in Prefill (1-64)

    // Debug
    output logic [3:0]                   dbg_fsm,
    output logic [9:0]                   dbg_cycle
);

    localparam int CYCLES_PER_ROW = INPUT_DIM / DSP_LANES;  // 2048/512 = 4
    localparam int REDUCE_DEPTH   = $clog2(DSP_LANES);      // log2(512) = 9

    typedef enum logic [3:0] { S_IDLE, S_PRELOAD, S_STREAM, S_DRAIN, S_REDUCE, S_STORE, S_DONE } st_t;
    st_t st;

    logic [$clog2(OUTPUT_DIM):0] row_cnt;
    logic [$clog2(CYCLES_PER_ROW+1):0] cyc_cnt;
    logic [$clog2(REDUCE_DEPTH+2):0] drain_cnt, reduce_cnt;
    logic        preload_active;
    logic        accum_clr;

    assign busy = (st != S_IDLE && st != S_DONE);
    assign activ_ready = (st == S_STREAM);
    assign weight_ready = (st == S_STREAM) || (st == S_PRELOAD);
    assign dbg_fsm = st;
    assign dbg_cycle = cyc_cnt;

    // Weight preload request
    assign wt_preload_req = (st == S_PRELOAD) && !preload_active;
    assign wt_preload_row = row_cnt;

    // =========================================================================
    // MAC Lanes: 512 × FP8×FP8 → sign-extend to 16-bit → DSP multiply
    // =========================================================================
    // MAC accumulator wire array (exposed for reduction tree)
    logic [DSP_LANES-1:0][ACCUM_W-1:0] mac_accum;

    genvar li;
    generate
        for (li = 0; li < DSP_LANES; li++) begin : g_mac
            logic signed [15:0] s0_a, s0_b;
            always_ff @(posedge clk) begin
                s0_a <= $signed(activ_data[li*DATA_W +: DATA_W]);
                s0_b <= $signed(weight_data[li*DATA_W +: DATA_W]);
            end

            // DSP multiply: 16×16 → 32
            (* multstyle = "dsp" *) logic signed [31:0] mult;
            assign mult = s0_a * s0_b;

            // Pipeline register for 250MHz (PIPE_STAGES=3)
            logic signed [31:0] mult_r1, mult_r2;
            always_ff @(posedge clk) begin mult_r1 <= mult; mult_r2 <= mult_r1; end

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n || accum_clr) mac_accum[li] <= 0;
                else if (st == S_STREAM || st == S_DRAIN)
                    mac_accum[li] <= mac_accum[li] + mult_r2[31:16];
            end
        end
    endgenerate

    // =========================================================================
    // Reduction Tree: 512→1, 9 stages registered
    // =========================================================================
    logic [REDUCE_DEPTH:0][DSP_LANES-1:0][ACCUM_W-1:0] reduce_tree;
    logic [REDUCE_DEPTH:0] reduce_valid;

    // Stage 0: latch MAC outputs
    always_ff @(posedge clk) begin
        for (int i = 0; i < DSP_LANES; i++)
            reduce_tree[0][i] <= mac_accum[i];
        reduce_valid[0] <= (st == S_DRAIN) && (drain_cnt == PIPE_STAGES);
    end

    genvar stage;
    generate
        for (stage = 1; stage <= REDUCE_DEPTH; stage++) begin : g_reduce
            localparam int N_IN = DSP_LANES >> (stage - 1);
            localparam int N_OUT = DSP_LANES >> stage;
            always_ff @(posedge clk) begin
                for (int pi = 0; pi < N_OUT; pi++)
                    reduce_tree[stage][pi] <= reduce_tree[stage-1][2*pi] + reduce_tree[stage-1][2*pi+1];
                reduce_valid[stage] <= reduce_valid[stage-1];
            end
        end
    endgenerate

    wire [ACCUM_W-1:0] reduced_sum = reduce_tree[REDUCE_DEPTH][0];

    // =========================================================================
    // FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; row_cnt <= 0; cyc_cnt <= 0;
            drain_cnt <= 0; reduce_cnt <= 0; preload_active <= 0;
            accum_clr <= 0; result_valid <= 0; result_data <= 0;
            result_row <= 0; result_last <= 0;
        end else begin
            result_valid <= 0; done <= 0;

            case (st)
                S_IDLE: begin
                    if (start) begin
                        row_cnt <= 0; cyc_cnt <= 0; st <= S_PRELOAD;
                    end
                end

                S_PRELOAD: begin
                    if (wt_preload_ack) begin
                        preload_active <= 1;
                        accum_clr <= 1;
                        cyc_cnt <= 0;
                        st <= S_STREAM;
                    end
                end

                S_STREAM: begin
                    accum_clr <= 0;
                    if (cyc_cnt == CYCLES_PER_ROW - 1) begin
                        cyc_cnt <= 0; drain_cnt <= 0; st <= S_DRAIN;
                    end else cyc_cnt <= cyc_cnt + 1;
                end

                S_DRAIN: begin
                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == PIPE_STAGES) begin
                        reduce_cnt <= 0; st <= S_REDUCE;
                    end
                end

                S_REDUCE: begin
                    reduce_cnt <= reduce_cnt + 1;
                    if (reduce_cnt == REDUCE_DEPTH) begin
                        result_data <= reduced_sum;
                        result_row <= row_cnt;
                        result_valid <= 1;
                        result_last <= (row_cnt == OUTPUT_DIM - 1);
                        st <= S_STORE;
                    end
                end

                S_STORE: begin
                    result_valid <= 0;
                    if (row_cnt == OUTPUT_DIM - 1) st <= S_DONE;
                    else begin
                        row_cnt <= row_cnt + 1; cyc_cnt <= 0; preload_active <= 0;
                        st <= S_PRELOAD;
                    end
                end

                S_DONE: begin
                    done <= 1;
                    if (!start) begin done <= 0; st <= S_IDLE; end
                end

                default: st <= S_IDLE;
            endcase
        end
    end

endmodule
