//=============================================================================
// systolic_array.sv — V2-Lite 64-Lane Systolic Array Controller
// Target: Intel Stratix 10 MX (1SM21BHU2F53E1VG), Quartus Pro 26.1
//
// Architecture:
//   64 fp8×fp8→fp16 MAC lanes. Activation-stationary within each row.
//   Weights stream from HBM2 via hbm2_weight_reader. Partial dot products
//   accumulate over K cycles per row, then a 64:1 pipelined adder tree
//   reduces lane results into a single row output.
//
// FSM: IDLE → WEIGHT_PRELOAD → STREAM → DRAIN → REDUCE → STORE → NEXT_ROW → DONE
//
// Parameterized for two projection types (V2-Lite):
//   Gate/Up:  INPUT_DIM=2048, OUTPUT_DIM=1408  (32 cycles/row, 1408 rows)
//   Down:     INPUT_DIM=1408, OUTPUT_DIM=2048  (22 cycles/row, 2048 rows)
//
// Double-buffering: compute row N+1 while storing row N result.
// Weight preload:   request row N+1 weights while computing row N.
// DSP chaining:     altera_mult_add cascaded with fabric accumulator.
//
// Resource (V2-Lite, DSP_LANES=64):
//   DSP blocks:  64  (multiply only, accumulation in fabric)
//   M20K:        < 4 (reduction pipeline registers)
//   ALMs:        ~6k (accumulators + reduction + control)
//=============================================================================

module systolic_array #(
    parameter int INPUT_DIM  = 2048,    // V2-Lite: activation / row dimension
    parameter int OUTPUT_DIM = 1408,    // V2-Lite: number of output rows
    parameter int DSP_LANES  = 64,      // V2-Lite: column parallelism
    parameter int DATA_W     = 8,       // FP8 E4M3
    parameter int ACCUM_W    = 24,      // accumulator width (fp16 + headroom)
    parameter logic [31:0] VERSION = 32'h0B061B01  // {day,month,year-2000,build#}
) (
    input  logic        clk,
    input  logic        rst_n,

    // ---- Control ----
    input  logic        start,
    output logic        busy,
    output logic        done,

    // ---- Activation input (DSP_LANES-wide, one beat per cycle) ----
    input  logic                               activ_valid,
    output logic                               activ_ready,
    input  logic [DSP_LANES*DATA_W-1:0]        activ_data,    // flat packed

    // ---- Weight input (DSP_LANES-wide, from hbm2_weight_reader) ----
    input  logic                               weight_valid,
    output logic                               weight_ready,
    input  logic [DSP_LANES*DATA_W-1:0]        weight_data,   // flat packed

    // ---- Weight preload request (to hbm2_weight_reader) ----
    output logic                               wt_preload_req,
    output logic [$clog2(OUTPUT_DIM)-1:0]      wt_preload_row,
    input  logic                               wt_preload_ack,

    // ---- Result output (one row result per output cycle) ----
    output logic                               result_valid,
    input  logic                               result_ready,
    output logic [ACCUM_W-1:0]                 result_data,
    output logic [$clog2(OUTPUT_DIM)-1:0]      result_row,
    output logic                               result_last,

    // ---- Observability / debug ----
    output logic [$clog2(OUTPUT_DIM)-1:0]      dbg_current_row,
    output logic [$clog2(INPUT_DIM/DSP_LANES):0] dbg_cycle_cnt,
    output logic [3:0]                          dbg_fsm_state,
    output logic                                dbg_preload_active,
    output logic                                dbg_stream_active,
    output logic [5:0]                          dbg_cycle_in_row,

    // ---- Performance counters ----
    output logic [31:0]                         perf_rows_done,
    output logic [31:0]                         perf_projections,
    output logic [31:0]                         perf_total_cycles
);

    //=========================================================================
    // Derived Parameters
    //=========================================================================
    localparam int CYCLES_PER_ROW   = INPUT_DIM / DSP_LANES;   // 2048/64=32 (gate/up) or 1408/64=22 (down)
    localparam int ROW_ADDR_W       = $clog2(OUTPUT_DIM > 1 ? OUTPUT_DIM : 2);
    localparam int CYCLE_CNT_W      = $clog2(CYCLES_PER_ROW > 1 ? CYCLES_PER_ROW + 1 : 3);
    localparam int MAC_PIPE_DEPTH   = 4;     // MAC pipeline drain cycles
    localparam int REDUCE_STAGES    = $clog2(DSP_LANES);  // 6 stages for 64 lanes
    localparam int REDUCE_WIDTH     = ACCUM_W + REDUCE_STAGES;  // guard bits
    localparam int DRAIN_CYCLES     = MAC_PIPE_DEPTH;

    //=========================================================================
    // FSM States
    //=========================================================================
    typedef enum logic [3:0] {
        S_IDLE,
        S_WT_PRELOAD,       // Request next-row weights from HBM2 reader
        S_WT_PRELOAD_WAIT,  // Wait for weight preload acknowledgement
        S_STREAM,           // Stream activations + weights into MAC lanes
        S_DRAIN,            // Drain MAC pipeline after last beat
        S_REDUCE,           // Pipelined reduction: 64 lanes → 1 result
        S_STORE,            // Store result to output + trigger next row preload
        S_NEXT_ROW,         // Increment row counter, loop or finish
        S_DONE              // Assert done, wait for start deassertion
    } state_t;

    state_t state, next_state;

    //=========================================================================
    // Control Registers
    //=========================================================================
    logic [ROW_ADDR_W-1:0]       current_row;
    logic [CYCLE_CNT_W-1:0]      cycle_count;
    logic [$clog2(DRAIN_CYCLES+2):0] drain_count;
    logic [$clog2(REDUCE_STAGES+2):0] reduce_count;
    logic                         preload_pending;
    logic                         result_buf_sel;   // 0/1 for double-buffered output

    //=========================================================================
    // MAC Lane Signals
    //=========================================================================
    logic                         mac_accum_clr;
    logic [DSP_LANES-1:0][DATA_W-1:0] mac_activ;
    logic [DSP_LANES-1:0][DATA_W-1:0] mac_weight;
    logic [DSP_LANES-1:0][ACCUM_W-1:0] mac_accum;
    logic [DSP_LANES-1:0]          mac_valid;

    //=========================================================================
    // Weight Preload Double-Buffering
    //=========================================================================
    logic [ROW_ADDR_W-1:0]       preload_next_row;
    logic                         preload_active;

    //=========================================================================
    // Result Double-Buffer
    //=========================================================================
    logic [1:0][ACCUM_W-1:0]     result_dbuf;
    logic [1:0][ROW_ADDR_W-1:0]  result_row_dbuf;
    logic [1:0]                  result_valid_dbuf;
    logic                         result_sel_write;  // which buffer to write
    logic                         result_sel_read;   // which buffer to read (output)

    //=========================================================================
    // MAC Lanes — Generate 64 fp8×fp8 multiply-accumulate instances
    //
    // Each lane:  input reg → altera_mult_add(DSP) → fabric accumulator
    // Pipeline:   Stage 0 (input reg) → Stage 1-2 (DSP mult) → Stage 3 (accum)
    // Total:      4 cycles from input to valid accumulator output.
    //
    // The altera_mult_add wrapper provides behavioral Icarus simulation
    // and Quartus inference to Stratix 10 variable-precision DSP blocks.
    //=========================================================================
    genvar li;
    generate
        for (li = 0; li < DSP_LANES; li++) begin : g_mac_lane

            // Stage 0: input registers
            logic signed [DATA_W-1:0] s0_a, s0_b;
            logic                     s0_v;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s0_a <= '0;
                    s0_b <= '0;
                    s0_v <= 1'b0;
                end else begin
                    s0_a <= $signed(mac_activ[li]);
                    s0_b <= $signed(mac_weight[li]);
                    s0_v <= 1'b1;  // valid is implicit during STREAM
                end
            end

            // Stage 1-2: DSP multiply (8b signed × 8b signed → 16b, 2 pipe stages)
            wire signed [15:0] product_full;
            logic signed [15:0] s2_product;
            logic               s2_v;

            altera_mult_add #(
                .A_WIDTH(DATA_W),
                .B_WIDTH(DATA_W),
                .PIPE_STAGES(2)
            ) u_dsp_mult (
                .clock (clk),
                .a     (s0_a),
                .b     (s0_b),
                .result(product_full)
            );

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s2_product <= 16'sd0;
                    s2_v       <= 1'b0;
                end else begin
                    s2_product <= product_full[15:0];
                    s2_v       <= s0_v;
                end
            end

            // Stage 3: Accumulate (saturating add to prevent overflow wrap)
            logic [ACCUM_W-1:0] lane_accum;

            function automatic logic [ACCUM_W-1:0] sat_add;
                input [ACCUM_W-1:0] a_val;
                input [ACCUM_W-1:0] b_val;
                logic [ACCUM_W-1:0] sum_raw;
                logic a_s, b_s, s_s;
                begin
                    sum_raw = a_val + b_val;
                    a_s = a_val[ACCUM_W-1];
                    b_s = b_val[ACCUM_W-1];
                    s_s = sum_raw[ACCUM_W-1];
                    if (!a_s && !b_s && s_s)
                        sat_add = {1'b0, {(ACCUM_W-1){1'b1}}};   // positive overflow → max
                    else if (a_s && b_s && !s_s)
                        sat_add = {1'b1, {(ACCUM_W-1){1'b0}}};    // negative overflow → min
                    else
                        sat_add = sum_raw;
                end
            endfunction

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    lane_accum <= '0;
                    mac_valid[li] <= 1'b0;
                end else if (mac_accum_clr) begin
                    lane_accum <= '0;
                    mac_valid[li] <= 1'b0;
                end else begin
                    if (s2_v) begin
                        lane_accum <= sat_add(lane_accum,
                            {{(ACCUM_W-16){s2_product[15]}}, s2_product});
                    end
                    mac_valid[li] <= s2_v;
                end
            end

            assign mac_accum[li] = lane_accum;
        end
    endgenerate

    //=========================================================================
    // Activation / Weight Fanout
    //=========================================================================
    always_comb begin
        for (int k = 0; k < DSP_LANES; k++) begin
            mac_activ[k]  = activ_data[k*DATA_W +: DATA_W];
            mac_weight[k] = weight_data[k*DATA_W +: DATA_W];
        end
    end

    //=========================================================================
    // Pipelined Reduction — 64:1 Adder Tree (6 registered stages)
    //
    // Each stage reduces pairs of values: N → N/2. Stage outputs registered.
    // Uses REDUCE_WIDTH = ACCUM_W + $clog2(DSP_LANES) to prevent overflow.
    //=========================================================================
    localparam int N_STAGES   = REDUCE_STAGES;   // 6 for 64 lanes
    localparam int N_LANES    = DSP_LANES;        // 64

    // Reduction tree: stage_reg[stage][index] holds stage output
    // Packed flat for each stage since we have different widths at each level
    logic [N_STAGES:0][N_LANES-1:0][REDUCE_WIDTH-1:0] reduce_tree;
    logic [N_STAGES:0]                                 reduce_valid;

    // Stage 0: sign-extend MAC accumulators
    always_ff @(posedge clk) begin
        for (int li = 0; li < N_LANES; li++) begin
            reduce_tree[0][li] <= {{(REDUCE_WIDTH-ACCUM_W){mac_accum[li][ACCUM_W-1]}},
                                    mac_accum[li]};
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            reduce_valid[0] <= 1'b0;
        else
            reduce_valid[0] <= (state == S_DRAIN) && (drain_count == DRAIN_CYCLES);
    end

    // Stages 1..N_STAGES: pairwise add + register
    genvar stage, idx;
    generate
        for (stage = 1; stage <= N_STAGES; stage++) begin : g_stage
            localparam int N_IN  = N_LANES >> (stage - 1);
            localparam int N_OUT = N_LANES >> stage;
            localparam int PAIRS = N_OUT;

            always_ff @(posedge clk) begin
                for (int pi = 0; pi < PAIRS; pi++) begin
                    reduce_tree[stage][pi] <=
                        reduce_tree[stage-1][2*pi] + reduce_tree[stage-1][2*pi + 1];
                end
            end

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    reduce_valid[stage] <= 1'b0;
                else
                    reduce_valid[stage] <= reduce_valid[stage-1];
            end
        end
    endgenerate

    // Final reduced value (from last stage, lane 0)
    wire [REDUCE_WIDTH-1:0] reduced_sum;
    assign reduced_sum = reduce_tree[N_STAGES][0];

    //=========================================================================
    // Main Controller FSM
    //=========================================================================
    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign dbg_current_row   = current_row;
    assign dbg_cycle_cnt     = cycle_count;
    assign dbg_fsm_state     = state;
    assign dbg_preload_active = (state == S_WT_PRELOAD) || (state == S_WT_PRELOAD_WAIT);
    assign dbg_stream_active  = (state == S_STREAM);
    assign dbg_cycle_in_row   = cycle_count[5:0];

    // Performance counters
    logic [31:0] _perf_rows, _perf_proj, _perf_cycles;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            _perf_rows   <= 32'd0;
            _perf_proj   <= 32'd0;
            _perf_cycles <= 32'd0;
        end else begin
            if (busy)
                _perf_cycles <= _perf_cycles + 32'd1;
            if (state == S_STORE)
                _perf_rows <= _perf_rows + 32'd1;
            if (state == S_DONE)
                _perf_proj <= _perf_proj + 32'd1;
        end
    end
    assign perf_rows_done    = _perf_rows;
    assign perf_projections  = _perf_proj;
    assign perf_total_cycles = _perf_cycles;

    // Preload row calculation (next row, wraps to 0 after last)
    wire [ROW_ADDR_W-1:0] next_row_calc;
    assign next_row_calc = (current_row == OUTPUT_DIM - 1) ? '0 : current_row + 1'b1;

    // Activation / Weight ready during STREAM state
    assign activ_ready = (state == S_STREAM);
    assign weight_ready = (state == S_STREAM);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            current_row        <= '0;
            cycle_count        <= '0;
            drain_count        <= '0;
            reduce_count       <= '0;
            preload_pending    <= 1'b0;
            preload_active     <= 1'b0;
            preload_next_row   <= '0;
            mac_accum_clr      <= 1'b0;
            result_sel_write   <= 1'b0;
            result_sel_read    <= 1'b0;
            result_valid_dbuf[0] <= 1'b0;
            result_valid_dbuf[1] <= 1'b0;
            result_dbuf[0]     <= '0;
            result_dbuf[1]     <= '0;
            result_row_dbuf[0] <= '0;
            result_row_dbuf[1] <= '0;
            done               <= 1'b0;
            wt_preload_req     <= 1'b0;
            wt_preload_row     <= '0;
            // Result output
            result_valid       <= 1'b0;
            result_data        <= '0;
            result_row         <= '0;
            result_last        <= 1'b0;
        end else begin
            // Defaults
            done             <= 1'b0;
            mac_accum_clr    <= 1'b0;
            wt_preload_req   <= 1'b0;

            // Result output handshake (STORE state drives this)
            if (result_valid && result_ready) begin
                result_valid <= 1'b0;
                // Mark the read buffer as consumed
                result_valid_dbuf[result_sel_read] <= 1'b0;
            end

            case (state)

                // ==============================================================
                // IDLE — Wait for start
                // ==============================================================
                S_IDLE: begin
                    if (start) begin
                        current_row      <= '0;
                        cycle_count      <= '0;
                        result_sel_write <= 1'b0;
                        result_sel_read  <= 1'b0;
                        result_valid_dbuf[0] <= 1'b0;
                        result_valid_dbuf[1] <= 1'b0;
                        // Preload row 0's weights
                        wt_preload_req   <= 1'b1;
                        wt_preload_row   <= '0;
                        preload_next_row <= '0;
                        state <= S_WT_PRELOAD;
                    end
                end

                // ==============================================================
                // WEIGHT_PRELOAD — Request weight preload from HBM2 reader
                // ==============================================================
                S_WT_PRELOAD: begin
                    if (!preload_active) begin
                        wt_preload_req <= 1'b1;
                        wt_preload_row <= current_row;
                        preload_active <= 1'b1;
                    end
                    state <= S_WT_PRELOAD_WAIT;
                end

                // ==============================================================
                // WEIGHT_PRELOAD_WAIT — Wait for HBM2 reader acknowledgement
                // ==============================================================
                S_WT_PRELOAD_WAIT: begin
                    if (wt_preload_ack) begin
                        preload_active <= 1'b0;
                        mac_accum_clr  <= 1'b1;  // clear MAC accumulators
                        cycle_count    <= '0;
                        state <= S_STREAM;
                    end
                end

                // ==============================================================
                // STREAM — Feed activations + weights into MAC lanes
                //
                // Each cycle: DSP_LANES activations × DSP_LANES weights
                // Total: CYCLES_PER_ROW cycles per row
                //
                // During the LAST cycle of the current row, request preload
                // for the NEXT row's weights (unless this is the last row).
                // ==============================================================
                S_STREAM: begin
                    // Only advance when valid data is consumed on both channels.
                    // This handles the HBM2 fill latency — the systolic array
                    // waits for weight_valid before processing each beat.
                    if (weight_valid && activ_valid) begin
                        if (cycle_count == CYCLES_PER_ROW - 2 && current_row != OUTPUT_DIM - 1) begin
                            // Second-to-last cycle: trigger preload for next row
                            wt_preload_req <= 1'b1;
                            wt_preload_row <= current_row + 1'b1;
                            preload_next_row <= current_row + 1'b1;
                        end

                        if (cycle_count == CYCLES_PER_ROW - 1) begin
                            // Last cycle of this row
                            cycle_count <= '0;
                            drain_count <= '0;
                            state <= S_DRAIN;
                        end else begin
                            cycle_count <= cycle_count + 1'b1;
                        end
                    end
                end

                // ==============================================================
                // DRAIN — Wait for MAC pipeline to flush
                // ==============================================================
                S_DRAIN: begin
                    if (drain_count == DRAIN_CYCLES) begin
                        reduce_count <= '0;
                        state <= S_REDUCE;
                    end else begin
                        drain_count <= drain_count + 1'b1;
                    end
                end

                // ==============================================================
                // REDUCE — Pipelined 64:1 adder tree
                //
                // The reduction tree was launched when drain completed
                // (reduce_valid[0] asserted). Each pipeline stage takes 1 cycle.
                // After N_STAGES cycles, the result is available.
                // ==============================================================
                S_REDUCE: begin
                    if (reduce_count == REDUCE_STAGES) begin
                        // Reduction complete: latch result into write buffer
                        result_dbuf[result_sel_write]
                            <= reduced_sum[ACCUM_W-1:0];
                        result_row_dbuf[result_sel_write] <= current_row;
                        result_valid_dbuf[result_sel_write] <= 1'b1;
                        state <= S_STORE;
                    end else begin
                        reduce_count <= reduce_count + 1'b1;
                    end
                end

                // ==============================================================
                // STORE — Output result from the read buffer (double-buffered)
                //
                // While storing row N's result, row N+1 is already preloading
                // or computing. This overlaps STORE with the next COMPUTE.
                // ==============================================================
                S_STORE: begin
                    // Present result from read buffer
                    if (!result_valid && result_valid_dbuf[result_sel_read]) begin
                        result_valid <= 1'b1;
                        result_data  <= result_dbuf[result_sel_read];
                        result_row   <= result_row_dbuf[result_sel_read];
                        result_last  <= (result_row_dbuf[result_sel_read]
                                         == OUTPUT_DIM - 1);
                        // Swap buffers: next write goes to the OTHER buffer
                        result_sel_write <= ~result_sel_write;
                        // Advance read pointer to the buffer we just wrote
                        result_sel_read  <= result_sel_write;
                        state <= S_NEXT_ROW;
                    end
                end

                // ==============================================================
                // NEXT_ROW — Check if all rows complete
                // ==============================================================
                S_NEXT_ROW: begin
                    if (current_row == OUTPUT_DIM - 1) begin
                        state <= S_DONE;
                    end else begin
                        current_row <= current_row + 1'b1;
                        state <= S_WT_PRELOAD;
                    end
                end

                // ==============================================================
                // DONE — All rows processed
                // ==============================================================
                S_DONE: begin
                    done <= 1'b1;
                    // Drain any remaining result
                    if (!result_valid && result_valid_dbuf[result_sel_read]) begin
                        result_valid <= 1'b1;
                        result_data  <= result_dbuf[result_sel_read];
                        result_row   <= result_row_dbuf[result_sel_read];
                        result_last  <= 1'b1;
                        result_valid_dbuf[result_sel_read] <= 1'b0;
                    end
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    //=========================================================================
    // Assertions (synthesis translate_off)
    //=========================================================================
    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (state == S_STREAM) begin
            if (cycle_count >= CYCLES_PER_ROW) begin
                $error("[SYSARR] cycle_count overflow: %0d >= %0d",
                       cycle_count, CYCLES_PER_ROW);
            end
        end
        if (state == S_REDUCE && reduce_count > REDUCE_STAGES + 1) begin
            $error("[SYSARR] reduce_count overflow: %0d", reduce_count);
        end
    end
    // synthesis translate_on

endmodule
