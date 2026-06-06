//=============================================================================
// fp4_gemm_engine.sv — 2D Systolic GEMM Engine (PRODUCTION)
//
// Replaces legacy fp4_linear_engine. Uses fp4_systolic_2d for weight-stationary
// matrix-vector multiplication with M×K 2D parallelism.
// Parameters from lpu_config_pkg.
//=============================================================================

`include "lpu_config.svh"

module fp4_gemm_engine #(
    parameter int M_OUT       = lpu_config_pkg::LPU_HIDDEN,
    parameter int K_TOTAL     = lpu_config_pkg::LPU_HIDDEN,
    parameter int LANES       = lpu_config_pkg::LPU_ARRAY_LANES,
    parameter int M_ROWS      = lpu_config_pkg::LPU_ARRAY_M_ROWS,
    parameter int ACCUM_WIDTH = 32,
    parameter int K_BEATS     = (K_TOTAL + LANES - 1) / LANES,
    parameter int M_PASSES    = (M_OUT + M_ROWS - 1) / M_ROWS
) (
    input  logic        clk,
    input  logic        rst_n,

    // Weight preload (fp4 E2M1) + scale preload (raw fp8, decoded at load)
    input  logic                         wt_wr_en,
    input  logic [$clog2(M_OUT)-1:0]     wt_wr_row,        // global output row
    input  logic [$clog2(K_TOTAL)-1:0]   wt_wr_col,        // global input col
    input  logic [3:0]                   wt_wr_data,        // fp4 weight
    input  logic [7:0]                   sc_wr_data,        // raw fp8 scale

    // Activation preload (streamed in K_BEATS)
    input  logic                         activ_wr_en,
    input  logic [$clog2(K_BEATS)-1:0]   activ_wr_beat,
    input  logic [LANES*8-1:0]           activ_wr_data,

    // Control
    input  logic                         start,
    output logic                         busy,
    output logic                         done,

    // Result stream
    output logic                         result_valid,
    output logic [$clog2(M_OUT)-1:0]     result_row,
    output logic [ACCUM_WIDTH-1:0]       result_data,
    input  logic                         result_ready
);

    localparam int ROW_BITS   = $clog2(M_ROWS);
    localparam int COL_BITS   = $clog2(LANES);
    localparam int PASS_BITS  = $clog2(M_PASSES + 1);
    // BEAT_BITS: max of beat-index width and drain-counter width (needs 3 bits
    // for 0..6 range).  Without this headroom the counter wraps in S_DRAIN and
    // the FSM hangs forever.
    localparam int BEAT_BITS  = $clog2(K_BEATS + 8);

    typedef enum logic [3:0] {
        S_IDLE,
        S_LOAD_WEIGHTS,
        S_FEED_BEATS,
        S_DRAIN,
        S_REDUCE_WAIT,        // wait for reduction to complete
        S_READOUT,
        S_NEXT_PASS,
        S_DONE
    } state_t;
    state_t state;

    // Weight preload addressing
    logic [ROW_BITS-1:0]   load_row;     // local row within current pass
    logic [COL_BITS-1:0]   load_col;
    logic [PASS_BITS-1:0]  current_pass;
    logic [BEAT_BITS-1:0]  beat_count;
    logic [$clog2(M_OUT)-1:0] global_row; // for output

    // Activation memory (Altera syncram IP)
    logic [LANES*8-1:0] activ_q;

    altera_syncram #(.WIDTH(LANES*8), .DEPTH(K_BEATS), .RAM_BLOCK_TYPE("AUTO"))
    u_activ (
        .clock(clk), .wren(activ_wr_en), .wraddress(activ_wr_beat),
        .data(activ_wr_data), .rdaddress(beat_count), .q(activ_q)
    );

    // 2D array interfaces
    logic                          array_wt_wr_en;
    logic [ROW_BITS-1:0]           array_wt_row;
    logic [COL_BITS-1:0]           array_wt_col;
    logic [11:0]                   array_sc_data;    // pre-decoded

    logic                          array_valid;
    logic [LANES*8-1:0]            array_activ;
    logic                          array_accum_clr;
    logic                          array_reduce_start;
    logic                          array_reduce_done;
    logic [M_ROWS*ACCUM_WIDTH-1:0] array_result;

    // Result gathering
    logic [ACCUM_WIDTH-1:0] result_buf [M_ROWS-1:0];
    logic [ROW_BITS-1:0]    result_idx;

    //=========================================================================
    // 2D Systolic Array
    //=========================================================================
    fp4_systolic_2d #(
        .LANES(LANES), .M_ROWS(M_ROWS), .ACCUM_WIDTH(ACCUM_WIDTH)
    ) u_array (
        .clk           (clk),
        .rst_n         (rst_n),
        .wt_wr_en      (array_wt_wr_en),
        .wt_wr_row     (array_wt_row),
        .wt_wr_col     (array_wt_col),
        .wt_wr_data    (wt_wr_data),
        .sc_wr_data    (array_sc_data),
        .valid_in      (array_valid),
        .activ_flat    (array_activ),
        .accum_clr     (array_accum_clr),
        .reduce_start  (array_reduce_start),
        .reduce_done   (array_reduce_done),
        .result_flat   (array_result)
    );

    // Pre-decode scale at load time
    assign array_sc_data = fp8_to_scaled12(sc_wr_data);

    // Weight load address mapping: global (row, col) → local (pass_row, col)
    // Use continuous assignments (not always_comb) to avoid Icarus
    // constant-select limitation.
    assign array_wt_row = wt_wr_row[ROW_BITS-1:0];
    assign array_wt_col = wt_wr_col[COL_BITS-1:0];

    // Direct pass-through: weight preload writes go straight to the array.
    // Weights are loaded externally before start, so no FSM gating needed.
    assign array_wt_wr_en = wt_wr_en;

    // Activation memory write handled by altera_syncram u_activ

    //=========================================================================
    // Main FSM
    //=========================================================================
    assign busy = (state != S_IDLE) && (state != S_DONE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            array_valid       <= 1'b0;
            array_activ       <= '0;
            array_accum_clr   <= 1'b0;
            array_reduce_start <= 1'b0;
            current_pass      <= '0;
            beat_count        <= '0;
            load_row          <= '0;
            load_col          <= '0;
            global_row        <= '0;
            result_idx        <= '0;
            result_valid      <= 1'b0;
            result_row        <= '0;
            result_data       <= '0;
            done              <= 1'b0;
        end else begin
            array_valid       <= 1'b0;
            array_accum_clr   <= 1'b0;
            array_reduce_start <= 1'b0;
            result_valid      <= 1'b0;
            done              <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        current_pass <= '0;
                        state <= S_LOAD_WEIGHTS;
                    end
                end

                //-------------------------------------------------------------
                // Phase 1: Load weights for current M_PASS
                //
                // Weights are loaded externally via wt_wr_en. The FSM waits
                // for all weights in the current pass to be loaded.
                // In production, weights stream from HBM via DMA.
                //
                // For bring-up: weights are pre-loaded before start, so this
                // state transitions immediately.
                //-------------------------------------------------------------
                S_LOAD_WEIGHTS: begin
                    // Weight loading is externally controlled via wt_wr_en.
                    // Here we just initialize the pass.
                    array_accum_clr <= 1'b1;   // clear accumulators for new token
                    beat_count <= '0;
                    state <= S_FEED_BEATS;
                end

                //-------------------------------------------------------------
                // Phase 2: Stream K_BEATS activation beats
                //-------------------------------------------------------------
                S_FEED_BEATS: begin
                    array_valid <= 1'b1;
                    array_activ <= activ_q;

                    if (beat_count == K_BEATS - 1) begin
                        beat_count <= '0;
                        state <= S_DRAIN;
                    end else begin
                        beat_count <= beat_count + 1'b1;
                    end
                end

                //-------------------------------------------------------------
                // Phase 3: Drain MAC pipeline + reduce
                //   fp4_mac: 4 stages + activ pipeline: 1 stage = 5 cycles
                //   After drain, 1-cycle reduction, then readout
                //-------------------------------------------------------------
                S_DRAIN: begin
                    beat_count <= beat_count + 1'b1;
                    if (beat_count >= 6) begin
                        beat_count <= '0;
                        array_reduce_start <= 1'b1;
                        state <= S_REDUCE_WAIT;
                    end
                end

                S_REDUCE_WAIT: begin
                    if (array_reduce_done) begin
                        for (int r = 0; r < M_ROWS; r++) begin
                            result_buf[r] <= array_result[r*ACCUM_WIDTH +: ACCUM_WIDTH];
                        end
                        result_idx <= '0;
                        state <= S_READOUT;
                    end
                end

                //-------------------------------------------------------------
                // Phase 5: Read out results
                //-------------------------------------------------------------
                S_READOUT: begin
                    result_valid <= 1'b1;
                    result_row   <= current_pass * M_ROWS + result_idx;
                    result_data  <= result_buf[result_idx];

                    if (result_ready) begin
                        if (result_idx == M_ROWS - 1 ||
                            (current_pass * M_ROWS + result_idx == M_OUT - 1)) begin
                            state <= S_NEXT_PASS;
                        end else begin
                            result_idx <= result_idx + 1'b1;
                        end
                    end
                end

                //-------------------------------------------------------------
                // Advance to next M_PASS or finish
                //-------------------------------------------------------------
                S_NEXT_PASS: begin
                    if (current_pass == M_PASSES - 1) begin
                        done <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        current_pass <= current_pass + 1'b1;
                        state <= S_LOAD_WEIGHTS;
                    end
                end

                S_DONE: begin
                    if (!start) state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
