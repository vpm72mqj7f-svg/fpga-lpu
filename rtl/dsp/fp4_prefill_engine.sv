//=============================================================================
// fp4_prefill_engine.sv — Prefill-Mode GEMM Engine (P0)
//
// STATUS: Reserved for future FPGA-side prefill (not active in current
// CPU-prefill-only architecture). Kept for reference and future extension.
//
// Wraps fp4_gemm_engine for batched prefill (P tokens in parallel).
// In prefill mode, M_ROWS rows process different tokens with shared weights.
//
// Operation:
//   For each output dimension d (0..M_OUT-1, step M_ROWS):
//     Load W[d:d+M_ROWS-1, :] into array
//     For each token batch b (0..ceil(P/M_ROWS)-1):
//       Feed activation batch[b*M_ROWS : (b+1)*M_ROWS] through array
//       Collect M_ROWS results (one per token × one output dim)
//
// Total passes: ceil(M_OUT/M_ROWS) × ceil(P/M_ROWS)
//
// Target: P=128 tokens, M_ROWS=32 → 4 batch passes × 224 output passes
//         = 896 passes, each ~61 cycles → ~55K cycles @ 450MHz = 122us
//=============================================================================
// STATUS: Reserved for future FPGA-side prefill (not active in current
//         CPU-prefill-only architecture). The production decode path does
//         not instantiate this module; all prefill runs on the host CPU.
//         Retained for possible future hybrid prefill tier and as a
//         reference for the batched-GEMM design pattern.

`include "lpu_config.svh"

module fp4_prefill_engine #(
    parameter int M_OUT       = lpu_config_pkg::LPU_HIDDEN,
    parameter int K_TOTAL     = lpu_config_pkg::LPU_HIDDEN,
    parameter int LANES       = lpu_config_pkg::LPU_ARRAY_LANES,
    parameter int M_ROWS      = lpu_config_pkg::LPU_ARRAY_M_ROWS,
    parameter int ACCUM_WIDTH = lpu_config_pkg::LPU_ACCUM_WIDTH,
    parameter int MAX_BATCH   = 128,
    parameter int K_BEATS     = (K_TOTAL + LANES - 1) / LANES,
    parameter int M_PASSES    = (M_OUT + M_ROWS - 1) / M_ROWS,
    parameter int B_PASSES    = (MAX_BATCH + M_ROWS - 1) / M_ROWS
) (
    input  logic        clk,
    input  logic        rst_n,

    // Weight preload (shared across batch)
    input  logic                         wt_wr_en,
    input  logic [$clog2(M_OUT)-1:0]     wt_wr_row,
    input  logic [$clog2(K_TOTAL)-1:0]   wt_wr_col,
    input  logic [3:0]                   wt_wr_data,
    input  logic [7:0]                   sc_wr_data,

    // Activation preload (P tokens × K_BEATS beats)
    input  logic                         activ_wr_en,
    input  logic [$clog2(MAX_BATCH)-1:0] activ_wr_token,   // which token
    input  logic [BEAT_WIDTH-1:0]        activ_wr_beat,
    input  logic [LANES*8-1:0]           activ_wr_data,

    // Control
    input  logic [$clog2(MAX_BATCH+1)-1:0] batch_size,    // actual P (0..MAX_BATCH)
    input  logic                         start,
    output logic                         busy,
    output logic                         done,

    // Result stream: [token_idx, output_dim, value]
    output logic                         result_valid,
    output logic [$clog2(MAX_BATCH)-1:0] result_token,
    output logic [$clog2(M_OUT)-1:0]     result_row,
    output logic [ACCUM_WIDTH-1:0]       result_data,
    input  logic                         result_ready
);

    localparam int ROW_BITS  = $clog2(M_ROWS);
    localparam int PASS_BITS = $clog2(M_PASSES + 1);
    localparam int BATCH_BITS = $clog2(B_PASSES + 1);
    // Safe beat-index width: ensure >=1 bit even when K_BEATS=1
    localparam int BEAT_WIDTH = $clog2(K_BEATS > 1 ? K_BEATS : 2);
    // Drain-counter width: must count 0..6 (requires 3 bits), independent of
    // K_BEATS.  Without this headroom the counter wraps and the FSM hangs.
    localparam int DRAIN_BITS = 3;  // $clog2(7)

    typedef enum logic [2:0] {
        S_IDLE, S_LOAD_W, S_FEED, S_DRAIN, S_REDUCE, S_OUTPUT, S_NEXT, S_DONE
    } state_t;
    state_t state;

    logic [PASS_BITS-1:0]  m_pass;       // which output-dim pass
    logic [BATCH_BITS-1:0] b_pass;       // which batch pass
    logic [ROW_BITS-1:0]   row_idx;      // row within current batch
    logic [ROW_BITS:0]     valid_rows;   // how many rows active this batch

    // Activation memory: MAX_BATCH banks × K_BEATS beats × LANES×8b
    // Banked Altera syncram IP — one bank per token for parallel reads
    logic [MAX_BATCH-1:0]            bank_wr_en;
    logic [LANES*8-1:0]              activ_q [MAX_BATCH-1:0];

    always_comb begin
        bank_wr_en = '0;
        if (activ_wr_en)
            bank_wr_en[activ_wr_token] = 1'b1;
    end

    for (genvar t = 0; t < MAX_BATCH; t++) begin : gen_token_bank
        altera_syncram #(.WIDTH(LANES*8), .DEPTH(K_BEATS), .RAM_BLOCK_TYPE("AUTO"))
        u_activ (
            .clock(clk), .wren(bank_wr_en[t]), .wraddress(activ_wr_beat),
            .data(activ_wr_data), .rdaddress(beat_cnt), .q(activ_q[t])
        );
    end

    // 2D Array interfaces
    logic                          arr_wt_wr_en;
    logic [ROW_BITS-1:0]           arr_wt_row;
    logic [$clog2(LANES)-1:0]      arr_wt_col;
    logic                          arr_valid;
    logic [LANES*8-1:0]            arr_activ;
    logic                          arr_accum_clr;
    logic                          arr_reduce_start;
    logic                          arr_reduce_done;
    logic [M_ROWS*ACCUM_WIDTH-1:0] arr_result;

    // Weight loading: replicate same weight row across all M_ROWS
    logic [$clog2(M_OUT)-1:0] weight_row_base;  // first output dim in this pass

    assign weight_row_base = m_pass * M_ROWS;

    // Weight load address mapping (continuous assignments to avoid Icarus
    // constant-select limitation in always_comb)
    assign arr_wt_row   = wt_wr_row[ROW_BITS-1:0];  // local row
    assign arr_wt_col   = wt_wr_col[$clog2(LANES)-1:0];
    assign arr_wt_wr_en = wt_wr_en;

    // Activation memory write: handled by bank_wr_en decode + altera_syncram

    // ── 2D Systolic Array ──
    wire [11:0] arr_sc_data = fp8_to_scaled12(sc_wr_data);

    fp4_systolic_2d #(.LANES(LANES), .M_ROWS(M_ROWS), .ACCUM_WIDTH(ACCUM_WIDTH))
    u_array (
        .clk, .rst_n,
        .wt_wr_en    (arr_wt_wr_en),
        .wt_wr_row   (arr_wt_row),
        .wt_wr_col   (arr_wt_col),
        .wt_wr_data  (wt_wr_data),
        .sc_wr_data  (arr_sc_data),
        .valid_in    (arr_valid),
        .activ_flat  (arr_activ),
        .accum_clr   (arr_accum_clr),
        .reduce_start(arr_reduce_start),
        .reduce_done (arr_reduce_done),
        .result_flat (arr_result)
    );

    // ── Activation feed: one beat per cycle, broadcast to array ──
    logic [DRAIN_BITS-1:0] beat_cnt;  // sized for drain (0..6), not just K_BEATS
    logic [ROW_BITS-1:0]   token_base;  // first token in current batch

    assign token_base = b_pass * M_ROWS;
    assign valid_rows = (batch_size - token_base >= M_ROWS) ?
                        M_ROWS : (batch_size - token_base);

    // Mux: for each row, select the right token's activation bank
    // Row r → token (token_base + r), if within valid_rows
    always_comb begin
        arr_activ = '0;
        for (int r = 0; r < M_ROWS; r++) begin
            if (r < valid_rows) begin
                arr_activ[r*LANES*8 +: LANES*8] =
                    activ_q[token_base + r];
            end
        end
    end

    // ── Main FSM ──
    assign busy = (state != S_IDLE) && (state != S_DONE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            m_pass     <= '0;
            b_pass     <= '0;
            beat_cnt   <= '0;
            arr_valid  <= 1'b0;
            arr_accum_clr   <= 1'b0;
            arr_reduce_start <= 1'b0;
            result_valid <= 1'b0;
            done        <= 1'b0;
        end else begin
            arr_valid  <= 1'b0;
            arr_accum_clr   <= 1'b0;
            arr_reduce_start <= 1'b0;
            result_valid <= 1'b0;
            done        <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        m_pass   <= '0;
                        b_pass   <= '0;
                        beat_cnt <= '0;
                        state <= S_LOAD_W;
                    end
                end

                // Wait for weight loading (external wt_wr_en controls this)
                // In production, weights stream from HBM in background
                S_LOAD_W: begin
                    arr_accum_clr <= 1'b1;  // clear for new batch
                    beat_cnt <= '0;
                    state <= S_FEED;
                end

                // Feed K_BEATS activation beats
                S_FEED: begin
                    arr_valid <= 1'b1;
                    if (beat_cnt == K_BEATS - 1) begin
                        beat_cnt <= '0;
                        state <= S_DRAIN;
                    end else begin
                        beat_cnt <= beat_cnt + 1'b1;
                    end
                end

                // Drain pipeline (6 cycles) + trigger reduction
                S_DRAIN: begin
                    beat_cnt <= beat_cnt + 1'b1;
                    if (beat_cnt >= 6) begin
                        beat_cnt <= '0;
                        arr_reduce_start <= 1'b1;
                        state <= S_REDUCE;
                    end
                end

                // Wait for reduction result
                S_REDUCE: begin
                    if (arr_reduce_done) begin
                        row_idx <= '0;
                        state <= S_OUTPUT;
                    end
                end

                // Output results: valid_rows entries
                S_OUTPUT: begin
                    result_valid <= 1'b1;
                    result_token <= token_base + row_idx;
                    result_row   <= weight_row_base;
                    result_data  <= arr_result[row_idx*ACCUM_WIDTH +: ACCUM_WIDTH];

                    if (result_ready) begin
                        if (row_idx == valid_rows - 1) begin
                            state <= S_NEXT;
                        end else begin
                            row_idx <= row_idx + 1'b1;
                        end
                    end
                end

                // Advance: next batch pass, or next output pass, or done
                S_NEXT: begin
                    if (b_pass == B_PASSES - 1 ||
                        (b_pass + 1) * M_ROWS >= batch_size) begin
                        // All tokens done for this output pass
                        b_pass <= '0;
                        if (m_pass == M_PASSES - 1) begin
                            done <= 1'b1;
                            state <= S_DONE;
                        end else begin
                            m_pass <= m_pass + 1'b1;
                            state <= S_LOAD_W;
                        end
                    end else begin
                        b_pass <= b_pass + 1'b1;
                        state <= S_LOAD_W;
                    end
                end

                S_DONE: if (!start) state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
