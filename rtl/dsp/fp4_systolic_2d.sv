//=============================================================================
// fp4_systolic_2d.sv — 2D Weight-Stationary Systolic Array
//
// Architecture:
//   M_ROWS rows × LANES columns of fp4_systolic_cell instances.
//   Weights pre-loaded per-cell. Activations broadcast per-column.
//   Each cell accumulates independently over K_BEATS.
//   After all beats, per-row adder tree reduces LANES→1.
//
// Resource (LANES=128, M_ROWS=32):
//   Cells:     4096
//   DSP:       8192 (87% of 9375)
//   Weight:    4096 × 4b  = 16 Kb
//   Scale:     4096 × 12b = 48 Kb (MLAB distributed)
//=============================================================================

module fp4_systolic_2d #(
    parameter int LANES       = 8,       // columns (K-direction parallelism)
    parameter int M_ROWS      = 4,       // rows (M-direction parallelism)
    parameter int ACCUM_WIDTH = 32
) (
    input  logic        clk,
    input  logic        rst_n,

    // Weight/scale preload (one cell at a time)
    input  logic                         wt_wr_en,
    input  logic [$clog2(M_ROWS)-1:0]    wt_wr_row,
    input  logic [$clog2(LANES)-1:0]     wt_wr_col,
    input  logic [3:0]                   wt_wr_data,       // fp4 E2M1
    input  logic [11:0]                  sc_wr_data,       // pre-decoded fp8

    // Activation input (LANES-wide, broadcast per column)
    input  logic                         valid_in,
    input  logic [LANES*8-1:0]           activ_flat,       // LANES × fp8

    // Control
    input  logic                         accum_clr,         // new token
    input  logic                         reduce_start,      // start reduction
    output logic                         reduce_done,       // reduction complete

    // Accumulator readout (M_ROWS results after reduction)
    output logic [M_ROWS*ACCUM_WIDTH-1:0] result_flat
);

    //=========================================================================
    // Cell Grid
    //=========================================================================
    logic [LANES*8-1:0] col_activ;
    logic               valid_s0;
    logic [ACCUM_WIDTH-1:0] cell_accum [M_ROWS-1:0][LANES-1:0];

    // Weight write decode — flat wires to avoid packed 2D indexing issues
    genvar rr, cc;
    generate
        for (rr = 0; rr < M_ROWS; rr++) begin : g_wt_row
            for (cc = 0; cc < LANES; cc++) begin : g_wt_col
                logic cell_wr;
                assign cell_wr = wt_wr_en &&
                    (rr == wt_wr_row) && (cc == wt_wr_col);

                fp4_systolic_cell #(.ACCUM_WIDTH(ACCUM_WIDTH)) u_cell (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .wt_wr_en   (cell_wr),
                    .wt_wr_data (wt_wr_data),
                    .sc_wr_data (sc_wr_data),
                    .activ_in   (col_activ[cc*8 +: 8]),
                    .valid_in   (valid_s0),
                    .accum_clr  (accum_clr),
                    .accum_out  (cell_accum[rr][cc])
                );
            end
        end
    endgenerate

    //=========================================================================
    // Activation pipeline (registered to avoid fanout timing issues)
    //=========================================================================
    logic [LANES*8-1:0] activ_s0;

    always_ff @(posedge clk) begin
        activ_s0 <= activ_flat;
        valid_s0 <= valid_in;
    end

    // Column broadcast
    assign col_activ = activ_s0;

    //=========================================================================
    // Reduction: per-row combinational adder tree, 1 pipeline register
    //
    // After all K_BEATS, cell accumulators hold the partial dot products.
    // For each row: sum = cell_accum[row][0] + ... + cell_accum[row][LANES-1]
    //
    // The reduction is registered once (single pipeline stage) since it
    // only fires once per token, not in the streaming critical path.
    //=========================================================================
    localparam int RED_WIDTH = ACCUM_WIDTH + $clog2(LANES);  // guard bits

    // Combinational per-row sum
    logic [RED_WIDTH-1:0] row_sum_comb [M_ROWS-1:0];

    always_comb begin
        for (int rr = 0; rr < M_ROWS; rr++) begin
            row_sum_comb[rr] = '0;
            for (int cc = 0; cc < LANES; cc++) begin
                row_sum_comb[rr] = row_sum_comb[rr] +
                    {{(RED_WIDTH-ACCUM_WIDTH){cell_accum[rr][cc][ACCUM_WIDTH-1]}},
                     cell_accum[rr][cc]};
            end
        end
    end

    // Register the reduction result (holds reduce_done for 2 cycles)
    always_ff @(posedge clk) begin
        if (reduce_start) begin
            for (int rr = 0; rr < M_ROWS; rr++) begin
                result_flat[rr*ACCUM_WIDTH +: ACCUM_WIDTH] <=
                    row_sum_comb[rr][ACCUM_WIDTH-1:0];
            end
            reduce_done <= 1'b1;
        end else if (reduce_done) begin
            // Hold reduce_done for one extra cycle so controller can sample it
            reduce_done <= 1'b0;
        end
    end

endmodule
