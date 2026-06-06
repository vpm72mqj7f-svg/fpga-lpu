//=============================================================================
// fp4_linear_engine.sv — tiny fp4 linear layer engine prototype
//
// Computes M_OUT output channels for one activation vector of K_TOTAL elements.
// This is an RTL bring-up model: preload small on-chip memories, then start.
//=============================================================================

module fp4_linear_engine #(
    parameter int M_OUT       = 2,
    parameter int K_TOTAL     = 8,
    parameter int LANES       = 4,
    parameter int GROUP_SIZE  = 4,
    parameter int NUM_GROUPS  = 8,
    parameter int ADDR_WIDTH  = $clog2(NUM_GROUPS),
    parameter int ACCUM_WIDTH = 32,
    parameter int K_BEATS     = (K_TOTAL + LANES - 1) / LANES,
    parameter int BEAT_W      = $clog2(K_BEATS > 1 ? K_BEATS : 2),
    parameter int NUM_EXPERTS = 1,
    parameter string NAME     = "le"  // debug: instance name
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Preload ports
    input  logic                         weight_wr_en,
    input  logic [$clog2(M_OUT)-1:0]     weight_wr_row,
    input  logic [BEAT_W-1:0]            weight_wr_beat,
    input  logic [LANES*4-1:0]           weight_wr_data,

    // Expert select (0 when NUM_EXPERTS=1)
    input  logic [$clog2(NUM_EXPERTS > 1 ? NUM_EXPERTS : 2)-1:0] expert_sel,

    input  logic                         activ_wr_en,
    input  logic [BEAT_W-1:0]            activ_wr_beat,
    input  logic [LANES*8-1:0]           activ_wr_data,

    input  logic                         scale_wr_en,
    input  logic [ADDR_WIDTH-1:0]        scale_wr_addr,
    input  logic [7:0]                   scale_wr_data,

    // Run control
    input  logic                         start,
    output logic                         busy,
    output logic                         done,

    // Result stream: one row per pulse
    output logic                         result_valid,
    output logic [$clog2(M_OUT)-1:0]     result_row,
    output logic [ACCUM_WIDTH-1:0]       result_data,
    input  logic                         result_ready
);

    typedef enum logic [2:0] {
        S_IDLE,
        S_ARRAY_START,
        S_FEED,
        S_WAIT,
        S_RESULT,
        S_DONE
    } state_t;

    state_t state;
    logic [$clog2(M_OUT>1 ? M_OUT : 2)-1:0] row_idx;
    logic [BEAT_W-1:0] beat_idx;

    // Weight/activation preload — Altera syncram IP
    // Multi-expert: weight RAM depth multiplied by NUM_EXPERTS, expert_sel offsets
    localparam int WT_DEPTH = M_OUT * K_BEATS * NUM_EXPERTS;
    localparam int EXPERT_STRIDE = M_OUT * K_BEATS;

    logic [LANES*4-1:0] weight_q;
    logic [LANES*8-1:0] activ_q;
    logic [$clog2(WT_DEPTH > 1 ? WT_DEPTH : 2)-1:0] wt_wr_addr, wt_rd_addr;

    assign wt_wr_addr = expert_sel * EXPERT_STRIDE + weight_wr_row * K_BEATS + weight_wr_beat;
    assign wt_rd_addr = expert_sel * EXPERT_STRIDE + row_idx * K_BEATS + beat_idx;

`ifdef DBG_PIPELINE
    // Capture preload signals at posedge for debug
    int le_dbg_cycle;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) le_dbg_cycle <= 0;
        else le_dbg_cycle <= le_dbg_cycle + 1;
    end
    always_ff @(posedge clk) begin
        if (weight_wr_en)
            $display("  [LE_PRE] %s wt_wr: cyc=%0d row=%0d beat=%0d addr=%0d data=0x%04x",
                     NAME, le_dbg_cycle, weight_wr_row, weight_wr_beat, wt_wr_addr, weight_wr_data);
    end
`endif

    altera_syncram #(.WIDTH(LANES*4), .DEPTH(WT_DEPTH), .RAM_BLOCK_TYPE("M20K"))
    u_weight (.clock(clk), .wren(weight_wr_en), .wraddress(wt_wr_addr),
              .data(weight_wr_data), .rdaddress(wt_rd_addr), .q(weight_q));

    altera_syncram #(.WIDTH(LANES*8), .DEPTH(K_BEATS), .RAM_BLOCK_TYPE("MLAB"))
    u_activ (.clock(clk), .wren(activ_wr_en), .wraddress(activ_wr_beat),
             .data(activ_wr_data), .rdaddress(beat_idx), .q(activ_q));

    logic array_start;
    logic array_k_valid;
    logic array_k_last;
    logic array_k_ready;
    logic array_busy;
    logic array_done;
    logic [15:0] array_elem_idx_base;
    logic [LANES*4-1:0] array_weight_flat;
    logic [LANES*8-1:0] array_activ_flat;
    logic [ACCUM_WIDTH-1:0] array_sum;
    logic [LANES*ACCUM_WIDTH-1:0] array_lanes;

    assign busy = (state != S_IDLE) && (state != S_DONE);

    fp4_systolic_array #(
        .LANES(LANES),
        .NUM_GROUPS(NUM_GROUPS),
        .GROUP_SIZE(GROUP_SIZE),
        .ADDR_WIDTH(ADDR_WIDTH),
        .ELEM_WIDTH(16),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .DRAIN_CYCLES(16)
    ) u_array (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (array_start),
        .k_valid          (array_k_valid),
        .k_last           (array_k_last),
        .elem_idx_base    (array_elem_idx_base),
        .weight_fp4_flat  (array_weight_flat),
        .activ_fp8_flat   (array_activ_flat),
        .k_ready          (array_k_ready),
        .scale_wr_en      (scale_wr_en),
        .scale_wr_addr    (scale_wr_addr),
        .scale_wr_data    (scale_wr_data),
        .busy             (array_busy),
        .result_valid     (array_done),
        .result_ready     (result_ready),
        .sum_result       (array_sum),
        .lane_result_flat (array_lanes)
    );

    //=========================================================================
    // Data source selection (TALOS-V2 pattern: explicit case-select mapping)
    //
    // The systolic array is time-multiplexed: same hardware, different data
    // sources per row. For multi-operation future use (Q/K/V/FC1/FC2),
    // add cases to the always_comb below.
    //=========================================================================
    always_comb begin
        // Default: current row/beat from Altera syncram
        array_weight_flat   = weight_q;
        array_activ_flat    = activ_q;
        array_elem_idx_base = beat_idx * LANES;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            row_idx <= '0;
            beat_idx <= '0;
            array_start <= 1'b0;
            array_k_valid <= 1'b0;
            array_k_last <= 1'b0;
            done <= 1'b0;
            result_valid <= 1'b0;
            result_row <= '0;
            result_data <= '0;
        end else begin
            array_start <= 1'b0;
            array_k_valid <= 1'b0;
            array_k_last <= 1'b0;
            done <= 1'b0;
            result_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        row_idx <= '0;
                        beat_idx <= '0;
                        array_start <= 1'b1;
                        state <= S_ARRAY_START;
                    end
                end

                S_ARRAY_START: begin
                    // Assert k_valid here so systolic array captures beat_idx=0 data
                    // on the next cycle (one cycle before S_FEED updates beat_idx).
                    // This fixes the beat-ordering race where k_valid and beat_idx
                    // were both updated in the same NBA cycle.
`ifdef DBG_PIPELINE
                    $display("  [LE_DBG] row=%0d beat=%0d wt_q=0x%04x act_q=0x%08x rdaddr=%0d",
                             row_idx, beat_idx, weight_q, activ_q,
                             row_idx * K_BEATS + beat_idx);
`endif
                    array_k_valid <= 1'b1;
                    array_k_last  <= (K_BEATS == 1);
                    state <= S_FEED;
                end

                S_FEED: begin
                    if (array_k_ready) begin
                        // Previous beat consumed; prepare next beat (or finish)
                        if (beat_idx == K_BEATS-1) begin
                            beat_idx <= '0;
                            state <= S_WAIT;
                        end else begin
                            beat_idx <= beat_idx + 1'b1;
                            array_k_valid <= 1'b1;
                            array_k_last  <= (beat_idx + 1 == K_BEATS-1);
                        end
                    end
                end

                S_WAIT: begin
                    if (array_done) begin
`ifdef DBG_PIPELINE
                        $display("  [LE_DBG] M_OUT=%0d row=%0d array_sum=0x%08h (%0d)",
                                 M_OUT, row_idx, array_sum, array_sum);
`endif
                        result_valid <= 1'b1;
                        result_row <= row_idx;
                        result_data <= array_sum;
                        state <= S_RESULT;
                    end
                end

                S_RESULT: begin
                    if (result_ready) begin
                        result_valid <= 1'b0;
                        if (row_idx == M_OUT-1) begin
                            done <= 1'b1;
                            state <= S_DONE;
                        end else begin
                            row_idx <= row_idx + 1'b1;
                            beat_idx <= '0;
                            array_start <= 1'b1;
                            state <= S_ARRAY_START;
                        end
                    end
                end

                S_DONE: begin
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
