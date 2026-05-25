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
    parameter int K_BEATS     = (K_TOTAL + LANES - 1) / LANES
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Preload ports
    input  logic                         weight_wr_en,
    input  logic [$clog2(M_OUT)-1:0]     weight_wr_row,
    input  logic [$clog2(K_BEATS)-1:0]   weight_wr_beat,
    input  logic [LANES*4-1:0]           weight_wr_data,

    input  logic                         activ_wr_en,
    input  logic [$clog2(K_BEATS)-1:0]   activ_wr_beat,
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
    logic [$clog2(M_OUT)-1:0] row_idx;
    logic [$clog2(K_BEATS)-1:0] beat_idx;

    logic [LANES*4-1:0] weight_mem [M_OUT*K_BEATS];
    logic [LANES*8-1:0] activ_mem  [K_BEATS];

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

    function automatic int mem_index(input int row, input int beat);
        mem_index = row * K_BEATS + beat;
    endfunction

    assign busy = (state != S_IDLE) && (state != S_DONE);

    always_ff @(posedge clk) begin
        if (weight_wr_en) begin
            weight_mem[mem_index(weight_wr_row, weight_wr_beat)] <= weight_wr_data;
        end
        if (activ_wr_en) begin
            activ_mem[activ_wr_beat] <= activ_wr_data;
        end
    end

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

    always_comb begin
        array_weight_flat = weight_mem[mem_index(row_idx, beat_idx)];
        array_activ_flat  = activ_mem[beat_idx];
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
                    state <= S_FEED;
                end

                S_FEED: begin
                    if (array_k_ready) begin
                        array_k_valid <= 1'b1;
                        array_k_last <= (beat_idx == K_BEATS-1);
                        if (beat_idx == K_BEATS-1) begin
                            beat_idx <= '0;
                            state <= S_WAIT;
                        end else begin
                            beat_idx <= beat_idx + 1'b1;
                        end
                    end
                end

                S_WAIT: begin
                    if (array_done) begin
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
            endcase
        end
    end

endmodule
