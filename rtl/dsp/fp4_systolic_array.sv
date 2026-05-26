//=============================================================================
// fp4_systolic_array.sv — minimal streaming fp4 vector-dot array
//
// Streams K dimension in beats of LANES elements. Internally reuses
// fp4_scaled_tile, whose MAC lanes accumulate over beats. k_last marks the
// final beat; done asserts after the tile pipeline drains.
//=============================================================================

module fp4_systolic_array #(
    parameter int LANES       = 4,
    parameter int NUM_GROUPS  = 512,
    parameter int GROUP_SIZE  = 16,
    parameter int ADDR_WIDTH  = $clog2(NUM_GROUPS),
    parameter int ELEM_WIDTH  = 16,
    parameter int ACCUM_WIDTH = 32,
    parameter int DRAIN_CYCLES = 8,
    // Sparse attention early termination (DSA/CSA)
    parameter int SPARSE_EN    = 0,
    parameter int SPARSE_MIN_BEATS = 2,
    parameter int SPARSE_THRESHOLD_Q12 = 1024  // 0.25 in Q12
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Control
    input  logic                         start,
    input  logic                         k_valid,
    input  logic                         k_last,
    input  logic [ELEM_WIDTH-1:0]        elem_idx_base,
    input  logic [LANES*4-1:0]           weight_fp4_flat,
    input  logic [LANES*8-1:0]           activ_fp8_flat,
    output logic                         k_ready,

    // Scale memory load port
    input  logic                         scale_wr_en,
    input  logic [ADDR_WIDTH-1:0]        scale_wr_addr,
    input  logic [7:0]                   scale_wr_data,

    // Result
    output logic                         busy,
    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [ACCUM_WIDTH-1:0]       sum_result,
    output logic [LANES*ACCUM_WIDTH-1:0] lane_result_flat
);

    typedef enum logic [2:0] {
        S_IDLE,
        S_RUN,
        S_DRAIN,
        S_OUTPUT,
        S_DONE
    } state_t;

    state_t state;
    logic [$clog2(DRAIN_CYCLES+1)-1:0] drain_count;
    logic accum_clr;
    logic tile_valid_out;
    logic [ACCUM_WIDTH-1:0] tile_sum;
    logic [LANES*ACCUM_WIDTH-1:0] tile_lanes;
    logic [5:0] beat_count;              // beats processed in current dot product
    logic [31:0] sparse_estimate;         // rough running score (no scale, for early check)

    assign k_ready = (state == S_RUN);
    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign sum_result = tile_sum;
    assign lane_result_flat = tile_lanes;
    assign result_valid = (state == S_OUTPUT);

    fp4_scaled_tile #(
        .LANES(LANES),
        .NUM_GROUPS(NUM_GROUPS),
        .GROUP_SIZE(GROUP_SIZE),
        .ADDR_WIDTH(ADDR_WIDTH),
        .ELEM_WIDTH(ELEM_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH)
    ) u_scaled_tile (
        .clk              (clk),
        .rst_n            (rst_n),
        .accum_clr        (accum_clr),
        .valid_in         (k_valid & k_ready),
        .elem_idx_base    (elem_idx_base),
        .weight_fp4_flat  (weight_fp4_flat),
        .activ_fp8_flat   (activ_fp8_flat),
        .scale_wr_en      (scale_wr_en),
        .scale_wr_addr    (scale_wr_addr),
        .scale_wr_data    (scale_wr_data),
        .valid_out        (tile_valid_out),
        .lane_result_flat (tile_lanes),
        .sum_result       (tile_sum)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            drain_count <= '0;
            accum_clr <= 1'b0;
            beat_count <= '0;
            sparse_estimate <= '0;
        end else begin
            accum_clr <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        accum_clr <= 1'b1;
                        beat_count <= '0;
                        sparse_estimate <= '0;
                        state <= S_RUN;
                    end
                end

                S_RUN: begin
                    if (k_valid && k_ready) begin
                        beat_count <= beat_count + 1'b1;
                        // Rough score estimate (weight × activation, no scale, fast)
                        sparse_estimate <= sparse_estimate +
                            (($signed(weight_fp4_flat[0*4+:4]) * $signed(activ_fp8_flat[0*8+:8])) +
                             ($signed(weight_fp4_flat[1*4+:4]) * $signed(activ_fp8_flat[1*8+:8])) +
                             ($signed(weight_fp4_flat[2*4+:4]) * $signed(activ_fp8_flat[2*8+:8])) +
                             ($signed(weight_fp4_flat[3*4+:4]) * $signed(activ_fp8_flat[3*8+:8])));
                        // Sparse early termination: if estimate below threshold after
                        // enough beats, skip remaining beats for this dot product.
                        if (SPARSE_EN && beat_count >= SPARSE_MIN_BEATS) begin
                            if (sparse_estimate < SPARSE_THRESHOLD_Q12 &&
                                sparse_estimate > -SPARSE_THRESHOLD_Q12) begin
                                drain_count <= DRAIN_CYCLES[$bits(drain_count)-1:0];
                                state <= S_DRAIN;
                            end
                        end
                    end
                    if (k_valid && k_ready && k_last) begin
                        drain_count <= DRAIN_CYCLES[$bits(drain_count)-1:0];
                        state <= S_DRAIN;
                    end
                end

                S_DRAIN: begin
                    if (drain_count == 0) begin
                        state <= S_OUTPUT;
                    end else begin
                        drain_count <= drain_count - 1'b1;
                    end
                end

                S_OUTPUT: begin
                    if (result_ready) begin
                        state <= S_DONE;
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
