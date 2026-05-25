//=============================================================================
// fp4_scaled_tile.sv — scale-reader aligned wrapper around fp4_systolic_tile
//=============================================================================

module fp4_scaled_tile #(
    parameter int LANES       = 4,
    parameter int NUM_GROUPS  = 512,
    parameter int GROUP_SIZE  = 16,
    parameter int ADDR_WIDTH  = $clog2(NUM_GROUPS),
    parameter int ELEM_WIDTH  = 16,
    parameter int ACCUM_WIDTH = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         accum_clr,

    input  logic                         valid_in,
    input  logic [ELEM_WIDTH-1:0]        elem_idx_base,
    input  logic [LANES*4-1:0]           weight_fp4_flat,
    input  logic [LANES*8-1:0]           activ_fp8_flat,

    // Broadcast scale-memory load port
    input  logic                         scale_wr_en,
    input  logic [ADDR_WIDTH-1:0]        scale_wr_addr,
    input  logic [7:0]                   scale_wr_data,

    output logic                         valid_out,
    output logic [LANES*ACCUM_WIDTH-1:0] lane_result_flat,
    output logic [ACCUM_WIDTH-1:0]       sum_result
);

    logic [LANES-1:0] scale_valid;
    logic [LANES*8-1:0] scale_flat;
    logic [LANES*4-1:0] weight_d1;
    logic [LANES*8-1:0] activ_d1;
    logic valid_d1;

    genvar i;
    generate
        for (i = 0; i < LANES; i++) begin : g_scale
            logic [7:0] scale_i;
            logic [ADDR_WIDTH-1:0] group_unused;
            logic [ELEM_WIDTH-1:0] elem_idx_i;

            assign elem_idx_i = elem_idx_base + i;
            assign scale_flat[i*8 +: 8] = scale_i;

            fp4_scale_reader #(
                .NUM_GROUPS(NUM_GROUPS),
                .GROUP_SIZE(GROUP_SIZE),
                .ADDR_WIDTH(ADDR_WIDTH),
                .ELEM_WIDTH(ELEM_WIDTH),
                .SCALE_WIDTH(8)
            ) u_scale_reader (
                .clk        (clk),
                .rst_n      (rst_n),
                .q_valid    (valid_in),
                .q_elem_idx (elem_idx_i),
                .q_ready    (),
                .r_valid    (scale_valid[i]),
                .r_scale    (scale_i),
                .r_group_id (group_unused),
                .wr_en      (scale_wr_en),
                .wr_addr    (scale_wr_addr),
                .wr_data    (scale_wr_data)
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_d1 <= '0;
            activ_d1  <= '0;
            valid_d1  <= 1'b0;
        end else begin
            weight_d1 <= weight_fp4_flat;
            activ_d1  <= activ_fp8_flat;
            valid_d1  <= valid_in;
        end
    end

    fp4_systolic_tile #(
        .LANES(LANES),
        .ACCUM_WIDTH(ACCUM_WIDTH)
    ) u_tile (
        .clk              (clk),
        .rst_n            (rst_n),
        .accum_clr        (accum_clr),
        .valid_in         (valid_d1 & (&scale_valid)),
        .weight_fp4_flat  (weight_d1),
        .scale_fp8_flat   (scale_flat),
        .activ_fp8_flat   (activ_d1),
        .valid_out        (valid_out),
        .lane_result_flat (lane_result_flat),
        .sum_result       (sum_result)
    );

endmodule
