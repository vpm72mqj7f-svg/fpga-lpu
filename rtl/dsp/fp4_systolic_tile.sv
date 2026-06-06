//=============================================================================
// fp4_systolic_tile.sv — small fp4 MAC tile wrapper
//
// Instantiates multiple fp4_mac lanes and exposes a vector interface.
// This is a building block for a larger systolic array, intentionally small.
//=============================================================================

`include "fp4_types.svh"

module fp4_systolic_tile #(
    parameter int LANES       = 4,
    parameter int ACCUM_WIDTH = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         accum_clr,

    input  logic                         valid_in,
    input  logic [LANES*4-1:0]           weight_fp4_flat,
    input  logic [LANES*12-1:0]          scale_fp8_flat,
    input  logic [LANES*8-1:0]           activ_fp8_flat,

    output logic                         valid_out,
    output logic [LANES*ACCUM_WIDTH-1:0] lane_result_flat,
    output logic [ACCUM_WIDTH-1:0]       sum_result
);

    logic [LANES-1:0] lane_valid;
    logic [ACCUM_WIDTH-1:0] lane_sum_terms [LANES];

    genvar i;
    generate
        for (i = 0; i < LANES; i++) begin : g_lane
            fp4_mac_input_t  mac_in;
            fp4_mac_output_t mac_out;

            always_comb begin
                mac_in.weight = weight_fp4_flat[i*4 +: 4];
                mac_in.scale  = scale_fp8_flat[i*12 +: 12];
                mac_in.activ  = activ_fp8_flat[i*8 +: 8];
                mac_in.valid  = valid_in;
            end

            fp4_mac #(
                .ACCUM_WIDTH(ACCUM_WIDTH),
                .VEC_LANES(1)
            ) u_mac (
                .clk       (clk),
                .rst_n     (rst_n),
                .accum_clr (accum_clr),
                .mac_in    (mac_in),
                .mac_out   (mac_out)
            );

            assign lane_result_flat[i*ACCUM_WIDTH +: ACCUM_WIDTH] = mac_out.result;
            assign lane_sum_terms[i] = mac_out.result;
            assign lane_valid[i] = mac_out.valid;
        end
    endgenerate

    always_comb begin
        sum_result = '0;
        for (int j = 0; j < LANES; j++) begin
            sum_result = sum_result + lane_sum_terms[j];
        end
    end

    always_comb begin
        valid_out = &lane_valid;
    end

endmodule
