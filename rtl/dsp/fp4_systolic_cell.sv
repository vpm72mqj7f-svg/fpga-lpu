//=============================================================================
// fp4_systolic_cell.sv — Single 2D Systolic Array Cell
//
// Weight-stationary: fp4 weight + pre-decoded fp8 scale are pre-loaded.
// Activation streams through, accumulator runs over K_BEATS.
//
// Each cell = 2 DSP blocks (base multiply + scale multiply).
// Accumulator cleared on new token (accum_clr).
//=============================================================================

`include "fp4_types.svh"

module fp4_systolic_cell #(
    parameter int ACCUM_WIDTH = 32
) (
    input  logic        clk,
    input  logic        rst_n,

    // Weight/scale preload
    input  logic        wt_wr_en,
    input  logic [3:0]  wt_wr_data,        // fp4 E2M1 encoded
    input  logic [11:0] sc_wr_data,        // pre-decoded fp8 scale (×256)

    // Activation input (from left neighbor or broadcast)
    input  logic [7:0]  activ_in,
    input  logic        valid_in,

    // Control
    input  logic        accum_clr,          // new token start

    // Accumulator readout
    output logic [ACCUM_WIDTH-1:0] accum_out
);

    //=========================================================================
    // Weight & Scale Storage (pre-loaded, stationary)
    //=========================================================================
    logic [3:0]  weight;
    logic [11:0] scale;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight <= 4'd0;
            scale  <= 12'd0;
        end else if (wt_wr_en) begin
            weight <= wt_wr_data;
            scale  <= sc_wr_data;
        end
    end

    //=========================================================================
    // fp4_mac Instance (4-stage pipeline)
    //=========================================================================
    fp4_mac_input_t  mac_in;
    fp4_mac_output_t mac_out;

    always_comb begin
        mac_in.weight = weight;
        mac_in.scale  = scale;
        mac_in.activ  = activ_in;
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

    assign accum_out = mac_out.result;

endmodule
