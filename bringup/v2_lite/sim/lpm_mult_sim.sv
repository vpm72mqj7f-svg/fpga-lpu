// lpm_mult_sim.sv — Icarus behavioral model for lpm_mult
// Quartus replaces this with actual DSP blocks during synthesis
module lpm_mult #(
    parameter LPM_WIDTHA = 8, LPM_WIDTHB = 8, LPM_WIDTHP = 16,
    parameter LPM_WIDTHS = 8, LPM_REPRESENTATION = "SIGNED",
    parameter LPM_PIPELINE = 2, LPM_TYPE = "LPM_MULT",
    parameter LPM_HINT = "UNUSED", parameter USE_EAB = "OFF"
) (
    input clock,
    input [LPM_WIDTHA-1:0] dataa,
    input [LPM_WIDTHB-1:0] datab,
    output [LPM_WIDTHP-1:0] result
);
    // Pipelined behavioral multiply
    logic signed [LPM_WIDTHA-1:0] a_r, a_rr;
    logic signed [LPM_WIDTHB-1:0] b_r, b_rr;
    logic signed [LPM_WIDTHP-1:0] mult_r;

    always_ff @(posedge clock) begin
        a_r <= $signed(dataa);
        b_r <= $signed(datab);
        a_rr <= a_r;
        b_rr <= b_r;
        mult_r <= a_rr * b_rr;
    end

    assign result = mult_r;
endmodule
