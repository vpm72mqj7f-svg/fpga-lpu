//=============================================================================
// q12_to_fp8_e4m3.sv — approximate signed Q12 to FP8 E4M3 encoder
//
// Bring-up encoder with nearest-value thresholds for common positive magnitudes.
// Supports sign bit. Saturates magnitude at ~8.0.
//=============================================================================

module q12_to_fp8_e4m3 (
    input  logic signed [31:0] x_q12,
    output logic [7:0]         fp8
);

    logic sign;
    logic [31:0] ax;
    logic [6:0] mag_code;

    always_comb begin
        sign = x_q12 < 0;
        ax = sign ? -x_q12 : x_q12;

        // Thresholds are midpoints in Q12.
        if (ax < 32'd512)        mag_code = 7'h00; // 0
        else if (ax < 32'd1536)  mag_code = 7'h28; // 0.25
        else if (ax < 32'd2560)  mag_code = 7'h30; // 0.5
        else if (ax < 32'd3584)  mag_code = 7'h34; // 0.75
        else if (ax < 32'd5120)  mag_code = 7'h38; // 1.0
        else if (ax < 32'd7168)  mag_code = 7'h3c; // 1.5
        else if (ax < 32'd10240) mag_code = 7'h40; // 2.0
        else if (ax < 32'd14336) mag_code = 7'h44; // 3.0
        else if (ax < 32'd20480) mag_code = 7'h48; // 4.0
        else if (ax < 32'd28672) mag_code = 7'h4c; // 6.0
        else                     mag_code = 7'h50; // 8.0 (saturating)

        fp8 = {sign, mag_code};
    end

endmodule
