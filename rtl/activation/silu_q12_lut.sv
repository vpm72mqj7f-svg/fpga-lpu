//=============================================================================
// silu_q12_lut.sv — signed Q12 SiLU approximation
//
// Input/output format: signed Q12 fixed point (1.0 = 4096).
// Piecewise-linear interpolation over knots [-8,-4,-2,-1,0,1,2,4,8].
//=============================================================================

module silu_q12_lut (
    input  logic signed [31:0] x_q12,
    output logic signed [31:0] y_q12
);

    function automatic logic signed [31:0] interp(
        input logic signed [31:0] x,
        input logic signed [31:0] x0,
        input logic signed [31:0] y0,
        input logic signed [31:0] x1,
        input logic signed [31:0] y1
    );
        logic signed [63:0] num;
        logic signed [31:0] den;
        begin
            num = $signed(x - x0) * $signed(y1 - y0);
            den = x1 - x0;
            interp = y0 + (num / den);
        end
    endfunction

    always_comb begin
        // Constants: Q12 values of x and SiLU(x)
        if (x_q12 <= -32'sd32768) begin          // x <= -8
            y_q12 = -32'sd11;                    // silu(-8) ~= -0.00268
        end else if (x_q12 < -32'sd16384) begin  // -8 .. -4
            y_q12 = interp(x_q12, -32'sd32768, -32'sd11,
                                    -32'sd16384, -32'sd295);
        end else if (x_q12 < -32'sd8192) begin   // -4 .. -2
            y_q12 = interp(x_q12, -32'sd16384, -32'sd295,
                                    -32'sd8192,  -32'sd976);
        end else if (x_q12 < -32'sd4096) begin   // -2 .. -1
            y_q12 = interp(x_q12, -32'sd8192, -32'sd976,
                                    -32'sd4096, -32'sd1102);
        end else if (x_q12 < 32'sd0) begin       // -1 .. 0
            y_q12 = interp(x_q12, -32'sd4096, -32'sd1102,
                                     32'sd0,     32'sd0);
        end else if (x_q12 < 32'sd4096) begin    // 0 .. 1
            y_q12 = interp(x_q12, 32'sd0,     32'sd0,
                                    32'sd4096,  32'sd2994);
        end else if (x_q12 < 32'sd8192) begin    // 1 .. 2
            y_q12 = interp(x_q12, 32'sd4096,  32'sd2994,
                                    32'sd8192,  32'sd7215);
        end else if (x_q12 < 32'sd16384) begin   // 2 .. 4
            y_q12 = interp(x_q12, 32'sd8192,  32'sd7215,
                                    32'sd16384, 32'sd16089);
        end else if (x_q12 < 32'sd32768) begin   // 4 .. 8
            y_q12 = interp(x_q12, 32'sd16384, 32'sd16089,
                                    32'sd32768, 32'sd32757);
        end else begin                           // x >= 8
            y_q12 = x_q12;
        end
    end

endmodule
