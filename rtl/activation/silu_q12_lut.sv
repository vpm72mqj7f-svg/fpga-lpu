//=============================================================================
// silu_q12_lut.sv — signed Q12 SiLU approximation
//
// Input/output format: signed Q12 fixed point (1.0 = 4096).
// Piecewise-linear interpolation over knots [-8,-4,-2,-1,0,1,2,4,8].
// DSP multiply via altera_mult_add IP (PIPE_STAGES=0 for combinational).
//=============================================================================

module silu_q12_lut (
    input  logic                 clk,      // required by altera_mult_add IP
    input  logic signed [31:0]   x_q12,
    output logic signed [31:0]   y_q12
);

    //=========================================================================
    // DSP: altera_mult_add for (x - x0) * (y1 - y0)
    //=========================================================================
    logic signed [31:0] dx, dy;
    logic signed [31:0] y0_sel, den_sel;
    wire  signed [63:0] dsp_prod;

    altera_mult_add #(.A_WIDTH(32), .B_WIDTH(32), .PIPE_STAGES(0))
    u_dsp (.clock(clk), .a(dx), .b(dy), .result(dsp_prod));

    always_comb begin
        // Defaults
        dx = 0; dy = 0; y0_sel = 0; den_sel = 1;

        if (x_q12 <= -32'sd32768) begin          // x <= -8
            y_q12 = -32'sd11;                    // silu(-8) ~= -0.00268
        end else if (x_q12 < -32'sd16384) begin  // -8 .. -4
            dx = x_q12 + 32'sd32768;             // x - (-32768)
            dy = -32'sd284;                      // -295 - (-11)
            y0_sel = -32'sd11;
            den_sel = 32'sd16384;                // -16384 - (-32768)
            y_q12 = y0_sel + (dsp_prod / den_sel);
        end else if (x_q12 < -32'sd8192) begin   // -4 .. -2
            dx = x_q12 + 32'sd16384;             // x - (-16384)
            dy = -32'sd681;                      // -976 - (-295)
            y0_sel = -32'sd295;
            den_sel = 32'sd8192;
            y_q12 = y0_sel + (dsp_prod / den_sel);
        end else if (x_q12 < -32'sd4096) begin   // -2 .. -1
            dx = x_q12 + 32'sd8192;              // x - (-8192)
            dy = -32'sd126;                      // -1102 - (-976)
            y0_sel = -32'sd976;
            den_sel = 32'sd4096;
            y_q12 = y0_sel + (dsp_prod / den_sel);
        end else if (x_q12 < 32'sd0) begin       // -1 .. 0
            dx = x_q12 + 32'sd4096;              // x - (-4096)
            dy = 32'sd1102;                      // 0 - (-1102)
            y0_sel = -32'sd1102;
            den_sel = 32'sd4096;
            y_q12 = y0_sel + (dsp_prod / den_sel);
        end else if (x_q12 < 32'sd4096) begin    // 0 .. 1
            dx = x_q12;                          // x - 0
            dy = 32'sd2994;                      // 2994 - 0
            y0_sel = 32'sd0;
            den_sel = 32'sd4096;
            y_q12 = y0_sel + (dsp_prod / den_sel);
        end else if (x_q12 < 32'sd8192) begin    // 1 .. 2
            dx = x_q12 - 32'sd4096;              // x - 4096
            dy = 32'sd4221;                      // 7215 - 2994
            y0_sel = 32'sd2994;
            den_sel = 32'sd4096;
            y_q12 = y0_sel + (dsp_prod / den_sel);
        end else if (x_q12 < 32'sd16384) begin   // 2 .. 4
            dx = x_q12 - 32'sd8192;              // x - 8192
            dy = 32'sd8874;                      // 16089 - 7215
            y0_sel = 32'sd7215;
            den_sel = 32'sd8192;
            y_q12 = y0_sel + (dsp_prod / den_sel);
        end else if (x_q12 < 32'sd32768) begin   // 4 .. 8
            dx = x_q12 - 32'sd16384;             // x - 16384
            dy = 32'sd16668;                     // 32757 - 16089
            y0_sel = 32'sd16089;
            den_sel = 32'sd16384;
            y_q12 = y0_sel + (dsp_prod / den_sel);
        end else begin                           // x >= 8
            y_q12 = x_q12;
        end
    end

endmodule
