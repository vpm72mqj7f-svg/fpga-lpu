//=============================================================================
// rms_norm.sv — signed Q12 RMSNorm (HIDDEN=8 bring-up, Icarus-friendly)
//=============================================================================

module rms_norm #(
    parameter int HIDDEN     = 8,
    parameter int SQRT_ITERS = 3,
    parameter int LATENCY    = SQRT_ITERS + 2
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic signed [31:0]           x0, x1, x2, x3, x4, x5, x6, x7,
    input  logic signed [31:0]           g0, g1, g2, g3, g4, g5, g6, g7,
    output logic                         valid_out,
    output logic signed [31:0]           y0, y1, y2, y3, y4, y5, y6, y7
);

    function automatic logic [31:0] isqrt(input logic [63:0] a);
        logic [63:0] g;
        begin
            if (a == 0) begin isqrt = 32'd1; end else begin
            g = 64'd1 << ($clog2(a) / 2);
            for (int i = 0; i < SQRT_ITERS; i++) begin
                g = (g + a / g) >> 1;
            end
            isqrt = g[31:0]; end
        end
    endfunction

    logic signed [31:0] xr0, xr1, xr2, xr3, xr4, xr5, xr6, xr7;
    logic signed [31:0] gr0, gr1, gr2, gr3, gr4, gr5, gr6, gr7;
    logic [31:0] rsqrt_val;
    logic [63:0] sum_x2;
    logic [$clog2(LATENCY+1)-1:0] delay_cnt;
    logic active;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            active <= 1'b0;
            delay_cnt <= '0;
            rsqrt_val <= '0;
            {xr0,xr1,xr2,xr3,xr4,xr5,xr6,xr7} <= '0;
            {gr0,gr1,gr2,gr3,gr4,gr5,gr6,gr7} <= '0;
            {y0,y1,y2,y3,y4,y5,y6,y7} <= '0;
        end else begin
            valid_out <= 1'b0;

            if (valid_in && !active) begin
                active <= 1'b1;
                delay_cnt <= (LATENCY - 1);
                {xr0,xr1,xr2,xr3,xr4,xr5,xr6,xr7} <= {x0,x1,x2,x3,x4,x5,x6,x7};
                {gr0,gr1,gr2,gr3,gr4,gr5,gr6,gr7} <= {g0,g1,g2,g3,g4,g5,g6,g7};
                sum_x2 = '0;
                sum_x2 = sum_x2 + $signed(x0) * $signed(x0);
                sum_x2 = sum_x2 + $signed(x1) * $signed(x1);
                sum_x2 = sum_x2 + $signed(x2) * $signed(x2);
                sum_x2 = sum_x2 + $signed(x3) * $signed(x3);
                sum_x2 = sum_x2 + $signed(x4) * $signed(x4);
                sum_x2 = sum_x2 + $signed(x5) * $signed(x5);
                sum_x2 = sum_x2 + $signed(x6) * $signed(x6);
                sum_x2 = sum_x2 + $signed(x7) * $signed(x7);
                rsqrt_val <= 32'd16777216 / isqrt(sum_x2 >> $clog2(HIDDEN));
            end else if (active) begin
                if (delay_cnt == 0) begin
                    valid_out <= 1'b1;
                    active <= 1'b0;
                    y0 <= ( ($signed({1'b0, xr0}) * $signed({1'b0, gr0})) >>> 12 )
                         * $signed({1'b0, rsqrt_val}) >>> 12;
                    y1 <= ( ($signed({1'b0, xr1}) * $signed({1'b0, gr1})) >>> 12 )
                         * $signed({1'b0, rsqrt_val}) >>> 12;
                    y2 <= ( ($signed({1'b0, xr2}) * $signed({1'b0, gr2})) >>> 12 )
                         * $signed({1'b0, rsqrt_val}) >>> 12;
                    y3 <= ( ($signed({1'b0, xr3}) * $signed({1'b0, gr3})) >>> 12 )
                         * $signed({1'b0, rsqrt_val}) >>> 12;
                    y4 <= ( ($signed({1'b0, xr4}) * $signed({1'b0, gr4})) >>> 12 )
                         * $signed({1'b0, rsqrt_val}) >>> 12;
                    y5 <= ( ($signed({1'b0, xr5}) * $signed({1'b0, gr5})) >>> 12 )
                         * $signed({1'b0, rsqrt_val}) >>> 12;
                    y6 <= ( ($signed({1'b0, xr6}) * $signed({1'b0, gr6})) >>> 12 )
                         * $signed({1'b0, rsqrt_val}) >>> 12;
                    y7 <= ( ($signed({1'b0, xr7}) * $signed({1'b0, gr7})) >>> 12 )
                         * $signed({1'b0, rsqrt_val}) >>> 12;
                end else begin
                    delay_cnt <= delay_cnt - 1'b1;
                end
            end
        end
    end

endmodule
