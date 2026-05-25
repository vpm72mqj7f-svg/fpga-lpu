//=============================================================================
// rms_norm.sv — signed Q12 RMSNorm (HIDDEN=8 bring-up, Icarus-friendly)
//
// SQRT_MODE: 0 = Newton-Raphson (fast, uses DSP multiply)
//            1 = Digit-recurrence (slow, pure LUT+FF, zero DSP)
//               Inspired by TALOS-V2 rms_scale_engine.sv
//=============================================================================

module rms_norm #(
    parameter int HIDDEN     = 8,
    parameter int SQRT_ITERS = 3,
    parameter int SQRT_MODE  = 0,    // 0=Newton-Raphson, 1=digit-recurrence
    parameter int LATENCY    = (SQRT_MODE == 1) ? 58 : (SQRT_ITERS + 2)
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic signed [31:0]           x0, x1, x2, x3, x4, x5, x6, x7,
    input  logic signed [31:0]           g0, g1, g2, g3, g4, g5, g6, g7,
    output logic                         valid_out,
    output logic signed [31:0]           y0, y1, y2, y3, y4, y5, y6, y7
);

    // Newton-Raphson sqrt (mode 0: fast, uses DSP)
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

    // Digit-recurrence sqrt (mode 1: slow, zero DSP, from TALOS-V2)
    function automatic logic [31:0] isqrt_digit(input logic [63:0] value);
        logic [65:0] rem;
        logic [32:0] root;
        logic [33:0] cand;
        integer bit_idx;
        begin
            rem  = 66'd0;
            root = 33'd0;
            for (bit_idx = 31; bit_idx >= 0; bit_idx = bit_idx - 1) begin
                rem  = (rem << 2) | ((value >> (bit_idx * 2)) & 64'd3);
                root = root << 1;
                cand = (root << 1) | 34'd1;
                if (rem >= cand) begin
                    rem  = rem - cand;
                    root = root + 33'd1;
                end
            end
            isqrt_digit = root[31:0];
        end
    endfunction

    // Combined digit-recurrence rsqrt (TALOS-V2 inspired: sqrt + div)
    // Matches Newton-Raphson formula: rsqrt = Q24 / sqrt(sum_sq / N)
    function automatic logic [31:0] digit_rsqrt;
        input logic [63:0] sumsq;
        input integer n_elem;
        logic [63:0] mean_sq;
        logic [31:0] denom;
        begin
            mean_sq = sumsq / n_elem;
            denom   = isqrt_digit(mean_sq);
            digit_rsqrt = (denom == 0) ? 32'd32767 : (32'd16777216 / denom);
        end
    endfunction
    function automatic logic [31:0] reciprocal_q24(input logic [31:0] divisor);
        logic [32:0] rem_div;
        logic [23:0] quot;
        integer div_bit;
        begin
            // Compute 2^24 / divisor using restoring division
            rem_div = 33'd0;
            quot    = 24'd0;
            for (div_bit = 23; div_bit >= 0; div_bit = div_bit - 1) begin
                rem_div = {rem_div[31:0], 1'b1};  // bring down a 1
                if (rem_div >= {1'b0, divisor}) begin
                    rem_div = rem_div - {1'b0, divisor};
                    quot[div_bit] = 1'b1;
                end
            end
            reciprocal_q24 = {8'd0, quot};
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
                if (SQRT_MODE == 1) begin
                    // TALOS-V2 digit-recurrence path (zero DSP)
                    rsqrt_val <= digit_rsqrt(sum_x2, HIDDEN);
                end else begin
                    // Newton-Raphson path (fast, uses DSP)
                    rsqrt_val <= 32'd16777216 / isqrt(sum_x2 >> $clog2(HIDDEN));
                end
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
