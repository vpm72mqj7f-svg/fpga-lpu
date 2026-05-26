//=============================================================================
// rms_norm.sv — signed Q12 RMSNorm (HIDDEN=8 bring-up, Icarus-friendly)
//
// Pipeline: 4-cycle compute + sqrt
//   Stage 0: Latch x[0:7], g[0:7]; pairwise sum-of-squares products
//   Stage 1: Reduce pairwise sums → total sum_sq; launch sqrt computation
//   Stage 2: Wait for sqrt (combinational function, registered here)
//   Stage 3: x*g multiply (registered)
//   Stage 4: *rsqrt multiply → output
//
// SQRT_MODE: 0 = Newton-Raphson (fast, uses DSP divide)
//            1 = Digit-recurrence (slow, pure LUT+FF, zero DSP)
//               Inspired by TALOS-V2 rms_scale_engine.sv
//
// TODO: Convert sqrt functions to proper multi-cycle state machines
//       for production use at >100 MHz. Current combinational functions
//       limit fmax to ~50-80 MHz depending on SQRT_ITERS.
//=============================================================================

module rms_norm #(
    parameter int HIDDEN     = 8,
    parameter int SQRT_ITERS = 3,
    parameter int SQRT_MODE  = 0,    // 0=Newton-Raphson, 1=digit-recurrence
    parameter int LATENCY    = 5     // pipeline depth (input → output)
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic signed [31:0]           x0, x1, x2, x3, x4, x5, x6, x7,
    input  logic signed [31:0]           g0, g1, g2, g3, g4, g5, g6, g7,
    output logic                         valid_out,
    output logic signed [31:0]           y0, y1, y2, y3, y4, y5, y6, y7
);

    //=========================================================================
    // Newton-Raphson sqrt (mode 0: fast, uses DSP)
    //=========================================================================
    function automatic logic [31:0] isqrt(input logic [63:0] a);
        logic [63:0] g;
        begin
            if (a == 0) begin
                isqrt = 32'd1;
            end else begin
                g = 64'd1 << ($clog2(a) / 2);
                for (int i = 0; i < SQRT_ITERS; i++) begin
                    g = (g + a / g) >> 1;
                end
                isqrt = g[31:0];
            end
        end
    endfunction

    //=========================================================================
    // Digit-recurrence sqrt (mode 1: slow, zero DSP, from TALOS-V2)
    //=========================================================================
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
            rem_div = 33'd0;
            quot    = 24'd0;
            for (div_bit = 23; div_bit >= 0; div_bit = div_bit - 1) begin
                rem_div = {rem_div[31:0], 1'b1};
                if (rem_div >= {1'b0, divisor}) begin
                    rem_div = rem_div - {1'b0, divisor};
                    quot[div_bit] = 1'b1;
                end
            end
            reciprocal_q24 = {8'd0, quot};
        end
    endfunction

    //=========================================================================
    // Pipeline registers
    //=========================================================================
    typedef enum logic [2:0] {
        S_IDLE, S_SOS_PAIR, S_SOS_REDUCE, S_SQRT, S_XG_MUL, S_OUTPUT
    } state_t;
    state_t state;

    // Stage 0→1: latched inputs
    logic signed [31:0] s0_x [HIDDEN-1:0];
    logic signed [31:0] s0_g [HIDDEN-1:0];

    // Stage 1→2: pairwise sum-of-squares
    logic signed [63:0] s1_ss [HIDDEN/2-1:0];  // 4 pairwise sums

    // Stage 2→3: total sum_sq + rsqrt
    logic [63:0] s2_sumsq;
    logic [31:0] s2_rsqrt;

    // Stage 3→4: x*g products
    logic signed [31:0] s3_xg [HIDDEN-1:0];

    // Pipeline valid shift
    logic [LATENCY-1:0] pipe_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            valid_out  <= 1'b0;
            pipe_valid <= '0;
            for (int i = 0; i < HIDDEN; i++) begin
                s0_x[i]  <= '0;
                s0_g[i]  <= '0;
                s3_xg[i] <= '0;
            end
            for (int i = 0; i < HIDDEN/2; i++) s1_ss[i] <= '0;
            s2_sumsq <= '0;
            s2_rsqrt <= '0;
            {y0,y1,y2,y3,y4,y5,y6,y7} <= '0;
        end else begin
            valid_out <= 1'b0;
            pipe_valid <= {pipe_valid[LATENCY-2:0], 1'b0};

            case (state)
                S_IDLE: begin
                    if (valid_in) begin
                        s0_x[0] <= x0; s0_x[1] <= x1; s0_x[2] <= x2; s0_x[3] <= x3;
                        s0_x[4] <= x4; s0_x[5] <= x5; s0_x[6] <= x6; s0_x[7] <= x7;
                        s0_g[0] <= g0; s0_g[1] <= g1; s0_g[2] <= g2; s0_g[3] <= g3;
                        s0_g[4] <= g4; s0_g[5] <= g5; s0_g[6] <= g6; s0_g[7] <= g7;
                        state <= S_SOS_PAIR;
                    end
                end

                // Stage 0→1: pairwise sum-of-squares products (4 DSPs)
                S_SOS_PAIR: begin
                    for (int i = 0; i < HIDDEN/2; i++) begin
                        s1_ss[i] <=
                            ($signed(s0_x[2*i]) * $signed(s0_x[2*i])) +
                            ($signed(s0_x[2*i+1]) * $signed(s0_x[2*i+1]));
                    end
                    state <= S_SOS_REDUCE;
                end

                // Stage 1→2: reduce pairwise sums, compute rsqrt
                S_SOS_REDUCE: begin
                    s2_sumsq <= (s1_ss[0] + s1_ss[1]) + (s1_ss[2] + s1_ss[3]);
                    // rsqrt computed combinationally, registered here
                    if (SQRT_MODE == 1) begin
                        s2_rsqrt <= digit_rsqrt(
                            (s1_ss[0] + s1_ss[1]) + (s1_ss[2] + s1_ss[3]),
                            HIDDEN);
                    end else begin
                        s2_rsqrt <= 32'd16777216 / isqrt(
                            ((s1_ss[0] + s1_ss[1]) + (s1_ss[2] + s1_ss[3]))
                            >> $clog2(HIDDEN));
                    end
                    state <= S_SQRT;
                end

                // Stage 2→3: sqrt result stable, just pass through
                S_SQRT: begin
                    state <= S_XG_MUL;
                end

                // Stage 3→4: x * gamma (Q12 multiply)
                S_XG_MUL: begin
                    for (int i = 0; i < HIDDEN; i++) begin
                        s3_xg[i] <= ($signed({1'b0, s0_x[i]}) *
                                     $signed({1'b0, s0_g[i]})) >>> 12;
                    end
                    state <= S_OUTPUT;
                end

                // Stage 4→output: x*g * rsqrt
                S_OUTPUT: begin
                    y0 <= ($signed(s3_xg[0]) * $signed({1'b0, s2_rsqrt})) >>> 12;
                    y1 <= ($signed(s3_xg[1]) * $signed({1'b0, s2_rsqrt})) >>> 12;
                    y2 <= ($signed(s3_xg[2]) * $signed({1'b0, s2_rsqrt})) >>> 12;
                    y3 <= ($signed(s3_xg[3]) * $signed({1'b0, s2_rsqrt})) >>> 12;
                    y4 <= ($signed(s3_xg[4]) * $signed({1'b0, s2_rsqrt})) >>> 12;
                    y5 <= ($signed(s3_xg[5]) * $signed({1'b0, s2_rsqrt})) >>> 12;
                    y6 <= ($signed(s3_xg[6]) * $signed({1'b0, s2_rsqrt})) >>> 12;
                    y7 <= ($signed(s3_xg[7]) * $signed({1'b0, s2_rsqrt})) >>> 12;
                    valid_out <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
