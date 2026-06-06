//=============================================================================
// rms_norm.sv — signed Q12 RMSNorm
//
// Pipeline: 5-stage FSM (SOS_PAIR → SOS_REDUCE → SQRT → XG_MUL → OUTPUT)
// SQRT_MODE: 0=Newton-Raphson (fast, DSP), 1=digit-recurrence (LUT only)
//
// TODO(prod): Convert sqrt combinational function to pipelined IP for >100MHz
//=============================================================================

`include "lpu_config.svh"

module rms_norm #(
    parameter int HIDDEN     = lpu_config_pkg::LPU_HIDDEN,
    parameter int SQRT_ITERS = 3,
    parameter int SQRT_MODE  = 0,    // 0=Newton-Raphson, 1=digit-recurrence
    parameter int LATENCY    = 5     // pipeline depth (input → output)
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic [HIDDEN*32-1:0]         x_flat,
    input  logic [HIDDEN*32-1:0]         g_flat,
    output logic                         valid_out,
    output logic [HIDDEN*32-1:0]         y_flat
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

    // ── Sum-of-squares multiply (combinational, Quartus infers DSP) ──
    wire signed [63:0] sos_prod [HIDDEN-1:0];    // x[i] * x[i]

    for (genvar d = 0; d < HIDDEN; d++) begin : gen_sos_mul
        assign sos_prod[d] = $signed(s0_x[d]) * $signed(s0_x[d]);
    end

    // ── X*G multiply (combinational, Quartus infers DSP) ──
    wire signed [63:0] xg_prod [HIDDEN-1:0];     // x[i] * g[i]

    for (genvar d = 0; d < HIDDEN; d++) begin : gen_xg_mul
        assign xg_prod[d] = $signed(s0_x[d]) * $signed(s0_g[d]);
    end

    // ── RMS output multiply (combinational, Quartus infers DSP) ──
    wire signed [63:0] rms_prod [HIDDEN-1:0];    // xg[i] * rsqrt

    for (genvar d = 0; d < HIDDEN; d++) begin : gen_rms_mul
        assign rms_prod[d] = $signed(s3_xg[d]) * $signed(s2_rsqrt);
    end

    // Stage 1→2: pairwise sum-of-squares
    logic signed [63:0] s1_ss [HIDDEN/2-1:0];

    // Combinational reduction of all s1_ss pairs
    // TODO(prod): replace with pipelined adder tree for synthesis at HIDDEN=7168
    logic [63:0] sos_total;
    always_comb begin
        sos_total = '0;
        for (int i = 0; i < HIDDEN/2; i++)
            sos_total = sos_total + s1_ss[i];
    end

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
            y_flat <= '0;
        end else begin
            valid_out <= 1'b0;
            pipe_valid <= {pipe_valid[LATENCY-2:0], 1'b0};

            case (state)
                S_IDLE: begin
                    if (valid_in) begin
                        for (int i = 0; i < HIDDEN; i++) begin
                            s0_x[i] <= $signed(x_flat[i*32+:32]);
                            s0_g[i] <= $signed(g_flat[i*32+:32]);
                        end
                        state <= S_SOS_PAIR;
                    end
                end

                // Stage 0→1: pairwise sum-of-squares (altera_mult_add DSP)
                S_SOS_PAIR: begin
                    for (int i = 0; i < HIDDEN/2; i++) begin
                        s1_ss[i] <= sos_prod[2*i] + sos_prod[2*i+1];
                    end
                    state <= S_SOS_REDUCE;
                end

                // Stage 1→2: reduce pairwise sums, compute rsqrt
                S_SOS_REDUCE: begin
                    s2_sumsq <= sos_total;
                    if (SQRT_MODE == 1) begin
                        s2_rsqrt <= digit_rsqrt(sos_total, HIDDEN);
                    end else begin
                        s2_rsqrt <= 32'd16777216 / isqrt(sos_total >> $clog2(HIDDEN));
                    end
                    state <= S_SQRT;
                end

                // Stage 2→3: sqrt result stable, just pass through
                S_SQRT: begin
                    state <= S_XG_MUL;
                end

                // Stage 3→4: x * gamma (altera_mult_add DSP)
                S_XG_MUL: begin
                    for (int i = 0; i < HIDDEN; i++) begin
                        s3_xg[i] <= xg_prod[i] >>> 12;
                    end
                    state <= S_OUTPUT;
                end

                // Stage 4→output: x*g * rsqrt (altera_mult_add DSP)
                S_OUTPUT: begin
                    for (int i = 0; i < HIDDEN; i++)
                        y_flat[i*32+:32] <= rms_prod[i] >>> 12;
                    valid_out <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
