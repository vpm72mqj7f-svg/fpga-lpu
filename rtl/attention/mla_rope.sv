//=============================================================================
// mla_rope.sv — Decoupled Rotary Position Embedding
//
// Applies rotation to Q (and optionally K) based on token position.
//   For each dimension pair (2i, 2i+1):
//     x' = x*cos(pos,i) - y*sin(pos,i)
//     y' = x*sin(pos,i) + y*cos(pos,i)
//
// Sin/cos values pre-loaded via config port. Q12 fixed-point.
// 2-cycle pipeline: multiply → sum.
//=============================================================================

module mla_rope #(
    parameter int HIDDEN      = 8,
    parameter int MAX_POS     = 64,
    parameter int COEFF_W     = 16,
    parameter int DATA_W      = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Vector input
    input  logic                         in_valid,
    input  logic [HIDDEN*DATA_W-1:0]     vec_flat,
    input  logic [$clog2(MAX_POS)-1:0]   pos,
    output logic                         in_ready,

    // Sin/cos LUT write port
    input  logic                         lut_wr_en,
    input  logic [$clog2(MAX_POS)-1:0]   lut_pos,
    input  logic [$clog2(HIDDEN/2)-1:0]  lut_pair,
    input  logic signed [COEFF_W-1:0]    lut_sin_data,
    input  logic signed [COEFF_W-1:0]    lut_cos_data,

    // Rotated output
    output logic                         out_valid,
    output logic [HIDDEN*DATA_W-1:0]     rot_flat
);

    localparam int N_PAIRS     = HIDDEN / 2;
    localparam int LUT_DEPTH   = MAX_POS * N_PAIRS;
    localparam int LUT_AW      = $clog2(LUT_DEPTH);

    // Flattened LUT addresses
    logic [LUT_AW-1:0]          lut_wr_addr;
    logic [LUT_AW-1:0]          lut_rd_addr;
    logic signed [COEFF_W-1:0]  sin_q, cos_q;

    assign lut_wr_addr = lut_pos * N_PAIRS + lut_pair;
    assign lut_rd_addr = pos_r  * N_PAIRS + pair_idx;

    // Altera syncram IP for sin/cos LUTs (ROM-style, MLAB for low latency)
    altera_syncram #(.WIDTH(COEFF_W), .DEPTH(LUT_DEPTH), .RAM_BLOCK_TYPE("MLAB"))
    u_sin_lut (.clock(clk), .wren(lut_wr_en), .wraddress(lut_wr_addr),
               .data(lut_sin_data), .rdaddress(lut_rd_addr), .q(sin_q));

    altera_syncram #(.WIDTH(COEFF_W), .DEPTH(LUT_DEPTH), .RAM_BLOCK_TYPE("MLAB"),
                      .INIT_VALUE(16'd4096))  // cos(0) = 1.0 in Q12
    u_cos_lut (.clock(clk), .wren(lut_wr_en), .wraddress(lut_wr_addr),
               .data(lut_cos_data), .rdaddress(lut_rd_addr), .q(cos_q));

    // Registered inputs
    logic signed [DATA_W-1:0] vec_r [HIDDEN];
    logic [$clog2(MAX_POS)-1:0] pos_r;

    // Pipeline stage 1: multiply results
    logic signed [DATA_W-1:0] x_cos, x_sin, y_cos, y_sin;

    // State
    logic pipe_active;
    logic stage2;

    assign in_ready = !pipe_active;

    // ── DSP: altera_mult_add IP instances (4 multipliers for RoPE) ──
    wire signed [DATA_W+COEFF_W-1:0] mul_x_cos, mul_x_sin, mul_y_cos, mul_y_sin;
    wire signed [DATA_W-1:0]         vec_even, vec_odd;

    assign vec_even = vec_r[pair_idx*2];
    assign vec_odd  = vec_r[pair_idx*2+1];

    altera_mult_add #(.A_WIDTH(DATA_W), .B_WIDTH(COEFF_W), .PIPE_STAGES(0))
    u_mul_x_cos (.clock(clk), .a(vec_even), .b(cos_q), .result(mul_x_cos));

    altera_mult_add #(.A_WIDTH(DATA_W), .B_WIDTH(COEFF_W), .PIPE_STAGES(0))
    u_mul_x_sin (.clock(clk), .a(vec_even), .b(sin_q), .result(mul_x_sin));

    altera_mult_add #(.A_WIDTH(DATA_W), .B_WIDTH(COEFF_W), .PIPE_STAGES(0))
    u_mul_y_cos (.clock(clk), .a(vec_odd),  .b(cos_q), .result(mul_y_cos));

    altera_mult_add #(.A_WIDTH(DATA_W), .B_WIDTH(COEFF_W), .PIPE_STAGES(0))
    u_mul_y_sin (.clock(clk), .a(vec_odd),  .b(sin_q), .result(mul_y_sin));

    // Main pipeline — processes one pair per cycle
    logic [$clog2(N_PAIRS)-1:0] pair_idx;
    logic                        pair_done;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_active <= 1'b0;
            stage2      <= 1'b0;
            pair_idx    <= '0;
            out_valid   <= 1'b0;
            rot_flat    <= '0;
            for (int d = 0; d < HIDDEN; d++) vec_r[d] <= '0;
            pos_r       <= '0;
            x_cos <= '0; x_sin <= '0;
            y_cos <= '0; y_sin <= '0;
        end else begin
            out_valid <= 1'b0;

            if (in_valid && in_ready) begin
                for (int d = 0; d < HIDDEN; d++)
                    vec_r[d] <= $signed(vec_flat[d*DATA_W +: DATA_W]);
                pos_r       <= pos;
                pair_idx    <= '0;
                pipe_active <= 1'b1;
                stage2      <= 1'b0;
            end

            if (pipe_active && !stage2) begin
                // Stage 1: altera_mult_add computes products, register results
                x_cos <= mul_x_cos >>> 12;
                x_sin <= mul_x_sin >>> 12;
                y_cos <= mul_y_cos >>> 12;
                y_sin <= mul_y_sin >>> 12;
                stage2 <= 1'b1;
            end

            if (pipe_active && stage2) begin
                // Stage 2: combine
                rot_flat[(pair_idx*2)*DATA_W   +: DATA_W] <= x_cos - y_sin;  // x'
                rot_flat[(pair_idx*2+1)*DATA_W +: DATA_W] <= x_sin + y_cos;  // y'

                if (pair_idx == (N_PAIRS - 1)) begin
                    out_valid   <= 1'b1;
                    pipe_active <= 1'b0;
                end else begin
                    pair_idx <= pair_idx + 1'b1;
                end
                stage2 <= 1'b0;
            end
        end
    end

endmodule
