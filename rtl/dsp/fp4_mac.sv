//=============================================================================
// fp4_mac.sv — fp4(E2M1) × fp8(E4M3) Multiply-Accumulate Unit
// Target: Intel Agilex 7 M-Series DSP (AGM 039-F)
//
// Pipeline: 3 stages
//   Stage 1: fp4 decode → signed 8b, fp8 decode → signed 12b
//   Stage 2: 8b×12b signed multiply → 20-bit product
//   Stage 3: 20b sign-extend → 32b accumulate
//
// Data flow:
//   fp4_weight → LUT decode → signed 8b (×16 scaled)
//   fp8_activ  → E4M3 decode → signed 12b (×256 scaled)
//   Multiply in DSP 18×19 mode, accumulate in 32b accumulator
//=============================================================================

`include "fp4_types.svh"

module fp4_mac #(
    parameter int ACCUM_WIDTH = 32,          // accumulator bit width
    parameter int VEC_LANES    = 2           // parallel MAC lanes per instance
) (
    input  logic                clk,
    input  logic                rst_n,

    // Control
    input  logic                accum_clr,   // clear accumulator (new token)

    // Input: fp4 weight stream + per-group fp8 scale
    input  fp4_mac_input_t      mac_in,

    // Output: accumulated result
    output fp4_mac_output_t     mac_out
);

    //=========================================================================
    // Stage 0: Input Register
    //=========================================================================
    logic [3:0]  s0_weight;
    logic [7:0]  s0_scale;
    logic [7:0]  s0_activ;
    logic        s0_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_weight <= 4'd0;
            s0_scale  <= 8'd0;
            s0_activ  <= 8'd0;
            s0_valid  <= 1'b0;
        end else begin
            s0_weight <= mac_in.weight;
            s0_scale  <= mac_in.scale;
            s0_activ  <= mac_in.activ;
            s0_valid  <= mac_in.valid;
        end
    end

    //=========================================================================
    // Stage 1: fp4 Decode + fp8 Decode → Signed 8-bit Operands
    //=========================================================================

    // --- fp4 weight decode ---
    // Extract fields
    logic        w_sign;
    logic [1:0]  w_exp;
    logic        w_mant;
    logic [2:0]  w_mag;

    assign w_sign = s0_weight[3];
    assign w_exp  = s0_weight[2:1];
    assign w_mant = s0_weight[0];
    assign w_mag  = s0_weight[2:0];  // magnitude index for LUT

    // LUT: magnitude → scaled integer (×16)
    logic [5:0]  w_mag_scaled;
    assign w_mag_scaled = fp4_mag_to_scaled(w_mag);

    // fp4 decoded as signed 8-bit: w_sign ? -w_mag : +w_mag
    // (values already ×16 for fractional representation)
    logic [7:0]  w_signed;
    always_comb begin
        if (w_mag == 3'd0) begin
            w_signed = 8'd0;  // zero
        end else if (w_sign) begin
            w_signed = -{2'd0, w_mag_scaled};  // negative
        end else begin
            w_signed = {2'd0, w_mag_scaled};   // positive
        end
    end

    // --- fp8 activation decode (E4M3) ---
    logic        a_sign;
    logic [3:0]  a_exp;
    logic [2:0]  a_mant;
    assign a_sign = s0_activ[7];
    assign a_exp  = s0_activ[6:3];
    assign a_mant = s0_activ[2:0];

    // --- fp8 per-group scale decode (E4M3, expected positive) ---
    logic        sc_sign;
    logic [3:0]  sc_exp;
    logic [2:0]  sc_mant;
    assign sc_sign = s0_scale[7];
    assign sc_exp  = s0_scale[6:3];
    assign sc_mant = s0_scale[2:0];

    // FP8 E4M3 → signed scaled integer:
    //   Normal:  (-1)^s × 2^(e-7) × (1 + m/8)  [e ≠ 0]
    //   Subnorm: (-1)^s × 2^(-6) × m/8          [e = 0]
    // Represent with ×256 scaling.
    //   value × 256 = (8+m)/8 × 2^(e-7) × 2^8 = (8+m) × 2^(e-2)  [normal]
    //   value × 256 = m/8 × 2^(-6) × 2^8 = m/2                     [subnorm]
    logic signed [11:0] a_scaled;
    always_comb begin
        if (a_exp == 4'd0) begin
            // Subnorm: m/2 (with round-down from LSB)
            a_scaled = $signed({1'b0, {8'd0, a_mant[2:1]}});
        end else if (a_exp < 4'd2) begin
            // e = 1: (8+m) >> 1
            a_scaled = $signed({1'b0, 1'b1, a_mant}) >>> 1;
        end else begin
            // e ≥ 2: (8+m) << (e-2), use 16-bit intermediate to avoid truncation
            logic [4:0]  shift;
            logic signed [15:0] a_full;
            shift  = a_exp - 4'd2;
            a_full = $signed({8'd0, 1'b1, a_mant}) << shift;
            // Saturate to 12-bit signed range
            if (a_full > 16'sd2047)      a_scaled = 12'sd2047;
            else if (a_full < -16'sd2048) a_scaled = -12'sd2048;
            else                          a_scaled = a_full[11:0];
        end
        if (a_sign) a_scaled = -a_scaled;
    end

    logic signed [11:0] sc_scaled;
    always_comb begin
        if (sc_exp == 4'd0) begin
            sc_scaled = $signed({1'b0, {8'd0, sc_mant[2:1]}});
        end else if (sc_exp < 4'd2) begin
            sc_scaled = $signed({1'b0, 1'b1, sc_mant}) >>> 1;
        end else begin
            logic [4:0]  shift;
            logic signed [15:0] sc_full;
            shift   = sc_exp - 4'd2;
            sc_full = $signed({8'd0, 1'b1, sc_mant}) << shift;
            if (sc_full > 16'sd2047)       sc_scaled = 12'sd2047;
            else if (sc_full < -16'sd2048) sc_scaled = -12'sd2048;
            else                            sc_scaled = sc_full[11:0];
        end
        if (sc_sign) sc_scaled = -sc_scaled;
    end

    // Register Stage 1 outputs
    logic [7:0]  s1_w_signed;
    logic [11:0] s1_a_scaled;
    logic [11:0] s1_sc_scaled;
    logic        s1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_w_signed  <= 8'd0;
            s1_a_scaled  <= 12'd0;
            s1_sc_scaled <= 12'd0;
            s1_valid     <= 1'b0;
        end else begin
            s1_w_signed  <= w_signed;
            s1_a_scaled  <= a_scaled;
            s1_sc_scaled <= sc_scaled;
            s1_valid     <= s0_valid;
        end
    end

    //=========================================================================
    // Stage 2: Multiply — (8b fp4 × 12b activation) × 12b scale
    //   fp4 and activation are both decoded as fixed-point. Scale is decoded
    //   as FP8 E4M3 ×256, so product is rescaled by >>>8.
    //=========================================================================
    logic signed [31:0] s2_product;
    logic               s2_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_product <= 32'sd0;
            s2_valid   <= 1'b0;
        end else begin
            logic signed [19:0] base_product;
            logic signed [31:0] scaled_product;
            base_product   = $signed(s1_w_signed) * $signed(s1_a_scaled);
            scaled_product = $signed(base_product) * $signed(s1_sc_scaled);
            s2_product     <= scaled_product >>> 8;
            s2_valid       <= s1_valid;
        end
    end

    //=========================================================================
    // Stage 3: Accumulate — 32b product → accumulator
    //   Sign-extend product, add to running sum. Saturation prevents overflow
    //   wrap-around on deep accumulations (inspired by TALOS-V2 sat16).
    //   Clear on accum_clr (new token start).
    //=========================================================================
    logic [ACCUM_WIDTH-1:0] accumulator;
    logic                   s3_valid;

    // Saturation: if operands have same sign but result has opposite sign,
    // clamp to max/min instead of wrapping around.
    function automatic logic [ACCUM_WIDTH-1:0] sat_acc;
        input [ACCUM_WIDTH-1:0] old_acc;
        input [ACCUM_WIDTH-1:0] add_val;
        logic [ACCUM_WIDTH-1:0] sum_raw;
        logic old_sign, val_sign, sum_sign;
        begin
            sum_raw  = old_acc + add_val;
            old_sign = old_acc[ACCUM_WIDTH-1];
            val_sign = add_val[ACCUM_WIDTH-1];
            sum_sign = sum_raw[ACCUM_WIDTH-1];
            // Overflow: both positive but sum went negative
            if (!old_sign && !val_sign && sum_sign)
                sat_acc = {1'b0, {(ACCUM_WIDTH-1){1'b1}}};  // max positive
            // Underflow: both negative but sum went positive
            else if (old_sign && val_sign && !sum_sign)
                sat_acc = {1'b1, {(ACCUM_WIDTH-1){1'b0}}};  // max negative
            else
                sat_acc = sum_raw;
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= '0;
            s3_valid    <= 1'b0;
        end else if (accum_clr) begin
            accumulator <= '0;
            s3_valid    <= 1'b0;
        end else begin
            if (s2_valid) begin
                accumulator <= sat_acc(accumulator, s2_product);
            end
            s3_valid <= s2_valid;
        end
    end

    assign mac_out.result = accumulator;
    assign mac_out.valid  = s3_valid;

endmodule
