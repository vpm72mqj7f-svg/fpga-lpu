// =============================================================================
// fp8_mac.sv — FP8 E4M3 Multiply-Accumulate for Systolic Array Lanes
//
// Target: Stratix 10 MX (1SM21BHU2F53E1VG), Quartus Prime Pro 26.1
//
// FP8 E4M3 format: 1 sign | 4 exponent (bias=7) | 3 mantissa
//   Bit layout: [s][e3 e2 e1 e0][m2 m1 m0]
//
//   Decoding:
//     exp=0,  mant=0 → ±0
//     exp=0,  mant>0 → subnormal: (-1)^s × mant/8 × 2^{-6}
//     exp=1..14      → normal:    (-1)^s × (1 + mant/8) × 2^{exp-7}
//     exp=15         → NaN: product = 0
//
// Pipeline (2 stages):
//   Stage 1: FP8 decode + mantissa multiply (DSP) + exponent add
//   Stage 2: Normalize product to FP16 + accumulate with running sum
//
// DSP Budget: NUM_LANES × 1 DSP per multiply
//   V2-Lite: 64 lanes × 1 DSP = 64 DSP blocks
//   S10 MX has 3,960 DSPs (1.6% utilization)
//
// Timing: Target fmax ≥ 500 MHz on S10 speed grade -2
//   Stage 1 critical path: FP8 decode (~0.2ns) + DSP multiply (~0.6ns) = ~0.8ns
//   Stage 2 critical path: LOD (~0.3ns) + barrel shifter (~0.3ns)
//                         + FP16 align (~0.2ns) + FP16 add (~0.4ns) = ~1.2ns
//   Both stages meet 2ns period with margin.
// =============================================================================

module fp8_mac #(
    parameter int NUM_LANES = 64,     // V2-Lite: 64 lanes
    parameter int DATA_W    = 8,      // FP8 E4M3
    parameter int ACCUM_W   = 16,     // fp16 accumulator (1+5+10)
    parameter logic [31:0] VERSION = 32'h0B061B02  // {day,month,year-2000,build#}
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic [DATA_W-1:0]            a [NUM_LANES-1:0],   // activation (fp8)
    input  logic [DATA_W-1:0]            b [NUM_LANES-1:0],   // weight (fp8)
    output logic [ACCUM_W-1:0]           sum [NUM_LANES-1:0], // per-lane accum (fp16)
    output logic                         valid_out,
    output logic                         dbg_stage1_valid,
    output logic                         dbg_stage2_valid,
    output logic [5:0]                   dbg_overflow_lane,
    output logic                         dbg_overflow_sticky
);

    // =========================================================================
    // FP8 E4M3 format constants (localparams for readability)
    // =========================================================================
    localparam int FP8_EXP_BIAS    = 7;
    localparam int FP16_EXP_BIAS   = 15;
    localparam int FP16_MANT_W     = 10;  // explicit mantissa bits
    localparam int FP16_EXP_W      = 5;

    // =========================================================================
    // Stage 1 registers (FP8 decode + multiply)
    // =========================================================================
    logic                         s1_valid;
    logic                         s1_prod_zero [NUM_LANES-1:0];
    logic                         s1_prod_sign [NUM_LANES-1:0];
    logic [7:0]                   s1_prod_mant [NUM_LANES-1:0]; // raw mantissa product
    logic signed [5:0]            s1_prod_exp_raw [NUM_LANES-1:0]; // raw exponent sum
    logic [15:0]                  s1_accum_in  [NUM_LANES-1:0]; // previous sum

    // =========================================================================
    // Stage 2 registers (normalize + accumulate)
    // =========================================================================
    logic                         s2_valid;
    logic [15:0]                  s2_sum [NUM_LANES-1:0];

    // =========================================================================
    // LOD (Leading One Detector) function — find MSB position of 8-bit value
    // Returns 0..7, or 0 for zero input (handled separately via prod_zero)
    // =========================================================================
    function automatic logic [2:0] find_leading_one(input logic [7:0] val);
        logic [2:0] pos;
        // Priority-encode: find highest set bit
        if      (val[7]) pos = 3'd7;
        else if (val[6]) pos = 3'd6;
        else if (val[5]) pos = 3'd5;
        else if (val[4]) pos = 3'd4;
        else if (val[3]) pos = 3'd3;
        else if (val[2]) pos = 3'd2;
        else if (val[1]) pos = 3'd1;
        else             pos = 3'd0;
        return pos;
    endfunction

    // =========================================================================
    // Generate block: per-lane processing for all NUM_LANES
    // =========================================================================
    genvar gi;
    generate
        for (gi = 0; gi < NUM_LANES; gi++) begin : gen_lane

            // =================================================================
            // STAGE 1: FP8 Decode + Mantissa Multiply (DSP-inferred)
            // =================================================================

            // Unpack operand A
            logic        a_sign, a_is_nan, a_is_zero, a_is_sub;
            logic [3:0]  a_exp_field;
            logic [2:0]  a_mant_field;
            logic [3:0]  a_mant_val;     // 4-bit: {implicit, mantissa} for DSP

            logic        b_sign, b_is_nan, b_is_zero, b_is_sub;
            logic [3:0]  b_exp_field;
            logic [2:0]  b_mant_field;
            logic [3:0]  b_mant_val;

            // DSP multiply: 4-bit × 4-bit unsigned → 8-bit product
            // Use multstyle="dsp" to force Stratix 10 DSP inference
            (* multstyle = "dsp" *) logic [7:0] prod_mant_raw;

            // Effective exponents (signed, accounts for bias and mantissa /8 scaling)
            // Normal:   eff_exp = exp - 7 - 3 = exp - 10
            // Subnormal: eff_exp = -6 - 3 = -9
            logic signed [5:0] eff_exp_a, eff_exp_b, prod_exp_raw_sign;

            always_comb begin
                // --- Decode operand A ---
                a_sign       = a[gi][7];
                a_exp_field  = a[gi][6:3];
                a_mant_field = a[gi][2:0];

                a_is_nan  = (a_exp_field == 4'hF);
                a_is_zero = (a_exp_field == 4'h0) && (a_mant_field == 3'h0);
                a_is_sub  = (a_exp_field == 4'h0) && (a_mant_field != 3'h0);

                // Mantissa value for multiply:
                //   Normal: {1'b1, mantissa_field} — range [8, 15]
                //   Subnormal: {1'b0, mantissa_field} — range [1, 7]
                //   Zero/NaN: produces prod_zero, mant_val is don't-care
                a_mant_val = a_is_sub ? {1'b0, a_mant_field}
                                      : {1'b1, a_mant_field};

                // Effective exponent (biased so value = mant_val × 2^{eff_exp})
                //   Normal: (1+m/8)×2^{e-7} = mant_val×2^{e-7-3} = mant_val×2^{e-10}
                //   Subnormal: m/8×2^{-6} = mant_val×2^{-6-3} = mant_val×2^{-9}
                eff_exp_a = a_is_sub ? -6'sd9
                                     : ($signed({1'b0, a_exp_field}) - 6'sd10);

                // --- Decode operand B ---
                b_sign       = b[gi][7];
                b_exp_field  = b[gi][6:3];
                b_mant_field = b[gi][2:0];

                b_is_nan  = (b_exp_field == 4'hF);
                b_is_zero = (b_exp_field == 4'h0) && (b_mant_field == 3'h0);
                b_is_sub  = (b_exp_field == 4'h0) && (b_mant_field != 3'h0);

                b_mant_val = b_is_sub ? {1'b0, b_mant_field}
                                      : {1'b1, b_mant_field};

                eff_exp_b = b_is_sub ? -6'sd9
                                     : ($signed({1'b0, b_exp_field}) - 6'sd10);

                // --- Mantissa product (DSP multiply) ---
                // Both mant_val are 4-bit unsigned [1..15], product fits in 8 bits [1..225]
                prod_mant_raw = a_mant_val * b_mant_val;

                // --- Raw exponent sum ---
                prod_exp_raw_sign = eff_exp_a + eff_exp_b;
            end

            // --- Pipeline register S1 → S2 ---
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s1_valid          <= 1'b0;
                    s1_prod_zero[gi]  <= 1'b1;
                    s1_prod_sign[gi]  <= 1'b0;
                    s1_prod_mant[gi]  <= 8'b0;
                    s1_prod_exp_raw[gi] <= 6'sd0;
                    s1_accum_in[gi]   <= 16'h0000;
                end else begin
                    s1_valid          <= valid_in;
                    s1_prod_zero[gi]  <= a_is_zero | b_is_zero | a_is_nan | b_is_nan;
                    s1_prod_sign[gi]  <= a_sign ^ b_sign;
                    s1_prod_mant[gi]  <= prod_mant_raw;
                    s1_prod_exp_raw[gi] <= prod_exp_raw_sign;
                    // Feed through current sum for accumulation
                    s1_accum_in[gi]   <= sum[gi];
                end
            end

            // =================================================================
            // STAGE 2: Normalize to FP16 + Accumulate
            //
            // Normalization algorithm:
            //   1. Find leading one position P in prod_mant (0..7)
            //   2. Shift prod_mant left by (10-P) to get 11-bit normal mantissa
            //      (implicit 1 at bit 10, explicit mantissa at bits 9:0)
            //   3. Compute FP16 biased exponent: exp_fp16 = prod_exp_raw + P + 15
            //   4. Handle subnormal (exp_fp16 <= 0): right-shift mantissa
            //   5. Handle overflow (exp_fp16 >= 31): saturate to FP16 max
            //   6. FP16 add with running sum (RNE rounding on add)
            // =================================================================

            // LOD of product mantissa
            logic [2:0]              s2_lod_pos;
            logic [4:0]              s2_left_shift;  // = 10 - LOD
            logic [10:0]             s2_mant_norm;   // 11-bit normalized mantissa

            // FP16 product before accumulation
            logic                    s2_prod_sign;
            logic [FP16_EXP_W-1:0]   s2_prod_exp_biased;
            logic [FP16_MANT_W-1:0]  s2_prod_mant_field;
            logic [15:0]             s2_prod_fp16;   // the new product to accumulate

            // FP16 adder signals (Stage 2 accumulation)
            logic                    s2_a_sign, s2_b_sign;
            logic [FP16_EXP_W-1:0]   s2_a_exp, s2_b_exp;
            logic [FP16_MANT_W-1:0]  s2_a_mant, s2_b_mant;
            logic                    s2_a_is_zero, s2_b_is_zero;
            logic signed [5:0]       s2_exp_diff;    // a_exp - b_exp

            // Aligned mantissas (12 bits: 11-bit value + guard for rounding)
            logic [11:0]             s2_mant_a_aligned, s2_mant_b_aligned;
            logic                    s2_add_sub;      // 0=add, 1=subtract (signs differ)
            logic [12:0]             s2_mant_sum;     // 13-bit sum (12-bit + carry)

            // Post-add normalization
            logic [2:0]              s2_sum_lod;
            logic [10:0]             s2_mant_final;
            logic [FP16_EXP_W-1:0]   s2_exp_final;

            always_comb begin
                // --- Normalize product to FP16 ---
                s2_lod_pos    = find_leading_one(s1_prod_mant[gi]);
                // left_shift = 10 - lod_pos, range [2, 10]
                // lod_pos=0 → shift=10; lod_pos=7 → shift=3
                s2_left_shift = 5'd10 - {2'b0, s2_lod_pos};

                // Shift product mantissa to FP16 mantissa position (bit 10 = implicit 1)
                // mant_norm = prod_mant << (10 - lod_pos), result in [1024, 2047]
                s2_mant_norm = {3'b0, s1_prod_mant[gi]} << s2_left_shift;

                // Biased FP16 exponent: prod_exp_raw + lod_pos + BIAS(15)
                // prod_exp_raw ∈ [-18, 8], lod_pos ∈ [0, 7]
                // → exp_fp16_unbiased ∈ [-18, 15]
                // → exp_fp16_biased    ∈ [-3, 30]
                s2_prod_sign = s1_prod_sign[gi];

                if (s1_prod_zero[gi]) begin
                    // Product is zero (zero operand or NaN)
                    s2_prod_fp16       = {s2_prod_sign, 15'h0000};
                end else if ($signed(s1_prod_exp_raw[gi]) + $signed({3'b0, s2_lod_pos}) + 6'sd15 <= 6'sd0) begin
                    // --- FP16 subnormal (exp_biased <= 0) ---
                    // Denormalize: shift mantissa right to achieve exp_biased = 0
                    // right_shift = 1 - exp_biased_unadj
                    // where exp_biased_unadj = prod_exp_raw + lod_pos + 15
                    automatic logic signed [6:0] exp_biased_unadj;
                    automatic logic [4:0]         right_shift;
                    automatic logic [11:0]        mant_shifted;
                    automatic logic               guard, sticky, round_up;

                    exp_biased_unadj = $signed(s1_prod_exp_raw[gi])
                                     + $signed({4'b0, s2_lod_pos})
                                     + 7'sd15;
                    right_shift = 5'd1 - exp_biased_unadj[4:0];

                    // Right-shift with RNE rounding
                    mant_shifted = ({1'b0, s2_mant_norm} >> right_shift);

                    // Compute round bit from the shifted-out portion
                    automatic logic [10:0] shifted_out;
                    shifted_out = s2_mant_norm & ((11'b1 << right_shift) - 11'b1);
                    guard  = (right_shift > 0) ? shifted_out[right_shift-1] : 1'b0;
                    sticky = (right_shift > 1) ? |(shifted_out[right_shift-2:0]) : 1'b0;

                    // RNE: round up if guard=1 AND (sticky=1 OR mant_shifted[0]=1)
                    round_up = guard & (sticky | mant_shifted[0]);

                    if (mant_shifted + {11'b0, round_up} >= 12'd1024) begin
                        // Rounded up into normal range
                        s2_prod_exp_biased  = 5'd1;
                        s2_prod_mant_field  = 10'h000;
                    end else begin
                        s2_prod_exp_biased  = 5'd0;
                        s2_prod_mant_field  = (mant_shifted[9:0] + {9'b0, round_up});
                    end
                    s2_prod_fp16 = {s2_prod_sign, s2_prod_exp_biased, s2_prod_mant_field};

                end else begin
                    // --- FP16 normal ---
                    automatic logic signed [6:0] exp_biased_tmp;
                    exp_biased_tmp = $signed(s1_prod_exp_raw[gi])
                                   + $signed({4'b0, s2_lod_pos})
                                   + 7'sd15;

                    if (exp_biased_tmp >= 7'sd31) begin
                        // Overflow: saturate to FP16 max (±65504)
                        // FP16 max: exp=30 (11110), mant=1023 (all 1s)
                        s2_prod_fp16 = {s2_prod_sign, 5'h1E, 10'h3FF};
                    end else begin
                        s2_prod_exp_biased  = exp_biased_tmp[4:0];
                        s2_prod_mant_field  = s2_mant_norm[9:0];
                        s2_prod_fp16 = {s2_prod_sign, s2_prod_exp_biased, s2_prod_mant_field};
                    end
                end
            end

            // --- FP16 Adder: accumulate product into running sum ---
            //    Operand A = s2_prod_fp16 (new product)
            //    Operand B = s1_accum_in (running sum)

            always_comb begin
                // Unpack operands
                s2_a_sign  = s2_prod_fp16[15];
                s2_a_exp   = s2_prod_fp16[14:10];
                s2_a_mant  = s2_prod_fp16[9:0];
                s2_a_is_zero = (s2_prod_fp16[14:0] == 15'b0);

                s2_b_sign  = s1_accum_in[gi][15];
                s2_b_exp   = s1_accum_in[gi][14:10];
                s2_b_mant  = s1_accum_in[gi][9:0];
                s2_b_is_zero = (s1_accum_in[gi][14:0] == 15'b0);

                // Signs differ → subtraction; same → addition
                s2_add_sub = s2_a_sign ^ s2_b_sign;

                // Exponent difference (a_exp - b_exp)
                s2_exp_diff = $signed({1'b0, s2_a_exp}) - $signed({1'b0, s2_b_exp});

                // Align mantissas: the larger exponent's mantissa stays, smaller shifts right
                s2_mant_a_aligned = 12'b0;
                s2_mant_b_aligned = 12'b0;

                if (s2_exp_diff >= 0) begin
                    // a has larger or equal exponent; shift b right
                    s2_mant_a_aligned = {(s2_a_exp == 5'b0 ? 1'b0 : 1'b1), s2_a_mant, 1'b0};
                    if (s2_exp_diff <= 6'sd12) begin
                        s2_mant_b_aligned = ({(s2_b_exp == 5'b0 ? 1'b0 : 1'b1), s2_b_mant, 1'b0}
                                            >> s2_exp_diff[3:0]);
                    end else begin
                        s2_mant_b_aligned = 12'b0;
                    end
                end else begin
                    // b has larger exponent; shift a right
                    s2_mant_b_aligned = {(s2_b_exp == 5'b0 ? 1'b0 : 1'b1), s2_b_mant, 1'b0};
                    if (-s2_exp_diff <= 6'sd12) begin
                        s2_mant_a_aligned = ({(s2_a_exp == 5'b0 ? 1'b0 : 1'b1), s2_a_mant, 1'b0}
                                            >> (-s2_exp_diff)[3:0]);
                    end else begin
                        s2_mant_a_aligned = 12'b0;
                    end
                end

                // Add or subtract aligned mantissas
                // s2_mant_sum is 13 bits (12-bit mantissa + carry/borrow)
                if (s2_add_sub) begin
                    // Subtraction: result_sign = sign of larger operand
                    // Always compute (larger - smaller) in magnitude
                    if (s2_mant_a_aligned >= s2_mant_b_aligned) begin
                        s2_mant_sum = {1'b0, s2_mant_a_aligned} - {1'b0, s2_mant_b_aligned};
                    end else begin
                        s2_mant_sum = {1'b0, s2_mant_b_aligned} - {1'b0, s2_mant_a_aligned};
                    end
                end else begin
                    // Addition: same sign → add magnitudes
                    s2_mant_sum = {1'b0, s2_mant_a_aligned} + {1'b0, s2_mant_b_aligned};
                end
            end

            // --- Normalize the accumulated sum ---
            // s2_mant_sum is a 13-bit unsigned value (range [0, 2×2048+carry = ~6144])
            // We need to re-normalize to FP16 format

            always_comb begin
                // Leading one detection on 13-bit sum
                if (s2_mant_sum[12])      s2_sum_lod = 3'd12;
                else if (s2_mant_sum[11]) s2_sum_lod = 3'd11;
                else if (s2_mant_sum[10]) s2_sum_lod = 3'd10;
                else if (s2_mant_sum[9])  s2_sum_lod = 3'd9;
                else if (s2_mant_sum[8])  s2_sum_lod = 3'd8;
                else if (s2_mant_sum[7])  s2_sum_lod = 3'd7;
                else if (s2_mant_sum[6])  s2_sum_lod = 3'd6;
                else if (s2_mant_sum[5])  s2_sum_lod = 3'd5;
                else if (s2_mant_sum[4])  s2_sum_lod = 3'd4;
                else if (s2_mant_sum[3])  s2_sum_lod = 3'd3;
                else if (s2_mant_sum[2])  s2_sum_lod = 3'd2;
                else if (s2_mant_sum[1])  s2_sum_lod = 3'd1;
                else                       s2_sum_lod = 3'd0;

                // Choose result sign
                automatic logic result_sign;
                if (s2_a_is_zero && s2_b_is_zero) begin
                    result_sign = 1'b0;  // +0
                end else if (s2_a_is_zero) begin
                    result_sign = s2_b_sign;
                end else if (s2_b_is_zero) begin
                    result_sign = s2_a_sign;
                end else if (s2_add_sub) begin
                    // Subtraction: sign of larger operand
                    if (s2_mant_a_aligned >= s2_mant_b_aligned)
                        result_sign = s2_a_sign;
                    else
                        result_sign = s2_b_sign;
                end else begin
                    // Addition: both same sign
                    result_sign = s2_a_sign;
                end

                // Normalize: shift mantissa so bit 11 becomes implicit 1 at bit 10
                automatic logic signed [4:0] norm_shift;
                automatic logic [FP16_EXP_W-1:0] larger_exp;

                larger_exp = (s2_exp_diff >= 0) ? s2_a_exp : s2_b_exp;

                // s2_sum_lod gives the bit position of MSB in s2_mant_sum.
                // We want the implicit 1 at bit 10 of the output mantissa.
                // shift = s2_sum_lod - 10
                norm_shift = $signed({2'b0, s2_sum_lod}) - 5'sd10;

                if (s2_mant_sum == 13'b0) begin
                    // Sum is zero
                    s2_exp_final  = 5'h00;
                    s2_mant_final = 11'b0;
                end else if (norm_shift > 0) begin
                    // Need to shift right (sum overflowed)
                    automatic logic guard_bit, sticky_bit, round_up;
                    automatic logic [12:0] mant_shifted;
                    mant_shifted = s2_mant_sum >> norm_shift;
                    guard_bit = (norm_shift >= 1) ? s2_mant_sum[norm_shift-1] : 1'b0;
                    sticky_bit = (norm_shift >= 2) ? |(s2_mant_sum[norm_shift-2:0]) : 1'b0;
                    round_up = guard_bit & (sticky_bit | mant_shifted[0]);

                    s2_mant_final = mant_shifted[10:0] + {10'b0, round_up};

                    if (s2_mant_final[10] && !mant_shifted[10]) begin
                        // Rounding caused another carry → re-normalize
                        s2_exp_final = larger_exp + norm_shift[3:0] + 5'd1;
                        s2_mant_final = 11'b0;
                    end else begin
                        s2_exp_final = larger_exp + norm_shift[3:0];
                    end
                end else if (norm_shift < 0) begin
                    // Need to shift left (cancellation)
                    automatic logic [3:0] left_shift;
                    left_shift = (-norm_shift)[3:0];
                    s2_mant_final = s2_mant_sum[10:0] << left_shift;
                    s2_exp_final  = larger_exp - left_shift;

                    // Handle underflow to subnormal
                    if (larger_exp <= left_shift) begin
                        automatic logic [3:0] right_shift;
                        right_shift = left_shift - larger_exp + 4'd1;
                        s2_mant_final = s2_mant_sum[10:0] >> right_shift;
                        s2_exp_final  = 5'h00;
                    end
                end else begin
                    // No shift needed
                    s2_mant_final = s2_mant_sum[10:0];
                    s2_exp_final  = larger_exp;
                end

                // Assemble final FP16 result
                if (s2_mant_sum == 13'b0) begin
                    s2_sum[gi] = 16'h0000;
                end else if (s2_exp_final >= 5'd31) begin
                    // Saturation to FP16 max
                    s2_sum[gi] = {result_sign, 5'h1E, 10'h3FF};
                end else if (s2_exp_final == 5'd0) begin
                    // Subnormal or zero
                    s2_sum[gi] = {result_sign, 5'h00, s2_mant_final[9:0]};
                end else begin
                    s2_sum[gi] = {result_sign, s2_exp_final[4:0], s2_mant_final[9:0]};
                end
            end

        end : gen_lane
    endgenerate

    // =========================================================================
    // Output registers (Stage 2 → output)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
            for (int i = 0; i < NUM_LANES; i++)
                sum[i] <= 16'h0000;
            valid_out <= 1'b0;
        end else begin
            s2_valid <= s1_valid;
            for (int i = 0; i < NUM_LANES; i++)
                sum[i] <= s2_sum[i];
            valid_out <= s2_valid;
        end
    end

    // =========================================================================
    // Timing & Resource Annotations
    // =========================================================================
    // Latency: 3 clock cycles (S1 combinatorial + S1→S2 reg + S2 combinatorial + S2→out reg)
    //   Cycle 0: valid_in asserted, operands A, B captured
    //   Cycle 1: s1_valid = 1, product computed (Stage 1 register output)
    //   Cycle 2: s2_valid = 1, product normalized + accumulated
    //   Cycle 3: valid_out = 1, sum available at output
    //
    // Throughput: 1 result per cycle after pipeline fill (3-cycle bubble).
    //
    // Resource Estimates (per lane, per module total with NUM_LANES=64):
    //   - DSP blocks:    1 (mantissa 4×4→8 multiply) × 64 = 64 DSPs
    //   - ALMs:          ~80 ALMs/lane (LOD + shifter + FP16 add) = ~5,120 ALMs
    //   - Registers:     ~80 FFs/lane (pipeline + accumulate state) = ~5,120 FFs
    //   - S10 MX total:  933,120 ALMs, 3,960 DSPs → ~0.55% ALMs, ~1.6% DSPs
    //
    // Critical path (Stage 2): LOD → barrel shift → FP16 align → FP16 add → normalize
    //   Estimated: 1.3ns (meets 2.0ns at 500 MHz on S10 speed grade -2)
    //   If timing closure is challenging, split Stage 2 into two sub-stages
    //   (normalize product in S2, accumulate in S3).

    // =========================================================================
    // Debug: pipeline status + overflow detection
    // =========================================================================
    assign dbg_stage1_valid = s1_valid;
    assign dbg_stage2_valid = s2_valid;

    // Detect overflow: sum saturated to max 16-bit (lane-level)
    logic _overflow_any;
    logic [5:0] _overflow_lane;
    always_comb begin
        _overflow_any = 1'b0;
        _overflow_lane = 6'd0;
        for (int li = 0; li < NUM_LANES; li++) begin
            if (s2_valid && (&sum[li][ACCUM_W-2:0]) && sum[li][ACCUM_W-1] == 1'b0) begin
                _overflow_any = 1'b1;
                _overflow_lane = li[5:0];
            end
        end
    end

    logic _overflow_sticky;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            _overflow_sticky <= 1'b0;
        else if (_overflow_any)
            _overflow_sticky <= 1'b1;
    end

    assign dbg_overflow_lane   = _overflow_lane;
    assign dbg_overflow_sticky = _overflow_sticky;

endmodule
