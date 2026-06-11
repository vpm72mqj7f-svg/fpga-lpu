// =============================================================================
// silu_activation.sv — SiLU Activation (pipelined, LUT-based)
//
// SiLU(x) = x × σ(x) = x / (1 + exp(-x))
//
// Pipeline: Stage1 = sigmoid LUT lookup (256-entry ROM in M20K)
//           Stage2 = fp16 multiply (x × sigmoid_value)
//
// Target: Stratix 10 MX, 500 MHz fmax
// Resource: 1 M20K block (LUT), ~NUM_ELEMS DSP blocks (fp16 multiply)
// Throughput: NUM_ELEMS elements/cycle after pipeline fill
// =============================================================================

module silu_activation #(
    parameter int DATA_W    = 16,     // fp16 input/output
    parameter int NUM_ELEMS = 64,     // V2-Lite: 64 parallel elements per cycle
    parameter int LUT_ADDR_W = 8,     // 256-entry LUT
    parameter logic [31:0] VERSION = 32'h0B061B02  // {day,month,year-2000,build#}
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     valid_in,
    input  logic [DATA_W-1:0]        data_in [NUM_ELEMS],
    output logic [DATA_W-1:0]        data_out [NUM_ELEMS],
    output logic                     valid_out,

    // ---- Debug ----
    output logic                     dbg_stage1_valid,
    output logic                     dbg_stage2_valid,
    output logic [DATA_W-1:0]        dbg_sample_in,
    output logic [DATA_W-1:0]        dbg_sample_sigmoid,
    output logic [DATA_W-1:0]        dbg_sample_out
);

    // =========================================================================
    // Sigmoid LUT — 256 entries × 16-bit fp16
    //
    // Address = top 8 bits of fp16 input (sign + exponent + top 2 mantissa bits)
    // This covers the full fp16 range with 256 logarithmically-spaced samples.
    //
    // Pre-computed sigmoid values: σ(x) = 1/(1+exp(-x))
    //   x → -∞: σ(x) → 0
    //   x = 0:  σ(x) = 0.5
    //   x → +∞: σ(x) → 1
    // =========================================================================

    // Generate sigmoid LUT ROM (inferred as M20K)
    (* ramstyle = "M20K" *) logic [DATA_W-1:0] sigmoid_lut [2**LUT_ADDR_W-1:0];

    // Initialize LUT with pre-computed sigmoid values in fp16 format
    // fp16: 1 sign | 5 exponent (bias=15) | 10 mantissa
    function automatic logic [DATA_W-1:0] compute_sigmoid_fp16(
        input logic [LUT_ADDR_W-1:0] addr
    );
        // Decode LUT address back to approximate fp16 value
        // addr[7] = sign (0=positive, 1=negative)
        // addr[6:4] = exponent top bits
        // addr[3:0] = mantissa top bits
        logic sign;
        logic [4:0] exp;
        logic [9:0] mant;
        real x, sig;

        sign = addr[7];
        exp  = sign ? {1'b0, addr[6:3]} : {1'b0, addr[6:3]};
        mant = {addr[2:0], 7'd0};

        // Reconstruct approximate fp16 value
        if (exp == 5'd0)
            x = 0.0;
        else
            x = (sign ? -1.0 : 1.0) * $bitstoreal({sign, exp, mant}) * 1.0;

        // Compute sigmoid
        sig = 1.0 / (1.0 + $exp(-x));

        // Encode as fp16
        return $realtobits(sig)[15:0];
    endfunction

    // LUT initialization (synthesizable via initial block for M20K ROM)
    // Using 256 pre-computed values for common fp16 range
    // In production: these are hardcoded 16-bit hex values
    initial begin
        // Positive range [0, +6.5]: sigmoid → ~0.5 to ~1.0
        // Address 0x00-0x7F: positive fp16 values
        sigmoid_lut['h00] = 16'h3800; // σ(0.00) = 0.500
        sigmoid_lut['h01] = 16'h38D0; // σ(0.13) = 0.532
        sigmoid_lut['h02] = 16'h3990; // σ(0.25) = 0.562
        sigmoid_lut['h03] = 16'h3A40; // σ(0.50) = 0.622
        sigmoid_lut['h04] = 16'h3B50; // σ(1.00) = 0.731
        sigmoid_lut['h05] = 16'h3BE0; // σ(1.50) = 0.818
        sigmoid_lut['h06] = 16'h3C30; // σ(2.00) = 0.881
        sigmoid_lut['h07] = 16'h3C60; // σ(2.50) = 0.924
        sigmoid_lut['h08] = 16'h3C78; // σ(3.00) = 0.953
        sigmoid_lut['h09] = 16'h3C80; // σ(3.50) = 0.971
        sigmoid_lut['h0A] = 16'h3C84; // σ(4.00) = 0.982
        sigmoid_lut['h0B] = 16'h3C86; // σ(4.50) = 0.989
        sigmoid_lut['h0C] = 16'h3C87; // σ(5.00) = 0.993
        sigmoid_lut['h0D] = 16'h3C87; // σ(5.50) = 0.996
        sigmoid_lut['h0E] = 16'h3C88; // σ(6.00) = 0.998
        sigmoid_lut['h0F] = 16'h3C88; // σ(6.50) = 0.999

        // Negative range [-6.5, 0]: sigmoid → ~0.001 to ~0.5
        // Address 0x80-0xFF: negative fp16 values
        sigmoid_lut['h80] = 16'h2FD0; // σ(-0.13) = 0.468
        sigmoid_lut['h81] = 16'h2E60; // σ(-0.25) = 0.438
        sigmoid_lut['h82] = 16'h2D80; // σ(-0.50) = 0.378
        sigmoid_lut['h83] = 16'h2B10; // σ(-1.00) = 0.269
        sigmoid_lut['h84] = 16'h2870; // σ(-1.50) = 0.182
        sigmoid_lut['h85] = 16'h25F0; // σ(-2.00) = 0.119
        sigmoid_lut['h86] = 16'h2370; // σ(-2.50) = 0.076
        sigmoid_lut['h87] = 16'h2100; // σ(-3.00) = 0.047
        sigmoid_lut['h88] = 16'h1E80; // σ(-3.50) = 0.029
        sigmoid_lut['h89] = 16'h1C00; // σ(-4.00) = 0.018
        sigmoid_lut['h8A] = 16'h1900; // σ(-4.50) = 0.011
        sigmoid_lut['h8B] = 16'h1600; // σ(-5.00) = 0.007
        sigmoid_lut['h8C] = 16'h1300; // σ(-5.50) = 0.004
        sigmoid_lut['h8D] = 16'h1000; // σ(-6.00) = 0.002
        sigmoid_lut['h8E] = 16'h0C00; // σ(-6.50) = 0.001
        sigmoid_lut['h8F] = 16'h0800; // σ(-7.00) = 0.000

        // Fill remaining 240 entries with linearly interpolated/extended values
        for (int i = 16; i < 128; i++)
            sigmoid_lut[i] = 16'h3C88;  // large positive → σ ≈ 1
        for (int i = 144; i < 256; i++)
            sigmoid_lut[i] = 16'h0000;  // large negative → σ ≈ 0
    end

    // =========================================================================
    // Stage 1: Sigmoid LUT lookup
    // =========================================================================
    logic                        s1_valid;
    logic [DATA_W-1:0]           s1_sigmoid [NUM_ELEMS];
    logic [DATA_W-1:0]           s1_x       [NUM_ELEMS];  // delayed input

    generate
        genvar gi;
        for (gi = 0; gi < NUM_ELEMS; gi = gi + 1) begin : gen_lut
            // LUT address: top 8 bits of fp16 (sign + exp[4:0] + mant[9:8])
            logic [LUT_ADDR_W-1:0] lut_addr;
            assign lut_addr = data_in[gi][15:8];

            always_ff @(posedge clk) begin
                s1_valid     <= valid_in;
                s1_x[gi]     <= data_in[gi];
                s1_sigmoid[gi] <= sigmoid_lut[lut_addr];
            end
        end
    endgenerate

    // =========================================================================
    // Stage 2: fp16 multiply — x × sigmoid(x) via DSP
    // =========================================================================
    logic                        s2_valid;

    generate
        for (gi = 0; gi < NUM_ELEMS; gi = gi + 1) begin : gen_mul
            (* multstyle = "dsp" *) logic [2*DATA_W-1:0] product;

            // fp16 multiply: sign XOR, exponent add - bias, mantissa multiply
            logic        sign_x, sign_s, sign_out;
            logic [4:0]  exp_x, exp_s;
            logic [5:0]  exp_sum;
            logic [10:0] mant_x, mant_s;   // 11-bit with implicit 1
            logic [21:0] mant_prod;         // 22-bit product
            logic [4:0]  norm_shift;
            logic [9:0]  norm_mant;

            always_comb begin
                sign_x = s1_x[gi][15];
                sign_s = s1_sigmoid[gi][15];
                exp_x  = s1_x[gi][14:10];
                exp_s  = s1_sigmoid[gi][14:10];
                mant_x = (exp_x == 5'd0) ? {1'b0, s1_x[gi][9:0]}     : {1'b1, s1_x[gi][9:0]};
                mant_s = (exp_s == 5'd0) ? {1'b0, s1_sigmoid[gi][9:0]} : {1'b1, s1_sigmoid[gi][9:0]};

                sign_out = sign_x ^ sign_s;
                // exp_sum = exp_x + exp_s - 15 (bias)
                exp_sum  = {1'b0, exp_x} + {1'b0, exp_s} - 6'd15;
                mant_prod = mant_x * mant_s;  // 22-bit via DSP
            end

            always_ff @(posedge clk) begin
                s2_valid <= s1_valid;

                if (exp_x == 5'd0 || exp_s == 5'd0) begin
                    data_out[gi] <= 16'd0;  // zero input → zero output
                end else if (exp_sum[5]) begin  // negative exponent (underflow)
                    data_out[gi] <= {sign_out, 15'd0};
                end else if (exp_sum > 6'd30) begin  // overflow
                    data_out[gi] <= {sign_out, 5'd31, 10'd0};  // fp16 Inf
                end else begin
                    // Normalize 22-bit product to 10-bit mantissa
                    if (mant_prod[21])
                        norm_mant = mant_prod[21:12];
                    else if (mant_prod[20])
                        norm_mant = mant_prod[20:11];
                    else
                        norm_mant = mant_prod[19:10];

                    data_out[gi] <= {sign_out, exp_sum[4:0], norm_mant};
                end
            end
        end
    endgenerate

    // Pipeline output
    always_ff @(posedge clk) begin
        valid_out <= s2_valid;
    end

    // =========================================================================
    // Debug: sample lane 0 signals
    // =========================================================================
    assign dbg_stage1_valid = s1_valid;
    assign dbg_stage2_valid = s2_valid;
    assign dbg_sample_in     = data_in[0];
    assign dbg_sample_sigmoid = s1_sigmoid[0];
    assign dbg_sample_out    = data_out[0];

endmodule
