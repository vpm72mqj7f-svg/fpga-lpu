//=============================================================================
// fp4_mac.sv — fp4(E2M1) × fp8(E4M3) Multiply-Accumulate Unit
// Target: Intel Agilex 7 M-Series DSP (AGM 039-F)
//
// Production: 4-stage pipeline, pre-decoded fp8 scales, DSP block retiming.
// Used as the fundamental compute cell in fp4_systolic_cell / fp4_gemm_engine.
//=============================================================================

`include "fp4_types.svh"

(* altera_attribute = "-name DSP_BLOCK_BALANCING AUTO" *)
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
    (* altera_attribute = "-name ALLOW_RETIMING ON" *)
    logic [3:0]  s0_weight;
    (* altera_attribute = "-name ALLOW_RETIMING ON" *)
    logic [11:0] s0_scale;
    (* altera_attribute = "-name ALLOW_RETIMING ON" *)
    logic [7:0]  s0_activ;
    logic        s0_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_weight <= 4'd0;
            s0_scale  <= 12'd0;
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
    // Stage 1: fp4 Decode + fp8 Decode → Signed Operands
    //=========================================================================

    // --- fp4 weight decode (E2M1) ---
    logic        w_sign;
    logic [2:0]  w_mag;

    assign w_sign = s0_weight[3];
    assign w_mag  = s0_weight[2:0];

    logic [5:0]  w_mag_scaled;
    assign w_mag_scaled = fp4_mag_to_scaled(w_mag);

    logic [7:0]  w_signed;
    always_comb begin
        if (w_mag == 3'd0) begin
            w_signed = 8'd0;
        end else if (w_sign) begin
            w_signed = -{2'd0, w_mag_scaled};
        end else begin
            w_signed = {2'd0, w_mag_scaled};
        end
    end

    // --- fp8 activation decode (E4M3, decoded on-the-fly) ---
    logic        a_sign;
    logic [3:0]  a_exp;
    logic [2:0]  a_mant;
    assign a_sign = s0_activ[7];
    assign a_exp  = s0_activ[6:3];
    assign a_mant = s0_activ[2:0];

    // Extract bit-fields outside always_comb to avoid Icarus constant-select
    // limitation in always_* processes.
    wire [1:0] a_mant_21 = a_mant[2:1];

    // --- Pre-decoded scale (12-bit signed, decoded at load time) ---
    // s0_scale is already decoded by fp4_scale_reader — no decode needed here.
    // This removes ~4 LUT levels from the MAC critical path at 450 MHz.
    logic signed [11:0] sc_scaled;
    assign sc_scaled = $signed(s0_scale);

    // Move a_full outside always_comb so the bit-select a_full[11:0] is
    // resolved in a continuous assignment.
    logic [4:0]        a_shift;
    logic signed [15:0] a_full;
    assign a_shift = a_exp - 4'd2;
    assign a_full  = $signed({8'd0, 1'b1, a_mant}) << a_shift;
    wire signed [11:0] a_full_lo = a_full[11:0];

    logic signed [11:0] a_scaled;
    always_comb begin
        if (a_exp == 4'd0) begin
            a_scaled = $signed({1'b0, {8'd0, a_mant_21}});
        end else if (a_exp < 4'd2) begin
            a_scaled = $signed({1'b0, 1'b1, a_mant}) >>> 1;
        end else begin
            if (a_full > 16'sd2047)      a_scaled = 12'sd2047;
            else if (a_full < -16'sd2048) a_scaled = -12'sd2048;
            else                          a_scaled = a_full_lo;
        end
        if (a_sign) a_scaled = -a_scaled;
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
    // Stage 2: Base Multiply — 8b fp4 × 12b activation → 20b product
    //   altera_mult_add DSP IP (combinational); output registered in always_ff
    //=========================================================================
    logic signed [19:0] s2_base_product;
    logic [11:0]        s2_sc_scaled;
    logic               s2_valid;
    wire  signed [19:0] base_product_full;

    altera_mult_add #(.A_WIDTH(8), .B_WIDTH(12), .PIPE_STAGES(0))
    u_dsp_base (
        .clock(clk), .a(s1_w_signed), .b(s1_a_scaled), .result(base_product_full)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_base_product <= 20'sd0;
            s2_sc_scaled    <= 12'd0;
            s2_valid        <= 1'b0;
        end else begin
            s2_base_product <= base_product_full[19:0];
            s2_sc_scaled    <= s1_sc_scaled;
            s2_valid        <= s1_valid;
        end
    end

    //=========================================================================
    // Stage 3: Scale Multiply — 20b base × 12b scale → 32b product
    //   altera_mult_add DSP IP (combinational); output registered in always_ff
    //   Rescale by >>>8 to match fixed-point.
    //=========================================================================
    logic signed [31:0] s3_product;
    logic               s3_valid;
    wire  signed [31:0] scale_product_full;

    altera_mult_add #(.A_WIDTH(20), .B_WIDTH(12), .PIPE_STAGES(0))
    u_dsp_scale (
        .clock(clk), .a(s2_base_product), .b(s2_sc_scaled), .result(scale_product_full)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_product <= 32'sd0;
            s3_valid   <= 1'b0;
        end else begin
            s3_product <= scale_product_full >>> 8;
            s3_valid   <= s2_valid;
        end
    end

    //=========================================================================
    // Stage 4: Accumulate — 32b product → accumulator
    //   Sign-extend product, add to running sum. Saturation prevents overflow
    //   wrap-around on deep accumulations (inspired by TALOS-V2 sat16).
    //   Clear on accum_clr (new token start).
    //=========================================================================
    logic [ACCUM_WIDTH-1:0] accumulator;
    logic                   s4_valid;

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
            if (!old_sign && !val_sign && sum_sign)
                sat_acc = {1'b0, {(ACCUM_WIDTH-1){1'b1}}};
            else if (old_sign && val_sign && !sum_sign)
                sat_acc = {1'b1, {(ACCUM_WIDTH-1){1'b0}}};
            else
                sat_acc = sum_raw;
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= '0;
            s4_valid    <= 1'b0;
        end else if (accum_clr) begin
            accumulator <= '0;
            s4_valid    <= 1'b0;
        end else begin
            if (s3_valid) begin
                accumulator <= sat_acc(accumulator, s3_product);
            end
            s4_valid <= s3_valid;
        end
    end

    assign mac_out.result = accumulator;
    assign mac_out.valid  = s4_valid;

endmodule
