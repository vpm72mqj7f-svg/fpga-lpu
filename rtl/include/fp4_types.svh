//=============================================================================
// fp4_types.svh — fp4 E2M1 / fp8 E4M3 type definitions and decode LUTs
// Target: Intel Agilex 7 M-Series (AGM 039-F)
// DeepSeek V4 Pro inference datapath
//=============================================================================

`ifndef FP4_TYPES_SVH
`define FP4_TYPES_SVH

//-----------------------------------------------------------------------------
// fp4 E2M1 format: {sign[3], exp[2:1], mant[0]}
//   Normal:  (-1)^s × 2^(e-1) × (1 + m/2),  e ∈ {1,2,3}
//   Subnorm: (-1)^s × m/2,                   e = 0
//   Values: 0, ±0.25, ±0.5, ±0.75, ±1.0, ±1.5, ±2.0, ±3.0
//-----------------------------------------------------------------------------

// Decoded fp4 value: sign, 3-bit magnitude index (0-7), 2-bit shared exponent
typedef struct packed {
    logic        sign;      // 0=positive, 1=negative
    logic [2:0]  mag;       // magnitude index into FP4_POS_VALUES[0:7]
    logic [1:0]  exp;       // exponent field raw (for scale combination)
} fp4_decoded_t;

// fp4 × fp8 MAC input bundle (one MAC operation)
typedef struct packed {
    logic [3:0]  weight;    // fp4 E2M1 encoded weight
    logic [7:0]  scale;     // fp8 E4M3 per-group scale
    logic [7:0]  activ;     // fp8 E4M3 activation
    logic        valid;     // input valid
} fp4_mac_input_t;

// fp4 × fp8 MAC output
typedef struct packed {
    logic [31:0] result;    // FP32 accumulated result
    logic        valid;     // output valid
} fp4_mac_output_t;

//-----------------------------------------------------------------------------
// LUT: fp4 encoded → positive value × 16 (4-bit fractional)
//   We scale by ×16 to represent 0.25→4, 0.5→8, etc. as integers.
//   sign is handled separately.
//-----------------------------------------------------------------------------
// fp4 index  | value | ×16 integer
//   0 (0000) |   0.0 |   0
//   1 (0001) |  0.25 |   4
//   2 (0010) |  0.5  |   8
//   3 (0011) |  0.75 |  12
//   4 (0100) |  1.0  |  16
//   5 (0101) |  1.5  |  24
//   6 (0110) |  2.0  |  32
//   7 (0111) |  3.0  |  48

function automatic logic [5:0] fp4_mag_to_scaled(logic [2:0] mag);
    case (mag)
        3'd0: fp4_mag_to_scaled = 6'd0;
        3'd1: fp4_mag_to_scaled = 6'd4;
        3'd2: fp4_mag_to_scaled = 6'd8;
        3'd3: fp4_mag_to_scaled = 6'd12;
        3'd4: fp4_mag_to_scaled = 6'd16;
        3'd5: fp4_mag_to_scaled = 6'd24;
        3'd6: fp4_mag_to_scaled = 6'd32;
        3'd7: fp4_mag_to_scaled = 6'd48;
    endcase
endfunction

//-----------------------------------------------------------------------------
// FP8 E4M3 helper: extract sign, exponent, mantissa
//   Format: {sign[7], exp[6:3], mant[2:0]}
//   Normal:  (-1)^s × 2^(e-7) × (1 + m/8), e ∈ {1..14}
//   Subnorm: (-1)^s × 2^(-6) × m/8,         e = 0
//-----------------------------------------------------------------------------
typedef struct packed {
    logic        sign;
    logic [3:0]  exp;
    logic [2:0]  mant;
} fp8_decoded_t;

function automatic fp8_decoded_t decode_fp8(input logic [7:0] fp8);
    decode_fp8.sign = fp8[7];
    decode_fp8.exp  = fp8[6:3];
    decode_fp8.mant = fp8[2:0];
endfunction

`endif // FP4_TYPES_SVH
