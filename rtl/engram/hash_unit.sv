//=============================================================================
// hash_unit.sv — 4-cycle pipelined hash for N-gram token IDs
//
// Mixes N_GRAMS token IDs into a 32-bit hash using multiply-xor-shift.
// FPGA-friendly: one 32×32 multiply per cycle (DSP), no variable shifts.
//
// Pipeline:
//   Cycle 0: h = (token0 ^ token1) * MURMUR_M
//   Cycle 1: h = (h ^ token2) * MURMUR_M
//   Cycle 2: h = (h ^ token3) * MURMUR_M
//   Cycle 3: final_mix(h) → hash_out, valid_out=1
//=============================================================================

module hash_unit #(
    parameter int N_GRAMS = 4
) (
    input  logic                     clk,
    input  logic                     rst_n,

    input  logic                     valid_in,
    input  logic [N_GRAMS*32-1:0]    token_ids_flat,
    output logic                     ready_out,

    output logic                     valid_out,
    output logic [31:0]              hash_out
);

    localparam logic [31:0] MURMUR_M = 32'h5bd1e995;
    localparam logic [31:0] FMIX_M   = 32'h85ebca6b;

    typedef enum logic [1:0] { ST_IDLE, ST_MIX1, ST_MIX2, ST_FINAL } state_t;
    state_t state;
    logic [31:0] h;
    logic [N_GRAMS*32-1:0] tokens_r;  // registered token_ids for pipeline

    assign ready_out = (state == ST_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_IDLE;
            h         <= '0;
            tokens_r  <= '0;
            valid_out <= 1'b0;
            hash_out  <= '0;
        end else begin
            valid_out <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (valid_in) begin
                        tokens_r <= token_ids_flat;
                        h     <= (token_ids_flat[0*32+:32] ^ token_ids_flat[1*32+:32]) * MURMUR_M;
                        state <= ST_MIX1;
                    end
                end

                ST_MIX1: begin
                    h     <= (h ^ tokens_r[2*32+:32]) * MURMUR_M;
                    state <= ST_MIX2;
                end

                ST_MIX2: begin
                    h     <= (h ^ tokens_r[3*32+:32]) * MURMUR_M;
                    state <= ST_FINAL;
                end

                ST_FINAL: begin
                    hash_out  <= (h ^ (h >> 16)) * FMIX_M;
                    valid_out <= 1'b1;
                    state     <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
