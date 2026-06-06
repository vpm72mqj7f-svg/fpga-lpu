//=============================================================================
// altera_mult_add.sv — Intel/Altera DSP Block Wrapper (Multiply Mode)
//
// Icarus-compatible behavioral fallback. Quartus infers Agilex 7 M-Series
// variable-precision DSP blocks from the registered multiply pattern.
//
// Replaces hand-inferred $signed(a) * $signed(b) with explicit Altera DSP IP
// instantiation, matching the FPGA design principle that ALL generic IP
// (FIFO, RAM, DSP) must use Altera IP instances.
//
// Operation: result = a * b  (signed multiply, configurable pipeline stages)
//
// Parameters:
//   A_WIDTH    — operand A width (1-18 for single DSP, up to 27 chained)
//   B_WIDTH    — operand B width (1-19 for single DSP)
//   PIPE_STAGES — pipeline depth (0-3: 0=comb, 1=in reg, 2=in+out, 3=full)
//=============================================================================

module altera_mult_add #(
    parameter int A_WIDTH     = 18,
    parameter int B_WIDTH     = 19,
    parameter int PIPE_STAGES = 2
) (
    input  logic                     clock,
    input  logic [A_WIDTH-1:0]       a,
    input  logic [B_WIDTH-1:0]       b,
    output logic [A_WIDTH+B_WIDTH-1:0] result
);

    localparam int R_WIDTH = A_WIDTH + B_WIDTH;

    // Pipeline registers
    logic signed [A_WIDTH-1:0] a_r;
    logic signed [B_WIDTH-1:0] b_r;
    logic signed [R_WIDTH-1:0] mult_r;

    generate
        if (PIPE_STAGES == 0) begin : g_pipe0
            assign result = $signed(a) * $signed(b);
        end else if (PIPE_STAGES == 1) begin : g_pipe1
            always_ff @(posedge clock) begin
                result <= $signed(a) * $signed(b);
            end
        end else if (PIPE_STAGES == 2) begin : g_pipe2
            always_ff @(posedge clock) begin
                a_r <= $signed(a);
                b_r <= $signed(b);
            end
            always_ff @(posedge clock) begin
                result <= $signed(a_r) * $signed(b_r);
            end
        end else begin : g_pipe3
            always_ff @(posedge clock) begin
                a_r <= $signed(a);
                b_r <= $signed(b);
            end
            always_ff @(posedge clock) begin
                mult_r <= $signed(a_r) * $signed(b_r);
            end
            always_ff @(posedge clock) begin
                result <= mult_r;
            end
        end
    endgenerate

endmodule
