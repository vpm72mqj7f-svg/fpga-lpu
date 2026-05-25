//=============================================================================
// fp4_scale_reader.sv — group-wise fp4 scale lookup
//
// Computes group_id = elem_idx / GROUP_SIZE and returns the stored scale.
// Default GROUP_SIZE=16 is the validated fp4 precision setting.
//=============================================================================

`include "fp4_params.svh"

module fp4_scale_reader #(
    parameter int NUM_GROUPS  = 512,
    parameter int GROUP_SIZE  = 16,
    parameter int ADDR_WIDTH  = $clog2(NUM_GROUPS),
    parameter int ELEM_WIDTH  = 16,
    parameter int SCALE_WIDTH = 8
) (
    input  logic                   clk,
    input  logic                   rst_n,

    // Query interface
    input  logic                   q_valid,
    input  logic [ELEM_WIDTH-1:0]  q_elem_idx,
    output logic                   q_ready,

    // Result interface (1-cycle latency)
    output logic                   r_valid,
    output logic [SCALE_WIDTH-1:0] r_scale,
    output logic [ADDR_WIDTH-1:0]  r_group_id,

    // Scale memory load port
    input  logic                   wr_en,
    input  logic [ADDR_WIDTH-1:0]  wr_addr,
    input  logic [SCALE_WIDTH-1:0] wr_data
);

    logic [SCALE_WIDTH-1:0] scale_mem [NUM_GROUPS];
    logic [ADDR_WIDTH-1:0] group_id;

    assign q_ready = 1'b1;

    always_comb begin
        group_id = (q_elem_idx / GROUP_SIZE);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_valid    <= 1'b0;
            r_scale    <= '0;
            r_group_id <= '0;
        end else begin
            if (wr_en) begin
                scale_mem[wr_addr] <= wr_data;
            end
            r_valid    <= q_valid;
            r_group_id <= group_id;
            if (q_valid) begin
                r_scale <= scale_mem[group_id];
            end
        end
    end

endmodule
