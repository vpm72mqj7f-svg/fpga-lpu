//=============================================================================
// fp4_scale_reader.sv — group-wise fp4 scale lookup (PRE-DECODED)
//
// Stores fp8 E4M3 scales as PRE-DECODED 12-bit signed values (×256).
// Decode happens at WRITE time (off critical path), so the read path
// is a simple 1-cycle BRAM lookup with zero combinational decode.
//
// Default GROUP_SIZE=16 is the validated fp4 precision setting.
//=============================================================================

`include "fp4_params.svh"
`include "fp4_types.svh"

module fp4_scale_reader #(
    parameter int NUM_GROUPS  = 512,
    parameter int GROUP_SIZE  = 16,
    parameter int ADDR_WIDTH  = $clog2(NUM_GROUPS),
    parameter int ELEM_WIDTH  = 16,
    parameter int SCALE_WIDTH = 12   // pre-decoded: 12-bit signed (×256)
) (
    input  logic                   clk,
    input  logic                   rst_n,

    // Query interface
    input  logic                   q_valid,
    input  logic [ELEM_WIDTH-1:0]  q_elem_idx,
    output logic                   q_ready,

    // Result interface (1-cycle latency, pre-decoded scale)
    output logic                   r_valid,
    output logic [SCALE_WIDTH-1:0] r_scale,
    output logic [ADDR_WIDTH-1:0]  r_group_id,

    // Scale memory load port (writes raw fp8, stores pre-decoded 12b)
    input  logic                   wr_en,
    input  logic [ADDR_WIDTH-1:0]  wr_addr,
    input  logic [7:0]             wr_data          // raw fp8 E4M3
);

    // MLAB distributed RAM — stores PRE-DECODED 12-bit scale values
    (* ramstyle = "MLAB" *) logic [SCALE_WIDTH-1:0] scale_mem [NUM_GROUPS];
    logic [ADDR_WIDTH-1:0] group_id;

    assign q_ready = 1'b1;

    // GROUP_SIZE is power-of-2 (default 16), use shift instead of divider
    always_comb begin
        group_id = q_elem_idx >> $clog2(GROUP_SIZE);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_valid    <= 1'b0;
            r_scale    <= '0;
            r_group_id <= '0;
        end else begin
            // Write: decode raw fp8 → pre-decoded 12-bit at load time
            if (wr_en) begin
                scale_mem[wr_addr] <= fp8_to_scaled12(wr_data);
            end
            // Read: direct lookup, no combinational decode
            r_valid    <= q_valid;
            r_group_id <= group_id;
            if (q_valid) begin
                r_scale <= scale_mem[group_id];
            end
        end
    end

endmodule
