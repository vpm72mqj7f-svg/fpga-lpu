//=============================================================================
// sram_cache.sv — direct-mapped embedding cache (SRAM)
//
// Uses lower INDEX_WIDTH bits of hash as address. Upper bits stored as tag.
// Direct-mapped: uniform hash distribution gives good hit rate.
// Embedding data stored as EMBED_DIM × 32-bit packed per entry.
//=============================================================================

module sram_cache #(
    parameter int NUM_ENTRIES   = 512,
    parameter int EMBED_DIM     = 8,
    parameter int INDEX_WIDTH   = $clog2(NUM_ENTRIES),
    parameter int TAG_WIDTH     = 32 - INDEX_WIDTH,
    parameter int DATA_WIDTH    = EMBED_DIM * 32
) (
    input  logic                     clk,
    input  logic                     rst_n,

    // Lookup interface (1-cycle: address in, hit+data out next cycle)
    input  logic                     lookup_valid,
    input  logic [31:0]              lookup_hash,
    output logic                     lookup_hit,
    output logic [DATA_WIDTH-1:0]    lookup_data,

    // Fill interface (write after miss)
    input  logic                     fill_valid,
    input  logic [31:0]              fill_hash,
    input  logic [DATA_WIDTH-1:0]    fill_data
);

    logic [INDEX_WIDTH-1:0]  lookup_idx;
    logic [TAG_WIDTH-1:0]    lookup_tag;
    logic [INDEX_WIDTH-1:0]  fill_idx;

    // Cache storage — use unpacked array for BRAM inference
    logic                    entry_valid  [NUM_ENTRIES];
    logic [TAG_WIDTH-1:0]    entry_tag    [NUM_ENTRIES];
    logic [DATA_WIDTH-1:0]   entry_data   [NUM_ENTRIES];

    assign lookup_idx = lookup_hash[INDEX_WIDTH-1:0];
    assign lookup_tag = lookup_hash[31:INDEX_WIDTH];
    assign fill_idx   = fill_hash[INDEX_WIDTH-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lookup_hit  <= 1'b0;
            lookup_data <= '0;
        end else begin
            // Read: check valid and tag match
            if (lookup_valid) begin
                lookup_hit  <= entry_valid[lookup_idx] &&
                              (entry_tag[lookup_idx] == lookup_tag);
                lookup_data <= entry_data[lookup_idx];
            end else begin
                lookup_hit  <= 1'b0;
            end

            // Write: fill on miss
            if (fill_valid) begin
                entry_valid[fill_idx] <= 1'b1;
                entry_tag[fill_idx]   <= fill_hash[31:INDEX_WIDTH];
                entry_data[fill_idx]  <= fill_data;
            end
        end
    end

endmodule
