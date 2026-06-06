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

    assign lookup_idx = lookup_hash[INDEX_WIDTH-1:0];
    assign lookup_tag = lookup_hash[31:INDEX_WIDTH];
    assign fill_idx   = fill_hash[INDEX_WIDTH-1:0];

    // Altera syncram IP: 3 independent simple-dual-port RAMs
    // M20K for tag/data (high density), MLAB for valid bits (low latency)
    logic          valid_q;
    logic [TAG_WIDTH-1:0]  tag_q;
    logic [DATA_WIDTH-1:0] data_q;

    altera_syncram #(.WIDTH(1), .DEPTH(NUM_ENTRIES), .RAM_BLOCK_TYPE("MLAB"))
    u_valid (
        .clock(clk), .wren(fill_valid), .wraddress(fill_idx), .data(1'b1),
        .rdaddress(lookup_idx), .q(valid_q)
    );

    altera_syncram #(.WIDTH(TAG_WIDTH), .DEPTH(NUM_ENTRIES), .RAM_BLOCK_TYPE("M20K"))
    u_tag (
        .clock(clk), .wren(fill_valid), .wraddress(fill_idx),
        .data(fill_hash[31:INDEX_WIDTH]), .rdaddress(lookup_idx), .q(tag_q)
    );

    altera_syncram #(.WIDTH(DATA_WIDTH), .DEPTH(NUM_ENTRIES), .RAM_BLOCK_TYPE("M20K"))
    u_data (
        .clock(clk), .wren(fill_valid), .wraddress(fill_idx),
        .data(fill_data), .rdaddress(lookup_idx), .q(data_q)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lookup_hit  <= 1'b0;
            lookup_data <= '0;
        end else begin
            if (lookup_valid) begin
                lookup_hit  <= valid_q && (tag_q == lookup_tag);
                lookup_data <= data_q;
            end else begin
                lookup_hit  <= 1'b0;
            end
        end
    end

endmodule
