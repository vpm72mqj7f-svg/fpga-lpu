//=============================================================================
// mla_kv_cache.sv — Hardware KV cache for compressed key/value storage
//
// Stores low-rank K_latent and V_latent for past tokens. Supports:
//   - Write: store compressed K/V for new token position
//   - Read: retrieve K/V for any cached position (1-cycle latency)
//   - Auto-increment write pointer (ring buffer, NUM_SLOTS deep)
//
// The cache stores COMPRESSED representations. Decompression (K_latent→K)
// happens on-the-fly during attention score computation.
//=============================================================================

module mla_kv_cache #(
    parameter int NUM_SLOTS  = 64,
    parameter int K_LATENT   = 4,
    parameter int V_LATENT   = 4,
    parameter int DATA_W     = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Write port
    input  logic                         wr_en,
    input  logic [K_LATENT*DATA_W-1:0]   K_latent_flat,
    input  logic [V_LATENT*DATA_W-1:0]   V_latent_flat,
    output logic [$clog2(NUM_SLOTS)-1:0] wr_addr,      // address written to

    // Read port (1-cycle: addr in, data out next cycle)
    input  logic                         rd_en,
    input  logic [$clog2(NUM_SLOTS)-1:0] rd_addr,
    output logic                         rd_valid,
    output logic [K_LATENT*DATA_W-1:0]   rd_K_flat,
    output logic [V_LATENT*DATA_W-1:0]   rd_V_flat,

    // Status
    output logic [$clog2(NUM_SLOTS)-1:0] fill_count,    // number of valid entries
    output logic                         full,
    output logic                         empty
);

    localparam int ADDR_W = $clog2(NUM_SLOTS);

    // Cache storage
    logic [K_LATENT*DATA_W-1:0] K_mem [NUM_SLOTS];
    logic [V_LATENT*DATA_W-1:0] V_mem [NUM_SLOTS];
    logic                        valid  [NUM_SLOTS];

    // Write pointer (ring buffer)
    logic [ADDR_W-1:0] wr_ptr;
    logic [ADDR_W-1:0] entry_count;

    assign wr_addr    = wr_ptr;
    assign fill_count = entry_count;
    assign full       = (entry_count == NUM_SLOTS);
    assign empty      = (entry_count == '0);

    // Read: 1-cycle latency
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid  <= 1'b0;
            rd_K_flat <= '0;
            rd_V_flat <= '0;
        end else begin
            if (rd_en) begin
                rd_K_flat <= K_mem[rd_addr];
                rd_V_flat <= V_mem[rd_addr];
                rd_valid  <= valid[rd_addr];
            end else begin
                rd_valid <= 1'b0;
            end
        end
    end

    // Write and pointer management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr      <= '0;
            entry_count <= '0;
            for (int i = 0; i < NUM_SLOTS; i++) begin
                valid[i]  <= 1'b0;
                K_mem[i]  <= '0;
                V_mem[i]  <= '0;
            end
        end else begin
            if (wr_en) begin
                K_mem[wr_ptr]  <= K_latent_flat;
                V_mem[wr_ptr]  <= V_latent_flat;
                valid[wr_ptr]  <= 1'b1;
                wr_ptr         <= (wr_ptr == (NUM_SLOTS - 1)) ? '0 : (wr_ptr + 1'b1);
                if (!valid[wr_ptr])
                    entry_count <= entry_count + 1'b1;
            end
        end
    end

endmodule
