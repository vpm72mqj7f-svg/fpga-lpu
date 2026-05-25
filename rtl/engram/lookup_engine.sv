//=============================================================================
// lookup_engine.sv — O(1) hash-based embedding lookup
//
// Pipeline: token_ids → hash_unit (4 cycles) → sram_cache (1 cycle)
//   Hit:  data returned in 6 cycles total (4 hash + 1 cache + 1 output)
//   Miss: LPDDR request, data returned after external memory latency
//
// Interfaces:
//   - input: token_ids_flat (N_GRAMS × 32-bit), with valid/ready handshake
//   - output: embedding_flat (EMBED_DIM × 32-bit), with valid/ready
//   - LPDDR: request/response for cache miss fill
//=============================================================================

module lookup_engine #(
    parameter int N_GRAMS           = 4,
    parameter int EMBED_DIM         = 8,
    parameter int NUM_CACHE_ENTRIES = 512,
    parameter int DATA_WIDTH        = EMBED_DIM * 32
) (
    input  logic                     clk,
    input  logic                     rst_n,

    // Token input
    input  logic                     in_valid,
    input  logic [N_GRAMS*32-1:0]    token_ids_flat,
    output logic                     in_ready,

    // Embedding output
    output logic                     out_valid,
    input  logic                     out_ready,
    output logic [DATA_WIDTH-1:0]    embedding_flat,

    // LPDDR miss interface
    output logic                     lpddr_rd_req,
    output logic [31:0]              lpddr_rd_addr,
    input  logic [DATA_WIDTH-1:0]    lpddr_rd_data,
    input  logic                     lpddr_rd_valid
);

    typedef enum logic [2:0] {
        S_IDLE,
        S_HASHING,
        S_CACHE_LOOKUP,
        S_RESOLVE,
        S_MISS_REQ,
        S_MISS_WAIT,
        S_OUTPUT
    } state_t;

    state_t state;

    // Hash unit signals
    logic        hash_valid_in;
    logic        hash_ready;
    logic        hash_valid_out;
    logic [31:0] hash_result;

    // Cache signals
    logic                     cache_lookup_valid;
    logic [31:0]              cache_lookup_hash;
    logic                     cache_hit;
    logic [DATA_WIDTH-1:0]    cache_data;

    // Stored hash for miss path
    logic [31:0] miss_hash;

    hash_unit #(
        .N_GRAMS(N_GRAMS)
    ) u_hash (
        .clk            (clk),
        .rst_n          (rst_n),
        .valid_in       (hash_valid_in),
        .token_ids_flat (token_ids_flat),
        .ready_out      (hash_ready),
        .valid_out      (hash_valid_out),
        .hash_out       (hash_result)
    );

    sram_cache #(
        .NUM_ENTRIES(NUM_CACHE_ENTRIES),
        .EMBED_DIM(EMBED_DIM)
    ) u_cache (
        .clk           (clk),
        .rst_n         (rst_n),
        .lookup_valid  (cache_lookup_valid),
        .lookup_hash   (cache_lookup_hash),
        .lookup_hit    (cache_hit),
        .lookup_data   (cache_data),
        .fill_valid    (lpddr_rd_valid),
        .fill_hash     (miss_hash),
        .fill_data     (lpddr_rd_data)
    );

    assign in_ready  = (state == S_IDLE);
    assign hash_valid_in = in_valid && in_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            cache_lookup_valid <= 1'b0;
            cache_lookup_hash  <= '0;
            miss_hash         <= '0;
            lpddr_rd_req      <= 1'b0;
            lpddr_rd_addr     <= '0;
            out_valid         <= 1'b0;
            embedding_flat    <= '0;
        end else begin
            cache_lookup_valid <= 1'b0;
            lpddr_rd_req      <= 1'b0;
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            case (state)
                S_IDLE: begin
                    if (in_valid && in_ready) begin
                        state <= S_HASHING;
                    end
                end

                S_HASHING: begin
                    // Wait for hash pipeline to complete (4 cycles)
                    if (hash_valid_out) begin
                        cache_lookup_valid <= 1'b1;
                        cache_lookup_hash  <= hash_result;
                        miss_hash          <= hash_result;
                        state <= S_CACHE_LOOKUP;
                    end
                end

                S_CACHE_LOOKUP: begin
                    state <= S_RESOLVE;
                end

                S_RESOLVE: begin
                    if (cache_hit) begin
                        out_valid      <= 1'b1;
                        embedding_flat <= cache_data;
                        state <= S_OUTPUT;
                    end else begin
                        lpddr_rd_req  <= 1'b1;
                        lpddr_rd_addr <= miss_hash;
                        state <= S_MISS_REQ;
                    end
                end

                S_MISS_REQ: begin
                    state <= S_MISS_WAIT;
                end

                S_MISS_WAIT: begin
                    if (lpddr_rd_valid) begin
                        embedding_flat <= lpddr_rd_data;
                        out_valid      <= 1'b1;
                        state <= S_OUTPUT;
                    end
                end

                S_OUTPUT: begin
                    if (out_ready) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
