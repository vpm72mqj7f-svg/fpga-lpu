//=============================================================================
// mla_kv_cache.sv — Hardware KV cache for compressed key/value storage
//
// Ring buffer, NUM_SLOTS deep. Stores K_latent + V_latent per token.
// Supports two storage modes:
//   STORE_W=32 (default): Q12 fixed-point — backward compatible
//   STORE_W=8  (FP8):    FP8 E4M3 — 4× KV capacity, FP8→Q12 LUT on read
//
// Output is always Q12 (32-bit) regardless of storage mode.
// Production: NUM_SLOTS = lpu_config_pkg::LPU_KV_CACHE_SLOTS
//=============================================================================

`include "lpu_config.svh"

module mla_kv_cache #(
    parameter int NUM_SLOTS  = lpu_config_pkg::LPU_KV_CACHE_SLOTS,
    parameter int K_LATENT   = 4,
    parameter int V_LATENT   = 4,
    parameter int DATA_W     = 32,     // output width (Q12=32)
    parameter int STORE_W    = 32      // storage width (32=Q12 legacy, 8=FP8)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Write port (normal decode path — Q12 input when STORE_W=32)
    input  logic                         wr_en,
    input  logic [K_LATENT*DATA_W-1:0]   K_latent_flat,
    input  logic [V_LATENT*DATA_W-1:0]   V_latent_flat,
    output logic [$clog2(NUM_SLOTS)-1:0] wr_addr,

    // Preload port (CPU prefill KV → DMA → FPGA HBM → cache)
    // Accepts DATA_W-wide data; internally packs to STORE_W
    input  logic                         preload_en,
    input  logic [K_LATENT*DATA_W-1:0]   preload_K_flat,
    input  logic [V_LATENT*DATA_W-1:0]   preload_V_flat,

    // Read port (always outputs Q12 regardless of STORE_W)
    input  logic                         rd_en,
    input  logic [$clog2(NUM_SLOTS)-1:0] rd_addr,
    output logic                         rd_valid,
    output logic [K_LATENT*DATA_W-1:0]   rd_K_flat,
    output logic [V_LATENT*DATA_W-1:0]   rd_V_flat,

    // Status
    output logic [CNT_W-1:0] fill_count,
    output logic                         full,
    output logic                         empty
);

    localparam int ADDR_W = $clog2(NUM_SLOTS);
    localparam int CNT_W  = $clog2(NUM_SLOTS+1);

    // Storage width per element
    localparam int K_STORE_TOTAL = K_LATENT * STORE_W;
    localparam int V_STORE_TOTAL = V_LATENT * STORE_W;

    // Cache storage — Altera syncram IP
    logic [K_STORE_TOTAL-1:0] K_q_store;
    logic [V_STORE_TOTAL-1:0] V_q_store;
    logic                     valid_q;

    logic eff_wr;
    logic [K_STORE_TOTAL-1:0] K_wr_store;
    logic [V_STORE_TOTAL-1:0] V_wr_store;

    // Pack DATA_W input → STORE_W storage
    generate
        if (STORE_W == 32) begin : g_store_q12
            // Q12 mode: pass-through
            assign K_wr_store = preload_en ? preload_K_flat : K_latent_flat;
            assign V_wr_store = preload_en ? preload_V_flat : V_latent_flat;
        end else begin : g_store_fp8
            // FP8 mode: select low byte of each 32-bit word
            // preload data arrives as FP8 in byte lanes
            for (genvar gi = 0; gi < K_LATENT; gi++) begin : gen_k_pack
                assign K_wr_store[gi*8 +: 8] = preload_en
                    ? preload_K_flat[gi*DATA_W +: 8]
                    : K_latent_flat[gi*DATA_W +: 8];
            end
            for (genvar gi = 0; gi < V_LATENT; gi++) begin : gen_v_pack
                assign V_wr_store[gi*8 +: 8] = preload_en
                    ? preload_V_flat[gi*DATA_W +: 8]
                    : V_latent_flat[gi*DATA_W +: 8];
            end
        end
    endgenerate

    assign eff_wr = wr_en || preload_en;

    altera_syncram #(.WIDTH(K_STORE_TOTAL), .DEPTH(NUM_SLOTS), .RAM_BLOCK_TYPE("M20K"))
    u_K (.clock(clk), .wren(eff_wr), .wraddress(wr_ptr), .data(K_wr_store),
         .rdaddress(rd_addr), .q(K_q_store));

    altera_syncram #(.WIDTH(V_STORE_TOTAL), .DEPTH(NUM_SLOTS), .RAM_BLOCK_TYPE("M20K"))
    u_V (.clock(clk), .wren(eff_wr), .wraddress(wr_ptr), .data(V_wr_store),
         .rdaddress(rd_addr), .q(V_q_store));

    altera_syncram #(.WIDTH(1), .DEPTH(NUM_SLOTS), .RAM_BLOCK_TYPE("MLAB"))
    u_valid (.clock(clk), .wren(eff_wr), .wraddress(wr_ptr), .data(1'b1),
             .rdaddress(rd_addr), .q(valid_q));

    // ── FP8→Q12 conversion LUT (used when STORE_W=8) ──
    // FP8 E4M3: s[7] e[6:3] m[2:0], bias=7
    // Q12 = round(fp8_value × 4096)
    function automatic logic signed [31:0] fp8_to_q12(input logic [7:0] fp8);
        logic sign;
        logic [3:0] exp;
        logic [2:0] mant;
        logic signed [31:0] mag;
        sign = fp8[7];
        exp  = fp8[6:3];
        mant = fp8[2:0];
        if (exp == 4'b0000) begin
            // Subnormal: value = 2^(-6) × m/8
            mag = mant * 32'sd8;  // × 4096 / 2^6 / 8 = 8
        end else begin
            // Normal: value = 2^(e-7) × (1 + m/8)
            // Q12 = 2^(e-7) × (8 + m) × 4096 / 8 = 2^(e-7) × (8+m) × 512
            mag = (32'sd8 + {29'b0, mant}) * 32'sd512;
            // Apply exponent (max shift 8 positions, e from 1 to 15)
            if (exp <= 4'd7)
                mag = mag >>> (4'd7 - exp);
            else
                mag = mag << (exp - 4'd7);
        end
        // Clamp to Q12 range
        if (mag > 32'sd32767) mag = 32'sd32767;
        fp8_to_q12 = sign ? -mag : mag;
    endfunction

    // Read path: convert from store format to Q12 output
    logic [K_LATENT*DATA_W-1:0] K_q;
    logic [V_LATENT*DATA_W-1:0] V_q;

    generate
        if (STORE_W == 32) begin : g_read_q12
            assign K_q = K_q_store;
            assign V_q = V_q_store;
        end else begin : g_read_fp8
            // FP8→Q12 per element
            for (genvar gi = 0; gi < K_LATENT; gi++) begin : gen_k_fp8_q12
                assign K_q[gi*DATA_W +: DATA_W] = fp8_to_q12(K_q_store[gi*8 +: 8]);
            end
            for (genvar gi = 0; gi < V_LATENT; gi++) begin : gen_v_fp8_q12
                assign V_q[gi*DATA_W +: DATA_W] = fp8_to_q12(V_q_store[gi*8 +: 8]);
            end
        end
    endgenerate

    // Write pointer (ring buffer)
    logic [ADDR_W-1:0] wr_ptr;
    logic [CNT_W-1:0]  entry_count;

    assign wr_addr    = wr_ptr;
    assign fill_count = entry_count;
    assign full       = (entry_count == NUM_SLOTS);
    assign empty      = (entry_count == '0);

    // Read capture
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid  <= 1'b0;
            rd_K_flat <= '0;
            rd_V_flat <= '0;
        end else begin
            if (rd_en) begin
                rd_K_flat <= K_q;
                rd_V_flat <= V_q;
                rd_valid  <= valid_q;
            end
        end
    end

    // Write and pointer management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr      <= '0;
            entry_count <= '0;
        end else begin
            if (eff_wr) begin
                wr_ptr <= (wr_ptr == (NUM_SLOTS - 1)) ? '0 : (wr_ptr + 1'b1);
                if (entry_count < NUM_SLOTS)
                    entry_count <= entry_count + 1'b1;
            end
        end
    end

endmodule
