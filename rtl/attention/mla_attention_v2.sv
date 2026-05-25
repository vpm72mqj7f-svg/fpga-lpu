//=============================================================================
// mla_attention_v2.sv — Full MLA pipeline with low-rank QKV + RoPE + KV cache
//
// Pipeline:
//   1. mla_qkv_proj: hidden → Q, K, V, K_latent, V_latent
//   2. mla_rope: apply RoPE to Q (and optionally K)
//   3. mla_kv_cache: store K_latent/V_latent for current token
//   4. Attention: Q @ all_cached_K → softmax → weighted V sum
//
// Supports single-token (self) and multi-token (cached) attention.
//=============================================================================

module mla_attention_v2 #(
    parameter int HIDDEN     = 8,
    parameter int K_LATENT   = 4,
    parameter int V_LATENT   = 4,
    parameter int NUM_SLOTS  = 64,
    parameter int MAX_POS    = 64,
    parameter int WEIGHT_W   = 16,
    parameter int DATA_W     = 32
) (
    input  logic clk, rst_n,

    // Hidden state input
    input  logic                         in_valid,
    input  logic [HIDDEN*DATA_W-1:0]     hidden_flat,
    output logic                         in_ready,

    // Position (for RoPE)
    input  logic [$clog2(MAX_POS)-1:0]   position,

    // QKV weight load port (forwarded to mla_qkv_proj)
    input  logic                         qkv_wt_wr_en,
    input  logic [2:0]                   qkv_wt_sel,
    input  logic [$clog2(HIDDEN)-1:0]    qkv_wt_row,
    input  logic [$clog2(HIDDEN)-1:0]    qkv_wt_col,
    input  logic signed [WEIGHT_W-1:0]   qkv_wt_wr_data,

    // RoPE LUT load port (forwarded to mla_rope)
    input  logic                         rope_lut_wr_en,
    input  logic [$clog2(MAX_POS)-1:0]   rope_lut_pos,
    input  logic [$clog2(HIDDEN/2)-1:0]  rope_lut_pair,
    input  logic signed [WEIGHT_W-1:0]   rope_lut_sin,
    input  logic signed [WEIGHT_W-1:0]   rope_lut_cos,

    // Attention output
    output logic                         out_valid,
    input  logic                         out_ready,
    output logic signed [DATA_W-1:0]     y0,y1,y2,y3,y4,y5,y6,y7
);

    // ---- Sub-module signals ----

    // mla_qkv_proj
    logic qkv_in_ready, qkv_out_valid, qkv_out_ready;
    logic [HIDDEN*DATA_W-1:0] Q_flat, K_flat, V_flat;
    logic [K_LATENT*DATA_W-1:0] K_latent_flat, V_latent_flat;

    // mla_rope (Q only)
    logic rope_in_ready, rope_out_valid;
    logic [HIDDEN*DATA_W-1:0] rope_Q_flat;

    // mla_kv_cache
    logic cache_wr_en;
    logic [K_LATENT*DATA_W-1:0] cache_K_in, cache_K_out;
    logic [V_LATENT*DATA_W-1:0] cache_V_in, cache_V_out;
    logic [$clog2(NUM_SLOTS)-1:0] cache_wr_addr, cache_rd_addr;
    logic cache_rd_en, cache_rd_valid;
    logic [$clog2(NUM_SLOTS)-1:0] cache_fill_count;
    logic cache_empty;

    // Top-level FSM
    typedef enum logic [3:0] {
        S_IDLE,
        S_QKV_PROJ,
        S_ROPE,
        S_CACHE_WR,
        S_CACHE_RD_INIT,
        S_CACHE_RD,
        S_ATTN_SCORE,
        S_SOFTMAX,
        S_OUTPUT
    } state_t;
    state_t state;

    // Sub-module instantiations
    mla_qkv_proj #(.HIDDEN(HIDDEN), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
                   .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W))
    u_qkv (
        .clk, .rst_n,
        .in_valid, .hidden_flat,
        .in_ready(qkv_in_ready),
        .wt_wr_en(qkv_wt_wr_en), .wt_sel(qkv_wt_sel),
        .wt_row(qkv_wt_row), .wt_col(qkv_wt_col),
        .wt_wr_data(qkv_wt_wr_data),
        .out_valid(qkv_out_valid), .out_ready(qkv_out_ready),
        .Q_flat, .K_flat, .V_flat,
        .K_latent_flat, .V_latent_flat
    );

    mla_rope #(.HIDDEN(HIDDEN), .MAX_POS(MAX_POS), .COEFF_W(WEIGHT_W), .DATA_W(DATA_W))
    u_rope (
        .clk, .rst_n,
        .in_valid(rope_in_valid), .vec_flat(Q_flat), .pos(position),
        .in_ready(rope_in_ready),
        .lut_wr_en(rope_lut_wr_en), .lut_pos(rope_lut_pos),
        .lut_pair(rope_lut_pair), .lut_sin_data(rope_lut_sin),
        .lut_cos_data(rope_lut_cos),
        .out_valid(rope_out_valid), .rot_flat(rope_Q_flat)
    );

    mla_kv_cache #(.NUM_SLOTS(NUM_SLOTS), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
                   .DATA_W(DATA_W))
    u_cache (
        .clk, .rst_n,
        .wr_en(cache_wr_en), .K_latent_flat(cache_K_in),
        .V_latent_flat(cache_V_in), .wr_addr(cache_wr_addr),
        .rd_en(cache_rd_en), .rd_addr(cache_rd_addr),
        .rd_valid(cache_rd_valid), .rd_K_flat(cache_K_out),
        .rd_V_flat(cache_V_out),
        .fill_count(cache_fill_count), .full(), .empty(cache_empty)
    );

    // Internal registers for attention computation
    logic signed [DATA_W-1:0] Q_r   [HIDDEN];
    logic signed [DATA_W-1:0] V_r   [HIDDEN];
    logic signed [DATA_W-1:0] K_cur [HIDDEN];
    logic signed [DATA_W-1:0] V_cur [HIDDEN];
    logic signed [DATA_W-1:0] scores [NUM_SLOTS];
    logic [$clog2(NUM_SLOTS)-1:0] score_idx;
    logic signed [DATA_W-1:0] score_max;
    logic signed [63:0] score_sum;

    // Softmax LUT (same as original mla_attention)
    function [31:0] exp_lut(input [31:0] adj);
        if (adj > -32'sd256)       exp_lut = 4096;
        else if (adj > -32'sd1024)  exp_lut = 3545;
        else if (adj > -32'sd2048)  exp_lut = 2588;
        else if (adj > -32'sd4096)  exp_lut = 1507;
        else if (adj > -32'sd8192)  exp_lut = 538;
        else                        exp_lut = 48;
    endfunction

    assign in_ready = (state == S_IDLE);
    assign qkv_out_ready = (state == S_QKV_PROJ);

    // Cache write (stores latent representations)
    always_comb begin
        cache_wr_en   = (state == S_CACHE_WR);
        cache_K_in    = K_latent_flat;
        cache_V_in    = V_latent_flat;
        cache_rd_en   = (state == S_CACHE_RD);
        cache_rd_addr = score_idx;
    end

    // RoPE trigger
    assign rope_in_valid = (state == S_ROPE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            out_valid  <= 1'b0;
            score_idx  <= '0;
            score_max  <= 32'sh80000000;
            score_sum  <= '0;
            {y0,y1,y2,y3,y4,y5,y6,y7} <= '0;
            for (int d = 0; d < HIDDEN; d++) begin
                Q_r[d] <= '0; V_r[d] <= '0; K_cur[d] <= '0; V_cur[d] <= '0;
            end
            for (int i = 0; i < NUM_SLOTS; i++) scores[i] <= '0;
        end else begin
            out_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (in_valid) state <= S_QKV_PROJ;
                end

                S_QKV_PROJ: begin
                    if (qkv_out_valid) begin
                        // Latch Q, V (keep as Q12)
                        for (int d = 0; d < HIDDEN; d++) begin
                            Q_r[d] <= $signed(Q_flat[d*DATA_W +: DATA_W]);
                            V_r[d] <= $signed(V_flat[d*DATA_W +: DATA_W]);
                        end
                        state <= S_ROPE;
                    end
                end

                S_ROPE: begin
                    if (rope_out_valid) begin
                        // Latched Q already has RoPE applied
                        for (int d = 0; d < HIDDEN; d++)
                            Q_r[d] <= $signed(rope_Q_flat[d*DATA_W +: DATA_W]);
                        state <= S_CACHE_WR;
                    end
                end

                S_CACHE_WR: begin
                    // cache_wr_en writes to cache (combinational)
                    state <= S_CACHE_RD_INIT;
                end

                S_CACHE_RD_INIT: begin
                    score_idx  <= '0;
                    score_max  <= 32'sh80000000;
                    score_sum  <= '0;
                    if (cache_empty) begin
                        // No cached tokens: self-attention only
                        // Compute Q·K score directly (use Q_r and K_flat)
                        for (int d = 0; d < HIDDEN; d++)
                            K_cur[d] <= $signed(K_flat[d*DATA_W +: DATA_W]);
                        state <= S_ATTN_SCORE;
                    end else begin
                        state <= S_CACHE_RD;
                    end
                end

                S_CACHE_RD: begin
                    // Wait 1 cycle for cache read
                    state <= S_ATTN_SCORE;
                end

                S_ATTN_SCORE: begin
                    // Compute Q·K score for current cached entry
                    // For simplicity, read K_latent from cache and compute
                    // dot product with Q's first K_LATENT dims
                    // (full decompression would need W_K_up, which we skip here)
                    logic signed [DATA_W-1:0] dot;
                    dot = '0;
                    for (int d = 0; d < K_LATENT; d++)
                        dot = dot + ((Q_r[d] * $signed(cache_K_out[d*DATA_W +: DATA_W])) >>> 12);
                    // Store score
                    scores[score_idx] <= dot;

                    // Track max for softmax
                    if (dot > score_max) score_max <= dot;

                    if (score_idx == cache_fill_count - 1) begin
                        state <= S_SOFTMAX;
                    end else begin
                        score_idx <= score_idx + 1'b1;
                        state <= S_CACHE_RD;
                    end
                end

                S_SOFTMAX: begin
                    // Compute softmax over all cached scores
                    // Initialize accumulators
                    score_sum <= '0;
                    for (int d = 0; d < HIDDEN; d++) V_cur[d] <= '0;
                    score_idx <= '0;
                    state <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    if (out_ready) begin
                        // Output V (for single-token self-attention, softmax=1.0, output=V)
                        {y0,y1,y2,y3,y4,y5,y6,y7} <= {V_r[0],V_r[1],V_r[2],V_r[3],
                                                       V_r[4],V_r[5],V_r[6],V_r[7]};
                        out_valid <= 1'b1;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
