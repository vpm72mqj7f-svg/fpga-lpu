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

`include "lpu_config.svh"

module mla_attention_v2 #(
    parameter int HIDDEN     = lpu_config_pkg::LPU_HIDDEN,
    parameter int K_LATENT   = lpu_config_pkg::LPU_K_LATENT,
    parameter int V_LATENT   = lpu_config_pkg::LPU_V_LATENT,
    parameter int NUM_SLOTS  = lpu_config_pkg::LPU_KV_CACHE_SLOTS,
    parameter int MAX_POS    = lpu_config_pkg::LPU_MAX_SEQ_LEN,
    parameter int WINDOW_SIZE = lpu_config_pkg::LPU_SLIDING_WINDOW,
    parameter int WEIGHT_W   = lpu_config_pkg::LPU_WEIGHT_WIDTH,
    parameter int DATA_W     = lpu_config_pkg::LPU_DATA_WIDTH
) (
    input  logic clk, rst_n,

    // Hidden state input
    input  logic                         in_valid,
    input  logic [HIDDEN*DATA_W-1:0]     hidden_flat,
    output logic                         in_ready,

    // Position (for RoPE)
    input  logic [$clog2(MAX_POS)-1:0]   position,

    // Sliding window control: 1 = window mode, 0 = full attention (backward compatible)
    input  logic                         window_mode,

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

    // KV cache preload (CPU prefill → DMA → FPGA HBM → cache)
    input  logic                         cache_preload_en,
    input  logic [K_LATENT*DATA_W-1:0]   cache_preload_K_flat,
    input  logic [V_LATENT*DATA_W-1:0]   cache_preload_V_flat,

    // Attention output
    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [HIDDEN*DATA_W-1:0]     y_flat
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
    logic [$clog2(NUM_SLOTS+1)-1:0] cache_fill_count;
    logic cache_empty;

    // Top-level FSM
    typedef enum logic [3:0] {
        S_IDLE,
        S_QKV_PROJ,
        S_ROPE,
        S_CACHE_WR,
        S_SCORE_INIT,
        S_CACHE_RD,
        S_ATTN_SCORE,
        S_EXP_LOOP,
        S_INV,
        S_ACCUM_INIT,
        S_CACHE_RD2,
        S_ACCUM_LOOP,
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
        .preload_en(cache_preload_en),
        .preload_K_flat(cache_preload_K_flat),
        .preload_V_flat(cache_preload_V_flat),
        .rd_en(cache_rd_en), .rd_addr(cache_rd_addr),
        .rd_valid(cache_rd_valid), .rd_K_flat(cache_K_out),
        .rd_V_flat(cache_V_out),
        .fill_count(cache_fill_count), .full(), .empty(cache_empty)
    );

    // Internal registers for attention computation
    logic signed [DATA_W-1:0] Q_r   [HIDDEN];
    logic signed [DATA_W-1:0] V_r   [HIDDEN];
    logic signed [DATA_W-1:0] scores [NUM_SLOTS];
    logic [$clog2(NUM_SLOTS)-1:0] score_idx;
    logic signed [DATA_W-1:0] score_max;

    // Sliding window registers
    logic                                    window_mode_r;       // latched at S_IDLE
    logic [$clog2(NUM_SLOTS)-1:0]            cache_wr_addr_latched; // physical addr of current write
    logic [$clog2(NUM_SLOTS)-1:0]            window_base;          // first physical addr in window
    logic [$clog2(NUM_SLOTS+1)-1:0]          window_count;         // entries to iterate

    // Softmax & weighted-sum accumulators
    logic [$clog2(NUM_SLOTS)-1:0] exp_idx;
    logic [$clog2(NUM_SLOTS)-1:0] accum_idx;
    logic signed [31:0]           exp_sum;
    logic signed [31:0]           inv_scale;
    logic signed [DATA_W-1:0]     V_acc [HIDDEN];

    // Pre-write cache state: latched before KV write so we know if
    // this token is the first one (self-attention only, no history).
    logic cache_was_empty;

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

    // Cache control (write + read for both score and accum passes)
    // Window mode: translate logical index to physical ring-buffer address
    always_comb begin
        cache_wr_en   = (state == S_CACHE_WR);
        cache_K_in    = K_latent_flat;
        cache_V_in    = V_latent_flat;
        cache_rd_en   = (state == S_CACHE_RD) || (state == S_CACHE_RD2);

        // Address generation with sliding window support
        if (window_mode_r && (cache_fill_count > WINDOW_SIZE)) begin
            logic [$clog2(NUM_SLOTS)-1:0] rel_idx, candidate;
            rel_idx  = (state == S_CACHE_RD2) ? accum_idx : score_idx;
            candidate = window_base + rel_idx;
            // Single-wrap correction (candidate can exceed NUM_SLOTS-1 by at most one wrap)
            cache_rd_addr = (candidate >= NUM_SLOTS[$clog2(NUM_SLOTS)-1:0])
                            ? (candidate - NUM_SLOTS[$clog2(NUM_SLOTS)-1:0])
                            : candidate;
        end else begin
            cache_rd_addr = (state == S_CACHE_RD2) ? accum_idx : score_idx;
        end
    end

    // RoPE trigger
    assign rope_in_valid = (state == S_ROPE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            out_valid  <= 1'b0;
            score_idx  <= '0;
            score_max  <= 32'sh80000000;
            exp_idx    <= '0;
            exp_sum    <= '0;
            inv_scale  <= '0;
            accum_idx  <= '0;
            y_flat     <= '0;
            for (int d = 0; d < HIDDEN; d++) begin
                Q_r[d] <= '0; V_r[d] <= '0; V_acc[d] <= '0;
            end
            for (int i = 0; i < NUM_SLOTS; i++) scores[i] <= '0;
        end else begin
            out_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (in_valid) begin
                        cache_was_empty <= cache_empty;
                        window_mode_r   <= window_mode;  // latch per-token for determinism
                        state <= S_QKV_PROJ;
                    end
                end

                S_QKV_PROJ: begin
                    if (qkv_out_valid) begin
                        for (int d = 0; d < HIDDEN; d++) begin
                            Q_r[d] <= $signed(Q_flat[d*DATA_W +: DATA_W]);
                            V_r[d] <= $signed(V_flat[d*DATA_W +: DATA_W]);
                        end
                        state <= S_ROPE;
                    end
                end

                S_ROPE: begin
                    if (rope_out_valid) begin
                        for (int d = 0; d < HIDDEN; d++)
                            Q_r[d] <= $signed(rope_Q_flat[d*DATA_W +: DATA_W]);
                        state <= S_CACHE_WR;
                    end
                end

                S_CACHE_WR: begin
                    cache_wr_addr_latched <= cache_wr_addr;  // latch for window base calc
                    state <= S_SCORE_INIT;
                end

                // ================================================================
                // Pass 1: Score computation — Q·K_latent for window positions
                // ================================================================
                S_SCORE_INIT: begin
                    score_idx  <= '0;
                    score_max  <= 32'sh80000000;
                    if (cache_was_empty) begin
                        state <= S_OUTPUT;  // first token — self-attn only, output V_r
                    end else begin
                        // Compute sliding window range
                        if (window_mode_r && (cache_fill_count > WINDOW_SIZE)) begin
                            window_count <= WINDOW_SIZE[$clog2(NUM_SLOTS+1)-1:0];
                            // window_base = (cache_wr_addr_latched - WINDOW_SIZE + 1) mod NUM_SLOTS
                            if (cache_wr_addr_latched >= (WINDOW_SIZE - 1))
                                window_base <= cache_wr_addr_latched - (WINDOW_SIZE - 1);
                            else
                                window_base <= cache_wr_addr_latched + NUM_SLOTS - (WINDOW_SIZE - 1);
                        end else begin
                            // Full attention: window = entire cache
                            window_count <= cache_fill_count;
                            window_base <= '0;
                        end
                        state <= S_CACHE_RD;
                    end
                end

                S_CACHE_RD: begin
                    state <= S_ATTN_SCORE;
                end

                S_ATTN_SCORE: begin
                    logic signed [DATA_W-1:0] dot;
                    dot = '0;
                    for (int d = 0; d < K_LATENT; d++)
                        dot = dot + ((Q_r[d] * $signed(cache_K_out[d*DATA_W +: DATA_W])) >>> 12);
                    scores[score_idx] <= dot;
                    if (dot > score_max) score_max <= dot;

                    if (score_idx == window_count - 1) begin
                        exp_idx  <= '0;
                        exp_sum  <= '0;
                        state <= S_EXP_LOOP;
                    end else begin
                        score_idx <= score_idx + 1'b1;
                        state <= S_CACHE_RD;
                    end
                end

                // ================================================================
                // Softmax: exp(score - max) → exp_sum for each position
                // ================================================================
                S_EXP_LOOP: begin
                    exp_sum <= exp_sum + $signed(exp_lut(scores[exp_idx] - score_max));

                    if (exp_idx == window_count - 1) begin
                        state <= S_INV;
                    end else begin
                        exp_idx <= exp_idx + 1'b1;
                    end
                end

                // ================================================================
                // Reciprocal: inv_scale = 4096 * 4096 / exp_sum (Q12)
                // Bring-up: use / operator. Production: DSP divider.
                // ================================================================
                S_INV: begin
                    if (exp_sum == 0)
                        inv_scale <= 4096;  // uniform weights
                    else
                        inv_scale <= (4096 * 4096) / exp_sum;
                    accum_idx <= '0;
                    for (int d = 0; d < HIDDEN; d++) V_acc[d] <= '0;
                    state <= S_ACCUM_INIT;
                end

                // ================================================================
                // Pass 2: Re-read cache, compute weighted V_latent sum
                // ================================================================
                S_ACCUM_INIT: begin
                    state <= S_CACHE_RD2;
                end

                S_CACHE_RD2: begin
                    state <= S_ACCUM_LOOP;
                end

                S_ACCUM_LOOP: begin
                    // weight_Q12 = exp(score - max) * inv_scale / 4096
                    //   = exp_lut(scores[accum_idx] - score_max) * inv_scale >>> 12
                    logic signed [31:0] exp_val;
                    logic signed [31:0] weight;
                    exp_val = $signed(exp_lut(scores[accum_idx] - score_max));
                    weight  = (exp_val * inv_scale) >>> 12;

                    for (int d = 0; d < V_LATENT; d++)
                        V_acc[d] <= V_acc[d] +
                            ((weight * $signed(cache_V_out[d*DATA_W +: DATA_W])) >>> 12);

                    if (accum_idx == window_count - 1) begin
                        state <= S_OUTPUT;
                    end else begin
                        accum_idx <= accum_idx + 1'b1;
                        state <= S_CACHE_RD2;
                    end
                end

                // ================================================================
                // Output: V_acc (weighted V_latent) on low dims, V_r on rest
                // ================================================================
                S_OUTPUT: begin
                    if (out_ready) begin
                        if (cache_was_empty) begin
                            // First token: self-attention only, output V_r
                            for (int d = 0; d < HIDDEN; d++)
                                y_flat[d*DATA_W+:DATA_W] <= V_r[d];
                        end else begin
                            // Multi-token: V_acc (weighted V_latent sum) in low dims
                            for (int d = 0; d < V_LATENT; d++)
                                y_flat[d*DATA_W+:DATA_W] <= V_acc[d];
                            for (int d = V_LATENT; d < HIDDEN; d++)
                                y_flat[d*DATA_W+:DATA_W] <= V_r[d];
                        end
                        out_valid <= 1'b1;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
