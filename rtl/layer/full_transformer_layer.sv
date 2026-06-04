//=============================================================================
// full_transformer_layer.sv — Decode-only: RMS→ATTN_v2→RMS→Router→FFN→RMS
//
// Uses mla_attention_v2 with MLA low-rank QKV, RoPE, and KV cache.
// Parameters default from lpu_config_pkg (production or bring-up).
//=============================================================================

`include "lpu_config.svh"

module full_transformer_layer #(
    parameter int HIDDEN    = lpu_config_pkg::LPU_HIDDEN,
    parameter int K_LATENT  = lpu_config_pkg::LPU_K_LATENT,
    parameter int V_LATENT  = lpu_config_pkg::LPU_V_LATENT,
    parameter int NUM_SLOTS = lpu_config_pkg::LPU_KV_CACHE_SLOTS,
    parameter int MAX_POS   = lpu_config_pkg::LPU_MAX_SEQ_LEN,
    parameter int WEIGHT_W  = lpu_config_pkg::LPU_WEIGHT_WIDTH,
    parameter int DATA_W    = lpu_config_pkg::LPU_DATA_WIDTH,
    localparam int INTER_L  = lpu_config_pkg::LPU_INTERMEDIATE,
    localparam int NUM_EXPERTS = lpu_config_pkg::LPU_NUM_EXPERTS,
    localparam int TOP_K    = lpu_config_pkg::LPU_TOP_K,
    localparam int FFN_EXPERTS = lpu_config_pkg::LPU_EXPERTS_PER_FPGA,
    localparam int BEAT_W   = $clog2(((HIDDEN + 3) / 4) > 1 ? ((HIDDEN + 3) / 4) : 2)
) (
    input  logic clk, rst_n,

    // RMSNorm gamma (shared across all 3 RMSNorm instances)
    input  logic gamma_wr_en,
    input  logic [$clog2(HIDDEN)-1:0] gamma_wr_idx,
    input  logic signed [31:0] gamma_wr_data,

    // MLA Attention v2: QKV weight preload
    input  logic                         attn_qkv_wt_wr_en,
    input  logic [2:0]                   attn_qkv_wt_sel,
    input  logic [$clog2(HIDDEN)-1:0]    attn_qkv_wt_row,
    input  logic [$clog2(HIDDEN)-1:0]    attn_qkv_wt_col,
    input  logic signed [WEIGHT_W-1:0]   attn_qkv_wt_wr_data,

    // MLA Attention v2: RoPE LUT preload
    input  logic                         attn_rope_lut_wr_en,
    input  logic [$clog2(MAX_POS)-1:0]   attn_rope_lut_pos,
    input  logic [$clog2(HIDDEN/2)-1:0]  attn_rope_lut_pair,
    input  logic signed [WEIGHT_W-1:0]   attn_rope_lut_sin,
    input  logic signed [WEIGHT_W-1:0]   attn_rope_lut_cos,

    // MLA Attention v2: token position (for RoPE)
    input  logic [$clog2(MAX_POS)-1:0]   token_position,

    // Router preload
    input  logic rtr_w_wr_en,
    input  logic [$clog2(NUM_EXPERTS)-1:0] rtr_w_wr_expert,
    input  logic [$clog2(HIDDEN)-1:0]      rtr_w_wr_idx,
    input  logic signed [31:0] rtr_w_wr_data,

    // FFN preload
    input  logic gate_w_wr_en, up_w_wr_en, down_w_wr_en,
    input  logic [$clog2(INTER_L)-1:0] gate_w_wr_row, up_w_wr_row,
    input  logic [$clog2(HIDDEN)-1:0]  down_w_wr_row,
    input  logic [BEAT_W-1:0] gate_w_wr_beat, up_w_wr_beat, down_w_wr_beat,
    input  logic [15:0] gate_w_wr_data, up_w_wr_data, down_w_wr_data,
    input  logic [$clog2(FFN_EXPERTS > 1 ? FFN_EXPERTS : 2)-1:0] ffn_expert_sel,

    // Scale preload
    input  logic scale_wr_en,
    input  logic [$clog2(lpu_config_pkg::LPU_SCALE_GROUPS)-1:0] scale_wr_addr,
    input  logic [7:0] scale_wr_data,

    // KV cache preload (CPU prefill → DMA → FPGA HBM → cache)
    input  logic                         cache_preload_en,
    input  logic [K_LATENT*DATA_W-1:0]   cache_preload_K_flat,
    input  logic [V_LATENT*DATA_W-1:0]   cache_preload_V_flat,

    // Local expert bitmap: 1 if expert is on this chip
    input  logic [lpu_config_pkg::LPU_NUM_EXPERTS-1:0] cfg_local_experts,

    // Activation I/O
    input  logic valid_in,
    input  logic [HIDDEN*32-1:0] a_flat,
    output logic valid_out, router_ok,
    output logic [HIDDEN*32-1:0] y_flat
);

    typedef enum logic [3:0] {
        S_IDLE, S_R1, S_ATTN, S_R2, S_RTR,
        S_MOE_SOFTMAX, S_FFN_LD, S_FFN, S_MOE_ACCUM,
        S_R3, S_OUT
    } st_t;
    st_t st;

    // FFN beat counter
    logic [$clog2(FFN_BEATS)-1:0] ffn_beat_cnt;

    // RMS flat ports
    logic r1_vi,r1_vo, r2_vi,r2_vo, r3_vi,r3_vo;
    logic [HIDDEN*32-1:0] r1_x_flat, r1_g_flat, r1_y_flat;
    logic [HIDDEN*32-1:0] r2_x_flat, r2_g_flat, r2_y_flat;
    logic [HIDDEN*32-1:0] r3_x_flat, r3_g_flat, r3_y_flat;

    // MLA Attention v2
    logic                         attn_in_valid, attn_in_ready;
    logic [HIDDEN*DATA_W-1:0]     attn_hidden_flat;
    logic                         attn_out_valid, attn_out_ready;
    logic [HIDDEN*DATA_W-1:0]     attn_out_flat;

    // Router
    logic rtr_vi, rtr_vo, rtr_ok;
    localparam int EXP_BITS = $clog2(NUM_EXPERTS > 1 ? NUM_EXPERTS : 2);
    localparam int K_BITS   = $clog2(TOP_K > 1 ? TOP_K : 2);
    // Iverilog workaround: connect unpacked array ports through wire intermediates
    // to avoid port connection bug where logic unpacked arrays are not driven.
    wire [EXP_BITS-1:0] rtr_top_idx [TOP_K];
    wire signed [31:0]  rtr_top_score [TOP_K];

    // MoE dispatch
    logic [K_BITS-1:0]     moe_idx;
    logic signed [31:0]    moe_softmax_w [TOP_K];
    logic signed [31:0]    moe_y_acc [HIDDEN];

    // FFN
    logic ffn_aen, ffn_start, ffn_done, ffn_rv;
    logic [BEAT_W-1:0] ffn_abeat;
    logic [31:0] ffn_adata;
    logic [$clog2(HIDDEN)-1:0] ffn_rrow;
    logic [31:0] ffn_rdata;
    logic signed [31:0] ffo [HIDDEN-1:0];

    // Q12→FP8 for FFN activation
    logic [7:0] r2_f8 [HIDDEN-1:0];

    // FFN activation beat count: HIDDEN fp8 elements, LANES=4 per beat
    localparam int FFN_BEATS = (HIDDEN + 3) / 4;
    localparam int FFN_EXP_W = $clog2(FFN_EXPERTS > 1 ? FFN_EXPERTS : 2);
    logic [FFN_EXP_W-1:0] ffn_expert_sel_int;

    //=========================================================================
    // MoE softmax: combinational over router top-K scores
    //=========================================================================
    function [31:0] moe_exp_lut(input [31:0] adj);
        if (adj > -32'sd256)       moe_exp_lut = 4096;
        else if (adj > -32'sd1024)  moe_exp_lut = 3545;
        else if (adj > -32'sd2048)  moe_exp_lut = 2588;
        else if (adj > -32'sd4096)  moe_exp_lut = 1507;
        else if (adj > -32'sd8192)  moe_exp_lut = 538;
        else                        moe_exp_lut = 48;
    endfunction

    // Combinational softmax datapath (TOP_K small: 2 bring-up, 6 production)
    logic signed [31:0] moe_sw_score_q12 [TOP_K];
    logic signed [31:0] moe_sw_max;
    logic signed [31:0] moe_sw_exp [TOP_K];
    logic signed [31:0] moe_sw_sum_chain [TOP_K:0];
    logic signed [31:0] moe_sw_sum;
    logic signed [31:0] moe_sw_weight [TOP_K];

    // Convert Q24 scores → Q12, find max, compute exp values
    always_comb begin
        for (int k = 0; k < TOP_K; k++)
            moe_sw_score_q12[k] = rtr_top_score[k] >>> 12;
        moe_sw_max = moe_sw_score_q12[0];
        for (int k = 1; k < TOP_K; k++)
            if (moe_sw_score_q12[k] > moe_sw_max)
                moe_sw_max = moe_sw_score_q12[k];
        for (int k = 0; k < TOP_K; k++)
            moe_sw_exp[k] = moe_exp_lut(moe_sw_score_q12[k] - moe_sw_max);
    end

    // Adder chain for exp sum (parameterized, no always_comb feedback)
    assign moe_sw_sum_chain[0] = 0;
    for (genvar gk = 0; gk < TOP_K; gk++) begin : gen_sw_chain
        assign moe_sw_sum_chain[gk+1] = moe_sw_sum_chain[gk] + moe_sw_exp[gk];
    end
    assign moe_sw_sum = moe_sw_sum_chain[TOP_K];

    // Normalize to Q12 softmax weights
    for (genvar gk = 0; gk < TOP_K; gk++) begin : gen_sw_weight
        assign moe_sw_weight[gk] = (moe_sw_sum == 0)
            ? (4096 / TOP_K)
            : ((moe_sw_exp[gk] * 4096) / moe_sw_sum);
    end

    //=========================================================================
    // Sub-module instantiations
    //=========================================================================

    rms_norm u_r1(.clk,.rst_n,.valid_in(r1_vi),
        .x_flat(r1_x_flat), .g_flat(r1_g_flat),
        .valid_out(r1_vo), .y_flat(r1_y_flat));

    mla_attention_v2 #(
        .HIDDEN(HIDDEN), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
        .NUM_SLOTS(NUM_SLOTS), .MAX_POS(MAX_POS),
        .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W)
    ) u_attn (
        .clk, .rst_n,
        .in_valid(attn_in_valid),
        .hidden_flat(attn_hidden_flat),
        .in_ready(attn_in_ready),
        .position(token_position),
        .window_mode(1'b0),  // full attention by default (window mode via future config)
        .qkv_wt_wr_en(attn_qkv_wt_wr_en),
        .qkv_wt_sel(attn_qkv_wt_sel),
        .qkv_wt_row(attn_qkv_wt_row),
        .qkv_wt_col(attn_qkv_wt_col),
        .qkv_wt_wr_data(attn_qkv_wt_wr_data),
        .rope_lut_wr_en(attn_rope_lut_wr_en),
        .rope_lut_pos(attn_rope_lut_pos),
        .rope_lut_pair(attn_rope_lut_pair),
        .rope_lut_sin(attn_rope_lut_sin),
        .rope_lut_cos(attn_rope_lut_cos),
        .cache_preload_en(cache_preload_en),
        .cache_preload_K_flat(cache_preload_K_flat),
        .cache_preload_V_flat(cache_preload_V_flat),
        .out_valid(attn_out_valid),
        .out_ready(attn_out_ready),
        .y_flat(attn_out_flat)
    );

    rms_norm u_r2(.clk,.rst_n,.valid_in(r2_vi),
        .x_flat(r2_x_flat), .g_flat(r2_g_flat),
        .valid_out(r2_vo), .y_flat(r2_y_flat));

    router_topk u_rtr(.clk,.rst_n,.w_wr_en(rtr_w_wr_en),
        .w_wr_expert(rtr_w_wr_expert),.w_wr_idx(rtr_w_wr_idx),.w_wr_data(rtr_w_wr_data),
        .valid_in(rtr_vi), .a_flat(r2_y_flat),
        .valid_out(rtr_vo),.result_ready(1'b1),
        .top_idx(rtr_top_idx),.top_score(rtr_top_score));

    // FFN expert_sel: top-level port during preload, internal FSM during compute
    logic [FFN_EXP_W-1:0] ffn_expert_sel_engine;
    assign ffn_expert_sel_engine = (st == S_IDLE) ? ffn_expert_sel : ffn_expert_sel_int;

    expert_ffn_engine_fp4_down #(.HIDDEN(HIDDEN),.INTER(lpu_config_pkg::LPU_INTERMEDIATE),.GROUP_SIZE(HIDDEN/lpu_config_pkg::LPU_SCALE_GROUPS),.NUM_EXPERTS(FFN_EXPERTS)) u_ffn(.clk,.rst_n,
        .expert_sel(ffn_expert_sel_engine),
        .activ_wr_en(ffn_aen),.activ_wr_beat(ffn_abeat),.activ_wr_data(ffn_adata),
        .scale_wr_en,.scale_wr_addr,.scale_wr_data,
        .gate_w_wr_en,.gate_w_wr_row,.gate_w_wr_beat,.gate_w_wr_data,
        .up_w_wr_en,.up_w_wr_row,.up_w_wr_beat,.up_w_wr_data,
        .down_w_wr_en,.down_w_wr_row,.down_w_wr_beat,.down_w_wr_data,
        .start(ffn_start),.busy(),.done(ffn_done),
        .result_valid(ffn_rv),.result_row(ffn_rrow),.result_data(ffn_rdata));

    rms_norm u_r3(.clk,.rst_n,.valid_in(r3_vi),
        .x_flat(r3_x_flat), .g_flat(r3_g_flat),
        .valid_out(r3_vo), .y_flat(r3_y_flat));

    //=========================================================================
    // Wiring
    //=========================================================================

    assign router_ok = rtr_ok;
    assign attn_out_ready = 1'b1;
    assign attn_hidden_flat = r1_y_flat;

    // Q12→FP8 encoders for RMSNorm2 output → FFN activation
    for (genvar gi = 0; gi < HIDDEN; gi++) begin : gen_q12_fp8
        q12_to_fp8_e4m3 e(.x_q12(r2_y_flat[gi*32+:32]), .fp8(r2_f8[gi]));
    end

    // Gamma preload (shared across all 3 RMSNorm instances)
    always_ff @(posedge clk) begin
        if (gamma_wr_en) begin
            r1_g_flat[gamma_wr_idx*32+:32] <= gamma_wr_data;
            r2_g_flat[gamma_wr_idx*32+:32] <= gamma_wr_data;
            r3_g_flat[gamma_wr_idx*32+:32] <= gamma_wr_data;
        end
        if (ffn_rv) ffo[ffn_rrow] <= ffn_rdata;
    end

    //=========================================================================
    // Main FSM — MoE multi-expert dispatch loop
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=S_IDLE; r1_vi<=0;r2_vi<=0;r3_vi<=0; attn_in_valid<=0; rtr_vi<=0;
            ffn_aen<=0; ffn_start<=0; valid_out<=0; ffn_beat_cnt<='0;
            moe_idx <= '0;
            for (int k = 0; k < TOP_K; k++) moe_softmax_w[k] <= '0;
            for (int d = 0; d < HIDDEN; d++) moe_y_acc[d] <= '0;
            ffn_expert_sel_int <= '0;
            y_flat <= '0;
        end else begin
            r1_vi<=0;r2_vi<=0;r3_vi<=0; attn_in_valid<=0; rtr_vi<=0;
            ffn_aen<=0; ffn_start<=0; valid_out<=0;
            case (st)
                S_IDLE: if (valid_in) begin
                    r1_x_flat <= a_flat;
                    r1_vi<=1; st<=S_R1;
                end
                S_R1: if (r1_vo) begin
                    attn_in_valid<=1; st<=S_ATTN;
                end
                S_ATTN: if (attn_out_valid) begin
                    r2_x_flat <= attn_out_flat;
                    r2_vi<=1; st<=S_R2;
                end
                S_R2: if (r2_vo) begin rtr_vi<=1; st<=S_RTR; end

                // ================================================================
                // Router done: compute softmax, init MoE loop, load first FFN beat
                // ================================================================
                S_RTR: if (rtr_vo) begin
                    rtr_ok <= cfg_local_experts[rtr_top_idx[0]];
                    // Latch combinational softmax weights, init accumulator
                    for (int k = 0; k < TOP_K; k++)
                        moe_softmax_w[k] <= moe_sw_weight[k];
                    moe_idx <= 0;
                    for (int d = 0; d < HIDDEN; d++)
                        moe_y_acc[d] <= 0;
                    // Start FFN activation load (first beat)
                    ffn_expert_sel_int <= rtr_top_idx[0];
                    ffn_beat_cnt <= '0;
                    ffn_aen <= 1; ffn_abeat <= '0;
                    ffn_adata <= {r2_f8[3],r2_f8[2],r2_f8[1],r2_f8[0]};
                    st <= S_FFN_LD;
                end

                // ================================================================
                // FFN activation load (remaining beats)
                // ================================================================
                S_FFN_LD: begin
                    ffn_aen <= 1; ffn_abeat <= ffn_beat_cnt + 1'b1;
                    ffn_adata <= {r2_f8[(ffn_beat_cnt+1)*4+3],
                                  r2_f8[(ffn_beat_cnt+1)*4+2],
                                  r2_f8[(ffn_beat_cnt+1)*4+1],
                                  r2_f8[(ffn_beat_cnt+1)*4+0]};
                    if (ffn_beat_cnt + 1 >= FFN_BEATS - 1) begin
                        ffn_start <= 1; st <= S_FFN;
                    end else begin
                        ffn_beat_cnt <= ffn_beat_cnt + 1'b1;
                    end
                end

                // ================================================================
                // Wait for FFN to complete, then accumulate weighted output
                // ================================================================
                S_FFN: if (ffn_done) begin
                    st <= S_MOE_ACCUM;
                end

                // ================================================================
                // MoE accumulation: weighted sum, then next expert or done
                // ================================================================
                S_MOE_ACCUM: begin
                    if (moe_idx < TOP_K - 1) begin
                        // Intermediate expert: accumulate and chain to next
                        for (int d = 0; d < HIDDEN; d++)
                            moe_y_acc[d] <= moe_y_acc[d]
                                + ((moe_softmax_w[moe_idx] * ffo[d]) >>> 12);
                        moe_idx <= moe_idx + 1;
                        ffn_expert_sel_int <= rtr_top_idx[moe_idx + 1];
                        ffn_start <= 1;
                        st <= S_FFN;
                    end else begin
                        // Last expert: write final accumulated output to RMS3
                        for (int i = 0; i < HIDDEN; i++)
                            r3_x_flat[i*32+:32] <= moe_y_acc[i]
                                + ((moe_softmax_w[moe_idx] * ffo[i]) >>> 12);
                        r3_vi <= 1; st <= S_R3;
                    end
                end

                S_R3: if (r3_vo) begin
                    y_flat <= r3_y_flat;
                    valid_out<=1; st<=S_OUT;
                end
                S_OUT: st<=S_IDLE;
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule
