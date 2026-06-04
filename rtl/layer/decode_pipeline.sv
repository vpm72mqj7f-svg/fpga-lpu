//=============================================================================
// decode_pipeline.sv — Batch accumulator + full transformer layer pipeline
//
// Wraps token_batch_accumulator around full_transformer_layer.
// Tokens are accumulated until BATCH_MIN=6 or 50ms timeout, then fed
// sequentially through the pipeline. Expert weights cached in FFN engine
// syncram are naturally reused across tokens in the same batch.
//
// Roofline: B>=6 changes OI from 2.8 to 14.9 MACs/byte (bandwidth→compute bound).
//=============================================================================

`include "lpu_config.svh"

module decode_pipeline #(
    parameter int HIDDEN    = lpu_config_pkg::LPU_HIDDEN,
    parameter int K_LATENT  = lpu_config_pkg::LPU_K_LATENT,
    parameter int V_LATENT  = lpu_config_pkg::LPU_V_LATENT,
    parameter int NUM_SLOTS = lpu_config_pkg::LPU_KV_CACHE_SLOTS,
    parameter int MAX_POS   = lpu_config_pkg::LPU_MAX_SEQ_LEN,
    parameter int WEIGHT_W  = lpu_config_pkg::LPU_WEIGHT_WIDTH,
    parameter int DATA_W    = lpu_config_pkg::LPU_DATA_WIDTH,
    parameter int MAX_BATCH = 32,
    parameter int BATCH_MIN = 6
) (
    input  logic clk, rst_n,

    // ---- Accumulator input (token stream from scheduler) ----
    input  logic                         token_valid,
    input  logic [HIDDEN*DATA_W-1:0]     token_data,
    output logic                         token_ready,

    // ---- Pipeline output (decoded tokens) ----
    output logic                         result_valid,
    output logic [HIDDEN*DATA_W-1:0]     result_data,
    input  logic                         result_ready,

    // ---- Batch status (observability) ----
    output logic                         batch_active,
    output logic [$clog2(MAX_BATCH):0]   batch_size,

    // ---- Pipeline control passthrough ----
    input  logic gamma_wr_en,
    input  logic [$clog2(HIDDEN)-1:0] gamma_wr_idx,
    input  logic signed [31:0] gamma_wr_data,

    input  logic attn_qkv_wt_wr_en,
    input  logic [2:0] attn_qkv_wt_sel,
    input  logic [$clog2(HIDDEN)-1:0] attn_qkv_wt_row, attn_qkv_wt_col,
    input  logic signed [WEIGHT_W-1:0] attn_qkv_wt_wr_data,

    input  logic attn_rope_lut_wr_en,
    input  logic [$clog2(MAX_POS)-1:0] attn_rope_lut_pos,
    input  logic [$clog2(HIDDEN/2)-1:0] attn_rope_lut_pair,
    input  logic signed [WEIGHT_W-1:0] attn_rope_lut_sin, attn_rope_lut_cos,

    input  logic [$clog2(MAX_POS)-1:0] token_position,
    input  logic rtr_w_wr_en,
    input  logic [$clog2(lpu_config_pkg::LPU_NUM_EXPERTS)-1:0] rtr_w_wr_expert,
    input  logic [$clog2(HIDDEN)-1:0] rtr_w_wr_idx,
    input  logic signed [31:0] rtr_w_wr_data,

    input  logic gate_w_wr_en, up_w_wr_en, down_w_wr_en,
    input  logic [$clog2(lpu_config_pkg::LPU_INTERMEDIATE)-1:0] gate_w_wr_row, up_w_wr_row,
    input  logic [$clog2(HIDDEN)-1:0] down_w_wr_row,
    input  logic [$clog2(((HIDDEN+3)/4)>1?((HIDDEN+3)/4):2)-1:0] gate_w_wr_beat, up_w_wr_beat, down_w_wr_beat,
    input  logic [15:0] gate_w_wr_data, up_w_wr_data, down_w_wr_data,
    input  logic [$clog2(lpu_config_pkg::LPU_EXPERTS_PER_FPGA>1?lpu_config_pkg::LPU_EXPERTS_PER_FPGA:2)-1:0] ffn_expert_sel,

    input  logic scale_wr_en,
    input  logic [$clog2(lpu_config_pkg::LPU_SCALE_GROUPS)-1:0] scale_wr_addr,
    input  logic [7:0] scale_wr_data,

    input  logic cache_preload_en,
    input  logic [K_LATENT*DATA_W-1:0] cache_preload_K_flat,
    input  logic [V_LATENT*DATA_W-1:0] cache_preload_V_flat,

    input  logic [lpu_config_pkg::LPU_NUM_EXPERTS-1:0] cfg_local_experts
);

    // ---- Batch accumulator signals ----
    logic                         acc_valid_out;
    logic [HIDDEN*DATA_W-1:0]     acc_data_out;
    logic                         acc_out_ready;
    logic                         acc_batch_active;
    logic [$clog2(MAX_BATCH):0]   acc_batch_size;
    logic                         acc_batch_first;
    logic                         acc_batch_last;

    // Control: when pipeline is busy, backpressure the accumulator
    logic pipeline_busy;

    token_batch_accumulator #(
        .MAX_BATCH(MAX_BATCH), .BATCH_MIN(BATCH_MIN),
        .DATA_W(HIDDEN*DATA_W)
    ) u_accumulator (
        .clk, .rst_n,
        .valid_in(token_valid), .data_in(token_data),
        .in_ready(token_ready),
        .valid_out(acc_valid_out), .data_out(acc_data_out),
        .out_ready(acc_out_ready),
        .batch_active(acc_batch_active),
        .batch_size(acc_batch_size),
        .batch_first(acc_batch_first),
        .batch_last(acc_batch_last)
    );

    // ---- Full transformer layer ----
    logic pipeline_valid_in;
    logic pipeline_valid_out;
    logic pipeline_router_ok;
    logic [HIDDEN*DATA_W-1:0] pipeline_y_flat;

    // Feed accumulator output → pipeline input
    assign pipeline_valid_in = acc_valid_out && !pipeline_busy;
    assign acc_out_ready     = !pipeline_busy;
    assign pipeline_busy      = 1'b0;  // pipeline is always ready (no internal backpressure in bring-up)

    full_transformer_layer #(
        .HIDDEN(HIDDEN), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
        .NUM_SLOTS(NUM_SLOTS), .MAX_POS(MAX_POS), .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W)
    ) u_pipeline (
        .clk, .rst_n,
        .gamma_wr_en, .gamma_wr_idx, .gamma_wr_data,
        .attn_qkv_wt_wr_en, .attn_qkv_wt_sel,
        .attn_qkv_wt_row, .attn_qkv_wt_col, .attn_qkv_wt_wr_data,
        .attn_rope_lut_wr_en, .attn_rope_lut_pos,
        .attn_rope_lut_pair, .attn_rope_lut_sin, .attn_rope_lut_cos,
        .token_position,
        .rtr_w_wr_en, .rtr_w_wr_expert, .rtr_w_wr_idx, .rtr_w_wr_data,
        .gate_w_wr_en, .gate_w_wr_row, .gate_w_wr_beat, .gate_w_wr_data,
        .up_w_wr_en, .up_w_wr_row, .up_w_wr_beat, .up_w_wr_data,
        .down_w_wr_en, .down_w_wr_row, .down_w_wr_beat, .down_w_wr_data,
        .ffn_expert_sel,
        .scale_wr_en, .scale_wr_addr, .scale_wr_data,
        .cache_preload_en, .cache_preload_K_flat, .cache_preload_V_flat,
        .cfg_local_experts,
        .valid_in(pipeline_valid_in),
        .a_flat(acc_data_out),
        .valid_out(pipeline_valid_out),
        .router_ok(pipeline_router_ok),
        .y_flat(pipeline_y_flat)
    );

    // Pipeline output
    assign result_valid = pipeline_valid_out;
    assign result_data  = pipeline_y_flat;
    assign batch_active = acc_batch_active;
    assign batch_size   = acc_batch_size;

endmodule
