//=============================================================================
// layer_compute_engine.sv — RMSNorm → Router → ExpertFFN → RMSNorm (flat ports)
//=============================================================================

`include "lpu_config.svh"

module layer_compute_engine #(
    parameter int HIDDEN = lpu_config_pkg::LPU_HIDDEN,
    parameter int INTER  = lpu_config_pkg::LPU_INTERMEDIATE,
    localparam int LCE_BEAT_W   = $clog2(((HIDDEN + 3) / 4) > 1 ? ((HIDDEN + 3) / 4) : 2),
    localparam int LCE_BEAT_W_I = $clog2(((INTER + 3) / 4) > 1 ? ((INTER + 3) / 4) : 2)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         gate_w_wr_en, up_w_wr_en, down_w_wr_en,
    input  logic [$clog2(INTER)-1:0]     gate_w_wr_row, up_w_wr_row,
    input  logic [$clog2(HIDDEN)-1:0]    down_w_wr_row,
    input  logic [LCE_BEAT_W-1:0]        gate_w_wr_beat, up_w_wr_beat,
    input  logic [LCE_BEAT_W_I-1:0]      down_w_wr_beat,
    input  logic [15:0]                  gate_w_wr_data, up_w_wr_data, down_w_wr_data,

    input  logic                         gamma_wr_en,
    input  logic [$clog2(HIDDEN)-1:0]    gamma_wr_idx,
    input  logic signed [31:0]           gamma_wr_data,

    input  logic                         scale_wr_en,
    input  logic [$clog2(lpu_config_pkg::LPU_SCALE_GROUPS)-1:0] scale_wr_addr,
    input  logic [7:0]                   scale_wr_data,

    // Router weight preload
    input  logic                         rtr_w_wr_en,
    input  logic [$clog2(lpu_config_pkg::LPU_NUM_EXPERTS)-1:0] rtr_w_wr_expert,
    input  logic [$clog2(HIDDEN)-1:0]    rtr_w_wr_idx,
    input  logic signed [31:0]           rtr_w_wr_data,

    input  logic                         valid_in,
    input  logic [HIDDEN*32-1:0]         a_flat,

    output logic                         valid_out,
    output logic                         router_ok,
    output logic [HIDDEN*32-1:0]         y_flat
);

    typedef enum logic [3:0] {
        S_IDLE, S_RMS1, S_FFN_LD, S_FFN_RUN, S_RMS2, S_OUTPUT
    } state_t;
    state_t state;

    // RMSNorm1 flat signals
    logic r1_vi, r1_vo;
    logic [HIDDEN*32-1:0] r1_x_flat, r1_g_flat, r1_y_flat;

    // FFN flat signals
    localparam int LCE_FFN_BEATS = (HIDDEN + 3) / 4;
    logic ffn_aen, ffn_go, ffn_start, ffn_done, ffn_rv;
    logic [LCE_BEAT_W-1:0] ffn_abeat;
    logic [31:0] ffn_adata;
    logic [$clog2(HIDDEN)-1:0] ffn_rrow;
    logic [31:0] ffn_rdata;
    logic [$clog2(LCE_FFN_BEATS)-1:0] ffn_beat_cnt;

    // RMSNorm2 flat signals
    logic r2_vi, r2_vo;
    logic [HIDDEN*32-1:0] r2_x_flat, r2_g_flat, r2_y_flat;

    // Captured FFN output
    logic signed [31:0] ffo [HIDDEN-1:0];
    logic rtr_vi, rtr_vo, rtr_ok;
    localparam int LCE_EXP_BITS = $clog2(lpu_config_pkg::LPU_NUM_EXPERTS > 1 ? lpu_config_pkg::LPU_NUM_EXPERTS : 2);
    localparam int LCE_TOP_K    = lpu_config_pkg::LPU_TOP_K;
    // Iverilog workaround: wire (not logic) unpacked arrays for port connection
    wire [LCE_EXP_BITS-1:0] rtr_top_idx [LCE_TOP_K];
    wire signed [31:0]      rtr_top_score [LCE_TOP_K];
    assign router_ok = rtr_ok;

    // Q12→FP8 encoders for RMSNorm1 output → FFN activation
    logic [7:0] r1_f8 [HIDDEN-1:0];
    for (genvar gi = 0; gi < HIDDEN; gi++) begin : gen_q12_fp8
        q12_to_fp8_e4m3 enc(.x_q12(r1_y_flat[gi*32+:32]), .fp8(r1_f8[gi]));
    end

    rms_norm u_r1 (.clk,.rst_n,.valid_in(r1_vi),
        .x_flat(r1_x_flat), .g_flat(r1_g_flat),
        .valid_out(r1_vo), .y_flat(r1_y_flat));

    expert_ffn_engine_fp4_down #(.HIDDEN(HIDDEN),.INTER(INTER),.GROUP_SIZE(HIDDEN/lpu_config_pkg::LPU_SCALE_GROUPS)) u_ffn (.clk,.rst_n,
        .activ_wr_en(ffn_aen),.activ_wr_beat(ffn_abeat),.activ_wr_data(ffn_adata),
        .scale_wr_en,.scale_wr_addr,.scale_wr_data,
        .gate_w_wr_en,.gate_w_wr_row,.gate_w_wr_beat,.gate_w_wr_data,
        .up_w_wr_en,.up_w_wr_row,.up_w_wr_beat,.up_w_wr_data,
        .down_w_wr_en,.down_w_wr_row,.down_w_wr_beat,.down_w_wr_data,
        .start(ffn_start),.busy(),.done(ffn_done),
        .result_valid(ffn_rv),.result_row(ffn_rrow),.result_data(ffn_rdata));

    router_topk u_router (.clk,.rst_n,.w_wr_en(rtr_w_wr_en),
        .w_wr_expert(rtr_w_wr_expert),.w_wr_idx(rtr_w_wr_idx),.w_wr_data(rtr_w_wr_data),
        .valid_in(rtr_vi),.a_flat(r1_y_flat),
        .valid_out(rtr_vo),.result_ready(1'b1),
        .top_idx(rtr_top_idx),.top_score(rtr_top_score));

    rms_norm u_r2 (.clk,.rst_n,.valid_in(r2_vi),
        .x_flat(r2_x_flat), .g_flat(r2_g_flat),
        .valid_out(r2_vo), .y_flat(r2_y_flat));

    // Gamma load
    always_ff @(posedge clk) begin
        if (gamma_wr_en) begin
            r1_g_flat[gamma_wr_idx*32+:32] <= gamma_wr_data;
            r2_g_flat[gamma_wr_idx*32+:32] <= gamma_wr_data;
        end
        if (ffn_rv) ffo[ffn_rrow] <= ffn_rdata;
        if (rtr_vo) rtr_ok <= (rtr_top_idx[0] == '0);
    end

    // FSM
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; r1_vi<=0; r2_vi<=0; ffn_aen<=0; ffn_start<=0; valid_out<=0;
            y_flat <= '0;
            ffn_abeat<='0; ffn_adata<='0; ffn_beat_cnt<='0;
            rtr_vi<=0; rtr_ok<=0;
        end else begin
            r1_vi<=0; r2_vi<=0; ffn_aen<=0; ffn_start<=0; rtr_vi<=0; valid_out<=0;
            case (state)
                S_IDLE: if (valid_in) begin
                    r1_x_flat <= a_flat;
                    r1_vi<=1; state<=S_RMS1;
                end

                S_RMS1: if (r1_vo) begin
                    ffn_beat_cnt <= '0;
                    ffn_aen<=1; ffn_abeat<='0;
                    ffn_adata<={r1_f8[3],r1_f8[2],r1_f8[1],r1_f8[0]};
                    state<=S_FFN_LD;
                end

                S_FFN_LD: begin
                    ffn_aen<=1; ffn_abeat<=ffn_beat_cnt+1'b1;
                    ffn_adata<={r1_f8[(ffn_beat_cnt+1)*4+3],
                                r1_f8[(ffn_beat_cnt+1)*4+2],
                                r1_f8[(ffn_beat_cnt+1)*4+1],
                                r1_f8[(ffn_beat_cnt+1)*4+0]};
                    if (ffn_beat_cnt + 1 >= LCE_FFN_BEATS - 1) begin
                        ffn_start<=1; rtr_vi<=1; state<=S_FFN_RUN;
                    end else begin
                        ffn_beat_cnt<=ffn_beat_cnt+1'b1;
                    end
                end

                S_FFN_RUN: if (ffn_done) begin
                    for (int i = 0; i < HIDDEN; i++)
                        r2_x_flat[i*32+:32] <= ffo[i];
                    r2_vi<=1; state<=S_RMS2;
                end

                S_RMS2: if (r2_vo) begin
                    y_flat <= r2_y_flat;
                    valid_out<=1; state<=S_OUTPUT;
                end

                S_OUTPUT: state<=S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
