//=============================================================================
// full_transformer_layer.sv — RMS→ATTN→RMS→Router→FFN→RMS (HIDDEN=8)
//=============================================================================

module full_transformer_layer (
    input  logic clk, rst_n,
    // RMSNorm gamma
    input  logic gamma_wr_en,
    input  logic [2:0] gamma_wr_idx,
    input  logic signed [31:0] gamma_wr_data,
    // Attention preload
    input  logic attn_score_wr_en, attn_v_wr_en,
    input  logic [5:0] attn_score_wr_idx, attn_v_wr_idx,
    input  logic signed [31:0] attn_score_wr_data, attn_v_wr_data,
    // Router preload
    input  logic rtr_w_wr_en,
    input  logic [1:0] rtr_w_wr_expert,
    input  logic [2:0] rtr_w_wr_idx,
    input  logic signed [31:0] rtr_w_wr_data,
    // FFN preload
    input  logic gate_w_wr_en, up_w_wr_en, down_w_wr_en,
    input  logic [1:0] gate_w_wr_row, up_w_wr_row,
    input  logic [2:0] down_w_wr_row,
    input  logic [0:0] gate_w_wr_beat, up_w_wr_beat, down_w_wr_beat,
    input  logic [15:0] gate_w_wr_data, up_w_wr_data, down_w_wr_data,
    // Scale preload
    input  logic scale_wr_en,
    input  logic [1:0] scale_wr_addr,
    input  logic [7:0] scale_wr_data,
    // Activation I/O
    input  logic valid_in,
    input  logic signed [31:0] a0,a1,a2,a3,a4,a5,a6,a7,
    output logic valid_out, router_ok,
    output logic signed [31:0] y0,y1,y2,y3,y4,y5,y6,y7
);
    typedef enum logic [3:0] {
        S_IDLE, S_R1, S_ATTN, S_R2, S_RTR, S_FFN_LD1, S_FFN_LD2,
        S_FFN, S_R3, S_OUT
    } st_t;
    st_t st;

    // RMS flash ports
    logic r1_vi,r1_vo, r2_vi,r2_vo, r3_vi,r3_vo;
    logic signed [31:0] r1x0,r1x1,r1x2,r1x3,r1x4,r1x5,r1x6,r1x7;
    logic signed [31:0] r1g0,r1g1,r1g2,r1g3,r1g4,r1g5,r1g6,r1g7;
    logic signed [31:0] r1y0,r1y1,r1y2,r1y3,r1y4,r1y5,r1y6,r1y7;
    logic signed [31:0] r2x0,r2x1,r2x2,r2x3,r2x4,r2x5,r2x6,r2x7;
    logic signed [31:0] r2g0,r2g1,r2g2,r2g3,r2g4,r2g5,r2g6,r2g7;
    logic signed [31:0] r2y0,r2y1,r2y2,r2y3,r2y4,r2y5,r2y6,r2y7;
    logic signed [31:0] r3x0,r3x1,r3x2,r3x3,r3x4,r3x5,r3x6,r3x7;
    logic signed [31:0] r3g0,r3g1,r3g2,r3g3,r3g4,r3g5,r3g6,r3g7;
    logic signed [31:0] r3y0,r3y1,r3y2,r3y3,r3y4,r3y5,r3y6,r3y7;
    // Attention
    logic attn_vi, attn_vo;
    logic signed [31:0] ay0,ay1,ay2,ay3,ay4,ay5,ay6,ay7;
    // Router
    logic rtr_vi, rtr_vo, rtr_ok;
    logic [1:0] rtr_t0, rtr_t1;
    logic signed [31:0] rtr_s0, rtr_s1;
    // FFN
    logic ffn_aen, ffn_start, ffn_done, ffn_rv;
    logic [0:0] ffn_abeat;
    logic [31:0] ffn_adata;
    logic [2:0] ffn_rrow;
    logic [31:0] ffn_rdata;
    logic signed [31:0] ffo [8];
    // Q12→FP8 for FFN activation
    logic [7:0] r2_f8 [8];

    rms_norm u_r1(.clk,.rst_n,.valid_in(r1_vi),
        .x0(r1x0),.x1(r1x1),.x2(r1x2),.x3(r1x3),.x4(r1x4),.x5(r1x5),.x6(r1x6),.x7(r1x7),
        .g0(r1g0),.g1(r1g1),.g2(r1g2),.g3(r1g3),.g4(r1g4),.g5(r1g5),.g6(r1g6),.g7(r1g7),
        .valid_out(r1_vo), .y0(r1y0),.y1(r1y1),.y2(r1y2),.y3(r1y3),.y4(r1y4),.y5(r1y5),.y6(r1y6),.y7(r1y7));
    mla_attention u_attn(.clk,.rst_n,.score_wr_en(attn_score_wr_en),.v_wr_en(attn_v_wr_en),
        .score_wr_idx(attn_score_wr_idx),.v_wr_idx(attn_v_wr_idx),
        .score_wr_data(attn_score_wr_data),.v_wr_data(attn_v_wr_data),
        .valid_in(attn_vi),.valid_out(attn_vo),
        .y0(ay0),.y1(ay1),.y2(ay2),.y3(ay3),.y4(ay4),.y5(ay5),.y6(ay6),.y7(ay7));
    rms_norm u_r2(.clk,.rst_n,.valid_in(r2_vi),
        .x0(r2x0),.x1(r2x1),.x2(r2x2),.x3(r2x3),.x4(r2x4),.x5(r2x5),.x6(r2x6),.x7(r2x7),
        .g0(r2g0),.g1(r2g1),.g2(r2g2),.g3(r2g3),.g4(r2g4),.g5(r2g5),.g6(r2g6),.g7(r2g7),
        .valid_out(r2_vo), .y0(r2y0),.y1(r2y1),.y2(r2y2),.y3(r2y3),.y4(r2y4),.y5(r2y5),.y6(r2y6),.y7(r2y7));
    router_topk u_rtr(.clk,.rst_n,.w_wr_en(rtr_w_wr_en),
        .w_wr_expert(rtr_w_wr_expert),.w_wr_idx(rtr_w_wr_idx),.w_wr_data(rtr_w_wr_data),
        .valid_in(rtr_vi), .a0(r2y0),.a1(r2y1),.a2(r2y2),.a3(r2y3),
        .a4(r2y4),.a5(r2y5),.a6(r2y6),.a7(r2y7),
        .valid_out(rtr_vo),.result_ready(1'b1),
        .top0_idx(rtr_t0),.top1_idx(rtr_t1),.top0_score(rtr_s0),.top1_score(rtr_s1));
    expert_ffn_engine_fp4_down #(.HIDDEN(8),.INTER(4)) u_ffn(.clk,.rst_n,
        .activ_wr_en(ffn_aen),.activ_wr_beat(ffn_abeat),.activ_wr_data(ffn_adata),
        .scale_wr_en,.scale_wr_addr,.scale_wr_data,
        .gate_w_wr_en,.gate_w_wr_row,.gate_w_wr_beat,.gate_w_wr_data,
        .up_w_wr_en,.up_w_wr_row,.up_w_wr_beat,.up_w_wr_data,
        .down_w_wr_en,.down_w_wr_row,.down_w_wr_beat,.down_w_wr_data,
        .start(ffn_start),.busy(),.done(ffn_done),
        .result_valid(ffn_rv),.result_row(ffn_rrow),.result_data(ffn_rdata));
    rms_norm u_r3(.clk,.rst_n,.valid_in(r3_vi),
        .x0(r3x0),.x1(r3x1),.x2(r3x2),.x3(r3x3),.x4(r3x4),.x5(r3x5),.x6(r3x6),.x7(r3x7),
        .g0(r3g0),.g1(r3g1),.g2(r3g2),.g3(r3g3),.g4(r3g4),.g5(r3g5),.g6(r3g6),.g7(r3g7),
        .valid_out(r3_vo), .y0(r3y0),.y1(r3y1),.y2(r3y2),.y3(r3y3),.y4(r3y4),.y5(r3y5),.y6(r3y6),.y7(r3y7));

    assign router_ok = rtr_ok;
    q12_to_fp8_e4m3 e0(.x_q12(r2y0),.fp8(r2_f8[0])); q12_to_fp8_e4m3 e1(.x_q12(r2y1),.fp8(r2_f8[1]));
    q12_to_fp8_e4m3 e2(.x_q12(r2y2),.fp8(r2_f8[2])); q12_to_fp8_e4m3 e3(.x_q12(r2y3),.fp8(r2_f8[3]));
    q12_to_fp8_e4m3 e4(.x_q12(r2y4),.fp8(r2_f8[4])); q12_to_fp8_e4m3 e5(.x_q12(r2y5),.fp8(r2_f8[5]));
    q12_to_fp8_e4m3 e6(.x_q12(r2y6),.fp8(r2_f8[6])); q12_to_fp8_e4m3 e7(.x_q12(r2y7),.fp8(r2_f8[7]));

    always_ff @(posedge clk) begin
        if (gamma_wr_en) case (gamma_wr_idx)
            0:begin r1g0<=gamma_wr_data;r2g0<=gamma_wr_data;r3g0<=gamma_wr_data;end
            1:begin r1g1<=gamma_wr_data;r2g1<=gamma_wr_data;r3g1<=gamma_wr_data;end
            2:begin r1g2<=gamma_wr_data;r2g2<=gamma_wr_data;r3g2<=gamma_wr_data;end
            3:begin r1g3<=gamma_wr_data;r2g3<=gamma_wr_data;r3g3<=gamma_wr_data;end
            4:begin r1g4<=gamma_wr_data;r2g4<=gamma_wr_data;r3g4<=gamma_wr_data;end
            5:begin r1g5<=gamma_wr_data;r2g5<=gamma_wr_data;r3g5<=gamma_wr_data;end
            6:begin r1g6<=gamma_wr_data;r2g6<=gamma_wr_data;r3g6<=gamma_wr_data;end
            7:begin r1g7<=gamma_wr_data;r2g7<=gamma_wr_data;r3g7<=gamma_wr_data;end
        endcase
        if (ffn_rv) ffo[ffn_rrow] <= ffn_rdata;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=S_IDLE; r1_vi<=0;r2_vi<=0;r3_vi<=0; attn_vi<=0; rtr_vi<=0;
            ffn_aen<=0; ffn_start<=0; valid_out<=0;
            {y0,y1,y2,y3,y4,y5,y6,y7}<='0;
        end else begin
            r1_vi<=0;r2_vi<=0;r3_vi<=0; attn_vi<=0; rtr_vi<=0;
            ffn_aen<=0; ffn_start<=0; valid_out<=0;
            case (st)
                S_IDLE: if (valid_in) begin
                    r1x0<=a0;r1x1<=a1;r1x2<=a2;r1x3<=a3;r1x4<=a4;r1x5<=a5;r1x6<=a6;r1x7<=a7;
                    r1_vi<=1; st<=S_R1;
                end
                S_R1: if (r1_vo) begin attn_vi<=1; st<=S_ATTN; end
                S_ATTN: if (attn_vo) begin
                    r2x0<=ay0;r2x1<=ay1;r2x2<=ay2;r2x3<=ay3;r2x4<=ay4;r2x5<=ay5;r2x6<=ay6;r2x7<=ay7;
                    r2_vi<=1; st<=S_R2;
                end
                S_R2: if (r2_vo) begin rtr_vi<=1; st<=S_RTR; end
                S_RTR: if (rtr_vo) begin rtr_ok<=(rtr_t0==2'd0);
                    ffn_aen<=1; ffn_abeat<=0; ffn_adata<={r2_f8[3],r2_f8[2],r2_f8[1],r2_f8[0]}; st<=S_FFN_LD1;
                end
                S_FFN_LD1: begin
                    ffn_aen<=1; ffn_abeat<=1; ffn_adata<={r2_f8[7],r2_f8[6],r2_f8[5],r2_f8[4]}; st<=S_FFN_LD2;
                end
                S_FFN_LD2: begin ffn_start<=1; st<=S_FFN; end
                S_FFN: if (ffn_done) begin
                    r3x0<=ffo[0];r3x1<=ffo[1];r3x2<=ffo[2];r3x3<=ffo[3];
                    r3x4<=ffo[4];r3x5<=ffo[5];r3x6<=ffo[6];r3x7<=ffo[7];
                    r3_vi<=1; st<=S_R3;
                end
                S_R3: if (r3_vo) begin
                    y0<=r3y0;y1<=r3y1;y2<=r3y2;y3<=r3y3;y4<=r3y4;y5<=r3y5;y6<=r3y6;y7<=r3y7;
                    valid_out<=1; st<=S_OUT;
                end
                S_OUT: st<=S_IDLE;
            endcase
        end
    end

endmodule
