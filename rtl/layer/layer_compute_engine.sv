//=============================================================================
// layer_compute_engine.sv — RMSNorm → ExpertFFN → RMSNorm (flat ports)
//=============================================================================

module layer_compute_engine (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         gate_w_wr_en, up_w_wr_en, down_w_wr_en,
    input  logic [1:0]                   gate_w_wr_row, up_w_wr_row,
    input  logic [2:0]                   down_w_wr_row,
    input  logic [0:0]                   gate_w_wr_beat, up_w_wr_beat, down_w_wr_beat,
    input  logic [15:0]                  gate_w_wr_data, up_w_wr_data, down_w_wr_data,

    input  logic                         gamma_wr_en,
    input  logic [2:0]                   gamma_wr_idx,
    input  logic signed [31:0]           gamma_wr_data,

    input  logic                         scale_wr_en,
    input  logic [1:0]                   scale_wr_addr,
    input  logic [7:0]                   scale_wr_data,

    // Router weight preload
    input  logic                         rtr_w_wr_en,
    input  logic [1:0]                   rtr_w_wr_expert,
    input  logic [2:0]                   rtr_w_wr_idx,
    input  logic signed [31:0]           rtr_w_wr_data,

    input  logic                         valid_in,
    input  logic signed [31:0]           a0, a1, a2, a3, a4, a5, a6, a7,

    output logic                         valid_out,
    output logic                         router_ok,
    output logic signed [31:0]           y0, y1, y2, y3, y4, y5, y6, y7
);

    typedef enum logic [3:0] {
        S_IDLE, S_RMS1, S_LD1, S_LD2, S_FFN_RUN, S_RMS2, S_OUTPUT
    } state_t;
    state_t state;

    // RMSNorm1 flat signals
    logic r1_vi, r1_vo;
    logic signed [31:0] r1_x0,r1_x1,r1_x2,r1_x3,r1_x4,r1_x5,r1_x6,r1_x7;
    logic signed [31:0] r1_g0,r1_g1,r1_g2,r1_g3,r1_g4,r1_g5,r1_g6,r1_g7;
    logic signed [31:0] r1_y0,r1_y1,r1_y2,r1_y3,r1_y4,r1_y5,r1_y6,r1_y7;

    // FFN flat signals
    logic ffn_aen, ffn_go, ffn_start, ffn_done, ffn_rv;
    logic [0:0] ffn_abeat;
    logic [31:0] ffn_adata;
    logic [2:0] ffn_rrow;
    logic [31:0] ffn_rdata;

    // RMSNorm2 flat signals
    logic r2_vi, r2_vo;
    logic signed [31:0] r2_x0,r2_x1,r2_x2,r2_x3,r2_x4,r2_x5,r2_x6,r2_x7;
    logic signed [31:0] r2_g0,r2_g1,r2_g2,r2_g3,r2_g4,r2_g5,r2_g6,r2_g7;
    logic signed [31:0] r2_y0,r2_y1,r2_y2,r2_y3,r2_y4,r2_y5,r2_y6,r2_y7;

    // Captured FFN output
    logic signed [31:0] ffo [8];
    int row_cnt;
    logic rtr_vi, rtr_vo, rtr_ok;
    logic [1:0] rtr_top0, rtr_top1;
    logic signed [31:0] rtr_s0, rtr_s1;
    assign router_ok = rtr_ok;

    // Q12→FP8 encoders for RMSNorm1 output → FFN activation
    logic [7:0] r1_f8_0, r1_f8_1, r1_f8_2, r1_f8_3, r1_f8_4, r1_f8_5, r1_f8_6, r1_f8_7;
    q12_to_fp8_e4m3 enc0(.x_q12(r1_y0),.fp8(r1_f8_0));
    q12_to_fp8_e4m3 enc1(.x_q12(r1_y1),.fp8(r1_f8_1));
    q12_to_fp8_e4m3 enc2(.x_q12(r1_y2),.fp8(r1_f8_2));
    q12_to_fp8_e4m3 enc3(.x_q12(r1_y3),.fp8(r1_f8_3));
    q12_to_fp8_e4m3 enc4(.x_q12(r1_y4),.fp8(r1_f8_4));
    q12_to_fp8_e4m3 enc5(.x_q12(r1_y5),.fp8(r1_f8_5));
    q12_to_fp8_e4m3 enc6(.x_q12(r1_y6),.fp8(r1_f8_6));
    q12_to_fp8_e4m3 enc7(.x_q12(r1_y7),.fp8(r1_f8_7));

    rms_norm u_r1 (.clk,.rst_n,.valid_in(r1_vi),
        .x0(r1_x0),.x1(r1_x1),.x2(r1_x2),.x3(r1_x3),.x4(r1_x4),.x5(r1_x5),.x6(r1_x6),.x7(r1_x7),
        .g0(r1_g0),.g1(r1_g1),.g2(r1_g2),.g3(r1_g3),.g4(r1_g4),.g5(r1_g5),.g6(r1_g6),.g7(r1_g7),
        .valid_out(r1_vo), .y0(r1_y0),.y1(r1_y1),.y2(r1_y2),.y3(r1_y3),.y4(r1_y4),.y5(r1_y5),.y6(r1_y6),.y7(r1_y7));

    expert_ffn_engine_fp4_down #(.HIDDEN(8),.INTER(4)) u_ffn (.clk,.rst_n,
        .activ_wr_en(ffn_aen),.activ_wr_beat(ffn_abeat),.activ_wr_data(ffn_adata),
        .scale_wr_en,.scale_wr_addr,.scale_wr_data,
        .gate_w_wr_en,.gate_w_wr_row,.gate_w_wr_beat,.gate_w_wr_data,
        .up_w_wr_en,.up_w_wr_row,.up_w_wr_beat,.up_w_wr_data,
        .down_w_wr_en,.down_w_wr_row,.down_w_wr_beat,.down_w_wr_data,
        .start(ffn_start),.busy(),.done(ffn_done),
        .result_valid(ffn_rv),.result_row(ffn_rrow),.result_data(ffn_rdata));

    router_topk u_router (.clk,.rst_n,.w_wr_en(rtr_w_wr_en),
        .w_wr_expert(rtr_w_wr_expert),.w_wr_idx(rtr_w_wr_idx),.w_wr_data(rtr_w_wr_data),
        .valid_in(rtr_vi),.a0(r1_y0),.a1(r1_y1),.a2(r1_y2),.a3(r1_y3),
        .a4(r1_y4),.a5(r1_y5),.a6(r1_y6),.a7(r1_y7),
        .valid_out(rtr_vo),.result_ready(1'b1),
        .top0_idx(rtr_top0),.top1_idx(rtr_top1),.top0_score(rtr_s0),.top1_score(rtr_s1));

    rms_norm u_r2 (.clk,.rst_n,.valid_in(r2_vi),
        .x0(r2_x0),.x1(r2_x1),.x2(r2_x2),.x3(r2_x3),.x4(r2_x4),.x5(r2_x5),.x6(r2_x6),.x7(r2_x7),
        .g0(r2_g0),.g1(r2_g1),.g2(r2_g2),.g3(r2_g3),.g4(r2_g4),.g5(r2_g5),.g6(r2_g6),.g7(r2_g7),
        .valid_out(r2_vo), .y0(r2_y0),.y1(r2_y1),.y2(r2_y2),.y3(r2_y3),.y4(r2_y4),.y5(r2_y5),.y6(r2_y6),.y7(r2_y7));

    // Gamma load
    always_ff @(posedge clk) begin
        if (gamma_wr_en) begin
            case (gamma_wr_idx)
                0: begin r1_g0<=gamma_wr_data; r2_g0<=gamma_wr_data; end
                1: begin r1_g1<=gamma_wr_data; r2_g1<=gamma_wr_data; end
                2: begin r1_g2<=gamma_wr_data; r2_g2<=gamma_wr_data; end
                3: begin r1_g3<=gamma_wr_data; r2_g3<=gamma_wr_data; end
                4: begin r1_g4<=gamma_wr_data; r2_g4<=gamma_wr_data; end
                5: begin r1_g5<=gamma_wr_data; r2_g5<=gamma_wr_data; end
                6: begin r1_g6<=gamma_wr_data; r2_g6<=gamma_wr_data; end
                7: begin r1_g7<=gamma_wr_data; r2_g7<=gamma_wr_data; end
            endcase
        end
        if (ffn_rv) ffo[ffn_rrow] <= ffn_rdata;
        if (rtr_vo) rtr_ok <= (rtr_top0 == 2'd0);
    end

    // FSM
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; r1_vi<=0; r2_vi<=0; ffn_aen<=0; ffn_start<=0; valid_out<=0;
            {y0,y1,y2,y3,y4,y5,y6,y7} <= '0;
            ffn_abeat<=0; ffn_adata<='0; row_cnt<=0;
            rtr_vi<=0; rtr_ok<=0;
        end else begin
            r1_vi<=0; r2_vi<=0; ffn_aen<=0; ffn_start<=0; rtr_vi<=0; valid_out<=0;
            case (state)
                S_IDLE: if (valid_in) begin
                    r1_x0<=a0; r1_x1<=a1; r1_x2<=a2; r1_x3<=a3;
                    r1_x4<=a4; r1_x5<=a5; r1_x6<=a6; r1_x7<=a7;
                    r1_vi<=1; state<=S_RMS1;
                end

                S_RMS1: if (r1_vo) begin
                    ffn_aen<=1; ffn_abeat<=0;
                    ffn_adata<= {r1_f8_3, r1_f8_2, r1_f8_1, r1_f8_0};
                    state<=S_LD1;
                end

                S_LD1: begin
                    ffn_aen<=1; ffn_abeat<=1;
                    ffn_adata<= {r1_f8_7, r1_f8_6, r1_f8_5, r1_f8_4};
                    state<=S_LD2;
                end

                S_LD2: begin
                    ffn_start<=1; rtr_vi<=1; row_cnt<=0; state<=S_FFN_RUN;
                end

                S_FFN_RUN: if (ffn_done) begin
                    r2_x0<=ffo[0]; r2_x1<=ffo[1]; r2_x2<=ffo[2]; r2_x3<=ffo[3];
                    r2_x4<=ffo[4]; r2_x5<=ffo[5]; r2_x6<=ffo[6]; r2_x7<=ffo[7];
                    r2_vi<=1; state<=S_RMS2;
                end

                S_RMS2: if (r2_vo) begin
                    y0<=r2_y0; y1<=r2_y1; y2<=r2_y2; y3<=r2_y3;
                    y4<=r2_y4; y5<=r2_y5; y6<=r2_y6; y7<=r2_y7;
                    valid_out<=1; state<=S_OUTPUT;
                end

                S_OUTPUT: state<=S_IDLE;
            endcase
        end
    end

endmodule
