//=============================================================================
// expert_ffn_engine_fp4_down.sv — Expert FFN with gate/up/down fp4 linear engines
//=============================================================================

module expert_ffn_engine_fp4_down #(
    parameter int HIDDEN = 8,
    parameter int INTER  = 4,
    parameter int LANES  = 4,
    parameter int GROUP_SIZE = 2,
    parameter int K_BEATS_H = (HIDDEN + LANES - 1) / LANES,
    parameter int K_BEATS_I = (INTER + LANES - 1) / LANES,
    parameter int BEAT_W_H  = $clog2(K_BEATS_H > 1 ? K_BEATS_H : 2),
    parameter int BEAT_W_I  = $clog2(K_BEATS_I > 1 ? K_BEATS_I : 2),
    parameter int NUM_EXPERTS = 1,
    localparam int EXPERT_W = $clog2(NUM_EXPERTS > 1 ? NUM_EXPERTS : 2),
    localparam int NUM_GROUPS_H = (HIDDEN + GROUP_SIZE - 1) / GROUP_SIZE,
    localparam int NUM_GROUPS_I = (INTER  + GROUP_SIZE - 1) / GROUP_SIZE,
    localparam int MAX_NUM_GROUPS = NUM_GROUPS_H > NUM_GROUPS_I ? NUM_GROUPS_H : NUM_GROUPS_I,
    localparam int SCALE_ADDR_W = $clog2(MAX_NUM_GROUPS > 1 ? MAX_NUM_GROUPS : 2)
) (
    input  logic clk,
    input  logic rst_n,

    // Expert select (0 when NUM_EXPERTS=1)
    input  logic [EXPERT_W-1:0] expert_sel,

    input  logic activ_wr_en,
    input  logic [BEAT_W_H-1:0] activ_wr_beat,
    input  logic [LANES*8-1:0] activ_wr_data,

    input  logic scale_wr_en,
    input  logic [SCALE_ADDR_W-1:0] scale_wr_addr,
    input  logic [7:0] scale_wr_data,

    input  logic gate_w_wr_en,
    input  logic [$clog2(INTER)-1:0] gate_w_wr_row,
    input  logic [BEAT_W_H-1:0] gate_w_wr_beat,
    input  logic [LANES*4-1:0] gate_w_wr_data,

    input  logic up_w_wr_en,
    input  logic [$clog2(INTER)-1:0] up_w_wr_row,
    input  logic [BEAT_W_H-1:0] up_w_wr_beat,
    input  logic [LANES*4-1:0] up_w_wr_data,

    input  logic down_w_wr_en,
    input  logic [$clog2(HIDDEN)-1:0] down_w_wr_row,
    input  logic [BEAT_W_I-1:0] down_w_wr_beat,
    input  logic [LANES*4-1:0] down_w_wr_data,

    input  logic start,
    output logic busy,
    output logic done,
    output logic result_valid,
    output logic [$clog2(HIDDEN)-1:0] result_row,
    output logic [31:0] result_data
);

    typedef enum logic [2:0] {S_IDLE, S_RUN_GU, S_MID, S_LOAD_DOWN, S_RUN_DOWN, S_DONE} state_t;
    state_t state;

    logic gate_start, up_start, down_start;
    logic gate_done, up_done, down_done;
    logic gate_rv, up_rv, down_rv;
    logic [$clog2(INTER)-1:0] gate_rr, up_rr;
    logic [$clog2(HIDDEN)-1:0] down_rr;
    logic [31:0] gate_rd, up_rd, down_rd;

    logic signed [31:0] gate_vec [INTER];
    logic signed [31:0] up_vec [INTER];
    logic signed [31:0] silu_vec [INTER];
    logic signed [31:0] mid_vec [INTER];
    logic [INTER*8-1:0] down_activ_pack;
    logic down_activ_wr_en;
    logic down_started;
    logic [BEAT_W_I-1:0] down_beat_cnt;
    wire  [LANES*8-1:0] down_activ_slice;

    assign down_activ_slice = down_activ_pack[down_beat_cnt * LANES*8 +: LANES*8];

    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign result_valid = down_rv;
    assign result_row = down_rr;
    assign result_data = down_rd;

    genvar gi;
    generate
        for (gi = 0; gi < INTER; gi++) begin : g_silu
            silu_q12_lut u_silu (.clk(clk), .x_q12(gate_vec[gi]), .y_q12(silu_vec[gi]));
            q12_to_fp8_e4m3 u_mid_enc (.x_q12(mid_vec[gi]), .fp8(down_activ_pack[gi*8 +: 8]));
        end
    endgenerate

    fp4_linear_engine #(.M_OUT(INTER), .K_TOTAL(HIDDEN), .LANES(LANES), .GROUP_SIZE(GROUP_SIZE), .NUM_GROUPS(NUM_GROUPS_H), .ADDR_WIDTH(SCALE_ADDR_W), .NUM_EXPERTS(NUM_EXPERTS), .NAME("gate")) u_gate (
        .clk(clk), .rst_n(rst_n), .weight_wr_en(gate_w_wr_en), .weight_wr_row(gate_w_wr_row),
        .weight_wr_beat(gate_w_wr_beat), .weight_wr_data(gate_w_wr_data),
        .expert_sel(expert_sel),
        .activ_wr_en(activ_wr_en), .activ_wr_beat(activ_wr_beat), .activ_wr_data(activ_wr_data),
        .scale_wr_en(scale_wr_en), .scale_wr_addr(scale_wr_addr), .scale_wr_data(scale_wr_data),
        .start(gate_start), .busy(), .done(gate_done), .result_valid(gate_rv), .result_row(gate_rr), .result_data(gate_rd),
        .result_ready(1'b1)
    );

    fp4_linear_engine #(.M_OUT(INTER), .K_TOTAL(HIDDEN), .LANES(LANES), .GROUP_SIZE(GROUP_SIZE), .NUM_GROUPS(NUM_GROUPS_H), .ADDR_WIDTH(SCALE_ADDR_W), .NUM_EXPERTS(NUM_EXPERTS), .NAME("up")) u_up (
        .clk(clk), .rst_n(rst_n), .weight_wr_en(up_w_wr_en), .weight_wr_row(up_w_wr_row),
        .weight_wr_beat(up_w_wr_beat), .weight_wr_data(up_w_wr_data),
        .expert_sel(expert_sel),
        .activ_wr_en(activ_wr_en), .activ_wr_beat(activ_wr_beat), .activ_wr_data(activ_wr_data),
        .scale_wr_en(scale_wr_en), .scale_wr_addr(scale_wr_addr), .scale_wr_data(scale_wr_data),
        .start(up_start), .busy(), .done(up_done), .result_valid(up_rv), .result_row(up_rr), .result_data(up_rd),
        .result_ready(1'b1)
    );

    fp4_linear_engine #(.M_OUT(HIDDEN), .K_TOTAL(INTER), .LANES(LANES), .GROUP_SIZE(GROUP_SIZE), .NUM_GROUPS(NUM_GROUPS_I), .ADDR_WIDTH(SCALE_ADDR_W), .NUM_EXPERTS(NUM_EXPERTS), .NAME("down")) u_down (
        .clk(clk), .rst_n(rst_n), .weight_wr_en(down_w_wr_en), .weight_wr_row(down_w_wr_row),
        .weight_wr_beat(down_w_wr_beat), .weight_wr_data(down_w_wr_data),
        .expert_sel(expert_sel),
        .activ_wr_en(down_activ_wr_en), .activ_wr_beat(down_beat_cnt), .activ_wr_data(down_activ_slice),
        .scale_wr_en(scale_wr_en), .scale_wr_addr(scale_wr_addr), .scale_wr_data(scale_wr_data),
        .start(down_start), .busy(), .done(down_done), .result_valid(down_rv), .result_row(down_rr), .result_data(down_rd),
        .result_ready(1'b1)
    );

    // ── DSP: altera_mult_add for gate * up multiply (INTER elements) ──
    wire signed [63:0] gate_up_prod [INTER-1:0];

    for (genvar i = 0; i < INTER; i++) begin : gen_gate_up
        altera_mult_add #(.A_WIDTH(32), .B_WIDTH(32), .PIPE_STAGES(0))
        u_mul (.clock(clk),
            .a($signed(silu_vec[i])), .b($signed(up_vec[i])),
            .result(gate_up_prod[i]));
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; gate_start <= 0; up_start <= 0; down_start <= 0; done <= 0; down_activ_wr_en <= 0; down_started <= 0;
            down_beat_cnt <= '0;
            for (int i=0; i<INTER; i++) begin gate_vec[i] <= 0; up_vec[i] <= 0; mid_vec[i] <= 0; end
        end else begin
            gate_start <= 0; up_start <= 0; down_start <= 0; done <= 0; down_activ_wr_en <= 0;
            if (gate_rv) begin
                gate_vec[gate_rr] <= gate_rd;
`ifdef DBG_PIPELINE
                $display("  [FFN_DBG] gate_vec[%0d] <= 0x%08h (%0d)", gate_rr, gate_rd, gate_rd);
`endif
            end
            if (up_rv) begin
                up_vec[up_rr] <= up_rd;
`ifdef DBG_PIPELINE
                $display("  [FFN_DBG] up_vec[%0d]   <= 0x%08h (%0d)", up_rr, up_rd, up_rd);
`endif
            end
            case (state)
                S_IDLE: if (start) begin
                    gate_start <= 1; up_start <= 1; state <= S_RUN_GU;
                end
                S_RUN_GU: if (gate_done && up_done) state <= S_MID;
                S_MID: begin
                    for (int i=0; i<INTER; i++) mid_vec[i] <= gate_up_prod[i] >>> 12;
`ifdef DBG_PIPELINE
                    for (int i=0; i<INTER; i++)
                        $display("  [FFN_DBG] S_MID i=%0d gate=0x%08h(%0d) up=0x%08h(%0d) silu=0x%08h(%0d) prod=0x%016h(%0d) mid=0x%016h(%0d)",
                                 i, gate_vec[i], gate_vec[i], up_vec[i], up_vec[i],
                                 silu_vec[i], silu_vec[i], gate_up_prod[i], gate_up_prod[i],
                                 gate_up_prod[i] >>> 12, gate_up_prod[i] >>> 12);
`endif
                    state <= S_LOAD_DOWN;
                end
                S_LOAD_DOWN: begin
                    down_activ_wr_en <= 1;
`ifdef DBG_PIPELINE
                    $display("  [FFN_DBG] S_LOAD_DOWN beat=%0d K_BEATS_I=%0d activ_slice=0x%08h fp8[0]=0x%02x fp8[1]=0x%02x fp8[2]=0x%02x fp8[3]=0x%02x",
                             down_beat_cnt, K_BEATS_I, down_activ_slice,
                             down_activ_pack[7:0], down_activ_pack[15:8],
                             down_activ_pack[23:16], down_activ_pack[31:24]);
                    for (int i=0; i<INTER; i++)
                        $display("  [FFN_DBG]   mid[%0d]=0x%08h(%0d) fp8=0x%02x",
                                 i, mid_vec[i], mid_vec[i], down_activ_pack[i*8 +: 8]);
`endif
                    if (down_beat_cnt == K_BEATS_I - 1) begin
                        down_beat_cnt <= '0;
                        down_started <= 0;
                        state <= S_RUN_DOWN;
                    end else begin
                        down_beat_cnt <= down_beat_cnt + 1'b1;
                    end
                end
                S_RUN_DOWN: begin
                    if (!down_started) begin
                        down_start <= 1;
                        down_started <= 1;
                    end
                    if (down_done) begin
                        done <= 1; state <= S_DONE;
                    end
                end
                S_DONE: if (!start) state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
