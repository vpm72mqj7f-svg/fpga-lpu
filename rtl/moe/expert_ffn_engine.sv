//=============================================================================
// expert_ffn_engine.sv — tiny Expert FFN control prototype
//
// HIDDEN=8, INTER=4 bring-up module:
//   gate = fp4_linear(x)
//   up   = fp4_linear(x)
//   mid  = silu(gate) * up     (silu via Q12 piecewise LUT)
//   out  = down_q12 * mid      (down weights loaded as Q12 fixed point)
//=============================================================================

module expert_ffn_engine #(
    parameter int HIDDEN = 8,
    parameter int INTER  = 4,
    parameter int LANES  = 4,
    parameter int K_BEATS = (HIDDEN + LANES - 1) / LANES,
    parameter int ACCUM_WIDTH = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Shared activation preload
    input  logic                         activ_wr_en,
    input  logic [$clog2(K_BEATS)-1:0]   activ_wr_beat,
    input  logic [LANES*8-1:0]           activ_wr_data,

    // Shared scale preload (for gate/up)
    input  logic                         scale_wr_en,
    input  logic [1:0]                   scale_wr_addr,
    input  logic [7:0]                   scale_wr_data,

    // Gate/up fp4 weight preload
    input  logic                         gate_w_wr_en,
    input  logic [$clog2(INTER)-1:0]     gate_w_wr_row,
    input  logic [$clog2(K_BEATS)-1:0]   gate_w_wr_beat,
    input  logic [LANES*4-1:0]           gate_w_wr_data,

    input  logic                         up_w_wr_en,
    input  logic [$clog2(INTER)-1:0]     up_w_wr_row,
    input  logic [$clog2(K_BEATS)-1:0]   up_w_wr_beat,
    input  logic [LANES*4-1:0]           up_w_wr_data,

    // Down weights as signed Q12 fixed-point values
    input  logic                         down_w_wr_en,
    input  logic [$clog2(HIDDEN)-1:0]    down_w_wr_row,
    input  logic [$clog2(INTER)-1:0]     down_w_wr_col,
    input  logic signed [31:0]           down_w_wr_data,

    input  logic                         start,
    output logic                         busy,
    output logic                         done,
    output logic                         result_valid,
    output logic [$clog2(HIDDEN)-1:0]    result_row,
    output logic signed [31:0]           result_data
);

    typedef enum logic [2:0] {
        S_IDLE,
        S_RUN_LINEAR,
        S_COMPUTE_MID,
        S_OUTPUT,
        S_DONE
    } state_t;

    state_t state;
    logic gate_start, up_start;
    logic gate_done, up_done;
    logic gate_result_valid, up_result_valid;
    logic [$clog2(INTER)-1:0] gate_result_row, up_result_row;
    logic [31:0] gate_result_data, up_result_data;

    logic signed [31:0] gate_vec [INTER];
    logic signed [31:0] up_vec   [INTER];
    logic signed [31:0] silu_vec [INTER];
    logic signed [31:0] mid_vec  [INTER];
    logic signed [31:0] down_w   [HIDDEN*INTER];
    logic [$clog2(HIDDEN)-1:0] out_row;

    genvar gi;
    generate
        for (gi = 0; gi < INTER; gi++) begin : g_silu
            silu_q12_lut u_silu (
                .x_q12(gate_vec[gi]),
                .y_q12(silu_vec[gi])
            );
        end
    endgenerate

    assign busy = (state != S_IDLE) && (state != S_DONE);

    fp4_linear_engine #(
        .M_OUT(INTER), .K_TOTAL(HIDDEN), .LANES(LANES),
        .GROUP_SIZE(4), .NUM_GROUPS(4), .ADDR_WIDTH(2)
    ) u_gate (
        .clk(clk), .rst_n(rst_n),
        .weight_wr_en(gate_w_wr_en), .weight_wr_row(gate_w_wr_row),
        .weight_wr_beat(gate_w_wr_beat), .weight_wr_data(gate_w_wr_data),
        .activ_wr_en(activ_wr_en), .activ_wr_beat(activ_wr_beat),
        .activ_wr_data(activ_wr_data),
        .scale_wr_en(scale_wr_en), .scale_wr_addr(scale_wr_addr),
        .scale_wr_data(scale_wr_data),
        .start(gate_start), .busy(), .done(gate_done),
        .result_valid(gate_result_valid), .result_row(gate_result_row),
        .result_data(gate_result_data)
    );

    fp4_linear_engine #(
        .M_OUT(INTER), .K_TOTAL(HIDDEN), .LANES(LANES),
        .GROUP_SIZE(4), .NUM_GROUPS(4), .ADDR_WIDTH(2)
    ) u_up (
        .clk(clk), .rst_n(rst_n),
        .weight_wr_en(up_w_wr_en), .weight_wr_row(up_w_wr_row),
        .weight_wr_beat(up_w_wr_beat), .weight_wr_data(up_w_wr_data),
        .activ_wr_en(activ_wr_en), .activ_wr_beat(activ_wr_beat),
        .activ_wr_data(activ_wr_data),
        .scale_wr_en(scale_wr_en), .scale_wr_addr(scale_wr_addr),
        .scale_wr_data(scale_wr_data),
        .start(up_start), .busy(), .done(up_done),
        .result_valid(up_result_valid), .result_row(up_result_row),
        .result_data(up_result_data)
    );

    function automatic int dw_index(input int row, input int col);
        dw_index = row * INTER + col;
    endfunction

    always_ff @(posedge clk) begin
        if (down_w_wr_en) begin
            down_w[dw_index(down_w_wr_row, down_w_wr_col)] <= down_w_wr_data;
        end
        if (gate_result_valid) gate_vec[gate_result_row] <= gate_result_data;
        if (up_result_valid)   up_vec[up_result_row]     <= up_result_data;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            gate_start <= 1'b0;
            up_start <= 1'b0;
            done <= 1'b0;
            result_valid <= 1'b0;
            result_row <= '0;
            result_data <= '0;
            out_row <= '0;
            for (int i = 0; i < INTER; i++) begin
                gate_vec[i] <= '0;
                up_vec[i] <= '0;
                mid_vec[i] <= '0;
            end
        end else begin
            gate_start <= 1'b0;
            up_start <= 1'b0;
            done <= 1'b0;
            result_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        gate_start <= 1'b1;
                        up_start <= 1'b1;
                        state <= S_RUN_LINEAR;
                    end
                end

                S_RUN_LINEAR: begin
                    if (gate_done && up_done) begin
                        state <= S_COMPUTE_MID;
                    end
                end

                S_COMPUTE_MID: begin
                    for (int i = 0; i < INTER; i++) begin
                        mid_vec[i] <= ($signed(silu_vec[i]) * $signed(up_vec[i])) >>> 12;
                    end
                    out_row <= '0;
                    state <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    logic signed [63:0] acc;
                    acc = '0;
                    for (int j = 0; j < INTER; j++) begin
                        acc = acc + ($signed(mid_vec[j]) * $signed(down_w[dw_index(out_row, j)]));
                    end
                    result_valid <= 1'b1;
                    result_row <= out_row;
                    result_data <= acc >>> 12;
                    if (out_row == HIDDEN-1) begin
                        state <= S_DONE;
                    end else begin
                        out_row <= out_row + 1'b1;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (!start) state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
