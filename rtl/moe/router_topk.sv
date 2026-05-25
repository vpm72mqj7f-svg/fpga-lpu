//=============================================================================
// router_topk.sv — MoE Router Top-K (HIDDEN=8, EXPERTS=4 bring-up)
//=============================================================================

module router_topk #(
    parameter int EXPERTS  = 4
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         w_wr_en,
    input  logic [1:0]                   w_wr_expert,
    input  logic [2:0]                   w_wr_idx,
    input  logic signed [31:0]           w_wr_data,

    input  logic                         valid_in,
    input  logic signed [31:0]           a0, a1, a2, a3, a4, a5, a6, a7,

    output logic                         valid_out,
    input  logic                         result_ready,
    output logic [1:0]                   top0_idx, top1_idx,
    output logic signed [31:0]           top0_score, top1_score
);

    // 4 experts × 8 hidden = 32 weights
    logic signed [31:0] w_e0h0, w_e0h1, w_e0h2, w_e0h3, w_e0h4, w_e0h5, w_e0h6, w_e0h7;
    logic signed [31:0] w_e1h0, w_e1h1, w_e1h2, w_e1h3, w_e1h4, w_e1h5, w_e1h6, w_e1h7;
    logic signed [31:0] w_e2h0, w_e2h1, w_e2h2, w_e2h3, w_e2h4, w_e2h5, w_e2h6, w_e2h7;
    logic signed [31:0] w_e3h0, w_e3h1, w_e3h2, w_e3h3, w_e3h4, w_e3h5, w_e3h6, w_e3h7;

    logic signed [31:0] ar0, ar1, ar2, ar3, ar4, ar5, ar6, ar7;
    logic active;
    logic holding;
    logic [0:0] delay;
    logic signed [63:0] s0, s1, s2, s3;
    logic signed [63:0] best, second;
    logic [1:0] bi, si;

    function automatic void set_w(input [1:0] ex, input [2:0] ix, input signed [31:0] d);
        case ({ex, ix})
            {2'd0, 3'd0}: w_e0h0 = d; {2'd0, 3'd1}: w_e0h1 = d;
            {2'd0, 3'd2}: w_e0h2 = d; {2'd0, 3'd3}: w_e0h3 = d;
            {2'd0, 3'd4}: w_e0h4 = d; {2'd0, 3'd5}: w_e0h5 = d;
            {2'd0, 3'd6}: w_e0h6 = d; {2'd0, 3'd7}: w_e0h7 = d;
            {2'd1, 3'd0}: w_e1h0 = d; {2'd1, 3'd1}: w_e1h1 = d;
            {2'd1, 3'd2}: w_e1h2 = d; {2'd1, 3'd3}: w_e1h3 = d;
            {2'd1, 3'd4}: w_e1h4 = d; {2'd1, 3'd5}: w_e1h5 = d;
            {2'd1, 3'd6}: w_e1h6 = d; {2'd1, 3'd7}: w_e1h7 = d;
            {2'd2, 3'd0}: w_e2h0 = d; {2'd2, 3'd1}: w_e2h1 = d;
            {2'd2, 3'd2}: w_e2h2 = d; {2'd2, 3'd3}: w_e2h3 = d;
            {2'd2, 3'd4}: w_e2h4 = d; {2'd2, 3'd5}: w_e2h5 = d;
            {2'd2, 3'd6}: w_e2h6 = d; {2'd2, 3'd7}: w_e2h7 = d;
            {2'd3, 3'd0}: w_e3h0 = d; {2'd3, 3'd1}: w_e3h1 = d;
            {2'd3, 3'd2}: w_e3h2 = d; {2'd3, 3'd3}: w_e3h3 = d;
            {2'd3, 3'd4}: w_e3h4 = d; {2'd3, 3'd5}: w_e3h5 = d;
            {2'd3, 3'd6}: w_e3h6 = d; {2'd3, 3'd7}: w_e3h7 = d;
        endcase
    endfunction

    always_ff @(posedge clk) begin
        if (w_wr_en) set_w(w_wr_expert, w_wr_idx, w_wr_data);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0; active <= 1'b0; delay <= '0;
            holding <= 1'b0;
            top0_idx <= '0; top1_idx <= '0; top0_score <= '0; top1_score <= '0;
        end else begin
            valid_out <= 1'b0;
            if (valid_in && !active) begin
                active <= 1'b1; delay <= 1'b1;
                ar0 <= a0; ar1 <= a1; ar2 <= a2; ar3 <= a3;
                ar4 <= a4; ar5 <= a5; ar6 <= a6; ar7 <= a7;
            end else if (active) begin
                if (delay == 0 && !holding) begin
                    holding <= 1'b1; active <= 1'b0;
                    // Dot products (same as before)
                    s0 = $signed(ar0)*$signed(w_e0h0) + $signed(ar1)*$signed(w_e0h1)
                       + $signed(ar2)*$signed(w_e0h2) + $signed(ar3)*$signed(w_e0h3)
                       + $signed(ar4)*$signed(w_e0h4) + $signed(ar5)*$signed(w_e0h5)
                       + $signed(ar6)*$signed(w_e0h6) + $signed(ar7)*$signed(w_e0h7);
                    s1 = $signed(ar0)*$signed(w_e1h0) + $signed(ar1)*$signed(w_e1h1)
                       + $signed(ar2)*$signed(w_e1h2) + $signed(ar3)*$signed(w_e1h3)
                       + $signed(ar4)*$signed(w_e1h4) + $signed(ar5)*$signed(w_e1h5)
                       + $signed(ar6)*$signed(w_e1h6) + $signed(ar7)*$signed(w_e1h7);
                    s2 = $signed(ar0)*$signed(w_e2h0) + $signed(ar1)*$signed(w_e2h1)
                       + $signed(ar2)*$signed(w_e2h2) + $signed(ar3)*$signed(w_e2h3)
                       + $signed(ar4)*$signed(w_e2h4) + $signed(ar5)*$signed(w_e2h5)
                       + $signed(ar6)*$signed(w_e2h6) + $signed(ar7)*$signed(w_e2h7);
                    s3 = $signed(ar0)*$signed(w_e3h0) + $signed(ar1)*$signed(w_e3h1)
                       + $signed(ar2)*$signed(w_e3h2) + $signed(ar3)*$signed(w_e3h3)
                       + $signed(ar4)*$signed(w_e3h4) + $signed(ar5)*$signed(w_e3h5)
                       + $signed(ar6)*$signed(w_e3h6) + $signed(ar7)*$signed(w_e3h7);
                    // Top-2 from 4 experts
                    best = s0; bi = 2'd0;
                    if (s1 > best) begin best = s1; bi = 2'd1; end
                    if (s2 > best) begin best = s2; bi = 2'd2; end
                    if (s3 > best) begin best = s3; bi = 2'd3; end
                    second = -64'sd1 << 62;
                    for (int e = 0; e < 4; e++) begin
                        if (e != bi) begin
                            if      (e==0 && s0 > second) begin second = s0; si = 2'd0; end
                            else if (e==1 && s1 > second) begin second = s1; si = 2'd1; end
                            else if (e==2 && s2 > second) begin second = s2; si = 2'd2; end
                            else if (e==3 && s3 > second) begin second = s3; si = 2'd3; end
                        end
                    end
                    top0_idx <= bi; top1_idx <= si;
                    top0_score <= best[31:0]; top1_score <= second[31:0];
                    valid_out <= 1'b1;
                end else if (delay == 1 && !holding) begin
                    delay <= 1'b0;
                end else if (holding) begin
                    if (result_ready) begin
                        valid_out <= 1'b0; holding <= 1'b0;
                    end
                end
            end
        end
    end

endmodule
