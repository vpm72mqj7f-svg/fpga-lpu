//=============================================================================
// mla_attention.sv — simplified attention (HIDDEN=8, packed-memory version)
//=============================================================================

module mla_attention (
    input  logic clk, rst_n,
    input  logic score_wr_en, v_wr_en,
    input  logic [5:0] score_wr_idx, v_wr_idx,
    input  logic signed [31:0] score_wr_data, v_wr_data,
    input  logic valid_in,
    output logic valid_out,
    output logic signed [31:0] y0,y1,y2,y3,y4,y5,y6,y7
);
    localparam int N2 = 64, H = 8;
    logic signed [N2*32-1:0] sc_p, vl_p;

    always_ff @(posedge clk) begin
        if (score_wr_en) sc_p[score_wr_idx*32 +: 32] <= score_wr_data;
        if (v_wr_en)     vl_p[v_wr_idx*32   +: 32] <= v_wr_data;
    end

    logic signed [31:0] c_out [H];
    always_comb begin : compute
        logic signed [31:0] mx, ex [N2], sm [N2];
        logic signed [63:0] esum;
        logic signed [31:0] sc [N2], vl [N2], adj;

        // Unpack
        for (int i = 0; i < N2; i++) begin
            sc[i] = sc_p[i*32 +: 32];
            vl[i] = vl_p[i*32 +: 32];
        end

        // Max
        mx = -32'sd1 << 30;
        for (int i = 0; i < N2; i++) if (sc[i] > mx) mx = sc[i];

        // Exp
        esum = 0;
        for (int i = 0; i < N2; i++) begin
            adj = sc[i] - mx;
            if (adj > -32'sd256)             ex[i] = 4096;
            else if (adj > -32'sd1024)       ex[i] = 3545;
            else if (adj > -32'sd2048)       ex[i] = 2588;
            else if (adj > -32'sd4096)       ex[i] = 1507;
            else if (adj > -32'sd8192)       ex[i] = 538;
            else                              ex[i] = 48;
            esum = esum + ex[i];
        end

        // Softmax
        for (int i = 0; i < N2; i++)
            sm[i] = (ex[i] * 4096) / (esum > 64 ? esum : 64);

        // V-weighted output
        for (int j = 0; j < H; j++) c_out[j] = 0;
        for (int i = 0; i < H; i++)
            for (int j = 0; j < H; j++)
                c_out[j] = c_out[j] + ((sm[i] * vl[j*H + i]) >>> 12);
    end

    logic vo_d1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin {y0,y1,y2,y3,y4,y5,y6,y7} <= '0; valid_out <= 0; vo_d1 <= 0; end
        else begin
            vo_d1 <= valid_in;
            if (vo_d1) begin
                y0<=c_out[0]; y1<=c_out[1]; y2<=c_out[2]; y3<=c_out[3];
                y4<=c_out[4]; y5<=c_out[5]; y6<=c_out[6]; y7<=c_out[7];
                valid_out <= 1;
            end else valid_out <= 0;
        end
    end

endmodule
