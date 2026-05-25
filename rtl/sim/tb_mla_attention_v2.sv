`timescale 1ns/1ps

module tb_mla_attention_v2;
    localparam int HIDDEN=8, K_LATENT=4, V_LATENT=4, NUM_SLOTS=64;
    localparam int MAX_POS=64, WEIGHT_W=16, DATA_W=32;
    localparam int Q12_ONE=4096, Q12_ZERO=0;

    logic clk, rst_n;
    logic in_valid, in_ready;
    logic [HIDDEN*DATA_W-1:0] hidden_flat;
    logic [5:0] position;

    // QKV weight load
    logic qkv_wt_wr_en;
    logic [2:0] qkv_wt_sel;
    logic [2:0] qkv_wt_row, qkv_wt_col;
    logic signed [WEIGHT_W-1:0] qkv_wt_wr_data;

    // RoPE LUT load
    logic rope_lut_wr_en;
    logic [5:0] rope_lut_pos;
    logic [1:0] rope_lut_pair;
    logic signed [WEIGHT_W-1:0] rope_lut_sin, rope_lut_cos;

    // Output
    logic out_valid, out_ready;
    logic signed [DATA_W-1:0] y0,y1,y2,y3,y4,y5,y6,y7;

    mla_attention_v2 #(.HIDDEN(HIDDEN), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
                       .NUM_SLOTS(NUM_SLOTS), .MAX_POS(MAX_POS),
                       .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W))
        dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task wait_cycles(input int n);
        for (int i = 0; i < n; i++) @(posedge clk);
    endtask

    function [HIDDEN*DATA_W-1:0] make_vec(input int base);
        reg [HIDDEN*DATA_W-1:0] v;
        for (int d = 0; d < HIDDEN; d++) v[d*DATA_W +: DATA_W] = base + d;
        make_vec = v;
    endfunction

    integer pass_count, fail_count;

    initial begin
        rst_n = 0; in_valid = 0; hidden_flat = '0; position = 0;
        qkv_wt_wr_en = 0; qkv_wt_sel = 0; qkv_wt_row = 0; qkv_wt_col = 0;
        qkv_wt_wr_data = 0; rope_lut_wr_en = 0; rope_lut_pos = 0;
        rope_lut_pair = 0; rope_lut_sin = 0; rope_lut_cos = 0;
        out_ready = 1;
        pass_count = 0; fail_count = 0;

        wait_cycles(4);
        rst_n = 1;
        wait_cycles(2);

        // ============================================
        // Load QKV weights (identity projection)
        // ============================================
        $display("Loading QKV weights...");
        // W_Q = identity (8×8)
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < HIDDEN; c++) begin
                @(posedge clk);
                qkv_wt_wr_en=1; qkv_wt_sel=0; qkv_wt_row=r; qkv_wt_col=c;
                qkv_wt_wr_data = (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); qkv_wt_wr_en=0;
            end
        end
        // W_K = I (8×4, first 4 cols of identity)
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < K_LATENT; c++) begin
                @(posedge clk);
                qkv_wt_wr_en=1; qkv_wt_sel=1; qkv_wt_row=r; qkv_wt_col=c;
                qkv_wt_wr_data = (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); qkv_wt_wr_en=0;
            end
        end
        // W_K_up = I (4×8, first 4 rows of identity)
        for (int r = 0; r < K_LATENT; r++) begin
            for (int c = 0; c < HIDDEN; c++) begin
                @(posedge clk);
                qkv_wt_wr_en=1; qkv_wt_sel=2; qkv_wt_row=r; qkv_wt_col=c;
                qkv_wt_wr_data = (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); qkv_wt_wr_en=0;
            end
        end
        // W_V = I (8×4)
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < V_LATENT; c++) begin
                @(posedge clk);
                qkv_wt_wr_en=1; qkv_wt_sel=3; qkv_wt_row=r; qkv_wt_col=c;
                qkv_wt_wr_data = (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); qkv_wt_wr_en=0;
            end
        end
        // W_V_up = I (4×8)
        for (int r = 0; r < V_LATENT; r++) begin
            for (int c = 0; c < HIDDEN; c++) begin
                @(posedge clk);
                qkv_wt_wr_en=1; qkv_wt_sel=4; qkv_wt_row=r; qkv_wt_col=c;
                qkv_wt_wr_data = (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); qkv_wt_wr_en=0;
            end
        end

        // Load RoPE identity (cos=1, sin=0 for all) — default already set
        $display("Weights loaded.");

        // ============================================
        // Test: Single-token self-attention passthrough
        // With identity weights and no cache: output = V = hidden
        // ============================================
        $display("Test: Single-token self-attention");
        @(posedge clk);
        in_valid = 1; hidden_flat = make_vec(100); position = 0;
        @(posedge clk);
        in_valid = 0;

        for (int cyc = 0; cyc < 100; cyc++) begin
            @(posedge clk);
            if (out_valid) begin
                $display("  Output: y=(%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d)",
                    y0,y1,y2,y3,y4,y5,y6,y7);
                // V = low-rank identity: dims 0-3 = hidden[0-3], dims 4-7 = 0
                if (y0 !== 100 || y1 !== 101 || y2 !== 102 || y3 !== 103) begin
                    $error("  [FAIL] V dims 0-3: expected 100-103, got %0d-%0d", y0,y3);
                    fail_count = fail_count + 1;
                end
                if (y4 !== 0 || y5 !== 0 || y6 !== 0 || y7 !== 0) begin
                    $error("  [FAIL] V dims 4-7: expected 0, got non-zero");
                    fail_count = fail_count + 1;
                end
                if (fail_count == 0) begin
                    $display("  [ OK ] Self-attention passthrough (V=hidden dims 0-3)");
                    pass_count = pass_count + 1;
                end
            end
        end

        $display("==============================");
        if (fail_count == 0)
            $display("PASS tb_mla_attention_v2 (%0d tests)", pass_count);
        else
            $display("FAIL tb_mla_attention_v2 (%0d pass, %0d fail)", pass_count, fail_count);
        $finish;
    end

endmodule
