`timescale 1ns/1ps

module tb_mla_qkv;
    localparam int HIDDEN=8, K_LATENT=4, V_LATENT=4, WEIGHT_W=16, DATA_W=32;
    localparam int MAX_POS=64, NUM_SLOTS=64, N_PAIRS=4;
    localparam int Q12_ONE=4096, Q12_ZERO=0;

    // ---- mla_qkv_proj signals ----
    logic clk, rst_n;
    logic in_valid, in_ready;
    logic [HIDDEN*DATA_W-1:0] hidden_flat;
    logic wt_wr_en;
    logic [2:0] wt_sel;
    logic [2:0] wt_row, wt_col;
    logic signed [WEIGHT_W-1:0] wt_wr_data;
    logic out_valid, out_ready;
    logic [HIDDEN*DATA_W-1:0] Q_flat, K_flat, V_flat;
    logic [K_LATENT*DATA_W-1:0] K_latent_flat, V_latent_flat;

    // ---- mla_rope signals ----
    logic rope_in_valid, rope_in_ready;
    logic [HIDDEN*DATA_W-1:0] rope_in_flat;
    logic [5:0] rope_pos;
    logic lut_wr_en;
    logic [1:0] lut_pair;
    logic signed [WEIGHT_W-1:0] lut_sin_data, lut_cos_data;
    logic rope_out_valid;
    logic [HIDDEN*DATA_W-1:0] rope_out_flat;

    // ---- mla_kv_cache signals ----
    logic cache_wr_en;
    logic [K_LATENT*DATA_W-1:0] cache_K_in, cache_K_out;
    logic [V_LATENT*DATA_W-1:0] cache_V_in, cache_V_out;
    logic [5:0] cache_wr_addr, cache_rd_addr;
    logic cache_rd_en, cache_rd_valid;
    logic [5:0] cache_fill_count;
    logic cache_full, cache_empty;

    // DUTs
    mla_qkv_proj #(.HIDDEN(HIDDEN), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
                   .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W))
        u_qkv (.*);

    mla_rope #(.HIDDEN(HIDDEN), .MAX_POS(MAX_POS), .COEFF_W(WEIGHT_W), .DATA_W(DATA_W))
        u_rope (.in_valid(rope_in_valid), .in_ready(rope_in_ready),
                .vec_flat(rope_in_flat), .pos(rope_pos),
                .lut_wr_en(lut_wr_en), .lut_pos(rope_pos), .lut_pair(lut_pair),
                .lut_sin_data(lut_sin_data), .lut_cos_data(lut_cos_data),
                .out_valid(rope_out_valid), .rot_flat(rope_out_flat), .*);

    mla_kv_cache #(.NUM_SLOTS(NUM_SLOTS), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
                   .DATA_W(DATA_W))
        u_cache (.wr_en(cache_wr_en), .K_latent_flat(cache_K_in),
                 .V_latent_flat(cache_V_in), .wr_addr(cache_wr_addr),
                 .rd_en(cache_rd_en), .rd_addr(cache_rd_addr),
                 .rd_valid(cache_rd_valid), .rd_K_flat(cache_K_out),
                 .rd_V_flat(cache_V_out), .fill_count(cache_fill_count),
                 .full(cache_full), .empty(cache_empty), .*);

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

    function [DATA_W-1:0] extract(input logic [HIDDEN*DATA_W-1:0] vec, input int d);
        extract = vec[d*DATA_W +: DATA_W];
    endfunction

    integer pass_count, fail_count;

    initial begin
        rst_n = 0;
        in_valid=0; hidden_flat='0; wt_wr_en=0; out_ready=1;
        rope_in_valid=0; rope_in_flat='0; rope_pos='0;
        lut_wr_en=0; lut_pair='0; lut_sin_data='0; lut_cos_data='0;
        cache_wr_en=0; cache_K_in='0; cache_V_in='0;
        cache_rd_en=0; cache_rd_addr='0;
        pass_count=0; fail_count=0;

        wait_cycles(4);
        rst_n = 1;
        wait_cycles(2);

        // ============================================
        // Test 1: mla_qkv_proj — identity weights
        // W_Q = identity, W_K × W_K_up = identity, etc.
        // Expected: Q=K=V=hidden
        // ============================================
        $display("Test 1: QKV identity projection");

        // Load W_Q = identity (Q12_ONE on diagonal)
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < HIDDEN; c++) begin
                @(posedge clk);
                wt_wr_en=1; wt_sel=0; wt_row=r; wt_col=c;
                wt_wr_data = (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); wt_wr_en=0;
            end
        end
        // Load W_K = [I | 0] (first 4 cols are identity)
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < K_LATENT; c++) begin
                @(posedge clk);
                wt_wr_en=1; wt_sel=1; wt_row=r; wt_col=c;
                wt_wr_data = (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); wt_wr_en=0;
            end
        end
        // Load W_K_up = identity (first 4 rows)
        for (int r = 0; r < K_LATENT; r++) begin
            for (int c = 0; c < HIDDEN; c++) begin
                @(posedge clk);
                wt_wr_en=1; wt_sel=2; wt_row=r; wt_col=c;
                wt_wr_data = (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); wt_wr_en=0;
            end
        end
        // Load W_V = [I | 0]
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < V_LATENT; c++) begin
                @(posedge clk);
                wt_wr_en=1; wt_sel=3; wt_row=r; wt_col=c;
                wt_wr_data = (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); wt_wr_en=0;
            end
        end
        // Load W_V_up = identity
        for (int r = 0; r < V_LATENT; r++) begin
            for (int c = 0; c < HIDDEN; c++) begin
                @(posedge clk);
                wt_wr_en=1; wt_sel=4; wt_row=r; wt_col=c;
                wt_wr_data = (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); wt_wr_en=0;
            end
        end

        // Send input
        @(posedge clk);
        in_valid = 1; hidden_flat = make_vec(100);
        @(posedge clk);
        in_valid = 0;

        // Wait for output
        for (int cyc = 0; cyc < 60; cyc++) begin
            @(posedge clk);
            if (out_valid) begin
                // Q = full identity (all 8 dims match)
                for (int d = 0; d < HIDDEN; d++) begin
                    if (extract(Q_flat,d) !== 100+d) begin
                        $error("  [FAIL] Q dim %0d: got %0d exp %0d",
                            d, extract(Q_flat,d), 100+d);
                        fail_count = fail_count + 1;
                    end
                end
                // K = low-rank identity (dims 0-3 preserved, 4-7 lost)
                for (int d = 0; d < K_LATENT; d++) begin
                    if (extract(K_flat,d) !== 100+d) begin
                        $error("  [FAIL] K dim %0d: got %0d exp %0d",
                            d, extract(K_flat,d), 100+d);
                        fail_count = fail_count + 1;
                    end
                end
                // K dims 4-7 should be 0 (rank-4 approximation of identity)
                for (int d = 4; d < HIDDEN; d++) begin
                    if (extract(K_flat,d) !== 0) begin
                        $error("  [FAIL] K dim %0d: got %0d exp 0", d, extract(K_flat,d));
                        fail_count = fail_count + 1;
                    end
                end
                // V = same as K
                for (int d = 0; d < K_LATENT; d++) begin
                    if (extract(V_flat,d) !== 100+d) begin
                        $error("  [FAIL] V dim %0d: got %0d exp %0d",
                            d, extract(V_flat,d), 100+d);
                        fail_count = fail_count + 1;
                    end
                end
                if (fail_count == 0) begin
                    $display("  [ OK ] Test 1: QKV projection (Q=identity, K/V=low-rank identity)");
                    pass_count = pass_count + 1;
                end
            end
        end

        wait_cycles(4);

        // ============================================
        // Test 2: RoPE identity (pos=0, cos=1, sin=0)
        // ============================================
        $display("Test 2: RoPE identity (pos=0, no rotation)");
        // LUT already has cos=1.0 (default), sin=0 (default) for all positions

        @(posedge clk);
        rope_in_valid = 1; rope_in_flat = make_vec(200); rope_pos = 0;
        @(posedge clk);
        rope_in_valid = 0;

        for (int cyc = 0; cyc < 30; cyc++) begin
            @(posedge clk);
            if (rope_out_valid) begin
                for (int d = 0; d < HIDDEN; d++) begin
                    if (extract(rope_out_flat,d) !== 200+d) begin
                        $error("  [FAIL] dim %0d: got %0d exp %0d", d,
                            extract(rope_out_flat,d), 200+d);
                        fail_count = fail_count + 1;
                    end
                end
                if (fail_count == 0) begin
                    $display("  [ OK ] Test 2: RoPE identity at pos=0");
                    pass_count = pass_count + 1;
                end
            end
        end

        wait_cycles(2);

        // ============================================
        // Test 3: RoPE 90-degree rotation at pos=1
        // Load: pair 0: cos=0, sin=Q12_ONE → (x,y)→(-y,x)
        // ============================================
        $display("Test 3: RoPE 90-degree rotation (pair 0)");
        @(posedge clk);
        lut_wr_en=1; rope_pos=1; lut_pair=0;
        lut_cos_data=0; lut_sin_data=Q12_ONE;
        @(posedge clk); lut_wr_en=0;

        @(posedge clk);
        rope_in_valid=1; rope_in_flat=make_vec(10); rope_pos=1;
        @(posedge clk);
        rope_in_valid=0;

        for (int cyc = 0; cyc < 30; cyc++) begin
            @(posedge clk);
            if (rope_out_valid) begin
                // pair 0: (10,11) → (-11,10)
                // other pairs unchanged
                if (extract(rope_out_flat,0) !== -11) begin
                    $error("  [FAIL] dim0: got %0d exp -11", extract(rope_out_flat,0));
                    fail_count = fail_count + 1;
                end
                if (extract(rope_out_flat,1) !== 10) begin
                    $error("  [FAIL] dim1: got %0d exp 10", extract(rope_out_flat,1));
                    fail_count = fail_count + 1;
                end
                if (extract(rope_out_flat,2) !== 12 || extract(rope_out_flat,3) !== 13) begin
                    $error("  [FAIL] other dims changed");
                    fail_count = fail_count + 1;
                end
                if (fail_count == 0) begin
                    $display("  [ OK ] Test 3: 90-degree rotation");
                    pass_count = pass_count + 1;
                end
            end
        end

        wait_cycles(2);

        // ============================================
        // Test 4: KV Cache write and read
        // ============================================
        $display("Test 4: KV Cache write/read");

        // Write 3 entries
        for (int i = 0; i < 3; i++) begin
            @(posedge clk);
            cache_wr_en = 1;
            for (int d = 0; d < K_LATENT; d++) begin
                cache_K_in[d*DATA_W +: DATA_W] = 1000 + i*10 + d;
                cache_V_in[d*DATA_W +: DATA_W] = 2000 + i*10 + d;
            end
            @(posedge clk);
            cache_wr_en = 0;
        end

        wait_cycles(2);

        // Read back and verify
        for (int i = 0; i < 3; i++) begin
            @(posedge clk);
            cache_rd_en = 1; cache_rd_addr = i;
            @(posedge clk);
            cache_rd_en = 0;
            if (cache_rd_valid) begin
                for (int d = 0; d < K_LATENT; d++) begin
                    if (cache_K_out[d*DATA_W +: DATA_W] !== (1000 + i*10 + d)) begin
                        $error("  [FAIL] K[%0d][%0d]: got %0d exp %0d",
                            i, d, cache_K_out[d*DATA_W +: DATA_W], 1000+i*10+d);
                        fail_count = fail_count + 1;
                    end
                    if (cache_V_out[d*DATA_W +: DATA_W] !== (2000 + i*10 + d)) begin
                        $error("  [FAIL] V[%0d][%0d]: got %0d exp %0d",
                            i, d, cache_V_out[d*DATA_W +: DATA_W], 2000+i*10+d);
                        fail_count = fail_count + 1;
                    end
                end
            end else begin
                $error("  [FAIL] cache read not valid for addr %0d", i);
                fail_count = fail_count + 1;
            end
        end

        if (fail_count == 0) begin
            $display("  [ OK ] Test 4: KV cache write/read");
            pass_count = pass_count + 1;
        end

        // Check fill count
        if (cache_fill_count !== 3) begin
            $error("  [FAIL] fill_count expected 3, got %0d", cache_fill_count);
            fail_count = fail_count + 1;
        end

        $display("==============================");
        if (fail_count == 0)
            $display("PASS tb_mla_qkv (%0d tests)", pass_count);
        else
            $display("FAIL tb_mla_qkv (%0d pass, %0d fail)", pass_count, fail_count);
        $finish;
    end

endmodule
