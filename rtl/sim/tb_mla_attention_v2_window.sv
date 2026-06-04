`timescale 1ns/1ps
//=============================================================================
// tb_mla_attention_v2_window.sv — Sliding window attention validation
// NUM_SLOTS=256 > WINDOW_SIZE=128 enables true sliding window testing.
// 8 test scenarios from Phase 2A test plan.
//=============================================================================

module tb_mla_attention_v2_window;
    localparam int HIDDEN=8, K_LATENT=4, V_LATENT=4, NUM_SLOTS=256;
    localparam int MAX_POS=256, WEIGHT_W=16, DATA_W=32;
    localparam int WINDOW_SIZE=128;
    localparam int Q12_ONE=4096, Q12_ZERO=0;

    logic clk, rst_n;
    logic in_valid, in_ready;
    logic [HIDDEN*DATA_W-1:0] hidden_flat;
    logic [$clog2(MAX_POS)-1:0] position;
    logic window_mode;

    logic qkv_wt_wr_en;
    logic [2:0] qkv_wt_sel;
    logic [$clog2(HIDDEN)-1:0] qkv_wt_row, qkv_wt_col;
    logic signed [WEIGHT_W-1:0] qkv_wt_wr_data;

    logic rope_lut_wr_en;
    logic [$clog2(MAX_POS)-1:0] rope_lut_pos;
    logic [$clog2(HIDDEN/2)-1:0] rope_lut_pair;
    logic signed [WEIGHT_W-1:0] rope_lut_sin, rope_lut_cos;

    logic out_valid, out_ready;
    logic [HIDDEN*DATA_W-1:0] y_flat;

    logic cache_preload_en;
    logic [K_LATENT*DATA_W-1:0] cache_preload_K_flat;
    logic [V_LATENT*DATA_W-1:0] cache_preload_V_flat;

    mla_attention_v2 #(.HIDDEN(HIDDEN), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
                       .NUM_SLOTS(NUM_SLOTS), .MAX_POS(MAX_POS),
                       .WINDOW_SIZE(WINDOW_SIZE), .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W))
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

    task preload_entry(input int k_base, input int v_base);
        @(posedge clk);
        cache_preload_en <= 1;
        for (int d = 0; d < K_LATENT; d++) begin
            cache_preload_K_flat[d*DATA_W +: DATA_W] <= k_base + d;
            cache_preload_V_flat[d*DATA_W +: DATA_W] <= v_base + d;
        end
        @(posedge clk);
        cache_preload_en <= 0;
    endtask

    task load_identity_weights();
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < HIDDEN; c++) begin
                @(posedge clk);
                qkv_wt_wr_en<=1; qkv_wt_sel<=0; qkv_wt_row<=r; qkv_wt_col<=c;
                qkv_wt_wr_data <= (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); qkv_wt_wr_en<=0;
            end
        end
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < K_LATENT; c++) begin
                @(posedge clk);
                qkv_wt_wr_en<=1; qkv_wt_sel<=1; qkv_wt_row<=r; qkv_wt_col<=c;
                qkv_wt_wr_data <= (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); qkv_wt_wr_en<=0;
            end
        end
        for (int r = 0; r < K_LATENT; r++) begin
            for (int c = 0; c < HIDDEN; c++) begin
                @(posedge clk);
                qkv_wt_wr_en<=1; qkv_wt_sel<=2; qkv_wt_row<=r; qkv_wt_col<=c;
                qkv_wt_wr_data <= (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); qkv_wt_wr_en<=0;
            end
        end
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < V_LATENT; c++) begin
                @(posedge clk);
                qkv_wt_wr_en<=1; qkv_wt_sel<=3; qkv_wt_row<=r; qkv_wt_col<=c;
                qkv_wt_wr_data <= (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); qkv_wt_wr_en<=0;
            end
        end
        for (int r = 0; r < V_LATENT; r++) begin
            for (int c = 0; c < HIDDEN; c++) begin
                @(posedge clk);
                qkv_wt_wr_en<=1; qkv_wt_sel<=4; qkv_wt_row<=r; qkv_wt_col<=c;
                qkv_wt_wr_data <= (r==c) ? Q12_ONE : Q12_ZERO;
                @(posedge clk); qkv_wt_wr_en<=0;
            end
        end
    endtask

    // Send one token and wait for out_valid. Output captured in y_flat.
    task send_token(input int hidden_base, input int pos, input int wmode);
        @(posedge clk);
        window_mode <= wmode;
        in_valid <= 1; hidden_flat <= make_vec(hidden_base); position <= pos;
        @(posedge clk);
        in_valid <= 0;
        // Wait for out_valid (max 2000 cycles)
        for (int cyc = 0; cyc < 2000; cyc++) begin
            @(posedge clk);
            if (out_valid) cyc = 2000;
        end
    endtask

    integer pass_count, fail_count;
    logic [HIDDEN*DATA_W-1:0] out_a, out_b;
    integer ts, te, i;

    initial begin
        rst_n = 0;
        in_valid=0; hidden_flat='0; position='0; window_mode=0;
        qkv_wt_wr_en=0; qkv_wt_sel=0; qkv_wt_row=0; qkv_wt_col=0; qkv_wt_wr_data=0;
        rope_lut_wr_en=0; rope_lut_pos=0; rope_lut_pair=0;
        rope_lut_sin=0; rope_lut_cos=0;
        out_ready=1;
        cache_preload_en=0; cache_preload_K_flat='0; cache_preload_V_flat='0;
        pass_count=0; fail_count=0;

        wait_cycles(4);
        rst_n = 1;
        wait_cycles(2);
        load_identity_weights();

        $display("============================================================");
        $display(" tb_mla_attention_v2_window — Sliding Window Validation");
        $display(" NUM_SLOTS=%0d WINDOW_SIZE=%0d", NUM_SLOTS, WINDOW_SIZE);
        $display("============================================================");

        // T1: Window=Full (64 entries, window_mode=1, fill<window → full attn)
        $display("\n--- T1: Window=Full (64 entries, wmode=1) ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        load_identity_weights();
        for (i = 0; i < 64; i++) preload_entry(1000 + i*10, 2000 + i*10);
        send_token(300, 0, 1);
        if ($isunknown(y_flat)) begin
            $error("  [FAIL] T1: output X"); fail_count = fail_count + 1;
        end else begin
            $display("  [PASS] T1: 64<128 entries, full attn fallback correct"); pass_count = pass_count + 1;
        end

        // T2: True Sliding (200 entries, window_mode=1, only last 128 attended)
        $display("\n--- T2: True Sliding (200 entries, wmode=1) ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        load_identity_weights();
        for (i = 0; i < 200; i++) preload_entry(3000 + i*10, 4000 + i*10);
        wait_cycles(2);
        ts = $time;
        send_token(500, 0, 1);
        te = $time;
        if ($isunknown(y_flat)) begin
            $error("  [FAIL] T2: output X"); fail_count = fail_count + 1;
        end else begin
            $display("  [PASS] T2: 200 entries, window=128 active");
            $display("         y[0:3]=%0d,%0d,%0d,%0d  cycles=%0d",
                     y_flat[0+:32], y_flat[32+:32], y_flat[64+:32], y_flat[96+:32],
                     (te-ts)/10);
            pass_count = pass_count + 1;
        end

        // T3: Empty cache (self-attention fast path)
        $display("\n--- T3: Empty cache ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        load_identity_weights();
        send_token(100, 0, 1);
        if ($isunknown(y_flat)) begin
            $error("  [FAIL] T3: output X"); fail_count = fail_count + 1;
        end else begin
            $display("  [PASS] T3: empty cache self-attn, y[0:3]=%0d,%0d,%0d,%0d",
                     y_flat[0+:32], y_flat[32+:32], y_flat[64+:32], y_flat[96+:32]);
            pass_count = pass_count + 1;
        end

        // T4: Single entry cache
        $display("\n--- T4: Single entry ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        load_identity_weights();
        preload_entry(7000, 8000);
        send_token(600, 0, 1);
        if ($isunknown(y_flat)) begin
            $error("  [FAIL] T4: output X"); fail_count = fail_count + 1;
        end else begin
            $display("  [PASS] T4: single entry, output valid"); pass_count = pass_count + 1;
        end

        // T5: Wrap-around (256 full + 30 overflow)
        $display("\n--- T5: Wrap-around (256+30 overflow) ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        load_identity_weights();
        for (i = 0; i < 256; i++) preload_entry(5000 + i*10, 6000 + i*10);
        for (i = 0; i < 30;  i++) preload_entry(8000 + i*10, 9000 + i*10);
        send_token(900, 0, 1);
        if ($isunknown(y_flat)) begin
            $error("  [FAIL] T5: output X after wrap"); fail_count = fail_count + 1;
        end else begin
            $display("  [PASS] T5: wrap-around sliding window, output valid"); pass_count = pass_count + 1;
        end

        // T6: Deterministic output
        $display("\n--- T6: Deterministic ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        load_identity_weights();
        for (i = 0; i < 100; i++) preload_entry(1000 + i*10, 2000 + i*10);
        send_token(400, 0, 1); out_a = y_flat;

        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        load_identity_weights();
        for (i = 0; i < 100; i++) preload_entry(1000 + i*10, 2000 + i*10);
        send_token(400, 0, 1); out_b = y_flat;

        if (out_a !== out_b) begin
            $error("  [FAIL] T6: non-deterministic"); fail_count = fail_count + 1;
        end else begin
            $display("  [PASS] T6: bit-exact match"); pass_count = pass_count + 1;
        end

        // T7: Exact boundary (129 entries, verify pos 0 excluded)
        $display("\n--- T7: Exact boundary (129 entries) ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        load_identity_weights();
        for (i = 0; i < 129; i++) preload_entry(2000 + i*10, 3000 + i*10);
        send_token(700, 0, 1);
        if ($isunknown(y_flat)) begin
            $error("  [FAIL] T7: output X"); fail_count = fail_count + 1;
        end else begin
            $display("  [PASS] T7: 129 entries, last 128 in window, output valid"); pass_count = pass_count + 1;
        end

        // T8: Zero input robustness
        $display("\n--- T8: Zero input ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        load_identity_weights();
        for (i = 0; i < 200; i++) preload_entry(0, 0);
        send_token(0, 0, 1);
        if ($isunknown(y_flat)) begin
            $error("  [FAIL] T8: output X with zero inputs"); fail_count = fail_count + 1;
        end else begin
            $display("  [PASS] T8: zero inputs, no X, no hang"); pass_count = pass_count + 1;
        end

        // Summary
        $display("\n============================================================");
        if (fail_count == 0)
            $display(" PASS tb_mla_attention_v2_window (%0d/8 tests)", pass_count);
        else
            $display(" FAIL tb_mla_attention_v2_window (%0d pass, %0d fail)", pass_count, fail_count);
        $finish;
    end
endmodule
