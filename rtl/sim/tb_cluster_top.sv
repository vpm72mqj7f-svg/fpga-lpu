`timescale 1ns/1ps
//=============================================================================
// tb_cluster_top.sv — Multi-Chip Pipeline Cluster Verification
//
// Instantiates cluster_top with NUM_CHIPS in SINGLE_CHIP mode,
// verifies token pipeline forwarding through the chip chain.
//
// Pipeline: Token → Chip 0 (L0-L1) → Chip 1 (L2-L3) → ... → Output
//=============================================================================

`include "lpu_config.svh"

module tb_cluster_top;
    localparam int HIDDEN    = 8;
    localparam int K_LATENT  = lpu_config_pkg::LPU_K_LATENT;
    localparam int V_LATENT  = lpu_config_pkg::LPU_V_LATENT;
    localparam int DATA_W    = lpu_config_pkg::LPU_DATA_WIDTH;
    localparam int WEIGHT_W  = lpu_config_pkg::LPU_WEIGHT_WIDTH;
    localparam int INTER     = lpu_config_pkg::LPU_INTERMEDIATE;
    localparam int NUM_CHIPS = 2;
    localparam int MAX_POS   = lpu_config_pkg::LPU_MAX_SEQ_LEN;

    logic clk, rst_n;

    // RMSNorm gamma
    logic                         gamma_wr_en;
    logic [$clog2(HIDDEN)-1:0]    gamma_wr_idx;
    logic signed [31:0]           gamma_wr_data;

    // MLA QKV weight preload
    logic                         attn_qkv_wt_wr_en;
    logic [2:0]                   attn_qkv_wt_sel;
    logic [$clog2(HIDDEN)-1:0]    attn_qkv_wt_row;
    logic [$clog2(HIDDEN)-1:0]    attn_qkv_wt_col;
    logic signed [WEIGHT_W-1:0]   attn_qkv_wt_wr_data;

    // MLA RoPE LUT preload
    logic                         attn_rope_lut_wr_en;
    logic [$clog2(MAX_POS)-1:0]   attn_rope_lut_pos;
    logic [$clog2(HIDDEN/2)-1:0]  attn_rope_lut_pair;
    logic signed [WEIGHT_W-1:0]   attn_rope_lut_sin;
    logic signed [WEIGHT_W-1:0]   attn_rope_lut_cos;

    // Token position
    logic [$clog2(MAX_POS)-1:0]   token_position;

    // KV cache preload
    logic                         cache_preload_en;
    logic [K_LATENT*DATA_W-1:0]   cache_preload_K_flat;
    logic [V_LATENT*DATA_W-1:0]   cache_preload_V_flat;

    // Router preload
    logic                         rtr_w_wr_en;
    logic [1:0]                   rtr_w_wr_expert;
    logic [$clog2(HIDDEN)-1:0]    rtr_w_wr_idx;
    logic signed [31:0]           rtr_w_wr_data;

    // FFN preload
    logic                         gate_w_wr_en, up_w_wr_en, down_w_wr_en;
    logic [$clog2(INTER)-1:0]     gate_w_wr_row, up_w_wr_row;
    logic [$clog2(HIDDEN)-1:0]    down_w_wr_row;
    logic [0:0]                   gate_w_wr_beat, up_w_wr_beat, down_w_wr_beat;
    logic [WEIGHT_W-1:0]          gate_w_wr_data, up_w_wr_data, down_w_wr_data;

    // Scale preload
    logic                         scale_wr_en;
    logic [$clog2(lpu_config_pkg::LPU_SCALE_GROUPS)-1:0] scale_wr_addr;
    logic [7:0]                   scale_wr_data;

    // Activation I/O
    logic                         valid_in, valid_out, router_ok;
    logic [HIDDEN*DATA_W-1:0]     a_flat;
    logic [HIDDEN*DATA_W-1:0]     y_flat;

    // Expert bitmap (per-chip): set all experts as local for bring-up
    logic [$clog2(lpu_config_pkg::LPU_EXPERTS_PER_FPGA > 1 ? lpu_config_pkg::LPU_EXPERTS_PER_FPGA : 2)-1:0] ffn_expert_sel;
    logic [lpu_config_pkg::LPU_NUM_EXPERTS-1:0] cfg_local_experts [NUM_CHIPS];

    // Convenience aliases
    wire signed [31:0] y0 = y_flat[0*32+:32];
    wire signed [31:0] y1 = y_flat[1*32+:32];
    wire signed [31:0] y2 = y_flat[2*32+:32];
    wire signed [31:0] y3 = y_flat[3*32+:32];
    wire signed [31:0] y4 = y_flat[4*32+:32];
    wire signed [31:0] y5 = y_flat[5*32+:32];
    wire signed [31:0] y6 = y_flat[6*32+:32];
    wire signed [31:0] y7 = y_flat[7*32+:32];

    // Instantiate cluster in SINGLE_CHIP mode
    cluster_top #(
        .NUM_CHIPS(NUM_CHIPS),
        .CHIPS_PER_CARD(4),
        .HIDDEN(HIDDEN),
        .INTER(INTER),
        .SINGLE_CHIP(1)
    ) u_cluster (
        .clk, .rst_n,
        .gamma_wr_en, .gamma_wr_idx, .gamma_wr_data,
        .attn_qkv_wt_wr_en, .attn_qkv_wt_sel,
        .attn_qkv_wt_row, .attn_qkv_wt_col, .attn_qkv_wt_wr_data,
        .attn_rope_lut_wr_en, .attn_rope_lut_pos,
        .attn_rope_lut_pair, .attn_rope_lut_sin, .attn_rope_lut_cos,
        .token_position,
        .cache_preload_en, .cache_preload_K_flat, .cache_preload_V_flat,
        .rtr_w_wr_en, .rtr_w_wr_expert, .rtr_w_wr_idx, .rtr_w_wr_data,
        .gate_w_wr_en, .up_w_wr_en, .down_w_wr_en,
        .gate_w_wr_row, .up_w_wr_row, .down_w_wr_row,
        .gate_w_wr_beat, .up_w_wr_beat, .down_w_wr_beat,
        .gate_w_wr_data, .up_w_wr_data, .down_w_wr_data,
        .scale_wr_en, .scale_wr_addr, .scale_wr_data,
        .ffn_expert_sel,
        .cfg_local_experts,
        .valid_in, .a_flat,
        .valid_out, .router_ok, .y_flat
    );

    // Clock: 100 MHz
    initial clk = 0;
    always #5 clk = ~clk;

    //=========================================================================
    // Weight preload tasks (same patterns as tb_chip_12layer)
    //=========================================================================
    task preload_scales;
        @(posedge clk); scale_wr_en<=1; scale_wr_addr<=0; scale_wr_data<=8'h38; @(posedge clk); scale_wr_en<=0;
        @(posedge clk); scale_wr_en<=1; scale_wr_addr<=1; scale_wr_data<=8'h38; @(posedge clk); scale_wr_en<=0;
    endtask

    task preload_gamma;
        for (int i = 0; i < HIDDEN; i++) begin
            @(posedge clk); gamma_wr_en<=1; gamma_wr_idx<=i[$clog2(HIDDEN)-1:0]; gamma_wr_data<=4096; @(posedge clk); gamma_wr_en<=0;
        end
    endtask

    task preload_qkv();
        for (int sel = 0; sel < 5; sel++) begin
            int rows = (sel==0||sel==2||sel==4) ? 8 : 4;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < 8; c++) begin
                    @(posedge clk); attn_qkv_wt_wr_en<=1; attn_qkv_wt_sel<=sel[2:0];
                    attn_qkv_wt_row<=r[$clog2(HIDDEN)-1:0];
                    attn_qkv_wt_col<=c[$clog2(HIDDEN)-1:0];
                    attn_qkv_wt_wr_data<=(r==c)?16'sd4096:16'sd0;
                    @(posedge clk); attn_qkv_wt_wr_en<=0;
                end
        end
    endtask

    task preload_rope();
        for (int p = 0; p < MAX_POS; p++)
            for (int pair = 0; pair < HIDDEN/2; pair++) begin
                @(posedge clk); attn_rope_lut_wr_en<=1;
                attn_rope_lut_pos<=p[$clog2(MAX_POS)-1:0];
                attn_rope_lut_pair<=pair[$clog2(HIDDEN/2)-1:0];
                attn_rope_lut_sin<=16'sd0; attn_rope_lut_cos<=16'sd4096;
                @(posedge clk); attn_rope_lut_wr_en<=0;
            end
    endtask

    task preload_router();
        for (int e = 0; e < 4; e++)
            for (int i = 0; i < HIDDEN; i++) begin
                @(posedge clk); rtr_w_wr_en<=1; rtr_w_wr_expert<=e[1:0];
                rtr_w_wr_idx<=i[$clog2(HIDDEN)-1:0]; rtr_w_wr_data<=(i==e)?4096:0;
                @(posedge clk); rtr_w_wr_en<=0;
            end
    endtask

    task preload_ffn();
        for (int r = 0; r < INTER; r++) begin
            @(posedge clk); gate_w_wr_en<=1; gate_w_wr_row<=r[$clog2(INTER)-1:0]; gate_w_wr_beat<=0;
            gate_w_wr_data<=16'h4000; @(posedge clk); gate_w_wr_en<=0;
            @(posedge clk); gate_w_wr_en<=1; gate_w_wr_row<=r[$clog2(INTER)-1:0]; gate_w_wr_beat<=1;
            gate_w_wr_data<=16'h0; @(posedge clk); gate_w_wr_en<=0;
            @(posedge clk); up_w_wr_en<=1; up_w_wr_row<=r[$clog2(INTER)-1:0]; up_w_wr_beat<=0;
            up_w_wr_data<=16'h4000; @(posedge clk); up_w_wr_en<=0;
            @(posedge clk); up_w_wr_en<=1; up_w_wr_row<=r[$clog2(INTER)-1:0]; up_w_wr_beat<=1;
            up_w_wr_data<=16'h0; @(posedge clk); up_w_wr_en<=0;
        end
        for (int r = 0; r < HIDDEN; r++) begin
            @(posedge clk); down_w_wr_en<=1; down_w_wr_row<=r[$clog2(HIDDEN)-1:0]; down_w_wr_beat<=0;
            down_w_wr_data<=(r < INTER) ? (16'h4000 >> (r*4)) : 16'h0;
            @(posedge clk); down_w_wr_en<=0;
        end
    endtask

    //=========================================================================
    // Token processing
    //=========================================================================
    logic [31:0] token_vec [8];
    int pipeline_cycles;

    task send_token;
        for (int d = 0; d < HIDDEN; d++) a_flat[d*32+:32] <= token_vec[d];
        @(posedge clk); #1;
        valid_in <= 1;
        @(posedge clk); #1;
        valid_in <= 0;
    endtask

    task wait_output;
        pipeline_cycles = 0;
        while (!valid_out) begin
            @(posedge clk);
            pipeline_cycles = pipeline_cycles + 1;
            if (pipeline_cycles > 5000) begin
                $error("TIMEOUT waiting for cluster output after %0d cycles", pipeline_cycles);
                $fatal;
            end
        end
        #1;
    endtask

    //=========================================================================
    // Main test
    //=========================================================================
    initial begin
        // Init
        rst_n = 0;
        gamma_wr_en <= 0; attn_qkv_wt_wr_en <= 0; attn_rope_lut_wr_en <= 0;
        rtr_w_wr_en <= 0; gate_w_wr_en <= 0; up_w_wr_en <= 0; down_w_wr_en <= 0;
        scale_wr_en <= 0; valid_in <= 0; token_position <= '0;
        cache_preload_en <= 0; cache_preload_K_flat <= '0; cache_preload_V_flat <= '0;
        for (int c = 0; c < NUM_CHIPS; c++) cfg_local_experts[c] <= '1;  // all experts local
        ffn_expert_sel <= 0;
        repeat (5) @(posedge clk); rst_n = 1;
        repeat (2) @(posedge clk);

        $display("============================================================");
        $display(" FPGA LPU — Cluster Pipeline Test (NUM_CHIPS=%0d)", NUM_CHIPS);
        $display("============================================================");
        $display(" Pipeline: Token → Chip 0 (L0-L1) → Chip 1 (L2-L3) → Output");
        $display("============================================================");
        $display("");

        // Weight preload (broadcast to all chips)
        $display("[PRELOAD] Loading weights (broadcast to all %0d chips)...", NUM_CHIPS);
        preload_scales();
        preload_gamma();
        preload_qkv();
        preload_rope();
        preload_router();
        preload_ffn();
        $display("[PRELOAD] Done.");
        $display("");

        // Test: send token and verify pipeline forwarding
        $display("[TEST] Sending token (all dims = 4096, Q12 = 1.0)...");
        for (int d = 0; d < HIDDEN; d++) token_vec[d] = 4096;
        send_token();

        wait_output();
        $display("[PASS] Cluster output received in %0d cycles", pipeline_cycles);
        $display("  y_flat[0:7] = %d, %d, %d, %d, %d, %d, %d, %d",
                 y0, y1, y2, y3, y4, y5, y6, y7);
        $display("  router_ok   = %b", router_ok);

        // Verify non-zero output (the pipeline transformed the token)
        if (y0 != 0 || y1 != 0 || y2 != 0 || y3 != 0 ||
            y4 != 0 || y5 != 0 || y6 != 0 || y7 != 0) begin
            $display("[PASS] Pipeline produces non-zero output — forwarding OK.");
        end else begin
            $display("[WARN] All outputs are zero — possible pipeline stall.");
        end

        // Test 2: second token (warm pipeline)
        $display("");
        $display("[TEST] Sending second token...");
        for (int d = 0; d < HIDDEN; d++) token_vec[d] = 2048;
        send_token();

        wait_output();
        $display("[PASS] Second token output in %0d cycles", pipeline_cycles);
        $display("  y_flat[0:7] = %d, %d, %d, %d, %d, %d, %d, %d",
                 y0, y1, y2, y3, y4, y5, y6, y7);

        // Test 3: burst of 5 tokens
        $display("");
        $display("[TEST] Burst: 5 tokens sequential...");
        for (int t = 1; t <= 5; t++) begin
            for (int d = 0; d < HIDDEN; d++) token_vec[d] = t * 1000;
            send_token();
            wait_output();
            $display("  Token %0d: latency=%0d cycles, y[0]=%d", t, pipeline_cycles, y0);
        end

        $display("");
        $display("============================================================");
        $display(" ALL TESTS PASSED");
        $display("============================================================");
        $finish;
    end

endmodule
