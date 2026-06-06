//=============================================================================
// tb_mla_qkv_prod.sv — Production-parameter MLA test (FPGA_LPU_PRODUCTION)
//
// Compile: iverilog -g2012 -DFPGA_LPU_PRODUCTION -I../include -o tb_mla_qkv_prod.vvp
//          ../attention/mla_rope.sv ../attention/mla_kv_cache.sv tb_mla_qkv_prod.sv
// Run: vvp tb_mla_qkv_prod.vvp
//
// FINDINGS (T2.5):
//   1. mla_qkv_proj.sv — dot-product hardcoded for HIDDEN=8 (fixed indices 0..7).
//      NOT parameterized. Needs systolic/MAC redesign for production.
//   2. mla_rope.sv — properly parameterized. Works with HIDDEN=7168, N_PAIRS=3584.
//      Pipeline takes ~2*N_PAIRS cycles per vector. Icarus handles this.
//   3. mla_kv_cache.sv — properly parameterized. Works with NUM_SLOTS=4096,
//      K_LATENT=512, V_LATENT=512. Uses BRAM-style unpacked arrays.
//   4. tb_mla_qkv.sv — overrides HIDDEN=8 via localparam, defeating production test.
//   5. Port widths: when K_LATENT changes from 4 to 512, K/V flat buses grow
//      from 128b to 16384b. Testbench must match.
//=============================================================================

`timescale 1ns/1ps

module tb_mla_qkv_prod;
`ifdef FPGA_LPU_PRODUCTION
    localparam int HIDDEN   = 7168;
    localparam int K_LATENT = 512;
    localparam int V_LATENT = 512;
    localparam int NUM_SLOTS = 4096;
`else
    localparam int HIDDEN   = 8;
    localparam int K_LATENT = 4;
    localparam int V_LATENT = 4;
    localparam int NUM_SLOTS = 64;
`endif
    localparam int MAX_POS  = 64;
    localparam int WEIGHT_W = 16;
    localparam int DATA_W   = 32;
    localparam int N_PAIRS  = HIDDEN / 2;
    localparam int Q12_ONE  = 4096;

    // ---- mla_rope signals ----
    logic clk, rst_n;
    logic rope_in_valid, rope_in_ready;
    logic [HIDDEN*DATA_W-1:0] rope_in_flat;
    logic [$clog2(MAX_POS)-1:0] rope_pos;
    logic lut_wr_en;
    logic [$clog2(N_PAIRS)-1:0] lut_pair;
    logic signed [WEIGHT_W-1:0] lut_sin_data, lut_cos_data;
    logic rope_out_valid;
    logic [HIDDEN*DATA_W-1:0] rope_out_flat;

    // ---- mla_kv_cache signals ----
    logic cache_wr_en;
    logic [K_LATENT*DATA_W-1:0] cache_K_in, cache_K_out;
    logic [V_LATENT*DATA_W-1:0] cache_V_in, cache_V_out;
    logic [$clog2(NUM_SLOTS)-1:0] cache_wr_addr, cache_rd_addr;
    logic cache_rd_en, cache_rd_valid;
    logic [$clog2(NUM_SLOTS+1)-1:0] cache_fill_count;
    logic cache_full, cache_empty;

    // DUTs
    mla_rope #(
        .HIDDEN(HIDDEN), .MAX_POS(MAX_POS),
        .COEFF_W(WEIGHT_W), .DATA_W(DATA_W)
    ) u_rope (
        .clk(clk), .rst_n(rst_n),
        .in_valid(rope_in_valid), .in_ready(rope_in_ready),
        .vec_flat(rope_in_flat), .pos(rope_pos),
        .lut_wr_en(lut_wr_en), .lut_pos(rope_pos), .lut_pair(lut_pair),
        .lut_sin_data(lut_sin_data), .lut_cos_data(lut_cos_data),
        .out_valid(rope_out_valid), .rot_flat(rope_out_flat)
    );

    mla_kv_cache #(
        .NUM_SLOTS(NUM_SLOTS), .K_LATENT(K_LATENT),
        .V_LATENT(V_LATENT), .DATA_W(DATA_W)
    ) u_cache (
        .clk(clk), .rst_n(rst_n),
        .wr_en(cache_wr_en),
        .K_latent_flat(cache_K_in), .V_latent_flat(cache_V_in),
        .wr_addr(cache_wr_addr),
        .preload_en(1'b0),
        .preload_K_flat('0),
        .preload_V_flat('0),
        .rd_en(cache_rd_en), .rd_addr(cache_rd_addr),
        .rd_valid(cache_rd_valid),
        .rd_K_flat(cache_K_out), .rd_V_flat(cache_V_out),
        .fill_count(cache_fill_count),
        .full(cache_full), .empty(cache_empty)
    );

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    task wait_cycles(input int n);
        integer i;
        i = 0;
        while (i < n) begin
            @(posedge clk);
            i = i + 1;
        end
    endtask

    integer pass_count, fail_count;
    integer cyc;

    // =========================================================================
    // Main
    // =========================================================================
    initial begin
        rst_n = 0;
        rope_in_valid = 0; rope_in_flat = '0; rope_pos = '0;
        lut_wr_en = 0; lut_pair = '0;
        lut_sin_data = '0; lut_cos_data = '0;
        cache_wr_en = 0; cache_K_in = '0; cache_V_in = '0;
        cache_rd_en = 0; cache_rd_addr = '0;
        pass_count = 0; fail_count = 0;

        wait_cycles(4);
        rst_n = 1;
        wait_cycles(2);

        $display("==============================================");
        $display("tb_mla_qkv_prod");
        $display("  HIDDEN=%0d  K_LATENT=%0d  V_LATENT=%0d",
                 HIDDEN, K_LATENT, V_LATENT);
        $display("  NUM_SLOTS=%0d  MAX_POS=%0d  N_PAIRS=%0d",
                 NUM_SLOTS, MAX_POS, N_PAIRS);
        $display("==============================================");

        // =====================================================================
        // Test P1: RoPE with production dimensions
        // Load LUT entry and verify rotation on a sparse vector.
        // Only the first few pairs are non-zero; all others are identity.
        // =====================================================================
        $display("Test P1: RoPE at production scale (first pair only)");

        // Load pos=1, pair=0: cos=0, sin=Q12_ONE (90-degree rotation)
        @(posedge clk);
        lut_wr_en = 1; rope_pos = 1; lut_pair = 0;
        lut_cos_data = 0; lut_sin_data = Q12_ONE;
        @(posedge clk); lut_wr_en = 0;

        // Send sparse vector: only dim0=10, dim1=11 are non-zero
        @(posedge clk);
        rope_in_valid = 1;
        rope_pos = 1;
        rope_in_flat = '0;
        rope_in_flat[0*DATA_W +: DATA_W] = 32'd10;
        rope_in_flat[1*DATA_W +: DATA_W] = 32'd11;
        @(posedge clk);
        rope_in_valid = 0;

        // Wait for output. With N_PAIRS=3584, each pair takes 2 cycles.
`ifdef FPGA_LPU_PRODUCTION
        $display("  Waiting up to 15000 cycles for RoPE pipeline...");
`endif
        cyc = 0;
        while (cyc < 15000 && !rope_out_valid) begin
            @(posedge clk);
            cyc = cyc + 1;
        end

        if (rope_out_valid) begin
            // Verify pair 0: (10,11) 90-deg -> (-11, 10)
            if (rope_out_flat[0*DATA_W +: DATA_W] !== $unsigned(-11)) begin
                $error("  [FAIL] P1 RoPE dim0: got %0d exp -11",
                       rope_out_flat[0*DATA_W +: DATA_W]);
                fail_count = fail_count + 1;
            end
            if (rope_out_flat[1*DATA_W +: DATA_W] !== 32'd10) begin
                $error("  [FAIL] P1 RoPE dim1: got %0d exp 10",
                       rope_out_flat[1*DATA_W +: DATA_W]);
                fail_count = fail_count + 1;
            end
            if (fail_count == 0) begin
                $display("  [ OK ] Test P1: RoPE 90-deg at production scale");
                pass_count = pass_count + 1;
            end
        end else begin
            $error("  [FAIL] P1: RoPE timeout after %0d cycles", cyc);
            fail_count = fail_count + 1;
        end

        wait_cycles(2);

        // =====================================================================
        // Test P2: KV Cache at production scale
        // Write a few entries and read them back.
        // NUM_SLOTS=4096, K_LATENT=512, V_LATENT=512.
        // =====================================================================
        $display("Test P2: KV Cache write/read at production scale");

        // Write one entry
        // Capture wr_addr BEFORE asserting wr_en (wr_addr == wr_ptr)
        @(posedge clk);
        cache_rd_addr = cache_wr_addr;  // Capture write address
        cache_wr_en = 1;
        cache_K_in = '0;
        cache_V_in = '0;
        cache_K_in[0*DATA_W +: DATA_W] = 32'd100;
        cache_K_in[1*DATA_W +: DATA_W] = 32'd101;
        cache_V_in[0*DATA_W +: DATA_W] = 32'd200;
        cache_V_in[1*DATA_W +: DATA_W] = 32'd201;
        @(posedge clk);
        cache_wr_en = 0;

        $display("  Wrote to slot %0d: K[0]=100 K[1]=101", cache_rd_addr);

        wait_cycles(2);

        // Read back from the captured address
        @(posedge clk);
        cache_rd_en = 1;
        // cache_rd_addr already holds the correct address
        @(posedge clk);
        cache_rd_en = 0;

        if (!cache_rd_valid) begin
            $error("  [FAIL] P2: rd_valid not asserted for slot %0d", cache_rd_addr);
            fail_count = fail_count + 1;
        end else begin
            if (cache_K_out[0*DATA_W +: DATA_W] !== 32'd100) begin
                $error("  [FAIL] P2: K[0] mismatch: got %0d exp 100",
                       cache_K_out[0*DATA_W +: DATA_W]);
                fail_count = fail_count + 1;
            end
            if (cache_V_out[0*DATA_W +: DATA_W] !== 32'd200) begin
                $error("  [FAIL] P2: V[0] mismatch: got %0d exp 200",
                       cache_V_out[0*DATA_W +: DATA_W]);
                fail_count = fail_count + 1;
            end
        end

        if (fail_count == 0) begin
            $display("  [ OK ] Test P2: KV cache write/read at production scale");
            pass_count = pass_count + 1;
        end

        // Check fill count
        if (cache_fill_count !== 1) begin
            $error("  [FAIL] P2: fill_count=%0d exp 1", cache_fill_count);
            fail_count = fail_count + 1;
        end

        wait_cycles(2);

        // =====================================================================
        // Summary
        // =====================================================================
        $display("==============================");
        if (fail_count == 0) begin
            $display("PASS tb_mla_qkv_prod (%0d tests)", pass_count);
        end else begin
            $display("FAIL tb_mla_qkv_prod (%0d pass, %0d fail)",
                     pass_count, fail_count);
        end
        $finish;
    end

endmodule
