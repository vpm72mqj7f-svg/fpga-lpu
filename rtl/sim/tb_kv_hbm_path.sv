`timescale 1ns/1ps
//=============================================================================
// tb_kv_hbm_path.sv — HBM → mla_kv_cache preload path verification
//
// Core path: HBM (behavioral) → kv_hbm_preloader → mla_kv_cache (BRAM)
// DMA bridge path tested separately (tb_kv_dma_bridge: 5/5 PASS).
//=============================================================================

module tb_kv_hbm_path;
    localparam int K_LATENT=4, V_LATENT=4, DATA_W=32;
    localparam int NUM_SLOTS=16;
    localparam int KV_ENTRY_BYTES = K_LATENT + V_LATENT;  // 8 bytes FP8

    logic clk, rst_n;

    // HBM behavioral model (256-bit wide, 32-bit word addressed)
    logic [31:0] sim_hbm [0:1023];

    // kv_hbm_preloader
    logic        pl_start, pl_done, pl_busy;
    logic [31:0] pl_hbm_base;
    logic [15:0] pl_num_entries;
    logic [31:0] pl_hbm_rd_addr;
    logic        pl_hbm_rd_en;
    logic [255:0] pl_hbm_rd_data;

    // mla_kv_cache (FP8 mode)
    logic cache_wr_en;
    logic [K_LATENT*DATA_W-1:0] cache_K_in, cache_V_in;
    logic [$clog2(NUM_SLOTS)-1:0] cache_wr_addr;
    logic cache_preload_en;
    logic [K_LATENT*DATA_W-1:0] cache_preload_K, cache_preload_V;
    logic cache_rd_en;
    logic [$clog2(NUM_SLOTS)-1:0] cache_rd_addr;
    logic cache_rd_valid;
    logic [K_LATENT*DATA_W-1:0] cache_rd_K, cache_rd_V;
    logic [$clog2(NUM_SLOTS+1)-1:0] cache_fill;
    logic cache_full, cache_empty;

    kv_hbm_preloader #(.K_LATENT(K_LATENT), .V_LATENT(V_LATENT), .DATA_W(DATA_W),
                       .KV_ENTRY_BYTES(KV_ENTRY_BYTES))
    u_preloader (
        .clk, .rst_n,
        .start(pl_start), .hbm_base_addr(pl_hbm_base),
        .num_entries(pl_num_entries), .done(pl_done), .busy(pl_busy),
        .hbm_rd_addr(pl_hbm_rd_addr), .hbm_rd_en(pl_hbm_rd_en),
        .hbm_rd_data(pl_hbm_rd_data),
        .preload_en(cache_preload_en),
        .preload_K_flat(cache_preload_K),
        .preload_V_flat(cache_preload_V)
    );

    mla_kv_cache #(.NUM_SLOTS(NUM_SLOTS), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
                   .DATA_W(DATA_W), .STORE_W(8))
    u_cache (.clk, .rst_n,
        .wr_en(cache_wr_en), .K_latent_flat(cache_K_in),
        .V_latent_flat(cache_V_in), .wr_addr(cache_wr_addr),
        .preload_en(cache_preload_en),
        .preload_K_flat(cache_preload_K),
        .preload_V_flat(cache_preload_V),
        .rd_en(cache_rd_en), .rd_addr(cache_rd_addr),
        .rd_valid(cache_rd_valid), .rd_K_flat(cache_rd_K),
        .rd_V_flat(cache_rd_V),
        .fill_count(cache_fill), .full(cache_full), .empty(cache_empty));

    initial clk = 0;
    always #5 clk = ~clk;

    // HBM read path
    always_comb begin
        pl_hbm_rd_data = '0;
        if (pl_hbm_rd_en) begin
            pl_hbm_rd_data = { sim_hbm[pl_hbm_rd_addr[31:2]+7],
                               sim_hbm[pl_hbm_rd_addr[31:2]+6],
                               sim_hbm[pl_hbm_rd_addr[31:2]+5],
                               sim_hbm[pl_hbm_rd_addr[31:2]+4],
                               sim_hbm[pl_hbm_rd_addr[31:2]+3],
                               sim_hbm[pl_hbm_rd_addr[31:2]+2],
                               sim_hbm[pl_hbm_rd_addr[31:2]+1],
                               sim_hbm[pl_hbm_rd_addr[31:2]+0] };
        end
    end

    // Init HBM with known FP8 KV data (bytes packed into 32-bit words)
    task init_hbm_fp8(input [7:0] k0,k1,k2,k3, v0,v1,v2,v3,
                      input int base_word);
        // K bytes → word 0, V bytes → word 1
        sim_hbm[base_word + 0] = {k3, k2, k1, k0};
        sim_hbm[base_word + 1] = {v3, v2, v1, v0};
    endtask

    integer pass, fail;
    initial begin
        rst_n=0; pl_start=0; pl_hbm_base=0; pl_num_entries=0;
        cache_wr_en=0; cache_rd_en=0; cache_rd_addr=0; pass=0; fail=0;

        // Init HBM: FP8 K=[1.0,0.5,-1.0,2.0], V=[2.5,-0.5,0.0,1.0]
        // FP8: 1.0=0x38, 0.5=0x30, -1.0=0xB8, 2.0=0x40, 2.5=0x42, -0.5=0xB0
        for (int i = 0; i < 1024; i++) sim_hbm[i] = '0;
        init_hbm_fp8(8'h38, 8'h30, 8'hB8, 8'h40,   // K
                     8'h42, 8'hB0, 8'h00, 8'h38,   // V
                     0);  // HBM base address = 0

        #20 rst_n=1; #10;

        $display("============================================================");
        $display(" tb_kv_hbm_path — HBM → mla_kv_cache preload");
        $display("============================================================");

        // Preload 1 entry from HBM to BRAM
        pl_hbm_base <= 32'd0;
        pl_num_entries <= 1;
        @(posedge clk); pl_start <= 1;
        @(posedge clk); pl_start <= 0;

        while (!pl_done) @(posedge clk);
        $display("\nPreload complete. fill_count=%0d", cache_fill);

        // Read back and verify
        @(posedge clk); cache_rd_en<=1; cache_rd_addr<=0;
        @(posedge clk); cache_rd_en<=0;
        @(posedge clk);  // rd_valid

        $display("K[0]=%0d (FP8 1.0 →Q12 exp 4096)", $signed(cache_rd_K[0*32+:32]));
        $display("K[1]=%0d (FP8 0.5 →Q12 exp 2048)", $signed(cache_rd_K[1*32+:32]));
        $display("K[2]=%0d (FP8 -1.0→Q12 exp -4096)", $signed(cache_rd_K[2*32+:32]));
        $display("K[3]=%0d (FP8 2.0 →Q12 exp 8192)", $signed(cache_rd_K[3*32+:32]));
        $display("V[0]=%0d (FP8 2.5 →Q12 exp 10240)", $signed(cache_rd_V[0*32+:32]));
        $display("V[1]=%0d (FP8 -0.5→Q12 exp -2048)", $signed(cache_rd_V[1*32+:32]));

        // Verify
        if ($signed(cache_rd_K[0*32+:32]) > 4080 && $signed(cache_rd_K[0*32+:32]) < 4112)
            begin $display("[PASS] K[0] 1.0→Q12"); pass=pass+1; end
        else begin $display("[FAIL] K[0]=%0d", $signed(cache_rd_K[0*32+:32])); fail=fail+1; end

        if ($signed(cache_rd_K[1*32+:32]) > 2030 && $signed(cache_rd_K[1*32+:32]) < 2070)
            begin $display("[PASS] K[1] 0.5→Q12"); pass=pass+1; end
        else begin $display("[FAIL] K[1]=%0d", $signed(cache_rd_K[1*32+:32])); fail=fail+1; end

        if ($signed(cache_rd_K[2*32+:32]) < -4080)
            begin $display("[PASS] K[2] -1.0→Q12"); pass=pass+1; end
        else begin $display("[FAIL] K[2]=%0d", $signed(cache_rd_K[2*32+:32])); fail=fail+1; end

        if ($signed(cache_rd_K[3*32+:32]) > 8170 && $signed(cache_rd_K[3*32+:32]) < 8220)
            begin $display("[PASS] K[3] 2.0→Q12"); pass=pass+1; end
        else begin $display("[FAIL] K[3]=%0d", $signed(cache_rd_K[3*32+:32])); fail=fail+1; end

        if ($signed(cache_rd_V[0*32+:32]) > 10220 && $signed(cache_rd_V[0*32+:32]) < 10260)
            begin $display("[PASS] V[0] 2.5→Q12"); pass=pass+1; end
        else begin $display("[FAIL] V[0]=%0d", $signed(cache_rd_V[0*32+:32])); fail=fail+1; end

        $display("\n============================================================");
        if (fail == 0)
            $display(" PASS tb_kv_hbm_path (%0d/%0d) — HBM→BRAM path verified", pass, pass+fail);
        else
            $display(" FAIL tb_kv_hbm_path");
        $finish;
    end
endmodule
