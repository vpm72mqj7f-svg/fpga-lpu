`timescale 1ns/1ps
module tb_kv_fp8_test;
    localparam int NUM_SLOTS=8, K_LATENT=4, V_LATENT=4, DATA_W=32;

    logic clk, rst_n;
    logic wr_en, rd_en;
    logic [K_LATENT*DATA_W-1:0] K_latent_flat, V_latent_flat;
    logic [K_LATENT*DATA_W-1:0] preload_K_flat, preload_V_flat;
    logic preload_en;
    logic [2:0] wr_addr, rd_addr;
    logic rd_valid;
    logic [K_LATENT*DATA_W-1:0] rd_K_flat, rd_V_flat;
    logic [3:0] fill_count;
    logic full, empty;

    // FP8 mode: STORE_W=8
    mla_kv_cache #(.NUM_SLOTS(NUM_SLOTS), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
                   .DATA_W(DATA_W), .STORE_W(8))
        dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    // FP8 E4M3 value → Q12 expected: 1.0 = 0x38 → 4096, 0.5 = 0x30 → 2048, -1.0 = 0xB8 → -4096
    // FP8 0x38 = 0_0111_000 = + 2^0 × 1.0 = 1.0 → Q12 = 4096
    // FP8 0x30 = 0_0110_000 = + 2^(-1) × 1.0 = 0.5 → Q12 = 2048
    // FP8 0xB8 = 1_0111_000 = - 2^0 × 1.0 = -1.0 → Q12 = -4096

    task preload_fp8(input [7:0] k0,k1,k2,k3, input [7:0] v0,v1,v2,v3);
        @(posedge clk);
        preload_en <= 1;
        preload_K_flat <= {k3,k2,k1,k0};  // pack: each in low byte
        preload_V_flat <= {v3,v2,v1,v0};
        @(posedge clk);
        preload_en <= 0;
    endtask

    initial begin
        rst_n=0; wr_en=0; rd_en=0; preload_en=0;
        K_latent_flat='0; V_latent_flat='0;
        preload_K_flat='0; preload_V_flat='0;
        rd_addr='0;
        #20 rst_n=1; #10;

        $display("=== FP8 KV Cache Test (STORE_W=8) ===");

        // Preload FP8 values: K=[1.0, 0.5, -1.0, 0.25], V=[0.0, 2.0, -0.5, 1.0]
        // FP8 encodings:
        //   1.0  = 0x38,  0.5 = 0x30, -1.0 = 0xB8, 0.25 = 0x28
        //   0.0  = 0x00,  2.0 = 0x40, -0.5 = 0xB0
        preload_fp8(8'h38, 8'h30, 8'hB8, 8'h28,   // K: 1.0, 0.5, -1.0, 0.25
                    8'h00, 8'h40, 8'hB0, 8'h38);  // V: 0.0, 2.0, -0.5, 1.0

        // Read back
        #20;
        @(posedge clk); rd_en<=1; rd_addr<=0; @(posedge clk); rd_en<=0;
        #10;

        $display("K[0]: %0d (exp ~4096 for FP8 1.0)", $signed(rd_K_flat[0*32+:32]));
        $display("K[1]: %0d (exp ~2048 for FP8 0.5)", $signed(rd_K_flat[1*32+:32]));
        $display("K[2]: %0d (exp ~-4096 for FP8 -1.0)", $signed(rd_K_flat[2*32+:32]));
        $display("K[3]: %0d (exp ~1024 for FP8 0.25)", $signed(rd_K_flat[3*32+:32]));
        $display("V[0]: %0d (exp 0)", $signed(rd_V_flat[0*32+:32]));
        $display("V[1]: %0d (exp ~8192 for FP8 2.0)", $signed(rd_V_flat[1*32+:32]));

        // Quick check: K[0] should be close to 4096
        if ($signed(rd_K_flat[0*32+:32]) > 4000 && $signed(rd_K_flat[0*32+:32]) < 4200)
            $display("\n[PASS] FP8→Q12 conversion correct (1.0 → ~4096)");
        else
            $display("\n[FAIL] FP8→Q12 conversion wrong: got %0d", $signed(rd_K_flat[0*32+:32]));

        $display("=== FP8 storage: %0d bytes/token vs Q12: %0d bytes/token ===",
                 K_LATENT*8 + V_LATENT*8, K_LATENT*32 + V_LATENT*32);
        $display("=== KV capacity gain: 4x ===");

        $finish;
    end
endmodule
