`timescale 1ns/1ps
module tb_kv_fp8_v2;
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

    mla_kv_cache #(.NUM_SLOTS(NUM_SLOTS), .K_LATENT(K_LATENT), .V_LATENT(V_LATENT),
                   .DATA_W(DATA_W), .STORE_W(8))
        dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    // FP8 encodings: 1.0=0x38, 0.5=0x30, -1.0=0xB8, 0.25=0x28, 2.0=0x40, -0.5=0xB0, 0.0=0x00

    task preload_fp8(input [7:0] k0,k1,k2,k3, input [7:0] v0,v1,v2,v3);
        @(posedge clk);
        preload_en <= 1;
        // Each FP8 byte in its own 32-bit lane (low byte)
        preload_K_flat <= { {24'b0, k3}, {24'b0, k2}, {24'b0, k1}, {24'b0, k0} };
        preload_V_flat <= { {24'b0, v3}, {24'b0, v2}, {24'b0, v1}, {24'b0, v0} };
        @(posedge clk);
        preload_en <= 0;
    endtask

    integer pass, fail;
    initial begin
        rst_n=0; wr_en=0; rd_en=0; preload_en=0;
        K_latent_flat='0; V_latent_flat='0;
        preload_K_flat='0; preload_V_flat='0;
        rd_addr='0; pass=0; fail=0;
        #20 rst_n=1; #10;

        $display("=== FP8 KV Cache (STORE_W=8) ===");

        // K=[1.0, 0.5, -1.0, 0.25], V=[2.0, -0.5, 0.0, 1.0]
        preload_fp8(8'h38, 8'h30, 8'hB8, 8'h28,
                    8'h40, 8'hB0, 8'h00, 8'h38);

        #20;
        @(posedge clk); rd_en<=1; rd_addr<=0; @(posedge clk); rd_en<=0;
        #10;

        $display("K[0]=%0d (exp 4096)", $signed(rd_K_flat[0*32+:32]));
        $display("K[1]=%0d (exp 2048)", $signed(rd_K_flat[1*32+:32]));
        $display("K[2]=%0d (exp -4096)", $signed(rd_K_flat[2*32+:32]));
        $display("K[3]=%0d (exp 1024)", $signed(rd_K_flat[3*32+:32]));
        $display("V[0]=%0d (exp 8192)", $signed(rd_V_flat[0*32+:32]));
        $display("V[1]=%0d (exp -2048)", $signed(rd_V_flat[1*32+:32]));

        // Tolerance: FP8→Q12 has LSB rounding error, ±16 OK
        if ($signed(rd_K_flat[0*32+:32]) > 4080 && $signed(rd_K_flat[0*32+:32]) < 4112) begin
            $display("[PASS] K[0] 1.0→Q12 correct"); pass=pass+1;
        end else begin $display("[FAIL] K[0]"); fail=fail+1; end

        if ($signed(rd_K_flat[1*32+:32]) > 2030 && $signed(rd_K_flat[1*32+:32]) < 2070) begin
            $display("[PASS] K[1] 0.5→Q12 correct"); pass=pass+1;
        end else begin $display("[FAIL] K[1]=%0d", $signed(rd_K_flat[1*32+:32])); fail=fail+1; end

        if ($signed(rd_K_flat[2*32+:32]) < -4080 && $signed(rd_K_flat[2*32+:32]) > -4112) begin
            $display("[PASS] K[2] -1.0→Q12 correct"); pass=pass+1;
        end else begin $display("[FAIL] K[2]=%0d", $signed(rd_K_flat[2*32+:32])); fail=fail+1; end

        if ($signed(rd_K_flat[3*32+:32]) > 1010 && $signed(rd_K_flat[3*32+:32]) < 1040) begin
            $display("[PASS] K[3] 0.25→Q12 correct"); pass=pass+1;
        end else begin $display("[FAIL] K[3]=%0d", $signed(rd_K_flat[3*32+:32])); fail=fail+1; end

        if ($signed(rd_V_flat[0*32+:32]) > 8170 && $signed(rd_V_flat[0*32+:32]) < 8220) begin
            $display("[PASS] V[0] 2.0→Q12 correct"); pass=pass+1;
        end else begin $display("[FAIL] V[0]=%0d", $signed(rd_V_flat[0*32+:32])); fail=fail+1; end

        if ($signed(rd_V_flat[1*32+:32]) < -2030 && $signed(rd_V_flat[1*32+:32]) > -2070) begin
            $display("[PASS] V[1] -0.5→Q12 correct"); pass=pass+1;
        end else begin $display("[FAIL] V[1]=%0d", $signed(rd_V_flat[1*32+:32])); fail=fail+1; end

        $display("\n=== FP8→Q12 LUT: %0d/%0d PASS ===", pass, pass+fail);
        $display("=== Storage: %0d bytes/token (vs Q12: %0d) ===",
                 K_LATENT*8+V_LATENT*8, K_LATENT*32+V_LATENT*32);
        $finish;
    end
endmodule
