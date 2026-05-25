`timescale 1ns/1ps

module tb_mhc_mixer;
    localparam int HIDDEN  = 8;
    localparam int N_HW    = 4;
    localparam int COEFF_W = 16;
    localparam int DATA_W  = 32;
    localparam int LAYER_W = HIDDEN * DATA_W;
    localparam int HW_W    = HIDDEN * N_HW * DATA_W;

    localparam int Q12_ONE  = 4096;
    localparam int Q12_ZERO = 0;
    localparam int Q12_HALF = 2048;

    logic clk, rst_n;
    logic in_valid, in_ready;
    logic [LAYER_W-1:0] layer_in_flat, residual_flat;
    logic out_valid;
    logic [HW_W-1:0] highway_flat;
    logic coeff_wr_en;
    logic [$clog2(N_HW)-1:0] coeff_hw_id;
    logic [1:0] coeff_col;
    logic signed [COEFF_W-1:0] coeff_wr_data;

    mhc_mixer #(.HIDDEN(HIDDEN), .N_HW(N_HW), .COEFF_W(COEFF_W), .DATA_W(DATA_W))
        dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task wait_cycles(input int n);
        for (int i = 0; i < n; i++) @(posedge clk);
    endtask

    function [LAYER_W-1:0] make_vec(input int base);
        reg [LAYER_W-1:0] v;
        for (int d = 0; d < HIDDEN; d++) v[d*DATA_W +: DATA_W] = base + d;
        make_vec = v;
    endfunction

    integer pass_count, fail_count;
    integer got_val, exp_val;

    initial begin
        rst_n = 0; in_valid = 0; layer_in_flat = '0; residual_flat = '0;
        coeff_wr_en = 0; coeff_hw_id = '0; coeff_col = '0; coeff_wr_data = '0;
        pass_count = 0; fail_count = 0;

        wait_cycles(4);
        rst_n = 1;
        wait_cycles(2);

        // ============================================
        // Test 1: Identity passthrough (layer=0, resid=1.0)
        // ============================================
        $display("Test 1: Identity mixing (passthrough)");
        for (int h = 0; h < N_HW; h++) begin
            @(posedge clk);
            coeff_wr_en = 1; coeff_hw_id = h; coeff_col = 0; coeff_wr_data = Q12_ZERO;
            @(posedge clk);
            coeff_wr_en = 0;
            @(posedge clk);
            coeff_wr_en = 1; coeff_hw_id = h; coeff_col = 1; coeff_wr_data = Q12_ONE;
            @(posedge clk);
            coeff_wr_en = 0;
        end

        @(posedge clk);
        in_valid = 1; layer_in_flat = make_vec(100); residual_flat = make_vec(200);
        @(posedge clk);
        in_valid = 0;

        // Wait for out_valid
        for (int cyc = 0; cyc < 30; cyc++) begin
            @(posedge clk);
            if (out_valid) begin
                for (int h = 0; h < N_HW; h++) begin
                    for (int d = 0; d < HIDDEN; d++) begin
                        got_val = highway_flat[(h*HIDDEN + d)*DATA_W +: DATA_W];
                        if (got_val !== (200 + d)) begin
                            $error("  [FAIL] hw=%0d d=%0d: got %0d exp %0d", h, d, got_val, 200+d);
                            fail_count = fail_count + 1;
                        end
                    end
                end
                if (fail_count == 0) begin
                    $display("  [ OK ] Test 1");
                    pass_count = pass_count + 1;
                end
            end
        end

        wait_cycles(4);

        // ============================================
        // Test 2: Half-half mixing
        // ============================================
        $display("Test 2: 50/50 mixing");
        for (int h = 0; h < N_HW; h++) begin
            @(posedge clk);
            coeff_wr_en = 1; coeff_hw_id = h; coeff_col = 0; coeff_wr_data = Q12_HALF;
            @(posedge clk);
            coeff_wr_en = 0;
            @(posedge clk);
            coeff_wr_en = 1; coeff_hw_id = h; coeff_col = 1; coeff_wr_data = Q12_HALF;
            @(posedge clk);
            coeff_wr_en = 0;
        end

        @(posedge clk);
        in_valid = 1; layer_in_flat = make_vec(100); residual_flat = make_vec(200);
        @(posedge clk);
        in_valid = 0;

        for (int cyc = 0; cyc < 30; cyc++) begin
            @(posedge clk);
            if (out_valid) begin
                for (int h = 0; h < N_HW; h++) begin
                    for (int d = 0; d < HIDDEN; d++) begin
                        got_val = highway_flat[(h*HIDDEN + d)*DATA_W +: DATA_W];
                        if (got_val < (150+d-1) || got_val > (150+d+1)) begin
                            $error("  [FAIL] hw=%0d d=%0d: got %0d exp ~%0d", h, d, got_val, 150+d);
                            fail_count = fail_count + 1;
                        end
                    end
                end
                if (fail_count == 0) begin
                    $display("  [ OK ] Test 2");
                    pass_count = pass_count + 1;
                end
            end
        end

        wait_cycles(4);

        // ============================================
        // Test 3: Per-highway different coefficients
        // ============================================
        $display("Test 3: Per-highway coefficient variation");
        // h0: (1.0, 0) — layer only
        @(posedge clk);
        coeff_wr_en = 1; coeff_hw_id = 0; coeff_col = 0; coeff_wr_data = Q12_ONE;
        @(posedge clk); coeff_wr_en = 0;
        @(posedge clk);
        coeff_wr_en = 1; coeff_hw_id = 0; coeff_col = 1; coeff_wr_data = Q12_ZERO;
        @(posedge clk); coeff_wr_en = 0;
        // h1: (0, 1.0) — residual only
        @(posedge clk);
        coeff_wr_en = 1; coeff_hw_id = 1; coeff_col = 0; coeff_wr_data = Q12_ZERO;
        @(posedge clk); coeff_wr_en = 0;
        @(posedge clk);
        coeff_wr_en = 1; coeff_hw_id = 1; coeff_col = 1; coeff_wr_data = Q12_ONE;
        @(posedge clk); coeff_wr_en = 0;
        // h2: (0.5, 0.5) — average
        @(posedge clk);
        coeff_wr_en = 1; coeff_hw_id = 2; coeff_col = 0; coeff_wr_data = Q12_HALF;
        @(posedge clk); coeff_wr_en = 0;
        @(posedge clk);
        coeff_wr_en = 1; coeff_hw_id = 2; coeff_col = 1; coeff_wr_data = Q12_HALF;
        @(posedge clk); coeff_wr_en = 0;
        // h3: (0.25, 0.75) — resid-biased
        @(posedge clk);
        coeff_wr_en = 1; coeff_hw_id = 3; coeff_col = 0; coeff_wr_data = 1024;
        @(posedge clk); coeff_wr_en = 0;
        @(posedge clk);
        coeff_wr_en = 1; coeff_hw_id = 3; coeff_col = 1; coeff_wr_data = 3072;
        @(posedge clk); coeff_wr_en = 0;

        @(posedge clk);
        in_valid = 1; layer_in_flat = make_vec(300); residual_flat = make_vec(400);
        @(posedge clk);
        in_valid = 0;

        for (int cyc = 0; cyc < 30; cyc++) begin
            @(posedge clk);
            if (out_valid) begin
                // h0: layer only = 300+d
                for (int d = 0; d < HIDDEN; d++) begin
                    got_val = highway_flat[(0*HIDDEN + d)*DATA_W +: DATA_W];
                    if (got_val !== 300 + d) begin
                        $error("  [FAIL] h0 d=%0d: got %0d exp %0d", d, got_val, 300+d);
                        fail_count = fail_count + 1;
                    end
                end
                // h1: resid only = 400+d
                for (int d = 0; d < HIDDEN; d++) begin
                    got_val = highway_flat[(1*HIDDEN + d)*DATA_W +: DATA_W];
                    if (got_val !== 400 + d) begin
                        $error("  [FAIL] h1 d=%0d: got %0d exp %0d", d, got_val, 400+d);
                        fail_count = fail_count + 1;
                    end
                end
                // h2: avg ~350+d
                for (int d = 0; d < HIDDEN; d++) begin
                    got_val = highway_flat[(2*HIDDEN + d)*DATA_W +: DATA_W];
                    if (got_val < 349+d || got_val > 351+d) begin
                        $error("  [FAIL] h2 d=%0d: got %0d exp ~%0d", d, got_val, 350+d);
                        fail_count = fail_count + 1;
                    end
                end
                // h3: 0.25*300 + 0.75*400 = 75 + 300 = 375+d
                for (int d = 0; d < HIDDEN; d++) begin
                    got_val = highway_flat[(3*HIDDEN + d)*DATA_W +: DATA_W];
                    if (got_val < 374+d || got_val > 376+d) begin
                        $error("  [FAIL] h3 d=%0d: got %0d exp ~%0d", d, got_val, 375+d);
                        fail_count = fail_count + 1;
                    end
                end
                if (fail_count == 0) begin
                    $display("  [ OK ] Test 3");
                    pass_count = pass_count + 1;
                end
            end
        end

        $display("==============================");
        if (fail_count == 0)
            $display("PASS tb_mhc_mixer (%0d/3 tests)", pass_count);
        else
            $display("FAIL tb_mhc_mixer (%0d pass, %0d fail)", pass_count, fail_count);
        $finish;
    end

endmodule
