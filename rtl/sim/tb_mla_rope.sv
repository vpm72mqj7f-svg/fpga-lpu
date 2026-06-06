//=============================================================================
// tb_mla_rope.sv — Standalone unit test for mla_rope
//
// Tests:
//   Test 1: Identity (pos=0, cos=1, sin=0 — default LUT) → vector unchanged
//   Test 2: 90-degree rotation (pair 0: cos=0, sin=1) → (x,y) → (-y,x)
//   Test 3: 180-degree rotation (pair 0: cos=-1, sin=0) → (x,y) → (-x,-y)
//   Test 4: 45-degree rotation (pair 0: cos=sin≈0.7071) → both rotated
//   Test 5: Random angle on all pairs (pos=5) → all pairs rotated
//=============================================================================

`timescale 1ns/1ps

module tb_mla_rope;
    localparam int HIDDEN   = 8;
    localparam int MAX_POS  = 64;
    localparam int COEFF_W  = 16;
    localparam int DATA_W   = 32;
    localparam int N_PAIRS  = HIDDEN / 2;

    localparam int Q12_ONE  = 4096;
    localparam int Q12_ZERO = 0;

    // DUT signals
    logic clk, rst_n;
    logic in_valid, in_ready;
    logic [HIDDEN*DATA_W-1:0] vec_flat;
    logic [$clog2(MAX_POS)-1:0] pos;
    logic lut_wr_en;
    logic [$clog2(MAX_POS)-1:0] lut_pos;
    logic [$clog2(HIDDEN/2)-1:0]  lut_pair;
    logic signed [COEFF_W-1:0] lut_sin_data, lut_cos_data;
    logic out_valid;
    logic [HIDDEN*DATA_W-1:0] rot_flat;

    mla_rope #(
        .HIDDEN(HIDDEN), .MAX_POS(MAX_POS),
        .COEFF_W(COEFF_W), .DATA_W(DATA_W)
    ) u_rope (.*);

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    // Helpers
    task wait_cycles(input int n);
        for (int i = 0; i < n; i++) @(posedge clk);
    endtask

    // Build a flat vector from an array of per-dimension values
    function automatic [HIDDEN*DATA_W-1:0] build_vec(
        input int v0, v1, v2, v3, v4, v5, v6, v7
    );
        reg [HIDDEN*DATA_W-1:0] result;
        result = {v7[DATA_W-1:0], v6[DATA_W-1:0], v5[DATA_W-1:0], v4[DATA_W-1:0],
                  v3[DATA_W-1:0], v2[DATA_W-1:0], v1[DATA_W-1:0], v0[DATA_W-1:0]};
        build_vec = result;
    endfunction

    // Build sequential vector: [base, base+1, ..., base+HIDDEN-1]
    function automatic [HIDDEN*DATA_W-1:0] make_vec(input int base);
        reg [HIDDEN*DATA_W-1:0] v;
        for (int d = 0; d < HIDDEN; d++) v[d*DATA_W +: DATA_W] = base + d;
        make_vec = v;
    endfunction

    function automatic [DATA_W-1:0] extract(
        input logic [HIDDEN*DATA_W-1:0] vec, input int d
    );
        extract = vec[d*DATA_W +: DATA_W];
    endfunction

    // Wait for out_valid
    task automatic wait_rope_out(output logic [HIDDEN*DATA_W-1:0] result,
                                  output logic ok);
        integer cyc;
        result = '0;
        ok = 1'b0;
        cyc = 0;
        while (cyc < 80 && !ok) begin
            @(posedge clk);
            if (out_valid) begin
                result = rot_flat;
                ok = 1'b1;
            end
            cyc = cyc + 1;
        end
        if (!ok) begin
            $error("  [FAIL] Timeout waiting for rope_out_valid");
        end
    endtask

    // Load LUT entry
    task load_lut(input int lut_pos_val, input int lut_pair_val,
                  input int cos_val, input int sin_val);
        @(posedge clk);
        lut_wr_en    <= 1;
        lut_pos      <= lut_pos_val;
        lut_pair     <= lut_pair_val;
        lut_cos_data <= cos_val;
        lut_sin_data <= sin_val;
        @(posedge clk);
        lut_wr_en    <= 0;
    endtask

    // Send a RoPE input vector
    task send_rope_vec(input int pos_val,
                       input logic [HIDDEN*DATA_W-1:0] v);
        @(posedge clk);
        in_valid <= 1;
        pos      <= pos_val;
        vec_flat <= v;
        @(posedge clk);
        in_valid <= 0;
    endtask

    integer pass_count, fail_count;
    logic [HIDDEN*DATA_W-1:0] rot_result;
    logic result_ok;

    // =========================================================================
    // Main test
    // =========================================================================
    initial begin
        // Init
        rst_n = 0;
        in_valid = 0; vec_flat = '0; pos = '0;
        lut_wr_en = 0; lut_pos = '0; lut_pair = '0;
        lut_sin_data = '0; lut_cos_data = '0;
        pass_count = 0; fail_count = 0;

        wait_cycles(4);
        rst_n = 1;
        wait_cycles(2);

        // =====================================================================
        // Test 1: Identity (default LUT: cos=1, sin=0)
        // Input: [0,1,2,3,4,5,6,7] → Output: [0,1,2,3,4,5,6,7]
        // =====================================================================
        $display("Test 1: RoPE identity (angle=0, default LUT)");

        send_rope_vec(0, make_vec(0));
        wait_rope_out(rot_result, result_ok);

        if (result_ok) begin
            for (int d = 0; d < HIDDEN; d++) begin
                if ($signed(extract(rot_result, d)) !== d) begin
                    $error("  [FAIL] dim %0d: got %0d exp %0d",
                           d, $signed(extract(rot_result, d)), d);
                    fail_count = fail_count + 1;
                end
            end
        end else begin
            fail_count = fail_count + 1;
        end

        if (fail_count == 0)
            $display("  [ OK ] Test 1: RoPE identity passed");

        wait_cycles(2);

        // =====================================================================
        // Test 2: 90-degree rotation (pair 0 only)
        // Load pos=1, pair=0: cos=0, sin=4096
        // Input: [10,11, 12,13, 14,15, 16,17]
        // Pair 0: (10,11) → (10*0 - 11*1, 10*1 + 11*0) = (-11, 10)
        // Other pairs: unchanged
        // =====================================================================
        $display("Test 2: RoPE 90-degree rotation (pair 0)");

        load_lut(1, 0, 0, Q12_ONE);

        send_rope_vec(1, make_vec(10));
        wait_rope_out(rot_result, result_ok);

        if (result_ok) begin
            if ($signed(extract(rot_result, 0)) !== -11) begin
                $error("  [FAIL] dim0: got %0d exp -11",
                       $signed(extract(rot_result, 0)));
                fail_count = fail_count + 1;
            end
            if ($signed(extract(rot_result, 1)) !== 10) begin
                $error("  [FAIL] dim1: got %0d exp 10",
                       $signed(extract(rot_result, 1)));
                fail_count = fail_count + 1;
            end
            for (int d = 2; d < HIDDEN; d++) begin
                if ($signed(extract(rot_result, d)) !== 10 + d) begin
                    $error("  [FAIL] dim %0d: got %0d exp %0d",
                           d, $signed(extract(rot_result, d)), 10 + d);
                    fail_count = fail_count + 1;
                end
            end
        end else begin
            fail_count = fail_count + 1;
        end

        if (fail_count == 0)
            $display("  [ OK ] Test 2: 90-degree rotation passed");

        wait_cycles(2);

        // =====================================================================
        // Test 3: 180-degree rotation (pair 0 only)
        // Load pos=2, pair=0: cos=-4096, sin=0
        // Input: [10,11, 12,13, 14,15, 16,17]
        // Pair 0: (10,11) → (-10, -11)
        // =====================================================================
        $display("Test 3: RoPE 180-degree rotation (pair 0)");

        load_lut(2, 0, -Q12_ONE, 0);

        send_rope_vec(2, make_vec(10));
        wait_rope_out(rot_result, result_ok);

        if (result_ok) begin
            if ($signed(extract(rot_result, 0)) !== -10) begin
                $error("  [FAIL] dim0: got %0d exp -10",
                       $signed(extract(rot_result, 0)));
                fail_count = fail_count + 1;
            end
            if ($signed(extract(rot_result, 1)) !== -11) begin
                $error("  [FAIL] dim1: got %0d exp -11",
                       $signed(extract(rot_result, 1)));
                fail_count = fail_count + 1;
            end
            for (int d = 2; d < HIDDEN; d++) begin
                if ($signed(extract(rot_result, d)) !== 10 + d) begin
                    $error("  [FAIL] dim %0d: got %0d exp %0d",
                           d, $signed(extract(rot_result, d)), 10 + d);
                    fail_count = fail_count + 1;
                end
            end
        end else begin
            fail_count = fail_count + 1;
        end

        if (fail_count == 0)
            $display("  [ OK ] Test 3: 180-degree rotation passed");

        wait_cycles(2);

        // =====================================================================
        // Test 4: 45-degree rotation (pair 0 only)
        // Load pos=3, pair=0: cos=2896, sin=2896  (~0.7071 * 4096)
        // Input: [100, 0, 12, 13, 14, 15, 16, 17]
        // Pair 0: (100, 0) → (100*2896>>12, 100*2896>>12) = (70, 70)
        // =====================================================================
        $display("Test 4: RoPE 45-degree rotation (pair 0)");

        load_lut(3, 0, 2896, 2896);

        // Construct [100, 0, 12, 13, 14, 15, 16, 17]
        send_rope_vec(3, build_vec(100, 0, 12, 13, 14, 15, 16, 17));
        wait_rope_out(rot_result, result_ok);

        if (result_ok) begin
            if ($signed(extract(rot_result, 0)) !== 70) begin
                $error("  [FAIL] 45deg dim0: got %0d exp 70",
                       $signed(extract(rot_result, 0)));
                fail_count = fail_count + 1;
            end
            if ($signed(extract(rot_result, 1)) !== 70) begin
                $error("  [FAIL] 45deg dim1: got %0d exp 70",
                       $signed(extract(rot_result, 1)));
                fail_count = fail_count + 1;
            end
            for (int d = 2; d < HIDDEN; d++) begin
                if ($signed(extract(rot_result, d)) !== 10 + d) begin
                    $error("  [FAIL] 45deg dim %0d: got %0d exp %0d",
                           d, $signed(extract(rot_result, d)), 10 + d);
                    fail_count = fail_count + 1;
                end
            end
        end else begin
            fail_count = fail_count + 1;
        end

        if (fail_count == 0)
            $display("  [ OK ] Test 4: 45-degree rotation passed");

        wait_cycles(2);

        // =====================================================================
        // Test 5: Mixed rotations across all pairs at pos=5
        // pair 0: cos=4096, sin=0     (identity)
        // pair 1: cos=0, sin=4096     (90 deg)
        // pair 2: cos=-4096, sin=0    (180 deg)
        // pair 3: cos=2896, sin=2896  (45 deg)
        // Input: [10,20, 30,40, 50,60, 70,80]
        // =====================================================================
        $display("Test 5: Mixed rotations across all pairs at pos=5");

        load_lut(5, 0, Q12_ONE, 0);
        load_lut(5, 1, 0, Q12_ONE);
        load_lut(5, 2, -Q12_ONE, 0);
        load_lut(5, 3, 2896, 2896);

        send_rope_vec(5, build_vec(10, 20, 30, 40, 50, 60, 70, 80));
        wait_rope_out(rot_result, result_ok);

        if (result_ok) begin
            // Pair 0 identity: (10,20) unchanged
            if ($signed(extract(rot_result, 0)) !== 10) begin
                $error("  [FAIL] T5 dim0: got %0d exp 10",
                       $signed(extract(rot_result, 0)));
                fail_count = fail_count + 1;
            end
            if ($signed(extract(rot_result, 1)) !== 20) begin
                $error("  [FAIL] T5 dim1: got %0d exp 20",
                       $signed(extract(rot_result, 1)));
                fail_count = fail_count + 1;
            end
            // Pair 1 90deg: (30,40) → (-40, 30)
            if ($signed(extract(rot_result, 2)) !== -40) begin
                $error("  [FAIL] T5 dim2: got %0d exp -40",
                       $signed(extract(rot_result, 2)));
                fail_count = fail_count + 1;
            end
            if ($signed(extract(rot_result, 3)) !== 30) begin
                $error("  [FAIL] T5 dim3: got %0d exp 30",
                       $signed(extract(rot_result, 3)));
                fail_count = fail_count + 1;
            end
            // Pair 2 180deg: (50,60) → (-50, -60)
            if ($signed(extract(rot_result, 4)) !== -50) begin
                $error("  [FAIL] T5 dim4: got %0d exp -50",
                       $signed(extract(rot_result, 4)));
                fail_count = fail_count + 1;
            end
            if ($signed(extract(rot_result, 5)) !== -60) begin
                $error("  [FAIL] T5 dim5: got %0d exp -60",
                       $signed(extract(rot_result, 5)));
                fail_count = fail_count + 1;
            end
            // Pair 3 45deg: (70,80)
            // x' = 70*2896>>12 - 80*2896>>12 = 49 - 56 = -7
            // y' = 70*2896>>12 + 80*2896>>12 = 49 + 56 = 105
            if ($signed(extract(rot_result, 6)) !== -7) begin
                $error("  [FAIL] T5 dim6: got %0d exp -7",
                       $signed(extract(rot_result, 6)));
                fail_count = fail_count + 1;
            end
            if ($signed(extract(rot_result, 7)) !== 105) begin
                $error("  [FAIL] T5 dim7: got %0d exp 105",
                       $signed(extract(rot_result, 7)));
                fail_count = fail_count + 1;
            end
        end else begin
            fail_count = fail_count + 1;
        end

        if (fail_count == 0)
            $display("  [ OK ] Test 5: Mixed rotations passed");

        wait_cycles(2);

        // =====================================================================
        // Summary
        // =====================================================================
        $display("==============================");
        if (fail_count == 0) begin
            $display("PASS tb_mla_rope (all tests)");
        end else begin
            $display("FAIL tb_mla_rope (%0d failures)", fail_count);
        end
        $finish;
    end

endmodule
