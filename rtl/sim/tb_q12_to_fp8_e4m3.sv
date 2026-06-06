`timescale 1ns/1ps

//=============================================================================
// tb_q12_to_fp8_e4m3.sv — self-checking testbench for Q12→FP8 E4M3 encoder
//
// Tests 15 cases covering: zero, positive, negative, saturating, sign bit.
// DUT is combinational — no clock needed, but we use one for structure.
//=============================================================================

module tb_q12_to_fp8_e4m3;

    logic signed [31:0] x_q12;
    logic [7:0]         fp8;
    logic [7:0]         expected;

    q12_to_fp8_e4m3 u_dut (.x_q12(x_q12), .fp8(fp8));

    integer pass_count, fail_count;

    task check(input string tc_name, input [31:0] x_val, input [7:0] exp);
        begin
            x_q12 = x_val;
            #1;  // allow combinational propagation
            expected = exp;
            if (fp8 !== exp) begin
                $error("[FAIL] %s: x=%0d → fp8=%02h, expected %02h",
                       tc_name, $signed(x_val), fp8, exp);
                fail_count = fail_count + 1;
            end else begin
                $display("  [ OK ] %s: x=%0d → fp8=%02h", tc_name, $signed(x_val), fp8);
                pass_count = pass_count + 1;
            end
        end
    endtask

    initial begin
        x_q12 = 0;
        pass_count = 0;
        fail_count = 0;
        #10;  // settle

        // ================================================================
        // Zero
        // ================================================================
        $display("Testing Q12 → FP8 E4M3 encoder");
        check("T01: zero",              32'd0,           8'h00);
        check("T02: negative zero-ish", -32'd1,           8'h80); // sign=1, ax=1, mag=0x00

        // ================================================================
        // Positive values — rise through the threshold table
        //   thresholds in Q12: 512, 1536, 2560, 3584, 5120, 7168, 10240,
        //                      14336, 20480, 28672
        // ================================================================
        check("T03: tiny + (below 512)",   32'd256,      8'h00);
        check("T04: threshold 512",        32'd512,      8'h28);  // 0.25
        check("T05: threshold 1536",       32'd1536,     8'h30);  // 0.5
        check("T06: threshold 2560",       32'd2560,     8'h34);  // 0.75
        check("T07: threshold 3584",       32'd3584,     8'h38);  // 1.0
        check("T08: Q12_ONE (4096)",       32'd4096,     8'h38);  // 1.0 (4096 < 5120)
        check("T09: threshold 5120",       32'd5120,     8'h3c);  // 1.5
        check("T10: value 8192 (2.0)",     32'd8192,     8'h40);  // 2.0
        check("T11: value 12288 (3.0)",      32'd12288,    8'h44);  // 3.0
        check("T12: value 16384 (4.0)",      32'd16384,    8'h48);  // 4.0
        check("T13: value 24576 (6.0)",      32'd24576,    8'h4c);  // 6.0
        check("T14: saturate to 8.0",      32'd32768,    8'h50);  // 8.0
        check("T15: max positive (2^31-1)", 32'h7FFFFFFF, 8'h50);  // 8.0 sat

        // ================================================================
        // Negative values — sign bit set
        // ================================================================
        check("T16: small negative",       -32'd100,     8'h80);  // ax<512 → 0x00
        check("T17: neg threshold 512",    -32'd512,     8'ha8);  // {1, 0x28}
        check("T18: neg Q12_ONE (-4096)",  -32'd4096,    8'hb8);  // {1, 0x38} = 1.0
        check("T19: neg large",            -32'd16384,   8'hc8);  // {1, 0x48} = 4.0
        check("T20: neg saturate",         -32'd32768,   8'hd0);  // {1, 0x50}

        // ================================================================
        // Boundary / edge cases
        // ================================================================
        // Just below each threshold
        check("T21: just below 512",        32'd511,      8'h00);
        check("T22: just below 1536",       32'd1535,     8'h28);
        // max 32-bit negative (2's complement min)
        check("T23: min signed (-2^31)",    32'h80000000, 8'hd0);  // neg saturate

        // ================================================================
        // Summary
        // ================================================================
        $display("==============================");
        if (fail_count == 0)
            $display("PASS tb_q12_to_fp8_e4m3 (%0d/%0d tests)", pass_count, pass_count);
        else
            $display("FAIL tb_q12_to_fp8_e4m3 (%0d pass, %0d fail)", pass_count, fail_count);
        $finish;
    end

endmodule
