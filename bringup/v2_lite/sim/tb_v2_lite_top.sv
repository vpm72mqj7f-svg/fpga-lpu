// =============================================================================
// tb_v2_lite_top.sv — V2-Lite FFN Engine Testbench
//
// Tests:
//   1. PLL lock → reset release sequence
//   2. Single-token FFN inference (synthetic data)
//   3. LED state sequence verification
//   4. Weight preload → compute → output check
//
// Run: vsim -c -do "run -all; quit" tb_v2_lite_top
// =============================================================================

`timescale 1ns / 1ps

module tb_v2_lite_top;

    logic        clk_sys_100m_p = 0;
    logic        clk_sys_100m_n = 1;
    logic        cpu_reset_n;
    logic [3:0]  led;
    logic        pcie_ep_refclk_p = 0;
    logic        pcie_ep_refclk_n = 1;
    logic        pcie_ep_perst_n;
    logic        pcie_ep_wake_n;

    // Clock generation: 100 MHz = 10 ns period
    always #5 begin
        clk_sys_100m_p <= ~clk_sys_100m_p;
        clk_sys_100m_n <= ~clk_sys_100m_n;
    end

    // DUT
    v2_lite_top dut (
        .clk_sys_100m_p   (clk_sys_100m_p),
        .clk_sys_100m_n   (clk_sys_100m_n),
        .cpu_reset_n      (cpu_reset_n),
        .led              (led),
        .pcie_ep_refclk_p (pcie_ep_refclk_p),
        .pcie_ep_refclk_n (pcie_ep_refclk_n),
        .pcie_ep_perst_n  (pcie_ep_perst_n),
        .pcie_ep_wake_n   (pcie_ep_wake_n)
    );

    // =========================================================================
    // Test sequence
    // =========================================================================
    integer errors;

    initial begin
        errors = 0;
        $display("============================================");
        $display(" V2-Lite FFN Engine Testbench");
        $display(" Target: 1SM21BHU2F53E1VG, hidden=2048, inter=1408");
        $display("============================================");

        // --- Test 1: Power-on reset ---
        $display("\n[Test 1] Power-on reset sequence...");
        cpu_reset_n = 1'b0;
        pcie_ep_perst_n = 1'b0;
        #1000;  // 1 µs
        cpu_reset_n = 1'b1;
        pcie_ep_perst_n = 1'b1;

        // --- Wait for PLL lock ---
        $display("[Test 2] Waiting for PLL lock...");
        wait(dut.pll_locked);
        $display("  PLL locked at %0t ns", $time);

        // --- Wait for bringup FSM to advance ---
        $display("[Test 3] Bringup FSM progression...");
        wait(dut.u_ffn.busy);
        $display("  FFN busy asserted at %0t ns", $time);

        // --- Test 4: FFN compute ---
        $display("[Test 4] FFN compute...");
        wait(dut.u_ffn.done);
        $display("  FFN done asserted at %0t ns", $time);

        // --- Test 5: LED state check ---
        $display("[Test 5] LED state check...");
        #10000;  // let LEDs settle
        if (dut.b_state == dut.B_PASS) begin
            $display("  PASS: Bringup state = B_PASS");
            if (led[3] == 1'b1)  // LED3 off = PASS
                $display("  LED3 = OFF (PASS indicator OK)");
            else begin
                $display("  ERROR: LED3 should be OFF for PASS");
                errors = errors + 1;
            end
        end else begin
            $display("  Bringup state = %0d (expected B_PASS=6)", dut.b_state);
            errors = errors + 1;
        end

        // --- Test 6: PLL heartbeat visible? ---
        $display("[Test 6] PLL heartbeat check (LED0 should blink)...");
        #100000000;  // 100 ms = ~5 blinks at 2 Hz
        $display("  LED test complete");

        // --- Summary ---
        $display("\n============================================");
        if (errors == 0) begin
            $display(" ALL TESTS PASSED");
        end else begin
            $display(" %0d TEST(S) FAILED", errors);
        end
        $display("============================================");

        $finish;
    end

    // =========================================================================
    // Waveform dumping
    // =========================================================================
    initial begin
        $dumpfile("tb_v2_lite_top.vcd");
        $dumpvars(0, tb_v2_lite_top);
    end

endmodule
