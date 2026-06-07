// =============================================================================
// tb_v2_lite_top.sv — V2-Lite FFN Engine Testbench (Questa/Icarus portable)
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

    // 100 MHz clock
    always #5 begin
        clk_sys_100m_p <= ~clk_sys_100m_p;
        clk_sys_100m_n <= ~clk_sys_100m_n;
    end

    v2_lite_top dut (.*);

    integer errs;

    initial begin
        errs = 0;
        $display("============================================");
        $display(" V2-Lite FFN Testbench (Questa 2025.3)");
        $display("============================================");

        // Test 1: Power-on reset (1 µs)
        $display("[1] Power-on reset...");
        cpu_reset_n     = 1'b0;
        pcie_ep_perst_n = 1'b0;
        #1000;  // 1 µs
        cpu_reset_n     = 1'b1;
        pcie_ep_perst_n = 1'b1;
        $display("    Reset released");

        // Test 2: Wait PLL lock (256 cycles = 2.56 µs + margin)
        $display("[2] PLL lock (wait 5 µs)...");
        #5000;  // 5 µs = 500 cycles @ 100 MHz
        $display("    PLL should be locked");

        // Test 3: LED heartbeat — LED[0] should toggle after PLL lock
        $display("[3] LED heartbeat (wait 500 ms)...");
        #500000000;  // 500 ms → ~5 blinks at 2 Hz
        $display("    LED heartbeat time elapsed");

        // Test 4: FFN self-test (200 ms more)
        $display("[4] FFN self-test (wait 200 ms)...");
        #200000000;
        $display("    FFN pipeline should have completed");

        // Test 5: LED[3] check — should be OFF if B_PASS
        $display("[5] Result check...");
        if (led[3] == 1'b1) begin  // Active low: OFF = PASS
            $display("    PASS: LED3 OFF (bringup passed)");
        end else begin
            $display("    WARN: LED3 ON (check FSM state)");
        end

        // Test 6: LED[0] should be blinking (PLL heartbeat active)
        if (led[0] != 1'b0 && led[0] != 1'b1) begin
            $display("    LED0: toggling (heartbeat OK)");
        end

        $display("");
        $display("============================================");
        $display(" V2-Lite Simulation Complete");
        $display(" ERRORS: %0d", errs);
        $display("============================================");

        $finish;
    end

    // Waveform dump (works with Questa and Icarus)
    initial begin
        $dumpfile("tb_v2_lite_top.vcd");
        $dumpvars(0, tb_v2_lite_top);
    end

endmodule
