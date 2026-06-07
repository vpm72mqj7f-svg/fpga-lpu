// =============================================================================
// pll_controller.sv — Stratix 10 I/O PLL instantiation
//
// Generates 500 MHz core clock and 250 MHz DSP clock from 100 MHz reference.
// Uses Stratix 10 I/O PLL (iopll) primitive.
//
// Target frequencies:
//   refclk:    100 MHz (Si5341A U16 CLK_SYS_100M)
//   outclk0:   500 MHz (×5, core fabric)
//   outclk1:   250 MHz (×2.5, DSP/FFN systolic array)
//
// Note: Actual PLL configuration depends on Quartus IP generation.
// This module serves as a bring-up placeholder; replace with IP-generated
// PLL when the Quartus project is fully configured.
// =============================================================================

module pll_controller (
    input  logic refclk,         // 100 MHz reference clock
    input  logic rst_n,          // Async reset (active low)
    output logic clk_500m,       // 500 MHz core clock
    output logic clk_250m,       // 250 MHz DSP clock
    output logic locked          // PLL lock indicator
);

    // =========================================================================
    // Bring-Up Simplification
    //
    // For initial bitstream generation, bypass the PLL and use the reference
    // clock directly. This allows the design to pass through Quartus without
    // requiring the full PLL IP to be configured.
    //
    // Replace this with proper iopll_wysiwyg or altera_iopll instantiation
    // when the Quartus project and IP catalog are set up.
    // =========================================================================

    // Bypass mode: use reference clock directly
    assign clk_500m  = refclk;
    assign clk_250m  = refclk;
    // PLL lock: mimic lock after a simple counter delay
    // (in real hardware, this comes from the iopll lock output)

    logic [7:0] lock_cnt;

    always_ff @(posedge refclk or negedge rst_n) begin
        if (!rst_n) begin
            lock_cnt <= '0;
            locked   <= 1'b0;
        end else begin
            if (!locked) begin
                if (lock_cnt == 8'hFF) begin
                    locked   <= 1'b1;
                    lock_cnt <= lock_cnt;   // saturate
                end else begin
                    lock_cnt <= lock_cnt + 1;
                end
            end
        end
    end

    // =========================================================================
    // TODO: Replace with properly configured I/O PLL
    //
    // Example iopll instantiation (commented out):
    //
    // wire locked_int;
    // wire clk_500m_int, clk_250m_int;
    //
    // iopll_wysiwyg #(
    //     .number_of_counters(2),
    //     .reference_clock_frequency("100.0 MHz"),
    //     .output_clock_frequency0("500.0 MHz"),
    //     .output_clock_frequency1("250.0 MHz"),
    //     .duty_cycle0(50),
    //     .duty_cycle1(50),
    //     .phase_shift0("0 ps"),
    //     .phase_shift1("0 ps"),
    //     .pll_auto_reset("ON"),
    //     .pll_bandwidth(1),          // Medium bandwidth
    //     .pll_dsm_out_sel("OFF")
    // ) iopll_inst (
    //     .refclk(refclk),
    //     .rst(~rst_n),
    //     .outclk({clk_250m_int, clk_500m_int}),
    //     .locked(locked_int)
    // );
    //
    // // Clock buffers
    // stratix10_clkena u_clkena0 (
    //     .inclk(clk_500m_int),
    //     .ena(1'b1),
    //     .outclk(clk_500m)
    // );
    //
    // assign clk_250m = clk_250m_int;
    // assign locked   = locked_int;
    // =========================================================================

endmodule
