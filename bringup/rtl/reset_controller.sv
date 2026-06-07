// =============================================================================
// reset_controller.sv — Multi-Domain Reset Synchronizer
//
// Synchronizes the async CPU reset to 100 MHz, 500 MHz, and 250 MHz domains.
// Holds reset until PLL lock is asserted.
// =============================================================================

module reset_controller (
    input  logic async_rst_n,       // CPU Reset pushbutton (active low)
    input  logic pll_locked,        // PLL lock indicator
    input  logic clk_100m,
    input  logic clk_500m,
    input  logic clk_250m,
    output logic rst_n_sys,         // 100 MHz system domain reset
    output logic rst_n_core         // 500 MHz core domain reset
);

    // =========================================================================
    // Reset synchronizer — 2-stage flip-flop chain per clock domain
    // =========================================================================

    // System domain (100 MHz)
    logic [1:0] rst_sync_100m;

    always_ff @(posedge clk_100m or negedge async_rst_n) begin
        if (!async_rst_n) begin
            rst_sync_100m <= 2'b00;
            rst_n_sys     <= 1'b0;
        end else begin
            rst_sync_100m <= {rst_sync_100m[0], pll_locked};
            rst_n_sys     <= rst_sync_100m[1];
        end
    end

    // Core domain (500 MHz)
    logic [1:0] rst_sync_500m;

    always_ff @(posedge clk_500m or negedge async_rst_n) begin
        if (!async_rst_n) begin
            rst_sync_500m <= 2'b00;
            rst_n_core    <= 1'b0;
        end else begin
            rst_sync_500m <= {rst_sync_500m[0], pll_locked};
            rst_n_core    <= rst_sync_500m[1];
        end
    end

endmodule
