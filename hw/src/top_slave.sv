//=============================================================================
// top_slave.sv — Slave FPGA (Compute Pipeline)
//
// One of 24 Slave chips in the 32-chip cluster. Each card has 3 Slaves
// (Chips 1-3) connected to the Master via C2C dual ring.
//
// 384 layers / 32 chips = 12 layers per chip (all chips, master or slave).
//
// Code uniformity: uses the SAME chip_top.sv RTL as Master, with
// IS_PCIE_MASTER=0. No PCIe IP — saves ~15-20% logic area vs Master.
// Configuration received via C2C control packets from the Master.
//=============================================================================

module top_slave (
    // ── HBM2e UIB ──
    // [TODO] QSYS-generated HBM port list

    // ── C2C (Chip-to-Chip) Dual Ring ──
    // [TODO] Map to F-Tile transceiver ports
    // output logic [3:0] c2c_tx_a_p, c2c_tx_a_n,
    // input  logic [3:0] c2c_rx_a_p, c2c_rx_a_n,
    // output logic [3:0] c2c_tx_b_p, c2c_tx_b_n,
    // input  logic [3:0] c2c_rx_b_p, c2c_rx_b_n,

    // ── QSFP-DD (F-Tile, Phase 2 cross-card) ──
    // [TODO] Multi-card inter-board links

    // ── Board Control ──
    input  logic        clk_board_100m,
    input  logic        cpu_reset_n,

    // ── Debug ──
    output logic [3:0]  debug_led,
    output logic        uart_tx
);

    // ========================================================================
    // Clock & Reset
    // ========================================================================
    logic clk_sys, clk_dsp, clk_hbm;
    logic rst_n_sys;

    // [TODO] PLL: 100 MHz → DSP 450 MHz (×9/2), HBM 450 MHz
    // DSP and HBM share same 450 MHz domain — no CDC needed between them
    // No PCIe clock domain needed for slave
    assign clk_sys  = clk_board_100m;
    assign clk_dsp  = clk_board_100m;
    assign clk_hbm  = clk_board_100m;

    // Reset synchronizer
    logic [2:0] rst_sr;
    always_ff @(posedge clk_sys or negedge cpu_reset_n) begin
        if (!cpu_reset_n) rst_sr <= '0;
        else              rst_sr <= {rst_sr[1:0], 1'b1};
    end
    assign rst_n_sys = rst_sr[2];

    // ========================================================================
    // LED heartbeat
    // ========================================================================
    logic [26:0] led_cnt;
    always_ff @(posedge clk_sys or negedge rst_n_sys) begin
        if (!rst_n_sys) led_cnt <= '0;
        else            led_cnt <= led_cnt + 1'b1;
    end
    assign debug_led[0] = led_cnt[26];

    // ========================================================================
    // Chip Configuration (via DIP switch or C2C control packet)
    // ========================================================================
    // Slave chips are configured with:
    //   - CHIP_ID: which position in the 32-chip pipeline (1-31)
    //   - LAYER_START/LAYER_END: which transformer layers to compute
    //   - EXPERT_BITMAP: which MoE experts are local
    //
    // On power-up, the slave waits for a C2C CONFIG packet from the master
    // before beginning computation. Defaults can be set via DIP switches.

    // [TODO] DIP switch input for CHIP_ID
    logic [4:0] dip_chip_id;
    assign dip_chip_id = 5'd1;  // placeholder: default to chip 1

    // ========================================================================
    // Chip Core — same chip_top.sv RTL as Master
    //
    // IS_PCIE_MASTER=0: PCIe logic gated by synthesis. Same compute pipeline.
    // 12 layers per chip, distributed across 32 chips = 384 total.
    // ========================================================================
    // [TODO] Uncomment after HBM and C2C IP are integrated:
    //
    // chip_top #(
    //     .CHIP_ID(dip_chip_id), .CARD_ID(dip_chip_id / 4),
    //     .LAYER_START(dip_chip_id * 12),
    //     .LAYER_END(dip_chip_id * 12 + 11),
    //     .IS_PCIE_MASTER(0)
    // ) u_chip (
    //     .clk, .rst_n(rst_n_sys),
    //     .c2c_rx_a(c2c_rx_a_int), .c2c_tx_a(c2c_tx_a_int),
    //     .c2c_rx_b(c2c_rx_b_int), .c2c_tx_b(c2c_tx_b_int),
    //     .pcie_host('0),       // unused on slave
    //     .pcie_fpga(),         // unused on slave
    //     .c2c_proxy()          // unused on slave (only chip 0 uses proxy)
    // );

    // ========================================================================
    // C2C Passthrough mode (before chip_top is enabled)
    // ========================================================================
    // In passthrough mode, the slave simply forwards C2C traffic without
    // any local processing. This allows incremental bring-up: first verify
    // the C2C ring works end-to-end with all slaves in passthrough, then
    // enable layer computation one chip at a time.

    // [TODO] Implement C2C passthrough MUX

    // ========================================================================
    // LED Status (Slave)
    // ========================================================================
    // LED[0]: heartbeat
    // LED[1]: C2C link A up
    // LED[2]: C2C link B up
    // LED[3]: Layer compute active
    assign debug_led[1] = 1'b0;  // [TODO] C2C link A status
    assign debug_led[2] = 1'b0;  // [TODO] C2C link B status
    assign debug_led[3] = 1'b0;  // [TODO] Compute active

    // ========================================================================
    // UART Debug
    // ========================================================================
    assign uart_tx = 1'b1;

endmodule
