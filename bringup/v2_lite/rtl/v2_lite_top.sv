// =============================================================================
// v2_lite_top.sv — DeepSeek V2-Lite FFN Top-Level (Production Quality)
//
// V2-Lite: 16B params, 27 layers
//   hidden=2048, inter=1408, 66 experts (64 routed + 2 shared), TOP_K=6, FP8
// Target: Stratix 10 MX (1SM21BHU2F53E1VG), DK-DEV-1SMX-H-A
// Quartus Prime Pro 26.1
//
// Clock domains:
//   clk_100m (100 MHz) — system control, LEDs, bringup FSM
//   clk_500m (500 MHz) — FFN engine, systolic arrays, HBM2 UIB
//   clk_250m (250 MHz) — DSP fabric clock
//
// Architecture: CPU Attention → PCIe → HBM2 (weights) → FFN Engine → PCIe → CPU
// IOPLL: Si5341A 100 MHz LVDS ref → 500 MHz / 250 MHz outputs
// HBM2: 2 stacks × 8 pseudo-channels, AXI4 per channel, 256 GB/s aggregate
//        (bring-up uses single 256-bit AXI4 channel)
// =============================================================================

module v2_lite_top (
    // ---- System Clock (100 MHz LVDS, Si5341A U16) ----
    input  logic        clk_sys_100m_p,
    input  logic        clk_sys_100m_n,

    // ---- HBM2 Reference Clocks (100 MHz LVDS) ----
    input  logic        hbm2_uib0_refclk_p,
    input  logic        hbm2_uib0_refclk_n,
    input  logic        hbm2_uib1_refclk_p,
    input  logic        hbm2_uib1_refclk_n,

    // ---- HBM2 AXI4 Master (to Intel HBM2 Controller IP) ----
    output logic [31:0] hbm2_axi_araddr,
    output logic [7:0]  hbm2_axi_arlen,
    output logic [2:0]  hbm2_axi_arsize,
    output logic        hbm2_axi_arvalid,
    input  logic        hbm2_axi_arready,
    input  logic [255:0] hbm2_axi_rdata,
    input  logic [1:0]  hbm2_axi_rresp,
    input  logic        hbm2_axi_rvalid,
    output logic        hbm2_axi_rready,
    input  logic        hbm2_axi_rlast,

    // ---- PCIe Reference Clocks ----
    input  logic        pcie_ep_refclk_p,
    input  logic        pcie_ep_refclk_n,

    // ---- PCIe Control ----
    input  logic        pcie_ep_perst_n,
    output logic        pcie_ep_wake_n,

    // ---- CPU Reset ----
    input  logic        cpu_reset_n,

    // ---- User LEDs (Active Low) ----
    output logic [3:0]  led
);

    // =========================================================================
    // V2-Lite Model Parameters
    // =========================================================================
    localparam int HIDDEN      = 2048;
    localparam int INTER       = 1408;
    localparam int NUM_EXPERTS = 66;     // 64 routed + 2 shared
    localparam int TOP_K       = 6;
    localparam int DATA_W      = 8;      // FP8 E4M3
    localparam int DSP_LANES   = 64;

    // =========================================================================
    // Clock & PLL
    // =========================================================================
    logic clk_100m, clk_500m, clk_250m;
    logic pll_locked;

    // LVDS input buffer (simplified for bringup — use ALTCLKCTRL in production)
    assign clk_100m = clk_sys_100m_p;

    // IOPLL: 100 MHz → 500 MHz + 250 MHz
    // Intel Stratix 10 I/O PLL — inferred as altera_iopll for Quartus synthesis
    altera_iopll #(
        .reference_clock_frequency("100.0 MHz"),
        .output_clock_frequency0("500.0 MHz"),
        .output_clock_frequency1("250.0 MHz"),
        .pll_operation_mode("direct"),
        .output_clock0_duty_cycle(50),
        .output_clock1_duty_cycle(50),
        .pll_auto_reset("ON")
    ) u_iopll (
        .refclk   (clk_100m),
        .rst      (~cpu_reset_n),
        .outclk0  (clk_500m),
        .outclk1  (clk_250m),
        .locked   (pll_locked)
    );

    // =========================================================================
    // Reset Release IP (required for Stratix 10 configuration)
    // =========================================================================
    logic reset_release;
    stratix10_reset_release u_reset_release (
        .ninit_done (reset_release)
    );

    // =========================================================================
    // Reset Controller
    // =========================================================================
    logic rst_n_sys, rst_n_core;
    reset_controller u_rst (
        .async_rst_n (cpu_reset_n & reset_release),
        .pll_locked  (pll_locked),
        .clk_100m    (clk_100m),
        .clk_500m    (clk_500m),
        .clk_250m    (clk_250m),
        .rst_n_sys   (rst_n_sys),
        .rst_n_core  (rst_n_core)
    );

    // =========================================================================
    // FFN Engine
    //
    // The FFN engine integrates the HBM2 weight reader internally.
    // Top-level routes AXI4 to the external HBM2 Controller IP.
    // Data I/O is via PCIe RX/TX (connected to the bringup FSM for
    // self-test, and to CPU via PCIe in production).
    // =========================================================================
    logic                         ffn_rx_valid, ffn_rx_ready;
    logic [HIDDEN*DATA_W-1:0]     ffn_rx_data;
    logic                         ffn_tx_valid, ffn_tx_ready;
    logic [HIDDEN*DATA_W-1:0]     ffn_tx_data;
    logic                         ffn_busy, ffn_done;
    logic [$clog2(NUM_EXPERTS)-1:0] ffn_expert_id [TOP_K];

    v2_lite_ffn_engine #(
        .HIDDEN      (HIDDEN),
        .INTER       (INTER),
        .NUM_EXPERTS (NUM_EXPERTS),
        .TOP_K       (TOP_K),
        .DATA_W      (DATA_W),
        .ACCUM_W     (24),
        .DSP_LANES   (DSP_LANES)
    ) u_ffn (
        .clk             (clk_500m),
        .rst_n           (rst_n_core),
        // PCIe data interface
        .pcie_rx_valid   (ffn_rx_valid),
        .pcie_rx_data    (ffn_rx_data),
        .pcie_rx_ready   (ffn_rx_ready),
        .pcie_tx_valid   (ffn_tx_valid),
        .pcie_tx_data    (ffn_tx_data),
        .pcie_tx_ready   (ffn_tx_ready),
        // HBM2 AXI4 — routed directly to top-level ports
        .m_axi_araddr    (hbm2_axi_araddr),
        .m_axi_arlen     (hbm2_axi_arlen),
        .m_axi_arsize    (hbm2_axi_arsize),
        .m_axi_arvalid   (hbm2_axi_arvalid),
        .m_axi_arready   (hbm2_axi_arready),
        .m_axi_rdata     (hbm2_axi_rdata),
        .m_axi_rresp     (hbm2_axi_rresp),
        .m_axi_rvalid    (hbm2_axi_rvalid),
        .m_axi_rready    (hbm2_axi_rready),
        .m_axi_rlast     (hbm2_axi_rlast),
        // Expert selection
        .expert_id       (ffn_expert_id),
        // Status
        .busy            (ffn_busy),
        .done            (ffn_done)
    );

    // Expert selection: default to first TOP_K experts for bringup self-test
    generate
        for (genvar _e = 0; _e < TOP_K; _e = _e + 1) begin : gen_expert_sel
            assign ffn_expert_id[_e] = _e[$clog2(NUM_EXPERTS)-1:0];
        end
    endgenerate

    assign ffn_tx_ready = 1'b1;  // always ready in bringup self-test

    // =========================================================================
    // Bring-Up Self-Test FSM
    //
    // Tests the FFN engine end-to-end by sending a test activation pattern
    // and checking for a non-zero result. This validates:
    //   1. PLL lock and clock domains
    //   2. FFN engine FSM progression through all states
    //   3. Systolic array compute (gate + up + down)
    //   4. HBM2 weight reading (if weights are preloaded)
    //   5. PCIe TX handshake
    // =========================================================================
    typedef enum logic [3:0] {
        B_IDLE, B_WAIT_PLL, B_SEND, B_WAIT, B_CHECK, B_PASS, B_FAIL
    } bstate_t;
    bstate_t b_state;

    always_ff @(posedge clk_100m or negedge rst_n_sys) begin
        if (!rst_n_sys) begin
            b_state <= B_IDLE;
        end else begin
            case (b_state)
                B_IDLE:      if (pll_locked)     b_state <= B_WAIT_PLL;
                B_WAIT_PLL:                       b_state <= B_SEND;
                B_SEND: begin
                    ffn_rx_valid <= 1'b1;
                    // Fill with test pattern: activation[i] = i (ramp)
                    for (int i = 0; i < HIDDEN; i++) begin
                        ffn_rx_data[i*DATA_W +: DATA_W] <= i[DATA_W-1:0];
                    end
                    b_state <= B_WAIT;
                end
                B_WAIT: begin
                    ffn_rx_valid <= 1'b0;
                    if (ffn_done) b_state <= B_CHECK;
                end
                B_CHECK: begin
                    // Check: any non-zero output indicates compute happened
                    if (|ffn_tx_data)
                        b_state <= B_PASS;
                    else
                        b_state <= B_FAIL;
                end
                B_PASS, B_FAIL: ;  // latched until reset
                default: b_state <= B_IDLE;
            endcase
        end
    end

    // =========================================================================
    // LED Debug Display
    //
    // LED[0]: PLL lock heartbeat (2 Hz blink when locked)
    // LED[1]: FFN engine busy
    // LED[2]: FFN engine done (brief pulse)
    // LED[3]: Bring-up result (off=pass, on=fail, blink=in progress)
    // =========================================================================
    led_controller u_led (
        .clk           (clk_100m),
        .rst_n         (rst_n_sys),
        .pll_locked    (pll_locked),
        .ffn_busy      (ffn_busy),
        .ffn_done      (ffn_done),
        .bringup_state (b_state),
        .led           (led)
    );

    // =========================================================================
    // PCIe wake — unused during bringup, tie inactive
    // =========================================================================
    assign pcie_ep_wake_n = 1'b1;

endmodule
