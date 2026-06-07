// =============================================================================
// bringup_top.sv — Stratix 10 MX Bring-Up Top-Level Module
//
// Target: Intel Stratix 10 MX Development Kit (DK-DEV-1SMX-H-A)
// Device: 1SM21BHU2F53E1VG
//
// Phase 1 bring-up: CPU attention + FPGA FFN on S10 MX dev kit.
// This top-level module connects board I/O to the FFN engine, using:
//   - System clock (100 MHz Si5341A) → PLL → 500 MHz core / 250 MHz DSP
//   - HBM2 for weight storage (8 GB, 2 stacks × 8 channels)
//   - PCIe Gen3 x16 for host-to-FPGA data transfer
//   - 4 User LEDs for debug status
//   - CPU Reset pushbutton
// =============================================================================

module bringup_top (
    // ---- 100 MHz System Clock (Si5341A U16, CLK_SYS_100M) ----
    input  logic        clk_sys_100m_p,
    input  logic        clk_sys_100m_n,

    // ---- 50 MHz System Clock (Si5338A U18, CLK_SYS_50M) ----
    input  logic        clk_sys_50m_p,
    input  logic        clk_sys_50m_n,

    // ---- 125 MHz Configuration Clock (Si510 U17, S10_OSC_CLK_1) ----
    input  logic        clk_osc_125m,

    // ---- Core Backup Clock (Si5338A U18, CLK_CORE_BAK) ----
    input  logic        clk_core_bak_p,
    input  logic        clk_core_bak_n,

    // ---- HBM2 UIB Reference Clocks (Si5341A U16) ----
    input  logic        hbm2_uib0_refclk_p,
    input  logic        hbm2_uib0_refclk_n,
    input  logic        hbm2_uib1_refclk_p,
    input  logic        hbm2_uib1_refclk_n,

    // ---- HBM2 ESRAM Reference Clocks (Si5341A U16) ----
    input  logic        hbm2_esram0_refclk_p,
    input  logic        hbm2_esram0_refclk_n,
    input  logic        hbm2_esram1_refclk_p,
    input  logic        hbm2_esram1_refclk_n,

    // ---- HBM2 Reference Resistors ----
    input  logic        hbm2_uib0_rzq,        // 240 Ohm 1%
    input  logic        hbm2_uib1_rzq,        // 240 Ohm 1%

    // ---- HBM2 ATB Test Points (optional, for debug) ----
    inout  wire  [3:0]  hbm2_atb_uib0,        // UIB00: AP26, AT29, AT28, AR26
    inout  wire  [3:0]  hbm2_atb_uib1,        // UIB01: U27, V28, U28, T27

    // ---- PCIe Endpoint Reference Clock (Si5341A U16, REFCLK_PCIE_EP) ----
    input  logic        pcie_ep_refclk_p,
    input  logic        pcie_ep_refclk_n,

    // ---- PCIe Endpoint Transceiver Clock (Si5338A U18, REFCLK_PCIE_EP1) ----
    input  logic        pcie_xcvr_refclk_p,
    input  logic        pcie_xcvr_refclk_n,

    // ---- PCIe Endpoint PERST and WAKE ----
    input  logic        pcie_ep_perst_n,       // PERST0 (S1 pushbutton, pin AH39)
    output logic        pcie_ep_wake_n,        // WAKE to host

    // ---- PCIe Transceiver Lanes (Endpoint — Bank 1C/1D/1E) ----
    // Placeholder: actual PCIe IP instantiation will drive these
    // input  logic [15:0]  pcie_ep_rx_p, pcie_ep_rx_n;
    // output logic [15:0]  pcie_ep_tx_p, pcie_ep_tx_n;

    // ---- DDR4 Component Memory Clocks (Si5338B U19) ----
    input  logic        ddr4_comp_clk_p,
    input  logic        ddr4_comp_clk_n,

    // ---- DDR4 DIMM Module Clocks (Si5338B U19) ----
    input  logic        ddr4_dimm_clk_p,
    input  logic        ddr4_dimm_clk_n,

    // ---- HiLo Memory Clocks (Si5338B U19) ----
    input  logic        hilo_clk_p,
    input  logic        hilo_clk_n,

    // ---- User LEDs (Active Low: illuminates when driven low) ----
    output logic [3:0]  led,                   // D7=BG12, D8=BF12, D9=BG11, D10=BH11

    // ---- CPU Reset Pushbutton (S10, pin BL14) ----
    input  logic        cpu_reset_n

    // ---- Configuration Status (driven by SDM, for monitoring) ----
    // These are typically auto-handled; exposed for debug if needed
    // output logic     conf_done,              // D14, AY39
    // output logic     cvp_conf_done          // D16, BC42
);

    // =========================================================================
    // Local Parameters
    // =========================================================================
    localparam int HIDDEN      = 2048;          // V2 Lite hidden dim
    localparam int INTER       = 1408;          // V2 Lite expert intermediate
    localparam int NUM_EXPERTS = 66;            // 64 routed + 2 shared
    localparam int TOP_K       = 6;             // activated per token
    localparam int DATA_W      = 8;             // FP8 weight/activation precision

    // =========================================================================
    // Clock & Reset
    // =========================================================================

    // Differential clock buffers
    logic clk_100m;
    logic clk_50m;

    // Stratix 10 LVDS input buffer primitives (uses IO_PLL)
    // During bring-up: use single-ended path for simplicity
    // Production: use dedicated LVDS clock input buffers
    assign clk_100m = clk_sys_100m_p;           // Simplified — add LVDS buffer IP
    assign clk_50m  = clk_sys_50m_p;            // Simplified — add LVDS buffer IP

    // PLL-generated clocks
    logic clk_core_500m;                        // 500 MHz core fabric clock
    logic clk_dsp_250m;                         // 250 MHz DSP clock
    logic clk_100m_locked;                      // PLL lock indicator

    // Reset synchronizer
    logic rst_n_core;
    logic rst_n_sys;

    // =========================================================================
    // PLL Controller — Generate 500 MHz / 250 MHz from 100 MHz reference
    // =========================================================================
    pll_controller u_pll (
        .refclk   (clk_100m),
        .rst_n    (cpu_reset_n),
        .clk_500m (clk_core_500m),
        .clk_250m (clk_dsp_250m),
        .locked   (clk_100m_locked)
    );

    // =========================================================================
    // Reset Controller — Synchronize async reset to each clock domain
    // =========================================================================
    reset_controller u_rst (
        .async_rst_n (cpu_reset_n),
        .pll_locked  (clk_100m_locked),
        .clk_100m    (clk_100m),
        .clk_500m    (clk_core_500m),
        .clk_250m    (clk_dsp_250m),
        .rst_n_sys   (rst_n_sys),
        .rst_n_core  (rst_n_core)
    );

    // =========================================================================
    // FFN Engine — CPU Attention → FPGA FFN → CPU via PCIe
    // =========================================================================

    // PCIe streaming interface (connected through PCIe IP in production)
    logic                          ffn_rx_valid;
    logic [HIDDEN*DATA_W-1:0]      ffn_rx_data;
    logic                          ffn_rx_ready;
    logic                          ffn_tx_valid;
    logic [HIDDEN*DATA_W-1:0]      ffn_tx_data;
    logic                          ffn_tx_ready;
    logic                          ffn_busy;
    logic                          ffn_done;

    // Expert routing (from CPU via PCIe sideband or control plane)
    logic [$clog2(NUM_EXPERTS)-1:0] ffn_expert_id [TOP_K];
    logic                           ffn_expert_valid;

    // Weight preload (one-time init from HBM2 controller — placeholder)
    logic                          ffn_wt_wr_en;
    logic [$clog2(NUM_EXPERTS)-1:0] ffn_wt_expert_id;
    logic [1:0]                    ffn_wt_type;       // 0=gate, 1=up, 2=down
    logic [$clog2(INTER)-1:0]      ffn_wt_row;
    logic [$clog2(HIDDEN)-1:0]     ffn_wt_col;
    logic [DATA_W-1:0]             ffn_wt_data;

    s10_ffn_engine #(
        .HIDDEN      (HIDDEN),
        .INTER       (INTER),
        .NUM_EXPERTS (NUM_EXPERTS),
        .TOP_K       (TOP_K),
        .DATA_W      (DATA_W)
    ) u_ffn (
        .clk          (clk_core_500m),
        .rst_n        (rst_n_core),

        .pcie_rx_valid (ffn_rx_valid),
        .pcie_rx_data  (ffn_rx_data),
        .pcie_rx_ready (ffn_rx_ready),
        .pcie_tx_valid (ffn_tx_valid),
        .pcie_tx_data  (ffn_tx_data),
        .pcie_tx_ready (ffn_tx_ready),

        .wt_wr_en      (ffn_wt_wr_en),
        .wt_expert_id  (ffn_wt_expert_id),
        .wt_type       (ffn_wt_type),
        .wt_row        (ffn_wt_row),
        .wt_col        (ffn_wt_col),
        .wt_data       (ffn_wt_data),

        .expert_id     (ffn_expert_id),
        .busy          (ffn_busy),
        .done          (ffn_done)
    );

    // =========================================================================
    // Bring-Up Loopback: CPU reset → idle → simple test pattern
    // During bring-up, no real PCIe or HBM2.  FFN is stimulated with
    // a simple state machine that feeds synthetic data through the engine
    // and verifies output correctness on LEDs.
    // =========================================================================

    typedef enum logic [3:0] {
        B_IDLE           = 4'h0,
        B_WAIT_PLL_LOCK  = 4'h1,
        B_INIT_WEIGHTS   = 4'h2,
        B_SEND_TEST_VEC  = 4'h3,
        B_WAIT_RESULT    = 4'h4,
        B_CHECK_RESULT   = 4'h5,
        B_PASS           = 4'h6,
        B_FAIL           = 4'h7
    } bringup_state_t;

    bringup_state_t b_state;
    logic [31:0]     cycle_cnt;
    logic [31:0]     stall_cnt;

    always_ff @(posedge clk_100m or negedge rst_n_sys) begin
        if (!rst_n_sys) begin
            b_state       <= B_IDLE;
            cycle_cnt     <= '0;
            stall_cnt     <= '0;
        end else begin
            cycle_cnt <= cycle_cnt + 1;

            case (b_state)
                B_IDLE: begin
                    if (clk_100m_locked)
                        b_state <= B_WAIT_PLL_LOCK;
                end

                B_WAIT_PLL_LOCK: begin
                    if (stall_cnt > 32'd1000)   // 10 µs @ 100 MHz
                        b_state <= B_SEND_TEST_VEC;
                    else
                        stall_cnt <= stall_cnt + 1;
                end

                B_SEND_TEST_VEC: begin
                    // Simulate: inject attention output into FFN
                    ffn_rx_valid <= 1'b1;
                    for (int i = 0; i < HIDDEN; i++)
                        ffn_rx_data[i*DATA_W +: DATA_W] <= 8'(i & 8'hFF);
                    b_state <= B_WAIT_RESULT;
                end

                B_WAIT_RESULT: begin
                    ffn_rx_valid <= 1'b0;
                    if (ffn_done)
                        b_state <= B_CHECK_RESULT;
                end

                B_CHECK_RESULT: begin
                    // Simple check: FFN output should be non-zero
                    if (|ffn_tx_data)
                        b_state <= B_PASS;
                    else
                        b_state <= B_FAIL;
                end

                B_PASS: begin
                    // Blink all LEDs in sequence — heart beat
                end

                B_FAIL: begin
                    // All LEDs on solid
                end

                default: b_state <= B_IDLE;
            endcase
        end
    end

    // =========================================================================
    // LED Controller — Debug status display
    // =========================================================================
    led_controller u_led (
        .clk         (clk_100m),
        .rst_n       (rst_n_sys),
        .pll_locked  (clk_100m_locked),
        .ffn_busy    (ffn_busy),
        .ffn_done    (ffn_done),
        .bringup_state (b_state),
        .led         (led)
    );

endmodule
