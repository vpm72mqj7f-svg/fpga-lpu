// =============================================================================
// v2_lite_top.sv — DeepSeek V2-Lite FFN Top-Level
// Target: S10 MX Dev Kit (1SM21BHU2F53E1VG)
//
// V2-Lite: 16B params, 27 layers
//   hidden=2048, inter=1408, 64 routed + 2 shared experts, TOP_K=6, FP8
//
// Architecture: CPU Attention → PCIe → HBM2 (weights) → FFN Engine → PCIe → CPU
// =============================================================================

module v2_lite_top (
    input  logic        clk_sys_100m_p,
    input  logic        clk_sys_100m_n,
    input  logic        cpu_reset_n,
    output logic [3:0]  led,
    input  logic        pcie_ep_refclk_p,
    input  logic        pcie_ep_refclk_n,
    input  logic        pcie_ep_perst_n,
    output logic        pcie_ep_wake_n
);

    // =========================================================================
    // V2-Lite Model Parameters
    // =========================================================================
    localparam int HIDDEN      = 2048;
    localparam int INTER       = 1408;
    localparam int NUM_EXPERTS = 66;     // 64 routed + 2 shared
    localparam int TOP_K       = 6;
    localparam int DATA_W      = 8;      // FP8

    // =========================================================================
    // Clock & Reset
    // =========================================================================
    logic clk_100m, clk_core, clk_dsp, pll_locked, rst_n_sys, rst_n_core;
    assign clk_100m = clk_sys_100m_p;

    pll_controller u_pll (
        .refclk   (clk_100m),
        .rst_n    (cpu_reset_n),
        .clk_500m (clk_core),
        .clk_250m (clk_dsp),
        .locked   (pll_locked)
    );

    reset_controller u_rst (
        .async_rst_n (cpu_reset_n),
        .pll_locked  (pll_locked),
        .clk_100m    (clk_100m),
        .clk_500m    (clk_core),
        .clk_250m    (clk_dsp),
        .rst_n_sys   (rst_n_sys),
        .rst_n_core  (rst_n_core)
    );

    // =========================================================================
    // FFN Engine
    // =========================================================================
    logic                          ffn_rx_valid, ffn_rx_ready;
    logic [HIDDEN*DATA_W-1:0]      ffn_rx_data;
    logic                          ffn_tx_valid, ffn_tx_ready;
    logic [HIDDEN*DATA_W-1:0]      ffn_tx_data;
    logic                          ffn_busy, ffn_done;
    logic [$clog2(NUM_EXPERTS)-1:0] ffn_expert_id [TOP_K];
    logic                          ffn_wt_wr_en;
    logic [$clog2(NUM_EXPERTS)-1:0] ffn_wt_expert_id;
    logic [1:0]                    ffn_wt_type;
    logic [$clog2(INTER)-1:0]      ffn_wt_row;
    logic [$clog2(HIDDEN)-1:0]     ffn_wt_col;
    logic [DATA_W-1:0]             ffn_wt_data;

    v2_lite_ffn_engine #(
        .HIDDEN      (HIDDEN),
        .INTER       (INTER),
        .NUM_EXPERTS (NUM_EXPERTS),
        .TOP_K       (TOP_K),
        .DATA_W      (DATA_W)
    ) u_ffn (
        .clk          (clk_core),
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
    // Bring-Up Self-Test FSM
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
                B_IDLE:  if (pll_locked) b_state <= B_WAIT_PLL;
                B_WAIT_PLL: b_state <= B_SEND;
                B_SEND: begin
                    ffn_rx_valid <= 1'b1;
                    for (int i = 0; i < HIDDEN; i++)
                        ffn_rx_data[i*DATA_W +: DATA_W] <= 8'(i);
                    b_state <= B_WAIT;
                end
                B_WAIT: begin
                    ffn_rx_valid <= 1'b0;
                    if (ffn_done) b_state <= B_CHECK;
                end
                B_CHECK: b_state <= (|ffn_tx_data) ? B_PASS : B_FAIL;
                B_PASS, B_FAIL: ; // latched
                default: b_state <= B_IDLE;
            endcase
        end
    end

    // =========================================================================
    // LED Debug Display
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

    assign pcie_ep_wake_n = 1'b1;  // unused during bringup

endmodule
