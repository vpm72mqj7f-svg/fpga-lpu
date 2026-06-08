// =============================================================================
// v2_lite_top.sv — Intel reference pinout (same as qts_hbm_top)
// Port names match QSF assignments from Intel DK-DEV-1SMX-H-A reference.
// =============================================================================
module v2_lite_top (
    input  wire        core_clk_iopll_ref_clk_clk,       // 100MHz, PIN_AU17/AU16
    input  wire        cpu_resetn,                         // PIN_BL14
    output logic [3:0] led                                 // BG12/BF12/BG11/BH11
);
    localparam int HIDDEN=2048, INTER=1408, NUM_EXPERTS=66, TOP_K=6, DATA_W=8;

    logic clk_100m, clk_core, pll_locked, rst_n_sys, rst_n_core;
    assign clk_100m = core_clk_iopll_ref_clk_clk;
    assign clk_core = clk_100m;
    logic [7:0] pc = 0;
    always_ff @(posedge clk_100m) if (pc < 255) pc <= pc + 1;
    assign pll_locked = (pc == 255);

    reset_controller u_rst (
        .async_rst_n(cpu_resetn), .pll_locked,
        .clk_100m, .clk_500m(clk_core), .clk_250m(clk_core),
        .rst_n_sys, .rst_n_core
    );

    logic pcie_rx_valid, pcie_rx_ready, pcie_tx_valid, pcie_tx_ready, ffn_busy, ffn_done;
    logic [HIDDEN*DATA_W-1:0] pcie_rx_data, pcie_tx_data;
    logic [$clog2(NUM_EXPERTS)-1:0] ffn_expert_id [TOP_K];

    v2_lite_ffn_engine #(
        .HIDDEN(HIDDEN), .INTER(INTER), .NUM_EXPERTS(NUM_EXPERTS),
        .TOP_K(TOP_K), .DATA_W(DATA_W)
    ) u_ffn (
        .clk(clk_core), .rst_n(rst_n_core),
        .pcie_rx_valid, .pcie_rx_data, .pcie_rx_ready,
        .pcie_tx_valid, .pcie_tx_data, .pcie_tx_ready,
        .m_axi_araddr(), .m_axi_arlen(), .m_axi_arsize(),
        .m_axi_arvalid(), .m_axi_arready(1'b0),
        .m_axi_rdata(256'd0), .m_axi_rresp(2'd0),
        .m_axi_rvalid(1'b0), .m_axi_rready(), .m_axi_rlast(1'b0),
        .expert_id(ffn_expert_id), .busy(ffn_busy), .done(ffn_done)
    );

    typedef enum logic [3:0] {B_IDLE, B_WAIT_PLL, B_SEND, B_WAIT, B_CHECK, B_PASS, B_FAIL} bst_t;
    bst_t bst = B_IDLE;
    always_ff @(posedge clk_100m or negedge rst_n_sys)
        if (!rst_n_sys) bst <= B_IDLE;
        else case (bst)
            B_IDLE:      if (pll_locked) bst <= B_WAIT_PLL;
            B_WAIT_PLL:  bst <= B_SEND;
            B_SEND: begin
                pcie_rx_valid <= 1'b1;
                for (int i = 0; i < HIDDEN; i++)
                    pcie_rx_data[i*DATA_W +: DATA_W] <= 8'(i);
                bst <= B_WAIT;
            end
            B_WAIT: begin
                pcie_rx_valid <= 1'b0;
                if (ffn_done) bst <= B_CHECK;
            end
            B_CHECK: bst <= (|pcie_tx_data) ? B_PASS : B_FAIL;
            B_PASS, B_FAIL: ;
        endcase

    assign pcie_tx_ready = 1'b1;
    generate for (genvar e = 0; e < TOP_K; e++) assign ffn_expert_id[e] = e; endgenerate

    led_controller u_led (.clk(clk_100m), .rst_n(rst_n_sys), .pll_locked, .ffn_busy, .ffn_done, .bringup_state(bst), .led);
endmodule
