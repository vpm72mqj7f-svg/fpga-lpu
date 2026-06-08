// =============================================================================
// v4_flash_top.sv — Intel reference pinout (same as qts_hbm_top)
// =============================================================================
module v4_flash_top (
    input  wire        core_clk_iopll_ref_clk_clk,
    input  wire        hbm_0_example_design_pll_ref_clk_clk,
    input  wire        cpu_resetn,
    output logic [3:0] led
);
    localparam int HIDDEN=7168, INTER=3072, NUM_EXPERTS=385, TOP_K=6, DATA_W=8;

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

    v4_flash_ffn_engine #(
        .HIDDEN(HIDDEN), .INTER(INTER), .NUM_EXPERTS(NUM_EXPERTS),
        .TOP_K(TOP_K), .DATA_W(DATA_W)
    ) u_ffn (
        .clk(clk_core), .rst_n(rst_n_core),
        .pcie_rx_valid, .pcie_rx_data, .pcie_rx_ready,
        .pcie_tx_valid, .pcie_tx_data, .pcie_tx_ready,
        .wt_wr_en(1'b0), .wt_expert_id('0), .wt_type(2'b0),
        .wt_row('0), .wt_col('0), .wt_data(8'd0),
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
                for (int _c = 0; _c < HIDDEN; _c += 2048)
                    for (int i = _c; i < _c + 2048 && i < HIDDEN; i++)
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
