//=============================================================================
// top_hbm_char.sv — HBM2e Characterization Project
//
// Purpose:  Measure HBM2e bandwidth across all 32 pseudo-channels.
//           Reports per-channel and aggregate bandwidth via UART.
//
// Go/No-Go: Read >= 800 GB/s, Write >= 700 GB/s aggregate
//=============================================================================

module top_hbm_char (
    input  logic        clk_board_100m,
    input  logic        cpu_reset_n,
    input  logic        start_button,
    output logic [7:0]  debug_led,
    output logic        uart_tx,
    input  logic        uart_rx
    // [QSYS] HBM2e AXI4 ports — 32 pseudo-channels
);

    // Clock & Reset
    logic clk_sys, clk_hbm;
    logic rst_n_sys;
    assign clk_sys = clk_board_100m;
    assign clk_hbm = clk_board_100m;  // [TODO: PLL 450 MHz]

    logic [2:0] rst_sr;
    always_ff @(posedge clk_sys or negedge cpu_reset_n) begin
        if (!cpu_reset_n) rst_sr <= '0;
        else              rst_sr <= {rst_sr[1:0], 1'b1};
    end
    assign rst_n_sys = rst_sr[2];

    // LED heartbeat
    logic [26:0] hb_cnt;
    always_ff @(posedge clk_sys) if (!rst_n_sys) hb_cnt <= '0; else hb_cnt <= hb_cnt + 1'b1;

    // UART
    logic uart_req, uart_ready;
    logic [7:0] uart_char;
    uart_debug u_uart (.clk(clk_sys), .rst_n(rst_n_sys),
        .print_req(uart_req), .print_char(uart_char),
        .print_ready(uart_ready), .uart_tx(uart_tx));

    // HBM BW Test
    logic test_start, test_done;
    logic [1:0] test_result;
    logic [31:0] write_bw, read_bw;
    hbm_bw_test u_hbm (.clk(clk_hbm), .rst_n(rst_n_sys),
        .start_test(test_start), .test_done(test_done),
        .test_result(test_result),
        .write_bw_mb_s(write_bw), .read_bw_mb_s(read_bw),
        .status_valid(), .status_char(),
        // [QSYS] connect AXI4 ports
        .m_axi_awid(), .m_axi_awaddr(), .m_axi_awlen(),
        .m_axi_awsize(), .m_axi_awburst(), .m_axi_awvalid(), .m_axi_awready(1'b0),
        .m_axi_wdata(), .m_axi_wstrb(), .m_axi_wlast(), .m_axi_wvalid(), .m_axi_wready(1'b0),
        .m_axi_bid('0), .m_axi_bresp('0), .m_axi_bvalid(1'b0), .m_axi_bready(),
        .m_axi_arid(), .m_axi_araddr(), .m_axi_arlen(),
        .m_axi_arsize(), .m_axi_arburst(), .m_axi_arvalid(), .m_axi_arready(1'b0),
        .m_axi_rid('0), .m_axi_rdata('0), .m_axi_rresp('0),
        .m_axi_rlast(1'b0), .m_axi_rvalid(1'b0), .m_axi_rready()
    );

    // Simple sequencer: start on button press
    logic start_d;
    always_ff @(posedge clk_sys) begin
        if (!rst_n_sys) begin test_start <= 0; start_d <= 0; end
        else begin
            start_d <= start_button;
            test_start <= start_button && !start_d;  // rising edge
        end
    end

    assign debug_led[0]   = hb_cnt[26];
    assign debug_led[3:1] = 3'd1;  // test ID = HBM
    assign debug_led[4]   = test_done;
    assign debug_led[5]   = (test_result == 2'd2);
    assign debug_led[6]   = (test_result == 2'd3);
    assign debug_led[7]   = uart_req;

endmodule
