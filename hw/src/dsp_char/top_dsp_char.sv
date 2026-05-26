//=============================================================================
// top_dsp_char.sv — DSP Array Characterization Project
//
// Purpose:  Sweep DSP array at target frequency, measure accuracy and power.
//           Runs fp4 MAC golden vector suite at 450 MHz.
//           DSP and HBM share the same clock domain.
//
// Go/No-Go: 0 errors, timing closed at 450 MHz
//=============================================================================

module top_dsp_char (
    input  logic        clk_board_100m,
    input  logic        cpu_reset_n,
    input  logic        start_button,
    output logic [7:0]  debug_led,
    output logic        uart_tx
);

    logic clk_sys, clk_dsp;
    logic rst_n_sys;
    assign clk_sys = clk_board_100m;
    assign clk_dsp = clk_board_100m;  // [TODO: PLL 450 MHz, ×9/2]

    logic [2:0] rst_sr;
    always_ff @(posedge clk_sys or negedge cpu_reset_n) begin
        if (!cpu_reset_n) rst_sr <= '0;
        else              rst_sr <= {rst_sr[1:0], 1'b1};
    end
    assign rst_n_sys = rst_sr[2];

    logic [26:0] hb_cnt;
    always_ff @(posedge clk_sys) if (!rst_n_sys) hb_cnt <= '0; else hb_cnt <= hb_cnt + 1'b1;

    // UART
    logic uart_req, uart_ready;
    logic [7:0] uart_char;
    uart_debug u_uart (.clk(clk_sys), .rst_n(rst_n_sys),
        .print_req(uart_req), .print_char(uart_char),
        .print_ready(uart_ready), .uart_tx(uart_tx));

    // DSP Stress Test
    logic test_start, test_done;
    logic [1:0] test_result;
    logic [31:0] errors, checked;
    logic scale_wr_en;
    logic [8:0] scale_addr;
    logic [7:0] scale_data;

    dsp_stress_test #(.TEST_MODE(0)) u_dsp (
        .clk(clk_dsp), .rst_n(rst_n_sys),
        .start_test(test_start), .test_done(test_done),
        .test_result(test_result),
        .errors_detected(errors), .vectors_checked(checked),
        .scale_wr_en(scale_wr_en), .scale_wr_addr(scale_addr),
        .scale_wr_data(scale_data),
        .status_valid(), .status_char()
    );

    logic start_d;
    always_ff @(posedge clk_sys) begin
        if (!rst_n_sys) begin test_start <= 0; start_d <= 0; end
        else begin start_d <= start_button; test_start <= start_button && !start_d; end
    end

    assign debug_led[0]   = hb_cnt[26];
    assign debug_led[3:1] = 3'd2;
    assign debug_led[4]   = test_done;
    assign debug_led[5]   = (test_result == 2'd2);
    assign debug_led[6]   = (test_result == 2'd3);
    assign debug_led[7]   = uart_req;

endmodule
