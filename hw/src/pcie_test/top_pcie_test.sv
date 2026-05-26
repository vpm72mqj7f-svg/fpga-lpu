//=============================================================================
// top_pcie_test.sv — PCIe 5.0 DMA Throughput Test (Master Chip Only)
//
// Purpose:  Measure PCIe 5.0 x16 DMA bandwidth in both directions.
//           Host must run companion driver (scripts/pcie_dma_test.py).
//
// Go/No-Go: H2D >= 28 GB/s, D2H >= 28 GB/s
//=============================================================================

module top_pcie_test (
    input  logic        clk_board_100m,
    input  logic        cpu_reset_n,
    input  logic        start_button,
    output logic [7:0]  debug_led,
    output logic        uart_tx,
    input  logic        uart_rx
    // [QSYS] PCIe R-Tile ports (16 lanes)
    // [QSYS] HBM2e AXI4 ports (DMA target buffer)
);

    logic clk_sys, clk_pcie, clk_hbm;
    logic rst_n_sys;
    assign clk_sys  = clk_board_100m;
    assign clk_pcie = clk_board_100m;  // [TODO: PLL 250 MHz]
    assign clk_hbm  = clk_board_100m;  // [TODO: PLL 450 MHz]

    logic [2:0] rst_sr;
    always_ff @(posedge clk_sys or negedge cpu_reset_n) begin
        if (!cpu_reset_n) rst_sr <= '0;
        else              rst_sr <= {rst_sr[1:0], 1'b1};
    end
    assign rst_n_sys = rst_sr[2];

    logic [26:0] hb_cnt;
    always_ff @(posedge clk_sys) if (!rst_n_sys) hb_cnt <= '0; else hb_cnt <= hb_cnt + 1'b1;

    logic uart_req, uart_ready;
    logic [7:0] uart_char;
    uart_debug u_uart (.clk(clk_sys), .rst_n(rst_n_sys),
        .print_req(uart_req), .print_char(uart_char),
        .print_ready(uart_ready), .uart_tx(uart_tx));

    //=========================================================================
    // PCIe DMA Test FSM
    //
    // Phase 1: Host → FPGA (H2D write): host writes 1 GB test buffer
    // Phase 2: FPGA → Host (D2H read):  host reads 1 GB back
    //=========================================================================
    typedef enum logic [2:0] { S_IDLE, S_WAIT_H2D, S_CHECK_H2D, S_WAIT_D2H, S_CHECK_D2H, S_DONE } st_t;
    st_t st;

    logic [31:0] bytes_received, bytes_sent;
    logic [63:0] cycle_count;
    logic [31:0] h2d_bw_mb_s, d2h_bw_mb_s;
    logic [1:0]  test_result;

    always_ff @(posedge clk_sys or negedge rst_n_sys) begin
        if (!rst_n_sys) begin
            st <= S_IDLE;
            bytes_received <= '0; bytes_sent <= '0;
            cycle_count <= '0;
            h2d_bw_mb_s <= '0; d2h_bw_mb_s <= '0;
            test_result <= 2'd0;
        end else begin
            case (st)
                S_IDLE: if (start_button) begin
                    cycle_count <= '0;
                    test_result <= 2'd1;
                    st <= S_WAIT_H2D;
                end
                S_WAIT_H2D: begin
                    cycle_count <= cycle_count + 1;
                    // [QSYS] Monitor PCIe RX DMA engine: bytes_received from BAR0
                    if (bytes_received >= 32'd1_073_741_824) st <= S_CHECK_H2D;  // 1 GB
                end
                S_CHECK_H2D: begin
                    h2d_bw_mb_s <= bytes_received / (cycle_count / 250);  // approx
                    st <= S_WAIT_D2H;
                end
                S_WAIT_D2H: begin
                    cycle_count <= cycle_count + 1;
                    // [QSYS] Monitor PCIe TX DMA engine
                    if (bytes_sent >= 32'd1_073_741_824) st <= S_CHECK_D2H;
                end
                S_CHECK_D2H: begin
                    d2h_bw_mb_s <= bytes_sent / (cycle_count / 250);
                    test_result <= (h2d_bw_mb_s >= 28000 && d2h_bw_mb_s >= 28000) ? 2'd2 : 2'd3;
                    st <= S_DONE;
                end
                S_DONE: ;
                default: st <= S_IDLE;
            endcase
        end
    end

    assign debug_led[0]   = hb_cnt[26];
    assign debug_led[3:1] = 3'd3;
    assign debug_led[4]   = (st == S_DONE);
    assign debug_led[5]   = (test_result == 2'd2);
    assign debug_led[6]   = (test_result == 2'd3);
    assign debug_led[7]   = uart_req;

endmodule
