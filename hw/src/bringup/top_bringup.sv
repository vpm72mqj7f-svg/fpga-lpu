//=============================================================================
// top_bringup.sv — Board Bring-Up Top Level
//
// Orchestrates go/no-go validation sequence for first FPGA power-on.
//
// Test Sequence (LED-coded):
//   LED[3:0] = 0000  → Power-on reset complete, waiting for start
//   LED[3:0] = 0001  → Test 1: HBM2e Bandwidth    [GO/NO-GO #1]
//   LED[3:0] = 0010  → Test 2: DSP Array Accuracy  [GO/NO-GO #2]
//   LED[3:0] = 0011  → Test 3: PCIe DMA Throughput [GO/NO-GO #3]
//   LED[3:0] = 0100  → Test 4: C2C Ring Link       [GO/NO-GO #4]
//   LED[3:0] = 0101  → Test 5: Full Layer Pipeline  [GO/NO-GO #5]
//   LED[3:0] = 1111  → ALL TESTS PASSED
//   LED[3:0] = 1010  → TEST FAILED (check UART)
//
// Start: Press button or send 'S' via UART
// Abort: Press reset or send 'A' via UART
//=============================================================================

module top_bringup #(
    parameter int CLK_FREQ_HZ = 100_000_000
) (
    // ── Board Control ──
    input  logic        clk_board_100m,
    input  logic        cpu_reset_n,
    input  logic        start_button,       // pushbutton to start test sequence

    // ── Debug ──
    output logic [7:0]  debug_led,
    output logic        uart_tx,
    input  logic        uart_rx,

    // ── HBM2e AXI4 (placeholder — replace with QSYS port list) ──
    // [QSYS] output logic [31:0]  hbm_awaddr, ...
    // [QSYS] input  logic [255:0] hbm_rdata, ...

    // ── PCIe R-Tile (placeholder — replace with QSYS port list) ──
    // [QSYS] input  logic [15:0] pcie_rx_p, pcie_rx_n, ...

    // ── C2C F-Tile (placeholder — replace with QSYS port list) ──
    // [QSYS] output logic [3:0] c2c_tx_a_p, c2c_tx_a_n, ...
);

    //=========================================================================
    // Clock & Reset
    //=========================================================================
    logic clk_sys;
    logic rst_n_sys;

    // [TODO: PLL] Replace with actual PLL instantiation
    assign clk_sys = clk_board_100m;

    // Reset synchronizer (3-stage)
    logic [2:0] rst_sr;
    always_ff @(posedge clk_sys or negedge cpu_reset_n) begin
        if (!cpu_reset_n) rst_sr <= '0;
        else              rst_sr <= {rst_sr[1:0], 1'b1};
    end
    assign rst_n_sys = rst_sr[2];

    //=========================================================================
    // LED heartbeat + status
    //=========================================================================
    logic [26:0] heartbeat_cnt;

    always_ff @(posedge clk_sys or negedge rst_n_sys) begin
        if (!rst_n_sys) heartbeat_cnt <= '0;
        else            heartbeat_cnt <= heartbeat_cnt + 1'b1;
    end

    //=========================================================================
    // UART Debug Console
    //=========================================================================
    logic        uart_print_req;
    logic [7:0]  uart_print_char;
    logic        uart_print_ready;

    uart_debug #(.CLK_FREQ_HZ(CLK_FREQ_HZ)) u_uart (
        .clk(clk_sys), .rst_n(rst_n_sys),
        .print_req(uart_print_req), .print_char(uart_print_char),
        .print_ready(uart_print_ready),
        .uart_tx(uart_tx)
    );

    //=========================================================================
    // Bring-Up Sequencer FSM
    //=========================================================================
    typedef enum logic [3:0] {
        SEQ_POR,              // 0: Power-on reset
        SEQ_WAIT_START,       // 1: Wait for start button
        SEQ_HBM_BW,           // 2: HBM2e bandwidth test
        SEQ_HBM_CHECK,        // 3: Check HBM result
        SEQ_DSP_ACC,          // 4: DSP accuracy test
        SEQ_DSP_CHECK,        // 5: Check DSP result
        SEQ_PCIE_DMA,         // 6: PCIe DMA test
        SEQ_PCIE_CHECK,       // 7: Check PCIe result
        SEQ_C2C_LINK,         // 8: C2C ring test
        SEQ_C2C_CHECK,        // 9: Check C2C result
        SEQ_LAYER_PIPE,       // 10: Full layer pipeline
        SEQ_LAYER_CHECK,      // 11: Check layer result
        SEQ_ALL_PASS,         // 12: All tests passed
        SEQ_FAIL              // 13: Test failed
    } seq_state_t;
    seq_state_t seq_state;

    // Test control/status
    logic        test_start;
    logic        test_done;
    logic [1:0]  test_result;    // 2=GO, 3=NO-GO
    logic [31:0] test_metric_0;  // generic metric (e.g., bandwidth)
    logic [31:0] test_metric_1;

    // UART message buffer
    logic [7:0]  msg_char;
    logic        msg_send;
    logic [3:0]  test_id_display;

    //=========================================================================
    // UART Message Printer (simple state machine)
    //=========================================================================
    typedef enum logic [1:0] { UM_IDLE, UM_SEND, UM_WAIT } uart_msg_state_t;
    uart_msg_state_t um_state;
    logic [7:0]  um_char;
    logic        um_trigger;
    logic [7:0]  msg_buf [0:63];
    logic [5:0]  msg_len;
    logic [5:0]  msg_idx;

    always_ff @(posedge clk_sys or negedge rst_n_sys) begin
        if (!rst_n_sys) begin
            um_state   <= UM_IDLE;
            uart_print_req <= 1'b0;
            uart_print_char <= '0;
            msg_idx <= '0;
        end else begin
            uart_print_req <= 1'b0;
            case (um_state)
                UM_IDLE: begin
                    if (um_trigger) begin
                        msg_idx <= '0;
                        um_state <= UM_SEND;
                    end
                end
                UM_SEND: begin
                    if (uart_print_ready && !uart_print_req) begin
                        uart_print_req  <= 1'b1;
                        uart_print_char <= msg_buf[msg_idx];
                        if (msg_idx == msg_len - 1) begin
                            um_state <= UM_IDLE;
                        end else begin
                            msg_idx <= msg_idx + 1;
                            um_state <= UM_WAIT;
                        end
                    end
                end
                UM_WAIT: begin
                    if (uart_print_ready) begin
                        um_state <= UM_SEND;
                    end
                end
                default: um_state <= UM_IDLE;
            endcase
        end
    end

    // Print a string
    task automatic uart_print_string(input string s);
        integer i;
        begin
            for (i = 0; i < s.len(); i = i + 1) begin
                msg_buf[i] = s[i];
            end
            msg_len = s.len();
            um_trigger = 1'b1;
            @(posedge clk_sys);
            um_trigger = 1'b0;
            // Wait for print to complete
            while (um_state != UM_IDLE) @(posedge clk_sys);
        end
    endtask

    //=========================================================================
    // Main Bring-Up Sequencer
    //=========================================================================
    logic [31:0] fail_code;
    logic        start_debounce;
    logic [7:0]  start_sr;

    // Start button debounce
    always_ff @(posedge clk_sys or negedge rst_n_sys) begin
        if (!rst_n_sys) begin
            start_sr      <= '0;
            start_debounce <= 1'b0;
        end else begin
            start_sr <= {start_sr[6:0], start_button};
            start_debounce <= &start_sr;  // stable high for 8 cycles
        end
    end

    always_ff @(posedge clk_sys or negedge rst_n_sys) begin
        if (!rst_n_sys) begin
            seq_state       <= SEQ_POR;
            test_start      <= 1'b0;
            test_id_display <= 4'd0;
            fail_code       <= '0;
        end else begin
            test_start <= 1'b0;

            case (seq_state)
                SEQ_POR: begin
                    test_id_display <= 4'd0;
                    seq_state <= SEQ_WAIT_START;
                end

                SEQ_WAIT_START: begin
                    // LED: heartbeat on bit 0, waiting on bits 3:1
                    if (start_debounce) begin
                        test_id_display <= 4'd1;
                        test_start <= 1'b1;
                        seq_state <= SEQ_HBM_BW;
                    end
                end

                // === TEST 1: HBM2e Bandwidth ===
                SEQ_HBM_BW: begin
                    test_start <= 1'b0;
                    if (test_done) begin
                        seq_state <= SEQ_HBM_CHECK;
                    end
                end

                SEQ_HBM_CHECK: begin
                    if (test_result == 2'd2) begin  // GO
                        test_id_display <= 4'd2;
                        test_start <= 1'b1;
                        seq_state <= SEQ_DSP_ACC;
                    end else begin
                        fail_code <= 32'd1;
                        seq_state <= SEQ_FAIL;
                    end
                end

                // === TEST 2: DSP Array Accuracy ===
                SEQ_DSP_ACC: begin
                    test_start <= 1'b0;
                    if (test_done) begin
                        seq_state <= SEQ_DSP_CHECK;
                    end
                end

                SEQ_DSP_CHECK: begin
                    if (test_result == 2'd2) begin
                        test_id_display <= 4'd3;
                        // PCIe only on master chip
                        if (1) begin  // [TODO] check IS_PCIE_MASTER
                            test_start <= 1'b1;
                            seq_state <= SEQ_PCIE_DMA;
                        end else begin
                            seq_state <= SEQ_C2C_LINK;
                        end
                    end else begin
                        fail_code <= 32'd2;
                        seq_state <= SEQ_FAIL;
                    end
                end

                // === TEST 3: PCIe DMA Throughput ===
                SEQ_PCIE_DMA: begin
                    test_start <= 1'b0;
                    if (test_done) begin
                        seq_state <= SEQ_PCIE_CHECK;
                    end
                end

                SEQ_PCIE_CHECK: begin
                    if (test_result != 2'd3) begin  // not NO-GO (WARN is OK)
                        test_id_display <= 4'd4;
                        test_start <= 1'b1;
                        seq_state <= SEQ_C2C_LINK;
                    end else begin
                        fail_code <= 32'd3;
                        seq_state <= SEQ_FAIL;
                    end
                end

                // === TEST 4: C2C Ring Link ===
                SEQ_C2C_LINK: begin
                    test_start <= 1'b0;
                    if (test_done) begin
                        seq_state <= SEQ_C2C_CHECK;
                    end
                end

                SEQ_C2C_CHECK: begin
                    if (test_result != 2'd3) begin
                        test_id_display <= 4'd5;
                        test_start <= 1'b1;
                        seq_state <= SEQ_LAYER_PIPE;
                    end else begin
                        fail_code <= 32'd4;
                        seq_state <= SEQ_FAIL;
                    end
                end

                // === TEST 5: Full Layer Pipeline ===
                SEQ_LAYER_PIPE: begin
                    test_start <= 1'b0;
                    if (test_done) begin
                        seq_state <= SEQ_LAYER_CHECK;
                    end
                end

                SEQ_LAYER_CHECK: begin
                    if (test_result != 2'd3) begin
                        seq_state <= SEQ_ALL_PASS;
                    end else begin
                        fail_code <= 32'd5;
                        seq_state <= SEQ_FAIL;
                    end
                end

                // === Terminal states ===
                SEQ_ALL_PASS: begin
                    test_id_display <= 4'hF;  // 1111 = ALL PASS
                    // Stay here, heartbeat continues
                end

                SEQ_FAIL: begin
                    test_id_display <= 4'hA;  // 1010 = FAIL
                    // Stay here, UART reports fail_code
                end

                default: seq_state <= SEQ_POR;
            endcase
        end
    end

    //=========================================================================
    // LED Display
    //=========================================================================
    // LED[0]   = heartbeat (~1 Hz)
    // LED[3:1] = test_id_display[2:0] during test, 111 after all pass
    // LED[4]   = test_result[0] (0=running, 1=done)
    // LED[5]   = GO (green)
    // LED[6]   = NO-GO (red)
    // LED[7]   = UART TX activity

    assign debug_led[0]   = heartbeat_cnt[26];              // ~0.75 Hz blink
    assign debug_led[3:1] = test_id_display[2:0];
    assign debug_led[4]   = test_done;
    assign debug_led[5]   = (test_result == 2'd2);           // GO
    assign debug_led[6]   = (test_result == 2'd3);           // NO-GO
    assign debug_led[7]   = uart_print_req;                  // UART activity

endmodule
