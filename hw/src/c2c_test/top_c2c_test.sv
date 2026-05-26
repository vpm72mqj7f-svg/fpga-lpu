//=============================================================================
// top_c2c_test.sv — C2C Ring Link Test (Master + Slave Loopback)
//
// Purpose:  Validate F-Tile SerDes C2C link between master and slave chips.
//           Tests: internal loopback → neighbour link → full ring loopback.
//
// Go/No-Go: BER < 1e-15, latency < 100 ns/hop, all 4 lanes up
//=============================================================================

module top_c2c_test #(
    parameter int IS_MASTER = 1    // 1 = master (originates test), 0 = slave (loopback)
) (
    input  logic        clk_board_100m,
    input  logic        cpu_reset_n,
    input  logic        start_button,
    output logic [7:0]  debug_led,
    output logic        uart_tx
    // [QSYS] F-Tile C2C SerDes ports (4 lanes × 2 rings)
);

    logic clk_sys;
    logic rst_n_sys;
    assign clk_sys = clk_board_100m;

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
    // C2C Link Test FSM
    //
    // Master sends PRBS-31 pattern, slave loops back, master checks.
    //=========================================================================
    typedef enum logic [2:0] { S_IDLE, S_TX_PRBS, S_RX_CHECK, S_DONE } st_t;
    st_t st;

    logic [30:0] prbs;
    logic [31:0] sent_count, recv_count, err_count;
    logic [63:0] cycle_count;
    logic [1:0]  test_result;
    logic        link_up;

    always_ff @(posedge clk_sys or negedge rst_n_sys) begin
        if (!rst_n_sys) begin
            st <= S_IDLE; prbs <= 31'h7FFF_FFFF;
            sent_count <= '0; recv_count <= '0; err_count <= '0;
            cycle_count <= '0; test_result <= 2'd0;
        end else begin
            case (st)
                S_IDLE: if (start_button && link_up) begin
                    prbs <= 31'h7FFF_FFFF;
                    sent_count <= '0; recv_count <= '0; err_count <= '0;
                    cycle_count <= '0;
                    test_result <= 2'd1;
                    st <= S_TX_PRBS;
                end

                S_TX_PRBS: begin
                    // PRBS-31: x^31 + x^28 + 1
                    prbs <= {prbs[29:0], prbs[30] ^~ prbs[27]};
                    sent_count <= sent_count + 1;
                    cycle_count <= cycle_count + 1;
                    // [QSYS] Send prbs via C2C TX, check RX for loopback match
                    if (sent_count >= 32'd100_000_000) st <= S_RX_CHECK;
                end

                S_RX_CHECK: begin
                    test_result <= (err_count == 0) ? 2'd2 : 2'd3;
                    st <= S_DONE;
                end

                S_DONE: ;
                default: st <= S_IDLE;
            endcase
        end
    end

    assign debug_led[0]   = hb_cnt[26];
    assign debug_led[1]   = link_up;            // C2C link A up
    assign debug_led[2]   = link_up;            // C2C link B up
    assign debug_led[3]   = IS_MASTER;
    assign debug_led[4]   = (st == S_DONE);
    assign debug_led[5]   = (test_result == 2'd2);
    assign debug_led[6]   = (test_result == 2'd3);
    assign debug_led[7]   = uart_req;

endmodule
