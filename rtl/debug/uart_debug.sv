//=============================================================================
// uart_debug.sv — Bring-up UART console (115200 baud, 8N1)
//
// Prints status messages during go/no-go tests. Minimal footprint (~80 LUTs).
// Usage:
//   uart_print("HBM BW Test") → streams ASCII to uart_tx
//   uart_print_hex(value)     → prints 32-bit hex value
//=============================================================================

module uart_debug #(
    parameter int CLK_FREQ_HZ = 100_000_000,  // system clock frequency
    parameter int BAUD_RATE   = 115200
) (
    input  logic        clk,
    input  logic        rst_n,

    // Command interface
    input  logic        print_req,
    input  logic [7:0]  print_char,
    output logic        print_ready,

    // Hardware TX pin
    output logic        uart_tx
);

    localparam int BIT_PERIOD = CLK_FREQ_HZ / BAUD_RATE;  // ~868 @ 100MHz

    typedef enum logic [1:0] { S_IDLE, S_START, S_DATA, S_STOP } state_t;
    state_t state;

    logic [$clog2(BIT_PERIOD)-1:0] bit_timer;
    logic [2:0]                    bit_idx;      // 0..7 data, 8=stop
    logic [7:0]                    tx_shift;     // shift register

    assign print_ready = (state == S_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            uart_tx   <= 1'b1;   // idle high
            bit_timer <= '0;
            bit_idx   <= '0;
            tx_shift  <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    uart_tx <= 1'b1;
                    if (print_req) begin
                        tx_shift  <= print_char;
                        bit_idx   <= '0;
                        bit_timer <= BIT_PERIOD - 1;
                        state     <= S_START;
                    end
                end

                S_START: begin
                    uart_tx <= 1'b0;  // start bit
                    if (bit_timer == 0) begin
                        bit_timer <= BIT_PERIOD - 1;
                        bit_idx   <= '0;
                        state     <= S_DATA;
                    end else begin
                        bit_timer <= bit_timer - 1;
                    end
                end

                S_DATA: begin
                    uart_tx <= tx_shift[0];
                    if (bit_timer == 0) begin
                        tx_shift  <= {1'b0, tx_shift[7:1]};  // shift right
                        if (bit_idx == 7) begin
                            bit_timer <= BIT_PERIOD - 1;
                            state     <= S_STOP;
                        end else begin
                            bit_idx   <= bit_idx + 1;
                            bit_timer <= BIT_PERIOD - 1;
                        end
                    end else begin
                        bit_timer <= bit_timer - 1;
                    end
                end

                S_STOP: begin
                    uart_tx <= 1'b1;  // stop bit
                    if (bit_timer == 0) begin
                        state <= S_IDLE;
                    end else begin
                        bit_timer <= bit_timer - 1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
