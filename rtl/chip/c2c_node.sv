//=============================================================================
// c2c_node.sv — minimal C2C ring node for multi-chip pipeline bring-up
//
// Receives pipeline-forward beats from previous chip, processes them
// (simulates layer compute by incrementing a data counter), and forwards
// to the next chip or back to Host.
//=============================================================================

module c2c_node #(parameter int NODE_ID = 0) (
    input  logic clk, rst_n,

    // Ring RX (from previous chip)
    input  logic        rx_valid,
    output logic        rx_ready,
    input  logic [7:0]  rx_dst,
    input  logic [15:0] rx_token_id,
    input  logic [31:0] rx_data,
    input  logic        rx_last,

    // Ring TX (to next chip)
    output logic        tx_valid,
    input  logic        tx_ready,
    output logic [7:0]  tx_dst,
    output logic [15:0] tx_token_id,
    output logic [31:0] tx_data,
    output logic        tx_last,

    // Host result (for last chip in chain)
    output logic        host_valid,
    output logic [15:0] host_token_id,
    output logic [31:0] host_data
);

    typedef enum logic [1:0] {S_IDLE, S_PROCESS, S_FWD} state_t;
    state_t state;

    assign rx_ready = (state == S_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            tx_valid <= 0; host_valid <= 0;
            tx_dst <= 0; tx_token_id <= 0; tx_data <= 0; tx_last <= 0;
            host_token_id <= 0; host_data <= 0;
        end else begin
            tx_valid <= 0; host_valid <= 0;

            case (state)
                S_IDLE: if (rx_valid) begin
                    state <= S_PROCESS;
                    // Simulate layer compute: increment data by 1
                    if (rx_last) begin
                        // Last chip in chain → output to host
                        host_valid <= 1;
                        host_token_id <= rx_token_id;
                        host_data <= rx_data + 32'd1;
                        state <= S_IDLE;
                    end else begin
                        // Forward to next chip with incremented data
                        tx_valid <= 1;
                        tx_dst <= rx_dst + 8'd1;   // next chip in pipeline
                        tx_token_id <= rx_token_id;
                        tx_data <= rx_data + 32'd1;
                        tx_last <= (NODE_ID == 2); // next node (3) is last in chain
                        state <= S_FWD;
                    end
                end

                S_PROCESS: state <= S_FWD; // one extra cycle

                S_FWD: begin
                    if (tx_ready) state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
