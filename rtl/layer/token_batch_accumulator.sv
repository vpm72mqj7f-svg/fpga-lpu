//=============================================================================
// token_batch_accumulator.sv — Token batch accumulation for decode efficiency
//
// Accumulates incoming tokens until BATCH_MIN reached or timeout expires,
// then dispatches them as a batch. This allows the downstream pipeline to
// reuse expert weights across B tokens, amortizing HBM weight loading.
//
// Roofline rationale: At B=1, OI=2.8 MACs/byte (bandwidth-bound, 78% DSP idle).
// At B>=6, OI>=14.9 MACs/byte (compute-bound, DSP fully utilized).
//=============================================================================

`include "lpu_config.svh"

module token_batch_accumulator #(
    parameter int MAX_BATCH       = 32,
    parameter int BATCH_MIN       = 6,
    parameter int DATA_W          = 256,    // HIDDEN*32 for production, 256 for bring-up
    parameter int TIMEOUT_CYCLES  = 5_000_000  // 50ms @ 100MHz
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Input: tokens arrive one at a time from upstream
    input  logic                         valid_in,
    input  logic [DATA_W-1:0]            data_in,
    output logic                         in_ready,

    // Output: tokens dispatched as a batch
    output logic                         valid_out,
    output logic [DATA_W-1:0]            data_out,
    input  logic                         out_ready,

    // Batch status for downstream pipeline optimization
    output logic                         batch_active,    // 1 during batch dispatch
    output logic [$clog2(MAX_BATCH):0]   batch_size,      // tokens in current batch
    output logic                         batch_first,     // first token of batch
    output logic                         batch_last       // last token of batch
);

    typedef enum logic [1:0] { S_IDLE, S_ACCUMULATE, S_DISPATCH } state_t;
    state_t state;

    // Token FIFO storage
    logic [DATA_W-1:0] fifo [MAX_BATCH];
    logic [$clog2(MAX_BATCH)-1:0] fifo_wr_ptr, fifo_rd_ptr;
    logic [$clog2(MAX_BATCH):0]   fifo_count;     // 0..MAX_BATCH

    // Timeout counter (cycles since first token in accumulate phase)
    logic [31:0] timer;

    assign in_ready = (state == S_IDLE) || ((state == S_ACCUMULATE) && (fifo_count < MAX_BATCH));

    // FIFO write
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_wr_ptr <= '0;
            for (int i = 0; i < MAX_BATCH; i++) fifo[i] <= '0;
        end else begin
            if (valid_in && in_ready) begin
                fifo[fifo_wr_ptr] <= data_in;
                fifo_wr_ptr <= (fifo_wr_ptr == (MAX_BATCH - 1)) ? '0 : (fifo_wr_ptr + 1'b1);
            end
        end
    end

    // FIFO read + batch control
    // _pos tracks which token (0-indexed) is being output this cycle
    logic [$clog2(MAX_BATCH)-1:0] output_pos;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            fifo_rd_ptr  <= '0;
            fifo_count   <= '0;
            timer        <= '0;
            valid_out    <= 1'b0;
            batch_active <= 1'b0;
            batch_first  <= 1'b0;
            batch_last   <= 1'b0;
            batch_size   <= '0;
            output_pos   <= '0;
        end else begin
            valid_out   <= 1'b0;
            batch_first <= 1'b0;
            batch_last  <= 1'b0;

            case (state)
                S_IDLE: begin
                    fifo_rd_ptr <= '0;
                    fifo_count  <= '0;
                    timer       <= '0;
                    batch_active <= 1'b0;
                    output_pos  <= '0;

                    if (valid_in) begin
                        fifo[fifo_wr_ptr] <= data_in;
                        fifo_wr_ptr <= 1'b1;
                        fifo_count  <= 1'b1;
                        fifo_rd_ptr <= '0;
                        timer       <= '0;
                        state <= S_ACCUMULATE;
                    end
                end

                S_ACCUMULATE: begin
                    timer <= timer + 1'b1;
                    if (valid_in && (fifo_count < MAX_BATCH)) begin
                        fifo[fifo_wr_ptr] <= data_in;
                        fifo_wr_ptr <= (fifo_wr_ptr == (MAX_BATCH - 1)) ? '0 : (fifo_wr_ptr + 1'b1);
                        fifo_count <= fifo_count + 1'b1;
                    end
                    if ((fifo_count >= BATCH_MIN) || (timer >= TIMEOUT_CYCLES)) begin
                        fifo_rd_ptr <= '0;
                        output_pos  <= '0;
                        batch_size  <= fifo_count;
                        batch_active <= 1'b1;
                        state <= S_DISPATCH;
                    end
                end

                S_DISPATCH: begin
                    if (out_ready) begin
                        data_out    <= fifo[fifo_rd_ptr];
                        valid_out   <= 1'b1;
                        batch_first <= (output_pos == '0);
                        batch_last  <= (output_pos == (fifo_count - 1));

                        if (fifo_rd_ptr == (fifo_count - 1)) begin
                            batch_active <= 1'b0;
                            state <= S_IDLE;
                        end else begin
                            fifo_rd_ptr <= fifo_rd_ptr + 1'b1;
                            output_pos  <= output_pos + 1'b1;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
