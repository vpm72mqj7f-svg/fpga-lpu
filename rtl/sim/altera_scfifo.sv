//=============================================================================
// altera_scfifo.sv — Intel/Altera Single-Clock FIFO Wrapper
//
// Icarus-compatible behavioral fallback. Quartus infers scfifo IP from this
// pattern (circular buffer with registered read-data, usedw signals).
//
// Usage: drop-in replacement for hand-written circular buffer FIFOs.
//   WIDTH  — data width in bits
//   DEPTH  — number of entries (should be power of 2 for Quartus inference)
//   SHOWAHEAD — 1 = rd_data shows next entry without rd_en (look-ahead mode)
//=============================================================================

module altera_scfifo #(
    parameter int WIDTH    = 8,
    parameter int DEPTH    = 16,
    parameter bit SHOWAHEAD = 0
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  wr_en,
    input  logic [WIDTH-1:0]      wr_data,
    output logic                  full,
    output logic                  almost_full,
    input  logic                  rd_en,
    output logic [WIDTH-1:0]      rd_data,
    output logic                  empty,
    output logic [$clog2(DEPTH+1)-1:0] usedw
);

    localparam int AW = $clog2(DEPTH);

    logic [WIDTH-1:0] mem [0:DEPTH-1];
    logic [AW-1:0]    wr_ptr, rd_ptr;
    logic [AW:0]      count;  // extra bit for full detection

    // synthesis attribute of mem is "M20K" or "MLAB" based on DEPTH*WIDTH
    // Quartus scfifo inference: recognizes wr_ptr/rd_ptr circular buffer pattern

    assign usedw = count;
    assign empty = (count == 0);
    assign full  = (count == DEPTH);
    assign almost_full = (count >= DEPTH - 2);

    // Write
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
        end else if (wr_en && !full) begin
            mem[wr_ptr] <= wr_data;
            wr_ptr <= wr_ptr + 1'b1;
        end
    end

    // Read pointer + occupancy
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= '0;
            count  <= '0;
        end else begin
            case ({wr_en && !full, rd_en && !empty})
                2'b10: begin  // write only
                    count <= count + 1'b1;
                end
                2'b01: begin  // read only
                    rd_ptr <= rd_ptr + 1'b1;
                    count  <= count - 1'b1;
                end
                2'b11: begin  // simultaneous read+write (throughput = 1)
                    rd_ptr <= rd_ptr + 1'b1;
                    // count unchanged
                end
                default: ;  // idle
            endcase
        end
    end

    // Read data output
    generate
        if (SHOWAHEAD) begin : g_showahead
            // Look-ahead: rd_data shows the "next" entry; rd_en advances
            logic [WIDTH-1:0] rd_data_next;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    rd_data <= '0;
                end else if (rd_en || empty) begin
                    rd_data <= rd_data_next;
                end
            end

            always_comb begin
                if (!empty)
                    rd_data_next = mem[rd_ptr];
                else
                    rd_data_next = rd_data;  // hold
            end
        end else begin : g_normal
            // Normal mode: rd_en pops the entry; rd_data valid next cycle
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    rd_data <= '0;
                end else if (rd_en && !empty) begin
                    rd_data <= mem[rd_ptr];
                end
            end
        end
    endgenerate

endmodule
