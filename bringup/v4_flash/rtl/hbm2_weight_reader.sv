// =============================================================================
// hbm2_weight_reader.sv — AXI4 Read Master for HBM2 Weight Streaming
//
// Reads expert weights from HBM2 and streams them to the systolic array.
// Intel HBM2 Controller IP exposes AXI4 per pseudo-channel (128-bit each).
// We aggregate 2 pseudo-channels for 256-bit AXI4 interface.
//
// Address map per expert (66 MB):
//   Gate: offset 0x000_0000 (22 MB), Up: 0x160_0000 (22 MB), Down: 0x2C0_0000 (22 MB)
//
// Performance: 256 GB/s HBM2 → 66M weights in ~258 µs per matrix (saturates array)
// =============================================================================

module hbm2_weight_reader #(
    parameter int AXI_DATA_W   = 256,
    parameter int AXI_ADDR_W   = 32,
    parameter int DATA_W       = 8,
    parameter int DSP_LANES    = 128,
    parameter int MAX_BURST    = 256       // AXI4 max burst length
) (
    input  logic                        clk,              // 500 MHz
    input  logic                        rst_n,

    // AXI4 Read Master
    output logic [AXI_ADDR_W-1:0]       m_axi_araddr,
    output logic [7:0]                  m_axi_arlen,
    output logic [2:0]                  m_axi_arsize,     // 3'b101 = 32 bytes (256-bit)
    output logic                        m_axi_arvalid,
    input  logic                        m_axi_arready,

    input  logic [AXI_DATA_W-1:0]       m_axi_rdata,
    input  logic [1:0]                  m_axi_rresp,
    input  logic                        m_axi_rvalid,
    output logic                        m_axi_rready,
    input  logic                        m_axi_rlast,

    // Weight stream output → systolic array
    output logic                        weight_valid,
    output logic [DSP_LANES*DATA_W-1:0] weight_data,      // flat packed 128 × FP8
    input  logic                        weight_ready,

    // Control
    input  logic                        start,
    input  logic [AXI_ADDR_W-1:0]       base_addr,
    input  logic [15:0]                 total_words,      // FP8 elements to read
    output logic                        busy,
    output logic                        done
);

    // =========================================================================
    // Constants
    // =========================================================================
    localparam int WORDS_PER_BEAT  = AXI_DATA_W / DATA_W;     // 32 FP8 per 256-bit beat
    localparam int BEATS_PER_BURST = MAX_BURST;                // 256 beats
    localparam int WORDS_PER_BURST = WORDS_PER_BEAT * BEATS_PER_BURST;  // 8,192 FP8
    localparam int BURST_SIZE_BYTES = BEATS_PER_BURST * (AXI_DATA_W / 8);  // 8 KB

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE, S_REQ, S_DATA, S_DRAIN
    } state_t;
    state_t state;

    // =========================================================================
    // Counters
    // =========================================================================
    logic [15:0] words_remaining;        // total words left to read
    logic [7:0]  burst_beat_cnt;         // beats within current burst (0..255)
    logic [AXI_ADDR_W-1:0] next_addr;    // next AXI4 read address
    logic [7:0]  next_burst_len;         // burst length for next request

    // =========================================================================
    // Ping-pong buffers: each holds WORDS_PER_BURST FP8 weights
    // =========================================================================
    // Buffer 0 and Buffer 1: double-buffered in M20K
    // Each buffer: 8,192 × 8 bit = 64 Kbit → 4 M20K blocks each
    localparam int BUF_DEPTH = WORDS_PER_BURST;
    (* ramstyle = "M20K" *) logic [DATA_W-1:0] buf0 [BUF_DEPTH-1:0];
    (* ramstyle = "M20K" *) logic [DATA_W-1:0] buf1 [BUF_DEPTH-1:0];

    logic                    buf_sel;           // 0 = filling buf0, streaming buf1
    logic [12:0]             buf_wr_addr;       // write address into fill buffer
    logic [12:0]             buf_rd_addr;       // read address from stream buffer
    logic                    buf_fill_done;     // fill buffer is complete
    logic                    buf_stream_done;   // stream buffer is consumed

    // =========================================================================
    // Deserializer: 256-bit AXI beat → 32 × FP8
    // =========================================================================
    logic [WORDS_PER_BEAT-1:0][DATA_W-1:0] deser_words;

    genvar gi;
    generate
        for (gi = 0; gi < WORDS_PER_BEAT; gi = gi + 1) begin : gen_deser
            assign deser_words[gi] = m_axi_rdata[gi*DATA_W +: DATA_W];
        end
    endgenerate

    // =========================================================================
    // Serializer: buffer → 128-lane stream
    // =========================================================================
    logic [DSP_LANES-1:0][DATA_W-1:0] stream_lanes;
    logic [DATA_W-1:0]                stream_buf_rd_data;

    generate
        for (gi = 0; gi < DSP_LANES; gi = gi + 1) begin : gen_stream
            assign weight_data[gi*DATA_W +: DATA_W] = stream_lanes[gi];
        end
    endgenerate

    // =========================================================================
    // Main FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            busy            <= 1'b0;
            done            <= 1'b0;
            m_axi_arvalid   <= 1'b0;
            m_axi_rready    <= 1'b0;
            weight_valid    <= 1'b0;
            buf_sel         <= 1'b0;
            buf_wr_addr     <= '0;
            buf_rd_addr     <= '0;
            buf_fill_done   <= 1'b0;
            buf_stream_done <= 1'b0;
            words_remaining <= '0;
            burst_beat_cnt  <= '0;
            next_addr       <= '0;
        end else begin
            done         <= 1'b0;
            m_axi_arvalid <= 1'b0;  // pulsed

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy            <= 1'b1;
                        words_remaining <= total_words;
                        next_addr       <= base_addr;
                        buf_sel         <= 1'b0;
                        buf_wr_addr     <= '0;
                        buf_rd_addr     <= '0;
                        buf_fill_done   <= 1'b0;
                        buf_stream_done <= 1'b1;  // nothing in stream buf yet
                        state           <= S_REQ;
                    end
                end

                // Issue AXI4 read request
                S_REQ: begin
                    if (!m_axi_arvalid) begin
                        next_burst_len = (words_remaining >= WORDS_PER_BURST) ?
                                         8'(MAX_BURST - 1) :  // 256 beats
                                         8'(words_remaining / WORDS_PER_BEAT - 1);

                        m_axi_araddr   <= next_addr;
                        m_axi_arlen    <= next_burst_len;
                        m_axi_arsize   <= 3'b101;     // 32 bytes
                        m_axi_arvalid  <= 1'b1;
                        burst_beat_cnt <= '0;
                        buf_wr_addr    <= '0;
                        buf_fill_done  <= 1'b0;
                        state          <= S_DATA;
                    end
                end

                // Receive AXI4 read data, write to fill buffer
                S_DATA: begin
                    m_axi_rready <= 1'b1;

                    if (m_axi_rvalid) begin
                        // Write 32 FP8 words into fill buffer
                        if (buf_sel == 1'b0) begin
                            for (int w = 0; w < WORDS_PER_BEAT; w++)
                                buf0[buf_wr_addr + w] <= deser_words[w];
                        end else begin
                            for (int w = 0; w < WORDS_PER_BEAT; w++)
                                buf1[buf_wr_addr + w] <= deser_words[w];
                        end

                        buf_wr_addr    <= buf_wr_addr + WORDS_PER_BEAT;
                        burst_beat_cnt <= burst_beat_cnt + 1;

                        if (m_axi_rlast) begin
                            m_axi_rready   <= 1'b0;
                            buf_fill_done  <= 1'b1;
                            words_remaining <= words_remaining - (burst_beat_cnt + 1) * WORDS_PER_BEAT;
                            next_addr      <= next_addr + (burst_beat_cnt + 1) * (AXI_DATA_W / 8);
                            state          <= S_DRAIN;
                        end
                    end
                end

                // Stream from completed buffer to systolic array
                S_DRAIN: begin
                    // Read from stream buffer (opposite of fill buffer)
                    if (buf_sel == 1'b0)
                        stream_buf_rd_data <= buf1[buf_rd_addr];
                    else
                        stream_buf_rd_data <= buf0[buf_rd_addr];

                    // Pack 128 FP8 into weight_data (done combinationally via gen_stream)
                    if (weight_valid && weight_ready) begin
                        buf_rd_addr <= buf_rd_addr + DSP_LANES;
                        if (buf_rd_addr + DSP_LANES >= BUF_DEPTH) begin
                            buf_stream_done <= 1'b1;
                            weight_valid    <= 1'b0;
                        end
                    end else if (!weight_valid) begin
                        weight_valid <= 1'b1;
                    end

                    // When stream buf is empty and fill buf is ready, swap
                    if (buf_stream_done && buf_fill_done) begin
                        buf_sel         <= ~buf_sel;
                        buf_rd_addr     <= '0;
                        buf_stream_done <= 1'b0;
                        buf_fill_done   <= 1'b0;

                        if (words_remaining > 0)
                            state <= S_REQ;  // more data to read
                        else begin
                            // All data read and streamed — drain remaining stream buf
                            if (buf_rd_addr + DSP_LANES >= BUF_DEPTH) begin
                                state <= S_IDLE;
                                done  <= 1'b1;
                                busy  <= 1'b0;
                            end
                        end
                    end
                end

            endcase
        end
    end

    // Combinational lane packing (simplified — actual implementation broadcasts stream_buf_rd_data)
    // In production: stream_buf_rd_data is a window into the stream buffer;
    // weight_data broadcasts to 128 systolic array lanes
    generate
        for (gi = 0; gi < DSP_LANES; gi = gi + 1) begin : gen_weight_bcast
            assign stream_lanes[gi] = 8'd0;  // placeholder: connect to stream_buf_rd_data
        end
    endgenerate

endmodule
