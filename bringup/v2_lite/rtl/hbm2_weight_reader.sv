// =============================================================================
// hbm2_weight_reader.sv — AXI4 Read Master for HBM2 Weight Streaming (V2-Lite)
//
// Reads expert weights from HBM2 and streams them to the systolic array.
// Intel HBM2 Controller IP exposes AXI4 per pseudo-channel (256-bit each).
//
// V2-Lite weight sizes (per expert):
//   Gate: 2048×1408 = 2,883,584 FP8 ≈ 2.88 MB
//   Up:   same ≈ 2.88 MB
//   Down: 1408×2048 = 2,883,584 FP8 ≈ 2.88 MB
//   Total per expert: ~8.65 MB (vs 66 MB for V4-Flash)
//   66 experts × 8.65 MB = 571 MB total (fits easily in 8 GB HBM2)
//
// Buffer architecture:
//   Interleaved 64-bank M20K design. Each AXI beat (256-bit = 32 FP8) is
//   distributed across banks. Even beats target banks 0..31; odd beats
//   target banks 32..63. Each bank is 128 entries deep.
//   Streaming reads all 64 banks in parallel → 64 FP8/cycle output.
//
// Performance: with 256-bit AXI at 500 MHz, fill bandwidth is 16 GB/s.
//   Stream bandwidth is 32 GB/s (64 FP8/cycle). With double-buffered
//   fill-ahead, effective throughput ~24 GB/s (75% of peak).
//   For full bandwidth, aggregate 2 pseudo-channels (512-bit AXI).
// =============================================================================

module hbm2_weight_reader #(
    parameter int AXI_DATA_W   = 256,
    parameter int AXI_ADDR_W   = 32,
    parameter int DATA_W       = 8,
    parameter int DSP_LANES    = 64,
    parameter int MAX_BURST    = 256,      // AXI4 max burst length
    parameter logic [31:0] VERSION = 32'h0B061B01  // {day,month,year-2000,build#}
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
    output logic [DSP_LANES*DATA_W-1:0] weight_data,      // flat packed 64 × FP8
    input  logic                        weight_ready,

    // Control
    input  logic                        start,
    input  logic [AXI_ADDR_W-1:0]       base_addr,
    input  logic [15:0]                 total_words,      // FP8 elements to read
    output logic                        busy,
    output logic                        done,

    // ---- Debug ----
    output logic [2:0]                  dbg_fsm_state,
    output logic                        dbg_buf_sel,
    output logic [6:0]                  dbg_rd_addr,
    output logic [6:0]                  dbg_wr_addr,
    output logic                        dbg_streaming,
    output logic                        dbg_filling,
    output logic [31:0]                 perf_bytes_read,
    output logic [31:0]                 perf_bursts_done,
    output logic [31:0]                 perf_beats_read
);

    // =========================================================================
    // Constants
    // =========================================================================
    localparam int WORDS_PER_BEAT  = AXI_DATA_W / DATA_W;     // 32 FP8 per 256-bit beat
    localparam int BEATS_PER_BURST = MAX_BURST;                // 256 beats
    localparam int WORDS_PER_BURST = WORDS_PER_BEAT * BEATS_PER_BURST;  // 8,192 FP8
    localparam int BURST_SIZE_BYTES = BEATS_PER_BURST * (AXI_DATA_W / 8);  // 8 KB
    localparam int BANK_DEPTH      = WORDS_PER_BURST / DSP_LANES;  // 128 entries per bank

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE, S_AR, S_RDATA, S_DRAIN
    } state_t;
    state_t state;

    // =========================================================================
    // Counters
    // =========================================================================
    logic [15:0] words_remaining;        // total words left to read
    logic [7:0]  burst_beat_cnt;         // beats within current burst (0..MAX_BURST-1)
    logic [AXI_ADDR_W-1:0] next_addr;    // next AXI4 read address
    logic [7:0]  next_burst_len;         // burst length for next request

    // =========================================================================
    // Interleaved Bank Buffers (64 banks × 128 entries, ping-pong)
    //
    // Write: 32 FP8 per AXI beat distributed across appropriate 32 of 64 banks.
    //   Even beats → banks 0..31, odd beats → banks 32..63.
    //   Address within bank = burst_beat_cnt / 2
    //
    // Read: all 64 banks read in parallel at the same address.
    //   One cycle produces 64 FP8 values for the systolic array.
    // =========================================================================
    logic [$clog2(BANK_DEPTH)-1:0]       bank_wr_addr;      // address to write within banks
    logic [$clog2(BANK_DEPTH)-1:0]       bank_rd_addr;      // address to read within banks

    // Read data from each bank (combinational)
    logic [DSP_LANES-1:0][DATA_W-1:0]    bank_rd_data;

    // Fill / stream control
    logic                    buf_sel;           // 0 = filling buf0, streaming buf1
    logic                    buf_fill_done;     // fill buffer just completed a burst
    logic                    buf_stream_has_data; // stream buffer has unread data
    logic                    buf_stream_done;   // stream buffer fully consumed

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
    // Interleaved M20K Banks — Generate 64 parallel read ports
    //
    // Each bank: 128 × 8-bit simple dual-port RAM (inferred as M20K)
    // Write port: shared — writes to the fill buffer when this bank is targeted
    // Read port:  independent — reads from the stream buffer in parallel
    // =========================================================================
    genvar bank;
    generate
        for (bank = 0; bank < DSP_LANES; bank++) begin : g_bank
            (* ramstyle = "M20K" *) logic [DATA_W-1:0] mem0 [BANK_DEPTH-1:0];
            (* ramstyle = "M20K" *) logic [DATA_W-1:0] mem1 [BANK_DEPTH-1:0];

            // Write enable: this bank is targeted during the appropriate half of the burst
            logic bank_we;
            assign bank_we = (state == S_RDATA) && m_axi_rvalid &&
                ((!burst_beat_cnt[0] && bank < WORDS_PER_BEAT) ||
                 ( burst_beat_cnt[0] && bank >= WORDS_PER_BEAT));

            // Write to fill buffer
            always_ff @(posedge clk) begin
                if (bank_we) begin
                    if (buf_sel == 1'b0)
                        mem0[bank_wr_addr] <= deser_words[bank % WORDS_PER_BEAT];
                    else
                        mem1[bank_wr_addr] <= deser_words[bank % WORDS_PER_BEAT];
                end
            end

            // Read from stream buffer (combinational — feeds systolic array)
            // buf_sel=0 means filling buf0, streaming buf1
            assign bank_rd_data[bank] = (buf_sel == 1'b0) ? mem1[bank_rd_addr]
                                                           : mem0[bank_rd_addr];
        end
    endgenerate

    // =========================================================================
    // Pack bank read data → flat weight_data for systolic array
    // =========================================================================
    generate
        for (gi = 0; gi < DSP_LANES; gi = gi + 1) begin : gen_stream_pack
            assign weight_data[gi*DATA_W +: DATA_W] = bank_rd_data[gi];
        end
    endgenerate

    // =========================================================================
    // Stream buffer status
    // =========================================================================
    // After swap, stream buffer has BANK_DEPTH unread rows
    // buf_stream_done when all rows have been streamed
    always_comb begin
        // Stream buffer has data if rd_addr hasn't reached the end
        buf_stream_has_data = (bank_rd_addr < BANK_DEPTH);
        buf_stream_done     = !buf_stream_has_data;
    end

    // =========================================================================
    // Main FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            busy               <= 1'b0;
            done               <= 1'b0;
            m_axi_arvalid      <= 1'b0;
            m_axi_rready       <= 1'b0;
            weight_valid       <= 1'b0;
            buf_sel            <= 1'b0;
            bank_wr_addr       <= '0;
            bank_rd_addr       <= '0;
            buf_fill_done      <= 1'b0;
            words_remaining    <= '0;
            burst_beat_cnt     <= '0;
            next_addr          <= '0;
        end else begin
            done          <= 1'b0;
            m_axi_arvalid <= 1'b0;  // pulsed
            m_axi_rready  <= 1'b0;  // default deasserted

            // ================================================================
            // Continuous streaming (not state-gated)
            // Stream from the current stream buffer whenever the consumer
            // is ready and we have data. This runs in parallel with fill.
            // ================================================================
            if (buf_stream_has_data) begin
                if (weight_valid && weight_ready) begin
                    // Advance read pointer
                    bank_rd_addr <= bank_rd_addr + 1'b1;
                    if (bank_rd_addr + 1'b1 >= BANK_DEPTH) begin
                        // Last read cycle — next time there's no data
                        weight_valid <= 1'b0;
                    end
                end else if (!weight_valid) begin
                    // Start streaming
                    weight_valid <= 1'b1;
                end
            end else begin
                weight_valid <= 1'b0;
            end

            case (state)

                S_IDLE: begin
                    if (start) begin
                        busy            <= 1'b1;
                        words_remaining <= total_words;
                        next_addr       <= base_addr;
                        buf_sel         <= 1'b0;
                        bank_wr_addr    <= '0;
                        bank_rd_addr    <= '0;
                        buf_fill_done   <= 1'b0;
                        weight_valid    <= 1'b0;
                        state           <= S_AR;
                    end
                end

                // Issue AXI4 read request
                S_AR: begin
                    if (!m_axi_arvalid) begin
                        // Calculate burst length in AXI beats
                        if (words_remaining >= WORDS_PER_BURST) begin
                            next_burst_len = 8'(MAX_BURST - 1);    // 255 (256 beats, 0-indexed)
                        end else begin
                            next_burst_len = 8'((words_remaining / WORDS_PER_BEAT) - 1);
                        end

                        m_axi_araddr   <= next_addr;
                        m_axi_arlen    <= next_burst_len;
                        m_axi_arsize   <= 3'b101;     // 32 bytes per beat (256-bit)
                        m_axi_arvalid  <= 1'b1;
                        burst_beat_cnt <= '0;
                        bank_wr_addr   <= '0;
                        buf_fill_done  <= 1'b0;
                        state          <= S_RDATA;
                    end
                end

                // Receive AXI4 read data, write to fill buffer
                S_RDATA: begin
                    m_axi_rready <= 1'b1;

                    if (m_axi_rvalid) begin
                        burst_beat_cnt <= burst_beat_cnt + 1'b1;

                        // bank_wr_addr advances every 2 beats (after both halves written)
                        if (burst_beat_cnt[0]) begin
                            // Odd beat just completed → all banks at this addr filled
                            bank_wr_addr <= bank_wr_addr + 1'b1;
                        end

                        if (m_axi_rlast) begin
                            m_axi_rready   <= 1'b0;
                            buf_fill_done  <= 1'b1;

                            // Adjust words_remaining
                            if (words_remaining >= WORDS_PER_BURST) begin
                                words_remaining <= words_remaining - WORDS_PER_BURST;
                            end else begin
                                words_remaining <= '0;
                            end
                            next_addr <= next_addr + (burst_beat_cnt + 1'b1) * (AXI_DATA_W / 8);
                            state     <= S_DRAIN;
                        end
                    end
                end

                // S_DRAIN: stream buffer has drained + fill buffer is ready → swap
                // The actual streaming happens in the always-active section above.
                S_DRAIN: begin
                    if (buf_stream_done && buf_fill_done) begin
                        // Swap buffers: the freshly filled buffer becomes the stream source
                        buf_sel         <= ~buf_sel;
                        bank_rd_addr    <= '0;
                        buf_fill_done   <= 1'b0;

                        if (words_remaining > 0) begin
                            state <= S_AR;  // more data to fetch
                        end else begin
                            // All data requested. If stream buffer is also done, finish.
                            state <= S_IDLE;
                            done  <= 1'b1;
                            busy  <= 1'b0;
                        end
                    end else if (buf_stream_done && !buf_fill_done) begin
                        // Stream buffer is empty but fill not yet complete.
                        // This can happen if stream drains faster than fill.
                        // Wait in S_DRAIN — fill continues via concurrent logic?
                        // Actually, S_RDATA completes before S_DRAIN, so fill IS done.
                        // The only case: we're draining the very last data and
                        // there's no more to fill. Wait for consumer.
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Debug: Status assignments
    // =========================================================================
    assign dbg_fsm_state = state;
    assign dbg_buf_sel   = buf_sel;
    assign dbg_rd_addr   = bank_rd_addr;
    assign dbg_wr_addr   = bank_wr_addr;
    assign dbg_streaming = weight_valid;
    assign dbg_filling   = (state == S_RDATA);

    // =========================================================================
    // Debug: Performance counters
    // =========================================================================
    logic [31:0] _perf_bytes, _perf_bursts, _perf_beats;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            _perf_bytes  <= 32'd0;
            _perf_bursts <= 32'd0;
            _perf_beats  <= 32'd0;
        end else begin
            if (m_axi_rvalid && m_axi_rready) begin
                _perf_bytes <= _perf_bytes + 32'd32;  // 256-bit = 32 bytes
                _perf_beats <= _perf_beats + 32'd1;
                if (m_axi_rlast)
                    _perf_bursts <= _perf_bursts + 32'd1;
            end
        end
    end
    assign perf_bytes_read  = _perf_bytes;
    assign perf_bursts_done = _perf_bursts;
    assign perf_beats_read  = _perf_beats;

endmodule
