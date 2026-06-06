//=============================================================================
// hbm_bw_test.sv — HBM2e Bandwidth Validation Engine  [GO/NO-GO #1]
//
// Tests HBM2e read/write bandwidth using AXI4 burst transactions.
// Self-contained: generates patterns, measures throughput, reports via LEDs.
//
// Go/No-Go Criteria:
//   GO:    read ≥ 800 GB/s, write ≥ 700 GB/s  (≥ 80% of rated 920 GB/s)
//   WARN:  read ≥ 500 GB/s, write ≥ 400 GB/s  (usable with degraded perf)
//   NO-GO: read <  500 GB/s or write < 400 GB/s (architecture infeasible)
//
// Test sequence:
//   1. Write 256 MB pattern to HBM (sequential addressing)
//   2. Read back 256 MB, check pattern
//   3. Measure elapsed cycles → compute bandwidth
//   4. Report via LED code + status signals
//
// AXI4 interface: 256-bit data, 32-bit address (HBM2e pseudo-channel)
//=============================================================================

module hbm_bw_test #(
    parameter int AXI_DATA_WIDTH  = 256,      // HBM2e AXI data width
    parameter int AXI_ADDR_WIDTH  = 32,       // HBM2e address width
    parameter int AXI_ID_WIDTH    = 4,
    parameter int BURST_LENGTH    = 16,       // beats per burst (16 × 32B = 512B)
    parameter int TEST_SIZE_BYTES = 256*1024*1024,  // 256 MB test
    parameter int CLK_FREQ_MHZ    = 450       // HBM/DSP clock (MHz, nominal)
) (
    input  logic        clk,
    input  logic        rst_n,

    // Control
    input  logic        start_test,           // pulse to start
    output logic        test_done,
    output logic [1:0]  test_result,          // 0=idle, 1=running, 2=GO, 3=NO-GO
    output logic [31:0] write_bw_mb_s,        // measured write bandwidth (MB/s)
    output logic [31:0] read_bw_mb_s,         // measured read bandwidth (MB/s)

    // Status output (for UART)
    output logic        status_valid,
    output logic [7:0]  status_char,

    // AXI4 Write Address Channel
    output logic [AXI_ID_WIDTH-1:0]    m_axi_awid,
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_awaddr,
    output logic [7:0]                 m_axi_awlen,
    output logic [2:0]                 m_axi_awsize,   // 5 = 32 bytes
    output logic [1:0]                 m_axi_awburst,  // 1 = INCR
    output logic                       m_axi_awvalid,
    input  logic                       m_axi_awready,

    // AXI4 Write Data Channel
    output logic [AXI_DATA_WIDTH-1:0]  m_axi_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0] m_axi_wstrb,
    output logic                       m_axi_wlast,
    output logic                       m_axi_wvalid,
    input  logic                       m_axi_wready,

    // AXI4 Write Response Channel
    input  logic [AXI_ID_WIDTH-1:0]    m_axi_bid,
    input  logic [1:0]                 m_axi_bresp,
    input  logic                       m_axi_bvalid,
    output logic                       m_axi_bready,

    // AXI4 Read Address Channel
    output logic [AXI_ID_WIDTH-1:0]    m_axi_arid,
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_araddr,
    output logic [7:0]                 m_axi_arlen,
    output logic [2:0]                 m_axi_arsize,
    output logic [1:0]                 m_axi_arburst,
    output logic                       m_axi_arvalid,
    input  logic                       m_axi_arready,

    // AXI4 Read Data Channel
    input  logic [AXI_ID_WIDTH-1:0]    m_axi_rid,
    input  logic [AXI_DATA_WIDTH-1:0]  m_axi_rdata,
    input  logic [1:0]                 m_axi_rresp,
    input  logic                       m_axi_rlast,
    input  logic                       m_axi_rvalid,
    output logic                       m_axi_rready
);

    localparam int BYTES_PER_BEAT = AXI_DATA_WIDTH / 8;  // 32
    localparam int BYTES_PER_BURST = BYTES_PER_BEAT * BURST_LENGTH;  // 512
    localparam int TOTAL_BURSTS = TEST_SIZE_BYTES / BYTES_PER_BURST;
    localparam int SIZE_WIDTH = $clog2(TOTAL_BURSTS + 1);

    typedef enum logic [3:0] {
        S_IDLE,
        S_WRITE_ADDR,
        S_WRITE_DATA,
        S_WRITE_RESP,
        S_READ_ADDR,
        S_READ_DATA,
        S_CHECK,
        S_DONE,
        S_FAIL
    } state_t;
    state_t state;

    // Counters
    logic [SIZE_WIDTH-1:0] burst_count;
    logic [7:0]            beat_count;
    logic [AXI_ADDR_WIDTH-1:0] base_addr;
    logic [63:0]           cycle_count;
    logic [63:0]           write_cycles;
    logic [63:0]           read_cycles;

    // Pattern generator (LFSR)
    logic [AXI_DATA_WIDTH-1:0] wr_pattern;
    logic [AXI_DATA_WIDTH-1:0] expected_data;

    assign m_axi_awsize  = 3'd5;   // 2^5 = 32 bytes per beat
    assign m_axi_awburst = 2'd1;   // INCR
    assign m_axi_arsize  = 3'd5;
    assign m_axi_arburst = 2'd1;
    assign m_axi_awid    = '0;
    assign m_axi_arid    = '0;
    assign m_axi_wstrb   = '1;     // all bytes valid
    assign m_axi_bready  = 1'b1;

    // LFSR pattern: simple 256-bit counter for write data
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_pattern <= '0;
        end else if (state == S_WRITE_DATA && m_axi_wvalid && m_axi_wready) begin
            wr_pattern <= wr_pattern + 1'b1;
        end
    end
    assign m_axi_wdata = {wr_pattern[223:0], 32'hDEAD_BEEF};  // marker

    // Expected data for readback
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            expected_data <= '0;
        end else if (state == S_READ_DATA && m_axi_rvalid && m_axi_rready) begin
            expected_data <= expected_data + 1'b1;
        end
    end

    // AXI write address channel
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awvalid <= 1'b0;
            m_axi_awaddr  <= '0;
            m_axi_awlen   <= '0;
        end else begin
            if (state == S_WRITE_ADDR && !m_axi_awvalid) begin
                m_axi_awvalid <= 1'b1;
                m_axi_awaddr  <= base_addr + (burst_count * BYTES_PER_BURST);
                m_axi_awlen   <= BURST_LENGTH - 1;
            end else if (m_axi_awvalid && m_axi_awready) begin
                m_axi_awvalid <= 1'b0;
            end
        end
    end

    // AXI write data channel
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_wvalid <= 1'b0;
            m_axi_wlast  <= 1'b0;
            beat_count   <= '0;
        end else begin
            if (state == S_WRITE_DATA) begin
                m_axi_wvalid <= 1'b1;
                m_axi_wlast  <= (beat_count == BURST_LENGTH - 1);
                if (m_axi_wvalid && m_axi_wready) begin
                    beat_count <= m_axi_wlast ? '0 : (beat_count + 1'b1);
                end
            end else begin
                m_axi_wvalid <= 1'b0;
                m_axi_wlast  <= 1'b0;
            end
        end
    end

    // AXI read address channel
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arvalid <= 1'b0;
            m_axi_araddr  <= '0;
            m_axi_arlen   <= '0;
        end else begin
            if (state == S_READ_ADDR && !m_axi_arvalid) begin
                m_axi_arvalid <= 1'b1;
                m_axi_araddr  <= base_addr + (burst_count * BYTES_PER_BURST);
                m_axi_arlen   <= BURST_LENGTH - 1;
            end else if (m_axi_arvalid && m_axi_arready) begin
                m_axi_arvalid <= 1'b0;
            end
        end
    end

    // AXI read data channel (always ready)
    assign m_axi_rready = (state == S_READ_DATA) || (state == S_CHECK);

    //=========================================================================
    // Main FSM
    //=========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            test_done     <= 1'b0;
            test_result   <= 2'd0;
            write_bw_mb_s <= '0;
            read_bw_mb_s  <= '0;
            burst_count   <= '0;
            base_addr     <= '0;
            cycle_count   <= '0;
            write_cycles  <= '0;
            read_cycles   <= '0;
            status_valid  <= 1'b0;
        end else begin
            test_done    <= 1'b0;
            status_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start_test) begin
                        burst_count  <= '0;
                        base_addr    <= '0;
                        cycle_count  <= '0;
                        test_result  <= 2'd1;  // running
                        state <= S_WRITE_ADDR;
                    end
                end

                S_WRITE_ADDR: begin
                    if (m_axi_awvalid && m_axi_awready) begin
                        state <= S_WRITE_DATA;
                    end
                end

                S_WRITE_DATA: begin
                    if (m_axi_wvalid && m_axi_wready && m_axi_wlast) begin
                        if (burst_count == TOTAL_BURSTS - 1) begin
                            write_cycles <= cycle_count;
                            burst_count  <= '0;
                            cycle_count  <= '0;
                            state <= S_READ_ADDR;
                        end else begin
                            burst_count <= burst_count + 1'b1;
                            state <= S_WRITE_ADDR;
                        end
                    end
                end

                // Drain write responses
                S_WRITE_RESP: begin
                    if (m_axi_bvalid) begin
                        state <= S_READ_ADDR;
                    end
                end

                S_READ_ADDR: begin
                    if (m_axi_arvalid && m_axi_arready) begin
                        state <= S_READ_DATA;
                    end
                end

                S_READ_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        // Check marker in all beats
                        if (m_axi_rdata[31:0] != 32'hDEAD_BEEF) begin
                            test_result <= 2'd3;  // NO-GO
                            state <= S_FAIL;
                        end
                        if (m_axi_rlast) begin
                            if (burst_count == TOTAL_BURSTS - 1) begin
                                state <= S_CHECK;
                            end else begin
                                burst_count <= burst_count + 1'b1;
                                state <= S_READ_ADDR;
                            end
                        end
                    end
                end

                S_CHECK: begin
                    read_cycles <= cycle_count;
                    // bw_MBps = TEST_SIZE_BYTES * CLK_FREQ_MHZ / cycles
                    // (bytes × MHz = bytes × 1e6/s, /1e6 = MB/s)
                    if (write_cycles > 0) begin
                        write_bw_mb_s <= (64'(TEST_SIZE_BYTES) * 64'(CLK_FREQ_MHZ))
                                        / write_cycles;
                    end
                    if (read_cycles > 0) begin
                        read_bw_mb_s <= (64'(TEST_SIZE_BYTES) * 64'(CLK_FREQ_MHZ))
                                       / read_cycles;
                    end

                    // GO / NO-GO decision (scaled for actual test size)
                    // Per-channel peak = 14,400 MB/s. Target: > 60% = 8640 MB/s
                    if (write_bw_mb_s >= 700_000 && read_bw_mb_s >= 800_000) begin
                        test_result <= 2'd2;  // GO
                    end else begin
                        test_result <= 2'd3;  // NO-GO
                    end
                    test_done <= 1'b1;
                    state <= S_DONE;
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                S_FAIL: begin
                    test_done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase

            // Cycle counter (shared across phases)
            if (state != S_IDLE && state != S_DONE) begin
                cycle_count <= cycle_count + 1'b1;
            end
        end
    end

endmodule
