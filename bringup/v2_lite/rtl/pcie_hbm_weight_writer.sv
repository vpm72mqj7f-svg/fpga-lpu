// =============================================================================
// pcie_hbm_weight_writer.sv — PCIe BAR0 → HBM2 AXI4 Weight Download Engine
//
// Host writes expert weights through PCIe BAR0 registers:
//   0x00  HBM_ADDR_LO   [31:0]  W  HBM2 target address low
//   0x04  HBM_ADDR_HI   [31:0]  W  HBM2 target address high (reserved)
//   0x08  BURST_COUNT   [31:0]  W  number of 256-beat AXI bursts to issue
//   0x0C  CONTROL       [31:0]  W  [0]=START [1]=ABORT
//   0x10  STATUS        [31:0]  R  [0]=BUSY [1]=DONE [2]=ERROR
//   0x20  DATA_PORT     [63:0]  W  64-bit data write port (4 writes → 1 AXI beat)
//
// Operation:
//   1. Host writes HBM_ADDR_LO, BURST_COUNT to BAR0
//   2. Host writes CONTROL[0]=1 (START)
//   3. Host streams 256-bit words through DATA_PORT (4 × 64-bit writes per beat)
//   4. Module issues AXI4 write bursts to HBM2
//   5. Host polls STATUS until DONE=1
//
// AXI4 write channel:
//   AW: 28-bit address, 8-bit burst length, 3-bit size(5=32B), 2-bit burst(INCR)
//   W:  256-bit data, 32-bit strobe
//   B:  9-bit ID, 2-bit response
//
// Performance:
//   At 250 MHz, 256-bit × 256 beats/burst = 8 KB/burst
//   One 64-bit DATA_PORT write per 4 cycles = 64-bit/cycle ≈ 2 GB/s data rate
//   HBM2 effective BW per pseudo-channel ≈ 14 GB/s (limited by 256-bit AXI)
// =============================================================================

module pcie_hbm_weight_writer #(
    parameter int AXI_ADDR_W  = 28,    // HBM2 AXI4 address width
    parameter int AXI_DATA_W  = 256,   // AXI4 data width
    parameter int AXI_ID_W    = 9,     // AXI4 ID width
    parameter int BAR_ADDR_W  = 8      // BAR0 address width (256 bytes)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // ---- PCIe BAR0 Register Interface (from PCIe HIP AXI4-Lite slave) ----
    input  logic                         bar_wvalid,
    input  logic [BAR_ADDR_W-1:0]        bar_waddr,
    input  logic [31:0]                  bar_wdata,
    output logic                         bar_wready,
    input  logic                         bar_rvalid,
    input  logic [BAR_ADDR_W-1:0]        bar_raddr,
    output logic [31:0]                  bar_rdata,
    output logic                         bar_rready,

    // ---- HBM2 AXI4 Write Master ----
    // AW channel
    output logic [AXI_ID_W-1:0]          m_axi_awid,
    output logic [AXI_ADDR_W-1:0]        m_axi_awaddr,
    output logic [7:0]                   m_axi_awlen,
    output logic [2:0]                   m_axi_awsize,
    output logic [1:0]                   m_axi_awburst,
    output logic [2:0]                   m_axi_awprot,
    output logic [3:0]                   m_axi_awqos,
    output logic                         m_axi_awvalid,
    input  logic                         m_axi_awready,

    // W channel
    output logic [AXI_DATA_W-1:0]        m_axi_wdata,
    output logic [AXI_DATA_W/8-1:0]      m_axi_wstrb,
    output logic                         m_axi_wlast,
    output logic                         m_axi_wvalid,
    input  logic                         m_axi_wready,

    // B channel
    input  logic [AXI_ID_W-1:0]          m_axi_bid,
    input  logic [1:0]                   m_axi_bresp,
    input  logic                         m_axi_bvalid,
    output logic                         m_axi_bready
);

    // =========================================================================
    // Register Map (BAR0 offsets)
    // =========================================================================
    localparam logic [BAR_ADDR_W-1:0] ADDR_HBM_LO    = 8'h00;
    localparam logic [BAR_ADDR_W-1:0] ADDR_HBM_HI    = 8'h04;
    localparam logic [BAR_ADDR_W-1:0] ADDR_BURST_CNT = 8'h08;
    localparam logic [BAR_ADDR_W-1:0] ADDR_CONTROL   = 8'h0C;
    localparam logic [BAR_ADDR_W-1:0] ADDR_STATUS    = 8'h10;
    localparam logic [BAR_ADDR_W-1:0] ADDR_DATA_PORT = 8'h20;

    // CONTROL bits
    localparam int CTRL_START = 0;
    localparam int CTRL_ABORT = 1;

    // STATUS bits
    localparam int STAT_BUSY  = 0;
    localparam int STAT_DONE  = 1;
    localparam int STAT_ERROR = 2;

    // =========================================================================
    // Internal Registers
    // =========================================================================
    logic [31:0]  hbm_addr_lo;
    logic [31:0]  hbm_addr_hi;
    logic [31:0]  burst_count;
    logic [31:0]  control;
    logic [31:0]  status;
    logic [63:0]  data_port;

    // =========================================================================
    // Data Packer: 4 × 64-bit BAR writes → 1 × 256-bit AXI beat
    // =========================================================================
    logic [1:0]   data_port_wr_cnt;    // 0..3, rolls over per AXI beat
    logic         data_port_ready;     // high when can accept next 64-bit write
    logic [AXI_DATA_W-1:0] wdata_buffer;
    logic         wdata_buffer_valid;
    logic         wdata_buffer_last;   // last beat of current burst

    logic [31:0]  beats_remaining;     // beats left in current burst
    logic [31:0]  bursts_remaining;    // bursts left in total transfer
    logic         transfer_active;

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE,
        S_START_BURST,     // Issue AW
        S_WRITE_DATA,      // Stream W beats
        S_WAIT_BRESP,      // Wait for B channel response
        S_NEXT_BURST,      // Check if more bursts remain
        S_DONE,            // Transfer complete
        S_ERROR            // Error state
    } state_t;

    state_t state;

    // =========================================================================
    // BAR0 Register Read/Write
    // =========================================================================
    assign bar_wready = 1'b1;  // Always ready to accept BAR writes
    assign bar_rready = 1'b1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hbm_addr_lo   <= 32'd0;
            hbm_addr_hi   <= 32'd0;
            burst_count   <= 32'd0;
            control       <= 32'd0;
            data_port     <= 64'd0;
        end else begin
            // Write registers
            if (bar_wvalid && bar_wready) begin
                case (bar_waddr)
                    ADDR_HBM_LO:    hbm_addr_lo   <= bar_wdata;
                    ADDR_HBM_HI:    hbm_addr_hi   <= bar_wdata;
                    ADDR_BURST_CNT: burst_count   <= bar_wdata;
                    ADDR_CONTROL:   control       <= bar_wdata;
                    ADDR_DATA_PORT: data_port     <= {data_port[31:0], bar_wdata}; // 2×32-bit → 64-bit
                    default: ;
                endcase
            end
            // Auto-clear CONTROL after START
            if (control[CTRL_START] && state != S_IDLE)
                control[CTRL_START] <= 1'b0;
        end
    end

    // BAR read mux
    always_comb begin
        bar_rdata = 32'd0;
        case (bar_raddr)
            ADDR_HBM_LO:    bar_rdata = hbm_addr_lo;
            ADDR_HBM_HI:    bar_rdata = hbm_addr_hi;
            ADDR_BURST_CNT: bar_rdata = burst_count;
            ADDR_CONTROL:   bar_rdata = control;
            ADDR_STATUS:    bar_rdata = status;
            default:        bar_rdata = 32'd0;
        endcase
    end

    // =========================================================================
    // Data Port Packer Counter
    // =========================================================================
    // data_port_ready: high when we're in write-data state and need more data
    assign data_port_ready = (state == S_WRITE_DATA) && m_axi_wready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_port_wr_cnt    <= 2'd0;
            wdata_buffer_valid  <= 1'b0;
            wdata_buffer        <= 256'd0;
        end else begin
            // Collect 4 × 64-bit writes into 256-bit word
            if (bar_wvalid && bar_wready && (bar_waddr == ADDR_DATA_PORT)) begin
                data_port_wr_cnt <= data_port_wr_cnt + 2'd1;
                // Latch 64-bit data into the correct 256-bit slice
                case (data_port_wr_cnt)
                    2'd0: wdata_buffer[63:0]    <= bar_wdata;
                    2'd1: wdata_buffer[127:64]  <= bar_wdata;
                    2'd2: wdata_buffer[191:128] <= bar_wdata;
                    2'd3: begin
                        wdata_buffer[255:192] <= bar_wdata;
                        wdata_buffer_valid     <= 1'b1;  // Full 256-bit word ready
                    end
                endcase
            end

            // Clear valid flag when AXI W channel consumes the data
            if (wdata_buffer_valid && m_axi_wready) begin
                wdata_buffer_valid <= 1'b0;
                data_port_wr_cnt   <= 2'd0;
            end
        end
    end

    // =========================================================================
    // AXI4 Write Channel Assignments (constant)
    // =========================================================================
    assign m_axi_awid    = '0;
    assign m_axi_awsize  = 3'd5;     // 32 bytes per beat (256-bit AXI data width)
    assign m_axi_awburst = 2'b01;    // INCR (incrementing burst)
    assign m_axi_awprot  = 3'b000;
    assign m_axi_awqos   = 4'd0;
    assign m_axi_wstrb   = {AXI_DATA_W/8{1'b1}};  // All bytes valid
    assign m_axi_bready  = 1'b1;     // Always ready for write response

    // =========================================================================
    // Main FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            status           <= 32'd0;
            m_axi_awvalid    <= 1'b0;
            m_axi_awaddr     <= {AXI_ADDR_W{1'b0}};
            m_axi_awlen      <= 8'd0;
            m_axi_wvalid     <= 1'b0;
            m_axi_wdata      <= 256'd0;
            m_axi_wlast      <= 1'b0;
            beats_remaining  <= 32'd0;
            bursts_remaining <= 32'd0;
            transfer_active  <= 1'b0;
        end else begin
            // Default: deassert pulsed signals
            m_axi_awvalid <= 1'b0;

            case (state)

                S_IDLE: begin
                    status <= 32'd0;
                    if (control[CTRL_START]) begin
                        // Validate: must have address and burst count
                        if (burst_count == 32'd0) begin
                            status[STAT_ERROR] <= 1'b1;
                            state <= S_ERROR;
                        end else begin
                            status[STAT_BUSY]      <= 1'b1;
                            bursts_remaining       <= burst_count;
                            hbm_addr_lo            <= hbm_addr_lo;  // latch
                            transfer_active        <= 1'b1;
                            state                  <= S_START_BURST;
                        end
                    end
                end

                // Issue AXI4 write address
                S_START_BURST: begin
                    if (!m_axi_awvalid) begin
                        // Set burst length: 256 beats per burst (max AXI4)
                        m_axi_awlen    <= 8'd255;   // 256 beats − 1
                        m_axi_awaddr   <= hbm_addr_lo[AXI_ADDR_W-1:0];
                        m_axi_awvalid  <= 1'b1;
                        beats_remaining <= 32'd256;
                        data_port_wr_cnt <= 2'd0;
                        wdata_buffer_valid <= 1'b0;
                        state <= S_WRITE_DATA;
                    end
                end

                // Stream write data beats
                S_WRITE_DATA: begin
                    if (wdata_buffer_valid) begin
                        // Drive W channel with packed 256-bit data
                        if (!m_axi_wvalid || (m_axi_wvalid && m_axi_wready)) begin
                            m_axi_wdata  <= wdata_buffer;
                            m_axi_wvalid <= 1'b1;
                            m_axi_wlast  <= (beats_remaining == 32'd1);
                            beats_remaining <= beats_remaining - 32'd1;

                            if (beats_remaining == 32'd1) begin
                                // Last beat of this burst
                                // Advance HBM2 address for next burst
                                hbm_addr_lo <= hbm_addr_lo + 32'd8192;  // 256 × 32B = 8KB
                                state <= S_WAIT_BRESP;
                            end
                        end
                    end

                    // Abort check
                    if (control[CTRL_ABORT]) begin
                        m_axi_wvalid <= 1'b0;
                        transfer_active <= 1'b0;
                        status[STAT_BUSY] <= 1'b0;
                        status[STAT_ERROR] <= 1'b1;
                        state <= S_ERROR;
                    end
                end

                // Wait for write response (B channel)
                S_WAIT_BRESP: begin
                    m_axi_wvalid <= 1'b0;  // deassert W valid
                    m_axi_wlast  <= 1'b0;

                    if (m_axi_bvalid && m_axi_bready) begin
                        // Check for write error
                        if (m_axi_bresp != 2'b00) begin
                            status[STAT_ERROR] <= 1'b1;
                            state <= S_ERROR;
                        end else begin
                            state <= S_NEXT_BURST;
                        end
                    end
                end

                // Check if more bursts remain
                S_NEXT_BURST: begin
                    bursts_remaining <= bursts_remaining - 32'd1;
                    if (bursts_remaining > 32'd1) begin
                        // More bursts: re-issue AW for next 8KB chunk
                        hbm_addr_lo <= hbm_addr_lo;  // already advanced
                        state <= S_START_BURST;
                    end else begin
                        state <= S_DONE;
                    end
                end

                // Transfer complete
                S_DONE: begin
                    status[STAT_BUSY] <= 1'b0;
                    status[STAT_DONE] <= 1'b1;
                    transfer_active <= 1'b0;
                    // Wait for host to read STATUS, then clear on next START
                    state <= S_IDLE;
                end

                // Error state (sticky until next START)
                S_ERROR: begin
                    if (control[CTRL_START])
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
