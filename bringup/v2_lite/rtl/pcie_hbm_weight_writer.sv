// =============================================================================
// pcie_hbm_weight_writer.sv — PCIe BAR0 → HBM2 AXI4 Weight Download Engine
//
// AVMM slave (PCIe clock domain) → CDC FIFO → AXI4 write master (core domain)
//
// Register Map (per v2_lite_pcie_regmap.atreg WT block, base 0x1000):
//   0x1000 WT_CONTROL        [31:0]  R/W  [0]=START [1]=ABORT [2]=AUTO_INCR
//   0x1004 WT_STATUS         [31:0]  R_O  [0]=BUSY [1]=DONE [2]=ERROR
//   0x1008 WT_HBM_ADDR_LO    [31:0]  R/W  HBM2 target address [27:0]
//   0x100C WT_HBM_ADDR_HI    [31:0]  R/W  HBM2 target address [63:32] (reserved)
//   0x1010 WT_BURST_COUNT    [31:0]  R/W  Number of 256-beat AXI bursts
//   0x1014 WT_BYTES_DONE     [31:0]  R_O  Total bytes written
//   0x1018 WT_ERROR_CODE     [31:0]  R/W/C Error flags
//   0x1020 WT_DATA_PORT      [31:0]  W_O  Stream writes pack to 256-bit AXI beats
//
// Clock domains:
//   pcie_clk (250MHz) — AVMM slave interface
//   core_clk (100MHz) — AXI4 write master interface
//   CDC FIFO bridges between them for control and data
// =============================================================================

module pcie_hbm_weight_writer #(
    parameter int AXI_ADDR_W  = 28,
    parameter int AXI_DATA_W  = 256,
    parameter int AXI_ID_W    = 9,
    parameter int BAR_ADDR_W  = 16     // BAR0 address width (64KB window)
) (
    // ---- PCIe Domain: AVMM Slave (from PCIe HIP rxm_bar0) ----
    input  logic                         pcie_clk,
    input  logic                         pcie_rst_n,
    input  logic [63:0]                  avs_address,      // byte address in BAR0
    input  logic [31:0]                  avs_writedata,
    output logic [31:0]                  avs_readdata,
    input  logic                         avs_write,
    input  logic                         avs_read,
    output logic                         avs_waitrequest,
    output logic                         avs_readdatavalid,
    input  logic [3:0]                   avs_byteenable,

    // ---- Core Domain: AXI4 Write Master (to HBM2 ed_synth) ----
    input  logic                         core_clk,
    input  logic                         core_rst_n,

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
    // Register Map (WT block, BAR0 base = 0x1000)
    // =========================================================================
    localparam logic [BAR_ADDR_W-1:0] WT_BASE       = 16'h1000;
    localparam logic [BAR_ADDR_W-1:0] ADDR_CONTROL  = WT_BASE + 16'h00;
    localparam logic [BAR_ADDR_W-1:0] ADDR_STATUS   = WT_BASE + 16'h04;
    localparam logic [BAR_ADDR_W-1:0] ADDR_HBM_LO   = WT_BASE + 16'h08;
    localparam logic [BAR_ADDR_W-1:0] ADDR_HBM_HI   = WT_BASE + 16'h0C;
    localparam logic [BAR_ADDR_W-1:0] ADDR_BURST_CNT= WT_BASE + 16'h10;
    localparam logic [BAR_ADDR_W-1:0] ADDR_BYTES_DONE=WT_BASE + 16'h14;
    localparam logic [BAR_ADDR_W-1:0] ADDR_ERROR     = WT_BASE + 16'h18;
    localparam logic [BAR_ADDR_W-1:0] ADDR_DATA_PORT = WT_BASE + 16'h20;

    localparam int CTRL_START = 0;
    localparam int CTRL_ABORT = 1;

    localparam int STAT_BUSY  = 0;
    localparam int STAT_DONE  = 1;
    localparam int STAT_ERROR = 2;

    // =========================================================================
    // PCIe Domain: AVMM Slave Register Access
    // =========================================================================
    logic [31:0]  hbm_addr_lo, hbm_addr_hi, burst_count, control, status;
    logic [31:0]  bytes_done, error_code;
    logic [31:0]  data_port_wr;
    logic         data_port_wr_pulse;

    // Simple AVMM: single-cycle reads (no waitrequest needed for register access)
    assign avs_waitrequest = 1'b0;  // registers are always ready

    always_ff @(posedge pcie_clk or negedge pcie_rst_n) begin
        if (!pcie_rst_n) begin
            hbm_addr_lo  <= 32'd0;
            hbm_addr_hi  <= 32'd0;
            burst_count  <= 32'd0;
            control      <= 32'd0;
            data_port_wr <= 32'd0;
            data_port_wr_pulse <= 1'b0;
        end else begin
            data_port_wr_pulse <= 1'b0;
            // Auto-clear START after one cycle
            if (control[CTRL_START]) control[CTRL_START] <= 1'b0;

            if (avs_write) begin
                unique case (avs_address[BAR_ADDR_W-1:0])
                    ADDR_HBM_LO:    hbm_addr_lo  <= avs_writedata;
                    ADDR_HBM_HI:    hbm_addr_hi  <= avs_writedata;
                    ADDR_BURST_CNT: burst_count  <= avs_writedata;
                    ADDR_CONTROL:   control      <= avs_writedata;
                    ADDR_DATA_PORT: begin
                        data_port_wr <= avs_writedata;
                        data_port_wr_pulse <= 1'b1;
                    end
                    default: ;
                endcase
            end
        end
    end

    // AVMM read mux (combinational)
    always_comb begin
        avs_readdata = 32'd0;
        unique case (avs_address[BAR_ADDR_W-1:0])
            ADDR_HBM_LO:    avs_readdata = hbm_addr_lo;
            ADDR_HBM_HI:    avs_readdata = hbm_addr_hi;
            ADDR_BURST_CNT: avs_readdata = burst_count;
            ADDR_CONTROL:   avs_readdata = control;
            ADDR_STATUS:    avs_readdata = status;
            ADDR_BYTES_DONE:avs_readdata = bytes_done;
            ADDR_ERROR:     avs_readdata = error_code;
            default:        avs_readdata = 32'd0;
        endcase
    end

    always_ff @(posedge pcie_clk or negedge pcie_rst_n) begin
        if (!pcie_rst_n)
            avs_readdatavalid <= 1'b0;
        else
            avs_readdatavalid <= avs_read;
    end

    // =========================================================================
    // CDC: Status readback from core domain → PCIe domain (toggle handshake)
    // =========================================================================
    logic        status_req_toggle, status_ack_toggle;
    logic        status_req_sync1, status_req_sync2;

    // PCIe domain: request status, receive via toggle handshake
    always_ff @(posedge pcie_clk or negedge pcie_rst_n) begin
        if (!pcie_rst_n) begin
            status_req_toggle <= 1'b0;
        end else begin
            // Receive: when ack toggles to match request, latch status data
            if (status_ack_sync2 == status_req_toggle)
                status_req_toggle <= ~status_req_toggle;
        end
    end

    // Synchronize ack from core to PCIe domain
    logic status_ack_sync1, status_ack_sync2;
    always_ff @(posedge pcie_clk or negedge pcie_rst_n) begin
        if (!pcie_rst_n) begin
            status_ack_sync1 <= 1'b0;
            status_ack_sync2 <= 1'b0;
        end else begin
            status_ack_sync1 <= status_ack_toggle;
            status_ack_sync2 <= status_ack_sync1;
        end
    end

    // Core domain: provide status when requested
    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            status_req_sync1 <= 1'b0;
            status_req_sync2 <= 1'b0;
            status_ack_toggle <= 1'b0;
        end else begin
            status_req_sync1 <= status_req_toggle;
            status_req_sync2 <= status_req_sync1;
            if (status_req_sync2 != status_ack_toggle) begin
                // Latch status + bytes_done on request
                status     <= {28'd0, transfer_error, wdata_buffer_valid, transfer_active, transfer_done};
                bytes_done <= perf_bytes_done;
                error_code <= {31'd0, transfer_error};
                status_ack_toggle <= status_req_sync2;
            end
        end
    end

    // =========================================================================
    // Core Domain: Data Packer + AXI4 Write FSM
    // =========================================================================
    // Data packer: 8 × 32-bit PCIe writes → 1 × 256-bit AXI beat
    // CDC: data_port_wr_pulse crosses with data_port_wr value
    logic [2:0]   data_wr_cnt;       // 0..7
    logic [AXI_DATA_W-1:0] wdata_buffer;
    logic         wdata_buffer_valid;
    (* preserve *) logic        data_wr_pulse_cdc;  // CDC: data write strobe

    // Simple pulse synchronizer for data write
    logic [2:0]   data_wr_pulse_sync;
    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n)
            data_wr_pulse_sync <= 3'd0;
        else
            data_wr_pulse_sync <= {data_wr_pulse_sync[1:0], data_port_wr_pulse};
    end
    assign data_wr_pulse_cdc = data_wr_pulse_sync[2:1] == 2'b01;  // rising edge

    // Core data packer
    logic [31:0] data_port_core;
    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            data_port_core <= 32'd0;
        end else if (data_port_wr_pulse) begin
            data_port_core <= data_port_wr;
        end
    end

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            data_wr_cnt       <= 3'd0;
            wdata_buffer      <= 256'd0;
            wdata_buffer_valid <= 1'b0;
        end else begin
            if (data_wr_pulse_cdc) begin
                data_wr_cnt <= data_wr_cnt + 3'd1;
                unique case (data_wr_cnt)
                    3'd0: wdata_buffer[31:0]    <= data_port_core;
                    3'd1: wdata_buffer[63:32]   <= data_port_core;
                    3'd2: wdata_buffer[95:64]   <= data_port_core;
                    3'd3: wdata_buffer[127:96]  <= data_port_core;
                    3'd4: wdata_buffer[159:128] <= data_port_core;
                    3'd5: wdata_buffer[191:160] <= data_port_core;
                    3'd6: wdata_buffer[223:192] <= data_port_core;
                    3'd7: begin
                        wdata_buffer[255:224] <= data_port_core;
                        wdata_buffer_valid <= 1'b1;
                    end
                endcase
            end
            if (wdata_buffer_valid && m_axi_wvalid && m_axi_wready)
                wdata_buffer_valid <= 1'b0;
        end
    end

    // =========================================================================
    // CDC: Control register crossing
    // =========================================================================
    logic [31:0] hbm_addr_lo_core, burst_count_core;
    logic        start_pulse, abort_pulse;

    // Toggle synchronizer for START and ABORT
    logic start_toggle, start_toggle_sync1, start_toggle_sync2, start_toggle_core;
    logic abort_toggle, abort_toggle_sync1, abort_toggle_sync2, abort_toggle_core;

    always_ff @(posedge pcie_clk or negedge pcie_rst_n) begin
        if (!pcie_rst_n) begin
            start_toggle <= 1'b0;
            abort_toggle <= 1'b0;
        end else begin
            if (control[CTRL_START])  start_toggle <= ~start_toggle;
            if (control[CTRL_ABORT])  abort_toggle <= ~abort_toggle;
        end
    end

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            start_toggle_sync1 <= 1'b0; start_toggle_sync2 <= 1'b0; start_toggle_core <= 1'b0;
            abort_toggle_sync1 <= 1'b0; abort_toggle_sync2 <= 1'b0; abort_toggle_core <= 1'b0;
            start_pulse <= 1'b0; abort_pulse <= 1'b0;
            hbm_addr_lo_core <= 32'd0; burst_count_core <= 32'd0;
        end else begin
            start_toggle_sync1 <= start_toggle;
            start_toggle_sync2 <= start_toggle_sync1;
            if (start_toggle_sync2 != start_toggle_core) begin
                start_toggle_core <= start_toggle_sync2;
                start_pulse       <= start_toggle_sync2;
                // Latch control values on START
                hbm_addr_lo_core  <= hbm_addr_lo;
                burst_count_core  <= burst_count;
            end else start_pulse <= 1'b0;

            abort_toggle_sync1 <= abort_toggle;
            abort_toggle_sync2 <= abort_toggle_sync1;
            if (abort_toggle_sync2 != abort_toggle_core) begin
                abort_toggle_core <= abort_toggle_sync2;
                abort_pulse       <= 1'b1;
            end else abort_pulse <= 1'b0;
        end
    end

    // =========================================================================
    // AXI4 Constants
    // =========================================================================
    assign m_axi_awid    = '0;
    assign m_axi_awsize  = 3'd5;      // 32 bytes per beat (256-bit)
    assign m_axi_awburst = 2'b01;     // INCR
    assign m_axi_awprot  = 3'b000;
    assign m_axi_awqos   = 4'd0;
    assign m_axi_wstrb   = {AXI_DATA_W/8{1'b1}};
    assign m_axi_bready  = 1'b1;

    // =========================================================================
    // Core AXI4 Write FSM
    // =========================================================================
    typedef enum logic [2:0] {
        S_IDLE, S_START_BURST, S_WRITE_DATA, S_WAIT_BRESP, S_NEXT_BURST, S_DONE, S_ERROR
    } state_t;
    state_t state;

    logic [31:0] hbm_addr_reg;
    logic [31:0] beats_remaining;
    logic [31:0] bursts_remaining;
    logic        transfer_active;
    logic        transfer_done, transfer_error;
    logic [31:0] perf_bytes_done;

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            state            <= S_IDLE;
            m_axi_awvalid    <= 1'b0;
            m_axi_awaddr     <= {AXI_ADDR_W{1'b0}};
            m_axi_awlen      <= 8'd0;
            m_axi_wvalid     <= 1'b0;
            m_axi_wdata      <= 256'd0;
            m_axi_wlast      <= 1'b0;
            beats_remaining  <= 32'd0;
            bursts_remaining <= 32'd0;
            transfer_active  <= 1'b0;
            transfer_done    <= 1'b0;
            transfer_error   <= 1'b0;
            perf_bytes_done  <= 32'd0;
            hbm_addr_reg     <= 32'd0;
        end else begin
            m_axi_awvalid <= 1'b0;  // pulsed
            transfer_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start_pulse) begin
                        if (burst_count_core == 32'd0) begin
                            transfer_error <= 1'b1;
                            state <= S_ERROR;
                        end else begin
                            bursts_remaining <= burst_count_core;
                            hbm_addr_reg     <= hbm_addr_lo_core;
                            transfer_active  <= 1'b1;
                            state <= S_START_BURST;
                        end
                    end
                end

                S_START_BURST: begin
                    if (!m_axi_awvalid) begin
                        m_axi_awlen    <= 8'd255;   // 256 beats − 1
                        m_axi_awaddr   <= hbm_addr_reg[AXI_ADDR_W-1:0];
                        m_axi_awvalid  <= 1'b1;
                        beats_remaining <= 32'd256;
                        data_wr_cnt     <= 3'd0;
                        state <= S_WRITE_DATA;
                    end
                end

                S_WRITE_DATA: begin
                    if (wdata_buffer_valid) begin
                        if (!m_axi_wvalid || (m_axi_wvalid && m_axi_wready)) begin
                            m_axi_wdata  <= wdata_buffer;
                            m_axi_wvalid <= 1'b1;
                            m_axi_wlast  <= (beats_remaining == 32'd1);
                            beats_remaining <= beats_remaining - 32'd1;
                            perf_bytes_done <= perf_bytes_done + 32'd32;

                            if (beats_remaining == 32'd1) begin
                                hbm_addr_reg <= hbm_addr_reg + 32'd8192;
                                state <= S_WAIT_BRESP;
                            end
                        end
                    end
                    if (abort_pulse) begin
                        m_axi_wvalid <= 1'b0;
                        transfer_active <= 1'b0;
                        transfer_error <= 1'b1;
                        state <= S_ERROR;
                    end
                end

                S_WAIT_BRESP: begin
                    m_axi_wvalid <= 1'b0;
                    m_axi_wlast  <= 1'b0;
                    if (m_axi_bvalid && m_axi_bready) begin
                        if (m_axi_bresp != 2'b00) begin
                            transfer_error <= 1'b1;
                            state <= S_ERROR;
                        end else state <= S_NEXT_BURST;
                    end
                end

                S_NEXT_BURST: begin
                    bursts_remaining <= bursts_remaining - 32'd1;
                    if (bursts_remaining > 32'd1)
                        state <= S_START_BURST;
                    else begin
                        transfer_done <= 1'b1;
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    transfer_active <= 1'b0;
                    state <= S_IDLE;
                end

                S_ERROR: begin
                    if (start_pulse)
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
