// =============================================================================
// pcie_hbm_weight_writer.sv — PCIe BAR0 → HBM2 AXI4 Weight Download Engine
// Single clock domain (core_clk). AVMM sync via 2-stage FF.
// =============================================================================

module pcie_hbm_weight_writer #(
    parameter int AXI_ADDR_W  = 28,
    parameter int AXI_DATA_W  = 256,
    parameter int AXI_TIMEOUT = 65536
) (
    input  logic                         clk,          // core_clk (100MHz)
    input  logic                         rst_n,

    // AVMM Slave (sync'd from PCIe domain via 2FF)
    input  logic [63:0]                  avs_address,
    input  logic [31:0]                  avs_writedata,
    output logic [31:0]                  avs_readdata,
    input  logic                         avs_write,
    input  logic                         avs_read,
    output logic                         avs_waitrequest,
    output logic                         avs_readdatavalid,
    input  logic [3:0]                   avs_byteenable,

    // AXI4 Write Master
    output logic [8:0]                   m_axi_awid,
    output logic [AXI_ADDR_W-1:0]        m_axi_awaddr,
    output logic [7:0]                   m_axi_awlen,
    output logic [2:0]                   m_axi_awsize,
    output logic [1:0]                   m_axi_awburst,
    output logic                         m_axi_awvalid,
    input  logic                         m_axi_awready,
    output logic [AXI_DATA_W-1:0]        m_axi_wdata,
    output logic [AXI_DATA_W/8-1:0]      m_axi_wstrb,
    output logic                         m_axi_wlast,
    output logic                         m_axi_wvalid,
    input  logic                         m_axi_wready,
    input  logic [1:0]                   m_axi_bresp,
    input  logic                         m_axi_bvalid,
    output logic                         m_axi_bready
);

    // =========================================================================
    // Register Map (BAR0 base = 0x000)
    // =========================================================================
    localparam logic [11:0] ADDR_WT_CTRL   = 12'h000;
    localparam logic [11:0] ADDR_WT_STATUS = 12'h004;
    localparam logic [11:0] ADDR_WT_HBM_LO = 12'h008;
    localparam logic [11:0] ADDR_WT_HBM_HI = 12'h00C;
    localparam logic [11:0] ADDR_WT_BURST  = 12'h010;
    localparam logic [11:0] ADDR_WT_BYTES  = 12'h014;
    localparam logic [11:0] ADDR_WT_ERROR  = 12'h018;
    localparam logic [11:0] ADDR_WT_DATA_LO= 12'h020;
    localparam logic [11:0] ADDR_WT_DATA_HI= 12'h024;

    // =========================================================================
    // 2-stage synchronizer for AVMM control signals
    // =========================================================================
    logic avs_write_s1, avs_write_s2, avs_write_s3;
    logic avs_read_s1,  avs_read_s2,  avs_read_s3;
    always_ff @(posedge clk) begin
        avs_write_s1 <= avs_write; avs_write_s2 <= avs_write_s1; avs_write_s3 <= avs_write_s2;
        avs_read_s1  <= avs_read;  avs_read_s2  <= avs_read_s1;  avs_read_s3  <= avs_read_s2;
    end
    wire avs_wr_pulse = avs_write_s2 && !avs_write_s3;  // rising edge detect
    wire avs_rd_pulse = avs_read_s2  && !avs_read_s3;

    assign avs_waitrequest = 1'b0;

    // =========================================================================
    // Register File
    // =========================================================================
    logic [31:0] reg_ctrl, reg_hbm_lo, reg_hbm_hi, reg_burst;
    logic [31:0] reg_status, reg_bytes, reg_error;
    logic [31:0] data_lo, data_hi;
    logic        data_commit;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_ctrl <= 0; reg_hbm_lo <= 0; reg_hbm_hi <= 0; reg_burst <= 0;
            reg_status <= 0; reg_bytes <= 0; reg_error <= 0;
            data_lo <= 0; data_hi <= 0; data_commit <= 0;
        end else begin
            data_commit <= 0;
            if (reg_ctrl[0]) reg_ctrl[0] <= 0;  // auto-clear START

            if (avs_wr_pulse) begin
                unique case (avs_address[11:0])
                    ADDR_WT_CTRL:    reg_ctrl   <= avs_writedata;
                    ADDR_WT_HBM_LO:  reg_hbm_lo <= avs_writedata;
                    ADDR_WT_HBM_HI:  reg_hbm_hi <= avs_writedata;
                    ADDR_WT_BURST:   reg_burst  <= avs_writedata;
                    ADDR_WT_DATA_LO: data_lo    <= avs_writedata;
                    ADDR_WT_DATA_HI: begin data_hi <= avs_writedata; data_commit <= 1'b1; end
                    default: ;
                endcase
            end
        end
    end

    // AVMM read (1-cycle delay)
    logic avs_rd_d1;
    always_ff @(posedge clk) begin
        avs_rd_d1 <= avs_rd_pulse;
        avs_readdatavalid <= avs_rd_d1;
    end

    always_comb begin
        avs_readdata = 32'd0;
        unique case (avs_address[11:0])
            ADDR_WT_CTRL:   avs_readdata = reg_ctrl;
            ADDR_WT_STATUS: avs_readdata = reg_status;
            ADDR_WT_HBM_LO: avs_readdata = reg_hbm_lo;
            ADDR_WT_HBM_HI: avs_readdata = reg_hbm_hi;
            ADDR_WT_BURST:  avs_readdata = reg_burst;
            ADDR_WT_BYTES:  avs_readdata = reg_bytes;
            ADDR_WT_ERROR:  avs_readdata = reg_error;
            default:        avs_readdata = 32'd0;  // stub regions return 0
        endcase
    end

    // =========================================================================
    // Data Packer: 4 × 64-bit pairs → 1 × 256-bit AXI beat
    // =========================================================================
    logic [1:0]  data_cnt;
    logic [AXI_DATA_W-1:0] wdata_buf;
    logic        wdata_rdy;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_cnt <= 0; wdata_buf <= 0; wdata_rdy <= 0;
        end else begin
            if (data_commit) begin
                data_cnt <= data_cnt + 2'd1;
                unique case (data_cnt)
                    2'd0: wdata_buf[63:0]    <= {data_hi, data_lo};
                    2'd1: wdata_buf[127:64]  <= {data_hi, data_lo};
                    2'd2: wdata_buf[191:128] <= {data_hi, data_lo};
                    2'd3: begin wdata_buf[255:192] <= {data_hi, data_lo}; wdata_rdy <= 1'b1; end
                endcase
            end
            if (wdata_rdy && m_axi_wvalid && m_axi_wready) wdata_rdy <= 0;
        end
    end

    // =========================================================================
    // Status update
    // =========================================================================
    always_ff @(posedge clk) begin
        reg_status <= {28'd0, transfer_error, data_commit, transfer_active, transfer_done};
        if (transfer_done) reg_bytes <= perf_bytes;
        if (transfer_error) reg_error[0] <= 1'b1;
        if (reg_ctrl[0])    reg_error <= 0;
    end

    // =========================================================================
    // AXI4 Constants
    // =========================================================================
    assign m_axi_awid    = 0;
    assign m_axi_awsize  = 3'd5;
    assign m_axi_awburst = 2'b01;
    assign m_axi_wstrb   = {AXI_DATA_W/8{1'b1}};
    assign m_axi_bready  = 1'b1;

    // =========================================================================
    // AXI4 Write FSM
    // =========================================================================
    typedef enum logic [2:0] { S_IDLE, S_AW, S_WDATA, S_BRESP, S_NEXT, S_DONE, S_ERR } state_t;
    state_t state;
    logic [31:0] hbm_addr, beats_left, bursts_left, timeout_cnt, perf_bytes;
    logic        transfer_active, transfer_done, transfer_error;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            m_axi_awvalid <= 0; m_axi_awaddr <= 0; m_axi_awlen <= 0;
            m_axi_wvalid <= 0; m_axi_wdata <= 0; m_axi_wlast <= 0;
            hbm_addr <= 0; beats_left <= 0; bursts_left <= 0; timeout_cnt <= 0;
            transfer_active <= 0; transfer_done <= 0; transfer_error <= 0;
            perf_bytes <= 0;
        end else begin
            m_axi_awvalid <= 0; transfer_done <= 0;

            case (state)
                S_IDLE: begin
                    if (reg_ctrl[0] && reg_burst != 0) begin
                        bursts_left <= reg_burst; hbm_addr <= reg_hbm_lo;
                        transfer_active <= 1; transfer_error <= 0; timeout_cnt <= 0;
                        state <= S_AW;
                    end
                end

                S_AW: begin
                    if (!m_axi_awvalid) begin
                        m_axi_awvalid <= 1; m_axi_awaddr <= hbm_addr[AXI_ADDR_W-1:0];
                        m_axi_awlen <= 255; beats_left <= 256;
                    end
                    if (m_axi_awvalid && m_axi_awready) begin
                        m_axi_awvalid <= 0; state <= S_WDATA; timeout_cnt <= 0;
                    end else begin
                        timeout_cnt <= timeout_cnt + 1;
                        if (timeout_cnt > AXI_TIMEOUT) state <= S_ERR;
                    end
                end

                S_WDATA: begin
                    if (wdata_rdy) begin
                        if (!m_axi_wvalid || (m_axi_wvalid && m_axi_wready)) begin
                            m_axi_wdata <= wdata_buf; m_axi_wvalid <= 1;
                            m_axi_wlast <= (beats_left == 1); beats_left <= beats_left - 1;
                            perf_bytes <= perf_bytes + 32;
                            if (beats_left == 1) begin
                                hbm_addr <= hbm_addr + 8192; state <= S_BRESP;
                            end
                        end
                    end
                    if (reg_ctrl[1]) begin m_axi_wvalid <= 0; transfer_error <= 1; state <= S_ERR; end
                    timeout_cnt <= timeout_cnt + 1;
                    if (timeout_cnt > AXI_TIMEOUT) state <= S_ERR;
                end

                S_BRESP: begin
                    m_axi_wvalid <= 0; m_axi_wlast <= 0;
                    if (m_axi_bvalid && m_axi_bready) begin
                        if (m_axi_bresp != 0) begin transfer_error <= 1; state <= S_ERR; end
                        else state <= S_NEXT;
                    end
                end

                S_NEXT: begin
                    bursts_left <= bursts_left - 1;
                    state <= (bursts_left > 1) ? S_AW : S_DONE;
                end

                S_DONE: begin transfer_active <= 0; transfer_done <= 1; state <= S_IDLE; end
                S_ERR:  begin transfer_active <= 0; if (reg_ctrl[0]) state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
