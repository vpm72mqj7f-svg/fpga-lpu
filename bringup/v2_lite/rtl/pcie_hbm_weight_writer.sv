// =============================================================================
// pcie_hbm_weight_writer.sv — Pure AXI4 Weight Write Engine (no AVMM)
// Controlled by v2_lite_bar0_regs via start/addr/burst/data signals.
// =============================================================================

module pcie_hbm_weight_writer #(
    parameter int AXI_ADDR_W  = 28,
    parameter int AXI_DATA_W  = 256,
    parameter int AXI_TIMEOUT = 65536
) (
    input  logic                         clk, rst_n,

    // Control (from v2_lite_bar0_regs)
    input  logic                         start,
    input  logic                         abort,
    input  logic [27:0]                  hbm_addr,
    input  logic [23:0]                  burst_count,
    output logic                         busy, done, error,
    output logic [31:0]                  bytes_written,

    // Data input: 64-bit pairs, commit on hi_write
    input  logic [31:0]                  data_lo, data_hi,
    input  logic                         data_commit,

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

    // Data packer: 4×64b → 256b
    logic [1:0]  data_cnt;
    logic [AXI_DATA_W-1:0] wdata_buf;
    logic        wdata_rdy;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin data_cnt <= 0; wdata_buf <= 0; wdata_rdy <= 0; end
        else begin
            if (data_commit) begin
                data_cnt <= data_cnt + 1;
                unique case (data_cnt)
                    2'd0: wdata_buf[63:0]    <= {data_hi, data_lo};
                    2'd1: wdata_buf[127:64]  <= {data_hi, data_lo};
                    2'd2: wdata_buf[191:128] <= {data_hi, data_lo};
                    2'd3: begin wdata_buf[255:192] <= {data_hi, data_lo}; wdata_rdy <= 1; end
                endcase
            end
            if (wdata_rdy && m_axi_wvalid && m_axi_wready) wdata_rdy <= 0;
        end
    end

    assign m_axi_awid=0; assign m_axi_awsize=3'd5; assign m_axi_awburst=2'b01;
    assign m_axi_wstrb={AXI_DATA_W/8{1'b1}}; assign m_axi_bready=1'b1;

    // FSM
    typedef enum logic [2:0] { S_IDLE, S_AW, S_WDATA, S_BRESP, S_NEXT, S_DONE, S_ERR } st_t;
    st_t st;
    logic [31:0] addr, beats, bursts, timeout, perf_bytes;
    logic active, finish, err;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=S_IDLE; m_axi_awvalid<=0;m_axi_awaddr<=0;m_axi_awlen<=0;
            m_axi_wvalid<=0;m_axi_wdata<=0;m_axi_wlast<=0;
            addr<=0;beats<=0;bursts<=0;timeout<=0;active<=0;finish<=0;err<=0;perf_bytes<=0;
        end else begin
            m_axi_awvalid<=0;finish<=0;
            case(st)
                S_IDLE: if(start&&burst_count!=0)begin
                    bursts<=burst_count;addr<=hbm_addr;active<=1;err<=0;timeout<=0;st<=S_AW;end
                S_AW: begin
                    if(!m_axi_awvalid)begin m_axi_awvalid<=1;m_axi_awaddr<=addr[AXI_ADDR_W-1:0];m_axi_awlen<=255;beats<=256;end
                    if(m_axi_awvalid&&m_axi_awready)begin m_axi_awvalid<=0;st<=S_WDATA;timeout<=0;end
                    else begin timeout<=timeout+1;if(timeout>AXI_TIMEOUT)st<=S_ERR;end
                end
                S_WDATA: begin
                    if(wdata_rdy)begin
                        if(!m_axi_wvalid||(m_axi_wvalid&&m_axi_wready))begin
                            m_axi_wdata<=wdata_buf;m_axi_wvalid<=1;m_axi_wlast<=(beats==1);beats<=beats-1;perf_bytes<=perf_bytes+32;
                            if(beats==1)begin addr<=addr+8192;st<=S_BRESP;end
                        end
                    end
                    if(abort)begin m_axi_wvalid<=0;err<=1;st<=S_ERR;end
                    timeout<=timeout+1;if(timeout>AXI_TIMEOUT)st<=S_ERR;
                end
                S_BRESP: begin
                    m_axi_wvalid<=0;m_axi_wlast<=0;
                    if(m_axi_bvalid&&m_axi_bready)begin if(m_axi_bresp!=0)begin err<=1;st<=S_ERR;end else st<=S_NEXT;end
                end
                S_NEXT:begin bursts<=bursts-1;st<=(bursts>1)?S_AW:S_DONE;end
                S_DONE:begin active<=0;finish<=1;st<=S_IDLE;end
                S_ERR:begin active<=0;if(start)st<=S_IDLE;end
                default:st<=S_IDLE;
            endcase
        end
    end

    assign busy=active; assign done=finish; assign error=err; assign bytes_written=perf_bytes;

endmodule
