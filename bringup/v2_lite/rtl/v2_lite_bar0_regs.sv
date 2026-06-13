// =============================================================================
// v2_lite_bar0_regs.sv — Unified PCIe BAR0 Register Map
// All blocks: SYS, WT, FFN, ACT, PERF, ERR
// Software reference: v2_lite_pcie_regmap.atreg
// =============================================================================

module v2_lite_bar0_regs #(
    parameter BAR_ADDR_W = 12       // 4KB BAR0
) (
    input  logic         clk, rst_n,

    // AVMM Slave (from PCIe HIP rxm_bar0, via 2FF sync in top-level)
    input  logic [63:0]  avs_address,
    input  logic [31:0]  avs_writedata,
    output logic [31:0]  avs_readdata,
    input  logic         avs_write, avs_read,
    output logic         avs_readdatavalid,

    // ---- WT: weight transfer signals ----
    output logic         wt_start,
    output logic         wt_abort,
    output logic [27:0]  wt_hbm_addr,
    output logic [23:0]  wt_burst_cnt,
    input  logic         wt_busy, wt_done, wt_error,
    input  logic [31:0]  wt_bytes_done,
    output logic [31:0]  wt_data_lo, wt_data_hi,
    output logic         wt_data_commit,   // pulse when HI written

    // ---- FFN: status readback from engine ----
    input  logic [3:0]   ffn_fsm_state,
    input  logic [2:0]   ffn_expert_cnt,
    input  logic         ffn_busy, ffn_done, ffn_pass, ffn_error,
    input  logic [31:0]  ffn_token_cnt, ffn_cycle_cnt,
    input  logic [31:0]  ffn_expert_total, ffn_axi_rbeat,

    // ---- SYS: global status ----
    input  logic [3:0]   sys_led,
    input  logic         sys_hbm_tg_pass, sys_pcie_pll_lock, sys_pcie_link_up,
    input  logic [7:0]   sys_pcie_ltssm,

    // ---- PERF: performance counters ----
    input  logic [31:0]  perf_hbm_bw_read, perf_hbm_bw_write,
    input  logic [31:0]  perf_pcie_bw_rx, perf_pcie_bw_tx,
    input  logic [31:0]  perf_total_cycles,

    // ---- ERR: error/diag ----
    input  logic [15:0]  err_sa_gate_fsm, err_sa_down_fsm,
    input  logic [2:0]   err_hbm2r_fsm, err_hbm2r_wr_wm, err_hbm2r_rd_wm,
    input  logic         err_ffn_merge, err_ffn_silu, err_axi_resp
);

    // =========================================================================
    // Address decode
    // =========================================================================
    wire is_wt   = (avs_address[11:8] == 4'h0);
    wire is_sys  = (avs_address[11:8] == 4'h1);
    wire is_ffn  = (avs_address[11:8] == 4'h2);
    wire is_act  = (avs_address[11:8] == 4'h3);
    wire is_perf = (avs_address[11:8] == 4'h4);
    wire is_err  = (avs_address[11:8] == 4'h5);

    // =========================================================================
    // AVMM sync + pulse detect
    // =========================================================================
    logic avs_wr_s1, avs_wr_s2, avs_wr_s3;
    logic avs_rd_s1, avs_rd_s2, avs_rd_s3;
    always_ff @(posedge clk) begin
        avs_wr_s1 <= avs_write; avs_wr_s2 <= avs_wr_s1; avs_wr_s3 <= avs_wr_s2;
        avs_rd_s1 <= avs_read;  avs_rd_s2 <= avs_rd_s1;  avs_rd_s3 <= avs_rd_s2;
    end
    wire wr_pulse = avs_wr_s2 && !avs_wr_s3;
    wire rd_pulse = avs_rd_s2  && !avs_rd_s3;

    logic rd_d1;
    always_ff @(posedge clk) begin rd_d1 <= rd_pulse; avs_readdatavalid <= rd_d1; end

    // =========================================================================
    // SYS Registers (0x100–0x1FF)
    // =========================================================================
    localparam ADDR_SYS_VERSION  = 12'h100;
    localparam ADDR_SYS_CONTROL  = 12'h108;
    localparam ADDR_SYS_STATUS   = 12'h10C;

    logic [31:0] sys_control;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) sys_control <= 0;
        else if (wr_pulse && avs_address[11:0] == ADDR_SYS_CONTROL)
            sys_control <= avs_writedata;
    end

    wire [31:0] sys_status = {
        12'd0,
        sys_pcie_ltssm,
        sys_pcie_link_up, sys_pcie_pll_lock, sys_hbm_tg_pass,
        sys_led
    };

    // =========================================================================
    // WT Registers (0x000–0x0FF)
    // =========================================================================
    localparam ADDR_WT_CTRL   = 12'h000;
    localparam ADDR_WT_STATUS = 12'h004;
    localparam ADDR_WT_HBM_LO = 12'h008;
    localparam ADDR_WT_HBM_HI = 12'h00C;
    localparam ADDR_WT_BURST  = 12'h010;
    localparam ADDR_WT_BYTES  = 12'h014;
    localparam ADDR_WT_ERROR  = 12'h018;
    localparam ADDR_WT_DATA_LO= 12'h020;
    localparam ADDR_WT_DATA_HI= 12'h024;

    logic [31:0] wt_ctrl, wt_hbm_lo, wt_hbm_hi, wt_burst;
    logic [31:0] wt_status, wt_bytes, wt_error_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wt_ctrl <= 0; wt_hbm_lo <= 0; wt_hbm_hi <= 0; wt_burst <= 0;
            wt_data_lo <= 0; wt_data_hi <= 0; wt_data_commit <= 0;
        end else begin
            wt_data_commit <= 0;
            if (wt_ctrl[0]) wt_ctrl[0] <= 0;
            if (wr_pulse) begin
                unique case (avs_address[11:0])
                    ADDR_WT_CTRL:    wt_ctrl    <= avs_writedata;
                    ADDR_WT_HBM_LO:  wt_hbm_lo  <= avs_writedata;
                    ADDR_WT_HBM_HI:  wt_hbm_hi  <= avs_writedata;
                    ADDR_WT_BURST:   wt_burst   <= avs_writedata;
                    ADDR_WT_DATA_LO: wt_data_lo <= avs_writedata;
                    ADDR_WT_DATA_HI: begin wt_data_hi <= avs_writedata; wt_data_commit <= 1'b1; end
                    default: ;
                endcase
            end
        end
    end

    assign wt_start      = wt_ctrl[0];
    assign wt_abort      = wt_ctrl[1];
    assign wt_hbm_addr   = wt_hbm_lo[27:0];
    assign wt_burst_cnt  = wt_burst[23:0];

    always_ff @(posedge clk) begin
        wt_status <= {28'd0, wt_error, wt_busy, wt_done};
        if (wt_done)  wt_bytes <= wt_bytes_done;
        if (wt_error) wt_error_reg[0] <= 1'b1;
        if (wt_ctrl[0]) wt_error_reg <= 0;
    end

    // =========================================================================
    // FFN Registers (0x200–0x2FF)
    // =========================================================================
    localparam ADDR_FFN_CONTROL    = 12'h200;
    localparam ADDR_FFN_STATUS     = 12'h204;
    localparam ADDR_FFN_TOKEN_CNT  = 12'h208;
    localparam ADDR_FFN_CYCLE_CNT  = 12'h20C;
    localparam ADDR_FFN_EXPERT_CNT = 12'h210;
    localparam ADDR_FFN_AXI_RBEAT  = 12'h214;
    localparam ADDR_FFN_THROUGHPUT = 12'h218;

    wire [31:0] ffn_status = {
        13'd0,
        ffn_expert_cnt,
        ffn_error, ffn_pass, ffn_done, ffn_busy,
        ffn_fsm_state
    };

    // =========================================================================
    // Read Mux
    // =========================================================================
    always_comb begin
        avs_readdata = 32'd0;
        unique case (avs_address[11:0])
            // SYS
            ADDR_SYS_VERSION:  avs_readdata = 32'h13061A02;  // {day(13), month(06), year-2000(26), build(2)}
            ADDR_SYS_CONTROL:  avs_readdata = sys_control;
            ADDR_SYS_STATUS:   avs_readdata = sys_status;
            // WT
            ADDR_WT_CTRL:      avs_readdata = wt_ctrl;
            ADDR_WT_STATUS:    avs_readdata = wt_status;
            ADDR_WT_HBM_LO:    avs_readdata = wt_hbm_lo;
            ADDR_WT_HBM_HI:    avs_readdata = wt_hbm_hi;
            ADDR_WT_BURST:     avs_readdata = wt_burst;
            ADDR_WT_BYTES:     avs_readdata = wt_bytes;
            ADDR_WT_ERROR:     avs_readdata = wt_error_reg;
            // FFN
            ADDR_FFN_CONTROL:   avs_readdata = 32'd0;
            ADDR_FFN_STATUS:    avs_readdata = ffn_status;
            ADDR_FFN_TOKEN_CNT: avs_readdata = ffn_token_cnt;
            ADDR_FFN_CYCLE_CNT: avs_readdata = ffn_cycle_cnt;
            ADDR_FFN_EXPERT_CNT:avs_readdata = ffn_expert_total;
            ADDR_FFN_AXI_RBEAT: avs_readdata = ffn_axi_rbeat;
            // PERF stubs → can be extended
            12'h400: avs_readdata = perf_hbm_bw_read;
            12'h404: avs_readdata = perf_hbm_bw_write;
            12'h408: avs_readdata = perf_pcie_bw_rx;
            12'h40C: avs_readdata = perf_pcie_bw_tx;
            12'h410: avs_readdata = perf_total_cycles;
            // ERR stubs → can be extended
            12'h500: avs_readdata = {16'd0, err_sa_down_fsm, err_sa_gate_fsm};
            12'h508: avs_readdata = {23'd0, err_hbm2r_rd_wm, err_hbm2r_wr_wm, err_hbm2r_fsm};
            default: avs_readdata = 32'd0;
        endcase
    end

endmodule
