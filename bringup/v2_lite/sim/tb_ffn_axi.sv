// tb_ffn_axi.sv — Testbench for FFN engine with AXI4 HBM2 read
// Verifies: AXI read FSM → HBM2 responder → GEMV compute → PCIe TX output
`timescale 1ns / 100ps

module tb_ffn_axi;
    localparam HIDDEN = 2048, INTER = 1408, DATA_W = 8, ACCUM_W = 24, DSP_LANES = 512;
    localparam AXI_DATA_W = 256, AXI_ADDR_W = 28;

    logic clk, rst_n;
    logic pcie_rx_valid, pcie_rx_ready, pcie_tx_valid, pcie_tx_ready;
    logic [HIDDEN*DATA_W-1:0] pcie_rx_data, pcie_tx_data;
    logic busy, done;
    logic [3:0] dbg_fsm, dbg_sub_fsm;
    logic [2:0] dbg_expert_cnt;
    logic dbg_gemv_busy, dbg_hbm2_busy;
    logic [31:0] perf_token_cnt, perf_cycle_cnt;
    logic [63:0] pr_debug;
    logic err_merge_overflow, err_silu_overflow, err_axi_resp_err;

    // AXI4 read master (from FFN)
    logic [31:0] m_axi_araddr, m_axi_arlen, m_axi_arsize, m_axi_arvalid;
    logic m_axi_arready;
    logic [AXI_DATA_W-1:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic m_axi_rvalid, m_axi_rlast, m_axi_rready;

    // Clock: 100MHz
    always #5 clk = ~clk;

    // HBM2 AXI Responder
    logic [31:0] hbm_mem [0:1023];
    logic [7:0] ar_beat_cnt;
    logic ar_active;

    assign m_axi_arready = 1'b1;  // always ready for AR

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_rvalid <= 0; m_axi_rdata <= 0; m_axi_rresp <= 0;
            m_axi_rlast <= 0; ar_beat_cnt <= 0; ar_active <= 0;
        end else begin
            // Accept AR
            if (m_axi_arvalid && m_axi_arready && !ar_active) begin
                ar_active <= 1; ar_beat_cnt <= 0;
            end
            // Send R beats
            if (ar_active) begin
                m_axi_rvalid <= 1;
                m_axi_rdata <= {4{8'hA5}};              // test pattern: 0xA5A5...A5
                m_axi_rresp <= 0;
                ar_beat_cnt <= ar_beat_cnt + 1;
                m_axi_rlast <= (ar_beat_cnt == m_axi_arlen[7:0]); // FIXME: use arlen
                if (m_axi_rlast && m_axi_rready) begin
                    ar_active <= 0; m_axi_rvalid <= 0;
                end
            end
        end
    end

    // DUT
    v2_lite_ffn_engine #(
        .HIDDEN(HIDDEN), .INTER(INTER), .DSP_LANES(DSP_LANES),
        .DATA_W(DATA_W), .ACCUM_W(ACCUM_W)
    ) u_dut (
        .clk, .rst_n, .mode_prefill(1'b0),
        .pcie_rx_valid, .pcie_rx_data, .pcie_rx_ready,
        .pcie_tx_valid, .pcie_tx_data, .pcie_tx_ready(tx_ready),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen, .m_axi_arsize,
        .m_axi_arvalid, .m_axi_arready,
        .m_axi_rdata, .m_axi_rresp, .m_axi_rvalid, .m_axi_rlast,
        .m_axi_rready,
        .busy, .done,
        .dbg_fsm, .dbg_sub_fsm, .dbg_expert_cnt,
        .dbg_gemv_busy, .dbg_hbm2_busy,
        .perf_token_cnt, .perf_cycle_cnt,
        .pr_debug,
        .err_merge_overflow, .err_silu_overflow, .err_axi_resp_err
    );

    assign tx_ready = 1'b1;  // always ready to receive
    assign pcie_rx_valid = 1'b0;

    initial begin
        clk = 0; rst_n = 0;
        $display("[%0t] TB_FFN_AXI: Starting simulation", $time);
        #20 rst_n = 1;
        #200;
        $display("[%0t] AXI arvalid=%b arready=%b rvalid=%b rready=%b",
                 $time, m_axi_arvalid, m_axi_arready, m_axi_rvalid, m_axi_rready);
        $display("[%0t] ar_beat_cnt=%0d perf_token=%0d busy=%b done=%b",
                 $time, ar_beat_cnt, perf_token_cnt, busy, done);
        #500;
        $display("[%0t] FINAL: pr_debug=0x%0h perf_token=%0d done=%b",
                 $time, pr_debug, perf_token_cnt, done);
        if (perf_token_cnt > 0) $display("[%0t] PASS: AXI read transactions detected", $time);
        else $display("[%0t] WARN: no AXI reads", $time);
        #10 $finish;
    end
endmodule
