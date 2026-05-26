//=============================================================================
// top_master.sv — Master FPGA (PCIe Host Interface)
//
// One of 8 Master chips in the 32-chip cluster. Each of the 8 PCIe cards has
// exactly 1 Master (Chip 0) + 3 Slaves (Chips 1-3), connected via C2C ring.
//
// 32 chips × 12 layers = 384 layers total.
// 8 Masters interconnect via PCIe 5.0 backplane to dual-socket host.
//
// Code uniformity: uses the SAME chip_top.sv RTL as Slave, with
// IS_PCIE_MASTER=1. The synthesis tool gates PCIe logic when parameter is 0.
//=============================================================================

module top_master (
    // ── PCIe 5.0 R-Tile (Host Interface) ──
    // [TODO] Replace with QSYS-generated PCIe port list
    // input  logic [15:0] pcie_rx_p, pcie_rx_n,
    // output logic [15:0] pcie_tx_p, pcie_tx_n,
    // input  logic        pcie_refclk_p, pcie_refclk_n,
    // input  logic        pcie_perst_n,

    // ── HBM2e UIB ──
    // [TODO] QSYS-generated HBM port list

    // ── C2C (Chip-to-Chip) Dual Ring ──
    // [TODO] Map to F-Tile transceiver ports
    // output logic [3:0] c2c_tx_a_p, c2c_tx_a_n,
    // input  logic [3:0] c2c_rx_a_p, c2c_rx_a_n,
    // output logic [3:0] c2c_tx_b_p, c2c_tx_b_n,
    // input  logic [3:0] c2c_rx_b_p, c2c_rx_b_n,

    // ── QSFP-DD (F-Tile, Phase 2) ──
    // [TODO] Multi-card inter-board links

    // ── Board Control ──
    input  logic        clk_board_100m,
    input  logic        cpu_reset_n,

    // ── Debug ──
    output logic [3:0]  debug_led,
    output logic        uart_tx
);

    // ========================================================================
    // Clock & Reset
    // ========================================================================
    logic clk_sys, clk_dsp, clk_pcie, clk_hbm;
    logic rst_n_sys;

    // [TODO] PLL: 100 MHz board clock → multiple domains
    //   clk_dsp  = 450 MHz (DSP systolic array, same as HBM)
    //   clk_pcie = 250 MHz (PCIe 5.0 reference)
    //   clk_hbm  = 450 MHz (HBM2e controller)
    //   clk_sys  = 100 MHz (control plane)
    //
    // PLL ratios: DSP = 100 × 9/2 = 450, PCIe = 100 × 5/2 = 250

    // Phase 1 workaround
    assign clk_sys  = clk_board_100m;
    assign clk_dsp  = clk_board_100m;
    assign clk_pcie = clk_board_100m;
    assign clk_hbm  = clk_board_100m;

    // Reset synchronizer
    logic [2:0] rst_sr;
    always_ff @(posedge clk_sys or negedge cpu_reset_n) begin
        if (!cpu_reset_n) rst_sr <= '0;
        else              rst_sr <= {rst_sr[1:0], 1'b1};
    end
    assign rst_n_sys = rst_sr[2];

    // ========================================================================
    // LED heartbeat
    // ========================================================================
    logic [26:0] led_cnt;
    always_ff @(posedge clk_sys or negedge rst_n_sys) begin
        if (!rst_n_sys) led_cnt <= '0;
        else            led_cnt <= led_cnt + 1'b1;
    end
    assign debug_led[0] = led_cnt[26];

    // ========================================================================
    // PCIe DMA Engine (Host ↔ FPGA)
    // ========================================================================
    // [TODO] Instantiate QSYS-generated R-Tile PCIe 5.0 IP
    // [TODO] Instantiate AXI4-Lite → register bridge for BAR0 MMIO
    // [TODO] Instantiate DMA descriptor engine

    // BAR0 Register placeholder signals
    logic [31:0] bar0_ctrl;
    logic [31:0] bar0_status;
    logic [63:0] bar0_desc_ring_base;
    logic [15:0] bar0_desc_head, bar0_desc_tail;

    // DMA stream placeholder
    logic        dma_h2f_valid, dma_h2f_ready;
    logic [31:0] dma_h2f_data;
    logic        dma_f2h_valid, dma_f2h_ready;
    logic [31:0] dma_f2h_data;

    // ========================================================================
    // KV DMA Engine (Host SSD → HBM KV cache blocks)
    // ========================================================================
    logic        kv_desc_valid, kv_desc_ready;
    logic [63:0] kv_desc_host_addr;
    logic [31:0] kv_desc_hbm_addr, kv_desc_length;
    logic [15:0] kv_desc_session_id;
    logic        kv_dma_req_valid, kv_dma_req_ready;
    logic [63:0] kv_dma_req_addr;
    logic [31:0] kv_dma_req_length;
    logic        kv_dma_rsp_valid, kv_dma_rsp_last;
    logic [255:0] kv_dma_rsp_data;
    logic [31:0] kv_hbm_wr_addr, kv_hbm_wr_data;
    logic        kv_hbm_wr_en;
    logic        kv_done;
    logic [15:0] kv_session_id;
    logic [31:0] kv_bytes_xfer;

    kv_dma_engine #(.BEAT_BYTES(32)) u_kv_dma (
        .clk              (clk_sys),
        .rst_n            (rst_n_sys),
        .desc_valid       (kv_desc_valid),
        .desc_ready       (kv_desc_ready),
        .desc_host_addr   (kv_desc_host_addr),
        .desc_hbm_addr    (kv_desc_hbm_addr),
        .desc_length      (kv_desc_length),
        .desc_session_id  (kv_desc_session_id),
        .dma_req_valid    (kv_dma_req_valid),
        .dma_req_ready    (kv_dma_req_ready),
        .dma_req_addr     (kv_dma_req_addr),
        .dma_req_length   (kv_dma_req_length),
        .dma_rsp_valid    (kv_dma_rsp_valid),
        .dma_rsp_data     (kv_dma_rsp_data),
        .dma_rsp_last     (kv_dma_rsp_last),
        .hbm_wr_addr      (kv_hbm_wr_addr),
        .hbm_wr_data      (kv_hbm_wr_data),
        .hbm_wr_en        (kv_hbm_wr_en),
        .done             (kv_done),
        .session_id       (kv_session_id),
        .bytes_transferred(kv_bytes_xfer)
    );

    // ========================================================================
    // Chip Core (Transformer Pipeline) — same RTL as Slave
    //
    // 384 layers / 32 chips = 12 layers per chip.
    // IS_PCIE_MASTER=1 enables PCIe DMA + KV offload + C2C ring origin.
    // ========================================================================
    // [TODO] Uncomment after PCIe and HBM IP are integrated:
    //
    // chip_top #(
    //     .CHIP_ID(dip_chip_id), .CARD_ID(0),
    //     .LAYER_START(dip_chip_id * 12),
    //     .LAYER_END(dip_chip_id * 12 + 11),
    //     .IS_PCIE_MASTER(1)
    // ) u_chip (
    //     .clk, .rst_n(rst_n_sys),
    //     .c2c_rx_a(c2c_rx_a_int), .c2c_tx_a(c2c_tx_a_int),
    //     .c2c_rx_b(c2c_rx_b_int), .c2c_tx_b(c2c_tx_b_int),
    //     .pcie_host(pcie_host_stream), .pcie_fpga(pcie_fpga_stream),
    //     .c2c_proxy(c2c_proxy_int)
    // );

    // ========================================================================
    // LED Status
    // ========================================================================
    // LED[0]: heartbeat
    // LED[1]: KV DMA active (kv_dma_req_valid)
    // LED[2]: PCIe link up
    // LED[3]: System error
    assign debug_led[1] = kv_dma_req_valid;
    assign debug_led[2] = 1'b0;  // [TODO] PCIe link status
    assign debug_led[3] = 1'b0;  // [TODO] Error monitor

    // ========================================================================
    // UART Debug
    // ========================================================================
    // [TODO] UART TX
    assign uart_tx = 1'b1;

endmodule
