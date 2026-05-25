//=============================================================================
// top.sv — DK-SI-AGM027 board-level wrapper
//
// This is a SKELETON. Fill blocks marked [TODO] after QSYS/Platform Designer
// generates the HBM and PCIe IP instances.
//
// Phase 1 goal: connect fp4_mac + scale_reader to HBM → run golden vectors
//               through PCIe, capture DSP output via Signal Tap.
//=============================================================================

module top (
    // ── PCIe 5.0 (R-Tile) ──
    // [TODO] Replace with QSYS-generated PCIe port list
    // input  logic [15:0] pcie_rx_p, pcie_rx_n,
    // output logic [15:0] pcie_tx_p, pcie_tx_n,
    // input  logic        pcie_refclk_p, pcie_refclk_n,
    // input  logic        pcie_perst_n,

    // ── HBM2e (UIB) ──
    // [TODO] Replace with QSYS-generated HBM port list

    // ── QSFP-DD (F-Tile, Phase 2) ──
    // [TODO] Add when multi-card bring-up starts

    // ── Board control ──
    input  logic        clk_board_100m,  // 100 MHz oscillator
    input  logic        cpu_reset_n,     // active-low reset from board

    // ── Debug ──
    output logic [3:0]  debug_led,       // user LEDs
    output logic        uart_tx          // debug UART
);

    // ========================================================================
    // Clock & Reset
    // ========================================================================

    logic clk_sys, clk_dsp, clk_hbm;
    logic rst_n_sys;

    // [TODO] Instantiate PLL to generate:
    //   clk_sys  = 100 MHz (control plane)
    //   clk_dsp  = 450 MHz (DSP datapath)
    //   clk_hbm  = from HBM IP reference clock

    // Phase 1 workaround: use board clock directly for bring-up
    assign clk_sys  = clk_board_100m;
    assign clk_dsp  = clk_board_100m;  // slow for bring-up; use PLL later
    assign clk_hbm  = clk_board_100m;

    // Reset synchronizer (async assert, sync deassert)
    logic [2:0] rst_sr;
    always_ff @(posedge clk_sys or negedge cpu_reset_n) begin
        if (!cpu_reset_n) rst_sr <= '0;
        else              rst_sr <= {rst_sr[1:0], 1'b1};
    end
    assign rst_n_sys = rst_sr[2];

    // ========================================================================
    // LED heartbeat (1 Hz blink)
    // ========================================================================
    logic [26:0] led_cnt;
    always_ff @(posedge clk_sys or negedge rst_n_sys) begin
        if (!rst_n_sys) led_cnt <= '0;
        else            led_cnt <= led_cnt + 1'b1;
    end
    assign debug_led[0] = led_cnt[26];  // ~0.75 Hz at 100 MHz

    // ========================================================================
    // Experiment 1: fp4 MAC + Scale Reader + Golden Vector Checker
    // ========================================================================

    // ── Scale memory write interface ──
    logic        scale_wr_en;
    logic [8:0]  scale_wr_addr;   // up to 512 groups
    logic [7:0]  scale_wr_data;

    // ── MAC input interface ──
    logic        mac_valid_in;
    logic [3:0]  mac_weight;      // fp4 code
    logic [7:0]  mac_scale;       // fp8 scale
    logic [7:0]  mac_activ;       // fp8 activation

    // ── MAC output ──
    logic        mac_valid_out;
    logic [31:0] mac_result;

    // ── Scale reader query (connected to MAC's internal scale read) ──
    logic        sr_q_valid;
    logic [15:0] sr_q_elem;
    logic        sr_r_valid;
    logic [7:0]  sr_r_scale;

    // ── FP4 MAC instance ──
    fp4_mac #(.ACCUM_WIDTH(32), .VEC_LANES(1)) u_mac (
        .clk       (clk_dsp),
        .rst_n     (rst_n_sys),
        .accum_clr (mac_accum_clr),
        .mac_in    ('{weight: mac_weight, scale: mac_scale,
                      activ: mac_activ, valid: mac_valid_in}),
        .mac_out   ('{result: mac_result, valid: mac_valid_out})
    );

    // ── Scale reader instance ──
    fp4_scale_reader #(
        .NUM_GROUPS(512), .GROUP_SIZE(16), .ELEM_WIDTH(16), .SCALE_WIDTH(8)
    ) u_sr (
        .clk, .rst_n(rst_n_sys),
        .q_valid(sr_q_valid), .q_elem_idx(sr_q_elem), .q_ready(),
        .r_valid(sr_r_valid), .r_scale(sr_r_scale), .r_group_id(),
        .wr_en(scale_wr_en), .wr_addr(scale_wr_addr), .wr_data(scale_wr_data)
    );

    // ========================================================================
    // Experiment control FSM (via MMIO — placeholder)
    // ========================================================================
    logic mac_accum_clr;
    logic [3:0]  exp_state;
    logic [31:0] golden_expected;
    logic [31:0] golden_mismatch_count;

    // [TODO] Instantiate a simple AXI4-Lite → register bridge:
    //   - Write scale memory (512 writes)
    //   - Write MAC inputs (weight, activation, scale)
    //   - Pulse accum_clr
    //   - Read MAC output
    //   - Compare to golden expected
    //   - Light LED[1] on mismatch

    // Placeholder: tie to zero for compilation test
    assign mac_valid_in   = 1'b0;
    assign mac_weight     = 4'h0;
    assign mac_scale      = 8'h38;  // 1.0
    assign mac_activ      = 8'h38;
    assign mac_accum_clr  = 1'b0;
    assign sr_q_valid     = 1'b0;
    assign sr_q_elem      = 16'd0;
    assign scale_wr_en    = 1'b0;
    assign scale_wr_addr  = 9'd0;
    assign scale_wr_data  = 8'd0;

    // ========================================================================
    // Phase 2: Full Transformer Layer Pipeline
    // ========================================================================
    //
    // Once Phase 1 MAC validation passes, enable the full layer pipeline.
    // The full_transformer_layer integrates:
    //   RMSNorm → Attention (MLA) → RMSNorm → Router (MoE) → FFN → RMSNorm
    //
    // Phase 2a: Add mhc_mixer after each sub-layer
    // Phase 2b: Add engram_lookup before attention
    // Phase 2c: Add mtp_head on chip 31 (multi-token prediction)
    // Phase 2d: Add kv_dma_engine on chip 0 (KV cache offload)
    //
    // Control: set exp_state[2] to enable Phase 2 pipeline
    // ========================================================================

    // Layer I/O signals
    logic        layer_start;
    logic        layer_k_valid;
    logic        layer_k_last;
    logic [15:0] layer_elem_idx;
    logic [31:0] layer_wt_fp4;       // LANES*4 bits = 16
    logic [63:0] layer_act_fp8;      // LANES*8 bits = 32
    logic        layer_k_ready;
    logic        layer_busy;
    logic        layer_result_valid;
    logic        layer_result_ready;
    logic [31:0] layer_sum_result;
    logic [127:0] layer_lane_result;  // LANES*32 bits = 128

    // Router/Expert control (placeholder for full layer integration)
    logic [3:0]  router_top_k;
    logic        router_valid_in;
    logic        router_valid_out;
    logic [7:0]  router_expert_id;
    logic        ffn_start;
    logic        ffn_valid_in;
    logic [15:0] ffn_token_id;
    logic [7:0]  ffn_expert_id;
    logic        ffn_valid_out;
    logic [31:0] ffn_y [8];

    // Attention score/V preload (Phase 1.5: simplified attention)
    logic        attn_score_wr_en;
    logic [5:0]  attn_score_wr_idx;
    logic [31:0] attn_score_wr_data;
    logic        attn_v_wr_en;
    logic [5:0]  attn_v_wr_idx;
    logic [31:0] attn_v_wr_data;

    // [TODO] Uncomment after Phase 1 MAC validation passes:
    //
    // full_transformer_layer #(
    //     .LANES(4), .NUM_GROUPS(512), .GROUP_SIZE(16),
    //     .ELEM_WIDTH(16), .ACCUM_WIDTH(32), .DRAIN_CYCLES(8)
    // ) u_layer (
    //     .clk               (clk_dsp),
    //     .rst_n             (rst_n_sys),
    //     .start             (layer_start),
    //     .k_valid           (layer_k_valid),
    //     .k_last            (layer_k_last),
    //     .elem_idx_base     (layer_elem_idx),
    //     .weight_fp4_flat   (layer_wt_fp4),
    //     .activ_fp8_flat    (layer_act_fp8),
    //     .k_ready           (layer_k_ready),
    //     .scale_wr_en       (scale_wr_en),
    //     .scale_wr_addr     (scale_wr_addr),
    //     .scale_wr_data     (scale_wr_data),
    //     .busy              (layer_busy),
    //     .result_valid      (layer_result_valid),
    //     .result_ready      (layer_result_ready),
    //     .sum_result        (layer_sum_result),
    //     .lane_result_flat  (layer_lane_result)
    // );
    //
    // // When layer_result_valid && !golden_mismatch: LED[2] = 1
    // assign debug_led[2] = layer_result_valid;
    //
    // // On golden mismatch: LED[1] = 1, latch mismatch_count
    // always_ff @(posedge clk_dsp) begin
    //     if (layer_result_valid && layer_sum_result !== golden_expected)
    //         golden_mismatch_count <= golden_mismatch_count + 1'b1;
    // end

    // Phase 2 placeholder: tie off unused signals
    assign layer_start        = 1'b0;
    assign layer_k_valid      = 1'b0;
    assign layer_k_last       = 1'b0;
    assign layer_elem_idx     = 16'd0;
    assign layer_wt_fp4       = 32'd0;
    assign layer_act_fp8      = 64'd0;
    assign layer_result_ready = 1'b1;
    assign attn_score_wr_en   = 1'b0;
    assign attn_score_wr_idx  = 6'd0;
    assign attn_score_wr_data = 32'd0;
    assign attn_v_wr_en       = 1'b0;
    assign attn_v_wr_idx      = 6'd0;
    assign attn_v_wr_data     = 32'd0;
    assign router_top_k       = 4'd2;
    assign router_valid_in    = 1'b0;
    assign ffn_start          = 1'b0;
    assign ffn_valid_in       = 1'b0;
    assign ffn_token_id       = 16'd0;
    assign ffn_expert_id      = 8'd0;

    // ========================================================================
    // Experiment state display on LEDs
    // ========================================================================
    // LED[0]: heartbeat (1 Hz)
    // LED[1]: Phase 1 MAC golden mismatch (latched)
    // LED[2]: Phase 2 layer result valid
    // LED[3]: System error (any unexpected condition)
    assign debug_led[1] = (golden_mismatch_count > 0);
    assign debug_led[3] = 1'b0;  // [TODO] wire to error monitor

    // ========================================================================
    // UART debug output (115200 baud, 8N1)
    // ========================================================================
    // [TODO] Instantiate simple UART TX for printf-style debug
    assign uart_tx = 1'b1;  // idle high

endmodule
