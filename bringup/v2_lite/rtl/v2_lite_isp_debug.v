// v2_lite_isp_debug.v — Multi-Instance ISP Debug Register Map
// 4 ISP instances: PCIE, HBM2, FFN, SYS
// Register map: see ISP_DEBUG_REGISTER_MAP.md
// CLI: quartus_issp --probe --instance=<ID>

module v2_lite_isp_debug #(
    parameter HEARTBEAT_WIDTH = 27  // ~1.3s at 100MHz
) (
    input  wire        clk,
    input  wire        rst_n,

    // === System ===
    input  wire [3:0]  led,

    // === PCIe ===
    input  wire        pcie_atx_pll_locked,
    input  wire [15:0] pcie_pll_locked_bank,

    // === HBM2 ===
    input  wire        tg0_0_pass,  tg0_0_fail,  tg0_0_timeout,
    input  wire        tg0_1_pass,  tg0_1_fail,  tg0_1_timeout,
    input  wire        tg1_0_pass,  tg1_0_fail,  tg1_0_timeout,
    input  wire        tg1_1_pass,  tg1_1_fail,  tg1_1_timeout,
    input  wire        tg2_0_pass,  tg2_0_fail,  tg2_0_timeout,
    input  wire        tg2_1_pass,  tg2_1_fail,  tg2_1_timeout,
    input  wire        tg3_0_pass,  tg3_0_fail,  tg3_0_timeout,
    input  wire        tg3_1_pass,  tg3_1_fail,  tg3_1_timeout,
    input  wire        tg4_0_pass,  tg4_0_fail,  tg4_0_timeout,
    input  wire        tg4_1_pass,  tg4_1_fail,  tg4_1_timeout,
    input  wire        tg5_0_pass,  tg5_0_fail,  tg5_0_timeout,
    input  wire        tg5_1_pass,  tg5_1_fail,  tg5_1_timeout,
    input  wire        tg6_0_pass,  tg6_0_fail,  tg6_0_timeout,
    input  wire        tg6_1_pass,  tg6_1_fail,  tg6_1_timeout,
    input  wire        tg7_0_pass,  tg7_0_fail,  tg7_0_timeout,
    input  wire        tg7_1_pass,  tg7_1_fail,  tg7_1_timeout,
    // hbm_temp & hbm_cattrip removed — dedicated IO buffer, can't fan out
    // TODO: read via HBM2 Qsys internal register access

    // === FFN ===
    input  wire [3:0]  ffn_state,
    input  wire        ffn_busy,
    input  wire        ffn_done,
    input  wire        ffn_pass,
    input  wire [15:0] ffn_tdata_lo,
    input  wire [15:0] ffn_tdata_hi,
    input  wire        ffn_arvalid,
    input  wire        ffn_arready
);

    // ========================================================================
    // PCIe Probe Wires
    // ========================================================================
    wire [31:0] pcie_probe0;  // LINK_STATUS
    wire [31:0] pcie_probe1;  // LANE_STATUS
    wire [31:0] pcie_probe2;  // ERROR_COUNTERS (TODO: real counters)

    assign pcie_probe0 = {12'd0,  1'b0 /*LINK_UP*/, 1'b0 /*PERSTN*/,
                          1'b0 /*CONFIG_DONE*/, pcie_atx_pll_locked,
                          6'd0 /*LINK_WIDTH*/, 5'd0 /*LINK_SPEED*/,
                          5'd0 /*LTSSM*/};
    assign pcie_probe1 = {16'd0 /*SIGNAL_DETECT*/, pcie_pll_locked_bank};
    assign pcie_probe2 = 32'd0;  // TODO: DL/TL error counters

    // ========================================================================
    // HBM2 Probe Wires
    // ========================================================================
    wire [31:0] hbm2_probe0;  // TG_STATUS
    wire [31:0] hbm2_probe1;  // TG_TIMEOUT
    wire [31:0] hbm2_probe2;  // STATUS

    wire [15:0] tg_pass   = {tg7_1_pass, tg7_0_pass, tg6_1_pass, tg6_0_pass,
                             tg5_1_pass, tg5_0_pass, tg4_1_pass, tg4_0_pass,
                             tg3_1_pass, tg3_0_pass, tg2_1_pass, tg2_0_pass,
                             tg1_1_pass, tg1_0_pass, tg0_1_pass, tg0_0_pass};
    wire [15:0] tg_fail   = {tg7_1_fail, tg7_0_fail, tg6_1_fail, tg6_0_fail,
                             tg5_1_fail, tg5_0_fail, tg4_1_fail, tg4_0_fail,
                             tg3_1_fail, tg3_0_fail, tg2_1_fail, tg2_0_fail,
                             tg1_1_fail, tg1_0_fail, tg0_1_fail, tg0_0_fail};
    wire [15:0] tg_timeout = {tg7_1_timeout, tg7_0_timeout, tg6_1_timeout, tg6_0_timeout,
                              tg5_1_timeout, tg5_0_timeout, tg4_1_timeout, tg4_0_timeout,
                              tg3_1_timeout, tg3_0_timeout, tg2_1_timeout, tg2_0_timeout,
                              tg1_1_timeout, tg1_0_timeout, tg0_1_timeout, tg0_0_timeout};
    wire        hbm_pll_locked = pcie_atx_pll_locked;  // HBM2 uses same core clock domain
    wire        ch_active = |tg_pass;

    assign hbm2_probe0 = {tg_fail, tg_pass};
    assign hbm2_probe1 = {16'd0, tg_timeout};
    assign hbm2_probe2 = {29'd0, ch_active, hbm_pll_locked, 1'b0 /*CATTRIP*/, 3'd0 /*TEMP*/};

    // ========================================================================
    // FFN Counters (performance measurement)
    // ========================================================================
    reg [15:0] ffn_token_cnt;
    reg [31:0] ffn_cycle_cnt;
    reg [15:0] ffn_ar_trans_cnt;
    reg [15:0] ffn_r_beat_cnt;

    wire ffn_done_pulse = ffn_done && !ffn_busy;  // one-shot on done

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ffn_token_cnt   <= 16'd0;
            ffn_cycle_cnt   <= 32'd0;
            ffn_ar_trans_cnt <= 16'd0;
            ffn_r_beat_cnt  <= 16'd0;
        end else begin
            ffn_cycle_cnt <= ffn_cycle_cnt + 32'd1;
            if (ffn_done_pulse)
                ffn_token_cnt <= ffn_token_cnt + 16'd1;
            if (ffn_arvalid && ffn_arready)
                ffn_ar_trans_cnt <= ffn_ar_trans_cnt + 16'd1;
            // R beat counting needs rvalid & rready — not at top level yet
            // TODO: expose FFN AXI read channel to top
        end
    end

    wire [31:0] ffn_probe0;  // STATUS
    wire [31:0] ffn_probe1;  // PERF
    wire [31:0] ffn_probe2;  // AXI_STATS
    wire [31:0] ffn_probe3;  // DATA

    assign ffn_probe0 = {16'd0, 8'd0 /*ERROR_CODE*/, ffn_pass, ffn_done, ffn_busy, ffn_pass, ffn_state};
    assign ffn_probe1 = {ffn_cycle_cnt[15:0], ffn_token_cnt};
    assign ffn_probe2 = {ffn_r_beat_cnt, ffn_ar_trans_cnt};
    assign ffn_probe3 = {ffn_tdata_hi, ffn_tdata_lo};

    // ========================================================================
    // SYS Probe + Source
    // ========================================================================
    wire [31:0] sys_probe0;  // STATUS
    wire [31:0] sys_source0; // CTRL (written by JTAG)

    assign sys_probe0 = {16'h0001 /*VERSION 1.0*/, 8'd0 /*CLK_STATUS*/, 4'd0 /*RESET*/, led};

    // SYS source: [0]=FFN_START, [1]=FFN_RESET, [2]=COUNTER_RESET
    // (source is read internally for debug; actual control logic TBD)

    // ========================================================================
    // ISP Instances
    // ========================================================================

    // --- PCIE ---
    altsource_probe #(
        .sld_auto_instance_index ("YES"),
        .instance_id              ("PCIE"),
        .probe_width              (96),     // 3 × 32-bit
        .source_width             (0),
        .source_initial_value     ("0"),
        .enable_metastability     ("YES")
    ) u_pcie_isp (
        .probe      ({pcie_probe2, pcie_probe1, pcie_probe0}),
        .source     (),
        .source_ena (1'b1),
        .clr        (1'b0)
    );

    // --- HBM2 ---
    altsource_probe #(
        .sld_auto_instance_index ("YES"),
        .instance_id              ("HBM2"),
        .probe_width              (96),     // 3 × 32-bit
        .source_width             (0),
        .source_initial_value     ("0"),
        .enable_metastability     ("YES")
    ) u_hbm2_isp (
        .probe      ({hbm2_probe2, hbm2_probe1, hbm2_probe0}),
        .source     (),
        .source_ena (1'b1),
        .clr        (1'b0)
    );

    // --- FFN ---
    altsource_probe #(
        .sld_auto_instance_index ("YES"),
        .instance_id              ("FFN"),
        .probe_width              (128),    // 4 × 32-bit
        .source_width             (0),
        .source_initial_value     ("0"),
        .enable_metastability     ("YES")
    ) u_ffn_isp (
        .probe      ({ffn_probe3, ffn_probe2, ffn_probe1, ffn_probe0}),
        .source     (),
        .source_ena (1'b1),
        .clr        (1'b0)
    );

    // --- SYS ---
    altsource_probe #(
        .sld_auto_instance_index ("YES"),
        .instance_id              ("SYS"),
        .probe_width              (32),     // 1 × 32-bit
        .source_width             (32),     // 1 × 32-bit
        .source_initial_value     ("0"),
        .enable_metastability     ("YES")
    ) u_sys_isp (
        .probe      (sys_probe0),
        .source     (sys_source0),
        .source_ena (1'b1),
        .clr        (1'b0)
    );

endmodule
