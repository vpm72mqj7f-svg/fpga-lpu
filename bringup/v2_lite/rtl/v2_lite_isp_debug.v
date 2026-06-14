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

    // === FFN (production engine debug ports) ===
    // Legacy wrapper signals (self-test FSM)
    input  wire [3:0]  ffn_state,
    input  wire        ffn_busy,
    input  wire        ffn_done,
    input  wire        ffn_pass,
    input  wire [15:0] ffn_tdata_lo,
    input  wire [15:0] ffn_tdata_hi,
    input  wire        ffn_arvalid,
    input  wire        ffn_arready,
    // Production engine debug (direct from v2_lite_ffn_engine.sv)
    input  wire [3:0]  ffn_dbg_fsm,
    input  wire [2:0]  ffn_dbg_expert_cnt,
    input  wire        ffn_dbg_gate_done,
    input  wire        ffn_dbg_up_done,
    input  wire        ffn_dbg_down_done,
    input  wire        ffn_dbg_silu_active,
    input  wire        ffn_dbg_merge_active,
    input  wire        ffn_dbg_hbm2_busy,
    input  wire        ffn_dbg_sa_active,
    input  wire [2:0]  ffn_dbg_hbm2r_fsm,
    input  wire [2:0]  ffn_dbg_hbm2r_wr_wm,
    input  wire [2:0]  ffn_dbg_hbm2r_rd_wm,
    input  wire [31:0] ffn_perf_token,
    input  wire [31:0] ffn_perf_cycle,
    input  wire [31:0] ffn_perf_expert,
    input  wire [31:0] ffn_perf_axi_rbeat,
    input  wire        ffn_err_merge_ovf,
    input  wire        ffn_err_silu_ovf,
    input  wire        ffn_err_axi_resp
);

    // ========================================================================
    // PCIe Probe Wires
    // ========================================================================
    wire [31:0] pcie_probe0;  // LINK_STATUS
    wire [31:0] pcie_probe1;  // LANE_STATUS
    wire [31:0] pcie_probe2;  // VERSION (铁律 2: highest probe word = version)

    // Version: {day[7:0], month[7:0], year-2000[7:0], build[7:0]}
    localparam PCIE_VERSION = 32'h0E061A03;  // 2026-06-14 build 3
    localparam HBM2_VERSION = 32'h0E061A03;
    localparam FFN_VERSION  = 32'h0E061A03;

    assign pcie_probe0 = {12'd0,  1'b0 /*LINK_UP*/, 1'b0 /*PERSTN*/,
                          1'b0 /*CONFIG_DONE*/, pcie_atx_pll_locked,
                          6'd0 /*LINK_WIDTH*/, 5'd0 /*LINK_SPEED*/,
                          5'd0 /*LTSSM*/};
    assign pcie_probe1 = {16'd0 /*SIGNAL_DETECT*/, pcie_pll_locked_bank};
    assign pcie_probe2 = PCIE_VERSION;

    // ========================================================================
    // HBM2 Probe Wires
    // ========================================================================
    wire [31:0] hbm2_probe0;  // TG_STATUS
    wire [31:0] hbm2_probe1;  // TG_TIMEOUT
    wire [31:0] hbm2_probe2;  // STATUS
    wire [31:0] hbm2_probe3;  // VERSION (铁律 2: highest probe word = version)

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
    assign hbm2_probe3 = HBM2_VERSION;

    // ========================================================================
    // FFN Probe — 7 × 32-bit = 224-bit
    // ========================================================================
    wire [31:0] ffn_probe0;  // STATUS
    wire [31:0] ffn_probe1;  // PERF_LO
    wire [31:0] ffn_probe2;  // PERF_HI
    wire [31:0] ffn_probe3;  // AXI_STATS
    wire [31:0] ffn_probe4;  // SA_STATUS
    wire [31:0] ffn_probe5;  // ERRORS
    wire [31:0] ffn_probe6;  // HBM2R_STATUS
    wire [31:0] ffn_probe7;  // VERSION (铁律 2: highest probe word = version)

    // probe0 STATUS: FFN FSM + submodule status
    assign ffn_probe0 = {
        6'd0,                        // [31:26] Reserved
        ffn_dbg_down_done,           // [25]
        ffn_dbg_up_done,             // [24]
        ffn_dbg_gate_done,           // [23]
        ffn_dbg_hbm2r_rd_wm,        // [22:20]
        ffn_dbg_hbm2r_wr_wm,        // [19:17]
        ffn_dbg_hbm2r_fsm,          // [16:14]
        ffn_dbg_merge_active,        // [13]
        1'b0,                        // [12] Reserved
        ffn_busy,                    // [11]
        ffn_dbg_sa_active,           // [10]
        ffn_dbg_silu_active,         // [9]
        ffn_dbg_hbm2_busy,           // [8]
        ffn_dbg_expert_cnt,         // [7:5]
        ffn_dbg_fsm                  // [4:0]  (10-state FSM, only [3:0] used)
    };

    // probe1 PERF_LO: token + cycle low bits
    assign ffn_probe1 = ffn_perf_token;       // 32-bit token count

    // probe2 PERF_HI: expert + cycle mid bits
    assign ffn_probe2 = {ffn_perf_expert[15:0], ffn_perf_cycle[15:0]};

    // probe3 AXI_STATS: AXI read beat count + AR transaction count
    assign ffn_probe3 = ffn_perf_axi_rbeat;

    // probe4 SA_STATUS: systolic array FSM + cycle counters
    assign ffn_probe4 = {
        12'd0,                        // [31:20] Reserved for SA down debug
        8'd0,                         // [19:12] Reserved for SA gate debug
        4'd0,                         // [11:8]  Reserved (future: sa_down_fsm)
        4'd0,                         // [7:4]   Reserved (future: sa_gate_fsm)
        ffn_dbg_fsm                   // [3:0]   FFN main FSM
    };

    // probe5 ERRORS: sticky error flags
    assign ffn_probe5 = {
        29'd0,                        // [31:3] Reserved
        ffn_err_axi_resp,             // [2]
        ffn_err_silu_ovf,             // [1]
        ffn_err_merge_ovf             // [0]
    };

    // probe6 HBM2R_STATUS: HBM2 reader detailed status
    assign ffn_probe6 = {
        23'd0,                        // [31:9] Reserved
        ffn_dbg_hbm2r_rd_wm,         // [8:6]
        ffn_dbg_hbm2r_wr_wm,         // [5:3]
        ffn_dbg_hbm2r_fsm            // [2:0]
    };

    // probe7 VERSION
    assign ffn_probe7 = FFN_VERSION;

    // ========================================================================
    // SYS Probe + Source
    // ========================================================================
    wire [31:0] sys_probe0;  // STATUS
    wire [31:0] sys_source0; // CTRL (written by JTAG)

    localparam SYS_VERSION = 32'h0E061A03;  // 2026-06-14 build 3

    assign sys_probe0 = {SYS_VERSION[31:16], 8'd0 /*CLK_STATUS*/, 4'd0 /*RESET*/, led};

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
        .probe_width              (128),    // 4 × 32-bit
        .source_width             (0),
        .source_initial_value     ("0"),
        .enable_metastability     ("YES")
    ) u_hbm2_isp (
        .probe      ({hbm2_probe3, hbm2_probe2, hbm2_probe1, hbm2_probe0}),
        .source     (),
        .source_ena (1'b1),
        .clr        (1'b0)
    );

    // --- FFN ---
    altsource_probe #(
        .sld_auto_instance_index ("YES"),
        .instance_id              ("FFN"),
        .probe_width              (256),    // 8 × 32-bit
        .source_width             (0),
        .source_initial_value     ("0"),
        .enable_metastability     ("YES")
    ) u_ffn_isp (
        .probe      ({ffn_probe7, ffn_probe6, ffn_probe5, ffn_probe4,
                      ffn_probe3, ffn_probe2, ffn_probe1, ffn_probe0}),
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
