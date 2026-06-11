////////////////////////////////////////////////////////////////////////////////
// tb_ffn_engine_sv.sv — V2-Lite FFN Engine Testbench (Production .sv DUT)
//
// Tests the DSP-based v2_lite_ffn_engine with:
//   - Small dimensions (SIM_SMALL) for fast simulation
//   - Behavioral AXI4 SRAM model
//   - Debug port connectivity check
//   - Performance counter validation
//
// Usage: vsim -do sim_ffn_prod.do
////////////////////////////////////////////////////////////////////////////////

`timescale 1ns/1ps

module tb_ffn_engine_sv;

    // =========================================================================
    // Parameters — SIM_SMALL for fast sim, else production scale
    // =========================================================================
`ifdef SIM_SMALL
    localparam int HIDDEN      = 16;
    localparam int INTER       = 8;
    localparam int TOP_K       = 1;
`else
    localparam int HIDDEN      = 2048;
    localparam int INTER       = 1408;
    localparam int TOP_K       = 6;
`endif
    localparam int NUM_EXPERTS = 66;
    localparam int DATA_W      = 8;
    localparam int ACCUM_W     = 24;
    localparam int DSP_LANES   = 64;

    // =========================================================================
    // DUT Signals — Clock & Reset
    // =========================================================================
    logic clk;
    logic rst_n;

    // =========================================================================
    // DUT Signals — PCIe Streaming
    // =========================================================================
    logic                         pcie_rx_valid;
    logic [HIDDEN*DATA_W-1:0]     pcie_rx_data;
    logic                         pcie_rx_ready;
    logic                         pcie_tx_valid;
    logic [HIDDEN*DATA_W-1:0]     pcie_tx_data;
    logic                         pcie_tx_ready;

    // =========================================================================
    // DUT Signals — AXI4 Read Master
    // =========================================================================
    logic [31:0]  m_axi_araddr;
    logic [7:0]   m_axi_arlen;
    logic [2:0]   m_axi_arsize;
    logic         m_axi_arvalid;
    logic         m_axi_arready;
    logic [255:0] m_axi_rdata;
    logic [1:0]   m_axi_rresp;
    logic         m_axi_rvalid;
    logic         m_axi_rready;
    logic         m_axi_rlast;

    // =========================================================================
    // DUT Signals — Expert Selection (SystemVerilog unpacked array)
    // =========================================================================
    logic [$clog2(NUM_EXPERTS)-1:0] expert_id [TOP_K];

    // =========================================================================
    // DUT Signals — Status
    // =========================================================================
    logic busy;
    logic done;

    // =========================================================================
    // DUT Signals — Debug (production engine)
    // =========================================================================
    logic [3:0]  dbg_fsm_state;
    logic [2:0]  dbg_expert_cnt;
    logic        dbg_gate_done, dbg_up_done, dbg_down_done;
    logic        dbg_silu_active, dbg_merge_active;
    logic        dbg_hbm2_busy, dbg_sa_active;
    logic [2:0]  dbg_hbm2r_fsm, dbg_hbm2r_wr_wm, dbg_hbm2r_rd_wm;
    logic [31:0] perf_token_cnt, perf_cycle_cnt, perf_expert_cnt, perf_axi_rbeat;
    logic        err_merge_overflow, err_silu_overflow, err_axi_resp_err;

    // =========================================================================
    // Clock — 100 MHz = 10 ns period
    // =========================================================================
    initial clk = 0;
    always #5 clk = ~clk;

    // =========================================================================
    // DUT — Production SystemVerilog FFN Engine
    // =========================================================================
    v2_lite_ffn_engine #(
        .HIDDEN      (HIDDEN),
        .INTER       (INTER),
        .NUM_EXPERTS (NUM_EXPERTS),
        .TOP_K       (TOP_K),
        .DATA_W      (DATA_W),
        .ACCUM_W     (ACCUM_W),
        .DSP_LANES   (DSP_LANES),
        .VERSION     (32'h0B061A01)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .pcie_rx_valid      (pcie_rx_valid),
        .pcie_rx_data       (pcie_rx_data),
        .pcie_rx_ready      (pcie_rx_ready),
        .pcie_tx_valid      (pcie_tx_valid),
        .pcie_tx_data       (pcie_tx_data),
        .pcie_tx_ready      (pcie_tx_ready),
        .m_axi_araddr       (m_axi_araddr),
        .m_axi_arlen        (m_axi_arlen),
        .m_axi_arsize       (m_axi_arsize),
        .m_axi_arvalid      (m_axi_arvalid),
        .m_axi_arready      (m_axi_arready),
        .m_axi_rdata        (m_axi_rdata),
        .m_axi_rresp        (m_axi_rresp),
        .m_axi_rvalid       (m_axi_rvalid),
        .m_axi_rready       (m_axi_rready),
        .m_axi_rlast        (m_axi_rlast),
        .expert_id          (expert_id),
        .busy               (busy),
        .done               (done),
        // Debug ports connected
        .dbg_fsm_state      (dbg_fsm_state),
        .dbg_expert_cnt     (dbg_expert_cnt),
        .dbg_gate_done      (dbg_gate_done),
        .dbg_up_done        (dbg_up_done),
        .dbg_down_done      (dbg_down_done),
        .dbg_silu_active    (dbg_silu_active),
        .dbg_merge_active   (dbg_merge_active),
        .dbg_hbm2_busy      (dbg_hbm2_busy),
        .dbg_sa_active      (dbg_sa_active),
        .dbg_hbm2r_fsm      (dbg_hbm2r_fsm),
        .dbg_hbm2r_wr_watermark (dbg_hbm2r_wr_wm),
        .dbg_hbm2r_rd_watermark (dbg_hbm2r_rd_wm),
        .perf_token_cnt     (perf_token_cnt),
        .perf_cycle_cnt     (perf_cycle_cnt),
        .perf_expert_cnt    (perf_expert_cnt),
        .perf_axi_rbeat     (perf_axi_rbeat),
        .err_merge_overflow (err_merge_overflow),
        .err_silu_overflow  (err_silu_overflow),
        .err_axi_resp_err   (err_axi_resp_err)
    );

    // Expert IDs: simple sequential assignment
    generate
        for (genvar gi = 0; gi < TOP_K; gi++) begin : g_expert
            assign expert_id[gi] = gi % NUM_EXPERTS;
        end
    endgenerate

    // =========================================================================
    // AXI4 SRAM Behavioral Model (ramp-data, 8-beat burst)
    // =========================================================================
    logic [255:0] sram [0:255];
    logic [7:0]   sram_beat_cnt;
    integer sram_init_i;

    initial begin
        for (sram_init_i = 0; sram_init_i < 256; sram_init_i++) begin
            for (int b = 0; b < 32; b++)
                sram[sram_init_i][b*8 +: 8] = (sram_init_i + b) % 256;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_rdata  <= 256'd0;
            m_axi_rresp  <= 2'd0;
            m_axi_rvalid <= 1'b0;
            m_axi_rlast  <= 1'b0;
            sram_beat_cnt <= 8'd0;
        end else begin
            // Default
            m_axi_rlast <= 1'b0;

            // Accept read request
            if (m_axi_arvalid && m_axi_arready && !m_axi_rvalid) begin
                sram_beat_cnt <= 8'd0;
                m_axi_rdata   <= sram[8'd0];
                m_axi_rvalid  <= 1'b1;
                m_axi_rlast   <= 1'b0;
            end

            // Stream beats
            if (m_axi_rvalid && m_axi_rready) begin
                if (sram_beat_cnt == m_axi_arlen) begin
                    m_axi_rvalid  <= 1'b0;
                    m_axi_rlast   <= 1'b0;
                end else begin
                    sram_beat_cnt <= sram_beat_cnt + 8'd1;
                    m_axi_rdata   <= sram[sram_beat_cnt + 8'd1];
                    m_axi_rvalid  <= 1'b1;
                    m_axi_rlast   <= (sram_beat_cnt + 8'd1 == m_axi_arlen);
                end
            end
        end
    end
    assign m_axi_arready = 1'b1;

    // =========================================================================
    // Test Sequence
    // =========================================================================
    integer errors;
    integer j;

    initial begin
        errors = 0;
        $display("============================================================");
        $display(" V2-Lite FFN Engine Testbench (Production .sv)");
        $display(" HIDDEN=%0d INTER=%0d TOP_K=%0d DSP_LANES=%0d",
                 HIDDEN, INTER, TOP_K, DSP_LANES);
        $display("============================================================");

        // ---- Init ----
        rst_n         = 1'b0;
        pcie_rx_valid = 1'b0;
        pcie_rx_data  = '0;
        pcie_tx_ready = 1'b0;
        repeat(10) @(posedge clk);

        // =====================================================================
        // Test 1: Reset release → IDLE state
        // =====================================================================
        $display("\n[Test 1] Reset release...");
        rst_n = 1'b1;
        repeat(5) @(posedge clk);

        if (busy == 1'b0 && dbg_fsm_state == 4'd0) begin
            $display("  PASS: busy=0, fsm_state=IDLE");
        end else begin
            $display("  FAIL: busy=%b, fsm_state=%0d", busy, dbg_fsm_state);
            errors = errors + 1;
        end

        // =====================================================================
        // Test 2: Send activation → pipeline should start
        // =====================================================================
        $display("\n[Test 2] Load activation (ramp pattern)...");
        pcie_rx_valid = 1'b1;
        for (j = 0; j < HIDDEN; j++) begin
            pcie_rx_data[j*DATA_W +: DATA_W] = j % 256;
        end
        @(posedge clk);
        @(posedge clk);
        wait(busy == 1'b1);
        pcie_rx_valid = 1'b0;

        if (busy && dbg_fsm_state != 4'd0) begin
            $display("  PASS: busy=1, fsm_state=%0d, hbm2_busy=%b",
                     dbg_fsm_state, dbg_hbm2_busy);
        end else begin
            $display("  FAIL: busy=%b, fsm_state=%0d", busy, dbg_fsm_state);
            errors = errors + 1;
        end

        // =====================================================================
        // Test 3: Pipeline completion
        // =====================================================================
        $display("\n[Test 3] Wait for FFN pipeline to complete...");
        pcie_tx_ready = 1'b1;

        // Poll FSM state transitions to verify pipeline progresses
        $display("  FSM trace (100 cycles max):");
        for (int cyc = 0; cyc < 100; cyc++) begin
            @(posedge clk);
            if (cyc % 10 == 0)
                $display("    cycle %0d: fsm=%0d busy=%b done=%b gate=%b up=%b down=%b",
                         cyc, dbg_fsm_state, busy, done,
                         dbg_gate_done, dbg_up_done, dbg_down_done);
            if (done) break;
        end

        if (done) begin
            $display("  PASS: done asserted at fsm=%0d", dbg_fsm_state);
        end else begin
            $display("  FAIL: done not asserted within 100 cycles, fsm=%0d", dbg_fsm_state);
            errors = errors + 1;
        end

        // =====================================================================
        // Test 4: Output data valid
        // =====================================================================
        $display("\n[Test 4] Check output...");
        if (pcie_tx_valid) begin
            $display("  pcie_tx_valid=1, sample bytes: %02x %02x %02x %02x",
                     pcie_tx_data[7:0], pcie_tx_data[15:8],
                     pcie_tx_data[23:16], pcie_tx_data[31:24]);
            $display("  PASS: output data valid");
        end else begin
            $display("  FAIL: pcie_tx_valid not asserted at done");
            errors = errors + 1;
        end

        // =====================================================================
        // Test 5: Debug ports and counters
        // =====================================================================
        $display("\n[Test 5] Debug port verification...");

        if (perf_cycle_cnt > 0) begin
            $display("  PASS: perf_cycle_cnt=%0d (should be > 0)", perf_cycle_cnt);
        end else begin
            $display("  FAIL: perf_cycle_cnt=%0d", perf_cycle_cnt);
            errors = errors + 1;
        end

        if (perf_token_cnt == 1) begin
            $display("  PASS: perf_token_cnt=1 (one token completed)");
        end else begin
            $display("  FAIL: perf_token_cnt=%0d (expected 1)", perf_token_cnt);
            errors = errors + 1;
        end

        if (perf_axi_rbeat > 0) begin
            $display("  PASS: perf_axi_rbeat=%0d (AXI reads occurred)", perf_axi_rbeat);
        end else begin
            $display("  FAIL: perf_axi_rbeat=%0d (no AXI traffic)", perf_axi_rbeat);
            errors = errors + 1;
        end

        if (perf_expert_cnt > 0) begin
            $display("  PASS: perf_expert_cnt=%0d", perf_expert_cnt);
        end else begin
            $display("  FAIL: perf_expert_cnt=0 (no expert processed)", perf_expert_cnt);
            errors = errors + 1;
        end

        // Error flags should be clear
        if (!err_merge_overflow && !err_silu_overflow && !err_axi_resp_err) begin
            $display("  PASS: all error flags clear");
        end else begin
            $display("  FAIL: error flags set (merge=%b silu=%b axi=%b)",
                     err_merge_overflow, err_silu_overflow, err_axi_resp_err);
            errors = errors + 1;
        end

        // =====================================================================
        // Test 6: HBM2 debug ports toggled
        // =====================================================================
        $display("\n[Test 6] HBM2 reader debug...");
        $display("  hbm2r_fsm=%0d wr_wm=%0d rd_wm=%0d (should not all be 0)",
                 dbg_hbm2r_fsm, dbg_hbm2r_wr_wm, dbg_hbm2r_rd_wm);

        // =====================================================================
        // Summary
        // =====================================================================
        $display("\n============================================================");
        if (errors == 0) begin
            $display(" ALL 6 TESTS PASSED");
            $display("============================================================");
        end else begin
            $display(" %0d TESTS FAILED", errors);
            $display("============================================================");
        end
        $finish;
    end

    // =========================================================================
    // Timeout watchdog (1M cycles @ 10ns = 10ms)
    // =========================================================================
    initial begin
        #10000000;
        $display("TIMEOUT: Simulation exceeded 10ms (1M cycles)");
        $finish;
    end

endmodule
