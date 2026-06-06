//=============================================================================
// tb_axi4_hbm_bw_bench.sv — HBM Bandwidth Benchmark with Pipelined Reads
//
// Measures actual RTL-simulated HBM bandwidth, replacing the circular 920×0.87.
//
// Test patterns:
//   P1: Sequential write (max write BW)
//   P2: Sequential read with pipelined AR (max read BW)
//   P3: Random 33 MB reads (MoE expert loading)
//
// Single pseudo-channel at 450 MHz: peak = 14,400 MB/s
//=============================================================================
`timescale 1ns / 1ps

module tb_axi4_hbm_bw_bench;

    localparam int AW = 256;
    localparam int AA = 32;
    localparam int AI = 4;
    localparam int MEM_KB = 256;           // 256 KB HBM model
    localparam int TEST_KB = 128;          // 128 KB test pattern
    localparam int BURST_BEATS = 256;           // AXI4 max burst length
    localparam int BYTES_PER_BURST = 256 * 32;  // 8192 bytes per burst
    localparam int TOTAL_BURSTS = (TEST_KB * 1024) / (BURST_BEATS * 32);

    logic clk, rst_n;

    // AW
    logic [AI-1:0]   awid;
    logic [AA-1:0]   awaddr;
    logic [7:0]      awlen;
    logic [2:0]      awsize;
    logic [1:0]      awburst;
    logic            awvalid, awready;

    // W
    logic [AW-1:0]   wdata;
    logic [AW/8-1:0] wstrb;
    logic            wlast;
    logic            wvalid, wready;

    // B
    logic [AI-1:0]  bid;
    logic [1:0]     bresp;
    logic           bvalid, bready;

    // AR
    logic [AI-1:0]  arid;
    logic [AA-1:0]  araddr;
    logic [7:0]     arlen;
    logic [2:0]     arsize;
    logic [1:0]     arburst;
    logic           arvalid, arready;

    // R
    logic [AI-1:0]  rid;
    logic [AW-1:0]  rdata;
    logic [1:0]     rresp;
    logic           rlast;
    logic           rvalid, rready;

    logic [63:0] wb, rb;
    logic [31:0] wbw, rbw;

    // HBM Model
    sim_axi4_hbm_model #(
        .MEM_SIZE_BYTES(MEM_KB * 1024),
        .READ_LATENCY(20),
        .WRITE_LATENCY(14),
        .BW_WINDOW_CYCLES(1000)
    ) u_hbm (
        .clk, .rst_n,
        .s_axi_awid(awid), .s_axi_awaddr(awaddr), .s_axi_awlen(awlen),
        .s_axi_awsize(awsize), .s_axi_awburst(awburst),
        .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wlast(wlast),
        .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bid(bid), .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_arid(arid), .s_axi_araddr(araddr), .s_axi_arlen(arlen),
        .s_axi_arsize(arsize), .s_axi_arburst(arburst),
        .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rid(rid), .s_axi_rdata(rdata), .s_axi_rresp(rresp),
        .s_axi_rlast(rlast), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .write_bytes_total(wb), .read_bytes_total(rb),
        .write_bw_mbps(wbw), .read_bw_mbps(rbw)
    );

    always #1.111 clk = ~clk;  // 450 MHz

    integer i, b;
    integer start_cycle, end_cycle, elapsed;
    integer total_data_bytes;
    real    bw_mbps;

    task write_bursts;
        input integer num_bursts;
        integer b_idx, beat;
        begin
            for (b_idx = 0; b_idx < num_bursts; b_idx++) begin
                @(posedge clk);
                awvalid <= 1;
                awaddr  <= b_idx * BYTES_PER_BURST;
                awlen   <= BURST_BEATS - 1;
                while (!(awvalid && awready)) @(posedge clk);
                awvalid <= 0;

                wvalid <= 1;
                for (beat = 0; beat < BURST_BEATS; beat++) begin
                    wlast <= (beat == BURST_BEATS - 1);
                    wdata <= {192'd0, 32'(32'hDEAD_BEEF + b_idx * 16 + beat)};
                    @(posedge clk);
                    while (!(wvalid && wready)) @(posedge clk);
                end
                wvalid <= 0;
                wlast  <= 0;
            end
        end
    endtask

    task read_bursts_pipelined;
        input integer num_bursts;
        integer b_idx, beat;
        integer pending_bursts;
        begin
            pending_bursts = 0;
            b_idx = 0;
            rready <= 1;

            // Pipeline: keep up to 2 AR ahead of data
            while (b_idx < num_bursts || pending_bursts > 0) begin
                // Issue AR if we have room in the pipeline
                if (b_idx < num_bursts && pending_bursts < 2) begin
                    @(posedge clk);
                    arvalid <= 1;
                    araddr  <= b_idx * BYTES_PER_BURST;
                    arlen   <= BURST_BEATS - 1;
                    while (!(arvalid && arready)) @(posedge clk);
                    arvalid <= 0;
                    b_idx = b_idx + 1;
                    pending_bursts = pending_bursts + 1;
                end else begin
                    @(posedge clk);
                end

                // Count completed bursts
                if (rvalid && rready && rlast) begin
                    pending_bursts = pending_bursts - 1;
                end
            end

            // Drain remaining data
            while (rvalid != 1 || pending_bursts > 0) begin
                @(posedge clk);
                if (rvalid && rready && rlast) begin
                    pending_bursts = pending_bursts - 1;
                end
            end

            rready <= 0;
        end
    endtask

    initial begin
        clk = 0; rst_n = 0;
        awvalid = 0; wvalid = 0; bready = 1;
        arvalid = 0; rready = 0;
        awid = 0; arsize = 5; arburst = 1;
        arid = 0; arsize = 5; arburst = 1;
        wstrb = '1;
        awsize = 5; awburst = 1;

        #10 rst_n = 1;
        #10;

        $display("═══════════════════════════════════════════════════════");
        $display("  HBM Bandwidth Benchmark — Pipelined AXI4");
        $display("═══════════════════════════════════════════════════════");
        $display("  HBM model: %0d KB, RD_LAT=20, WR_LAT=14", MEM_KB);
        $display("  Test size: %0d KB, %0d bursts of %0d beats",
                 TEST_KB, TOTAL_BURSTS, BURST_BEATS);
        $display("  Per-channel peak @ 450 MHz: 14,400 MB/s");
        $display();

        // ── P1: Sequential Write ──
        $display("── P1: Sequential Write (%0d bursts) ──", TOTAL_BURSTS);
        start_cycle = u_hbm.bw_cycle_counter;
        write_bursts(TOTAL_BURSTS);
        end_cycle = u_hbm.bw_cycle_counter;
        $display("  Write bytes: %0d, HBM write BW: %0d MB/s", wb, wbw);
        $display("  Write efficiency: %.1f%%", 100.0 * wbw / 14400.0);
        $display();

        // ── P2: Sequential Read with Pipelined AR ──
        $display("── P2: Sequential Read with Pipelined AR ──");
        @(posedge clk); @(posedge clk); @(posedge clk);
        total_data_bytes = TOTAL_BURSTS * BURST_BEATS * 32;
        elapsed = 0;
        start_cycle = u_hbm.bw_cycle_counter;
        read_bursts_pipelined(TOTAL_BURSTS);
        end_cycle = u_hbm.bw_cycle_counter;
        elapsed = end_cycle - start_cycle;
        $display("  Read bytes: %0d, HBM read BW: %0d MB/s", rb, rbw);
        if (elapsed > 0) begin
            bw_mbps = (total_data_bytes * 450.0) / elapsed;
            $display("  Read cycles: %0d, BW = %0d MB/s (%.1f%% peak)",
                     elapsed, bw_mbps, 100.0 * bw_mbps / 14400.0);
        end
        $display("  Read efficiency (HBM): %.1f%%", 100.0 * rbw / 14400.0);
        $display();

        // ── P3: Streaming large read (MoE expert: 33MB contiguous, 256-beat bursts) ──
        $display("── P3: Streaming Large Read (MoE Expert 33MB-style, %0dKB) ──", TEST_KB);
        @(posedge clk); @(posedge clk);
        total_data_bytes = TEST_KB * 1024;

        // Issue AR and wait for all data — pipelined
        rready <= 1;
        for (b = 0; b < TOTAL_BURSTS; b++) begin
            @(posedge clk);
            arvalid <= 1;
            araddr  <= b * BYTES_PER_BURST;
            arlen   <= BURST_BEATS - 1;
            while (!(arvalid && arready)) @(posedge clk);
            arvalid <= 0;
        end
        // Drain all read data
        repeat (TOTAL_BURSTS) begin
            while (!(rvalid && rready && rlast)) @(posedge clk);
        end
        rready <= 0;

        $display("  Read bytes: %0d, HBM read BW: %0d MB/s", rb, rbw);
        $display("  Large-transfer read efficiency: %.1f%%", 100.0 * rbw / 14400.0);
        $display();

        // ── Summary ──
        $display("═══════════════════════════════════════════════════════");
        $display("  Bandwidth Benchmark Results (Single Pseudo-Channel)");
        $display("═══════════════════════════════════════════════════════");
        $display("  Theoretical peak:  14,400 MB/s (100.0%%)");
        if (wbw > 0)
            $display("  Sequential write:  %0d MB/s (%.1f%%)", wbw, 100.0*wbw/14400.0);
        if (rbw > 0)
            $display("  Sequential read:   %0d MB/s (%.1f%%)", rbw, 100.0*rbw/14400.0);
        $display();
        $display("  For 32-channel HBM2e stack (@80%% of per-channel measured):");
        $display("  Read BW  = 32 × measured_read × 0.80");
        $display("  Write BW = 32 × measured_write × 0.80");
        $display();

        #100;
        $display("=== Benchmark Complete ===");
        $finish;
    end

endmodule
