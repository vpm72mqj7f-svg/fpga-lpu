//=============================================================================
// tb_axi4_hbm_sanity.sv — Minimal AXI4 write+read sanity check
// One burst write (addr=0, 16 beats), one burst read (addr=0, 16 beats).
//=============================================================================
`timescale 1ns / 1ps

module tb_axi4_hbm_sanity;

    localparam int AW = 256;
    localparam int AA = 32;
    localparam int AI = 4;

    logic clk, rst_n;

    // AW
    logic [AI-1:0]  awid;
    logic [AA-1:0]  awaddr;
    logic [7:0]     awlen;
    logic [2:0]     awsize;
    logic [1:0]     awburst;
    logic           awvalid, awready;

    // W
    logic [AW-1:0]  wdata;
    logic [AW/8-1:0] wstrb;
    logic           wlast;
    logic           wvalid, wready;

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

    // HBM model
    sim_axi4_hbm_model #(
        .MEM_SIZE_BYTES(65536), .READ_LATENCY(5), .WRITE_LATENCY(3)
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

    integer i;
    logic [31:0] marker;

    initial begin
        clk = 0; rst_n = 0;
        awvalid = 0; wvalid = 0; bready = 1;
        arvalid = 0; rready = 0;
        awid = 0; awaddr = 0; awlen = 15; awsize = 5; awburst = 1;
        wstrb = '1;
        arid = 0; araddr = 0; arlen = 15; arsize = 5; arburst = 1;

        #10 rst_n = 1;
        #10;

        $display("=== AXI4 HBM Sanity Test ===");
        $display("READ_LAT=5, WRITE_LAT=3");

        // ── Write one burst (16 beats) to addr=0 ──
        $display("--- Write Phase ---");
        @(posedge clk);
        awvalid <= 1;
        awaddr <= 0;
        awlen <= 15;  // 16 beats

        // Send AW
        while (!(awvalid && awready)) @(posedge clk);
        $display("  AW handshake @ %0t, addr=0x%0h", $time, awaddr);
        awvalid <= 0;

        // Send W beats — set data BEFORE the handshake loop
        wvalid <= 1;
        for (i = 0; i < 16; i++) begin
            wlast <= (i == 15);
            wdata <= {192'd0, 32'hDEAD0000 + i[31:0]};
            @(posedge clk);
            while (!(wvalid && wready)) @(posedge clk);
            // Data is already set for next beat on the @(posedge clk)
        end
        wvalid <= 0;
        wlast <= 0;

        // Wait for B response
        while (!bvalid) @(posedge clk);
        $display("  B response @ %0t: bid=%0d bresp=%0d", $time, bid, bresp);

        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);

        // Check memory directly
        $display("--- Memory Check (addr=0 words 0-7) ---");
        $display("  mem[0] = 0x%08h", u_hbm.mem[0]);
        $display("  mem[1] = 0x%08h", u_hbm.mem[1]);
        $display("  mem[2] = 0x%08h", u_hbm.mem[2]);
        $display("  mem[3] = 0x%08h", u_hbm.mem[3]);

        // ── Read one burst (16 beats) from addr=0 ──
        $display("--- Read Phase ---");
        @(posedge clk);
        arvalid <= 1;
        araddr <= 0;
        arlen <= 15;
        rready <= 1;

        while (!(arvalid && arready)) @(posedge clk);
        $display("  AR handshake @ %0t, addr=0x%0h", $time, araddr);
        arvalid <= 0;

        // Wait for read data
        i = 0;
        while (i < 16) begin
            while (!rvalid) @(posedge clk);
            $display("  R beat %0d @ %0t: data[31:0]=0x%08h, expected=0x%08h, %s",
                     i, $time, rdata[31:0], 32'hDEAD0000 + i,
                     rdata[31:0] == (32'hDEAD0000 + i) ? "OK" : "FAIL");
            if (rlast) $display("  RLAST asserted");
            i = i + 1;
            @(posedge clk);
        end
        rready <= 0;

        // Check marker in first word
        marker = u_hbm.mem[0];
        $display();
        $display("=== Final: mem[0] = 0x%08h, first read data[31:0] = (see above) ===", marker);

        #100;
        $display("=== Test Complete ===");
        $finish;
    end

endmodule
