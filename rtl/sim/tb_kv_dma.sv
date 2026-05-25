`timescale 1ns/1ps

module tb_kv_dma;
    localparam int BEAT_BYTES = 32;
    localparam int WORDS_PER_BEAT = BEAT_BYTES / 4;

    logic clk, rst_n;

    // Descriptor
    logic desc_valid, desc_ready;
    logic [63:0] desc_host_addr;
    logic [31:0] desc_hbm_addr;
    logic [31:0] desc_length;
    logic [15:0] desc_session_id;

    // DMA request
    logic dma_req_valid, dma_req_ready;
    logic [63:0] dma_req_addr;
    logic [31:0] dma_req_length;

    // DMA response
    logic dma_rsp_valid, dma_rsp_last;
    logic [BEAT_BYTES*8-1:0] dma_rsp_data;

    // HBM
    logic [31:0] hbm_wr_addr;
    logic [31:0] hbm_wr_data;
    logic hbm_wr_en;

    // Status
    logic done;
    logic [15:0] session_id;
    logic [31:0] bytes_transferred;

    // HBM memory model
    logic [31:0] hbm_mem [1024];

    // Track total expected bytes for rsp_last
    logic [31:0] total_host_bytes;
    logic [31:0] host_bytes_done;

    kv_dma_engine #(.BEAT_BYTES(BEAT_BYTES)) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    // Host responder with byte tracking for correct rsp_last
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_req_ready  <= 1'b1;
            dma_rsp_valid  <= 1'b0;
            dma_rsp_data   <= '0;
            dma_rsp_last   <= 1'b0;
            host_bytes_done <= '0;
            total_host_bytes <= '0;
        end else begin
            dma_rsp_valid <= 1'b0;
            dma_rsp_last  <= 1'b0;

            // Latch total length when new descriptor accepted
            if (desc_valid && desc_ready) begin
                total_host_bytes <= desc_length;
                host_bytes_done  <= '0;
            end

            if (dma_req_valid && dma_req_ready) begin
                dma_rsp_valid <= 1'b1;
                for (int w = 0; w < 8; w++)
                    dma_rsp_data[w*32 +: 32] <= dma_req_addr[31:0] + w;

                // rsp_last: this is the last beat if remaining <= BEAT_BYTES
                if (total_host_bytes - host_bytes_done <= BEAT_BYTES)
                    dma_rsp_last <= 1'b1;

                host_bytes_done <= host_bytes_done + dma_req_length;
            end
        end
    end

    // HBM write capture — word-addressed
    always_ff @(posedge clk) begin
        if (hbm_wr_en)
            hbm_mem[hbm_wr_addr[31:2]] <= hbm_wr_data;
    end

    task wait_cycles(input int n);
        for (int i = 0; i < n; i++) @(posedge clk);
    endtask

    task submit_desc(input [63:0] host_addr, input [31:0] hbm_addr,
                     input [31:0] len, input [15:0] sid);
        @(posedge clk);
        desc_valid       = 1;
        desc_host_addr   = host_addr;
        desc_hbm_addr    = hbm_addr;
        desc_length      = len;
        desc_session_id  = sid;
        @(posedge clk);
        desc_valid       = 0;
    endtask

    function [31:0] read_hbm(input [31:0] byte_addr);
        read_hbm = hbm_mem[byte_addr[31:2]];
    endfunction

    integer pass_count, fail_count;

    initial begin
        rst_n = 0;
        desc_valid = 0; desc_host_addr = '0; desc_hbm_addr = '0;
        desc_length = '0; desc_session_id = '0;
        pass_count = 0; fail_count = 0;

        wait_cycles(4);
        rst_n = 1;
        wait_cycles(2);

        // ============================================
        // Test 1: Single-beat transfer (16 bytes)
        // ============================================
        $display("Test 1: Single-beat transfer (16 bytes)");
        submit_desc(64'h1000_0000, 32'h0000_0100, 32'd16, 16'd42);

        for (int cyc = 0; cyc < 30; cyc++) begin
            @(posedge clk);
            if (done) begin
                if (session_id !== 16'd42) begin
                    $error("  [FAIL] session_id: got %0d exp 42", session_id);
                    fail_count = fail_count + 1;
                end
                if (bytes_transferred !== 32'd16) begin
                    $error("  [FAIL] bytes: got %0d exp 16", bytes_transferred);
                    fail_count = fail_count + 1;
                end
                // Check HBM (word-addressed: addr>>2)
                for (int w = 0; w < 4; w++) begin
                    if (read_hbm(32'h100 + w*4) !== (32'h1000_0000 + w)) begin
                        $error("  [FAIL] HBM[0x%h]: got 0x%h exp 0x%h",
                            32'h100 + w*4, read_hbm(32'h100 + w*4), 32'h1000_0000 + w);
                        fail_count = fail_count + 1;
                    end
                end
                if (fail_count == 0) begin
                    $display("  [ OK ] Test 1: single-beat 16B, session 42");
                    pass_count = pass_count + 1;
                end
            end
        end

        wait_cycles(4);

        // ============================================
        // Test 2: Multi-beat transfer (100 bytes = 3×32 + 4)
        // ============================================
        $display("Test 2: Multi-beat transfer (100 bytes)");
        submit_desc(64'h2000_0000, 32'h0000_0200, 32'd100, 16'd7);

        for (int cyc = 0; cyc < 60; cyc++) begin
            @(posedge clk);
            if (done) begin
                if (bytes_transferred !== 32'd100) begin
                    $error("  [FAIL] bytes: got %0d exp 100", bytes_transferred);
                    fail_count = fail_count + 1;
                end
                if (session_id !== 16'd7) begin
                    $error("  [FAIL] session_id: got %0d exp 7", session_id);
                    fail_count = fail_count + 1;
                end
                // Check word 0 of each beat
                if (read_hbm(32'h200) !== 32'h2000_0000) begin
                    $error("  [FAIL] Beat0: got 0x%h", read_hbm(32'h200));
                    fail_count = fail_count + 1;
                end
                // Beat 1 starts at 0x220 (byte addr 0x200 + 32)
                if (read_hbm(32'h220) !== 32'h2000_0020) begin
                    $error("  [FAIL] Beat1: got 0x%h", read_hbm(32'h220));
                    fail_count = fail_count + 1;
                end
                // Beat 2 starts at 0x240 (byte addr 0x200 + 64)
                if (read_hbm(32'h240) !== 32'h2000_0040) begin
                    $error("  [FAIL] Beat2: got 0x%h", read_hbm(32'h240));
                    fail_count = fail_count + 1;
                end
                if (fail_count == 0) begin
                    $display("  [ OK ] Test 2: multi-beat 100B, session 7");
                    pass_count = pass_count + 1;
                end
            end
        end

        $display("==============================");
        if (fail_count == 0)
            $display("PASS tb_kv_dma (%0d tests)", pass_count);
        else
            $display("FAIL tb_kv_dma (%0d pass, %0d fail)", pass_count, fail_count);
        $finish;
    end

endmodule
