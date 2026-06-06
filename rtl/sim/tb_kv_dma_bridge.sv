`timescale 1ns/1ps
//=============================================================================
// tb_kv_dma_bridge.sv - Testbench for kv_dma_bridge (PCIe-to-HBM DMA bridge)
//
// Tests:
//   T1: Single-token PCIe->HBM forward path (DMA descriptor -> HBM write)
//   T2: Multi-token transfer (3 tokens, sequential DMA requests)
//   T3: Buffer swap behavior (verify buf_a_active toggle, buf_b_ready clear)
//   T4: Back-to-back transfers (two separate DMA operations)
//   T5: Different payload sizes (1, 2, 5 tokens)
//
// Uses KV_ENTRY_BYTES=128 (4 beats/entry) for fast simulation.
// Self-checking with PASS/FAIL summary.
//=============================================================================

module tb_kv_dma_bridge;

    // -- Test parameters (override production defaults for fast sim) --
    localparam int TEST_KV_ENTRY_BYTES  = 128;
    localparam int TEST_MAX_TOKENS      = 64;
    localparam int TEST_PCIE_BEAT_BYTES = 32;
    localparam int TEST_BEATS_PER_ENTRY = TEST_KV_ENTRY_BYTES / TEST_PCIE_BEAT_BYTES;  // 4

    // -- DUT signals --
    logic        clk, rst_n;
    logic        start_dma;
    logic [31:0] host_addr_base;
    logic [31:0] hbm_addr_base;
    logic [15:0] num_tokens;
    logic        dma_done;
    logic [15:0] tokens_transferred;

    logic        buf_a_active;
    logic        buf_b_ready;
    logic        swap_buffers;

    // PCIe DMA request
    logic        pcie_req_valid;
    logic        pcie_req_ready;
    logic [63:0] pcie_req_addr;
    logic [31:0] pcie_req_length;

    // PCIe DMA response
    logic        pcie_rsp_valid;
    logic [255:0] pcie_rsp_data;
    logic        pcie_rsp_last;

    // HBM write port
    logic [31:0]  hbm_wr_addr;
    logic [255:0] hbm_wr_data;
    logic         hbm_wr_en;

    // -- Instantiate DUT --
    kv_dma_bridge #(
        .KV_ENTRY_BYTES (TEST_KV_ENTRY_BYTES),
        .MAX_TOKENS     (TEST_MAX_TOKENS),
        .PCIE_BEAT_BYTES(TEST_PCIE_BEAT_BYTES),
        .BEATS_PER_ENTRY(TEST_BEATS_PER_ENTRY)
    ) dut (.*);

    // -- Clock generator: 100 MHz (10 ns period) --
    initial clk = 0;
    always #5 clk = ~clk;

    // -- HBM memory model (byte-addressable, 256-bit words) --
    // Address as byte offset; store 256-bit words indexed by byte_addr[31:5] (32B aligned)
    logic [255:0] hbm_mem [0:4095];  // enough for 128KB of 256-bit words
    logic [31:0]  hbm_wr_count;      // debug: number of HBM writes

    always_ff @(posedge clk) begin
        if (hbm_wr_en) begin
            hbm_mem[hbm_wr_addr[31:5]] <= hbm_wr_data;
            hbm_wr_count <= hbm_wr_count + 1;
        end
    end

    // -- PCIe Responder Model --
    // On receiving a DMA read request, the responder sends back TEST_BEATS_PER_ENTRY
    // beats of data. Data pattern: each 256-bit beat = 8 x 32-bit words,
    // each word = host_byte_addr + beat_offset + word_offset.
    // This makes each 32-bit word within a beat unique and verifiable.

    typedef enum logic [1:0] { RSP_IDLE, RSP_ACTIVE, RSP_DONE } rsp_state_t;
    rsp_state_t rsp_state;

    logic [31:0] rsp_host_addr;       // latched request address
    logic [31:0] rsp_beat_base;       // byte address at start of current token
    logic [4:0]  rsp_beat_idx;        // 0..TEST_BEATS_PER_ENTRY-1
    logic [15:0] rsp_tokens_sent;     // count of tokens fully responded
    logic [15:0] rsp_tokens_total;    // total tokens requested

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pcie_req_ready   <= 1'b1;
            pcie_rsp_valid   <= 1'b0;
            pcie_rsp_data    <= '0;
            pcie_rsp_last    <= 1'b0;
            rsp_state        <= RSP_IDLE;
            rsp_host_addr    <= '0;
            rsp_beat_base    <= '0;
            rsp_beat_idx     <= '0;
            rsp_tokens_sent  <= '0;
            rsp_tokens_total <= '0;
        end else begin
            pcie_rsp_valid <= 1'b0;
            pcie_rsp_last  <= 1'b0;

            case (rsp_state)
                RSP_IDLE: begin
                    pcie_req_ready <= 1'b1;
                    if (pcie_req_valid && pcie_req_ready) begin
                        // Latch the request
                        rsp_host_addr    <= pcie_req_addr[31:0];
                        rsp_beat_base    <= pcie_req_addr[31:0];
                        rsp_beat_idx     <= '0;
                        rsp_tokens_total <= rsp_tokens_sent + 1'b1;
                        pcie_req_ready   <= 1'b0;
                        rsp_state        <= RSP_ACTIVE;
                    end
                end

                RSP_ACTIVE: begin
                    pcie_req_ready <= 1'b0;
                    // Send one beat per cycle while active
                    pcie_rsp_valid <= 1'b1;

                    // Build data pattern: each 32-bit word = host_byte_addr + word_offset
                    for (int w = 0; w < 8; w++) begin
                        pcie_rsp_data[w*32 +: 32] <= rsp_beat_base +
                            (rsp_beat_idx * TEST_PCIE_BEAT_BYTES) + (w * 4);
                    end

                    // Last beat in this token?
                    if (rsp_beat_idx == TEST_BEATS_PER_ENTRY - 1) begin
                        pcie_rsp_last     <= 1'b1;
                        rsp_beat_idx      <= '0;
                        rsp_tokens_sent   <= rsp_tokens_sent + 1'b1;
                        rsp_host_addr     <= rsp_host_addr + TEST_KV_ENTRY_BYTES;
                        rsp_beat_base     <= rsp_host_addr + TEST_KV_ENTRY_BYTES;
                        rsp_state         <= RSP_DONE;
                    end else begin
                        rsp_beat_idx <= rsp_beat_idx + 1'b1;
                    end
                end

                RSP_DONE: begin
                    pcie_req_ready <= 1'b1;
                    // Check if another request is pending
                    if (pcie_req_valid && pcie_req_ready) begin
                        rsp_host_addr    <= pcie_req_addr[31:0];
                        rsp_beat_base    <= pcie_req_addr[31:0];
                        rsp_beat_idx     <= '0;
                        rsp_tokens_total <= rsp_tokens_sent + 1'b1;
                        pcie_req_ready   <= 1'b0;
                        rsp_state        <= RSP_ACTIVE;
                    end else begin
                        rsp_state <= RSP_IDLE;
                    end
                end
            endcase
        end
    end

    // -- Helper: read HBM 32-bit word at byte address --
    function logic [31:0] read_hbm_word(input logic [31:0] byte_addr);
        logic [255:0] beat;
        beat = hbm_mem[byte_addr[31:5]];
        // byte_addr[4:2] selects which 32-bit word within the 256-bit beat
        read_hbm_word = beat[(byte_addr[4:2])*32 +: 32];
    endfunction

    // -- Utility tasks --
    task wait_cycles(input int n);
        for (int i = 0; i < n; i++) @(posedge clk);
    endtask

    // Pulse start_dma
    task pulse_start(input [31:0] host_addr, input [31:0] hbm_addr,
                     input [15:0] ntokens);
        @(posedge clk);
        start_dma      <= 1'b1;
        host_addr_base <= host_addr;
        hbm_addr_base  <= hbm_addr;
        num_tokens     <= ntokens;
        @(posedge clk);
        start_dma      <= 1'b0;
    endtask

    // -- Counters --
    integer pass_count, fail_count, test_num;

    // -- Watchdog --
    initial begin
        #20000000;  // 20 ms
        $error("WATCHDOG TIMEOUT");
        $finish;
    end

    //=====================================================================
    // Main test sequence
    //=====================================================================
    initial begin
        // Init
        rst_n          = 1'b0;
        start_dma      = 1'b0;
        host_addr_base = '0;
        hbm_addr_base  = '0;
        num_tokens     = '0;
        swap_buffers   = 1'b0;
        pass_count     = 0;
        fail_count     = 0;
        test_num       = 0;

        wait_cycles(5);
        rst_n = 1'b1;
        wait_cycles(3);

        $display("==================================================================");
        $display(" tb_kv_dma_bridge - PCIe DMA Bridge Verification");
        $display(" KV_ENTRY_BYTES=%0d  BEATS_PER_ENTRY=%0d  PCIE_BEAT_BYTES=%0d",
                 TEST_KV_ENTRY_BYTES, TEST_BEATS_PER_ENTRY, TEST_PCIE_BEAT_BYTES);
        $display("==================================================================");
        $display("");

        //===============================================================
        // Test 1: Single-token PCIe->HBM forward path
        //===============================================================
        test_num = 1;
        $display("--- Test %0d: Single-token PCIe->HBM forward path ---", test_num);

        hbm_wr_count <= '0;
        rsp_tokens_sent <= '0;

        // Start DMA for 1 token: host=0x1000_0000, hbm=0x0000_0100
        pulse_start(32'h1000_0000, 32'h0000_0100, 16'd1);

        // Wait for dma_done with timeout
        fork : t1_wait
            begin
                for (int cyc = 0; cyc < 500; cyc++) begin
                    @(posedge clk);
                    if (dma_done) begin
                        // Verify tokens_transferred
                        if (tokens_transferred !== 16'd1) begin
                            $error("  [FAIL] T1: tokens_transferred=%0d, expected 1",
                                   tokens_transferred);
                            fail_count = fail_count + 1;
                        end else begin
                            $display("  [ OK ] T1: tokens_transferred = 1");
                        end

                        // Verify HBM contents: TEST_BEATS_PER_ENTRY beats written
                        // Data pattern: each beat = 8 x 32-bit words
                        // word = host_addr_base + beat*PCIE_BEAT_BYTES + w*4
                        begin
                            integer b, w, local_fails;
                            logic [31:0] byte_addr, exp_val, got_val;
                            local_fails = 0;
                            for (b = 0; b < TEST_BEATS_PER_ENTRY; b++) begin
                                for (w = 0; w < 8; w++) begin
                                    byte_addr = 32'h0000_0100 +
                                        (b * TEST_PCIE_BEAT_BYTES) + (w * 4);
                                    exp_val = 32'h1000_0000 +
                                        (b * TEST_PCIE_BEAT_BYTES) + (w * 4);
                                    got_val = read_hbm_word(byte_addr);
                                    if (got_val !== exp_val) begin
                                        $error("  [FAIL] T1: HBM[0x%08h] (beat %0d word %0d) = 0x%08h, exp 0x%08h",
                                               byte_addr, b, w, got_val, exp_val);
                                        local_fails = local_fails + 1;
                                    end
                                end
                            end
                            if (local_fails == 0) begin
                                $display("  [ OK ] T1: All %0d beats x %0d words verified",
                                         TEST_BEATS_PER_ENTRY, 8);
                            end else begin
                                fail_count = fail_count + local_fails;
                            end
                        end

                        // Verify buf_b_ready asserted at done
                        if (buf_b_ready !== 1'b1) begin
                            $error("  [FAIL] T1: buf_b_ready not asserted at dma_done");
                            fail_count = fail_count + 1;
                        end else begin
                            $display("  [ OK ] T1: buf_b_ready asserted at dma_done");
                        end

                        disable t1_wait;
                    end
                end
                $error("  [FAIL] T1: TIMEOUT waiting for dma_done");
                fail_count = fail_count + 1;
            end
        join

        if (fail_count == 0) begin
            $display("  [PASS] Test %0d: Single-token forward path", test_num);
            pass_count = pass_count + 1;
        end
        $display("");

        wait_cycles(10);

        //===============================================================
        // Test 2: Multi-token transfer (3 tokens)
        //===============================================================
        test_num = 2;
        $display("--- Test %0d: Multi-token transfer (3 tokens) ---", test_num);

        hbm_wr_count <= '0;
        rsp_tokens_sent <= '0;

        // Start DMA for 3 tokens: host=0x2000_0000, hbm=0x0000_0400
        pulse_start(32'h2000_0000, 32'h0000_0400, 16'd3);

        fork : t2_wait
            begin
                for (int cyc = 0; cyc < 1000; cyc++) begin
                    @(posedge clk);
                    if (dma_done) begin
                        if (tokens_transferred !== 16'd3) begin
                            $error("  [FAIL] T2: tokens_transferred=%0d, expected 3",
                                   tokens_transferred);
                            fail_count = fail_count + 1;
                        end else begin
                            $display("  [ OK ] T2: tokens_transferred = 3");
                        end

                        // Verify HBM for all 3 tokens
                        begin
                            integer tok, b, w, local_fails;
                            logic [31:0] byte_addr, exp_val, got_val;
                            local_fails = 0;
                            for (tok = 0; tok < 3; tok++) begin
                                for (b = 0; b < TEST_BEATS_PER_ENTRY; b++) begin
                                    for (w = 0; w < 8; w++) begin
                                        byte_addr = 32'h0000_0400 +
                                            (tok * TEST_KV_ENTRY_BYTES) +
                                            (b * TEST_PCIE_BEAT_BYTES) + (w * 4);
                                        exp_val = 32'h2000_0000 +
                                            (tok * TEST_KV_ENTRY_BYTES) +
                                            (b * TEST_PCIE_BEAT_BYTES) + (w * 4);
                                        got_val = read_hbm_word(byte_addr);
                                        if (got_val !== exp_val) begin
                                            $error("  [FAIL] T2: tok=%0d beat=%0d w=%0d HBM[0x%08h]=0x%08h exp 0x%08h",
                                                   tok, b, w, byte_addr, got_val, exp_val);
                                            local_fails = local_fails + 1;
                                        end
                                    end
                                end
                            end
                            if (local_fails == 0) begin
                                $display("  [ OK ] T2: All 3 tokens x %0d beats verified",
                                         TEST_BEATS_PER_ENTRY);
                            end else begin
                                fail_count = fail_count + local_fails;
                            end
                        end

                        disable t2_wait;
                    end
                end
                $error("  [FAIL] T2: TIMEOUT");
                fail_count = fail_count + 1;
            end
        join

        if (fail_count == 0) begin
            $display("  [PASS] Test %0d: Multi-token transfer", test_num);
            pass_count = pass_count + 1;
        end
        $display("");

        wait_cycles(10);

        //===============================================================
        // Test 3: Buffer swap behavior
        //===============================================================
        test_num = 3;
        $display("--- Test %0d: Buffer swap behavior ---", test_num);

        // Track initial buffer state
        $display("  Initial: buf_a_active=%b, buf_b_ready=%b",
                 buf_a_active, buf_b_ready);

        // buf_a_active should be 1 from reset (unless swapped earlier).
        // During T2, dma_done set buf_b_ready=1. Verify that.
        // Toggle swap_buffers and verify buf_a_active flips and buf_b_ready clears
        @(posedge clk);
        swap_buffers <= 1'b1;
        @(posedge clk);
        swap_buffers <= 1'b0;
        @(posedge clk);
        #1;

        if (buf_a_active !== 1'b0) begin
            $error("  [FAIL] T3: buf_a_active=%b after swap, expected 0", buf_a_active);
            fail_count = fail_count + 1;
        end else begin
            $display("  [ OK ] T3: buf_a_active toggled to 0 after swap");
        end

        if (buf_b_ready !== 1'b0) begin
            $error("  [FAIL] T3: buf_b_ready=%b after swap, expected 0 (cleared)",
                   buf_b_ready);
            fail_count = fail_count + 1;
        end else begin
            $display("  [ OK ] T3: buf_b_ready cleared after swap");
        end

        // Toggle again and verify it returns
        @(posedge clk);
        swap_buffers <= 1'b1;
        @(posedge clk);
        swap_buffers <= 1'b0;
        @(posedge clk);
        #1;

        if (buf_a_active !== 1'b1) begin
            $error("  [FAIL] T3: buf_a_active=%b after 2nd swap, expected 1",
                   buf_a_active);
            fail_count = fail_count + 1;
        end else begin
            $display("  [ OK ] T3: buf_a_active toggled back to 1 after 2nd swap");
        end

        if (fail_count == 0) begin
            $display("  [PASS] Test %0d: Buffer swap", test_num);
            pass_count = pass_count + 1;
        end
        $display("");

        wait_cycles(5);

        //===============================================================
        // Test 4: Back-to-back transfers
        //===============================================================
        test_num = 4;
        $display("--- Test %0d: Back-to-back transfers ---", test_num);

        // First transfer: 2 tokens
        hbm_wr_count <= '0;
        rsp_tokens_sent <= '0;
        pulse_start(32'h3000_0000, 32'h0000_0800, 16'd2);

        fork : t4a_wait
            begin
                for (int cyc = 0; cyc < 500; cyc++) begin
                    @(posedge clk);
                    if (dma_done) begin
                        if (tokens_transferred !== 16'd2) begin
                            $error("  [FAIL] T4a: tokens_transferred=%0d, expected 2",
                                   tokens_transferred);
                            fail_count = fail_count + 1;
                        end else begin
                            $display("  [ OK ] T4a: First transfer done (2 tokens)");
                        end

                        // Spot-check first and last words
                        if (read_hbm_word(32'h0800) !== 32'h3000_0000) begin
                            $error("  [FAIL] T4a: HBM[0x0800]=0x%08h, exp 0x3000_0000",
                                   read_hbm_word(32'h0800));
                            fail_count = fail_count + 1;
                        end
                        // Last word of last token: addr 0x0800 + 2*128 - 4 = 0x08FC
                        if (read_hbm_word(32'h08FC) !== 32'h3000_00FC) begin
                            $error("  [FAIL] T4a: HBM[0x08FC]=0x%08h, exp 0x3000_00FC",
                                   read_hbm_word(32'h08FC));
                            fail_count = fail_count + 1;
                        end else begin
                            $display("  [ OK ] T4a: HBM spot-checks pass");
                        end

                        disable t4a_wait;
                    end
                end
                $error("  [FAIL] T4a: TIMEOUT");
                fail_count = fail_count + 1;
            end
        join

        wait_cycles(10);

        // Second transfer: 1 token, different HBM region
        rsp_tokens_sent <= '0;
        pulse_start(32'h4000_0000, 32'h0000_0A00, 16'd1);

        fork : t4b_wait
            begin
                for (int cyc = 0; cyc < 500; cyc++) begin
                    @(posedge clk);
                    if (dma_done) begin
                        if (tokens_transferred !== 16'd1) begin
                            $error("  [FAIL] T4b: tokens_transferred=%0d, expected 1",
                                   tokens_transferred);
                            fail_count = fail_count + 1;
                        end else begin
                            $display("  [ OK ] T4b: Second transfer done (1 token)");
                        end

                        // Verify second transfer didn't corrupt first
                        if (read_hbm_word(32'h0800) !== 32'h3000_0000) begin
                            $error("  [FAIL] T4b: First xfer corrupted at 0x0800");
                            fail_count = fail_count + 1;
                        end
                        if (read_hbm_word(32'h0A00) !== 32'h4000_0000) begin
                            $error("  [FAIL] T4b: HBM[0x0A00]=0x%08h, exp 0x4000_0000",
                                   read_hbm_word(32'h0A00));
                            fail_count = fail_count + 1;
                        end else begin
                            $display("  [ OK ] T4b: Second transfer correct, first intact");
                        end

                        disable t4b_wait;
                    end
                end
                $error("  [FAIL] T4b: TIMEOUT");
                fail_count = fail_count + 1;
            end
        join

        if (fail_count == 0) begin
            $display("  [PASS] Test %0d: Back-to-back transfers", test_num);
            pass_count = pass_count + 1;
        end
        $display("");

        wait_cycles(10);

        //===============================================================
        // Test 5: Different payload sizes
        //===============================================================
        test_num = 5;
        $display("--- Test %0d: Different payload sizes ---", test_num);

        // 5a: 1 token
        rsp_tokens_sent <= '0;
        pulse_start(32'h5000_0000, 32'h0000_1000, 16'd1);
        fork : t5a_wait
            begin
                for (int cyc = 0; cyc < 500; cyc++) begin
                    @(posedge clk);
                    if (dma_done) begin
                        if (tokens_transferred === 16'd1 &&
                            read_hbm_word(32'h1000) === 32'h5000_0000) begin
                            $display("  [ OK ] T5a: 1-token xfer OK");
                        end else begin
                            $error("  [FAIL] T5a: 1-token xfer");
                            fail_count = fail_count + 1;
                        end
                        disable t5a_wait;
                    end
                end
                $error("  [FAIL] T5a: TIMEOUT");
                fail_count = fail_count + 1;
            end
        join

        wait_cycles(10);

        // 5b: 2 tokens
        rsp_tokens_sent <= '0;
        pulse_start(32'h5100_0000, 32'h0000_1200, 16'd2);
        fork : t5b_wait
            begin
                for (int cyc = 0; cyc < 500; cyc++) begin
                    @(posedge clk);
                    if (dma_done) begin
                        if (tokens_transferred === 16'd2 &&
                            read_hbm_word(32'h1200) === 32'h5100_0000 &&
                            read_hbm_word(32'h1280) === 32'h5100_0080) begin
                            $display("  [ OK ] T5b: 2-token xfer OK");
                        end else begin
                            $error("  [FAIL] T5b: 2-token xfer");
                            fail_count = fail_count + 1;
                        end
                        disable t5b_wait;
                    end
                end
                $error("  [FAIL] T5b: TIMEOUT");
                fail_count = fail_count + 1;
            end
        join

        wait_cycles(10);

        // 5c: 5 tokens
        rsp_tokens_sent <= '0;
        pulse_start(32'h5200_0000, 32'h0000_1400, 16'd5);
        fork : t5c_wait
            begin
                for (int cyc = 0; cyc < 1500; cyc++) begin
                    @(posedge clk);
                    if (dma_done) begin
                        begin
                            integer tok, local_fails;
                            logic [31:0] exp_first, got_first;
                            local_fails = 0;
                            if (tokens_transferred !== 16'd5) begin
                                $error("  [FAIL] T5c: tokens_transferred=%0d, expected 5",
                                       tokens_transferred);
                                local_fails = local_fails + 1;
                            end
                            // Spot-check each token's first word
                            for (tok = 0; tok < 5; tok++) begin
                                exp_first = 32'h5200_0000 +
                                    (tok * TEST_KV_ENTRY_BYTES);
                                got_first = read_hbm_word(32'h1400 +
                                    (tok * TEST_KV_ENTRY_BYTES));
                                if (got_first !== exp_first) begin
                                    $error("  [FAIL] T5c: token %0d first word 0x%08h != 0x%08h",
                                           tok, got_first, exp_first);
                                    local_fails = local_fails + 1;
                                end
                            end
                            if (local_fails == 0)
                                $display("  [ OK ] T5c: 5-token xfer OK");
                            else
                                fail_count = fail_count + local_fails;
                        end
                        disable t5c_wait;
                    end
                end
                $error("  [FAIL] T5c: TIMEOUT");
                fail_count = fail_count + 1;
            end
        join

        if (fail_count == 0) begin
            $display("  [PASS] Test %0d: Different payload sizes", test_num);
            pass_count = pass_count + 1;
        end
        $display("");

        //===============================================================
        // Summary
        //===============================================================
        $display("==================================================================");
        if (fail_count == 0) begin
            $display(" PASS  tb_kv_dma_bridge  (%0d/%0d tests)", pass_count, 5);
        end else begin
            $display(" FAIL  tb_kv_dma_bridge  (%0d pass, %0d failures)",
                     pass_count, fail_count);
        end
        $display("==================================================================");
        $finish;
    end

endmodule
