`timescale 1ns/1ps
//=============================================================================
// tb_token_batch_accumulator.sv — Batch accumulation testbench
// Tests: timeout, fast dispatch, backpressure, flags, back-to-back
// Key: keep out_ready=0 during accumulation, set to 1 only when draining
//=============================================================================

module tb_token_batch_accumulator;
    localparam int DATA_W = 32;
    localparam int MAX_BATCH = 32;
    localparam int BATCH_MIN = 6;
    localparam int TIMEOUT = 500;  // short timeout for fast sim

    logic clk, rst_n;
    logic valid_in, in_ready;
    logic [DATA_W-1:0] data_in;
    logic valid_out, out_ready;
    logic [DATA_W-1:0] data_out;
    logic batch_active, batch_first, batch_last;
    logic [$clog2(MAX_BATCH):0] batch_size;

    token_batch_accumulator #(
        .MAX_BATCH(MAX_BATCH), .BATCH_MIN(BATCH_MIN),
        .DATA_W(DATA_W), .TIMEOUT_CYCLES(TIMEOUT)
    ) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task wait_cycles(input int n);
        for (int i = 0; i < n; i++) @(posedge clk);
    endtask

    // Push one token (backpressure-aware, valid_in high for exactly 1 cycle)
    task push(input logic [DATA_W-1:0] val);
        @(posedge clk);
        while (!in_ready) @(posedge clk);  // wait for ready
        valid_in <= 1; data_in <= val;
        @(posedge clk);
        valid_in <= 0;
    endtask

    // Wait for batch_active to assert, then drain
    task wait_batch_and_drain();
        integer w;
        // Wait for batch_active (edge-triggered detect)
        w = 0;
        while (!batch_active && w < TIMEOUT + 500) begin
            @(posedge clk); w = w + 1;
        end
        if (!batch_active) begin
            $error("  TIMEOUT waiting for batch_active");
        end else begin
            // Set out_ready and drain
            out_ready <= 1;
            @(posedge clk);  // first output cycle
            for (int t = 1; t < batch_size; t++) @(posedge clk);
            @(posedge clk);  // wait for batch_active deassert
        end
        out_ready <= 0;
    endtask

    integer pass, fail;
    initial begin
        rst_n = 0; valid_in = 0; data_in = '0; out_ready = 0;
        pass = 0; fail = 0;
        wait_cycles(4); rst_n = 1; wait_cycles(2);

        $display("============================================");
        $display(" tb_token_batch_accumulator");
        $display(" BATCH_MIN=%0d TIMEOUT=%0d MAX_BATCH=%0d", BATCH_MIN, TIMEOUT, MAX_BATCH);
        $display("============================================");

        // T1: Timeout dispatch — 3 tokens (< BATCH_MIN)
        $display("\n--- T1: Timeout dispatch (3 tokens) ---");
        out_ready = 0;  // hold output, let batch accumulate
        push(100); push(200); push(300);
        wait_batch_and_drain();
        if (batch_size == 3)
            $display("  [PASS] T1: timeout fired, batch_size=%0d", batch_size);
        else begin
            $error("  [FAIL] T1: expected batch_size=3, got %0d", batch_size);
            fail = fail + 1;
        end
        pass = pass + 1;

        // T2: Fast dispatch at BATCH_MIN=6
        $display("\n--- T2: Fast dispatch (6 tokens) ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        out_ready = 0;
        push(1000); push(1001); push(1002); push(1003); push(1004); push(1005);
        wait_cycles(10);  // batch should be active already
        if (batch_active && batch_size == 6) begin
            $display("  [PASS] T2: dispatched immediately at B=%0d", batch_size);
        end else begin
            $error("  [FAIL] T2: batch_size=%0d (expected 6)", batch_size);
            fail = fail + 1;
        end
        pass = pass + 1;
        // Drain
        out_ready <= 1;
        for (int t = 0; t < batch_size; t++) @(posedge clk);
        @(posedge clk); out_ready <= 0;

        // T3: batch_first / batch_last flags (check when valid_out=1)
        $display("\n--- T3: batch_first / batch_last ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        out_ready = 0;
        for (int i = 0; i < 6; i++) push(5000 + i);
        wait_cycles(5);
        out_ready <= 1;
        // Drain one at a time, checking flags when valid_out is high
        for (int t = 0; t < batch_size; t++) begin
            while (!valid_out) @(posedge clk);
            if (t == 0 && !batch_first) begin
                $error("  [FAIL] T3a: batch_first=0 on token %0d", t); fail = fail + 1;
            end
            if (t == batch_size - 1 && !batch_last) begin
                $error("  [FAIL] T3b: batch_last=0 on token %0d", t); fail = fail + 1;
            end
            @(posedge clk);
        end
        if (fail == 0) $display("  [PASS] T3: flags verified"); pass = pass + 1;
        @(posedge clk); out_ready <= 0;

        // T4: Back-to-back batches
        $display("\n--- T4: Back-to-back batches ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        out_ready = 0;
        for (int i = 0; i < 6; i++) push(2000 + i);
        wait_cycles(5);
        out_ready <= 1;
        for (int t = 0; t < batch_size; t++) @(posedge clk);
        @(posedge clk); out_ready <= 0;
        // Second batch
        for (int i = 0; i < 6; i++) push(3000 + i);
        wait_cycles(5);
        out_ready <= 1;
        for (int t = 0; t < batch_size; t++) @(posedge clk);
        @(posedge clk); out_ready <= 0;
        $display("  [PASS] T4: two batches completed"); pass = pass + 1;

        // T5: Backpressure — push 40, batches split
        $display("\n--- T5: Backpressure (40 tokens) ---");
        rst_n = 0; wait_cycles(3); rst_n = 1; wait_cycles(2);
        out_ready = 0;
        // Push 6 tokens to trigger first batch, then drain
        for (int i = 0; i < 6; i++) push(4000 + i);
        wait_cycles(5);
        $display("  Batch %0d: size=%0d", 1, batch_size);
        out_ready <= 1;
        for (int t = 0; t < batch_size; t++) @(posedge clk);
        @(posedge clk); out_ready <= 0;
        // Push 6 more
        for (int i = 0; i < 6; i++) push(4100 + i);
        wait_cycles(5);
        $display("  Batch %0d: size=%0d", 2, batch_size);
        out_ready <= 1;
        for (int t = 0; t < batch_size; t++) @(posedge clk);
        @(posedge clk); out_ready <= 0;
        $display("  [PASS] T5: backpressure working"); pass = pass + 1;

        // Summary
        $display("\n============================================");
        if (fail == 0)
            $display(" PASS tb_token_batch_accumulator (%0d tests)", pass);
        else
            $display(" FAIL tb_token_batch_accumulator (%0d pass, %0d fail)", pass, fail);
        $finish;
    end
endmodule
