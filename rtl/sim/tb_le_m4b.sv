`timescale 1ns/1ps

module tb_fp4_linear_engine;
    localparam int M_OUT = 4;
    localparam int K_TOTAL = 8;
    localparam int LANES = 4;
    localparam int K_BEATS = 2;

    logic clk;
    logic rst_n;
    logic weight_wr_en;
    logic [$clog2(M_OUT)-1:0] weight_wr_row;
    logic [$clog2(K_BEATS)-1:0] weight_wr_beat;
    logic [LANES*4-1:0] weight_wr_data;
    logic activ_wr_en;
    logic [$clog2(K_BEATS)-1:0] activ_wr_beat;
    logic [LANES*8-1:0] activ_wr_data;
    logic scale_wr_en;
    logic [1:0] scale_wr_addr;
    logic [7:0] scale_wr_data;
    logic start;
    logic busy;
    logic done;
    logic result_valid;
    logic [$clog2(M_OUT)-1:0] result_row;
    logic [31:0] result_data;
    logic result_ready = 1'b1;

    int seen_rows;
    logic [31:0] results [M_OUT];

    fp4_linear_engine #(
        .M_OUT(M_OUT),
        .K_TOTAL(K_TOTAL),
        .LANES(LANES),
        .GROUP_SIZE(2),
        .NUM_GROUPS(4),
        .ADDR_WIDTH(2)
    ) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task write_scale(input [1:0] addr, input [7:0] data);
        begin
            @(posedge clk);
            scale_wr_en <= 1'b1;
            scale_wr_addr <= addr;
            scale_wr_data <= data;
            @(posedge clk);
            scale_wr_en <= 1'b0;
        end
    endtask

    task write_activ(input int beat, input [LANES*8-1:0] data);
        begin
            @(posedge clk);
            activ_wr_en <= 1'b1;
            activ_wr_beat <= beat[$clog2(K_BEATS)-1:0];
            activ_wr_data <= data;
            @(posedge clk);
            activ_wr_en <= 1'b0;
        end
    endtask

    task write_weight(input int row, input int beat, input [LANES*4-1:0] data);
        begin
            @(posedge clk);
            weight_wr_en <= 1'b1;
            weight_wr_row <= row[$clog2(M_OUT)-1:0];
            weight_wr_beat <= beat[$clog2(K_BEATS)-1:0];
            weight_wr_data <= data;
            @(posedge clk);
            weight_wr_en <= 1'b0;
        end
    endtask

    initial begin
        rst_n = 0;
        weight_wr_en = 0;
        weight_wr_row = 0;
        weight_wr_beat = 0;
        weight_wr_data = '0;
        activ_wr_en = 0;
        activ_wr_beat = 0;
        activ_wr_data = '0;
        scale_wr_en = 0;
        scale_wr_addr = 0;
        scale_wr_data = 0;
        start = 0;
        seen_rows = 0;
        for (int ii = 0; ii < M_OUT; ii = ii + 1) results[ii] = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;

        // With GROUP_SIZE=2: elem 0-1→g0, 2-3→g1, 4-5→g2, 6-7→g3
        write_scale(0, 8'h38);
        write_scale(1, 8'h40);
        write_scale(2, 8'h38);
        write_scale(3, 8'h40);

        // activations all +1.0 in both K beats
        write_activ(0, {4{8'h38}});
        write_activ(1, {4{8'h38}});

        // row0 weights all +1.0: beat0 sum=4*4096, beat1 scale2 sum=4*8192 -> 0xC000
        write_weight(0, 0, {4{4'h4}});
        write_weight(0, 1, {4{4'h4}});
        // row1 weights all +2.0: beat0 sum=4*8192, beat1 scale2 sum=4*16384 -> 0x18000
        write_weight(1, 0, {4{4'h6}});
        write_weight(1, 1, {4{4'h6}});
        // row2 weights all +1.0 (same as row0)
        write_weight(2, 0, {4{4'h4}});
        write_weight(2, 1, {4{4'h4}});
        // row3 weights all +2.0 (same as row1)
        write_weight(3, 0, {4{4'h6}});
        write_weight(3, 1, {4{4'h6}});

        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        for (int cyc = 0; cyc < 500; cyc++) begin
            @(posedge clk);
            if (result_valid) begin
                results[result_row] = result_data;
                seen_rows++;
                $display("row %0d result=0x%08h", result_row, result_data);
            end
            if (done) begin
                #1;
                if (seen_rows != M_OUT) begin
                    $error("expected %0d result rows, got %0d", M_OUT, seen_rows);
                    $fatal;
                end
                if (results[0] !== 32'h0000c000) begin
                    $error("row0 expected 0x0000c000, got 0x%08h", results[0]);
                    $fatal;
                end
                if (results[1] !== 32'h00018000) begin
                    $error("row1 expected 0x00018000, got 0x%08h", results[1]);
                    $fatal;
                end
                if (results[2] !== 32'h0000c000) begin
                    $error("row2 expected 0x0000c000, got 0x%08h", results[2]);
                    $fatal;
                end
                if (results[3] !== 32'h00018000) begin
                    $error("row3 expected 0x00018000, got 0x%08h", results[3]);
                    $fatal;
                end
                $display("PASS tb_le_m4b (M_OUT=4)");
                $finish;
            end
        end
        $error("timeout waiting for done");
        $fatal;
    end
endmodule
