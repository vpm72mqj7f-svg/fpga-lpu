`timescale 1ns/1ps

module tb_expert_ffn_engine;
    localparam int HIDDEN = 8;
    localparam int INTER = 4;
    localparam int LANES = 4;
    localparam int K_BEATS = 2;

    logic clk, rst_n;
    logic activ_wr_en;
    logic [$clog2(K_BEATS)-1:0] activ_wr_beat;
    logic [LANES*8-1:0] activ_wr_data;
    logic scale_wr_en;
    logic [1:0] scale_wr_addr;
    logic [7:0] scale_wr_data;
    logic gate_w_wr_en;
    logic [$clog2(INTER)-1:0] gate_w_wr_row;
    logic [$clog2(K_BEATS)-1:0] gate_w_wr_beat;
    logic [LANES*4-1:0] gate_w_wr_data;
    logic up_w_wr_en;
    logic [$clog2(INTER)-1:0] up_w_wr_row;
    logic [$clog2(K_BEATS)-1:0] up_w_wr_beat;
    logic [LANES*4-1:0] up_w_wr_data;
    logic down_w_wr_en;
    logic [$clog2(HIDDEN)-1:0] down_w_wr_row;
    logic [$clog2(INTER)-1:0] down_w_wr_col;
    logic signed [31:0] down_w_wr_data;
    logic start, busy, done;
    logic result_valid;
    logic [$clog2(HIDDEN)-1:0] result_row;
    logic signed [31:0] result_data;

    logic [31:0] results [HIDDEN];
    int seen;

    expert_ffn_engine dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task write_scale(input [1:0] addr, input [7:0] data);
        @(posedge clk); scale_wr_en <= 1; scale_wr_addr <= addr; scale_wr_data <= data;
        @(posedge clk); scale_wr_en <= 0;
    endtask

    task write_activ(input int beat, input [LANES*8-1:0] data);
        @(posedge clk); activ_wr_en <= 1; activ_wr_beat <= beat[$clog2(K_BEATS)-1:0]; activ_wr_data <= data;
        @(posedge clk); activ_wr_en <= 0;
    endtask

    task write_gate(input int row, input int beat, input [LANES*4-1:0] data);
        @(posedge clk); gate_w_wr_en <= 1; gate_w_wr_row <= row[$clog2(INTER)-1:0]; gate_w_wr_beat <= beat[$clog2(K_BEATS)-1:0]; gate_w_wr_data <= data;
        @(posedge clk); gate_w_wr_en <= 0;
    endtask

    task write_up(input int row, input int beat, input [LANES*4-1:0] data);
        @(posedge clk); up_w_wr_en <= 1; up_w_wr_row <= row[$clog2(INTER)-1:0]; up_w_wr_beat <= beat[$clog2(K_BEATS)-1:0]; up_w_wr_data <= data;
        @(posedge clk); up_w_wr_en <= 0;
    endtask

    task write_down(input int row, input int col, input signed [31:0] data);
        @(posedge clk); down_w_wr_en <= 1; down_w_wr_row <= row[$clog2(HIDDEN)-1:0]; down_w_wr_col <= col[$clog2(INTER)-1:0]; down_w_wr_data <= data;
        @(posedge clk); down_w_wr_en <= 0;
    endtask

    initial begin
        rst_n = 0;
        activ_wr_en = 0; scale_wr_en = 0; gate_w_wr_en = 0; up_w_wr_en = 0; down_w_wr_en = 0;
        activ_wr_beat = 0; activ_wr_data = '0;
        scale_wr_addr = 0; scale_wr_data = 0;
        gate_w_wr_row = 0; gate_w_wr_beat = 0; gate_w_wr_data = '0;
        up_w_wr_row = 0; up_w_wr_beat = 0; up_w_wr_data = '0;
        down_w_wr_row = 0; down_w_wr_col = 0; down_w_wr_data = 0;
        start = 0; seen = 0;
        for (int i = 0; i < HIDDEN; i++) results[i] = 0;
        repeat (4) @(posedge clk); rst_n = 1;

        write_scale(0, 8'h38); // scale 1.0
        write_scale(1, 8'h38);
        write_activ(0, {4{8'h38}});
        write_activ(1, {4{8'h38}});

        // gate/up: all rows sum 8 ones => 8.0 = 0x8000 in Q12
        for (int r = 0; r < INTER; r++) begin
            write_gate(r, 0, {4{4'h4}});
            write_gate(r, 1, {4{4'h4}});
            write_up(r, 0, {4{4'h4}});
            write_up(r, 1, {4{4'h4}});
        end

        // down identity for first four hidden outputs, zeros elsewhere
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < INTER; c++) begin
                write_down(r, c, (r == c) ? 32'sd4096 : 32'sd0);
            end
        end

        @(posedge clk); start <= 1;
        @(posedge clk); start <= 0;

        for (int cyc = 0; cyc < 500; cyc++) begin
            @(posedge clk);
            if (result_valid) begin
                results[result_row] = result_data;
                seen++;
                $display("row %0d result=0x%08h", result_row, result_data);
            end
            if (done) begin
                #1;
                if (seen != HIDDEN) begin
                    $error("expected %0d result rows, got %0d", HIDDEN, seen);
                    $fatal;
                end
                for (int r = 0; r < HIDDEN; r++) begin
                    if (r < INTER) begin
                        if (results[r] !== 32'h00040000) begin
                            $error("row %0d expected 0x00040000, got 0x%08h", r, results[r]);
                            $fatal;
                        end
                    end else if (results[r] !== 32'h00000000) begin
                        $error("row %0d expected 0, got 0x%08h", r, results[r]);
                        $fatal;
                    end
                end
                $display("PASS tb_expert_ffn_engine");
                $finish;
            end
        end
        $error("timeout waiting for done");
        $fatal;
    end
endmodule
