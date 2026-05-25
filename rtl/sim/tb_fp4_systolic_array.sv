`timescale 1ns/1ps

module tb_fp4_systolic_array;
    localparam int LANES = 4;
    logic clk;
    logic rst_n;
    logic start;
    logic k_valid;
    logic k_last;
    logic [15:0] elem_idx_base;
    logic [LANES*4-1:0] weight_fp4_flat;
    logic [LANES*8-1:0] activ_fp8_flat;
    logic k_ready;
    logic scale_wr_en;
    logic [1:0] scale_wr_addr;
    logic [7:0] scale_wr_data;
    logic busy;
    logic result_valid;
    logic result_ready;
    logic [31:0] sum_result;
    logic [LANES*32-1:0] lane_result_flat;

    fp4_systolic_array #(
        .LANES(LANES),
        .NUM_GROUPS(4),
        .GROUP_SIZE(16),
        .ADDR_WIDTH(2),
        .ELEM_WIDTH(16),
        .DRAIN_CYCLES(16)
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

    task drive_beat(input [15:0] base, input bit last);
        begin
            elem_idx_base <= base;
            for (int i = 0; i < LANES; i++) begin
                weight_fp4_flat[i*4 +: 4] <= 4'h4;  // +1.0
                activ_fp8_flat [i*8 +: 8] <= 8'h38; // +1.0
            end
            @(posedge clk);
            k_valid <= 1'b1;
            k_last <= last;
            @(posedge clk);
            k_valid <= 1'b0;
            k_last <= 1'b0;
        end
    endtask

    initial begin
        rst_n = 0;
        start = 0;
        k_valid = 0;
        k_last = 0;
        elem_idx_base = 0;
        weight_fp4_flat = '0;
        activ_fp8_flat = '0;
        scale_wr_en = 0;
        scale_wr_addr = 0;
        scale_wr_data = 0;
        result_ready = 1'b1;
        repeat (4) @(posedge clk);
        rst_n = 1;

        write_scale(0, 8'h38); // scale 1.0
        write_scale(1, 8'h40); // scale 2.0

        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        drive_beat(16'd0, 1'b0);   // group 0, +4096/lane
        drive_beat(16'd16, 1'b1);  // group 1, +8192/lane

        for (int cyc = 0; cyc < 40; cyc++) begin
            @(posedge clk);
            if (result_valid) begin
                #1;
                if (sum_result !== 32'h0000c000) begin
                    $error("sum expected 0x0000c000, got 0x%08h", sum_result);
                    $fatal;
                end
                for (int i = 0; i < LANES; i++) begin
                    if (lane_result_flat[i*32 +: 32] !== 32'h00003000) begin
                        $error("lane %0d expected 0x00003000, got 0x%08h",
                               i, lane_result_flat[i*32 +: 32]);
                        $fatal;
                    end
                end
                $display("PASS tb_fp4_systolic_array");
                $finish;
            end
        end
        $error("timeout waiting for done");
        $fatal;
    end
endmodule
