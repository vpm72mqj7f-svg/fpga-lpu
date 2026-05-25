`timescale 1ns/1ps

module tb_fp4_scaled_tile;
    localparam int LANES = 4;
    logic clk;
    logic rst_n;
    logic accum_clr;
    logic valid_in;
    logic [15:0] elem_idx_base;
    logic [LANES*4-1:0] weight_fp4_flat;
    logic [LANES*8-1:0] activ_fp8_flat;
    logic scale_wr_en;
    logic [1:0] scale_wr_addr;
    logic [7:0] scale_wr_data;
    logic valid_out;
    logic [LANES*32-1:0] lane_result_flat;
    logic [31:0] sum_result;

    fp4_scaled_tile #(
        .LANES(LANES),
        .NUM_GROUPS(4),
        .GROUP_SIZE(16),
        .ADDR_WIDTH(2),
        .ELEM_WIDTH(16)
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

    task clear_accum;
        begin
            @(posedge clk);
            accum_clr <= 1'b1;
            @(posedge clk);
            accum_clr <= 1'b0;
        end
    endtask

    task drive_common(input [15:0] base_idx);
        begin
            elem_idx_base <= base_idx;
            for (int i = 0; i < LANES; i++) begin
                weight_fp4_flat[i*4 +: 4] <= 4'h4;  // +1.0
                activ_fp8_flat [i*8 +: 8] <= 8'h38; // +1.0
            end
            @(posedge clk);
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task wait_cycles(input int n);
        begin
            for (int i = 0; i < n; i++) @(posedge clk);
            #1;
        end
    endtask

    initial begin
        rst_n = 0;
        accum_clr = 0;
        valid_in = 0;
        elem_idx_base = 0;
        weight_fp4_flat = '0;
        activ_fp8_flat = '0;
        scale_wr_en = 0;
        scale_wr_addr = 0;
        scale_wr_data = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;

        write_scale(0, 8'h38); // +1.0
        write_scale(1, 8'h40); // +2.0

        clear_accum();
        drive_common(16'd0);  // group 0, all lane expected 4096, sum 16384
        wait_cycles(24);
        if (sum_result !== 32'h00004000) begin
            $error("group0 sum expected 0x00004000, got 0x%08h", sum_result);
            $fatal;
        end
        $display("[ OK ] group0 scale=1.0 sum=0x%08h", sum_result);

        clear_accum();
        drive_common(16'd16); // group 1, all lane expected 8192, sum 32768
        wait_cycles(24);
        if (sum_result !== 32'h00008000) begin
            $error("group1 sum expected 0x00008000, got 0x%08h", sum_result);
            $fatal;
        end
        $display("[ OK ] group1 scale=2.0 sum=0x%08h", sum_result);

        $display("PASS tb_fp4_scaled_tile");
        $finish;
    end
endmodule
