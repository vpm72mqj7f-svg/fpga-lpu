`timescale 1ns/1ps
`include "fp4_types.svh"

module tb_fp4_scale_reader;
    logic clk;
    logic rst_n;
    logic q_valid;
    logic [15:0] q_elem_idx;
    logic q_ready;
    logic r_valid;
    logic [11:0] r_scale;
    logic [4:0] r_group_id;
    logic wr_en;
    logic [4:0] wr_addr;
    logic [7:0] wr_data;

    fp4_scale_reader #(
        .NUM_GROUPS(32),
        .GROUP_SIZE(16),
        .ADDR_WIDTH(5),
        .ELEM_WIDTH(16),
        .SCALE_WIDTH(12)
    ) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task write_scale(input [4:0] addr, input [7:0] data);
        begin
            @(posedge clk);
            wr_en <= 1'b1;
            wr_addr <= addr;
            wr_data <= data;
            @(posedge clk);
            wr_en <= 1'b0;
        end
    endtask

    task query(input [15:0] elem, input [4:0] exp_gid, input [11:0] exp_scale);
        begin
            @(posedge clk);
            q_valid <= 1'b1;
            q_elem_idx <= elem;
            @(posedge clk);
            q_valid <= 1'b0;
            #1;
            if (!r_valid || r_group_id !== exp_gid || r_scale !== exp_scale) begin
                $error("elem=%0d gid=%0d/%0d scale=%0d/%0d valid=%0b",
                       elem, r_group_id, exp_gid, r_scale, exp_scale, r_valid);
                $fatal;
            end
        end
    endtask

    initial begin
        rst_n = 0;
        q_valid = 0;
        q_elem_idx = 0;
        wr_en = 0;
        wr_addr = 0;
        wr_data = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;

        write_scale(0, 8'h11);
        write_scale(1, 8'h22);
        write_scale(2, 8'h33);
        write_scale(31, 8'h38);  // 1.0

        // Expected values are pre-decoded 12-bit (via fp8_to_scaled12)
        // 8'h11 = fp8 with exp=2, mant=1 → pre-decoded = 9
        // 8'h22 = fp8 with exp=4, mant=2 → pre-decoded = 40
        // 8'h33 = fp8 with exp=6, mant=3 → pre-decoded = 176
        // 8'h38 = fp8 1.0 → pre-decoded = 256
        query(0,   0,  fp8_to_scaled12(8'h11));
        query(15,  0,  fp8_to_scaled12(8'h11));
        query(16,  1,  fp8_to_scaled12(8'h22));
        query(31,  1,  fp8_to_scaled12(8'h22));
        query(32,  2,  fp8_to_scaled12(8'h33));
        query(511, 31, fp8_to_scaled12(8'h38));

        $display("PASS tb_fp4_scale_reader");
        $finish;
    end
endmodule
