`timescale 1ns/1ps

module tb_rms_norm;
    localparam int HIDDEN = 8;
    logic clk, rst_n;
    logic valid_in, valid_out;
    logic signed [31:0] x0, x1, x2, x3, x4, x5, x6, x7;
    logic signed [31:0] g0, g1, g2, g3, g4, g5, g6, g7;
    logic signed [31:0] y0, y1, y2, y3, y4, y5, y6, y7;

    rms_norm dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 0;
        valid_in = 0;
        {x0,x1,x2,x3,x4,x5,x6,x7} = '0;
        {g0,g1,g2,g3,g4,g5,g6,g7} = '0;
        g0 = 4096; g1 = 4096; g2 = 4096; g3 = 4096;
        g4 = 4096; g5 = 4096; g6 = 4096; g7 = 4096;
        repeat (4) @(posedge clk);
        rst_n = 1;

        // Test: all inputs 4096, gamma 4096 → output 4096
        x0 = 4096; x1 = 4096; x2 = 4096; x3 = 4096;
        x4 = 4096; x5 = 4096; x6 = 4096; x7 = 4096;
        @(posedge clk);
        #1; valid_in = 1;
        @(posedge clk);
        #1; valid_in = 0;

        for (int cyc = 0; cyc < 20; cyc++) begin
            @(posedge clk);
            if (valid_out) begin
                #1;
                if (y0 != 32'sd4096 || y1 != 32'sd4096 || y2 != 32'sd4096 || y3 != 32'sd4096 ||
                    y4 != 32'sd4096 || y5 != 32'sd4096 || y6 != 32'sd4096 || y7 != 32'sd4096) begin
                    $error("got %0d %0d %0d %0d %0d %0d %0d %0d",
                           y0, y1, y2, y3, y4, y5, y6, y7);
                    $fatal;
                end
                $display("PASS tb_rms_norm (identity case)");
                $finish;
            end
        end
        $error("timeout");
        $fatal;
    end
endmodule
