`timescale 1ns/1ps

module tb_rms_norm;
    localparam int HIDDEN = 8;
    logic clk, rst_n;
    logic valid_in, valid_out;
    logic [HIDDEN*32-1:0] x_flat;
    logic [HIDDEN*32-1:0] g_flat;
    logic [HIDDEN*32-1:0] y_flat;

    rms_norm dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 0;
        valid_in = 0;
        x_flat = '0;
        g_flat = '0;
        for (int i = 0; i < HIDDEN; i++) g_flat[i*32+:32] = 4096;
        repeat (4) @(posedge clk);
        rst_n = 1;

        // Test: all inputs 4096, gamma 4096 → output 4096
        for (int i = 0; i < HIDDEN; i++) x_flat[i*32+:32] = 4096;
        @(posedge clk);
        #1; valid_in <= 1;
        @(posedge clk);
        #1; valid_in <= 0;

        for (int cyc = 0; cyc < 20; cyc++) begin
            @(posedge clk);
            if (valid_out) begin
                #1;
                for (int i = 0; i < HIDDEN; i++) begin
                    if (y_flat[i*32+:32] != 32'sd4096) begin
                        $error("y[%0d] = %0d, expected 4096", i, y_flat[i*32+:32]);
                        $fatal;
                    end
                end
                $display("PASS tb_rms_norm (identity case)");
                $finish;
            end
        end
        $error("timeout");
        $fatal;
    end
endmodule
