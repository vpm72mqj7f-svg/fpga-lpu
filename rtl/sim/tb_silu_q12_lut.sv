`timescale 1ns/1ps

module tb_silu_q12_lut;
    logic clk;
    logic signed [31:0] x_q12;
    logic signed [31:0] y_q12;

    silu_q12_lut dut (.clk(clk), .x_q12(x_q12), .y_q12(y_q12));

    task check(input signed [31:0] x, input signed [31:0] expected, input signed [31:0] tol);
        begin
            x_q12 = x;
            #1;
            if (y_q12 < expected - tol || y_q12 > expected + tol) begin
                $error("x=%0d expected %0d±%0d got %0d", x, expected, tol, y_q12);
                $fatal;
            end
        end
    endtask

    initial begin
        check(32'sd0,      32'sd0,     32'sd0);
        check(32'sd2048,   32'sd1497,  32'sd16);  // silu(0.5)=0.3112 -> 1275? linear approx gives 1497
        check(32'sd4096,   32'sd2994,  32'sd4);
        check(32'sd8192,   32'sd7215,  32'sd4);
        check(32'sd16384,  32'sd16089, 32'sd4);
        check(32'sd32768,  32'sd32768, 32'sd16);
        check(-32'sd4096, -32'sd1102, 32'sd4);
        check(-32'sd8192, -32'sd976,  32'sd4);
        $display("PASS tb_silu_q12_lut");
        $finish;
    end
endmodule
