module test_timing;
    reg clk = 0;
    always #5 clk = ~clk;
    reg sig = 0;
    initial begin
        $display("t=%0t: start", $time);
        repeat (3) @(posedge clk);
        $display("t=%0t: after 3 posedges", $time);
        sig = 1;
        @(posedge clk);
        $display("t=%0t: after sig=1", $time);
        sig = 0;
        repeat (2) @(posedge clk);
        $display("t=%0t: done", $time);
        $finish;
    end
    always_ff @(posedge clk) begin
        if (sig)
            $display("t=%0t: sig=1 captured", $time);
    end
endmodule
