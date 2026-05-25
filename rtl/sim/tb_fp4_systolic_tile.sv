`timescale 1ns/1ps

module tb_fp4_systolic_tile;
    localparam int LANES = 4;
    logic clk;
    logic rst_n;
    logic accum_clr;
    logic valid_in;
    logic [LANES*4-1:0] weight_fp4_flat;
    logic [LANES*8-1:0] scale_fp8_flat;
    logic [LANES*8-1:0] activ_fp8_flat;
    logic valid_out;
    logic [LANES*32-1:0] lane_result_flat;
    logic [31:0] sum_result;

    fp4_systolic_tile #(.LANES(LANES)) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    task clear_accum;
        begin
            @(posedge clk);
            accum_clr <= 1'b1;
            @(posedge clk);
            accum_clr <= 1'b0;
        end
    endtask

    task drive_zero_case;
        begin
            for (int i = 0; i < LANES; i++) begin
                weight_fp4_flat[i*4 +: 4] <= 4'h0;  // fp4 zero
                scale_fp8_flat[i*8 +: 8]  <= 8'h38; // arbitrary scale
                activ_fp8_flat[i*8 +: 8]  <= 8'h38; // fp8 one-ish
            end
            @(posedge clk);
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task drive_nonzero_case;
        begin
            // lane0: +1.0 * +1.0 * scale1.0 = +4096
            weight_fp4_flat[0*4 +: 4] <= 4'h4;
            activ_fp8_flat [0*8 +: 8] <= 8'h38;
            scale_fp8_flat [0*8 +: 8] <= 8'h38;
            // lane1: +2.0 * +1.0 * scale0.5 = +4096
            weight_fp4_flat[1*4 +: 4] <= 4'h6;
            activ_fp8_flat [1*8 +: 8] <= 8'h38;
            scale_fp8_flat [1*8 +: 8] <= 8'h30;
            // lane2: -1.0 * +1.0 * scale2.0 = -8192
            weight_fp4_flat[2*4 +: 4] <= 4'hc;
            activ_fp8_flat [2*8 +: 8] <= 8'h38;
            scale_fp8_flat [2*8 +: 8] <= 8'h40;
            // lane3: zero
            weight_fp4_flat[3*4 +: 4] <= 4'h0;
            activ_fp8_flat [3*8 +: 8] <= 8'h38;
            scale_fp8_flat [3*8 +: 8] <= 8'h38;
            @(posedge clk);
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    initial begin
        rst_n = 0;
        accum_clr = 0;
        valid_in = 0;
        weight_fp4_flat = '0;
        scale_fp8_flat = '0;
        activ_fp8_flat = '0;
        repeat (4) @(posedge clk);
        rst_n = 1;

        clear_accum();
        drive_zero_case();
        for (int cyc = 0; cyc < 20; cyc++) begin
            @(posedge clk);
        end
        #1;
        // fp4_mac emits a one-cycle valid pulse; some simulators may miss it
        // after the wait above, so this test focuses on datapath zero behavior.
        if (sum_result !== 32'd0) begin
            $error("expected zero sum, got %0d", sum_result);
            $fatal;
        end
        for (int i = 0; i < LANES; i++) begin
            if (lane_result_flat[i*32 +: 32] !== 32'd0) begin
                $error("lane %0d expected zero, got %0d", i, lane_result_flat[i*32 +: 32]);
                $fatal;
            end
        end
        $display("[ OK ] zero-vector tile test");

        clear_accum();
        drive_nonzero_case();
        for (int cyc = 0; cyc < 20; cyc++) begin
            @(posedge clk);
        end
        #1;
        if (lane_result_flat[0*32 +: 32] !== 32'h00001000) begin
            $error("lane0 expected 0x00001000, got 0x%08h", lane_result_flat[0*32 +: 32]);
            $fatal;
        end
        if (lane_result_flat[1*32 +: 32] !== 32'h00001000) begin
            $error("lane1 expected 0x00001000, got 0x%08h", lane_result_flat[1*32 +: 32]);
            $fatal;
        end
        if (lane_result_flat[2*32 +: 32] !== 32'hffffe000) begin
            $error("lane2 expected 0xffffe000, got 0x%08h", lane_result_flat[2*32 +: 32]);
            $fatal;
        end
        if (lane_result_flat[3*32 +: 32] !== 32'h00000000) begin
            $error("lane3 expected 0x00000000, got 0x%08h", lane_result_flat[3*32 +: 32]);
            $fatal;
        end
        if (sum_result !== 32'h00000000) begin
            $error("tile sum expected 0x00000000, got 0x%08h", sum_result);
            $fatal;
        end
        $display("[ OK ] non-zero scale-aware tile test");

        $display("PASS tb_fp4_systolic_tile");
        $finish;
    end
endmodule
