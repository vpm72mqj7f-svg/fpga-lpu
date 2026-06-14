// gemv_test_top.sv — Standalone GEMV array DSP test
module gemv_test_top(input clk, input rst_n, output [3:0] led);
    wire gemv_busy, gemv_done, gemv_result_valid;
    wire [4095:0] gemv_activ, gemv_weight;
    wire [23:0] gemv_result;
    wire [10:0] gemv_row;
    wire gemv_last;
    wire [3:0] gemv_fsm;
    wire [9:0] gemv_cycle;
    wire [23:0] gemv_dbg;

    reg [7:0] ramp;
    always_ff @(posedge clk) ramp <= ramp + 1;
    assign gemv_activ = {512{ramp}};
    assign gemv_weight = {512{~ramp}};

    ffn_gemv_array #(.DSP_LANES(512),.INPUT_DIM(2048),.OUTPUT_DIM(1408)) u_gemv(
        .clk,.rst_n,.start(1'b1),.busy(gemv_busy),.done(gemv_done),
        .activ_valid(1'b1),.activ_ready(),.activ_data(gemv_activ),
        .weight_valid(1'b1),.weight_ready(),.weight_data(gemv_weight),
        .wt_preload_req(),.wt_preload_row(),.wt_preload_ack(1'b1),
        .result_valid(gemv_result_valid),.result_ready(1'b1),
        .result_data(gemv_result),.result_row(gemv_row),.result_last(gemv_last),
        .mode_prefill(1'b0),.prefill_tokens(6'd1),
        .dbg_fsm(gemv_fsm),.dbg_cycle(gemv_cycle),.dbg_reduced_out(gemv_dbg)
    );

    // Crucial: XOR ALL output data bits into LEDs to prevent DSP optimization
    wire led0 = gemv_done ^ gemv_busy ^ gemv_result_valid ^ gemv_fsm[0];
    wire led1 = ^gemv_result;        // 24-bit XOR of result
    wire led2 = ^gemv_dbg;           // 24-bit XOR of reduced output
    wire led3 = ^gemv_activ[127:0];  // partial activ XOR
    assign led = {led3, led2, led1, led0};
endmodule
