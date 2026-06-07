// =============================================================================
// led_controller.sv — User LED Driver for Debug Status Display
//
// Drives 4 user LEDs (active-low on DK-DEV-1SMX-H-A).
// Encoding:
//   PLL locked     → LED[0] blinks at 2 Hz
//   FFN busy       → LED[1] on
//   FFN done       → LED[2] on (brief pulse)
//   Bring-up state → LED[3] encodes pass/fail
// =============================================================================

module led_controller #(
    parameter int CLK_FREQ_HZ = 100_000_000    // 100 MHz system clock
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        pll_locked,
    input  logic        ffn_busy,
    input  logic        ffn_done,
    input  logic [3:0]  bringup_state,          // bring-up FSM state
    output logic [3:0]  led                     // Active low: LED on = logic 0
);

    // =========================================================================
    // Heartbeat Timer — 2 Hz toggle (~25M cycles @ 100 MHz)
    // =========================================================================
    localparam int HEARTBEAT_HALF = CLK_FREQ_HZ / 4;   // 0.25s = 2.5 Hz toggle
    localparam int DONE_PULSE_LEN = CLK_FREQ_HZ / 10;  // 0.1s done pulse

    logic [$clog2(HEARTBEAT_HALF)-1:0] hb_cnt;
    logic                               hb_toggle;
    logic [$clog2(DONE_PULSE_LEN)-1:0]  done_cnt;
    logic                               done_pulse;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hb_cnt    <= '0;
            hb_toggle <= 1'b0;
            done_cnt  <= '0;
            done_pulse <= 1'b0;
        end else begin
            // Heartbeat counter
            if (hb_cnt == HEARTBEAT_HALF - 1) begin
                hb_cnt    <= '0;
                hb_toggle <= ~hb_toggle;
            end else begin
                hb_cnt <= hb_cnt + 1;
            end

            // Done pulse (stretched)
            if (ffn_done) begin
                done_cnt  <= '0;
                done_pulse <= 1'b1;
            end else if (done_pulse) begin
                if (done_cnt == DONE_PULSE_LEN - 1) begin
                    done_cnt  <= '0;
                    done_pulse <= 1'b0;
                end else begin
                    done_cnt <= done_cnt + 1;
                end
            end
        end
    end

    // =========================================================================
    // LED Encoding (Active Low: ON = 0)
    // =========================================================================
    // led[0]: PLL lock heartbeat     — blinks when locked, off when unlocked
    // led[1]: FFN busy               — on during FFN compute
    // led[2]: FFN done               — on for 0.1s after completion
    // led[3]: Bring-up result        — off=pass, on=fail
    //
    // Bring-up state encoding for led[3]:
    //   B_PASS (4'h6) → LED off (pass)
    //   B_FAIL (4'h7) → LED solid on (fail)
    //   others        → LED blinks

    logic led0, led1, led2, led3;

    assign led0 = pll_locked ? ~hb_toggle : 1'b1;    // blink when locked
    assign led1 = ~ffn_busy;                          // on during FFN busy
    assign led2 = ~done_pulse;                        // pulse on done

    always_comb begin
        case (bringup_state)
            4'h6:       led3 = 1'b1;   // B_PASS: LED off (logic 1)
            4'h7:       led3 = 1'b0;   // B_FAIL: LED on (logic 0)
            default:    led3 = hb_toggle ? 1'b0 : 1'b1;  // blink during others
        endcase
    end

    assign led = {led3, led2, led1, led0};

endmodule
