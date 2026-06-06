//=============================================================================
// mhc_mixer.sv — manifold-constrained hyper-connection mixing matrix
//
// For each highway h and hidden dim d:
//   highway[h*HIDDEN + d] = mix[h][0] * layer_in[d] + mix[h][1] * residual[d]
//
// Mixing coefficients are Q12 fixed-point, pre-loaded via config port.
// Pipeline: 1 warmup + N_HW run + 1 done = N_HW+2 cycles total.
// DSP: one 18×19 multiply per hidden dim per cycle (Agilex 7 M-Series).
//=============================================================================

(* altera_attribute = "-name DSP_BLOCK_BALANCING AUTO" *)
module mhc_mixer #(
    parameter int HIDDEN    = 8,
    parameter int N_HW      = 4,
    parameter int COEFF_W   = 16,
    parameter int DATA_W    = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Data input (flattened: HIDDEN × DATA_W bits)
    input  logic                         in_valid,
    input  logic [HIDDEN*DATA_W-1:0]     layer_in_flat,
    input  logic [HIDDEN*DATA_W-1:0]     residual_flat,
    output logic                         in_ready,

    // Data output (flattened: HIDDEN × N_HW × DATA_W bits)
    output logic                         out_valid,
    output logic [HIDDEN*N_HW*DATA_W-1:0] highway_flat,

    // Coefficient load port
    input  logic                         coeff_wr_en,
    input  logic [$clog2(N_HW)-1:0]      coeff_hw_id,
    input  logic [1:0]                   coeff_col,
    input  logic signed [COEFF_W-1:0]    coeff_wr_data
);

    localparam int HW_BITS = $clog2(N_HW);

    // Mixing matrix: N_HW × 2, Q12 coefficients (split for Icarus)
    logic signed [COEFF_W-1:0] mix_a [N_HW];  // layer weight
    logic signed [COEFF_W-1:0] mix_b [N_HW];  // residual weight

    // Pipeline state
    typedef enum logic [1:0] { S_IDLE, S_WARMUP, S_RUN, S_DONE } state_t;
    state_t state;

    // Registered inputs
    logic [HIDDEN*DATA_W-1:0] layer_r;
    logic [HIDDEN*DATA_W-1:0] resid_r;

    // Current highway index
    logic [HW_BITS-1:0] hw_idx;
    logic               last_hw;

    assign in_ready = (state == S_IDLE);
    assign last_hw  = (hw_idx == (N_HW - 1));

    // ── DSP: altera_mult_add IP instances (2 × HIDDEN multipliers) ──
    // Coefficient mux: select mix_a/mix_b[hw_idx+1] or [0] based on state
    logic signed [COEFF_W-1:0] coeff_a_sel;
    logic signed [COEFF_W-1:0] coeff_b_sel;
    logic [HW_BITS-1:0]        next_hw_idx;

    assign next_hw_idx = hw_idx + 1'b1;

    always_comb begin
        if (state == S_WARMUP) begin
            coeff_a_sel = mix_a[0];
            coeff_b_sel = mix_b[0];
        end else if (last_hw) begin
            // Hold current coefficients; next_hw_idx would wrap and select mix[0]
            coeff_a_sel = mix_a[hw_idx];
            coeff_b_sel = mix_b[hw_idx];
        end else begin
            coeff_a_sel = mix_a[next_hw_idx];
            coeff_b_sel = mix_b[next_hw_idx];
        end
    end

    // Registered products and their sum (combinational from registered products)
    logic signed [DATA_W+COEFF_W-1:0] prod_l [HIDDEN];
    logic signed [DATA_W+COEFF_W-1:0] prod_r [HIDDEN];
    logic signed [DATA_W-1:0]        sum_c    [HIDDEN];

    for (genvar d = 0; d < HIDDEN; d++) begin : gen_dsp
        wire signed [DATA_W+COEFF_W-1:0] mul_l, mul_r;

        altera_mult_add #(.A_WIDTH(DATA_W), .B_WIDTH(COEFF_W), .PIPE_STAGES(0))
        u_mul_l (.clock(clk),
            .a($signed(layer_r[d*DATA_W +: DATA_W])),
            .b(coeff_a_sel), .result(mul_l));

        altera_mult_add #(.A_WIDTH(DATA_W), .B_WIDTH(COEFF_W), .PIPE_STAGES(0))
        u_mul_r (.clock(clk),
            .a($signed(resid_r[d*DATA_W +: DATA_W])),
            .b(coeff_b_sel), .result(mul_r));

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                prod_l[d] <= '0;
                prod_r[d] <= '0;
            end else if (state == S_WARMUP || state == S_RUN) begin
                prod_l[d] <= mul_l;
                prod_r[d] <= mul_r;
            end
        end

        assign sum_c[d] = (prod_l[d] >>> 12) + (prod_r[d] >>> 12);
    end

    // Coefficient write port
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int h = 0; h < N_HW; h++) begin
                mix_a[h] <= '0;
                mix_b[h] <= '0;
            end
        end else if (coeff_wr_en) begin
            if (coeff_col == 1'b0)
                mix_a[coeff_hw_id] <= coeff_wr_data;
            else
                mix_b[coeff_hw_id] <= coeff_wr_data;
        end
    end

    // Main FSM + datapath
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            hw_idx      <= '0;
            layer_r     <= '0;
            resid_r     <= '0;
            out_valid   <= 1'b0;
            highway_flat <= '0;
        end else begin
            out_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (in_valid) begin
                        layer_r <= layer_in_flat;
                        resid_r <= residual_flat;
                        hw_idx  <= '0;
                        state   <= S_WARMUP;
                    end
                end

                S_WARMUP: begin
                    // Products computed by altera_mult_add + registered above
                    hw_idx <= '0;
                    state  <= S_RUN;
                end

                S_RUN: begin
                    // Write current highway result (from products computed last cycle)
                    for (int d = 0; d < HIDDEN; d++) begin
                        highway_flat[(hw_idx*HIDDEN + d)*DATA_W +: DATA_W] <= sum_c[d];
                    end

                    if (last_hw) begin
                        state <= S_DONE;
                    end else begin
                        hw_idx <= hw_idx + 1'b1;
                    end
                end

                S_DONE: begin
                    // Last highway was already written in final S_RUN cycle.
                    // (Writing again here would use stale products from the
                    //  coefficient-wrap bug — removed to avoid overwrite.)
                    out_valid <= 1'b1;
                    state     <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
