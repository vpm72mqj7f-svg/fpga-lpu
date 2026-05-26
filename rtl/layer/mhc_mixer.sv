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

    // Registered products (from previous cycle's multiply)
    logic signed [DATA_W+COEFF_W-1:0] prod_l [HIDDEN];
    logic signed [DATA_W+COEFF_W-1:0] prod_r [HIDDEN];

    // Current highway index
    logic [HW_BITS-1:0] hw_idx;
    logic               last_hw;

    // Combinational: sum = (prod_l >> 12) + (prod_r >> 12) for output write
    logic signed [DATA_W-1:0] sum_c [HIDDEN];

    assign in_ready = (state == S_IDLE);
    assign last_hw  = (hw_idx == (N_HW - 1));

    // Combinational sum from registered products
    for (genvar d = 0; d < HIDDEN; d++) begin : gen_sum
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
            for (int d = 0; d < HIDDEN; d++) begin
                prod_l[d] <= '0;
                prod_r[d] <= '0;
            end
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
                    // Compute products for highway 0 (registered, used in S_RUN)
                    for (int d = 0; d < HIDDEN; d++) begin
                        prod_l[d] <= $signed(layer_r[d*DATA_W +: DATA_W]) * mix_a[0];
                        prod_r[d] <= $signed(resid_r[d*DATA_W +: DATA_W]) * mix_b[0];
                    end
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
                        // Compute products for next highway
                        hw_idx <= hw_idx + 1'b1;
                        for (int d = 0; d < HIDDEN; d++) begin
                            prod_l[d] <= $signed(layer_r[d*DATA_W +: DATA_W]) * mix_a[hw_idx + 1'b1];
                            prod_r[d] <= $signed(resid_r[d*DATA_W +: DATA_W]) * mix_b[hw_idx + 1'b1];
                        end
                    end
                end

                S_DONE: begin
                    // Write last highway result (from products computed in final S_RUN)
                    for (int d = 0; d < HIDDEN; d++) begin
                        highway_flat[(hw_idx*HIDDEN + d)*DATA_W +: DATA_W] <= sum_c[d];
                    end
                    out_valid <= 1'b1;
                    state     <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
