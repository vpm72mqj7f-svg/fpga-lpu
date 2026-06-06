//=============================================================================
// mla_qkv_proj.sv — MLA low-rank Q/K/V projection engine
//
// Computes Q, K, V from hidden state using low-rank factorized projections:
//   Q = hidden × W_Q             (dense, HIDDEN → HIDDEN)
//   K_latent = hidden × W_K      (compress, HIDDEN → K_LATENT)
//   K = K_latent × W_K_up        (decompress, K_LATENT → HIDDEN)
//   V_latent = hidden × W_V      (compress, HIDDEN → V_LATENT)
//   V = V_latent × W_V_up        (decompress, V_LATENT → HIDDEN)
//
// One dot product per cycle, all input dims computed in parallel.
// Total latency: HIDDEN + K_LATENT + V_LATENT + HIDDEN + HIDDEN + 2 = ~34 cycles
//=============================================================================

module mla_qkv_proj #(
    parameter int HIDDEN    = 8,
    parameter int K_LATENT  = 4,
    parameter int V_LATENT  = 4,
    parameter int WEIGHT_W  = 16,
    parameter int DATA_W    = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Hidden state input
    input  logic                         in_valid,
    input  logic [HIDDEN*DATA_W-1:0]     hidden_flat,
    output logic                         in_ready,

    // Weight load port
    input  logic                         wt_wr_en,
    input  logic [2:0]                   wt_sel,     // 0=Q, 1=K, 2=K_up, 3=V, 4=V_up
    input  logic [$clog2(HIDDEN)-1:0]    wt_row,
    input  logic [$clog2(HIDDEN)-1:0]    wt_col,
    input  logic signed [WEIGHT_W-1:0]   wt_wr_data,

    // Q/K/V output (flattened)
    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [HIDDEN*DATA_W-1:0]     Q_flat,
    output logic [HIDDEN*DATA_W-1:0]     K_flat,
    output logic [HIDDEN*DATA_W-1:0]     V_flat,
    output logic [K_LATENT*DATA_W-1:0]   K_latent_flat,
    output logic [V_LATENT*DATA_W-1:0]   V_latent_flat
);

    localparam int DIM_BITS   = $clog2(HIDDEN);
    localparam int KL_BITS    = $clog2(K_LATENT);
    localparam int VL_BITS    = $clog2(V_LATENT);
    localparam int MAX_DIM    = HIDDEN;  // max of HIDDEN, K_LATENT, V_LATENT

    // Weight storage: use 5 separate 2D arrays
    logic signed [WEIGHT_W-1:0] W_Q  [HIDDEN][HIDDEN];
    logic signed [WEIGHT_W-1:0] W_K  [HIDDEN][K_LATENT];
    logic signed [WEIGHT_W-1:0] W_Ku [K_LATENT][HIDDEN];
    logic signed [WEIGHT_W-1:0] W_V  [HIDDEN][V_LATENT];
    logic signed [WEIGHT_W-1:0] W_Vu [V_LATENT][HIDDEN];

    // Pipeline
    typedef enum logic [3:0] {
        S_IDLE, S_Q, S_K, S_V, S_K_UP, S_V_UP, S_FLAT, S_OUTPUT
    } state_t;
    state_t state;

    // Registered inputs
    logic signed [DATA_W-1:0] hidden_r [HIDDEN];
    logic signed [DATA_W-1:0] K_lat_r  [K_LATENT];
    logic signed [DATA_W-1:0] V_lat_r  [V_LATENT];

    // Output dim iterator
    logic [DIM_BITS-1:0] out_idx;           // current output dimension

    // Output accumulators
    logic signed [DATA_W-1:0] Q_r [HIDDEN];
    logic signed [DATA_W-1:0] K_r [HIDDEN];
    logic signed [DATA_W-1:0] V_r [HIDDEN];

    // Combinational dot product for current out_idx — parameterized for-loop
    logic signed [DATA_W-1:0] dot_product;
    always_comb begin
        dot_product = '0;
        case (state)
            S_Q: for (int i = 0; i < HIDDEN; i++)
                     dot_product += ($signed(hidden_r[i]) * $signed(W_Q[i][out_idx]) >>> 12);
            S_K: for (int i = 0; i < HIDDEN; i++)
                     dot_product += ($signed(hidden_r[i]) * $signed(W_K[i][out_idx]) >>> 12);
            S_V: for (int i = 0; i < HIDDEN; i++)
                     dot_product += ($signed(hidden_r[i]) * $signed(W_V[i][out_idx]) >>> 12);
            S_K_UP: for (int i = 0; i < K_LATENT; i++)
                         dot_product += ($signed(K_lat_r[i]) * $signed(W_Ku[i][out_idx]) >>> 12);
            S_V_UP: for (int i = 0; i < V_LATENT; i++)
                         dot_product += ($signed(V_lat_r[i]) * $signed(W_Vu[i][out_idx]) >>> 12);
            default: dot_product = '0;
        endcase
    end

    assign in_ready = (state == S_IDLE);

    // Weight storage — survives reset (production: altera_syncram BRAM)
    // synthesis translate_off
    initial begin
        for (int r = 0; r < HIDDEN; r++) begin
            for (int c = 0; c < HIDDEN; c++)     W_Q[r][c]  = '0;
            for (int c = 0; c < K_LATENT; c++)   W_K[r][c]  = '0;
            for (int c = 0; c < V_LATENT; c++)   W_V[r][c]  = '0;
        end
        for (int r = 0; r < K_LATENT; r++)
            for (int c = 0; c < HIDDEN; c++)     W_Ku[r][c] = '0;
        for (int r = 0; r < V_LATENT; r++)
            for (int c = 0; c < HIDDEN; c++)     W_Vu[r][c] = '0;
    end
    // synthesis translate_on

    always_ff @(posedge clk) begin
        if (wt_wr_en) begin
            case (wt_sel)
                0: W_Q [wt_row][wt_col] <= wt_wr_data;
                1: W_K [wt_row][wt_col] <= wt_wr_data;
                2: W_Ku[wt_row][wt_col] <= wt_wr_data;
                3: W_V [wt_row][wt_col] <= wt_wr_data;
                4: W_Vu[wt_row][wt_col] <= wt_wr_data;
                default: ;
            endcase
        end
    end

    // Main FSM
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            out_idx  <= '0;
            out_valid <= 1'b0;
            Q_flat   <= '0; K_flat <= '0; V_flat <= '0;
            K_latent_flat <= '0; V_latent_flat <= '0;
            for (int d = 0; d < HIDDEN; d++) begin
                hidden_r[d] <= '0; Q_r[d] <= '0; K_r[d] <= '0; V_r[d] <= '0;
            end
            for (int d = 0; d < K_LATENT; d++) K_lat_r[d] <= '0;
            for (int d = 0; d < V_LATENT; d++) V_lat_r[d] <= '0;
        end else begin
            out_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (in_valid) begin
                        for (int d = 0; d < HIDDEN; d++)
                            hidden_r[d] <= $signed(hidden_flat[d*DATA_W +: DATA_W]);
                        out_idx <= '0;
                        state <= S_Q;
                    end
                end

                S_Q: begin
                    Q_r[out_idx] <= dot_product;
                    if (out_idx == (HIDDEN - 1)) begin
                        out_idx <= '0;
                        state <= S_K;
                    end else begin
                        out_idx <= out_idx + 1'b1;
                    end
                end

                S_K: begin
                    K_lat_r[out_idx[KL_BITS-1:0]] <= dot_product;
                    if (out_idx == (K_LATENT - 1)) begin
                        out_idx <= '0;
                        state <= S_V;
                    end else begin
                        out_idx <= out_idx + 1'b1;
                    end
                end

                S_V: begin
                    V_lat_r[out_idx[VL_BITS-1:0]] <= dot_product;
                    if (out_idx == (V_LATENT - 1)) begin
                        out_idx <= '0;
                        state <= S_K_UP;
                    end else begin
                        out_idx <= out_idx + 1'b1;
                    end
                end

                S_K_UP: begin
                    K_r[out_idx] <= dot_product;
                    if (out_idx == (HIDDEN - 1)) begin
                        out_idx <= '0;
                        state <= S_V_UP;
                    end else begin
                        out_idx <= out_idx + 1'b1;
                    end
                end

                S_V_UP: begin
                    V_r[out_idx] <= dot_product;
                    if (out_idx == (HIDDEN - 1)) begin
                        state <= S_FLAT;
                    end else begin
                        out_idx <= out_idx + 1'b1;
                    end
                end

                S_FLAT: begin
                    for (int d = 0; d < HIDDEN; d++) begin
                        Q_flat[d*DATA_W +: DATA_W] <= Q_r[d];
                        K_flat[d*DATA_W +: DATA_W] <= K_r[d];
                        V_flat[d*DATA_W +: DATA_W] <= V_r[d];
                    end
                    for (int d = 0; d < K_LATENT; d++)
                        K_latent_flat[d*DATA_W +: DATA_W] <= K_lat_r[d];
                    for (int d = 0; d < V_LATENT; d++)
                        V_latent_flat[d*DATA_W +: DATA_W] <= V_lat_r[d];
                    out_valid <= 1'b1;
                    state <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    if (out_ready) begin
                        state <= S_IDLE;
                        out_valid <= 1'b0;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
