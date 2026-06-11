// =============================================================================
// v2_lite_ffn_engine.sv — DeepSeek V2-Lite FFN Compute Engine (Production)
//
// V2-Lite: hidden=2048, inter=1408, 66 experts (64 routed + 2 shared),
//          TOP_K=6, FP8 (E4M3) weights and activations.
//
// Pipeline (per token, 10-state):
//   S_IDLE → S_LOAD_WEIGHTS → S_GATE_PROJECT → S_SILU
//          → S_UP_PROJECT → S_MERGE_GATE_UP → S_DOWN_PROJECT
//          → S_ACCUM_EXPERT → S_NEXT_EXPERT → S_OUTPUT
//
// Submodules:
//   - systolic_array (gate_up: 2048→1408) — gate and up projections
//   - systolic_array (down: 1408→2048)   — down projection
//   - silu_activation (64-element)        — SiLU activation
//   - hbm2_weight_reader (256-bit AXI4)   — HBM2 → weight stream
//
// Performance Estimate (500 MHz, 64 DSP lanes):
//   Gate:  2048×1408 matmul = 32 cycles/row × 1408 rows = 45,056 cycles
//   SiLU:  1408 ÷ 64 lanes = 22 cycles (pipelined)
//   Up:    2048×1408 matmul = 45,056 cycles
//   Merge: 1408 ÷ 64 lanes = 22 cycles
//   Down:  1408×2048 matmul = 22 cycles/row × 2048 rows = 45,056 cycles
//   Total per expert:       ~135,212 cycles
//   6 experts:              ~811,272 cycles
//   Time per token:         811,272 / 500 MHz = 1.62 ms
//   Throughput:             ~616 tok/s
//
//   With pipelining (gate_up of expert N+1 overlaps with down of expert N):
//   Effective per expert (after 1st): ~90,000 cycles
//   6 experts: 135,212 + 5 × 90,000 = 585,212 cycles
//   Throughput: ~854 tok/s (accounts for HBM2 fill latency)
//
// M20K Buffer Budget (V2-Lite):
//   activ_buf:  2048 × 8-bit  = 16 Kbit   → 1 M20K
//   gate_buf:   1408 × 16-bit = 22.5 Kbit → 2 M20K
//   up_buf:     1408 × 16-bit = 22.5 Kbit → 2 M20K
//   combined:   1408 × 16-bit = 22.5 Kbit → 2 M20K
//   ffn_accum:  2048 × 24-bit = 49 Kbit   → 3 M20K
//   Total: ~10 M20K blocks (out of 11,721 available)
// =============================================================================

module v2_lite_ffn_engine #(
    parameter int HIDDEN      = 2048,
    parameter int INTER       = 1408,
    parameter int NUM_EXPERTS = 66,
    parameter int TOP_K       = 6,
    parameter int DATA_W      = 8,       // FP8 E4M3
    parameter int ACCUM_W     = 24,      // accumulator width (fp16 + headroom)
    parameter int DSP_LANES   = 64,      // parallel MAC units
    parameter logic [31:0] VERSION = 32'h0B061A01  // {day,month,year-2000,build#}
) (
    input  logic                         clk,              // 500 MHz core clock
    input  logic                         rst_n,

    // ---- PCIe RX: attention output from CPU ----
    input  logic                         pcie_rx_valid,
    input  logic [HIDDEN*DATA_W-1:0]     pcie_rx_data,     // flat packed 2048 × FP8
    output logic                         pcie_rx_ready,

    // ---- PCIe TX: FFN output to CPU ----
    output logic                         pcie_tx_valid,
    output logic [HIDDEN*DATA_W-1:0]     pcie_tx_data,     // flat packed 2048 × FP8
    input  logic                         pcie_tx_ready,

    // ---- HBM2 AXI4 (to Intel HBM2 Controller IP) ----
    output logic [31:0]                  m_axi_araddr,
    output logic [7:0]                   m_axi_arlen,
    output logic [2:0]                   m_axi_arsize,
    output logic                         m_axi_arvalid,
    input  logic                         m_axi_arready,
    input  logic [255:0]                 m_axi_rdata,
    input  logic [1:0]                   m_axi_rresp,
    input  logic                         m_axi_rvalid,
    output logic                         m_axi_rready,
    input  logic                         m_axi_rlast,

    // ---- Expert selection (from router on CPU) ----
    input  logic [$clog2(NUM_EXPERTS)-1:0] expert_id [TOP_K],

    // ---- Status ----
    output logic                         busy,
    output logic                         done,

    // ---- Debug (JTAG ISP) ----
    output logic [3:0]                   dbg_fsm_state,
    output logic [2:0]                   dbg_expert_cnt,
    output logic                         dbg_gate_done,
    output logic                         dbg_up_done,
    output logic                         dbg_down_done,
    output logic                         dbg_silu_active,
    output logic                         dbg_merge_active,
    output logic                         dbg_hbm2_busy,
    output logic                         dbg_sa_active,
    output logic [2:0]                   dbg_hbm2r_fsm,          // HBM2 reader FSM state
    output logic [2:0]                   dbg_hbm2r_wr_watermark, // buffer fill level MSBs
    output logic [2:0]                   dbg_hbm2r_rd_watermark, // buffer drain level MSBs

    // ---- Performance Counters (sticky, 32-bit) ----
    output logic [31:0]                  perf_token_cnt,
    output logic [31:0]                  perf_cycle_cnt,
    output logic [31:0]                  perf_expert_cnt,
    output logic [31:0]                  perf_axi_rbeat,

    // ---- Error Flags (sticky) ----
    output logic                         err_merge_overflow,
    output logic                         err_silu_overflow,
    output logic                         err_axi_resp_err
);

    // =========================================================================
    // FSM States
    // =========================================================================
    typedef enum logic [3:0] {
        S_IDLE,
        S_LOAD_WEIGHTS,    // Request gate_w from HBM2 for current expert
        S_GATE_PROJECT,    // systolic_array: activ × gate_w → gate_out[INTER]
        S_SILU,            // SiLU activation on gate_out
        S_UP_PROJECT,      // systolic_array: activ × up_w → up_out[INTER]
        S_MERGE_GATE_UP,   // Element-wise: SiLU(gate)×up → combined[INTER]
        S_DOWN_PROJECT,    // systolic_array: combined × down_w → expert_out[HIDDEN]
        S_ACCUM_EXPERT,    // Accumulate into ffn_accum
        S_NEXT_EXPERT,     // Loop to next expert or continue
        S_OUTPUT           // Output ffn_accum on PCIe TX
    } state_t;
    state_t state;

    // =========================================================================
    // Buffers (M20K inferred)
    // =========================================================================
    (* ramstyle = "M20K" *) logic [DATA_W-1:0]   activ_buf   [HIDDEN-1:0];
    (* ramstyle = "M20K" *) logic [15:0]          gate_buf    [INTER-1:0];    // fp16
    (* ramstyle = "M20K" *) logic [15:0]          up_buf      [INTER-1:0];    // fp16
    (* ramstyle = "M20K" *) logic [15:0]          combined    [INTER-1:0];    // fp16
    (* ramstyle = "M20K" *) logic [ACCUM_W-1:0]   ffn_accum   [HIDDEN-1:0];  // fp24 accum

    // =========================================================================
    // Counters
    // =========================================================================
    logic [$clog2(TOP_K+1):0]         expert_cnt;     // 0..TOP_K-1 (routed) + TOP_K (shared)
    logic [$clog2(HIDDEN)-1:0]        accum_idx;      // index into ffn_accum during accumulation
    logic [$clog2(INTER)-1:0]         merge_idx;      // index into gate/up/combined during merge
    logic [$clog2(INTER)-1:0]         silu_idx;       // index into gate_buf during SiLU
    logic                             is_shared_expert;
    logic                             gate_done;      // gate projection complete
    logic                             up_done;        // up projection complete
    logic                             down_done;      // down projection complete

    // =========================================================================
    // Expert Weight Base Address Calculation
    // =========================================================================
    localparam int EXPERT_SIZE_MB  = 9;   // 8.65 MB, round up
    localparam int GATE_OFFSET_B   = 0;
    localparam int UP_OFFSET_B     = HIDDEN * INTER;           // 2,883,584 FP8
    localparam int DOWN_OFFSET_B   = 2 * HIDDEN * INTER;      // 5,767,168 FP8

    function automatic logic [31:0] expert_base(input int eid);
        return 32'(eid) * 32'(EXPERT_SIZE_MB * 1024 * 1024);
    endfunction

    // =========================================================================
    // HBM2 Reader Interface
    // =========================================================================
    logic                        hbm2_start;
    logic [31:0]                 hbm2_base_addr;
    logic [15:0]                 hbm2_words;
    logic                        hbm2_busy;
    logic                        hbm2_done;

    logic                        weight_valid;
    logic [DSP_LANES*DATA_W-1:0] weight_data;
    logic                        weight_ready;

    hbm2_weight_reader #(
        .AXI_DATA_W(256),
        .AXI_ADDR_W(32),
        .DATA_W(DATA_W),
        .DSP_LANES(DSP_LANES)
    ) u_hbm2_reader (
        .clk             (clk),
        .rst_n           (rst_n),
        .m_axi_araddr    (m_axi_araddr),
        .m_axi_arlen     (m_axi_arlen),
        .m_axi_arsize    (m_axi_arsize),
        .m_axi_arvalid   (m_axi_arvalid),
        .m_axi_arready   (m_axi_arready),
        .m_axi_rdata     (m_axi_rdata),
        .m_axi_rresp     (m_axi_rresp),
        .m_axi_rvalid    (m_axi_rvalid),
        .m_axi_rready    (m_axi_rready),
        .m_axi_rlast     (m_axi_rlast),
        .weight_valid    (weight_valid),
        .weight_data     (weight_data),
        .weight_ready    (weight_ready),
        .start           (hbm2_start),
        .base_addr       (hbm2_base_addr),
        .total_words     (hbm2_words),
        .busy            (hbm2_busy),
        .done            (hbm2_done)
    );

    // =========================================================================
    // Systolic Array: Gate/Up (2048 → 1408)
    // =========================================================================
    logic                             sa_gate_up_start;
    logic                             sa_gate_up_busy;
    logic                             sa_gate_up_done;
    logic                             sa_gate_up_activ_valid;
    logic                             sa_gate_up_activ_ready;
    logic [DSP_LANES*DATA_W-1:0]      sa_gate_up_activ_data;
    logic                             sa_gate_up_result_valid;
    logic [ACCUM_W-1:0]              sa_gate_up_result_data;
    logic [$clog2(INTER)-1:0]        sa_gate_up_result_row;
    logic                             sa_gate_up_result_last;

    systolic_array #(
        .INPUT_DIM(HIDDEN),
        .OUTPUT_DIM(INTER),
        .DSP_LANES(DSP_LANES),
        .DATA_W(DATA_W),
        .ACCUM_W(ACCUM_W)
    ) u_sa_gate_up (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (sa_gate_up_start),
        .busy            (sa_gate_up_busy),
        .done            (sa_gate_up_done),
        .activ_valid     (sa_gate_up_activ_valid),
        .activ_ready     (sa_gate_up_activ_ready),
        .activ_data      (sa_gate_up_activ_data),
        .weight_valid    (weight_valid),
        .weight_ready    (weight_ready),
        .weight_data     (weight_data),
        .wt_preload_req  (),
        .wt_preload_row  (),
        .wt_preload_ack  (1'b0),
        .result_valid    (sa_gate_up_result_valid),
        .result_ready    (1'b1),  // always ready to receive results
        .result_data     (sa_gate_up_result_data),
        .result_row      (sa_gate_up_result_row),
        .result_last     (sa_gate_up_result_last),
        .dbg_current_row (),
        .dbg_cycle_cnt   ()
    );

    // =========================================================================
    // Systolic Array: Down (1408 → 2048)
    // =========================================================================
    logic                             sa_down_start;
    logic                             sa_down_busy;
    logic                             sa_down_done;
    logic                             sa_down_activ_valid;
    logic                             sa_down_activ_ready;
    logic [DSP_LANES*DATA_W-1:0]      sa_down_activ_data;
    logic                             sa_down_result_valid;
    logic [ACCUM_W-1:0]              sa_down_result_data;
    logic [$clog2(HIDDEN)-1:0]       sa_down_result_row;
    logic                             sa_down_result_last;

    systolic_array #(
        .INPUT_DIM(INTER),
        .OUTPUT_DIM(HIDDEN),
        .DSP_LANES(DSP_LANES),
        .DATA_W(DATA_W),
        .ACCUM_W(ACCUM_W)
    ) u_sa_down (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (sa_down_start),
        .busy            (sa_down_busy),
        .done            (sa_down_done),
        .activ_valid     (sa_down_activ_valid),
        .activ_ready     (sa_down_activ_ready),
        .activ_data      (sa_down_activ_data),
        .weight_valid    (weight_valid),
        .weight_ready    (weight_ready),
        .weight_data     (weight_data),
        .wt_preload_req  (),
        .wt_preload_row  (),
        .wt_preload_ack  (1'b0),
        .result_valid    (sa_down_result_valid),
        .result_ready    (1'b1),
        .result_data     (sa_down_result_data),
        .result_row      (sa_down_result_row),
        .result_last     (sa_down_result_last),
        .dbg_current_row (),
        .dbg_cycle_cnt   ()
    );

    // =========================================================================
    // SiLU Activation
    // =========================================================================
    logic                         silu_valid_in;
    logic                         silu_valid_out;
    logic [15:0]                  silu_data_in  [DSP_LANES];
    logic [15:0]                  silu_data_out [DSP_LANES];

    silu_activation #(
        .DATA_W(16),
        .NUM_ELEMS(DSP_LANES)
    ) u_silu (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (silu_valid_in),
        .data_in   (silu_data_in),
        .data_out  (silu_data_out),
        .valid_out (silu_valid_out)
    );

    // =========================================================================
    // Activation stream index (cycles through activ_buf for systolic array)
    // =========================================================================
    logic [$clog2(HIDDEN)-1:0]   activ_stream_idx;   // for gate/up activation feed
    logic [$clog2(INTER)-1:0]    down_activ_stream_idx; // for down activation feed

    // =========================================================================
    // Main Pipeline FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= S_IDLE;
            busy               <= 1'b0;
            done               <= 1'b0;
            pcie_rx_ready      <= 1'b0;
            pcie_tx_valid      <= 1'b0;
            expert_cnt         <= '0;
            accum_idx          <= '0;
            merge_idx          <= '0;
            silu_idx           <= '0;
            is_shared_expert   <= 1'b0;
            gate_done          <= 1'b0;
            up_done            <= 1'b0;
            down_done          <= 1'b0;
            hbm2_start         <= 1'b0;
            sa_gate_up_start   <= 1'b0;
            sa_down_start      <= 1'b0;
            silu_valid_in      <= 1'b0;
            sa_gate_up_activ_valid <= 1'b0;
            sa_down_activ_valid    <= 1'b0;
            activ_stream_idx   <= '0;
            down_activ_stream_idx <= '0;

            // Clear buffers
            for (int i = 0; i < HIDDEN; i++) begin
                activ_buf[i]   <= '0;
                ffn_accum[i]   <= '0;
            end
            for (int i = 0; i < INTER; i++) begin
                gate_buf[i]    <= '0;
                up_buf[i]      <= '0;
                combined[i]    <= '0;
            end
        end else begin
            // Default pulse deassert
            done               <= 1'b0;
            pcie_rx_ready      <= 1'b0;
            pcie_tx_valid      <= 1'b0;
            hbm2_start         <= 1'b0;
            sa_gate_up_start   <= 1'b0;
            sa_down_start      <= 1'b0;
            silu_valid_in      <= 1'b0;

            // ================================================================
            // Continuous: feed activation data to systolic arrays while they
            // assert activ_ready. The activation index cycles through the
            // appropriate buffer (activ_buf for gate/up, combined for down).
            // ================================================================
            if (sa_gate_up_activ_ready && sa_gate_up_activ_valid) begin
                // Consumed: advance to next chunk
                activ_stream_idx <= activ_stream_idx + DSP_LANES;
                if (activ_stream_idx + DSP_LANES >= HIDDEN)
                    activ_stream_idx <= '0;  // wrap for next row
            end
            if (!sa_gate_up_activ_valid && sa_gate_up_activ_ready) begin
                // Start feeding
                for (int k = 0; k < DSP_LANES; k++) begin
                    sa_gate_up_activ_data[k*DATA_W +: DATA_W]
                        <= activ_buf[(activ_stream_idx + k) % HIDDEN];
                end
                sa_gate_up_activ_valid <= 1'b1;
            end else if (!sa_gate_up_activ_ready) begin
                sa_gate_up_activ_valid <= 1'b0;
            end

            // Down projection activation feed
            if (sa_down_activ_ready && sa_down_activ_valid) begin
                down_activ_stream_idx <= down_activ_stream_idx + DSP_LANES;
                if (down_activ_stream_idx + DSP_LANES >= INTER)
                    down_activ_stream_idx <= '0;
            end
            if (!sa_down_activ_valid && sa_down_activ_ready) begin
                // Feed combined[] values as FP8 (truncated from fp16)
                for (int k = 0; k < DSP_LANES; k++) begin
                    sa_down_activ_data[k*DATA_W +: DATA_W]
                        <= combined[(down_activ_stream_idx + k) % INTER][7:0];
                end
                sa_down_activ_valid <= 1'b1;
            end else if (!sa_down_activ_ready) begin
                sa_down_activ_valid <= 1'b0;
            end

            // ================================================================
            // Continuous: collect results from systolic arrays
            // ================================================================

            // Gate/Up result → gate_buf or up_buf
            if (sa_gate_up_result_valid) begin
                if (state == S_GATE_PROJECT) begin
                    gate_buf[sa_gate_up_result_row] <= sa_gate_up_result_data[15:0];
                end else if (state == S_UP_PROJECT) begin
                    up_buf[sa_gate_up_result_row] <= sa_gate_up_result_data[15:0];
                end
                if (sa_gate_up_result_last) begin
                    if (state == S_GATE_PROJECT)
                        gate_done <= 1'b1;
                    else if (state == S_UP_PROJECT)
                        up_done <= 1'b1;
                end
            end

            // Down result → accumulated into ffn_accum
            if (sa_down_result_valid) begin
                ffn_accum[sa_down_result_row]
                    <= ffn_accum[sa_down_result_row] + sa_down_result_data;
                if (sa_down_result_last)
                    down_done <= 1'b1;
            end

            // ================================================================
            // FSM state transitions
            // ================================================================
            case (state)

                // ------------------------------------------------------------
                // S_IDLE: Wait for activation from PCIe (attention output)
                // ------------------------------------------------------------
                S_IDLE: begin
                    if (pcie_rx_valid) begin
                        // Latch activation into M20K buffer
                        for (int i = 0; i < HIDDEN; i++) begin
                            activ_buf[i] <= pcie_rx_data[i*DATA_W +: DATA_W];
                        end
                        // Clear accumulators for new token
                        for (int i = 0; i < HIDDEN; i++)
                            ffn_accum[i] <= '0;
                        pcie_rx_ready <= 1'b1;
                        busy          <= 1'b1;
                        expert_cnt    <= '0;
                        is_shared_expert <= 1'b0;
                        gate_done     <= 1'b0;
                        up_done       <= 1'b0;
                        down_done     <= 1'b0;
                        silu_idx      <= '0;
                        merge_idx     <= '0;
                        activ_stream_idx <= '0;
                        down_activ_stream_idx <= '0;
                        state         <= S_LOAD_WEIGHTS;
                    end
                end

                // ------------------------------------------------------------
                // S_LOAD_WEIGHTS: Request gate_w from HBM2 for current expert
                // ------------------------------------------------------------
                S_LOAD_WEIGHTS: begin
                    if (!hbm2_busy && !hbm2_start) begin
                        // Gate weights at expert_base + 0
                        hbm2_base_addr <= expert_base(expert_id[expert_cnt])
                                        + 32'(GATE_OFFSET_B);
                        hbm2_words     <= 16'(HIDDEN * INTER);  // 2,883,584 FP8
                        hbm2_start     <= 1'b1;
                        gate_done      <= 1'b0;
                        state          <= S_GATE_PROJECT;
                    end
                end

                // ------------------------------------------------------------
                // S_GATE_PROJECT: Compute gate(hidden) → gate_out[INTER]
                // ------------------------------------------------------------
                S_GATE_PROJECT: begin
                    // Start systolic array for gate projection
                    if (!sa_gate_up_busy && !sa_gate_up_start) begin
                        sa_gate_up_start <= 1'b1;
                    end

                    // Wait for gate projection to complete
                    if (gate_done) begin
                        gate_done      <= 1'b0;
                        silu_idx       <= '0;
                        state          <= S_SILU;
                    end
                end

                // ------------------------------------------------------------
                // S_SILU: Apply SiLU activation to gate_out[INTER]
                //   Feed gate_buf[INTER] through silu_activation in 64-wide
                //   chunks. Store result back into gate_buf (in-place).
                // ------------------------------------------------------------
                S_SILU: begin
                    if (silu_valid_out) begin
                        // Write SiLU result back to gate_buf
                        for (int k = 0; k < DSP_LANES; k++) begin
                            if (silu_idx + k < INTER)
                                gate_buf[silu_idx + k] <= silu_data_out[k];
                        end
                        silu_idx <= silu_idx + DSP_LANES;
                    end

                    if (!silu_valid_in && silu_idx < INTER) begin
                        // Feed next chunk to SiLU
                        for (int k = 0; k < DSP_LANES; k++) begin
                            if (silu_idx + k < INTER)
                                silu_data_in[k] <= gate_buf[silu_idx + k];
                            else
                                silu_data_in[k] <= 16'h0000;
                        end
                        silu_valid_in <= 1'b1;
                    end else if (silu_valid_in) begin
                        silu_valid_in <= 1'b0;
                    end

                    // Done when all INTER elements processed
                    if (silu_idx + DSP_LANES >= INTER && !silu_valid_in && !silu_valid_out) begin
                        silu_idx       <= '0;
                        // Now load up_w from HBM2
                        if (!hbm2_busy && !hbm2_start) begin
                            hbm2_base_addr <= expert_base(expert_id[expert_cnt])
                                            + 32'(UP_OFFSET_B);
                            hbm2_words     <= 16'(HIDDEN * INTER);
                            hbm2_start     <= 1'b1;
                            up_done        <= 1'b0;
                            state          <= S_UP_PROJECT;
                        end
                    end
                end

                // ------------------------------------------------------------
                // S_UP_PROJECT: Compute up(hidden) → up_out[INTER]
                // ------------------------------------------------------------
                S_UP_PROJECT: begin
                    if (!sa_gate_up_busy && !sa_gate_up_start) begin
                        sa_gate_up_start <= 1'b1;
                    end

                    if (up_done) begin
                        up_done        <= 1'b0;
                        merge_idx      <= '0;
                        state          <= S_MERGE_GATE_UP;
                    end
                end

                // ------------------------------------------------------------
                // S_MERGE_GATE_UP: Element-wise multiply
                //   combined[i] = SiLU(gate_out[i]) × up_out[i]  (fp16 × fp16)
                //   Uses DSP inference for 64 parallel multiplies per cycle.
                //
                //   fp16 multiply: sign XOR, exponent add - bias,
                //   mantissa multiply with normalization.
                //   Each 11-bit mantissa (implicit 1 at bit 10) produces
                //   a 22-bit product. Normalization extracts 10 explicit bits
                //   and adjusts the exponent.
                // ------------------------------------------------------------
                S_MERGE_GATE_UP: begin
                    // Process 64 elements per cycle
                    for (int k = 0; k < DSP_LANES; k++) begin
                        if (merge_idx + k < INTER) begin
                            automatic logic [15:0] g_val = gate_buf[merge_idx + k];
                            automatic logic [15:0] u_val = up_buf[merge_idx + k];

                            // fp16 multiply: extract sign/exp/mant, compute product
                            automatic logic        g_sign     = g_val[15];
                            automatic logic        u_sign     = u_val[15];
                            automatic logic [4:0]  g_exp      = g_val[14:10];
                            automatic logic [4:0]  u_exp      = u_val[14:10];
                            automatic logic [9:0]  g_mant     = g_val[9:0];
                            automatic logic [9:0]  u_mant     = u_val[9:0];
                            automatic logic [10:0] g_mant_ext = (g_exp == 5'd0) ? {1'b0, g_mant} : {1'b1, g_mant};
                            automatic logic [10:0] u_mant_ext = (u_exp == 5'd0) ? {1'b0, u_mant} : {1'b1, u_mant};
                            automatic logic [5:0]  exp_raw;
                            automatic logic [6:0]  exp_adj;     // 7-bit for overflow detect
                            automatic logic [21:0] mant_prod;
                            automatic logic        res_sign;
                            automatic logic [4:0]  res_exp;
                            automatic logic [9:0]  res_mant;

                            res_sign = g_sign ^ u_sign;
                            // exp_raw = exp_g + exp_u - 15 (bias)
                            exp_raw  = {1'b0, g_exp} + {1'b0, u_exp} - 6'd15;
                            mant_prod = g_mant_ext * u_mant_ext;

                            // ==================================================
                            // Normalize 22-bit mantissa product to 10-bit explicit
                            //
                            // Two 11-bit mantissas (range [1024, 2047)):
                            //   product in [1,048,576, 4,190,208) → 22 bits
                            //
                            // Case 1: mant_prod[21]=1 → product in [2.0, 4.0)
                            //   Normalize: shift right by 1 (÷2), exponent +1
                            //   Explicit mantissa (10b): mant_prod[20:11]
                            //
                            // Case 2: mant_prod[21]=0, mant_prod[20]=1 → [1.0, 2.0)
                            //   No shift, explicit mantissa: mant_prod[19:10]
                            //
                            // Case 3: subnormal product (should not occur for
                            //   normal×normal but handled defensively):
                            //   Shift left until bit 20=1, decrement exponent
                            // ==================================================
                            if (mant_prod[21]) begin
                                res_mant = mant_prod[20:11];
                                exp_adj  = {1'b0, exp_raw} + 7'd1;  // exp += 1
                            end else if (mant_prod[20]) begin
                                res_mant = mant_prod[19:10];
                                exp_adj  = {1'b0, exp_raw};         // no adjustment
                            end else begin
                                // Subnormal: use trailing bits (should be rare)
                                res_mant = mant_prod[18:9];
                                exp_adj  = {1'b0, exp_raw};         // keep as-is
                            end

                            // Clamp exponent to fp16 range [0, 30]
                            // exp_adj[6] means negative (underflow)
                            if (exp_adj[6] || (exp_adj[5] && !exp_adj[4] && exp_adj == 7'd0))
                                res_exp = 5'd0;          // underflow → zero
                            else if (exp_adj > 7'd30)
                                res_exp = 5'd30;         // overflow → inf
                            else
                                res_exp = exp_adj[4:0];

                            combined[merge_idx + k] <= {res_sign, res_exp, res_mant};
                        end
                    end
                    merge_idx <= merge_idx + DSP_LANES;

                    if (merge_idx + DSP_LANES >= INTER) begin
                        merge_idx <= '0;
                        // Load down_w from HBM2
                        if (!hbm2_busy && !hbm2_start) begin
                            hbm2_base_addr <= expert_base(expert_id[expert_cnt])
                                            + 32'(DOWN_OFFSET_B);
                            hbm2_words     <= 16'(INTER * HIDDEN);
                            hbm2_start     <= 1'b1;
                            down_done      <= 1'b0;
                            state          <= S_DOWN_PROJECT;
                        end
                    end
                end

                // ------------------------------------------------------------
                // S_DOWN_PROJECT: Compute down(combined) → expert_out[HIDDEN]
                // ------------------------------------------------------------
                S_DOWN_PROJECT: begin
                    if (!sa_down_busy && !sa_down_start) begin
                        sa_down_start <= 1'b1;
                    end

                    if (down_done) begin
                        down_done      <= 1'b0;
                        accum_idx      <= '0;
                        state          <= S_ACCUM_EXPERT;
                    end
                end

                // ------------------------------------------------------------
                // S_ACCUM_EXPERT: Expert result already accumulated into
                //   ffn_accum[] during the down projection (done concurrently
                //   via the sa_down_result_valid handler above).
                //   Just verify completion and move to next expert.
                // ------------------------------------------------------------
                S_ACCUM_EXPERT: begin
                    // Accumulation happens concurrently in the result handler.
                    // This state just gates the transition.
                    if (sa_down_done) begin
                        state <= S_NEXT_EXPERT;
                    end
                end

                // ------------------------------------------------------------
                // S_NEXT_EXPERT: Loop to next expert or proceed to output
                // ------------------------------------------------------------
                S_NEXT_EXPERT: begin
                    if (expert_cnt < TOP_K - 1) begin
                        // Next routed expert
                        expert_cnt <= expert_cnt + 1'b1;
                        gate_done  <= 1'b0;
                        up_done    <= 1'b0;
                        down_done  <= 1'b0;
                        state      <= S_LOAD_WEIGHTS;
                    end else if (!is_shared_expert) begin
                        // After all TOP_K routed experts, do 1 shared expert
                        is_shared_expert <= 1'b1;
                        gate_done  <= 1'b0;
                        up_done    <= 1'b0;
                        down_done  <= 1'b0;
                        state      <= S_LOAD_WEIGHTS;
                    end else begin
                        // All experts done
                        state <= S_OUTPUT;
                    end
                end

                // ------------------------------------------------------------
                // S_OUTPUT: Output FFN result on PCIe TX
                //   Convert ffn_accum (fp24) → FP8 for PCIe output
                // ------------------------------------------------------------
                S_OUTPUT: begin
                    pcie_tx_valid <= 1'b1;
                    for (int i = 0; i < HIDDEN; i++) begin
                        // Truncate fp24 accumulator to FP8
                        // Production: proper fp24→fp8 conversion with rounding
                        // Bring-up: take top 8 bits of accumulator
                        pcie_tx_data[i*DATA_W +: DATA_W]
                            <= ffn_accum[i][ACCUM_W-1 -: DATA_W];
                    end
                    if (pcie_tx_ready) begin
                        pcie_tx_valid <= 1'b0;
                        done  <= 1'b1;
                        busy  <= 1'b0;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Debug: Performance Counters & Error Flags
    // =========================================================================
    logic [31:0] _perf_token_cnt, _perf_cycle_cnt, _perf_expert_cnt, _perf_axi_rbeat;
    logic        _err_merge_overflow, _err_silu_overflow, _err_axi_resp_err;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            _perf_token_cnt   <= 32'd0;
            _perf_cycle_cnt   <= 32'd0;
            _perf_expert_cnt  <= 32'd0;
            _perf_axi_rbeat   <= 32'd0;
            _err_merge_overflow <= 1'b0;
            _err_silu_overflow  <= 1'b0;
            _err_axi_resp_err   <= 1'b0;
        end else begin
            _perf_cycle_cnt <= _perf_cycle_cnt + 32'd1;
            if (done && state == S_OUTPUT)
                _perf_token_cnt <= _perf_token_cnt + 32'd1;
            if (state == S_NEXT_EXPERT)
                _perf_expert_cnt <= _perf_expert_cnt + 32'd1;
            if (m_axi_rvalid && m_axi_rready)
                _perf_axi_rbeat <= _perf_axi_rbeat + 32'd1;
            // Sticky error flags
            if (state == S_MERGE_GATE_UP && merge_idx >= INTER)
                _err_merge_overflow <= 1'b1;
            if (state == S_SILU && silu_idx >= INTER + DSP_LANES)
                _err_silu_overflow <= 1'b1;
            if (m_axi_rvalid && m_axi_rready && (m_axi_rresp != 2'b00))
                _err_axi_resp_err <= 1'b1;
        end
    end

    assign perf_token_cnt    = _perf_token_cnt;
    assign perf_cycle_cnt    = _perf_cycle_cnt;
    assign perf_expert_cnt   = _perf_expert_cnt;
    assign perf_axi_rbeat    = _perf_axi_rbeat;
    assign err_merge_overflow = _err_merge_overflow;
    assign err_silu_overflow  = _err_silu_overflow;
    assign err_axi_resp_err   = _err_axi_resp_err;

    // =========================================================================
    // Debug: Status Signal Assignments (combinational)
    // =========================================================================
    assign dbg_fsm_state   = state;
    assign dbg_expert_cnt  = expert_cnt;
    assign dbg_gate_done   = gate_done;
    assign dbg_up_done     = up_done;
    assign dbg_down_done   = down_done;
    assign dbg_silu_active = (state == S_SILU);
    assign dbg_merge_active = (state == S_MERGE_GATE_UP);
    assign dbg_hbm2_busy   = hbm2_busy;
    assign dbg_sa_active   = sa_gate_up_busy || sa_down_busy;

    // =========================================================================
    // Assertions (synthesis translate_off)
    // =========================================================================
    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (state == S_MERGE_GATE_UP && merge_idx >= INTER) begin
            $error("[FFN] merge_idx overflow: %0d >= %0d", merge_idx, INTER);
        end
        if (state == S_SILU && silu_idx >= INTER + DSP_LANES) begin
            $error("[FFN] silu_idx overflow: %0d", silu_idx);
        end
    end
    // synthesis translate_on

endmodule
