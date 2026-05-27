//=============================================================================
// kv_dma_bridge.sv — KV Cache DMA Bridge (P0)
//
// Double-buffered DMA engine for CPU prefill → FPGA HBM KV cache transfer.
//
// Operation:
//   CPU produces KV entries in host memory (pinned DMA buffer)
//   → PCIe DMA to FPGA HBM
//   → FPGA decode reads KV entries from HBM
//
// Double buffering:
//   Buffer A: FPGA decode reads from HBM bank A
//   Buffer B: CPU prefill + DMA writes to HBM bank B
//   Swap when B is ready and A is drained
//
// Bandwidth: PCIe 5.0 x16 → 28 GB/s effective
//   KV entry: 1 KB per token (K_latent + V_latent, fp8)
//   Transfer rate: 28M tokens/s → 35 ns/token
//   For P=128 chunk: 128 KB → 4.6 us transfer time
//=============================================================================

module kv_dma_bridge #(
    parameter int KV_ENTRY_BYTES  = 1024,     // K_latent(512) + V_latent(512) fp8
    parameter int MAX_TOKENS      = 4096,     // max KV entries per HBM bank
    parameter int PCIE_BEAT_BYTES = 32,       // 256-bit PCIe beat
    parameter int BEATS_PER_ENTRY = KV_ENTRY_BYTES / PCIE_BEAT_BYTES  // 32
) (
    input  logic        clk,
    input  logic        rst_n,

    // Control
    input  logic        start_dma,           // pulse to start transfer
    input  logic [31:0] host_addr_base,      // host physical address
    input  logic [31:0] hbm_addr_base,       // HBM destination address
    input  logic [15:0] num_tokens,          // tokens to transfer
    output logic        dma_done,
    output logic [15:0] tokens_transferred,

    // Buffer management
    output logic        buf_a_active,        // which buffer FPGA is reading
    output logic        buf_b_ready,         // buffer B has fresh data
    input  logic        swap_buffers,        // FPGA requests buffer swap

    // ── PCIe DMA Request (to host) ──
    output logic        pcie_req_valid,
    input  logic        pcie_req_ready,
    output logic [63:0] pcie_req_addr,       // host physical address
    output logic [31:0] pcie_req_length,     // bytes

    // ── PCIe DMA Response (from host) ──
    input  logic        pcie_rsp_valid,
    input  logic [255:0] pcie_rsp_data,      // 256-bit beat
    input  logic        pcie_rsp_last,

    // ── HBM Write Port ──
    output logic [31:0] hbm_wr_addr,
    output logic [255:0] hbm_wr_data,
    output logic        hbm_wr_en
);

    typedef enum logic [2:0] {
        S_IDLE, S_REQ, S_XFER, S_FLUSH, S_DONE
    } state_t;
    state_t state;

    logic [31:0] host_addr_r;
    logic [31:0] hbm_addr_r;
    logic [15:0] tokens_remaining;
    logic [15:0] tokens_done;
    logic [4:0]  beat_idx;     // 0..BEATS_PER_ENTRY-1

    assign tokens_transferred = tokens_done;

    // PCIe DMA request
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pcie_req_valid  <= 1'b0;
            pcie_req_addr   <= '0;
            pcie_req_length <= '0;
        end else begin
            if (state == S_REQ && !pcie_req_valid) begin
                pcie_req_valid  <= 1'b1;
                pcie_req_addr   <= {32'd0, host_addr_r};
                pcie_req_length <= KV_ENTRY_BYTES;  // one KV entry per request
            end else if (pcie_req_valid && pcie_req_ready) begin
                pcie_req_valid <= 1'b0;
            end
        end
    end

    // HBM write
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hbm_wr_en   <= 1'b0;
            hbm_wr_addr <= '0;
            hbm_wr_data <= '0;
            beat_idx    <= '0;
        end else begin
            hbm_wr_en <= 1'b0;

            if (state == S_XFER && pcie_rsp_valid) begin
                hbm_wr_en   <= 1'b1;
                hbm_wr_addr <= hbm_addr_r + (tokens_done * KV_ENTRY_BYTES) +
                               (beat_idx * PCIE_BEAT_BYTES);
                hbm_wr_data <= pcie_rsp_data;

                if (pcie_rsp_last || beat_idx == BEATS_PER_ENTRY - 1) begin
                    beat_idx <= '0;
                end else begin
                    beat_idx <= beat_idx + 1'b1;
                end
            end
        end
    end

    // Main FSM
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            dma_done         <= 1'b0;
            host_addr_r      <= '0;
            hbm_addr_r       <= '0;
            tokens_remaining <= '0;
            tokens_done      <= '0;
            buf_a_active     <= 1'b1;
            buf_b_ready      <= 1'b0;
        end else begin
            dma_done <= 1'b0;

            // Buffer swap
            if (swap_buffers) begin
                buf_a_active <= ~buf_a_active;
                buf_b_ready  <= 1'b0;
            end

            case (state)
                S_IDLE: begin
                    if (start_dma) begin
                        host_addr_r      <= host_addr_base;
                        hbm_addr_r       <= hbm_addr_base;
                        tokens_remaining <= num_tokens;
                        tokens_done      <= '0;
                        state <= S_REQ;
                    end
                end

                S_REQ: begin
                    if (pcie_req_valid && pcie_req_ready) begin
                        state <= S_XFER;
                    end
                end

                S_XFER: begin
                    if (pcie_rsp_valid && pcie_rsp_last) begin
                        tokens_done      <= tokens_done + 1'b1;
                        tokens_remaining <= tokens_remaining - 1'b1;
                        host_addr_r      <= host_addr_r + KV_ENTRY_BYTES;

                        if (tokens_remaining == 1) begin
                            state <= S_FLUSH;
                        end else begin
                            state <= S_REQ;  // request next KV entry
                        end
                    end
                end

                S_FLUSH: begin
                    // Wait for last HBM write to complete
                    dma_done    <= 1'b1;
                    buf_b_ready <= 1'b1;   // buffer B now has fresh data
                    state <= S_DONE;
                end

                S_DONE: begin
                    if (!start_dma) state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
