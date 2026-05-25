//=============================================================================
// kv_dma_engine.sv — KV Cache Offload DMA Engine
//
// Transfers KV cache blocks from host SSD to FPGA HBM via PCIe DMA.
// Descriptor-driven: host_addr, hbm_addr, length, session_id per transfer.
//
// Beat size: 256-bit (8×32b words). Burst of beats per descriptor.
// FSM: IDLE → REQ → WAIT_RSP → WRITE → NEXT → ... → DONE
//=============================================================================

module kv_dma_engine #(
    parameter int BEAT_BYTES  = 32,          // bytes per DMA beat (256-bit)
    parameter int WORD_BYTES  = 4,           // bytes per HBM word (32-bit)
    parameter int WORDS_PER_BEAT = BEAT_BYTES / WORD_BYTES  // 8 words per beat
) (
    input  logic         clk,
    input  logic         rst_n,

    // Descriptor input
    input  logic         desc_valid,
    output logic         desc_ready,
    input  logic [63:0]  desc_host_addr,
    input  logic [31:0]  desc_hbm_addr,
    input  logic [31:0]  desc_length,        // bytes to transfer
    input  logic [15:0]  desc_session_id,

    // Host DMA request (to PCIe / host bridge)
    output logic         dma_req_valid,
    input  logic         dma_req_ready,
    output logic [63:0]  dma_req_addr,
    output logic [31:0]  dma_req_length,     // bytes for this beat/request

    // Host DMA response (from PCIe / host bridge)
    input  logic         dma_rsp_valid,
    input  logic [BEAT_BYTES*8-1:0] dma_rsp_data,
    input  logic         dma_rsp_last,

    // HBM write port
    output logic [31:0]  hbm_wr_addr,
    output logic [31:0]  hbm_wr_data,
    output logic         hbm_wr_en,

    // Status
    output logic         done,
    output logic [15:0]  session_id,
    output logic [31:0]  bytes_transferred
);

    typedef enum logic [2:0] {
        S_IDLE, S_REQ, S_WAIT_RSP, S_WRITE, S_NEXT, S_DONE
    } state_t;
    state_t state;

    // Transfer tracking
    logic [63:0] host_addr_r;
    logic [31:0] hbm_addr_r;
    logic [31:0] remain_bytes;
    logic [15:0] sess_id_r;
    logic [31:0] bytes_done;

    // HBM write sub-beat counter
    logic [$clog2(WORDS_PER_BEAT)-1:0] wd_idx;
    logic [BEAT_BYTES*8-1:0]          rsp_data_r;
    logic                              rsp_last_r;

    assign desc_ready   = (state == S_IDLE);
    assign session_id   = sess_id_r;
    assign bytes_transferred = bytes_done;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            dma_req_valid <= 1'b0;
            dma_req_addr  <= '0;
            dma_req_length<= '0;
            hbm_wr_en    <= 1'b0;
            hbm_wr_addr  <= '0;
            hbm_wr_data  <= '0;
            done         <= 1'b0;
            host_addr_r  <= '0;
            hbm_addr_r   <= '0;
            remain_bytes <= '0;
            sess_id_r    <= '0;
            bytes_done   <= '0;
            wd_idx       <= '0;
            rsp_data_r   <= '0;
            rsp_last_r   <= 1'b0;
        end else begin
            done         <= 1'b0;
            dma_req_valid <= 1'b0;
            hbm_wr_en    <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (desc_valid) begin
                        host_addr_r  <= desc_host_addr;
                        hbm_addr_r   <= desc_hbm_addr;
                        remain_bytes <= desc_length;
                        sess_id_r    <= desc_session_id;
                        bytes_done   <= '0;
                        state <= S_REQ;
                    end
                end

                S_REQ: begin
                    dma_req_valid  <= 1'b1;
                    dma_req_addr   <= host_addr_r;
                    if (remain_bytes < BEAT_BYTES)
                        dma_req_length <= remain_bytes;
                    else
                        dma_req_length <= BEAT_BYTES;
                    if (dma_req_ready) begin
                        state <= S_WAIT_RSP;
                    end
                end

                S_WAIT_RSP: begin
                    if (dma_rsp_valid) begin
                        rsp_data_r <= dma_rsp_data;
                        rsp_last_r <= dma_rsp_last;
                        wd_idx     <= '0;
                        state      <= S_WRITE;
                    end
                end

                S_WRITE: begin
                    // Write one 32-bit word to HBM
                    hbm_wr_en   <= 1'b1;
                    hbm_wr_addr <= hbm_addr_r + (wd_idx * WORD_BYTES);
                    hbm_wr_data <= rsp_data_r[wd_idx*32 +: 32];
                    if (wd_idx == (WORDS_PER_BEAT - 1)) begin
                        // Done writing this beat
                        state <= S_NEXT;
                    end else begin
                        wd_idx <= wd_idx + 1'b1;
                    end
                end

                S_NEXT: begin
                    // Update bookkeeping
                    if (remain_bytes <= BEAT_BYTES) begin
                        bytes_done <= bytes_done + remain_bytes;
                        remain_bytes <= '0;
                    end else begin
                        bytes_done <= bytes_done + BEAT_BYTES;
                        remain_bytes <= remain_bytes - BEAT_BYTES;
                        host_addr_r  <= host_addr_r + BEAT_BYTES;
                        hbm_addr_r   <= hbm_addr_r + BEAT_BYTES;
                    end

                    if (rsp_last_r || remain_bytes <= BEAT_BYTES) begin
                        done  <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        state <= S_REQ;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
