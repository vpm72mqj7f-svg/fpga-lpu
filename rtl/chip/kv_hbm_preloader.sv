//=============================================================================
// kv_hbm_preloader.sv — HBM → mla_kv_cache KV preload engine
//
// Reads KV entries from HBM and feeds them to mla_kv_cache.preload port.
// One entry per cycle. Used to load CPU-prefill KV cache into FPGA BRAM.
//
// Data path: CPU Prefill → PCIe DMA → kv_dma_bridge → HBM (write)
//            kv_hbm_preloader → HBM (read) → mla_kv_cache.preload (BRAM)
//=============================================================================

module kv_hbm_preloader #(
    parameter int K_LATENT     = 512,
    parameter int V_LATENT     = 512,
    parameter int DATA_W       = 32,
    parameter int KV_ENTRY_BYTES = (K_LATENT + V_LATENT),  // FP8: 1024 bytes
    parameter int WORDS_PER_ENTRY = KV_ENTRY_BYTES / (DATA_W/8),  // 32 words @32b
    parameter int ADDR_W       = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,

    // Control
    input  logic                         start,
    input  logic [ADDR_W-1:0]            hbm_base_addr,
    input  logic [15:0]                  num_entries,
    output logic                         done,
    output logic                         busy,

    // HBM read port (sim_axi4_hbm_model compatible)
    output logic [ADDR_W-1:0]            hbm_rd_addr,
    output logic                         hbm_rd_en,
    input  logic [255:0]                 hbm_rd_data,    // 256-bit HBM beat

    // mla_kv_cache preload port
    output logic                         preload_en,
    output logic [K_LATENT*DATA_W-1:0]   preload_K_flat,
    output logic [V_LATENT*DATA_W-1:0]   preload_V_flat
);

    typedef enum logic [1:0] { S_IDLE, S_READ, S_DONE } state_t;
    state_t state;

    logic [ADDR_W-1:0] base_addr_r;
    logic [15:0]       remain;
    logic [15:0]       entry_idx;

    // HBM reads 256-bit beats = 32 bytes.
    localparam int BEATS_PER_ENTRY = (KV_ENTRY_BYTES + 31) / 32;  // ceil division
    localparam int BEAT_CNT_W = $clog2(BEATS_PER_ENTRY > 1 ? BEATS_PER_ENTRY : 2);
    logic [BEAT_CNT_W-1:0] beat_cnt;

    // Accumulate a full KV entry from HBM beats (min 256 bits for 1 beat)
    localparam int ENTRY_BUF_W = (KV_ENTRY_BYTES * 8 > 256) ? KV_ENTRY_BYTES * 8 : 256;
    logic [ENTRY_BUF_W-1:0] entry_buf;
    logic                         entry_ready;
    logic [63:0]                  cycle_count;

    assign busy = (state != S_IDLE);

    // HBM read: one beat per cycle
    assign hbm_rd_en   = (state == S_READ);
    assign hbm_rd_addr = base_addr_r + (entry_idx * KV_ENTRY_BYTES) + (beat_cnt * 32);

    // Entry buffer: accumulate 32-byte beats into 1024-byte entry
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            entry_buf   <= '0;
            entry_ready <= 1'b0;
            beat_cnt    <= '0;
        end else begin
            entry_ready <= 1'b0;
            if (state == S_READ) begin
                // Store current beat into entry buffer
                entry_buf[beat_cnt*256 +: 256] <= hbm_rd_data;
                if (beat_cnt == (BEATS_PER_ENTRY - 1)) begin
                    entry_ready <= 1'b1;
                    beat_cnt <= '0;
                end else begin
                    beat_cnt <= beat_cnt + 1'b1;
                end
            end
        end
    end

    // Preload output: drive mla_kv_cache.preload when entry is ready
    generate
        for (genvar gi = 0; gi < K_LATENT; gi++) begin : gen_k_preload
            // FP8 K_latent values stored in entry_buf[gi*8 +: 8]
            // Output in Q12 lane: {24'b0, fp8_byte}
            assign preload_K_flat[gi*DATA_W +: DATA_W] =
                { {24{1'b0}}, entry_buf[gi*8 +: 8] };
        end
        for (genvar gi = 0; gi < V_LATENT; gi++) begin : gen_v_preload
            assign preload_V_flat[gi*DATA_W +: DATA_W] =
                { {24{1'b0}}, entry_buf[(K_LATENT + gi)*8 +: 8] };
        end
    endgenerate

    assign preload_en = entry_ready && (state == S_READ);

    // Main FSM
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            base_addr_r <= '0;
            remain      <= '0;
            entry_idx   <= '0;
            cycle_count <= '0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        base_addr_r <= hbm_base_addr;
                        remain      <= num_entries;
                        entry_idx   <= '0;
                        beat_cnt    <= '0;
                        cycle_count <= '0;
                        state <= S_READ;
                    end
                end

                S_READ: begin
                    cycle_count <= cycle_count + 1'b1;
                    if (entry_ready) begin
                        if (remain <= 1) begin
                            done  <= 1'b1;
                            state <= S_DONE;
                        end else begin
                            remain    <= remain - 1'b1;
                            entry_idx <= entry_idx + 1'b1;
                        end
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
