//=============================================================================
// sim_axi4_hbm_model.sv — Behavioral AXI4-256 HBM2e Slave Model
//
// Replaces the circular 920×0.87 assumption with actual RTL-simulated bandwidth.
//
// HBM2e Reference: 32 pseudo-channels, 256-bit per channel, 920 GB/s aggregate
// Single pseudo-channel peak at 450 MHz: 256b × 450 MHz = 14.4 GB/s
//
// Icarus-compatible. B response queue uses altera_scfifo (Altera IP wrapper)
// instead of hand-written circular buffer.
//=============================================================================

module sim_axi4_hbm_model #(
    parameter int AXI_DATA_WIDTH   = 256,
    parameter int AXI_ADDR_WIDTH   = 32,
    parameter int AXI_ID_WIDTH     = 4,
    parameter int MEM_SIZE_BYTES   = 65536,      // 64 KB (16384 words) — fast sim
    parameter int READ_LATENCY     = 20,
    parameter int WRITE_LATENCY    = 14,
    parameter int BW_WINDOW_CYCLES = 10000
) (
    input  logic        clk,
    input  logic        rst_n,

    // AXI4 Slave — Write Address
    input  logic [AXI_ID_WIDTH-1:0]   s_axi_awid,
    input  logic [AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic [7:0]                s_axi_awlen,
    input  logic [2:0]                s_axi_awsize,
    input  logic [1:0]                s_axi_awburst,
    input  logic                      s_axi_awvalid,
    output logic                      s_axi_awready,

    // AXI4 Slave — Write Data
    input  logic [AXI_DATA_WIDTH-1:0]  s_axi_wdata,
    input  logic [AXI_DATA_WIDTH/8-1:0] s_axi_wstrb,
    input  logic                       s_axi_wlast,
    input  logic                       s_axi_wvalid,
    output logic                       s_axi_wready,

    // AXI4 Slave — Write Response
    output logic [AXI_ID_WIDTH-1:0]   s_axi_bid,
    output logic [1:0]                s_axi_bresp,
    output logic                      s_axi_bvalid,
    input  logic                      s_axi_bready,

    // AXI4 Slave — Read Address
    input  logic [AXI_ID_WIDTH-1:0]   s_axi_arid,
    input  logic [AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic [7:0]                s_axi_arlen,
    input  logic [2:0]                s_axi_arsize,
    input  logic [1:0]                s_axi_arburst,
    input  logic                      s_axi_arvalid,
    output logic                      s_axi_arready,

    // AXI4 Slave — Read Data
    output logic [AXI_ID_WIDTH-1:0]   s_axi_rid,
    output logic [AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output logic [1:0]                s_axi_rresp,
    output logic                      s_axi_rlast,
    output logic                      s_axi_rvalid,
    input  logic                      s_axi_rready,

    // Bandwidth monitoring
    output logic [63:0] write_bytes_total,
    output logic [63:0] read_bytes_total,
    output logic [31:0] write_bw_mbps,
    output logic [31:0] read_bw_mbps
);

    localparam int BYTES_PER_BEAT = 32;            // 256 / 8
    localparam int MEM_WORDS      = MEM_SIZE_BYTES / 4;
    localparam int ADDR_MASK      = MEM_SIZE_BYTES - 1;

    // Memory: 32-bit word addressed
    logic [31:0] mem [0:MEM_WORDS-1];

    // ── Write pipeline — pipelined: AW/W independent from B response ──
    logic [AXI_ID_WIDTH-1:0]   w_saved_id;
    logic [AXI_ADDR_WIDTH-1:0] w_saved_addr;
    logic [7:0]                w_saved_len;
    logic [7:0]                w_beat_count;
    logic                      w_burst_active;

    // ── B response: Altera scfifo for ID storage + ready-time queue ──
    localparam int B_QUEUE_DEPTH = 8;

    logic [AXI_ID_WIDTH-1:0]  b_fifo_wr_data;
    logic                     b_fifo_wr_en;
    logic                     b_fifo_rd_en;
    logic [AXI_ID_WIDTH-1:0]  b_fifo_rd_data;
    logic                     b_fifo_empty;
    logic [$clog2(B_QUEUE_DEPTH+1)-1:0] b_fifo_usedw;

    // Ready-time queue: absolute sim-cycle when each B entry matures
    logic [31:0] b_ready_time [0:B_QUEUE_DEPTH-1];
    logic [2:0]  b_rt_wr_ptr, b_rt_rd_ptr;
    logic [3:0]  b_rt_count;
    logic [31:0] sim_cycle;

    altera_scfifo #(
        .WIDTH(AXI_ID_WIDTH), .DEPTH(B_QUEUE_DEPTH), .SHOWAHEAD(1)
    ) u_b_fifo (
        .clk, .rst_n,
        .wr_en(b_fifo_wr_en), .wr_data(b_fifo_wr_data),
        .rd_en(b_fifo_rd_en), .rd_data(b_fifo_rd_data),
        .full(), .almost_full(), .empty(b_fifo_empty), .usedw(b_fifo_usedw)
    );

    // ── Read pipeline state machine ──
    localparam logic [1:0] R_IDLE      = 2'd0;
    localparam logic [1:0] R_WAIT_LAT  = 2'd1;
    localparam logic [1:0] R_SEND_DATA = 2'd2;

    logic [1:0] rstate;
    logic [AXI_ID_WIDTH-1:0]   r_saved_id;
    logic [AXI_ADDR_WIDTH-1:0] r_saved_addr;
    logic [7:0]                r_saved_len;
    logic [7:0]                r_beat_count;
    logic [7:0]                r_latency_count;

    // Bandwidth counters
    logic [63:0] w_bytes_window;
    logic [63:0] r_bytes_window;
    logic [31:0] bw_cycle_counter;

    // Address calculation wires
    logic [31:0] w_byte_addr;
    logic [31:0] r_byte_addr;
    logic [31:0] w_word_idx;
    logic [31:0] r_word_idx;
    integer      w_word_idx_int;
    integer      r_word_idx_int;

    assign w_byte_addr = w_saved_addr + (w_beat_count * 32);
    assign r_byte_addr = r_saved_addr + (r_beat_count * 32);
    assign w_word_idx  = (w_byte_addr & ADDR_MASK) >> 2;
    assign r_word_idx  = (r_byte_addr & ADDR_MASK) >> 2;

    // AXI4 control signals
    assign s_axi_bresp   = 2'b00;
    assign s_axi_rresp   = 2'b00;
    assign s_axi_rid     = r_saved_id;
    assign s_axi_rlast   = (r_beat_count == r_saved_len);

    // ── Read data output (direct memory access) ──
    logic [31:0] rdata_w0, rdata_w1, rdata_w2, rdata_w3;
    logic [31:0] rdata_w4, rdata_w5, rdata_w6, rdata_w7;

    assign rdata_w0 = mem[r_word_idx + 0];
    assign rdata_w1 = mem[r_word_idx + 1];
    assign rdata_w2 = mem[r_word_idx + 2];
    assign rdata_w3 = mem[r_word_idx + 3];
    assign rdata_w4 = mem[r_word_idx + 4];
    assign rdata_w5 = mem[r_word_idx + 5];
    assign rdata_w6 = mem[r_word_idx + 6];
    assign rdata_w7 = mem[r_word_idx + 7];

    assign s_axi_rdata = {
        rdata_w7, rdata_w6, rdata_w5, rdata_w4,
        rdata_w3, rdata_w2, rdata_w1, rdata_w0
    };

    // ── Write pipeline AW/W ──
    assign s_axi_awready = !w_burst_active;
    assign s_axi_wready  = w_burst_active;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_saved_id      <= '0;
            w_saved_addr    <= '0;
            w_saved_len     <= '0;
            w_beat_count    <= '0;
            w_burst_active  <= 1'b0;
            s_axi_bid       <= '0;
            s_axi_bvalid    <= 1'b0;
            sim_cycle       <= '0;
            b_rt_wr_ptr     <= '0;
            b_rt_rd_ptr     <= '0;
            b_rt_count       <= '0;
            b_fifo_wr_en    <= 1'b0;
            b_fifo_rd_en    <= 1'b0;
        end else begin
            sim_cycle <= sim_cycle + 1'b1;

            // ── AW acceptance ──
            if (s_axi_awvalid && s_axi_awready) begin
                w_saved_id   <= s_axi_awid;
                w_saved_addr <= s_axi_awaddr;
                w_saved_len  <= s_axi_awlen;
                w_beat_count <= '0;
                w_burst_active <= 1'b1;
            end

            // ── W data acceptance ──
            b_fifo_wr_en <= 1'b0;  // default
            if (s_axi_wvalid && s_axi_wready) begin
                mem[w_word_idx + 0] <= s_axi_wdata[31:0];
                mem[w_word_idx + 1] <= s_axi_wdata[63:32];
                mem[w_word_idx + 2] <= s_axi_wdata[95:64];
                mem[w_word_idx + 3] <= s_axi_wdata[127:96];
                mem[w_word_idx + 4] <= s_axi_wdata[159:128];
                mem[w_word_idx + 5] <= s_axi_wdata[191:160];
                mem[w_word_idx + 6] <= s_axi_wdata[223:192];
                mem[w_word_idx + 7] <= s_axi_wdata[255:224];

                if (s_axi_wlast) begin
                    w_burst_active <= 1'b0;
                    // Push ID to Altera scfifo + ready-time to shadow queue
                    if (b_rt_count < B_QUEUE_DEPTH) begin
                        b_fifo_wr_data <= w_saved_id;
                        b_fifo_wr_en   <= 1'b1;
                        b_ready_time[b_rt_wr_ptr] <= sim_cycle + WRITE_LATENCY;
                        b_rt_wr_ptr <= b_rt_wr_ptr + 1'b1;
                        b_rt_count  <= b_rt_count + 1'b1;
                    end
                end else begin
                    w_beat_count <= w_beat_count + 1'b1;
                end
            end

            // ── B response output (showahead: rd_data already has head entry) ──
            b_fifo_rd_en <= 1'b0;  // default
            if (!s_axi_bvalid && b_rt_count > 0 && !b_fifo_empty &&
                sim_cycle >= b_ready_time[b_rt_rd_ptr]) begin
                s_axi_bid    <= b_fifo_rd_data;   // showahead: head entry visible
                s_axi_bvalid <= 1'b1;
                b_fifo_rd_en <= 1'b1;             // advance FIFO to next entry
                b_rt_rd_ptr  <= b_rt_rd_ptr + 1'b1;
                b_rt_count   <= b_rt_count - 1'b1;
            end

            // Clear bvalid when accepted
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end
        end
    end

    // ── Read pipeline — pipelined: accept next AR during data transfer ──
    logic                       next_ar_valid;
    logic [AXI_ID_WIDTH-1:0]    next_ar_id;
    logic [AXI_ADDR_WIDTH-1:0]  next_ar_addr;
    logic [7:0]                 next_ar_len;

    assign s_axi_arready = (rstate == R_IDLE) || (rstate == R_SEND_DATA && !next_ar_valid);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rstate          <= R_IDLE;
            r_saved_id      <= '0;
            r_saved_addr    <= '0;
            r_saved_len     <= '0;
            r_beat_count    <= '0;
            r_latency_count <= '0;
            s_axi_rvalid    <= 1'b0;
            next_ar_valid   <= 1'b0;
            next_ar_id      <= '0;
            next_ar_addr    <= '0;
            next_ar_len     <= '0;
        end else begin
            // Accept new AR (either directly or into next_ar buffer)
            if (s_axi_arvalid && s_axi_arready) begin
                if (rstate == R_IDLE) begin
                    r_saved_id   <= s_axi_arid;
                    r_saved_addr <= s_axi_araddr;
                    r_saved_len  <= s_axi_arlen;
                    r_beat_count <= '0;
                    r_latency_count <= '0;
                    rstate <= R_WAIT_LAT;
                end else begin
                    next_ar_id    <= s_axi_arid;
                    next_ar_addr  <= s_axi_araddr;
                    next_ar_len   <= s_axi_arlen;
                    next_ar_valid <= 1'b1;
                end
            end

            case (rstate)
                R_IDLE: begin
                    if (next_ar_valid) begin
                        r_saved_id   <= next_ar_id;
                        r_saved_addr <= next_ar_addr;
                        r_saved_len  <= next_ar_len;
                        r_beat_count <= '0;
                        r_latency_count <= '0;
                        next_ar_valid <= 1'b0;
                        rstate <= R_WAIT_LAT;
                    end
                end

                R_WAIT_LAT: begin
                    if (r_latency_count >= READ_LATENCY) begin
                        s_axi_rvalid <= 1'b1;
                        rstate <= R_SEND_DATA;
                    end else begin
                        r_latency_count <= r_latency_count + 1'b1;
                    end
                end

                R_SEND_DATA: begin
                    if (s_axi_rvalid && s_axi_rready) begin
                        if (s_axi_rlast) begin
                            s_axi_rvalid <= 1'b0;
                            if (next_ar_valid) begin
                                r_saved_id   <= next_ar_id;
                                r_saved_addr <= next_ar_addr;
                                r_saved_len  <= next_ar_len;
                                r_beat_count <= '0;
                                r_latency_count <= '0;
                                next_ar_valid <= 1'b0;
                                rstate <= R_WAIT_LAT;
                            end else begin
                                rstate <= R_IDLE;
                            end
                        end else begin
                            r_beat_count <= r_beat_count + 1'b1;
                        end
                    end
                end

                default: rstate <= R_IDLE;
            endcase
        end
    end

    // ── Bandwidth monitoring ──
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_bytes_total <= '0;
            read_bytes_total  <= '0;
            w_bytes_window    <= '0;
            r_bytes_window    <= '0;
            write_bw_mbps     <= '0;
            read_bw_mbps      <= '0;
            bw_cycle_counter  <= '0;
        end else begin
            if (s_axi_wvalid && s_axi_wready) begin
                write_bytes_total <= write_bytes_total + 32;
                w_bytes_window    <= w_bytes_window + 32;
            end

            if (s_axi_rvalid && s_axi_rready) begin
                read_bytes_total <= read_bytes_total + 32;
                r_bytes_window   <= r_bytes_window + 32;
            end

            if (bw_cycle_counter >= BW_WINDOW_CYCLES) begin
                if (bw_cycle_counter > 0) begin
                    write_bw_mbps <= (w_bytes_window * 450) / bw_cycle_counter;
                    read_bw_mbps  <= (r_bytes_window * 450) / bw_cycle_counter;
                end
                w_bytes_window   <= '0;
                r_bytes_window   <= '0;
                bw_cycle_counter <= '0;
            end else begin
                bw_cycle_counter <= bw_cycle_counter + 1'b1;
            end
        end
    end

endmodule
