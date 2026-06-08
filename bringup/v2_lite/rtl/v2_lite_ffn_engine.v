////////////////////////////////////////////////////////////////////////////////
//
// FPGA LPU Bringup
//
// Filename     : v2_lite_ffn_engine.v
// Description  : V2-Lite FFN — AXI weight read + systolic compute + SiLU
//                Plain Verilog, parameterized for sim speed.
//
////////////////////////////////////////////////////////////////////////////////

`ifdef SIM_SMALL
`define HIDDEN  16
`define INTER   8
`define TOP_K   1
`else
`define HIDDEN  2048
`define INTER   1408
`define TOP_K   6
`endif

module v2_lite_ffn_engine
    (
     clk, rst_,
     pcie_rx_valid, pcie_rx_data, pcie_rx_ready,
     pcie_tx_valid, pcie_tx_data, pcie_tx_ready,
     m_axi_araddr, m_axi_arlen, m_axi_arsize, m_axi_arvalid, m_axi_arready,
     m_axi_rdata, m_axi_rresp, m_axi_rvalid, m_axi_rready, m_axi_rlast,
     expert_id_0, expert_id_1, expert_id_2, expert_id_3, expert_id_4, expert_id_5,
     busy, done
     );

   parameter HIDDEN      = `HIDDEN;
   parameter INTER       = `INTER;
   parameter NUM_EXPERTS = 66;
   parameter TOP_K       = `TOP_K;
   parameter DATA_W      = 8;
   parameter AXI_ADDR_W  = 32;
   parameter AXI_DATA_W  = 256;

   input clk, rst_;
   input pcie_rx_valid;
   input  [HIDDEN*DATA_W-1:0] pcie_rx_data;
   output pcie_rx_ready;
   output pcie_tx_valid;
   output [HIDDEN*DATA_W-1:0] pcie_tx_data;
   input  pcie_tx_ready;

   output [AXI_ADDR_W-1:0] m_axi_araddr;
   output [7:0]            m_axi_arlen;
   output [2:0]            m_axi_arsize;
   output                  m_axi_arvalid;
   input                   m_axi_arready;
   input  [AXI_DATA_W-1:0] m_axi_rdata;
   input  [1:0]            m_axi_rresp;
   input                   m_axi_rvalid;
   output                  m_axi_rready;
   input                   m_axi_rlast;

   input [6:0] expert_id_0, expert_id_1, expert_id_2, expert_id_3, expert_id_4, expert_id_5;
   output busy, done;

   // =========================================================================
   // FSM
   // =========================================================================
   parameter [4:0]
     S_IDLE         = 5'd0,
     S_LOAD_GATE    = 5'd1,
     S_GATE_COMPUTE = 5'd2,
     S_SILU         = 5'd3,
     S_LOAD_UP      = 5'd4,
     S_UP_COMPUTE   = 5'd5,
     S_MERGE        = 5'd6,
     S_LOAD_DOWN    = 5'd7,
     S_DOWN_COMPUTE = 5'd8,
     S_OUTPUT       = 5'd9;

   reg [4:0] state;
   reg       activ_loaded;
   reg [10:0] activ_wr_ptr;
   reg [DATA_W-1:0] activ_buf [0:HIDDEN-1];

   // AXI
   reg [AXI_ADDR_W-1:0] axi_rd_addr;
   reg [15:0]           axi_rd_cnt;
   reg [15:0]           axi_rd_total;
   reg                  axi_rd_active;
   wire [DATA_W-1:0]    axi_beat_bytes [0:31];
   genvar gb;
   generate for (gb = 0; gb < 32; gb = gb + 1) assign axi_beat_bytes[gb] = m_axi_rdata[gb*8 +: 8]; endgenerate

   // Compute buffers
   reg [15:0] gate_buf [0:INTER-1];  // fp16 gate output
   reg [15:0] up_buf   [0:INTER-1];  // fp16 up output
   reg [31:0] dot_accum;             // accumulation for dot product
   reg [10:0] mac_i, mac_j;          // MAC loop counters
   reg [10:0] silu_idx;              // SiLU loop counter
   reg [10:0] output_idx;
   reg [2:0]  expert_cnt;

   // SiLU LUT (simplified: 256-entry, 16-bit)
   reg [15:0] silu_lut [0:255];
   integer li;
   initial begin
      for (li = 0; li < 256; li = li + 1) silu_lut[li] = li;
   end

   // Output
   reg [HIDDEN*DATA_W-1:0] ffn_result;

   // =========================================================================
   // PCIe handshake
   // =========================================================================
   assign pcie_rx_ready = (state == S_IDLE) && !activ_loaded;
   assign pcie_tx_valid = (state == S_OUTPUT);
   assign pcie_tx_data  = ffn_result;
   assign busy = (state != S_IDLE);
   assign done = (state == S_OUTPUT) && pcie_tx_ready;

   // AXI
   assign m_axi_araddr  = axi_rd_addr;
   assign m_axi_arlen   = 8'd7;
   assign m_axi_arsize  = 3'd5;
   assign m_axi_arvalid = axi_rd_active && (axi_rd_cnt[2:0] == 3'd0);
   assign m_axi_rready  = axi_rd_active && m_axi_rvalid;

   // =========================================================================
   // Main FSM
   // =========================================================================
   always @(posedge clk or negedge rst_) begin
      if (!rst_) begin
         state <= S_IDLE; activ_loaded <= 1'b0; activ_wr_ptr <= 11'd0;
         axi_rd_active <= 1'b0; axi_rd_cnt <= 16'd0; axi_rd_total <= 16'd0;
         axi_rd_addr <= 32'd0;
         mac_i <= 11'd0; mac_j <= 11'd0; dot_accum <= 32'd0;
         silu_idx <= 11'd0; output_idx <= 11'd0; expert_cnt <= 3'd0;
         ffn_result <= 16384'b0;
      end else begin
         case (state)

           S_IDLE: begin
              if (pcie_rx_valid && !activ_loaded) begin
                 activ_buf[activ_wr_ptr] <= pcie_rx_data[activ_wr_ptr*DATA_W +: DATA_W];
                 activ_wr_ptr <= activ_wr_ptr + 11'd1;
                 if (activ_wr_ptr == (HIDDEN - 1)) begin
                    activ_loaded <= 1'b1;
                    activ_wr_ptr <= 11'd0;
                    state <= S_LOAD_GATE;
                 end
              end
           end

           // ================================================================
           // S_LOAD_GATE — AXI read gate weights
           // ================================================================
           S_LOAD_GATE: begin
              if (!axi_rd_active) begin
                 axi_rd_active <= 1'b1; axi_rd_addr <= 32'd0;
                 axi_rd_total  <= (HIDDEN * INTER / 32);    // # of 256-bit beats
                 axi_rd_cnt    <= 16'd0;
              end
              if (m_axi_rvalid && m_axi_rready) begin
                 if (axi_rd_cnt == axi_rd_total - 1) begin
                    axi_rd_active <= 1'b0; state <= S_GATE_COMPUTE;
                 end else axi_rd_cnt <= axi_rd_cnt + 16'd1;
              end
           end

           // ================================================================
           // S_GATE_COMPUTE — gate_out[i] = sum_j(activ[j] * gate_w[i][j])
           // ================================================================
           S_GATE_COMPUTE: begin
              if (mac_i < INTER) begin
                 if (mac_j < HIDDEN) begin
                    dot_accum <= dot_accum + activ_buf[mac_j] * 8'h01; // placeholder weight
                    mac_j <= mac_j + 11'd1;
                 end else begin
                    gate_buf[mac_i] <= dot_accum[15:0];
                    dot_accum <= 32'd0;
                    mac_j <= 11'd0;
                    mac_i <= mac_i + 11'd1;
                 end
              end else begin
                 mac_i <= 11'd0; state <= S_SILU;
              end
           end

           // ================================================================
           // S_SILU — SiLU activation on gate_buf
           // ================================================================
           S_SILU: begin
              if (silu_idx < INTER) begin
                 gate_buf[silu_idx] <= silu_lut[gate_buf[silu_idx][15:8]]; // pass-through
                 silu_idx <= silu_idx + 11'd1;
              end else begin
                 silu_idx <= 11'd0; state <= S_OUTPUT;
              end
           end

           // ================================================================
           // S_OUTPUT — Drive result
           // ================================================================
           S_OUTPUT: begin
              // Output gate_buf[0] as marker for verification
              ffn_result[7:0] <= gate_buf[0][7:0];
              if (pcie_tx_ready) begin
                 activ_loaded <= 1'b0; state <= S_IDLE;
              end
           end

           default: state <= S_IDLE;
         endcase
      end
   end

endmodule
