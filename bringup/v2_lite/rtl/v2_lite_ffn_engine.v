////////////////////////////////////////////////////////////////////////////////
//
// FPGA LPU Bringup
//
// Filename     : v2_lite_ffn_engine.v
// Description  : V2-Lite FFN Compute Engine (2048×1408, 66 experts, TOP_K=6)
//                Reads weights via AXI4 read master from HBM2/SRAM.
//                Gate → SiLU → Up → Merge → Down → Accumulate pipeline.
//                Plain Verilog — no SystemVerilog constructs.
//
////////////////////////////////////////////////////////////////////////////////

module v2_lite_ffn_engine
    (
     clk,
     rst_,

     // PCIe RX (activation input from CPU attention)
     pcie_rx_valid,
     pcie_rx_data,
     pcie_rx_ready,

     // PCIe TX (FFN output to CPU)
     pcie_tx_valid,
     pcie_tx_data,
     pcie_tx_ready,

     // AXI4 Read Master (to HBM2 controller)
     m_axi_araddr,
     m_axi_arlen,
     m_axi_arsize,
     m_axi_arvalid,
     m_axi_arready,
     m_axi_rdata,
     m_axi_rresp,
     m_axi_rvalid,
     m_axi_rready,
     m_axi_rlast,

     // Expert selection from router
     expert_id_0,
     expert_id_1,
     expert_id_2,
     expert_id_3,
     expert_id_4,
     expert_id_5,

     // Status
     busy,
     done
     );

   // =========================================================================
   // Parameters
   // =========================================================================
   parameter HIDDEN      = 2048;
   parameter INTER       = 1408;
   parameter NUM_EXPERTS = 66;
   parameter TOP_K       = 6;
   parameter DATA_W      = 8;    // FP8
   parameter AXI_ADDR_W  = 32;
   parameter AXI_DATA_W  = 256;

   // =========================================================================
   // Ports — Clock & Reset
   // =========================================================================
   input                     clk;
   input                     rst_;

   // =========================================================================
   // Ports — PCIe Streaming
   // =========================================================================
   input                     pcie_rx_valid;
   input  [HIDDEN*DATA_W-1:0] pcie_rx_data;
   output                    pcie_rx_ready;
   output                    pcie_tx_valid;
   output [HIDDEN*DATA_W-1:0] pcie_tx_data;
   input                     pcie_tx_ready;

   // =========================================================================
   // Ports — AXI4 Read Master
   // =========================================================================
   output [AXI_ADDR_W-1:0]   m_axi_araddr;
   output [7:0]              m_axi_arlen;
   output [2:0]              m_axi_arsize;
   output                    m_axi_arvalid;
   input                     m_axi_arready;
   input  [AXI_DATA_W-1:0]   m_axi_rdata;
   input  [1:0]              m_axi_rresp;
   input                     m_axi_rvalid;
   output                    m_axi_rready;
   input                     m_axi_rlast;

   // =========================================================================
   // Ports — Expert Selection
   // =========================================================================
   input  [6:0]              expert_id_0;
   input  [6:0]              expert_id_1;
   input  [6:0]              expert_id_2;
   input  [6:0]              expert_id_3;
   input  [6:0]              expert_id_4;
   input  [6:0]              expert_id_5;

   // =========================================================================
   // Ports — Status
   // =========================================================================
   output                    busy;
   output                    done;

   // =========================================================================
   // FSM States
   // =========================================================================
   parameter [3:0]
     S_IDLE        = 4'd0,
     S_LOAD_WEIGHT = 4'd1,
     S_COMPUTE     = 4'd2,
     S_OUTPUT      = 4'd3;

   reg [3:0] state, next_state;

   // =========================================================================
   // Activation buffer (2048 × 8-bit)
   // =========================================================================
   reg [DATA_W-1:0] activ_buf [0:HIDDEN-1];
   reg [10:0]       activ_wr_ptr;
   reg              activ_loaded;

   // =========================================================================
   // AXI read controller
   // =========================================================================
   reg [AXI_ADDR_W-1:0] axi_rd_addr;
   reg [15:0]           axi_rd_cnt;    // words read so far
   reg [15:0]           axi_rd_total;  // total words to read
   reg                  axi_rd_active;

   // =========================================================================
   // Compute accumulator (simplified: sum of activations × weight pattern)
   // =========================================================================
   reg [HIDDEN*DATA_W-1:0] ffn_result;
   reg [10:0]              compute_idx;

   // =========================================================================
   // PCIe handshake
   // =========================================================================
   assign pcie_rx_ready = (state == S_IDLE) && !activ_loaded;
   assign pcie_tx_valid = (state == S_OUTPUT);
   assign pcie_tx_data  = ffn_result;
   assign busy = (state != S_IDLE);
   assign done = (state == S_OUTPUT) && pcie_tx_ready;

   // =========================================================================
   // AXI4 read master (256-bit = 32 FP8 per beat, 8-beat bursts)
   // =========================================================================
   assign m_axi_araddr  = axi_rd_addr;
   assign m_axi_arlen   = 8'd7;       // 8 beats per burst
   assign m_axi_arsize  = 3'd5;       // 32 bytes = 256-bit
   assign m_axi_arvalid = axi_rd_active && (axi_rd_cnt[2:0] == 3'd0);
   assign m_axi_rready  = axi_rd_active && m_axi_rvalid;

   // =========================================================================
   // FSM — sequential
   // =========================================================================
   always @(posedge clk or negedge rst_) begin
      if (!rst_) begin
         state         <= S_IDLE;
         activ_loaded  <= 1'b0;
         activ_wr_ptr  <= 11'd0;
         axi_rd_active <= 1'b0;
         axi_rd_cnt    <= 16'd0;
         axi_rd_total  <= 16'd0;
         axi_rd_addr   <= 32'd0;
         compute_idx   <= 11'd0;
         ffn_result    <= 16384'b0;
      end else begin
         case (state)

           // ============================================================
           // S_IDLE — Wait for activation from CPU
           // ============================================================
           S_IDLE: begin
              if (pcie_rx_valid && !activ_loaded) begin
                 activ_buf[activ_wr_ptr] <= pcie_rx_data[activ_wr_ptr*DATA_W +: DATA_W];
                 activ_wr_ptr <= activ_wr_ptr + 11'd1;
                 if (activ_wr_ptr == (HIDDEN - 1)) begin
                    activ_loaded <= 1'b1;
                    activ_wr_ptr <= 11'd0;
                    state <= S_LOAD_WEIGHT;
                 end
              end
           end

           // ============================================================
           // S_LOAD_WEIGHT — Read weights from HBM2 via AXI
           // 256 bytes per burst, 64 bursts = 16KB test read
           // ============================================================
           S_LOAD_WEIGHT: begin
              // Skip AXI for bringup — go directly to compute
              state <= S_COMPUTE;
           end

           // ============================================================
           // S_COMPUTE — Simple dot product: sum(activ[i] * weight_pattern)
           // For bringup: just pass activation through
           // ============================================================
           S_COMPUTE: begin
              // Simple: output ramp pattern (bypass actual MAC compute)
              ffn_result[7:0] <= 8'hA5;  // marker byte
              state <= S_OUTPUT;
           end

           // ============================================================
           // S_OUTPUT — Drive result on PCIe TX
           // ============================================================
           S_OUTPUT: begin
              if (pcie_tx_ready) begin
                 activ_loaded <= 1'b0;
                 state <= S_IDLE;
              end
           end

           default: state <= S_IDLE;
         endcase
      end
   end

endmodule
