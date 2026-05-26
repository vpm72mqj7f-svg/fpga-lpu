//=============================================================================
// chip_top.sv — Single FPGA chip top-level wrapper
//
// Each chip carries 1-2 Transformer layers + 12-14 experts.
// Chip 0 of each card is PCIe master; chips 1-3 use C2C proxy.
//
// Parameters from per-chip config:
//   CHIP_ID, CARD_ID    — global identity (0-31, 0-7)
//   LAYER_START, LAYER_END — which layers this chip computes
//   EXPERT_BITMAP[11:0] — which experts are local
//   IS_PCIE_MASTER      — 1 if chip 0 of card (has R-Tile PCIe IP)
//=============================================================================

`include "avalon_stream.svh"
`include "c2c_packet.svh"
`include "pcie_dma.svh"

module chip_top #(
    parameter int CHIP_ID        = 0,
    parameter int CARD_ID        = 0,
    parameter int LAYER_START    = 0,
    parameter int LAYER_END      = 1,
    parameter int IS_PCIE_MASTER = 1
) (
    input  logic clk, rst_n,

    // === C2C Dual Ring ===
    // Ring A (clockwise: 0→1→2→3→0)
    input  c2c_link_t c2c_rx_a,
    output c2c_link_t c2c_tx_a,
    // Ring B (counter-clockwise)
    input  c2c_link_t c2c_rx_b,
    output c2c_link_t c2c_tx_b,

    // === PCIe (Chip 0 only) ===
    input  pcie_dma_stream_t pcie_host,
    output pcie_dma_stream_t pcie_fpga,

    // === C2C Proxy (cross-card forwarding) ===
    output pcie_c2c_proxy_t  c2c_proxy
);

    // Config registers (set via PCIe BAR0 on chip 0, via C2C CTRL on others)
    logic [5:0]  cfg_layer_start, cfg_layer_end;
    logic [11:0] cfg_expert_bitmap;

    // === Pipeline ingress / egress ===
    // Token coming from previous chip (or Host for layer 0)
    logic        pipe_in_valid;
    logic [15:0] pipe_in_token_id;
    logic [31:0] pipe_in_hidden [8];  // 8-element hidden state

    // Token going to next chip (or Host for last layer)
    logic        pipe_out_valid;
    logic [15:0] pipe_out_token_id;
    logic [31:0] pipe_out_hidden [8];

    // MoE dispatch / reduce interfaces
    logic        moe_disp_valid, moe_disp_ready;
    logic [11:0] moe_disp_expert;
    logic [15:0] moe_disp_token;
    logic [31:0] moe_disp_activation [8];

    logic        moe_red_valid, moe_red_ready;
    logic [11:0] moe_red_expert;
    logic [15:0] moe_red_token;
    logic [31:0] moe_red_result [8];

    // === Host-facing ports (Chip 0 only) ===
    // Weight preload, config, activation I/O multiplexed through PCIe DMA

    // === Layer compute engine ===
    full_transformer_layer u_layer (
        .clk, .rst_n,
        .gamma_wr_en(1'b0), .gamma_wr_idx('0), .gamma_wr_data('0),
        .attn_qkv_wt_wr_en(1'b0), .attn_qkv_wt_sel('0),
        .attn_qkv_wt_row('0), .attn_qkv_wt_col('0), .attn_qkv_wt_wr_data('0),
        .attn_rope_lut_wr_en(1'b0), .attn_rope_lut_pos('0),
        .attn_rope_lut_pair('0), .attn_rope_lut_sin('0), .attn_rope_lut_cos('0),
        .token_position('0),
        .rtr_w_wr_en(1'b0), .rtr_w_wr_expert('0), .rtr_w_wr_idx('0),
        .rtr_w_wr_data('0),
        .gate_w_wr_en(1'b0), .up_w_wr_en(1'b0), .down_w_wr_en(1'b0),
        .gate_w_wr_row('0), .up_w_wr_row('0), .down_w_wr_row('0),
        .gate_w_wr_beat('0), .up_w_wr_beat('0), .down_w_wr_beat('0),
        .gate_w_wr_data('0), .up_w_wr_data('0), .down_w_wr_data('0),
        .scale_wr_en(1'b0), .scale_wr_addr('0), .scale_wr_data('0),
        .valid_in(1'b0),
        .a0('0),.a1('0),.a2('0),.a3('0),.a4('0),.a5('0),.a6('0),.a7('0),
        .valid_out(), .router_ok(),
        .y0(),.y1(),.y2(),.y3(),.y4(),.y5(),.y6(),.y7()
    );

    // === Pipeline forward logic ===
    // If this chip is the last for its assigned layers, forward to next chip.
    // Otherwise, loop back internally for the next layer on this chip.
    logic [5:0] next_layer;
    assign next_layer = LAYER_END + 6'd1;  // next layer in pipeline

    // === C2C message routing ===
    // Ring A: forward messages to next chip in ring (clockwise)
    // Ring B: forward messages to previous chip (counter-clockwise)
    // Each message is routed based on dst_chip field in header

    // C2C TX: select between Ring A and Ring B based on shortest path
    // (simplified: Ring A for 0→1,1→2,2→3,3→0 hops; Ring B for opposite)

    // === PCIe Proxy (Chip 0 only) ===
    generate
        if (IS_PCIE_MASTER) begin : g_pcie_master
            // PCIe DMA engine: desc ring → H2D/D2H streams
            // BAR0 register file
            // C2C proxy: forward chip 1-3 traffic to/from PCIe
            // (placeholder — full DMA engine not implemented in bring-up)
        end else begin : g_c2c_slave
            // Chips 1-3: all host interaction via C2C → Chip 0 proxy
            // Config/weight preload received over C2C CTRL messages
            // Activation in/out over C2C PIPELINE_FWD messages
        end
    endgenerate

    // === Assign chip identity ===
    // Set via top-level parameters or register writes (PCIe BAR0 / C2C CTRL).
    assign cfg_layer_start  = LAYER_START;
    assign cfg_layer_end    = LAYER_END;

endmodule
