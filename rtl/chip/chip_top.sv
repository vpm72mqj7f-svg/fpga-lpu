//=============================================================================
// chip_top.sv — Single FPGA chip top-level wrapper
//
// Each chip carries 1-2 Transformer layers + 12-14 experts.
// Chip 0 of each card is PCIe master; chips 1-3 use C2C proxy.
//
// Two compile modes (set via QSF global_assignment -name VERILOG_MACRO):
//
//   FPGA_LPU_SINGLE_CHIP defined:
//     Bare-metal bring-up. C2C/PCIe ports removed. Direct weight/activation I/O.
//     Use for single-board development, lab testing, and RTL simulation.
//
//   FPGA_LPU_SINGLE_CHIP NOT defined (default):
//     Production multi-chip mode. C2C dual ring + PCIe DMA ports present.
//     Use for 32-chip cluster synthesis.
//
// Parameter SINGLE_CHIP (runtime) gates internal pipeline/MoE logic.
// Macro FPGA_LPU_SINGLE_CHIP (compile-time) removes C2C/PCIe interface ports.
//
// Parameters from per-chip config:
//   CHIP_ID, CARD_ID    — global identity (0-31, 0-7)
//   LAYER_START, LAYER_END — which layers this chip computes
//   EXPERT_BITMAP[11:0] — which experts are local
//   IS_PCIE_MASTER      — 1 if chip 0 of card (has R-Tile PCIe IP)
//=============================================================================

`include "lpu_config.svh"

`ifndef FPGA_LPU_SINGLE_CHIP
`include "avalon_stream.svh"
`include "c2c_packet.svh"
`include "pcie_dma.svh"
`endif

module chip_top #(
    parameter int CHIP_ID        = 0,
    parameter int CARD_ID        = 0,
    parameter int LAYER_START    = 0,
    parameter int LAYER_END      = 1,
    parameter int IS_PCIE_MASTER = 1,
    parameter int SINGLE_CHIP    = 0,
    parameter int HIDDEN         = lpu_config_pkg::LPU_HIDDEN,
    parameter int INTER          = lpu_config_pkg::LPU_INTERMEDIATE
) (
    input  logic clk, rst_n,

`ifndef FPGA_LPU_SINGLE_CHIP
    // === C2C Dual Ring ===
    input  c2c_link_t c2c_rx_a,
    output c2c_link_t c2c_tx_a,
    input  c2c_link_t c2c_rx_b,
    output c2c_link_t c2c_tx_b,

    // === PCIe (Chip 0 only) ===
    input  pcie_dma_stream_t pcie_host,
    output pcie_dma_stream_t pcie_fpga,

    // === C2C Proxy (cross-card forwarding) ===
    output pcie_c2c_proxy_t  c2c_proxy,
`endif

    // === SINGLE_CHIP bring-up pass-through ports ===
    input  logic                         gamma_wr_en,
    input  logic [$clog2(HIDDEN)-1:0]    gamma_wr_idx,
    input  logic signed [31:0]           gamma_wr_data,

    input  logic                         attn_qkv_wt_wr_en,
    input  logic [2:0]                   attn_qkv_wt_sel,
    input  logic [$clog2(HIDDEN)-1:0]    attn_qkv_wt_row,
    input  logic [$clog2(HIDDEN)-1:0]    attn_qkv_wt_col,
    input  logic signed [15:0]           attn_qkv_wt_wr_data,

    input  logic                         attn_rope_lut_wr_en,
    input  logic [5:0]                   attn_rope_lut_pos,
    input  logic [$clog2(HIDDEN/2)-1:0]  attn_rope_lut_pair,
    input  logic signed [15:0]           attn_rope_lut_sin,
    input  logic signed [15:0]           attn_rope_lut_cos,

    input  logic [$clog2(64)-1:0]        token_position,

    input  logic                         cache_preload_en,
    input  logic [lpu_config_pkg::LPU_K_LATENT*lpu_config_pkg::LPU_DATA_WIDTH-1:0] cache_preload_K_flat,
    input  logic [lpu_config_pkg::LPU_V_LATENT*lpu_config_pkg::LPU_DATA_WIDTH-1:0] cache_preload_V_flat,

    input  logic                         rtr_w_wr_en,
    input  logic [1:0]                   rtr_w_wr_expert,
    input  logic [$clog2(HIDDEN)-1:0]    rtr_w_wr_idx,
    input  logic signed [31:0]           rtr_w_wr_data,

    input  logic                         gate_w_wr_en, up_w_wr_en, down_w_wr_en,
    input  logic [$clog2(INTER)-1:0]     gate_w_wr_row, up_w_wr_row,
    input  logic [$clog2(HIDDEN)-1:0]    down_w_wr_row,
    input  logic [0:0]                   gate_w_wr_beat, up_w_wr_beat, down_w_wr_beat,
    input  logic [15:0]                  gate_w_wr_data, up_w_wr_data, down_w_wr_data,
    input  logic [$clog2(lpu_config_pkg::LPU_TOTAL_PER_FPGA > 1 ? lpu_config_pkg::LPU_TOTAL_PER_FPGA : 2)-1:0] ffn_expert_sel,

    input  logic                         scale_wr_en,
    input  logic [$clog2(lpu_config_pkg::LPU_SCALE_GROUPS)-1:0] scale_wr_addr,
    input  logic [7:0]                   scale_wr_data,

    // Local expert bitmap: 1=expert resident on this chip
    input  logic [lpu_config_pkg::LPU_NUM_EXPERTS-1:0] cfg_local_experts,

    input  logic                         valid_in,
    input  logic [HIDDEN*32-1:0]         a_flat,

    output logic                         valid_out,
    output logic                         router_ok,
    output logic [HIDDEN*32-1:0]         y_flat
);

    // Config registers (set via PCIe BAR0 on chip 0, via C2C CTRL on others)
    logic [5:0]  cfg_layer_start, cfg_layer_end;

    // === Pipeline ingress / egress (unused in SINGLE_CHIP bring-up) ===
    logic        pipe_in_valid;
    logic [15:0] pipe_in_token_id;
    logic [31:0] pipe_in_hidden [8];

    logic        pipe_out_valid;
    logic [15:0] pipe_out_token_id;
    logic [31:0] pipe_out_hidden [8];

    // MoE dispatch / reduce interfaces (unused in bring-up)
    logic        moe_disp_valid, moe_disp_ready;
    logic [11:0] moe_disp_expert;
    logic [15:0] moe_disp_token;
    logic [31:0] moe_disp_activation [8];

    logic        moe_red_valid, moe_red_ready;
    logic [11:0] moe_red_expert;
    logic [15:0] moe_red_token;
    logic [31:0] moe_red_result [8];

    // === Layer compute engine ===
    full_transformer_layer #(
        .HIDDEN(HIDDEN)
    ) u_layer (
        .clk, .rst_n,
        .gamma_wr_en, .gamma_wr_idx, .gamma_wr_data,
        .attn_qkv_wt_wr_en, .attn_qkv_wt_sel,
        .attn_qkv_wt_row, .attn_qkv_wt_col, .attn_qkv_wt_wr_data,
        .attn_rope_lut_wr_en, .attn_rope_lut_pos,
        .attn_rope_lut_pair, .attn_rope_lut_sin, .attn_rope_lut_cos,
        .token_position,
        .cache_preload_en, .cache_preload_K_flat, .cache_preload_V_flat,
        .rtr_w_wr_en, .rtr_w_wr_expert, .rtr_w_wr_idx, .rtr_w_wr_data,
        .gate_w_wr_en, .up_w_wr_en, .down_w_wr_en,
        .gate_w_wr_row, .up_w_wr_row, .down_w_wr_row,
        .gate_w_wr_beat, .up_w_wr_beat, .down_w_wr_beat,
        .gate_w_wr_data, .up_w_wr_data, .down_w_wr_data,
        .ffn_expert_sel,
        .scale_wr_en, .scale_wr_addr, .scale_wr_data,
        .cfg_local_experts,
        .valid_in,
        .a_flat,
        .valid_out,
        .router_ok,
        .y_flat
    );

    // === Pipeline forward logic ===
    logic [5:0] next_layer;
    assign next_layer = LAYER_END + 6'd1;

    // === C2C / PCIe (production only, stub) ===
    // In SINGLE_CHIP mode, these are present but unused.
    // In production mode, C2C ring + PCIe DMA are fully active.

    generate
        if (IS_PCIE_MASTER) begin : g_pcie_master
            // PCIe DMA engine: desc ring → H2D/D2H streams
            // BAR0 register file
            // C2C proxy: forward chip 1-3 traffic to/from PCIe
        end else begin : g_c2c_slave
            // Chips 1-3: all host interaction via C2C → Chip 0 proxy
            // Config/weight preload received over C2C CTRL messages
            // Activation in/out over C2C PIPELINE_FWD messages
        end
    endgenerate

    // === Assign chip identity ===
    assign cfg_layer_start = LAYER_START;
    assign cfg_layer_end   = LAYER_END;

endmodule
