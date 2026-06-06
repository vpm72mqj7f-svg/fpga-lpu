//=============================================================================
// cluster_top.sv — Multi-chip pipeline cluster wrapper
//
// Instantiates NUM_CHIPS chip_top instances connected via pipeline forwarding.
// Each chip processes its assigned layers; tokens flow sequentially through
// the chain: Chip 0 → Chip 1 → ... → Chip N-1.
//
// Weight preload signals are broadcast to all chips. Each chip internally
// filters by its LAYER_START/LAYER_END config.
//
// Parameters:
//   NUM_CHIPS       — chips in cluster (default 4, production 32)
//   CHIPS_PER_CARD  — chips sharing one PCIe master (default 4)
//   HIDDEN          — model hidden dimension
//   INTER           — FFN intermediate dimension
//
// Compile modes:
//   FPGA_LPU_SINGLE_CHIP defined — all chip_tops in bring-up mode
//   Default                        — production multi-chip mode with C2C
//=============================================================================

`include "lpu_config.svh"

`ifndef FPGA_LPU_SINGLE_CHIP
`include "c2c_packet.svh"
`include "pcie_dma.svh"
`endif

module cluster_top #(
    parameter int NUM_CHIPS       = 4,
    parameter int CHIPS_PER_CARD  = 4,
    parameter int HIDDEN          = lpu_config_pkg::LPU_HIDDEN,
    parameter int INTER           = lpu_config_pkg::LPU_INTERMEDIATE,
    parameter int SINGLE_CHIP     = 0
) (
    input  logic clk, rst_n,

`ifndef FPGA_LPU_SINGLE_CHIP
    // === C2C Dual Ring (chip-to-chip, within card) ===
    // Ring A: chip[i].tx_a → chip[i+1].rx_a (clockwise)
    // Ring B: chip[i].tx_b → chip[i-1].rx_b (counter-clockwise)
    input  c2c_link_t c2c_rx_a_ext,   // external ring A input (from previous card)
    output c2c_link_t c2c_tx_a_ext,   // external ring A output (to next card)
    input  c2c_link_t c2c_rx_b_ext,
    output c2c_link_t c2c_tx_b_ext,

    // === PCIe (card-level, Chip 0 only) ===
    input  pcie_dma_stream_t pcie_host,
    output pcie_dma_stream_t pcie_fpga,
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

    input  logic                         scale_wr_en,
    input  logic [$clog2(lpu_config_pkg::LPU_SCALE_GROUPS)-1:0] scale_wr_addr,
    input  logic [7:0]                   scale_wr_data,

    input  logic [$clog2(lpu_config_pkg::LPU_EXPERTS_PER_FPGA > 1 ? lpu_config_pkg::LPU_EXPERTS_PER_FPGA : 2)-1:0] ffn_expert_sel,
    input  logic [lpu_config_pkg::LPU_NUM_EXPERTS-1:0] cfg_local_experts [NUM_CHIPS],

    input  logic                         valid_in,
    input  logic [HIDDEN*32-1:0]         a_flat,

    output logic                         valid_out,
    output logic                         router_ok,
    output logic [HIDDEN*32-1:0]         y_flat
);

    // =========================================================================
    // Pipeline interconnect: chip[i].y_flat → chip[i+1].a_flat
    // =========================================================================
    logic [NUM_CHIPS-1:0]        chip_valid_in;
    logic [NUM_CHIPS-1:0]        chip_valid_out;
    logic [NUM_CHIPS-1:0]        chip_router_ok;
    logic [HIDDEN*32-1:0]        chip_a_flat [NUM_CHIPS];
    logic [HIDDEN*32-1:0]        chip_y_flat [NUM_CHIPS];

    // First chip receives external input
    assign chip_valid_in[0] = valid_in;
    assign chip_a_flat[0]  = a_flat;

    // Pipeline forwarding: chip[i] output → chip[i+1] input
    generate
        for (genvar i = 1; i < NUM_CHIPS; i++) begin : g_pipeline_chain
            assign chip_valid_in[i] = chip_valid_out[i-1];
            assign chip_a_flat[i]   = chip_y_flat[i-1];
        end
    endgenerate

    // Last chip output → external
    assign valid_out  = chip_valid_out[NUM_CHIPS-1];
    assign router_ok  = chip_router_ok[NUM_CHIPS-1];
    assign y_flat     = chip_y_flat[NUM_CHIPS-1];

    // =========================================================================
    // Chip instances
    // =========================================================================
    generate
        for (genvar gi = 0; gi < NUM_CHIPS; gi++) begin : g_chip
            localparam int LAYER_START = gi * 2;       // 2 layers per chip
            localparam int LAYER_END   = gi * 2 + 1;
            localparam int CARD_ID     = gi / CHIPS_PER_CARD;
            localparam int PCIE_MASTER = (gi % CHIPS_PER_CARD == 0) ? 1 : 0;

`ifndef FPGA_LPU_SINGLE_CHIP
            // Per-chip C2C/PCIe internal wires
            c2c_link_t         c2c_rx_a_w, c2c_tx_a_w;
            c2c_link_t         c2c_rx_b_w, c2c_tx_b_w;
            pcie_dma_stream_t  pcie_host_w, pcie_fpga_w;

            // Only chip 0 drives external C2C ring + PCIe ports.
            // Future: per-card masters with inter-card C2C wiring.
            if (gi == 0) begin : g_ext_master
                assign c2c_rx_a_w  = c2c_rx_a_ext;
                assign c2c_tx_a_ext = c2c_tx_a_w;
                assign c2c_rx_b_w  = c2c_rx_b_ext;
                assign c2c_tx_b_ext = c2c_tx_b_w;
                assign pcie_host_w = pcie_host;
                assign pcie_fpga   = pcie_fpga_w;
            end else begin : g_ext_slave
                assign c2c_rx_a_w  = '0;
                assign c2c_rx_b_w  = '0;
                assign pcie_host_w = '0;
            end
`endif

            chip_top #(
                .CHIP_ID        (gi),
                .CARD_ID        (CARD_ID),
                .LAYER_START    (LAYER_START),
                .LAYER_END      (LAYER_END),
                .IS_PCIE_MASTER (PCIE_MASTER),
                .SINGLE_CHIP    (SINGLE_CHIP),
                .HIDDEN         (HIDDEN),
                .INTER          (INTER)
            ) u_chip (
                .clk            (clk),
                .rst_n          (rst_n),
`ifndef FPGA_LPU_SINGLE_CHIP
                .c2c_rx_a       (c2c_rx_a_w),
                .c2c_tx_a       (c2c_tx_a_w),
                .c2c_rx_b       (c2c_rx_b_w),
                .c2c_tx_b       (c2c_tx_b_w),
                .pcie_host      (pcie_host_w),
                .pcie_fpga      (pcie_fpga_w),
                .c2c_proxy      (),
`endif
                .gamma_wr_en    (gamma_wr_en),
                .gamma_wr_idx   (gamma_wr_idx),
                .gamma_wr_data  (gamma_wr_data),
                .attn_qkv_wt_wr_en   (attn_qkv_wt_wr_en),
                .attn_qkv_wt_sel     (attn_qkv_wt_sel),
                .attn_qkv_wt_row     (attn_qkv_wt_row),
                .attn_qkv_wt_col     (attn_qkv_wt_col),
                .attn_qkv_wt_wr_data (attn_qkv_wt_wr_data),
                .attn_rope_lut_wr_en (attn_rope_lut_wr_en),
                .attn_rope_lut_pos   (attn_rope_lut_pos),
                .attn_rope_lut_pair  (attn_rope_lut_pair),
                .attn_rope_lut_sin   (attn_rope_lut_sin),
                .attn_rope_lut_cos   (attn_rope_lut_cos),
                .token_position      (token_position),
                .cache_preload_en    (cache_preload_en),
                .cache_preload_K_flat(cache_preload_K_flat),
                .cache_preload_V_flat(cache_preload_V_flat),
                .rtr_w_wr_en    (rtr_w_wr_en),
                .rtr_w_wr_expert(rtr_w_wr_expert),
                .rtr_w_wr_idx   (rtr_w_wr_idx),
                .rtr_w_wr_data  (rtr_w_wr_data),
                .gate_w_wr_en   (gate_w_wr_en),
                .up_w_wr_en     (up_w_wr_en),
                .down_w_wr_en   (down_w_wr_en),
                .gate_w_wr_row  (gate_w_wr_row),
                .up_w_wr_row    (up_w_wr_row),
                .down_w_wr_row  (down_w_wr_row),
                .gate_w_wr_beat (gate_w_wr_beat),
                .up_w_wr_beat   (up_w_wr_beat),
                .down_w_wr_beat (down_w_wr_beat),
                .gate_w_wr_data (gate_w_wr_data),
                .up_w_wr_data   (up_w_wr_data),
                .down_w_wr_data (down_w_wr_data),
                .scale_wr_en    (scale_wr_en),
                .scale_wr_addr  (scale_wr_addr),
                .scale_wr_data  (scale_wr_data),
                .ffn_expert_sel (ffn_expert_sel),
                .cfg_local_experts(cfg_local_experts[gi]),
                .valid_in       (chip_valid_in[gi]),
                .a_flat         (chip_a_flat[gi]),
                .valid_out      (chip_valid_out[gi]),
                .router_ok      (chip_router_ok[gi]),
                .y_flat         (chip_y_flat[gi])
            );
        end
    endgenerate

endmodule
