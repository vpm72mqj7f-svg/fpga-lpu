//=============================================================================
// lpu_config.svh — FPGA LPU Shared Configuration Package
//
// Two modes, controlled by `define FPGA_LPU_PRODUCTION:
//
//   Simulation (default): HIDDEN=8, INTER=4  — fast, Icarus-friendly
//   Production:            HIDDEN=7168, INTER=3072 — DeepSeek V4 Pro
//
// Usage:
//   In Quartus QSF (production):
//     set_global_assignment -name VERILOG_MACRO "FPGA_LPU_PRODUCTION"
//
//   In iverilog (simulation):
//     iverilog -g2012 ...  (no define needed, defaults to bring-up)
//
//   Per-module override:
//     full_transformer_layer #(.HIDDEN(256)) ...  (overrides package default)
//=============================================================================

`ifndef LPU_CONFIG_SVH
`define LPU_CONFIG_SVH

package lpu_config_pkg;

`ifdef FPGA_LPU_PRODUCTION

    // ========================================================================
    // PRODUCTION: DeepSeek V4 Pro
    // ========================================================================
    parameter int LPU_HIDDEN          = 7168;    // model hidden dimension
    parameter int LPU_INTERMEDIATE    = 3072;    // FFN intermediate dimension
    parameter int LPU_K_LATENT        = 512;     // MLA K low-rank dimension
    parameter int LPU_V_LATENT        = 512;     // MLA V low-rank dimension
    parameter int LPU_NUM_HEADS       = 128;     // attention heads
    parameter int LPU_QK_ROPE_DIM     = 64;      // RoPE dimension per head
    parameter int LPU_V_HEAD_DIM      = 128;     // V head dimension
    parameter int LPU_NUM_EXPERTS     = 384;     // total MoE experts
    parameter int LPU_TOP_K           = 6;       // routed experts per token
    parameter int LPU_EXPERTS_PER_FPGA = 12;     // base experts per chip (unique)
    parameter int LPU_HOT_EXPERTS     = 12;      // hot experts to replicate
    parameter int LPU_HOT_REPLICAS    = 16;      // replicas per hot expert (1=no rep)
    parameter int LPU_MAX_SEQ_LEN     = 4096;    // maximum sequence length (positions)
    parameter int LPU_SLIDING_WINDOW  = 128;     // sliding window attention size
    parameter int LPU_NUM_LAYERS      = 61;      // transformer layers
    parameter int LPU_VOCAB_SIZE      = 129280;  // vocabulary size

    // Total experts per chip (base + replicas)
    parameter int LPU_TOTAL_PER_FPGA  = LPU_EXPERTS_PER_FPGA + LPU_HOT_EXPERTS;

    // Systolic array sizing (per chip)
    parameter int LPU_ARRAY_LANES     = 128;     // K-direction parallelism
    parameter int LPU_ARRAY_M_ROWS    = 32;      // M-direction parallelism
    parameter int LPU_ACCUM_WIDTH     = 32;      // accumulator width

    // Memory sizing
    parameter int LPU_KV_CACHE_SLOTS  = 4096;    // KV cache entries
    parameter int LPU_SCALE_GROUPS    = 448;     // fp4 scale groups (7168/16)
    parameter int LPU_WEIGHT_WIDTH    = 16;      // Q12 signed weight width
    parameter int LPU_DATA_WIDTH      = 32;      // Q12 hidden state width

    // Clock frequencies (MHz)
    parameter int LPU_CLK_DSP_MHZ     = 450;     // DSP/HBM clock
    parameter int LPU_CLK_SYS_MHZ     = 100;     // System/control clock
    parameter int LPU_CLK_PCIE_MHZ    = 250;     // PCIe clock

`else

    // ========================================================================
    // BRING-UP / SIMULATION: Minimal parameters for fast iteration
    // ========================================================================
    parameter int LPU_HIDDEN          = 8;
    parameter int LPU_INTERMEDIATE    = 4;
    parameter int LPU_K_LATENT        = 4;
    parameter int LPU_V_LATENT        = 4;
    parameter int LPU_NUM_HEADS       = 4;
    parameter int LPU_QK_ROPE_DIM     = 4;
    parameter int LPU_V_HEAD_DIM      = 8;
    parameter int LPU_NUM_EXPERTS     = 4;
    parameter int LPU_TOP_K           = 2;
    parameter int LPU_EXPERTS_PER_FPGA = 4;       // base experts per chip
    parameter int LPU_HOT_EXPERTS     = 0;        // no replication in bring-up
    parameter int LPU_HOT_REPLICAS    = 1;        // (unused when HOT_EXPERTS=0)
    parameter int LPU_MAX_SEQ_LEN     = 64;       // maximum sequence length (positions)
    parameter int LPU_SLIDING_WINDOW  = 128;     // sliding window size (>=NUM_SLOTS→full attention)
    parameter int LPU_NUM_LAYERS      = 12;
    parameter int LPU_TOTAL_PER_FPGA  = LPU_EXPERTS_PER_FPGA + LPU_HOT_EXPERTS;
    parameter int LPU_VOCAB_SIZE      = 16;

    // Systolic array sizing (small, for iverilog)
    parameter int LPU_ARRAY_LANES     = 8;
    parameter int LPU_ARRAY_M_ROWS    = 4;
    parameter int LPU_ACCUM_WIDTH     = 32;

    // Memory sizing (small)
    parameter int LPU_KV_CACHE_SLOTS  = 64;
    parameter int LPU_SCALE_GROUPS    = 4;       // 8 elements/16 = 1, rounded up
    parameter int LPU_WEIGHT_WIDTH    = 16;
    parameter int LPU_DATA_WIDTH      = 32;

    // Clock frequencies (MHz)
    parameter int LPU_CLK_DSP_MHZ     = 100;     // simulation at 100 MHz
    parameter int LPU_CLK_SYS_MHZ     = 100;
    parameter int LPU_CLK_PCIE_MHZ    = 100;

`endif

endpackage

`endif // LPU_CONFIG_SVH
