//=============================================================================
// pcie_dma.svh — PCIe 5.0 DMA descriptor and interface (Chip 0 only)
//
// R-Tile PCIe 5.0 x16 Hard IP provides BAR0 (MMIO registers, 64MB)
// and BAR2 (HBM aperture, 32GB). The DMA engine on Chip 0 exposes
// a descriptor ring for Host → FPGA and FPGA → Host transfers.
//=============================================================================

`ifndef PCIE_DMA_SVH
`define PCIE_DMA_SVH

// DMA descriptor (Host ↔ FPGA)
typedef struct packed {
    logic [63:0] src_addr;     // 64-bit physical address
    logic [63:0] dst_addr;
    logic [31:0] length;       // bytes to transfer
    logic [15:0] flags;        // bit0: intr_on_done, bit1: chain
    logic [15:0] status;       // written by DMA engine on completion
} pcie_dma_desc_t;

// BAR0 register map (Chip 0 only)
typedef struct packed {
    logic [31:0] ctrl;          // [0] go, [1] reset, [3:2] mode
    logic [31:0] status;        // [0] done, [1] error, [15:8] chip_id
    logic [63:0] desc_ring_base; // host physical address of descriptor ring
    logic [15:0] desc_ring_size; // number of descriptors
    logic [15:0] desc_head;      // written by host (next desc to submit)
    logic [15:0] desc_tail;      // written by FPGA (next desc completed)
    logic [31:0] irq_mask;       // MSI-X interrupt mask
    logic [31:0] perf_counters [8]; // DMA performance counters
} pcie_bar0_regs_t;

// PCIe DMA engine interface (Chip 0 internal)
typedef struct packed {
    // Host → FPGA
    logic        h2f_valid;
    logic        h2f_ready;
    logic [31:0] h2f_data;
    logic [3:0]  h2f_keep;
    logic        h2f_last;

    // FPGA → Host
    logic        f2h_valid;
    logic        f2h_ready;
    logic [31:0] f2h_data;
    logic [3:0]  f2h_keep;
    logic        f2h_last;
} pcie_dma_stream_t;

// C2C proxy bridge: forwards Chip 1-3 traffic to/from PCIe
typedef struct packed {
    logic        upstream_valid;
    logic        upstream_ready;
    logic [7:0]  upstream_chip;   // which chip (1-3) the traffic belongs to
    logic [31:0] upstream_data;

    logic        downstream_valid;
    logic        downstream_ready;
    logic [7:0]  downstream_chip;
    logic [31:0] downstream_data;
} pcie_c2c_proxy_t;

`endif
