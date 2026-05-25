//=============================================================================
// c2c_packet.svh — Chip-to-Chip SerDes frame format
//
// F-Tile SerDes supports up to 4088-byte payloads per frame.
// We define a header + payload format that multiplexes pipeline forward,
// MoE dispatch, and MoE reduce onto the same physical link.
//
// Frame overhead: 24B header + CRC
// Max payload:    4088B
//=============================================================================

`ifndef C2C_PACKET_SVH
`define C2C_PACKET_SVH

`include "avalon_stream.svh"

// C2C header: identifies message type and routing info
typedef struct packed {
    logic [2:0]  msg_type;     // 0=PIPE_FWD, 1=MOE_DISP, 2=MOE_REDU, 3=CTRL
    logic        ring;         // 0=Ring A (CW), 1=Ring B (CCW)
    logic [7:0]  src_chip;     // origin chip global_id
    logic [7:0]  dst_chip;     // final destination chip global_id
    logic [15:0] seq_id;       // sequence number for reorder/drop detection
    logic [15:0] payload_len;  // bytes in payload (0-4088)
} c2c_header_t;

// Control message: register read/write, expert bitmap config
typedef struct packed {
    logic        is_write;
    logic [15:0] addr;
    logic [31:0] data;
} c2c_ctrl_t;

parameter C2C_MSG_PIPE_FWD  = 3'd0;
parameter C2C_MSG_MOE_DISP  = 3'd1;
parameter C2C_MSG_MOE_REDU  = 3'd2;
parameter C2C_MSG_CTRL      = 3'd3;

// Per-chip C2C interface (one per ring direction)
typedef struct packed {
    logic        tx_valid;
    logic        tx_ready;
    c2c_header_t tx_header;
    logic [4087:0] tx_payload;   // max payload (8-bit aligned)

    logic        rx_valid;
    logic        rx_ready;
    c2c_header_t rx_header;
    logic [4087:0] rx_payload;
} c2c_link_t;

`endif
