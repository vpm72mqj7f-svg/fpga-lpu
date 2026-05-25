//=============================================================================
// avalon_stream.svh — lightweight valid/ready streaming interface
//
// Used for all token-level data movement: pipeline forward, MoE dispatch,
// activation ingress/egress. Packed struct avoids port explosion.
//=============================================================================

`ifndef AVALON_STREAM_SVH
`define AVALON_STREAM_SVH

typedef struct packed {
    logic        valid;
    logic [7:0]  dest;      // destination chip global_id (0-31)
    logic [7:0]  src;       // source chip global_id
    logic [5:0]  layer;     // layer index (0-60) or layer bitmap for expert dispatch
    logic [15:0] token_id;  // token/session identifier
    logic [31:0] data;      // payload (hidden state element, Q12)
} stream_beat_t;

// Pipeline forward: one beat per (token, layer) to pass hidden state to next chip
typedef struct packed {
    logic        valid;
    logic [7:0]  src_chip;
    logic [7:0]  dst_chip;
    logic [15:0] token_id;
    logic [7:0]  byte_en;   // bytes valid in hidden_state
    logic [31:0] hidden;    // one element of hidden state (Q12)
} pipeline_fwd_beat_t;

// MoE dispatch: send activation to remote chip for expert computation
typedef struct packed {
    logic        valid;
    logic [7:0]  src_chip;
    logic [7:0]  dst_chip;
    logic [11:0] expert_id;
    logic [15:0] token_id;
    logic [31:0] activation; // hidden state element (FP8 encoded)
} moe_dispatch_beat_t;

// MoE reduce: remote expert result sent back to requesting chip
typedef struct packed {
    logic        valid;
    logic [7:0]  src_chip;
    logic [7:0]  dst_chip;
    logic [11:0] expert_id;
    logic [15:0] token_id;
    logic [31:0] result;     // expert output element (Q12)
} moe_reduce_beat_t;

`endif
