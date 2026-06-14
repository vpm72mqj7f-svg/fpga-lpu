# V2-Lite HBM2 Weight Address Map

**S10 MX HBM2: 4 GB, AXI4 28-bit address (256 MB window), 256-bit data**

---

## 1. Per-Expert Weight Layout (FP8, 1 byte/element)

```
V2-Lite: d=2048, inter=1408

Expert N base = N × EXPERT_STRIDE (9 MB)

Offset      Size        Content             AXI read burst
─────────────────────────────────────────────────────────
0x000000    2,883,584   gate[2048×1408]     ARADDR=base+0x000000, LEN=255
0x2C0000    2,883,584   up[2048×1408]       ARADDR=base+0x2C0000, LEN=255
0x580000    2,883,584   down[1408×2048]     ARADDR=base+0x580000, LEN=255
0x840000    1,048,576   (padding to 9MB)    ─
─────────────────────────────────────────────────────────
EXPERT_STRIDE = 0x900000 = 9,437,184 bytes = 9 MB
```

## 2. Expert Slot Mapping (6 experts for V2-Lite Phase 1)

```
Expert ID    HBM2 Base Address    Content
───────────────────────────────────────────────────────
0            0x00000000           gate + up + down
1            0x00900000
2            0x01200000
3            0x01B00000
4            0x02400000
5            0x02D00000
───────────────────────────────────────────────────────
Total: 6 × 9 MB = 54 MB (out of 4 GB)
```

## 3. AXI Read Parameters

```
Parameter           Value
───────────────────────────────────────────────
AXI data width      256-bit (32 bytes/beat)
AXI address width   28-bit
AXI burst type      INCR
AXI burst length    255 (256 beats × 32B = 8,192 bytes/burst)
ARADDR step         8,192 bytes between bursts
ARADDR alignment    256-bit (32-byte aligned)

Gate reads:  ceil(2,883,584 / 8,192) = 352 bursts
Up reads:    ceil(2,883,584 / 8,192) = 352 bursts
Down reads:  ceil(2,883,584 / 8,192) = 352 bursts
Total per expert: 1,056 bursts
```

## 4. FFN Read Sequence (per expert, per GEMV row)

```
For each row r = 0..OUTPUT_DIM-1:

  Gate:  read INPUT_DIM elements starting at expert_base + 0x000000
         AXI: ARADDR = base + r*INPUT_DIM*1, LEN = ceil(INPUT_DIM/32)-1
         = 2048 elements ÷ 32 = 64 beats → ARLEN = 63

  Up:    same as Gate (same weight matrix dimensions)

  Down:  read INPUT_DIM elements starting at expert_base + 0x580000
         AXI: ARADDR = base + 0x580000 + r*INTER*1, LEN = ceil(INTER/32)-1
         = 1408 elements ÷ 32 = 44 beats → ARLEN = 43
```

## 5. Self-Test Bringup: Preload Expert 0

```
Phase 1 bringup:
  1. Host writes test weights to Expert 0 via PCIe BAR0 WT engine
  2. FFN reads Expert 0 from HBM2 via AXI4 (ffn_axi_ar*)
  3. This establishes the real HBM2→DSP data path, preventing Quartus DSP removal
```

## 6. Complete Address Table (RTL-ready)

```
Constants:
  EXPERT_STRIDE  = 9,437,184 = 0x900000
  GATE_OFFSET    = 0
  UP_OFFSET      = 2,883,584 = 0x2C0000
  DOWN_OFFSET    = 5,767,168 = 0x580000
  ROW_BYTES_GATE = 2,048     (d × 1B FP8)
  ROW_BYTES_DOWN = 1,408     (inter × 1B FP8)
  ARLEN_GATE     = 63         (2048/32 - 1)
  ARLEN_UP       = 63         (same as gate)
  ARLEN_DOWN     = 43         (1408/32 - 1)
  BURSTS_GATE    = 352        (2,883,584 / 8,192)
  BURSTS_DOWN    = 352        (same)
```

### Expert 0 (base = 0x00000000)

| Row | Gate ARADDR (hex) | Down ARADDR (hex) |
|-----|-------------------|-------------------|
| 0   | 0x00000000        | 0x00580000        |
| 1   | 0x00000800        | 0x00580580        |
| 2   | 0x00001000        | 0x00580B00        |
| ... | ...               | ...               |
| r   | 0x000000 + r×2048 | 0x580000 + r×1408 |
| 1407| 0x002BFF80        | 0x007BF580        |

### Expert 1 (base = 0x00900000)

| Row | Gate ARADDR (hex) | Down ARADDR (hex) |
|-----|-------------------|-------------------|
| 0   | 0x00900000        | 0x00E80000        |
| r   | 0x900000 + r×2048 | 0xE80000 + r×1408 |

### Expert N (base = N × 0x00900000)

| Row | Gate ARADDR                    | Down ARADDR                    |
|-----|--------------------------------|--------------------------------|
| 0   | N × 0x00900000                 | N × 0x00900000 + 0x00580000   |
| r   | base + r × 2048                | base + 0x580000 + r × 1408    |

## 7. RTL Address Generator

```systemverilog
// In v2_lite_ffn_engine.sv or hbm2_weight_reader.sv:

localparam int EXPERT_STRIDE     = 28'h0900000;  // 9 MB
localparam int GATE_OFFSET       = 28'h0000000;  // gate weight offset
localparam int UP_OFFSET         = 28'h02C0000;  // up weight offset
localparam int DOWN_OFFSET       = 28'h0580000;  // down weight offset
localparam int ROW_BYTES_GATE    = 2048;         // d × 1B
localparam int ROW_BYTES_DOWN    = 1408;         // inter × 1B
localparam int BURST_LEN_GATE    = 63;           // 2048/32 - 1
localparam int BURST_LEN_DOWN    = 43;           // 1408/32 - 1

// Expert base address (selected by current expert FSM)
wire [27:0] expert_base;
assign expert_base = (expert_id * EXPERT_STRIDE);

// Gate/Up row read address (same dimensions for gate and up)
wire [27:0] gate_row_addr;
assign gate_row_addr = expert_base + GATE_OFFSET + (row_idx * ROW_BYTES_GATE);

// Down row read address
wire [27:0] down_row_addr;
assign down_row_addr = expert_base + DOWN_OFFSET + (row_idx * ROW_BYTES_DOWN);

// Address mux per FSM state
always_comb begin
    case (fsm_state)
        LOAD_GATE: begin
            m_axi_araddr = gate_row_addr;
            m_axi_arlen  = BURST_LEN_GATE;
        end
        LOAD_UP: begin
            m_axi_araddr = gate_row_addr;  // same as gate
            m_axi_arlen  = BURST_LEN_GATE;
        end
        LOAD_DOWN: begin
            m_axi_araddr = down_row_addr;
            m_axi_arlen  = BURST_LEN_DOWN;
        end
        default: begin
            m_axi_araddr = 28'd0;
            m_axi_arlen  = 8'd0;
        end
    endcase
end
```
