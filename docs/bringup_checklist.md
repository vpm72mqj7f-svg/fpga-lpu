# FPGA LPU Phase 1 Bring-Up Checklist

> **Target**: DK-DEV-AGM039EA (Intel Agilex 7 M-Series HBM2e Development Kit)
> **FPGA Device**: AGMF039R47A1E2VR0 (ES) / AGMF039R47A1E1VC (production)
> **User Guide**: Intel doc 782461 (Revision 2025.07.30)
> **Kit Contents**: Dev board + DDR5 16GB DIMM + IO48 HPS daughter board + 240W PSU
> **Price**: ~$8,000-12,000 (Mouser/element14)
> **Duration**: 8 weeks

### Quick Reference Card

| Item | Spec |
|------|------|
| FPGA | AGMF039R47A (R-Tile + F-Tile ×3 + HBM2e) |
| DSP | 12,300 (supports fp4×fp8 native: 11.07 TMACs) |
| HBM2e | 32 GB, 920 GB/s, 2048-bit Avalon-MM |
| PCIe | 5.0 ×16 via MCIO cable (Amphenol HMC74-0631, sold separately) |
| DDR5 | 16 GB RDIMM (Micron MTC10F1084S1RC56BG1) |
| QSFP-DD | 2× ports (F-Tile ×8 transceivers each) |
| QSFP-DD800 | 1× port (F-Tile ×4 transceivers) |
| FMC+ | 1× connector (×16 FGT lanes + 64 I/O) |
| Clocks on board | 100 MHz (PCIe), 156.25 MHz, 245.76 MHz, 312.50 MHz, 390.625 MHz |
| DSP target | 390.625 MHz × PLL → ~450 MHz |
| Power | 240W adapter, SmartVID (LTC3888 @ 0x55) |
| Board design files | `board_design_files/` in kit installer (schematics, layout, BOM) |
| BSP download | Intel FPGA Dev Kit page → "Agilex 7 M-Series HBM2e Dev Kit" |

---

## Pre-Board (Week 0: Before the board arrives)

### □ 0.1 Software tools

- [ ] Install Quartus Prime Pro 24.3+ (with Agilex 7 M-Series device support)
- [ ] Install Intel HBM2e IP license (included with Quartus Pro)
- [ ] Install R-Tile PCIe 5.0 IP license (included with Quartus Pro)
- [ ] Download DK-DEV-AGM039EA BSP from Intel/Altera website
  - Direct link: https://www.intel.com/content/www/us/en/products/details/fpga/development-kits/agilex/agm039.html
  - The BSP installer includes `board_design_files/` (schematics, layout, BOM)
  - Also includes `examples/` with Golden Top, BTS, PCIe, Memory, and XCVR design examples
- [ ] Install the BSP: run `./install.sh` (Linux) or `install.bat` (Windows)
- [ ] Verify `quartus_sh --version`
- [ ] Build and run the BSP's Golden Top example design
  - This validates: PCIe link up, HBM read/write, DDR5 access, clock tree, power sequencing
  - If this doesn't work, everything else is blocked — debug until it passes

### □ 0.2 RTL code freeze for Phase 1

- [ ] Set up git tag `phase1-freeze` on the current RTL
- [ ] Run `quartus_map` on `hw/quartus/fpga_lpu.qpf` (target: AGMF039R47A1E2VR0)
- [ ] Fix any Quartus-specific syntax issues:
  - Quartus Pro 24.3 supports SystemVerilog 2017
  - Check: unpacked array indexing, `$clog2` usage, generate blocks
  - Common issues: part-select width mismatch, `always_comb` sensitivity list
- [ ] Run `quartus_sta` to check setup/hold on every module
  - Expected: many timing violations on first pass — this is normal
  - Gate: no more than 20 unique failing paths after Week 2
- [ ] Generate SmartVID constraint file (required for AGM 039):
  - Quartus auto-generates from device OPN: `quartus_sh --flow compile` extracts VID settings
  - Add `set_global_assignment -name USE_PWRMGT_SCL SDM_IO14` (per doc 782461)

### □ 0.3 Python golden model freeze

- [ ] Freeze `scripts/simulation/gen_tb_vectors.py` — 15 golden MAC tests
- [ ] Freeze `scripts/simulation/gen_ffn_tb_vectors.py` — 2 Expert FFN golden cases
- [ ] Freeze `scripts/simulation/gen_layer_golden.py` — 2 layer golden cases
- [ ] Generate all golden vectors and commit as `hw/test_vectors/*.hex`

### □ 0.4 QSYS / Platform Designer setup

- [ ] Generate HBM2e subsystem in QSYS
  - 1 HBM2e stack, 32 GB, 2048-bit Avalon-MM interface
  - Set reference clock to board oscillator frequency
- [ ] Generate PCIe 5.0 EP subsystem in QSYS
  - R-Tile, x16, Gen5
  - BAR0: 64KB MMIO (control registers)
  - BAR2: 32GB prefetchable (HBM aperture)
- [ ] Export QSYS to `hw/ip/hbm_sys/` and `hw/ip/pcie_sys/`
- [ ] Instantiate both in `hw/src/top.sv`

---

## Week 1-2: Board Arrival — Hello World + HBM

### □ W1.1 Power-on check

- [ ] Connect board to host PC via PCIe slot (or PCIe cable if using external enclosure)
- [ ] Connect 12V AUX power if required
- [ ] Install Intel FPGA SDK for OpenCL / oneAPI (for driver)
- [ ] `lspci` should show Intel device 0x0000 (or similar Agilex ID)
- [ ] Load default bitstream from BSP — verify PCIe link up at Gen5 x16

### □ W1.2 HBM sequential R/W test

- [ ] Load bitstream with HBM example design
- [ ] Write 1GB pattern (0xDEADBEEF) to HBM base address via PCIe BAR2
- [ ] Read back 1GB, compare
- [ ] Measure sequential bandwidth: `bytes_transferred / time`
  - Target: ≥ 800 GB/s (87% of theoretical 920)
  - Gate: ≥ 700 GB/s
- [ ] If < 700 GB/s: check HBM reference clock, PLL lock, Avalon-MM burst length

### □ W1.3 LED heartbeat + UART

- [ ] Load bitstream with `top.sv` as-is (placeholder)
- [ ] Verify `debug_led[0]` blinks at ~1 Hz
- [ ] Verify `uart_tx` sends "FPGA LPU boot\r\n" string at 115200 baud
  - [TODO] Implement UART TX if not already done — ~50 lines SV

---

## Week 2-3: fp4 MAC Precision (Go/No-Go #1)

### □ W2.1 Scale memory load via MMIO

- [ ] Implement AXI4-Lite → register bridge for `scale_wr_en/addr/data`
  - 512 writes to fill scale memory (group_size=16, up to 8192 elements)
  - Map to BAR0 offset 0x0000_1000
- [ ] Verify scale memory content by reading back via register bridge
- [ ] Test: write 0x38 to group 0, verify readback

### □ W2.2 MAC golden vector runner

- [ ] Implement FSM that:
  1. Loads scale memory (512 groups)
  2. Loads MAC test inputs (weight, activation, scale index) from BAR0 MMIO
  3. Pulses `accum_clr` before each test sequence
  4. Streams inputs to fp4_mac (back-to-back valid)
  5. Waits for pipeline drain (8 cycles after last input)
  6. Latches `mac_result` into a readable register
  7. Asserts `test_done` interrupt bit
- [ ] Test: run T1 (single multiply: fp4 +1.0 × fp8 +1.0) → verify result = 0x00001000

### □ W2.3 Signal Tap setup

- [ ] Open Signal Tap II, create `hw/scripts/debug.stp`
- [ ] Add trigger: `mac_valid_in && weight == 4'h4` (first beat of a test)
- [ ] Add capture nodes:
  - `u_mac.s0_weight` (4b) — fp4 input
  - `u_mac.s1_w_signed` (8b) — decoded fp4
  - `u_mac.s1_a_scaled` (12b) — decoded fp8 activation
  - `u_mac.s1_sc_scaled` (12b) — decoded fp8 scale
  - `u_mac.s2_product` (32b) — scaled product
  - `u_mac.accumulator` (32b) — running accumulator
  - `u_mac.mac_out.result` (32b) — final result
- [ ] Set sample depth: 4K samples
- [ ] Set clock: `clk_dsp`

### □ W2.4 Run all 15 golden tests on hardware

- [ ] Python script `hw/scripts/run_golden_tests.py`:
  - Reads `tb_golden_pkg.sv` → extracts 15 test vectors
  - Writes each test to FPGA via PCIe MMIO (BAR0)
  - Reads back MAC result
  - Compares to expected (already in Python golden)
  - Reports PASS/FAIL per test
- [ ] Run full suite
  - Target: 15/15 PASS
  - Gate: ≥ 13/15 PASS (Go/No-Go #1)

### □ W2.5 Per-bit comparison (if any test fails)

- [ ] Set Signal Tap trigger to the specific failing test's input beat
- [ ] Capture pipeline waveform
- [ ] Compare each pipeline stage output to Python golden:
  - Stage 0: s0_weight/s0_activ (should match Python decode input)
  - Stage 1: s1_w_signed / s1_a_scaled / s1_sc_scaled
  - Stage 2: s2_product (multiply result)
  - Stage 3: accumulator
- [ ] Root-cause any mismatch → fix RTL, re-run

### □ Go/No-Go #1 decision

```
✓ PASS: 15/15 tests match Python golden, cosine ≥ 0.995
        → Continue to HBM bandwidth experiment

△ WARN: 13-14/15 pass, one test has small rounding diff (≤ 2 ULP)
        → Analyze root cause; if DSP rounding artifact (not logic bug), continue
        → Document as "known DSP behavior" for future reference

✗ STOP: < 13/15 pass, or any mismatch > 2 ULP
        → Debug with Signal Tap for 1 week
        → If not resolved, consider fp8 fallback path
        → Decision: continue Phase 1 with fp8 weights or stop project
```

---

## Week 3-5: HBM Bandwidth (Go/No-Go #2)

### □ W3.1 Sequential bandwidth calibration

- [ ] DMA engine (or memcpy loop via PCIe BAR2) writes 1GB pattern
- [ ] Read back sequentially, measure time
- [ ] Plot bandwidth vs transfer size (64B → 32MB)
- [ ] Record: peak bandwidth, 90% bandwidth point, minimum efficient size

### □ W3.2 MoE expert random access pattern

- [ ] Place 12 expert blocks (33 MB each) in HBM at known addresses
- [ ] Generate Zipf-distributed access trace (α=1.0):
  - Python: `scripts/simulation/gen_hbm_trace.py` → 100K addresses
  - Load trace into FPGA BRAM (or stream from PCIe)
- [ ] DMA engine reads from trace addresses, measures total_time
- [ ] Effective BW = total_bytes_read / total_time
- [ ] Also measure: max single-access latency, p99 latency

### □ W3.3 Bank conflict measurement

- [ ] Repeat random access test with different expert layouts:
  - A) Naive: contiguous 33MB blocks (default)
  - B) Interleaved: 32KB stripes across pseudo-channels
  - C) Hashed: address XOR with token_id to spread access
- [ ] Compare effective bandwidth of A vs B vs C
- [ ] Record: bank conflict count (from HBM performance counters, if available)

### □ W3.4 Dual-buffer overlap test

- [ ] Implement weight double-buffer: buffer_A + buffer_B
- [ ] FSM: load buffer_B from HBM while DSP computes from buffer_A
- [ ] Signal Tap simultaneously triggers on:
  - `hbm_read_req_valid` (HBM read active)
  - `dsp_compute_busy` (DSP busy)
- [ ] Compute overlap ratio:
  - overlap% = (cycles_both_active / total_cycles) × 100
  - Target: ≥ 80%

### □ Go/No-Go #2 decision

```
✓ PASS: MoE random BW ≥ 550 GB/s, overlap ≥ 80%
        → Continue to single-layer end-to-end

△ WARN: 400-550 GB/s, overlap 60-80%
        → Throughput will be 20-30% lower than modelled
        → Still viable; recalculate $/M token with corrected numbers

✗ STOP: < 400 GB/s or overlap < 60%
        → Reassess: increase per-chip HBM, reduce expert count, or accept lower perf
```

---

## Week 5-7: Single-Layer End-to-End (Go/No-Go #3)

### □ W5.1 Synthesize full_transformer_layer

- [ ] Instantiate `full_transformer_layer.sv` in `top.sv`
- [ ] Generate weight preload data from existing Python golden generator
- [ ] Map weight preload to BAR0 MMIO writes (hundreds of writes — OK for bring-up)
- [ ] Run Quartus full compilation
- [ ] Check resource usage: DSP count, BRAM, logic utilization
- [ ] Fix any timing violations (especially in the combinational softmax block)

### □ W5.2 Run layer golden case on hardware

- [ ] Load C0 golden weights (uniform input, identity attention/V/FFN)
- [ ] Trigger `valid_in` → wait for `valid_out`
- [ ] Read `y0..y7` from result registers
- [ ] Compare to Python golden expected values
- [ ] C0 target: exact match on all 8 outputs (FPGA = Python golden)

### □ W5.3 Run mixed-input case

- [ ] Load C1 golden weights
- [ ] Trigger and read results
- [ ] C1 target: match within ±4 (FPGA rounding vs Python)

### □ W5.4 Latency measurement

- [ ] Signal Tap: trigger on `valid_in` rising edge
- [ ] Capture all internal valid handshakes:
  - r1_vo, attn_vo, r2_vo, rtr_vo, ffn_done, r3_vo, valid_out
- [ ] Record per-stage latency (in clock cycles)
- [ ] Compare to Icarus simulation values
- [ ] Gate: per-stage latency ≤ 2× simulation

### □ Go/No-Go #3 decision

```
✓ PASS: C0 exact match, C1 within ±4, latency ≤ 2× simulation
        → Phase 1 complete — move to Phase 2 (8-chip card)

△ WARN: C0 matches but C1 out of tolerance, or latency 2-3× simulation
        → Debug and fix; extend Phase 1 by 1-2 weeks

✗ STOP: C0 fails or latency > 3× simulation
        → Hardware architecture flaw; re-evaluate
```

---

## Week 8: Wrap-Up & Decision

### □ W8.1 Phase 1 report

- [ ] Compile all three experiment results into a single report
- [ ] Update `docs/fpga_inference_cluster_proposal.md` with hardware-validated numbers
- [ ] Update BP PPT with Go/No-Go outcomes
- [ ] If all three gates pass: update TCO with corrected latency/BW numbers

### □ W8.2 Team decision meeting

- [ ] Present: fp4 cos=?, HBM BW=?, layer latency=?
- [ ] Against targets: PASS / WARN / STOP
- [ ] Decision options:
  - **All PASS**: Proceed to Phase 2 (order 8× AGM 039 chips, design 4-chip card PCB)
  - **1-2 WARN**: Proceed with adjusted performance targets
  - **Any STOP**: Halt, re-evaluate architecture, consider fp8 fallback
- [ ] Budget implications (Phase 2: ~¥3M for 8 chips + PCB prototype)

### □ W8.3 Phase 2 prep (if GO)

- [ ] Order 8 × AGM 039-F chips (delivery 8-12 weeks)
- [ ] Start PCB schematic for 4-chip card
- [ ] Extend RTL with C2C ring controller (already defined in `rtl/interfaces/`)
- [ ] Run multi-chip C2C simulations (already have `c2c_node.sv` + `tb_c2c_ring.sv`)

---

## Appendix: Quick Reference

### A. Development Kit Details

| Field | Value |
|------|-------|
| Ordering code (ES) | DK-DEV-AGM039FES |
| Ordering code (production) | DK-DEV-AGM039EA |
| FPGA device (ES) | AGMF039R47A1E2VR0 |
| FPGA device (production) | AGMF039R47A1E1VC |
| User guide | Intel doc 782461 (Rev 2025.07.30) |
| BSP download | https://www.intel.com/content/www/us/en/products/details/fpga/development-kits/agilex/agm039.html |
| Mouser order | https://eu.mouser.com/ProductDetail/Altera/DK-DEV-AGM039FES |
| Community | https://www.rocketboards.org/ |

### B. Key files for Phase 1

| File | Purpose |
|------|---------|
| `hw/quartus/fpga_lpu.qpf` | Quartus project |
| `hw/quartus/fpga_lpu.qsf` | Project settings + file list (20 RTL files) |
| `hw/constraints/fpga_lpu.sdc` | Timing constraints (100/156/390 MHz, DSP multicycle) |
| `hw/src/top.sv` | Board-level wrapper (LED, fp4_mac, scale_reader, FSM skeleton) |
| `rtl/dsp/fp4_mac.sv` | fp4 MAC (scale-aware, 3-stage pipeline) |
| `rtl/dsp/fp4_scale_reader.sv` | group_size=16 scale lookup |
| `rtl/layer/full_transformer_layer.sv` | Full layer (use in Week 5) |
| `scripts/simulation/gen_tb_vectors.py` | 15 MAC golden tests |
| `scripts/simulation/gen_layer_golden.py` | 2 layer golden cases |

### C. Clock Reference (from Intel doc 782461)

| Signal | Frequency | Used for |
|--------|-----------|----------|
| CLK_100M_PCIE | 100.00 MHz | R-Tile PCIe refclk |
| SAMPLE_CLK | 312.50 MHz | Sample / general |
| CLK_156_25MHZ | 156.25 MHz | F-Tile / QSFP-DD |
| CLK_245_76MHZ | 245.76 MHz | QSFP-DD transceivers |
| CLK_390_625MHZ (×3) | 390.625 MHz | **DSP PLL source** (→ 450 MHz) |
| DDR5 | 5600 Mbps | Board DIMM (Micron MTC10F1084S1RC56BG1) |
| HBM2e | 450 MHz effective | UIB → 2048-bit @ 920 GB/s |

### D. Useful Quartus commands

```bash
# Full compilation (Phase 1: start with this)
quartus_sh --flow compile fpga_lpu

# Synthesis only (fast iteration during RTL debugging)
quartus_map fpga_lpu

# Timing analysis
quartus_sta fpga_lpu

# Program device via JTAG
quartus_pgm -c 1 -m jtag -o "p;output_files/fpga_lpu.sof"

# Signal Tap capture
quartus_stp --stp_file hw/scripts/debug.stp
```

### E. MCIO PCIe Cable (required — not included with kit)

- **Part**: Amphenol HMC74-0631 (or compatible MCIO ×16 cable assembly)
- **Purpose**: R-Tile PCIe 5.0 ×16 breakout to host system
- **Note**: The DK-DEV-AGM039EA board uses MCIO connector for PCIe, NOT a standard CEM edge connector. Purchase separately.

### F. Hardware Budget Estimate

| Item | Qty | Est. Cost | Source |
|------|-----|-----------|--------|
| DK-DEV-AGM039EA | 1 | $8-12K | Mouser / DigiKey / Intel direct |
| MCIO PCIe cable | 1 | $200-400 | Amphenol / distributor |
| Quartus Pro License | 1 | $4K/yr | Intel FPGA licensing portal |
| **Total** | | **~$13-17K** | |
