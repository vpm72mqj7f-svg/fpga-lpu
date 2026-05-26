# FPGA LPU — Board Bring-Up: Go / No-Go Checklist

## Test Sequence

| Order | Test | LED Code | Module |
|-------|------|----------|--------|
| 1 | HBM2e Bandwidth | `0001` | `hbm_bw_test.sv` |
| 2 | DSP Array Accuracy | `0010` | `dsp_stress_test.sv` |
| 3 | PCIe DMA Throughput | `0011` | `pcie_dma_test.sv` |
| 4 | C2C Ring Link | `0100` | `c2c_node.sv` (loopback) |
| 5 | Full Layer Pipeline | `0101` | `full_transformer_layer.sv` |

LED Codes:
- `1111` = ALL TESTS PASSED
- `1010` = TEST FAILED (connect UART for fail_code and details)
- LED[0] always shows ~0.75 Hz heartbeat while FPGA is alive

---

## Test 1: HBM2e Bandwidth Validation  [GO/NO-GO #1]

**Why this is #1:** The entire architecture assumes 920 GB/s HBM2e bandwidth. If HBM can't deliver, the LPU can't meet latency targets.

### Procedure
1. QSYS: generate HBM2e controller IP (4 stacks, 32 pseudo-channels)
2. Wire `hbm_bw_test.sv` to one pseudo-channel AXI4 interface
3. Run test: writes 256 MB pattern, reads back, measures throughput
4. Repeat for all 32 pseudo-channels in parallel

### Go/No-Go Criteria

| Result | Criteria | Action |
|--------|----------|--------|
| **GO** | Read >= 800 GB/s, Write >= 700 GB/s (>= 80% of rated) | Proceed to Test 2 |
| **WARN** | Read >= 500 GB/s, Write >= 400 GB/s | Investigate AXI burst size, clock freq; may need retuning |
| **NO-GO** | Read < 500 GB/s or Write < 400 GB/s | **STOP. Architecture infeasible.** Check HBM refclk, UIB placement, QSYS config |

### Debug Hints
- Check `hbm_refclk_p/n` is 450 MHz (measure with oscilloscope)
- Verify UIB placement constraints in QSF
- Check AXI4 data width = 256-bit per pseudo-channel
- Monitor HBM temperature via JTAG
- Try reducing burst length (16 → 8) if initial latency is high

---

## Test 2: DSP Array Accuracy & Timing  [GO/NO-GO #2]

**Why this is #2:** The fp4 MAC is the core compute primitive. If it produces wrong results or can't close timing at 450 MHz, the entire datapath is invalid.

### Procedure
1. Instantiate `dsp_stress_test.sv` with target parameters (LANES=4)
2. Run in Mode 0 (sweep): exhaustively test all fp4 × fp8 value pairs
3. Run in Mode 2 (max toggle): verify power delivery under worst-case switching
4. Check timing reports: `report_timing -setup -npaths 100`

### Go/No-Go Criteria

| Result | Criteria | Action |
|--------|----------|--------|
| **GO** | 0 errors in sweep mode, timing closed at 450 MHz | Proceed to Test 3 |
| **WARN** | 0 errors but timing at 350-450 MHz | Reduce target fmax; throughput degrades proportionally |
| **NO-GO** | Any MAC errors OR timing < 350 MHz | **STOP.** Check DSP block configuration, pipeline stages, power supply noise |

### Debug Hints
- Check `clk_dsp` frequency at PLL output (should be 450 MHz, = HBM clock)
- Verify `DSP_BLOCK_BALANCING AUTO` attribute is set
- Check fp4_mac pipeline: should be 4 stages (S0→S1→S2→S3→S4)
- If timing fails: run `report_timing -setup -npaths 100 -detail full_path`
- If errors: run Mode 0 with reduced clock, binary-search for failure point

---

## Test 3: PCIe 5.0 DMA Throughput  [GO/NO-GO #3]

**Why this is #3:** Host-to-FPGA weight streaming requires >= 28 GB/s PCIe bandwidth. If PCIe is slow, weight loading becomes the bottleneck.

**Chip scope:** Master (Chip 0) only. Slaves skip this test.

### Procedure
1. QSYS: generate R-Tile PCIe 5.0 x16 IP
2. Host: load driver, allocate 1 GB DMA buffer
3. Host → FPGA: DMA write 1 GB, measure throughput
4. FPGA → Host: DMA read 1 GB, measure throughput

### Go/No-Go Criteria

| Result | Criteria | Action |
|--------|----------|--------|
| **GO** | H2D >= 28 GB/s, D2H >= 28 GB/s | Proceed to Test 4 |
| **WARN** | H2D >= 16 GB/s, D2H >= 16 GB/s | Acceptable with smaller weight batches |
| **NO-GO** | H2D < 16 GB/s or D2H < 16 GB/s | Check PCIe link training, lane count, BIOS settings |

### Debug Hints
- Verify PCIe link trains at Gen5 x16: `lspci -vv` on host
- Check R-Tile placement in QSF (must be in R-Tile bank)
- Try reducing DMA transfer size to isolate throughput vs latency issues
- Enable PCIe AER (Advanced Error Reporting) on host

---

## Test 4: C2C Ring Link  [GO/NO-GO #4]

**Why this is #4:** The 32-chip pipeline depends on C2C ring. If the ring doesn't work, chips can't communicate.

**Chip scope:** All chips. Master originates, slaves forward.

### Procedure
1. QSYS: generate F-Tile SerDes IP (4 lanes per direction)
2. Master: send loopback packet, measure round-trip latency
3. Master → Slave 1 → Slave 2 → Slave 3 → Master (full ring)
4. Check BER (Bit Error Rate) over 1e12 bits

### Go/No-Go Criteria

| Result | Criteria | Action |
|--------|----------|--------|
| **GO** | BER < 1e-15, latency < 100 ns/hop | Proceed to Test 5 |
| **WARN** | BER < 1e-12, latency < 200 ns/hop | May need FEC or retry for critical control messages |
| **NO-GO** | BER >= 1e-12 or link fails to train | **STOP.** Check SerDes refclk, F-Tile placement, signal integrity |

### Debug Hints
- Check F-Tile refclk (156.25 MHz typical for 25G lanes)
- Verify F-Tile placement constraints
- Use internal loopback first (PMA loopback), then external
- Check eye diagram with on-die scope if available

---

## Test 5: Full Layer Pipeline  [GO/NO-GO #5]

**Why this is #5:** Integration test — does the full RMS→ATTN→RMS→Router→FFN→RMS pipeline work end-to-end with real weights?

### Procedure
1. Load known weights (golden vectors from C reference model)
2. Feed one token with known hidden state
3. Verify output matches golden C model within 1 ULP
4. Measure per-layer latency

### Go/No-Go Criteria

| Result | Criteria | Action |
|--------|----------|--------|
| **GO** | Output matches C model, latency < 500 cycles | Ready for multi-layer deployment |
| **WARN** | Output within 10 ULP, latency < 1000 cycles | Debug numerical precision; may need more fp8 mantissa bits |
| **NO-GO** | Output mismatch > 10 ULP or pipeline hangs | Check all sub-module FSMs; run each module's unit test |

### Debug Hints
- Run `tb_full_transformer_layer.sv` in simulation with same weight vectors
- Capture ILA trace at each sub-module boundary (rms_norm → attn → rms_norm → router → ffn → rms_norm)
- Check KV cache fill_count doesn't overflow (should be fixed in current RTL)
- Verify SiLU LUT values match golden C model

---

## Power-Up Sequence

```
1. Apply 12V board power
2. Check power rails (via PMBus/I2C):
   - VCC 0.8V (core) — within ±3%
   - VCCP 1.0V (HBM) — within ±3%
   - VCCIO 1.2V — within ±5%
3. Assert cpu_reset_n (low → high after clocks stable)
4. LED[0] should start blinking (~0.75 Hz)
5. Press start_button or send 'S' over UART
6. Monitor LED[3:1] for test progress
7. If LED = 1111: ALL PASS. If LED = 1010: read fail_code over UART
```

## UART Console (115200 8N1)

Send:
- `S` — Start test sequence
- `A` — Abort current test
- `R` — Report last test result
- `H` — Help / list commands

Receive:
- Test start/stop messages
- Measured metrics (bandwidth, errors, latency)
- fail_code on error (see table below)

### fail_code Reference

| Code | Test | Common Cause |
|------|------|-------------|
| 1 | HBM BW | HBM refclk missing, UIB placement wrong, QSYS config error |
| 2 | DSP Acc | MAC pipeline depth wrong, timing violation, power noise |
| 3 | PCIe DMA | Link training failed, wrong lane count, BIOS MMIO config |
| 4 | C2C Link | SerDes PLL unlock, signal integrity, F-Tile placement |
| 5 | Layer Pipe | FSM hang, KV cache overflow (fixed), numerical divergence |

## After All Tests Pass

1. Load per-layer weights via PCIe DMA
2. Run multi-layer pipeline (12 layers per chip)
3. Enable C2C forwarding for multi-chip pipeline
4. Measure end-to-end token latency
5. Compare against simulation baseline (182K cycles @ 100 MHz for 384 layers)

### Production Target Metrics

| Metric | Target | Simulation Baseline |
|--------|--------|-------------------|
| Per-layer latency (steady state) | < 125 ns | 485 cycles @ 100 MHz = 4.85 us |
| Per-chip latency (12 layers) | < 1.5 us | 5,820 cycles @ 100 MHz = 58.2 us |
| Cluster latency (384 layers) | < 50 us | 182,208 cycles @ 100 MHz = 1.82 ms |
| HBM bandwidth | >= 800 GB/s | N/A (not simulated) |
| PCIe DMA throughput | >= 28 GB/s | N/A (not simulated) |
| fp4 MAC accuracy | < 1 ULP error | 0 errors (15 golden tests) |

> Note: Simulation times are at 100 MHz system clock. Target production times assume
> 450 MHz DSP/HBM clock (4.5× faster than simulation) with pipelined control path.
> DSP @ 450 MHz = 2.22 ns period. Gap analysis needed after timing closure.

---

## Signal Tap / ILA Probe Points

Pre-wire these signals for ILA capture during bring-up:

| Probe | Width | Module | Purpose |
|-------|-------|--------|---------|
| `seq_state` | 4 | `top_bringup` | Bring-up FSM state |
| `test_result` | 2 | `top_bringup` | GO/NO-GO/WARN/RUNNING |
| `m_axi_awvalid/awready` | 2 | `hbm_bw_test` | HBM write handshake |
| `m_axi_rvalid/rready` | 2 | `hbm_bw_test` | HBM read handshake |
| `array_result_valid` | 1 | `dsp_stress_test` | DSP test output valid |
| `errors_detected` | 32 | `dsp_stress_test` | DSP error counter |
| `st` (FSM state) | 4 | `full_transformer_layer` | Layer pipeline state |
| `valid_in/valid_out` | 2 | `full_transformer_layer` | Layer handshake |
| `entry_count` | 7 | `mla_kv_cache` | KV cache fill level |
