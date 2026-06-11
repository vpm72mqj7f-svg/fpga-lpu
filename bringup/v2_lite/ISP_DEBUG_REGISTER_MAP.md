# V2-Lite Full Debug Register Map

> **设计原则**: 寄存器表先行（Register Map First）。每个子系统独立 ISP 实例，probe 读状态，source 写控制。CLI 直接读写，不依赖 GUI。

---

## ISP 实例总览

| 实例 ID | 子系统 | Probe 数 | Source 数 | 用途 |
|---------|--------|----------|-----------|------|
| `PCIE` | PCIe XCVR | 3 × 32-bit | 0 | 链路状态、lane 状态、错误计数 |
| `HBM2` | HBM2 Memory | 3 × 32-bit | 0 | TG 状态、温度、通道状态 |
| `FFN` | FFN Engine | 4 × 32-bit | 0 | 状态、性能计数、AXI 统计 |
| `SYS` | System | 1 × 32-bit | 1 × 32-bit | 系统状态 + 控制寄存器 |

---

## 1. PCIE — PCIe XCVR 子系统

### Probe 0: PCIE_LINK_STATUS (offset 0x00, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[4:0]` | `LTSSM_STATE` | PCIe LTSSM state: 0=Detect, 1=Polling, 2=Config, 3=L0, 4=Recovery, ... |
| `[9:5]` | `LINK_SPEED` | Negotiated link speed: 1=Gen1, 2=Gen2, 3=Gen3 |
| `[15:10]` | `LINK_WIDTH` | Active lane count: 1=x1, 2=x2, 4=x4, 8=x8, 16=x16 |
| `[16]` | `ATX_PLL_LOCK` | ATX PLL locked |
| `[17]` | `CONFIG_DONE` | PCIe HIP configuration done |
| `[18]` | `PERSTN` | Current PERST# pin state (0=in reset) |
| `[19]` | `LINK_UP` | PCIe link is up (L0 state) |
| `[31:20]` | *(reserved)* | |

### Probe 1: PCIE_LANE_STATUS (offset 0x01, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[15:0]` | `PLL_LOCK[15:0]` | Per-lane PLL lock. Bit N = lane N. Bank mapping: [0]=1c_0, [5]=1c_5, [6]=1d_0, ..., [15]=1e_3 |
| `[31:16]` | `SIGNAL_DETECT[15:0]` | Per-lane RX signal detect. `1` = electrical signal present |

### Probe 2: PCIE_ERROR_COUNTERS (offset 0x02, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[15:0]` | `DL_FE_ERR` | Data Link layer framing error count (saturating) |
| `[31:16]` | `TL_ERR` | Transaction layer error count (saturating) |

---

## 2. HBM2 — HBM2 Memory 子系统

### Probe 0: HBM2_TG_STATUS (offset 0x00, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[15:0]` | `TG_PASS[15:0]` | Per-channel traffic generator pass. Bit N = channel N (tg0_0..tg7_1) |
| `[31:16]` | `TG_FAIL[15:0]` | Per-channel traffic generator fail |

### Probe 1: HBM2_TG_TIMEOUT (offset 0x01, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[15:0]` | `TG_TIMEOUT[15:0]` | Per-channel traffic generator timeout |
| `[31:16]` | *(reserved)* | |

### Probe 2: HBM2_STATUS (offset 0x02, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[2:0]` | `TEMP[2:0]` | *(TODO: HBM2 IO buffer, can't fan out — read via internal reg)* |
| `[3]` | `CATTRIP` | *(TODO: same as above)* |
| `[4]` | `PLL_LOCK` | HBM2 core PLL locked |
| `[5]` | `CH_ACTIVE` | Any channel active |
| `[31:6]` | *(reserved)* | |

---

## 3. FFN — Feed-Forward Network Engine

### Probe 0: FFN_STATUS (offset 0x00, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[3:0]` | `STATE` | FSM state: 0=IDLE, 1=START, 2=BUSY, 5=PASS |
| `[4]` | `BUSY` | Engine is processing |
| `[5]` | `DONE` | Computation completed |
| `[6]` | `PASS` | Self-test passed |
| `[7]` | `ERROR` | Error detected |
| `[15:8]` | `ERROR_CODE` | Error code (0=no error) |
| `[31:16]` | *(reserved)* | |

### Probe 1: FFN_PERF (offset 0x01, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[15:0]` | `TOKEN_COUNT` | Tokens processed since last reset |
| `[31:16]` | `CYCLE_COUNT[15:0]` | Cycle counter LSB — for throughput measurement |

### Probe 2: FFN_AXI_STATS (offset 0x02, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[15:0]` | `AR_TRANS` | AXI read address transactions issued |
| `[31:16]` | `R_BEATS` | AXI read data beats received |

### Probe 3: FFN_DATA (offset 0x03, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[15:0]` | `TDATA_LO[15:0]` | FFN output data [15:0] (latest) |
| `[31:16]` | `TDATA_HI[15:0]` | FFN output data [31:16] (latest) |

---

## 4. SYS — 系统控制与状态

### Probe 0: SYS_STATUS (offset 0x00, RO)

| Bits | Field | Description |
|------|-------|-------------|
| `[3:0]` | `LED[3:0]` | Board LED: [0]=HBM2, [1]=~PCIe_PLL, [2]=FFN, [3]=heartbeat |
| `[7:4]` | `RESET_STATUS` | Reset state flags |
| `[15:8]` | `CLK_STATUS` | Clock status flags |
| `[31:16]` | `VERSION` | Firmware version (0x0001 = v1.0) |

### Source 0: SYS_CTRL (offset 0x00, WO)

| Bits | Field | Description |
|------|-------|-------------|
| `[0]` | `FFN_START` | Write `1` to start FFN self-test |
| `[1]` | `FFN_RESET` | Write `1` to reset FFN engine |
| `[2]` | `COUNTER_RESET` | Write `1` to reset all performance counters |
| `[31:3]` | *(reserved)* | |

---

## CLI 使用

### 读所有状态
```powershell
quartus_issp --probe --instance=PCIE   # PCIe 链路状态
quartus_issp --probe --instance=HBM2   # HBM2 状态
quartus_issp --probe --instance=FFN    # FFN 状态
quartus_issp --probe --instance=SYS    # 系统状态
```

### 写控制
```powershell
quartus_issp --source --instance=SYS --source_index=0 --value=1  # 启动 FFN
```

### 健康检查一键脚本
```powershell
$pcie = quartus_issp --probe --instance=PCIE --probe_index=0
Write-Output "PCIe PLL: $(($pcie -shr 16) -band 1)"
Write-Output "PCIe Lanes Locked: $(($pcie -shr 32) -band 0xFFFF)"
$hbm2 = quartus_issp --probe --instance=HBM2 --probe_index=0
Write-Output "HBM2 TG Pass: $(($hbm2 -band 0xFFFF))"
$ffn = quartus_issp --probe --instance=FFN --probe_index=0
Write-Output "FFN State: $(($ffn -band 0xF))"
Write-Output "FFN Pass: $(($ffn -shr 6) -band 1)"
```

---

## 硬件实现

| 文件 | 内容 |
|------|------|
| `v2_lite_isp_debug.v` | 4 个 ISP 实例（PCIE/HBM2/FFN/SYS），信号打包 + 性能计数器 |

---

## 版本历史

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-06-11 | Initial register map: PCIE (3 probes), HBM2 (3 probes), FFN (4 probes), SYS (1P+1S) |
