# 32 Chip FPGA 集群芯片级分解方案

## 一、拓扑总览

```text
┌─────────────────────────────────────────────────────────────────┐
│                     单台 4U Server                               │
│                                                                  │
│  ┌─ Card 0 ──────────┐  ┌─ Card 1 ──────────┐   ...  Card 7    │
│  │ Chip0 (PCIe Master)│  │ Chip8              │                 │
│  │ L00-01 + Embedding│  │ L08-09             │                 │
│  │ Chip1 Chip2 Chip3  │  │ Chip9 10 11        │                 │
│  │ L02-07             │  │ L10-14             │                 │
│  │ C2C Dual Ring A/B  │  │ C2C Dual Ring A/B  │                 │
│  └────────────────────┘  └────────────────────┘                 │
│       ↕ PCIe 5.0 x16         ↕ PCIe 5.0 x16                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              PCIe 5.0 Backplane (P2P DMA)                │   │
│  │         CPU 0 Root Complex  |  CPU 1 Root Complex        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 二、层到芯片映射

384 层分配到 32 芯片：32 chips × 12 layers = 384

每卡 4 芯片（1 Master + 3 Slave），通过 C2C 双环互联。
8 个 Master 通过 PCIe 5.0 背板互联到双路 Host。

| Card | Chip | Global ID | Role | Layers | Expert Range |
|------|------|-----------|------|--------|-------------|
| 0 | 0 | 00 | **Master** | L000-011 + Emb | E000-047 |
| 0 | 1 | 01 | Slave | L012-023 | E048-095 |
| 0 | 2 | 02 | Slave | L024-035 | E096-143 |
| 0 | 3 | 03 | Slave | L036-047 | E144-191 |
| 1 | 0 | 04 | **Master** | L048-059 | E192-239 |
| 1 | 1 | 05 | Slave | L060-071 | E240-287 |
| 1 | 2 | 06 | Slave | L072-083 | E288-335 |
| 1 | 3 | 07 | Slave | L084-095 | E336-383 |
| 2 | 0 | 08 | **Master** | L096-107 | E000-047 |
| 2 | 1 | 09 | Slave | L108-119 | E048-095 |
| 2 | 2 | 10 | Slave | L120-131 | E096-143 |
| 2 | 3 | 11 | Slave | L132-143 | E144-191 |
| 3 | 0 | 12 | **Master** | L144-155 | E192-239 |
| 3 | 1 | 13 | Slave | L156-167 | E240-287 |
| 3 | 2 | 14 | Slave | L168-179 | E288-335 |
| 3 | 3 | 15 | Slave | L180-191 | E336-383 |
| 4 | 0 | 16 | **Master** | L192-203 | E000-047 |
| 4 | 1 | 17 | Slave | L204-215 | E048-095 |
| 4 | 2 | 18 | Slave | L216-227 | E096-143 |
| 4 | 3 | 19 | Slave | L228-239 | E144-191 |
| 5 | 0 | 20 | **Master** | L240-251 | E192-239 |
| 5 | 1 | 21 | Slave | L252-263 | E240-287 |
| 5 | 2 | 22 | Slave | L264-275 | E288-335 |
| 5 | 3 | 23 | Slave | L276-287 | E336-383 |
| 6 | 0 | 24 | **Master** | L288-299 | E000-047 |
| 6 | 1 | 25 | Slave | L300-311 | E048-095 |
| 6 | 2 | 26 | Slave | L312-323 | E096-143 |
| 6 | 3 | 27 | Slave | L324-335 | E144-191 |
| 7 | 0 | 28 | **Master** | L336-347 | E192-239 |
| 7 | 1 | 29 | Slave | L348-359 | E240-287 |
| 7 | 2 | 30 | Slave | L360-371 | E288-335 |
| 7 | 3 | 31 | Slave | L372-383 + lm_head + MTP | E336-383 |

关键设计：
- 8 个 Master（每卡 Chip 0）：PCIe 5.0 主机接口 + C2C 环起点
- 24 个 Slave（每卡 Chip 1-3）：纯计算，通过 C2C 环通信
- 芯片 31 (最后一个 Slave): L372-383 + lm_head projection + Multi-Token Prediction
- 每 chip 均匀 12 层，代码完全统一（chip_top.sv 参数化 IS_PCIE_MASTER）

## 三、卡内 C2C 双环拓扑

每张 PCIe 加速卡有 4 颗 AGM 039：

```text
        ┌─────────────────────────────────────┐
        │          FPGA Accelerator Card       │
        │                                      │
        │   Chip 0 ◄════ Ring A (CW) ════► Chip 1
        │    ▲                                  │
        │    ║ Ring B (CCW)                     │
        │    ║                                  │
        │   Chip 3 ◄════ Ring A  ═══════► Chip 2
        │                                      │
        │   PCIe 5.0 x16 CEM (Chip 0 only)     │
        └──────────────────────────────────────┘
```

### C2C SerDes 参数

| 参数 | 值 |
|------|-----|
| Link rate | 56 Gbps PAM4 per lane |
| Lanes per link | 8 |
| Effective BW | ~448 Gbps = 56 GB/s per direction |
| Hop latency | ~10 ns (PCB trace + SerDes) |
| 编码 | 64b/66b |
| 每 frame 开销 | 24B header + CRC |

### 双环冗余

- **Ring A** (顺时针): 0→1, 1→2, 2→3, 3→0
- **Ring B** (逆时针): 0→3, 3→2, 2→1, 1→0

任意两芯片间选择短路径。单链路故障不影响通信（走另一方向）。

## 四、Chip 0 vs Chip 1/2/3 差异

| 功能 | Chip 0 | Chip 1-3 |
|------|--------|----------|
| PCIe 5.0 EP Hard IP | ✓ R-Tile | ✗ |
| P2P DMA Engine | ✓ 自研 RTL | ✗ |
| BAR0 寄存器 | ✓ 64 MB | ✗ (通过 C2C proxy) |
| BAR2 HBM aperture | ✓ 32 GB per chip mapped | ✗ (通过 C2C proxy) |
| C2C proxy bridge | ✓ 转发 chip 1-3 的 PCIe 流量 | ✗ |
| C2C SerDes | ✓ F-Tile ×2 | ✓ F-Tile ×2 |
| 完整 RTL 模块 | ✓ | ✓ (无 PCIe 逻辑) |

### PCIe Proxy 原理

```
Host ↔ PCIe ↔ Chip 0 (BAR4 64MB 映射 4 芯片寄存器)
                  │
                  ├── Chip 0 registers (direct)
                  ├── Chip 1 registers (C2C forward via Ring A)
                  ├── Chip 2 registers (C2C forward via Ring A)
                  └── Chip 3 registers (C2C forward via Ring A)
```

Chip 0 的 BAR4 将 4 片芯片的寄存器空间统一映射到 64 MB PCIe BAR，Host 通过 Chip 0 的 C2C proxy bridge 透明访问 Chip 1-3。

## 五、Pipeline Forward 协议

Token 沿层序流水线在各芯片间单向传递：

```text
Host (PCIe DMA)
  → Chip 00 (L00-01) → C2C Ring → Chip 01 (L02-03) → ...
  → Chip 03 (L06-07) → PCIe P2P → Chip 04 (L08-09) → ...
  → Chip 31 (L59-60) → PCIe DMA → Host
```

### 帧格式

```text
┌──────┬─────────┬─────────┬──────────┬──────────────────────────┐
│valid │ src_chip│ dst_chip│ token_id │ hidden_state[8] (Q12×8)  │
│ 1b   │ 8b      │ 8b      │ 16b      │ 256b                     │
└──────┴─────────┴─────────┴──────────┴──────────────────────────┘
```

- 单 token hidden state = 8 elements × 32-bit Q12 = 256 bits
- 每 token 在 61 层流水线上经过 32 跳
- 同一 token 的连续层在不同 chip 上并行处理（pipeline parallelism）

### 同卡 vs 跨卡

| 场景 | 介质 | 延迟 |
|------|------|------|
| 同卡 chip → chip (相邻) | C2C SerDes (10ns) | ~10 ns |
| 同卡 chip → chip (对端, 2-hop) | C2C (2× hop) | ~20 ns + 1 cycle |
| 跨卡 chip → chip | PCIe 5.0 P2P DMA | ~260 ns |

## 六、MoE Dispatch / Reduce

### 流程

```text
1. Router 在每个 MoE 层后选出 Top-6 专家
2. 对于不在本地的专家：
   a. Dispatch: 发送 activation 到专家所在 chip (by C2C or PCIe P2P)
   b. Remote chip: 计算 Routed Expert FFN
   c. Reduce: 将专家输出送回请求 chip
3. 本地专家直接在本地计算（无 C2C/PCIe 开销）
```

### 帧格式

**Dispatch:**
```text
┌──────┬─────────┬─────────┬──────────┬──────────┬────────────────┐
│valid │ src_chip│ dst_chip│ expert_id│ token_id │ activation[8]  │
│ 1b   │ 8b      │ 8b      │ 12b      │ 16b      │ 256b (FP8)     │
└──────┴─────────┴─────────┴──────────┴──────────┴────────────────┘
```

**Reduce:**
```text
┌──────┬─────────┬─────────┬──────────┬──────────┬────────────────┐
│valid │ src_chip│ dst_chip│ expert_id│ token_id │ result[8]      │
│ 1b   │ 8b      │ 8b      │ 12b      │ 16b      │ 256b (Q12)     │
└──────┴─────────┴─────────┴──────────┴──────────┴────────────────┘
```

## 七、每 Chip 资源分配

| 资源 | 用量 (2-layer chip) | 用量 (1-layer chip) |
|------|--------------------|--------------------|
| HBM 权重 (fp4) | ~0.7 GB | ~0.4 GB |
| HBM KV 区 | ~22 GB | ~22.3 GB |
| HBM 总占用 | ~22.7 GB | ~22.7 GB |
| HBM 余量 | ~9.3 GB | ~9.3 GB |
| SRAM 确定性权重 | ~26 MB (double-buffer) | ~13 MB |
| SRAM 专家 buffer | ~2 MB | ~2 MB |
| DSP 利用率 (B=1 decode) | ~50% (0-hit layers) | ~50% |
| DSP 利用率 (B=8 decode) | ~25% | ~25% |
| C2C 带宽占用 | ~5.4 GB/s | ~5.4 GB/s |
| PCIe 带宽占用 (chip 0) | ~0.3 GB/s (control) | — |

## 八、RTL 接口文件清单

| 文件 | 内容 |
|------|------|
| `rtl/interfaces/avalon_stream.svh` | valid/ready 流接口、pipeline_forward_beat、moe_dispatch/reduce_beat |
| `rtl/interfaces/c2c_packet.svh` | C2C header、消息类型、per-chip link 封装 |
| `rtl/interfaces/pcie_dma.svh` | PCIe DMA descriptor、BAR0 寄存器、C2C proxy 接口 |
| `rtl/chip/chip_top.sv` | 单 chip 顶层：layer 实例化 + C2C/PCIe 端口 + pipeline forward 逻辑 |

## 九、Weight Layout Compiler 输出

Weight Layout Compiler（`scripts/vllm_serve/weight_layout.py`）为每个 chip 生成：

```text
per-chip config:
  - layer_start, layer_end
  - expert list (with replica count for hot replication)
  - HBM base address for weight region
  - HBM base address for KV cache region
  - C2C routing table (next_hop per destination chip)
```

此配置在 chip 初始化时通过 PCIe (chip 0) 或 C2C CTRL 消息 (chip 1-3) 加载到寄存器。
