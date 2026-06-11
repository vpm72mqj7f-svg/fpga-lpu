# V2-Lite FPGA Debug & Register Interface — 工程检查清单

> **文档类型**: 硬件调试接口设计规范 (Checklist + Spec)  
> **参考**: Arrive Technologies `.atreg` 规范, Intel `altsource_probe` 用户指南  
> **日期**: 2026-06-11  
> **关联文档**: `AF_V2LITE_Design_Spec_Rev1.1.md`

---

## 1. 模块层次与接口总览

```
v2_lite_full (top) ─── v2_lite_full_top.v
│
├── u_hbm: ed_synth (Qsys) ─── HBM2 Controller + TG
├── u_pcie: pcie_xcvr_system (Qsys) ─── PCIe XCVR + ATX PLL
├── u_ffn: v2_lite_ffn_engine ─── FFN 主引擎
│   ├── u_hbm2_reader: hbm2_weight_reader ─── AXI4 权重读取器
│   ├── u_sa_gate_up: systolic_array ─── Gate/Up 脉动阵列
│   ├── u_sa_down: systolic_array ─── Down 脉动阵列
│   └── u_silu: silu_activation ─── SiLU 激活函数
├── u_isp: v2_lite_isp_debug ─── JTAG ISP 调试聚合器
└── u_gpio: q_sys_gpio ─── HBM2 TG 状态汇总
```

## 2. 接口类型定义 (Arrive .atreg 规范)

| 类型 | 缩写 | 含义 | 硬件实现 |
|------|------|------|---------|
| Read-Only | `R_O` | 只读状态/计数器 | `output wire [31:0]` 直接寄存器 |
| Read/Write | `R/W` | 可读写配置 | `input wire [31:0]` + `output wire [31:0]` |
| Write-1-to-Clear | `R/W/C` | 写1清除 sticky bit | `input wire [31:0]` + 内部分频逻辑 |
| Write-Only | `W_O` | 只写触发 | `input wire [31:0]` |

### 2.1 版本寄存器格式 (Arrive .atreg)

```
[31:24] day    — 编译日 (1–31)
[23:16] month  — 编译月 (1–12)
[15:08] year   — 编译年 − 2000
[07:00] number — 当日编译序号 (1-indexed)

示例: 0x0B061A01 = 2026-06-11, Build #1
```

---

## 3. 各模块 Debug/寄存器 Checklist

### 3.1 `v2_lite_ffn_engine` — FFN 主引擎 🔴 需补齐

**当前接口**: `clk, rst_n, pcie_rx_*, pcie_tx_*, m_axi_*, expert_id, busy, done`  
**debug 输出**: 无  
**版本寄存器**: 无  
**性能计数器**: 无

#### 3.1.1 需新增端口

```systemverilog
// ---- Version Register ----
parameter logic [31:0] FFN_VERSION = 32'h0B061A01  // 只读参数

// ---- Debug Status (32-bit packed for ISP) ----
output logic [3:0]  dbg_fsm_state,       // FSM state (0..9, 10 states)
output logic        dbg_expert_cnt_v,    // expert_cnt valid
output logic [2:0]  dbg_expert_cnt,      // 当前 expert 索引 (0..TOP_K)
output logic        dbg_gate_done,       // Gate projection 完成
output logic        dbg_up_done,         // Up projection 完成
output logic        dbg_down_done,       // Down projection 完成
output logic        dbg_silu_active,     // SiLU 处理中
output logic        dbg_merge_active,    // Merge 处理中
output logic        dbg_hbm2_busy,       // HBM2 reader busy (透传)
output logic        dbg_sa_active,       // 任一 systolic_array active

// ---- Performance Counters (32-bit, sticky, read via ISP) ----
output logic [31:0] perf_token_cnt,      // 完成的 token 数
output logic [31:0] perf_cycle_cnt,      // 总周期数
output logic [31:0] perf_expert_cnt,     // 累计处理的 expert 数
output logic [31:0] perf_axi_rbeat,      // AXI 读数据 beat 数

// ---- Error Flags (sticky, write-1-clear via future BAR) ----
output logic        err_merge_overflow,  // merge_idx 溢出
output logic        err_silu_overflow,   // silu_idx 溢出
output logic        err_axi_resp_err     // AXI RRESP error
```

#### 3.1.2 内部逻辑修改

```systemverilog
// 版本寄存器 — 直接 parameter
localparam logic [31:0] VERSION = FFN_VERSION;

// 性能计数器
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        perf_cycle_cnt  <= 32'd0;
        perf_token_cnt  <= 32'd0;
        perf_expert_cnt <= 32'd0;
        perf_axi_rbeat  <= 32'd0;
    end else begin
        perf_cycle_cnt  <= perf_cycle_cnt + 32'd1;         // 每个周期+
        if (done)
            perf_token_cnt  <= perf_token_cnt + 32'd1;     // token 完成
        if (state == S_NEXT_EXPERT)   // 每个 expert 循环
            perf_expert_cnt <= perf_expert_cnt + 32'd1;
        if (m_axi_rvalid && m_axi_rready)                 // AXI R beat
            perf_axi_rbeat  <= perf_axi_rbeat + 32'd1;
    end
end

// Debug 状态组合输出
assign dbg_fsm_state     = state;
assign dbg_expert_cnt    = expert_cnt;
assign dbg_expert_cnt_v  = (state != S_IDLE) && (state != S_OUTPUT);
assign dbg_gate_done     = gate_done;
assign dbg_up_done       = up_done;
assign dbg_down_done     = down_done;
assign dbg_silu_active   = (state == S_SILU);
assign dbg_merge_active  = (state == S_MERGE_GATE_UP);
assign dbg_hbm2_busy     = hbm2_busy;
assign dbg_sa_active     = sa_gate_up_busy || sa_down_busy;

// 错误标志 — 在已有的 assertion block 中设置
assign err_merge_overflow = ...; // sticky
assign err_silu_overflow  = ...;
assign err_axi_resp_err   = (m_axi_rvalid && (m_axi_rresp != 2'b00));
```

#### 3.1.3 寄存器地址映射 (未来 PCIe BAR0)

| 偏移 | 名称 | 宽度 | 类型 | 描述 |
|------|------|------|------|------|
| 0x00 | FFN_VERSION | 32 | R_O | `{day, month, year, number}` |
| 0x04 | FFN_STATUS | 32 | R_O | `{fsm[3:0], busy, done, expert[2:0], gate, up, down, ...}` |
| 0x08 | FFN_PERF_TOKEN | 32 | R_O | 累计 token |
| 0x0C | FFN_PERF_CYCLE | 32 | R_O | 累计周期 |
| 0x10 | FFN_PERF_EXPERT | 32 | R_O | 累计 expert |
| 0x14 | FFN_PERF_AXI_RBEAT | 32 | R_O | AXI 读 beat 数 |
| 0x18 | FFN_ERROR | 32 | R/W/C | 错误 sticky bits |

### 3.1.4 Checklist

- [ ] 添加 `FFN_VERSION` parameter
- [ ] 添加 `dbg_fsm_state[3:0]` 输出
- [ ] 添加 `dbg_expert_cnt[2:0]` 输出
- [ ] 添加 `dbg_gate_done, dbg_up_done, dbg_down_done` 输出
- [ ] 添加 `dbg_silu_active, dbg_merge_active` 输出
- [ ] 添加 `dbg_hbm2_busy, dbg_sa_active` 输出
- [ ] 添加 4 个性能计数器 `perf_token_cnt, perf_cycle_cnt, perf_expert_cnt, perf_axi_rbeat`
- [ ] 添加 3 个错误标志 `err_merge_overflow, err_silu_overflow, err_axi_resp_err`
- [ ] 组合逻辑赋值所有 debug 输出
- [ ] 综合验证: debug 端口不出现在关键路径上

---

### 3.2 `systolic_array` — 脉动阵列 🔴 需补齐

**当前接口**: `clk, rst_n, start, busy, done, activ_*, weight_*, result_*, dbg_current_row, dbg_cycle_cnt`  
**debug 输出**: 仅有 `dbg_current_row`, `dbg_cycle_cnt`（且在 FFN engine 中悬空）  
**版本寄存器**: 无  
**性能计数器**: 无

#### 3.2.1 需新增端口

```systemverilog
// ---- Version ----
parameter logic [31:0] SA_VERSION = 32'h0B061A01

// ---- Debug Status ----
output logic [3:0]  dbg_fsm_state,      // FSM: IDLE(0)..DONE(8)
output logic        dbg_preload_active,  // 权重预取活跃
output logic        dbg_stream_active,   // STREAM 状态
output logic [5:0]  dbg_cycle_in_row,    // 当前行的周期计数 (6bit, max 63)
// 已有的 dbg_current_row, dbg_cycle_cnt 保留

// ---- Performance Counters ----
output logic [31:0] perf_rows_done,     // 完成的行数
output logic [31:0] perf_projections,   // 完成的 projection 数
output logic [31:0] perf_total_cycles   // 总活跃周期数
```

#### 3.2.2 内部逻辑

```systemverilog
// 性能计数器
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        perf_rows_done    <= 32'd0;
        perf_projections  <= 32'd0;
        perf_total_cycles <= 32'd0;
    end else begin
        if (busy)
            perf_total_cycles <= perf_total_cycles + 32'd1;
        if (state == S_STORE)
            perf_rows_done <= perf_rows_done + 32'd1;
        if (state == S_DONE)  // 每个 projection 完成
            perf_projections <= perf_projections + 32'd1;
    end
end

assign dbg_fsm_state     = state;
assign dbg_preload_active = (state == S_WT_PRELOAD) || (state == S_WT_PRELOAD_WAIT);
assign dbg_stream_active  = (state == S_STREAM);
assign dbg_cycle_in_row   = cycle_count[5:0];
```

#### 3.2.3 Checklist

- [ ] 添加 `SA_VERSION` parameter
- [ ] 添加 `dbg_fsm_state[3:0]` 输出
- [ ] 添加 `dbg_preload_active, dbg_stream_active` 输出
- [ ] 添加 `dbg_cycle_in_row[5:0]` 输出
- [ ] 添加 3 个性能计数器 `perf_rows_done, perf_projections, perf_total_cycles`
- [ ] FFN engine 中连接 `.dbg_current_row()`, `.dbg_cycle_cnt()`（修复悬空）
- [ ] FFN engine 中连接 `dbg_cycle_in_row` → ISP

---

### 3.3 `hbm2_weight_reader` — HBM2 权重读取器 🔴 需补齐

**当前接口**: `clk, rst_n, m_axi_*, weight_*, start, base_addr, total_words, busy, done`  
**debug 输出**: 无  
**版本寄存器**: 无

#### 3.3.1 需新增端口

```systemverilog
// ---- Version ----
parameter logic [31:0] HBM2R_VERSION = 32'h0B061A01

// ---- Debug Status ----
output logic [2:0]  dbg_fsm_state,      // FSM: IDLE(0)..DRAIN(3)
output logic        dbg_buf_sel,         // 当前填充/stream buffer 选择
output logic [6:0]  dbg_rd_addr,         // 当前 bank 读地址 (水位线)
output logic [6:0]  dbg_wr_addr,         // 当前 bank 写地址 (水位线)
output logic        dbg_streaming,       // 正在 stream
output logic        dbg_filling,         // 正在填充

// ---- Performance Counters ----
output logic [31:0] perf_bytes_read,     // 累计读字节数
output logic [31:0] perf_bursts_done,    // 完成的 AXI burst 数
output logic [31:0] perf_beats_read      // 完成的 AXI R beat 数
```

#### 3.3.2 内部逻辑

```systemverilog
// 性能计数器
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        perf_bytes_read  <= 32'd0;
        perf_bursts_done <= 32'd0;
        perf_beats_read  <= 32'd0;
    end else begin
        if (m_axi_rvalid && m_axi_rready) begin
            perf_bytes_read <= perf_bytes_read + 32'd32;  // 256-bit = 32 bytes
            perf_beats_read <= perf_beats_read + 32'd1;
        end
        if (m_axi_rvalid && m_axi_rready && m_axi_rlast)
            perf_bursts_done <= perf_bursts_done + 32'd1;
    end
end

assign dbg_fsm_state = state;
assign dbg_buf_sel   = buf_sel;
assign dbg_rd_addr   = bank_rd_addr;
assign dbg_wr_addr   = bank_wr_addr;
assign dbg_streaming = weight_valid;
assign dbg_filling   = (state == S_RDATA);
```

#### 3.3.3 Checklist

- [ ] 添加 `HBM2R_VERSION` parameter
- [ ] 添加 `dbg_fsm_state[2:0]` 输出
- [ ] 添加 `dbg_buf_sel, dbg_rd_addr, dbg_wr_addr` 输出
- [ ] 添加 `dbg_streaming, dbg_filling` 输出
- [ ] 添加 3 个性能计数器
- [ ] FFN engine 中透传 HBM2 reader 的 debug 端口到顶层

---

### 3.4 `silu_activation` — SiLU 激活函数 🔴 需补齐

**当前接口**: `clk, rst_n, valid_in, data_in[NUM_ELEMS], data_out[NUM_ELEMS], valid_out`  
**debug 输出**: 无  
**版本寄存器**: 无

#### 3.4.1 需新增端口

```systemverilog
// ---- Version ----
parameter logic [31:0] SILU_VERSION = 32'h0B061A01

// ---- Debug Status ----
output logic        dbg_stage1_valid,    // Stage1 LUT lookup active
output logic        dbg_stage2_valid,    // Stage2 fp16 multiply active
output logic [15:0] dbg_sample_in,       // 采样: data_in[0] 用于波形观察
output logic [15:0] dbg_sample_sigmoid,  // 采样: sigmoid 值 (Stage1 out)
output logic [15:0] dbg_sample_out       // 采样: data_out[0] (Stage2 out)
```

#### 3.4.2 Checklist

- [ ] 添加 `SILU_VERSION` parameter
- [ ] 添加 `dbg_stage1_valid, dbg_stage2_valid` 流水线状态
- [ ] 添加 3 个采样端口 (lane 0 的输入/中间/输出)
- [ ] FFN engine 中连接 SiLU debug 端口

---

### 3.5 `fp8_mac` — FP8 MAC 单元 🔴 需补齐

**当前接口**: `clk, rst_n, valid_in, a[NUM_LANES], b[NUM_LANES], sum[NUM_LANES], valid_out`  
**debug 输出**: 无  
**版本寄存器**: 无

#### 3.5.1 需新增端口

```systemverilog
// ---- Version ----
parameter logic [31:0] FP8MAC_VERSION = 32'h0B061A01

// ---- Debug Status ----
output logic        dbg_stage1_valid,    // Stage1 decode+multiply active
output logic        dbg_stage2_valid,    // Stage2 normalize+accumulate active
output logic [5:0]  dbg_overflow_lane,   // 最后溢出的 lane index
output logic        dbg_overflow_sticky  // 任一 lane 溢出的 sticky flag
```

> **注**: `fp8_mac` 由 `systolic_array` 的 generate loop 例化 64 次。debug 端口需被 `systolic_array` 打包聚合后向上传递。

#### 3.5.2 Checklist

- [ ] 添加 `FP8MAC_VERSION` parameter
- [ ] 添加 `dbg_stage1_valid, dbg_stage2_valid` 流水线状态
- [ ] 添加 `dbg_overflow_lane[5:0]`, `dbg_overflow_sticky` 溢出检测
- [ ] `systolic_array` 中聚合 64 个 `fp8_mac` 的 debug 输出

---

### 3.6 `v2_lite_isp_debug` — ISP 调试聚合器 🟡 需扩展

**当前状态**: 4 个 ISP 实例 (PCIE/HBM2/FFN/SYS)，FFN 部分连接的是自检 wrapper 信号  
**需要**: 连接到生产 FFN 引擎的真实 debug 端口

#### 3.6.1 FFN ISP probe 扩展 (当前 128-bit → 256-bit 或拆分为更多 probe word)

```verilog
// 现有 FFN probe (128-bit):
//   ffn_probe0[31:0] = STATUS
//   ffn_probe1[31:0] = PERF
//   ffn_probe2[31:0] = AXI_STATS
//   ffn_probe3[31:0] = DATA

// 扩展方案: 增加 ffn_probe4, ffn_probe5, ffn_probe6
wire [31:0] ffn_probe4;  // SUBMODULE STATUS
wire [31:0] ffn_probe5;  // SA Gate/Up Status
wire [31:0] ffn_probe6;  // SA Down Status

// FFN probe width: 128 → 224 (7 × 32-bit)
// 或者保持 128-bit，用 MUX 切换视角 (更省 SLD 资源)

// 推荐: 关键信号常驻，详细信息通过 SYS source 选择
```

#### 3.6.2 FFN ISP probe bit 分配 (修订版)

| Word | Bits | Signal | 描述 |
|------|------|--------|------|
| probe0 STATUS | [3:0] | `ffn_fsm` | FFN 主 FSM 状态 |
| | [6:4] | `expert_cnt` | 当前 expert 索引 |
| | [7] | `hbm_busy` | HBM2 reader 活跃 |
| | [8] | `sa_active` | 任一 SA 活跃 |
| | [9] | `silu_active` | SiLU 活跃 |
| | [10] | `merge_active` | Merge 活跃 |
| | [11] | `busy` | FFN busy |
| | [12] | `done` | FFN done |
| | [16:13] | `hbm2r_fsm` | HBM2 reader FSM |
| | [19:17] | `hbm2r_wr_watermark[2:0]` | Buffer 写水位线 (MSB 3 bits) |
| | [22:20] | `hbm2r_rd_watermark[2:0]` | Buffer 读水位线 (MSB 3 bits) |
| | [23] | `gate_done` | Gate 投影完成 |
| | [24] | `up_done` | Up 投影完成 |
| | [25] | `down_done` | Down 投影完成 |
| | [31:26] | Reserved | |
| probe1 PERF_LO | [31:0] | `perf_token_cnt[31:0]` | Token 计数器低 32-bit |
| probe2 PERF_HI | [15:0] | `perf_cycle_cnt[15:0]` | 周期计数低 16-bit |
| | [31:16] | `perf_expert_cnt[15:0]` | Expert 计数低 16-bit |
| probe3 AXI | [15:0] | `perf_axi_rbeat[15:0]` | AXI 读 beat 低 16-bit |
| | [31:16] | `perf_axi_ar_trans` | AXI AR 事务计数 |
| probe4 SA_STATUS | [3:0] | `sa_gate_fsm` | Gate/Up SA FSM |
| | [7:4] | `sa_down_fsm` | Down SA FSM |
| | [13:8] | `sa_gate_cycle` | Gate/Up 当前行内周期 |
| | [19:14] | `sa_down_cycle` | Down 当前行内周期 |
| | [31:20] | Reserved | |
| probe5 ERRORS | [0] | `err_axi_resp` | AXI 响应错误 |
| | [1] | `err_merge_ovf` | Merge 溢出 |
| | [2] | `err_silu_ovf` | SiLU 溢出 |
| | [31:3] | Reserved | |

**ISP 实例配置变更**:
```verilog
// 原来: probe_width = 128 (4 × 32-bit)
// 现在: probe_width = 192 (6 × 32-bit)  ← altsource_probe 支持到 512-bit
altsource_probe #(
    .sld_auto_instance_index ("YES"),
    .instance_id              ("FFN"),
    .probe_width              (192),     // 6 × 32-bit
    .source_width             (0),
    .enable_metastability     ("YES")
) u_ffn_isp (
    .probe  ({ffn_probe5, ffn_probe4, ffn_probe3, ffn_probe2, ffn_probe1, ffn_probe0}),
    ...
);
```

#### 3.6.3 Checklist (ISP)

- [ ] FFN probe 扩展到 192-bit (或 224-bit)
- [ ] probe0 STATUS 重新分配 FFN + HBM2 reader 状态
- [ ] probe1/2 映射性能计数器
- [ ] probe3 映射 AXI 状态
- [ ] probe4 新增 SA 状态 (gate/up + down)
- [ ] probe5 新增错误寄存器
- [ ] HBM2 ISP probe 补充 HBM2 reader 的 bytes_read/bursts 计数器
- [ ] SYS ISP source bit[0] 连接 FFN reset, bit[1] 连接 FFN start

---

### 3.7 `v2_lite_full_top` — 顶层连接 🔴 需重连

#### 3.7.1 Checklist

- [ ] 声明所有新增 debug wire
- [ ] FFN 实例化连接所有新端口
- [ ] `(* keep *)` 标记所有 debug wire (防止优化)
- [ ] ISP 实例化更新 probe 连接
- [ ] SignalTap STP 更新关键信号列表
- [ ] 自检 FSM 保留但改为旁路模式（生产模式下由 PCIe 驱动）

---

## 4. 实现优先级

| 优先级 | 模块 | 工作量 | 依赖 |
|--------|------|--------|------|
| **P0** | `v2_lite_ffn_engine` debug + 计数器 | 中 (~60行) | 无 |
| **P0** | `v2_lite_isp_debug` FFN probe 扩展 | 小 (~30行) | P0 FFN |
| **P0** | `v2_lite_full_top` 重连 | 小 (~40行) | P0 FFN + P0 ISP |
| P1 | `systolic_array` debug + 计数器 | 中 (~40行) | P0 FFN |
| P1 | `hbm2_weight_reader` debug + 计数器 | 中 (~50行) | P0 FFN |
| P2 | `silu_activation` debug 采样 | 小 (~20行) | P0 FFN |
| P2 | `fp8_mac` overflow 检测 | 小 (~15行) | P1 SA |

> **P0 = DSP 编译前必须完成，P1 = 首次编译后立即补齐，P2 = 时序验证阶段补齐**

## 5. 验证方法

### 5.1 综合验证
```bash
quartus_syn v2_lite_full
# 检查: debug 端口未被优化掉 (Synthesis Report → Removed Registers)
# 检查: 关键路径不经过 debug 组合逻辑
```

### 5.2 ISP 读回验证
```bash
# ic31 上:
quartus_stp -t read_isp_local.tcl
# 期望输出:
#   PCIE: PLL locked, all lanes OK
#   HBM2: TG pass, bytes_read > 0
#   FFN:  FSM progressing, token_cnt > 0, expert_cnt > 0
#   SYS:  Version = 0x0B061A01
```

### 5.3 SignalTap 验证
- 触发条件: `dbg_ffn_state == S_OUTPUT` (token 完成)
- 关键信号: `perf_token_cnt`, `perf_cycle_cnt`, `perf_axi_rbeat`
- 期望: 一个 token 的 FFN 延迟 < 2ms @ 100MHz (200,000 周期)

---

## 6. 文件修改汇总

| 文件 | 操作 | 优先级 |
|------|------|--------|
| `v2_lite/rtl/v2_lite_ffn_engine.sv` | 添加 debug 端口 + 计数器 + 版本号 | P0 |
| `v2_lite/rtl/v2_lite_isp_debug.v` | 扩展 FFN probe 到 192/224-bit | P0 |
| `v2_lite/rtl/v2_lite_full_top.v` | 重连 debug wire + ISP | P0 |
| `v2_lite/rtl/systolic_array.sv` | 添加 debug 端口 + 计数器 | P1 |
| `v2_lite/rtl/hbm2_weight_reader.sv` | 添加 debug 端口 + 计数器 | P1 |
| `v2_lite/rtl/silu_activation.sv` | 添加 debug 采样 | P2 |
| `v2_lite/rtl/fp8_mac.sv` | 添加 overflow 检测 | P2 |
| `v2_lite/v2_lite_full.stp` | 更新 SignalTap 信号列表 | P1 |
| `v2_lite/read_isp_local.tcl` | 更新 ISP 读回脚本 | P1 |
