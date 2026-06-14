# RTL 开发 SOP — 严格流程

**服务器：** ARM仿真 172.18.40.202 | 综合 172.18.10.31  
**寄存器规范：** `docs/v2_lite_pcie_regmap.atreg`（唯一真相源）  
**设计指引：** `FFN_ENGINE_SPEC.md` + `HBM2_WEIGHT_MAP.md` + `GEMV_ENGINE_DESIGN.md`

---

## Step 1: 写 RTL（本地）

```
☐ 1.1 打开 docs/v2_lite_pcie_regmap.atreg，确认要实现的寄存器地址和 bit 定义
☐ 1.2 打开 FFN_ENGINE_SPEC.md，确认 DSP 配比、FSM 状态、时序约束
☐ 1.3 打开 HBM2_WEIGHT_MAP.md，确认 expert 地址计算
☐ 1.4 写 RTL 文件 (.sv)，模块名和端口名必须与 atreg 一致
☐ 1.5 自检：寄存器 offset 是否与 atreg 对齐？bit 位是否匹配？
☐ 1.6 自检：AXI 接口信号名是否与 v2_lite_full_top.sv 中 ed_synth 的端口一致？
```

**规则：不准 AI 一次生成全部代码——AI 只能用来做逐模块的语法检查、地址计算验证、diff 对比。每次自己写完后用 AI review。**

---

## Step 2: 仿真（ARM 服务器 172.18.40.202）

```
☐ 2.1 登录: ssh huahuan@172.18.40.202  (密码 Admin123)
☐ 2.2 把 RTL 文件 scp 到 /home/huahuan/v2_lite_sim/
☐ 2.3 写 testbench（或复用已有 tb），重点验证：
       - 寄存器读写（BAR0 AVMM → reg 值一致）
       - AXI 地址生成（验证 gate/up/down 地址公式）
       - FSM 状态跳转
☐ 2.4 运行 Verilator: verilator --cc --build -j 32 <rtl_file>.sv tb_<rtl_file>.cpp
☐ 2.5 确认: 所有 test case PASS（截图或保存 log）
☐ 2.6 确认: 无 latch warning、无 combinational loop
```

**规则：Verilator PASS 是进 Step 3 的前置条件。没过不准 commit。**

---

## Step 3: 版本管理

```
☐ 3.1 git status — 确认改动的文件列表
☐ 3.2 git add <rtl_file.sv> <testbench> <atreg 如有更新>
☐ 3.3 git commit -m "feat: <模块名> — <一句话改动描述>"
      格式: feat: v2_lite_bar0_regs — add FFN_DATA_LO/HI/VALID registers
            fix:  ffn_gemv_array — add preserve on mac_accum, prevent DSP removal
            spec: HBM2_WEIGHT_MAP — define expert address layout
☐ 3.4 git push origin master
☐ 3.5 确认 push 成功: git log --oneline -3
```

**规则：每次 commit 只改一个逻辑块。不许一次 commit 里混 RTL + QSF + SDC。**

---

## Step 4: 跑综合（综合服务器 172.18.10.31）

```
☐ 4.1 登录综合服务器
☐ 4.2 cd /home/ic-server31/bringup/v2_lite_full 或项目路径
☐ 4.3 git pull origin master — 拉取最新 RTL
☐ 4.4 跑 fix_qsf: quartus_sh -t scripts/fix_qsf_pcie_ep.tcl
       （修复 QSF 里 pcie_ep 条目丢失问题）
☐ 4.5 确认 QSYS_FILE 包含:
       - ed_synth.qsys
       - pcie_ep.qsys  ← 关键！不能是 pcie_xcvr_system
☐ 4.6 确认 SYSTEMVERILOG_FILE 包含:
       - v2_lite_full_top.sv
       - v2_lite_bar0_regs.sv
       - pcie_hbm_weight_writer.sv
       - ffn_gemv_array.sv
       - v2_lite_ffn_engine.sv
       - hbm2_weight_reader.sv
       - silu_activation.sv
☐ 4.7 跑综合: bash scripts/synth_v2_full.sh
☐ 4.8 等待完成（通常 30-90 分钟）
```

**规则：QSF 每次编译前必须跑 fix_qsf。综合脚本里也加一行 `quartus_sh -t scripts/fix_qsf_pcie_ep.tcl` 作为第一行。**

---

## Step 5: 看结果

```
☐ 5.1 检查综合成功: grep "successful" v2_lite_full_syn.log
☐ 5.2 检查 DSP 数量: grep "DSP block" output_files/v2_lite_full.map.rpt
       期望: DSP > 0（具体数量取决于当前配比，512 MAC ≈ 512 DSP）
       如果 DSP = 0 → 返回 Step 1，检查 weight 路径是否从 HBM2 来
☐ 5.3 检查 ALM 数量: grep "Total comb" output_files/v2_lite_full.map.rpt
☐ 5.4 检查 Timing: grep "Slack" output_files/v2_lite_full.sta.rpt
       期望: 所有 slack ≥ 0 ns
       如果有负 slack: 记录最差路径的 start/end point，返回 Step 1 加 pipeline
☐ 5.5 检查 Warning: grep "^Warning" v2_lite_full_syn.log | wc -l
       新产生的 warning（之前没有的）必须解释原因
☐ 5.6 把综合结果写到 TASKS.md:
       DSP count: 512 / 3,960
       ALM: XXXXX
       Slack: +0.xxx ns
       Warnings: N（其中新产生 M 个，原因: ...）
```

**规则：结果必须写进 TASKS.md，不能只在聊天里说"过了"。下次 session 进来能直接看到上次综合数据。**

---

## 快速参考

```
服务器          IP              用途
────────────────────────────────────────────
ARM 仿真       172.18.40.202    Verilator, testbench, c_ref 编译
综合            172.18.10.31     Quartus syn/fit/asm

关键文件                        位置
────────────────────────────────────────────
寄存器规范                      docs/v2_lite_pcie_regmap.atreg
FFN 设计指引                    FFN_ENGINE_SPEC.md
HBM2 地址映射                   HBM2_WEIGHT_MAP.md
GEMV 引擎设计                   GEMV_ENGINE_DESIGN.md
QSF 修复脚本                    scripts/fix_qsf_pcie_ep.tcl
综合脚本                        scripts/synth_v2_full.sh
任务板                          TASKS.md
```

## 禁止事项

```
❌ 不先看 atreg 就写 RTL
❌ Verilator 没过就 commit
❌ 一次 commit 混改多个不相关的文件
❌ 综合前不跑 fix_qsf
❌ DSP=0 就当"综合过了"
❌ 综合结果只发聊天不写 TASKS.md
❌ AI 一次性生成整个模块（只能逐模块 review）
❌ 改了 atreg 不同步更新 pcie_driver.h（软件侧）
```
