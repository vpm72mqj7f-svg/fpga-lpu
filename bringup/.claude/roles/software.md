# Software Role — FPGA Backend + llama.cpp Integration

**Session:** fpgalpu 主 session  
**Server:** huahuan@172.18.40.202 (ARM N2 256核 + RTX 4090D)  
**Last Update:** 2026-06-13

## 已完成

### ggml_backend_fpga (llama.cpp 侧)
- `ggml-backend-fpga.h` — 公开 API，112 行
- `ggml-backend-fpga.cpp` — 5 个 vtable 全部实现 (Registry/Device/Backend/BufferType/Buffer)，705 行
- Stub 模式：heap 分配 + memcpy，无硬件可跑
- `test_fpga_backend.cpp` — 10/10 单元测试通过
- `patch_llama_fpga.py` — 一键打 patch 到 llama.cpp-v4
- `--fpga-moe` CLI flag 已加入 common.h/arg.cpp/common.cpp
- CMake 集成: `-DGGML_FPGA=ON -DGGML_FPGA_STUB_MODE=ON`
- 远程编译通过，测试通过

### c_ref/fpga/ (主机侧驱动)
- `pcie_driver.h/c` — UIO BAR mmap + 寄存器读写 + DMA 写入 (270 行)
- `gguf_export.py` — GGUF → fp4 packed .bin 导出
- `Makefile` — 编译 libfpga_backend.so
- `tests/test_dma_loopback.c` — BAR0 映射 + 寄存器读写测试
- `tests/test_expert_load.c` — Expert 权重加载 + 带宽 benchmark
- `tests/test_fpga_e2e.c` — 端到端: 加载 → 发激活 → 跑 FFN → 读结果

### V2-Lite P/D 分离验证
- CPU baseline: 17.6 tok/s
- GPU-All: 86.3 tok/s
- PD-Split (attn→CPU, FFN→GPU): 31.0 tok/s
- 结论: V2-Lite 太小，P/D 分离意义不大；直接走 FPGA FFN 验证

### 寄存器契约
- `v2_lite/v2_lite_weight_writer.atreg` — 按项目 .atreg 格式写的寄存器规范
- `pcie_driver.h` 寄存器宏已对齐

## 需要 RTL 侧配合

1. **确认 weight_writer.atreg 寄存器地址** — 综合后地址有无偏移？
2. **pcie_hbm_weight_writer.sv 模块** — PCIe BAR0 → AXI4 write bridge
3. **连 AXI write channel 到 ed_synth** — ffn_axi_aw*/w*/b* 目前接 0
4. **FFN engine pcie_rx_* 接 weight writer 的激活输出** — 目前接 self-test FSM
5. **.sof 就绪后通知** — 我这侧 UIO 驱动立即可测

## 下一步 (等 RTL 就绪后)

1. 远程 ARM 服务器上 `make arm64` 交叉编译 c_ref/fpga/
2. `scp` 到 ARM 服务器
3. `test_dma_loopback` → `test_expert_load` → `llama-server --fpga-moe`
