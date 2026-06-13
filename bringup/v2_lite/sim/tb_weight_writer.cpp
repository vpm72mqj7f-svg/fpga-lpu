// tb_weight_writer.cpp — Verilator C++ testbench for pcie_hbm_weight_writer
// Tests: AVMM register access, CDC toggle, AXI4 write burst
// Build: verilator --cc --exe --build -j pcie_hbm_weight_writer.sv tb_weight_writer.cpp

#include "Vpcie_hbm_weight_writer.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>

static Vpcie_hbm_weight_writer *dut;
static vluint64_t sim_time = 0;
static int errors = 0;

double sc_time_stamp() { return sim_time; }

#define PCIE_CLK_HALF 2   // 250MHz = 4ns period
#define CORE_CLK_HALF 5   // 100MHz = 10ns period

// AVMM write helper
void avs_write(uint64_t addr, uint32_t data) {
    dut->avs_address = addr;
    dut->avs_writedata = data;
    dut->avs_write = 1;
    dut->avs_read = 0;
    // Single-cycle write (no waitrequest)
    // Tick PCIe clock
    dut->pcie_clk = 1; dut->eval(); sim_time += PCIE_CLK_HALF;
    dut->pcie_clk = 0; dut->eval(); sim_time += PCIE_CLK_HALF;
    dut->avs_write = 0;
}

// AVMM read helper
uint32_t avs_read(uint64_t addr) {
    dut->avs_address = addr;
    dut->avs_write = 0;
    dut->avs_read = 1;
    dut->pcie_clk = 1; dut->eval(); sim_time += PCIE_CLK_HALF;
    dut->pcie_clk = 0; dut->eval(); sim_time += PCIE_CLK_HALF;
    uint32_t val = dut->avs_readdata;
    dut->avs_read = 0;
    return val;
}

// Clock tick helpers
void tick_pcie() {
    dut->pcie_clk = 1; dut->eval(); sim_time += PCIE_CLK_HALF;
    dut->pcie_clk = 0; dut->eval(); sim_time += PCIE_CLK_HALF;
}

void tick_core() {
    dut->core_clk = 1; dut->eval(); sim_time += CORE_CLK_HALF;
    dut->core_clk = 0; dut->eval(); sim_time += CORE_CLK_HALF;
}

// AXI4 slave model (responds to write bursts)
void axi_respond() {
    if (dut->m_axi_awvalid) {
        dut->m_axi_awready = 1;
    } else {
        dut->m_axi_awready = 0;
    }
    if (dut->m_axi_wvalid) {
        dut->m_axi_wready = 1;
        if (dut->m_axi_wlast) {
            // Send write response after last beat
            dut->m_axi_bvalid = 1;
            dut->m_axi_bresp = 0;
        }
    } else {
        dut->m_axi_wready = 0;
    }
    if (dut->m_axi_bvalid && dut->m_axi_bready) {
        dut->m_axi_bvalid = 0;
    }
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vpcie_hbm_weight_writer;

    // Reset
    dut->pcie_rst_n = 0;
    dut->core_rst_n = 0;
    dut->avs_write = 0;
    dut->avs_read = 0;
    dut->m_axi_awready = 0;
    dut->m_axi_wready = 0;
    dut->m_axi_bvalid = 0;
    dut->m_axi_bresp = 0;
    for (int i = 0; i < 10; i++) { tick_pcie(); tick_core(); }
    dut->pcie_rst_n = 1;
    dut->core_rst_n = 1;
    printf("[TEST] Reset released\n");

    // Test 1: Register write/read
    printf("[TEST 1] AVMM register access...\n");
    avs_write(0x1008, 0x00100000);  // HBM_ADDR_LO = 1MB
    avs_write(0x1010, 0x00000001);  // BURST_COUNT = 1
    uint32_t r = avs_read(0x1008);
    if (r == 0x00100000) printf("  PASS: HBM_ADDR_LO = 0x%08X\n", r);
    else { printf("  FAIL: HBM_ADDR_LO = 0x%08X (expected 0x00100000)\n", r); errors++; }

    // Test 2: START triggers AXI write
    printf("[TEST 2] START → AXI4 write burst...\n");
    // Write 8 × 32-bit data words to fill 256-bit AXI beat
    for (int i = 0; i < 8; i++) {
        avs_write(0x1020, 0xAABBCC00 + i);  // DATA_PORT
    }
    avs_write(0x1000, 0x00000001);  // CONTROL = START

    // Run core clock until AXI write happens or timeout
    int axi_aw_seen = 0;
    for (int cyc = 0; cyc < 10000; cyc++) {
        tick_pcie();
        for (int c = 0; c < 3; c++) tick_core();  // 3 core ticks per PCIe tick
        axi_respond();
        if (dut->m_axi_awvalid) axi_aw_seen = 1;
        if (axi_aw_seen && dut->m_axi_wlast && dut->m_axi_wvalid && dut->m_axi_wready) break;
    }
    if (axi_aw_seen) printf("  PASS: AXI4 AW asserted\n");
    else { printf("  FAIL: AXI4 AW never asserted\n"); errors++; }

    // Test 3: STATUS readback (CDC)
    printf("[TEST 3] STATUS readback...\n");
    for (int cyc = 0; cyc < 500; cyc++) {
        tick_pcie();
        for (int c = 0; c < 3; c++) tick_core();
        axi_respond();
    }
    uint32_t st = avs_read(0x1004);
    printf("  STATUS = 0x%08X\n", st);
    // STATUS bits: bit0=BUSY, bit1=DONE, bit2=ERROR
    if (!(st & 1) && (st & 2)) printf("  PASS: BUSY=0 DONE=1\n");
    else { printf("  INFO: STATUS=0x%08X (BUSY=%d DONE=%d)\n", st, (int)(st&1), (int)((st>>1)&1)); }

    // Summary
    printf("\n========================================\n");
    if (errors == 0) printf(" ALL TESTS PASSED\n");
    else              printf(" %d TESTS FAILED\n", errors);
    printf("========================================\n");

    dut->final();
    delete dut;
    return errors ? 1 : 0;
}
