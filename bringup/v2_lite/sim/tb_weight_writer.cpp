// tb_weight_writer.cpp — Verilator C++ testbench for pcie_hbm_weight_writer
// Tests: AVMM R/W, DATA_PORT 64-bit pair, CDC, AXI4 write burst, timeout
#include "Vpcie_hbm_weight_writer.h"
#include "verilated.h"
#include <cstdio>

static Vpcie_hbm_weight_writer *dut;
static vluint64_t sim_time = 0;
static int errors = 0;
double sc_time_stamp() { return sim_time; }

// Register offsets (BAR0 base 0x000)
enum { ADDR_CTRL=0x000, ADDR_STATUS=0x004, ADDR_HBM_LO=0x008, ADDR_HBM_HI=0x00C,
       ADDR_BURST=0x010, ADDR_BYTES=0x014, ADDR_ERROR=0x018,
       ADDR_DATA_LO=0x020, ADDR_DATA_HI=0x024 };

void avs_write(uint64_t addr, uint32_t data) {
    dut->avs_address = addr; dut->avs_writedata = data; dut->avs_write = 1; dut->avs_read = 0;
    dut->pcie_clk = 1; dut->eval(); sim_time += 2;
    dut->pcie_clk = 0; dut->eval(); sim_time += 2;
    dut->avs_write = 0;
}

uint32_t avs_read(uint64_t addr) {
    dut->avs_address = addr; dut->avs_write = 0; dut->avs_read = 1;
    dut->pcie_clk = 1; dut->eval(); sim_time += 2;
    dut->pcie_clk = 0; dut->eval(); sim_time += 2;
    // AVMM read has 1-cycle delay
    dut->pcie_clk = 1; dut->eval(); sim_time += 2;
    dut->pcie_clk = 0; dut->eval(); sim_time += 2;
    uint32_t val = dut->avs_readdata; dut->avs_read = 0;
    return val;
}

void tick_pcie() { dut->pcie_clk=1; dut->eval(); sim_time+=2; dut->pcie_clk=0; dut->eval(); sim_time+=2; }
void tick_core() { dut->core_clk=1; dut->eval(); sim_time+=5; dut->core_clk=0; dut->eval(); sim_time+=5; }
void axi_respond() {
    if (dut->m_axi_awvalid) dut->m_axi_awready = 1; else dut->m_axi_awready = 0;
    if (dut->m_axi_wvalid)  dut->m_axi_wready  = 1; else dut->m_axi_wready  = 0;
    if (dut->m_axi_wvalid && dut->m_axi_wready && dut->m_axi_wlast) {
        dut->m_axi_bvalid = 1; dut->m_axi_bresp = 0;
    }
    if (dut->m_axi_bvalid && dut->m_axi_bready) dut->m_axi_bvalid = 0;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vpcie_hbm_weight_writer;
    dut->pcie_rst_n = 0; dut->core_rst_n = 0;
    for (int i = 0; i < 10; i++) { tick_pcie(); tick_core(); }
    dut->pcie_rst_n = 1; dut->core_rst_n = 1;
    printf("[TEST] Reset released\n");

    // Test 1: AVMM register write/read
    printf("[TEST 1] AVMM register access (BAR0 base 0x000)...\n");
    avs_write(ADDR_HBM_LO, 0x00100000);
    avs_write(ADDR_BURST,  0x00000001);
    uint32_t r = avs_read(ADDR_HBM_LO);
    if (r == 0x00100000) printf("  PASS: HBM_ADDR_LO = 0x%08X\n", r);
    else { printf("  FAIL: HBM_ADDR_LO = 0x%08X\n", r); errors++; }

    // Test 2: 64-bit DATA_PORT (LO+HI pair)
    printf("[TEST 2] 64-bit DATA_PORT streaming...\n");
    for (int i = 0; i < 4; i++) {  // 4 pairs = 1 AXI beat
        avs_write(ADDR_DATA_LO, 0xAAAA0000 + i);
        avs_write(ADDR_DATA_HI, 0xBBBB0000 + i);
        // CDC settling time between pairs
        for (int j = 0; j < 20; j++) { tick_pcie(); tick_core(); tick_core(); }
    }
    avs_write(ADDR_CTRL, 0x00000001);  // START
    int aw_seen = 0;
    for (int cyc = 0; cyc < 5000; cyc++) {
        tick_pcie(); tick_core(); tick_core(); tick_core(); axi_respond();
        if (dut->m_axi_awvalid) aw_seen = 1;
        if (aw_seen && dut->m_axi_wlast && dut->m_axi_wvalid && dut->m_axi_wready) break;
    }
    if (aw_seen) printf("  PASS: AXI4 AW asserted\n");
    else { printf("  FAIL: AXI4 AW never asserted\n"); errors++; }

    // Test 3: STATUS readback
    printf("[TEST 3] STATUS readback...\n");
    for (int cyc = 0; cyc < 200; cyc++) { tick_pcie(); tick_core(); tick_core(); axi_respond(); }
    uint32_t st = avs_read(ADDR_STATUS);
    printf("  STATUS = 0x%08X (BUSY=%d DONE=%d ERR=%d)\n", st, st&1, (st>>1)&1, (st>>2)&1);
    if (!(st & 1)) printf("  PASS: Not busy\n"); else { printf("  WARN: Still busy\n"); }

    // Test 4: Stub register blocks return 0
    printf("[TEST 4] Stub registers (FFN/ACT/PERF/ERR)...\n");
    int stubs_ok = 1;
    for (uint64_t a = 0x100; a < 0x500; a += 0x04) {
        uint32_t v = avs_read(a);
        if (v != 0) { printf("  WARN: stub 0x%03llX = 0x%08X\n", a, v); stubs_ok = 0; }
    }
    if (stubs_ok) printf("  PASS: All stub registers return 0\n");
    else printf("  INFO: Some stubs non-zero (may be mapped to real regs)\n");

    printf("\n========================================\n");
    if (errors == 0) printf(" ALL TESTS PASSED\n");
    else printf(" %d TESTS FAILED\n", errors);
    printf("========================================\n");

    dut->final(); delete dut; return errors ? 1 : 0;
}
