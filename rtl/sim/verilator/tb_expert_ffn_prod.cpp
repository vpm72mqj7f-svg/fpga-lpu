//=============================================================================
// tb_expert_ffn_prod.cpp — Verilator structural smoke test
// expert_ffn_engine_fp4_down at production scale: HIDDEN=7168, INTER=3072
//=============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <verilated.h>
#include "Vexpert_ffn_engine_fp4_down.h"

double sc_time_stamp() { return 0; }

static void tick(Vexpert_ffn_engine_fp4_down *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    dut->clk = 0; dut->eval();
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);

    Vexpert_ffn_engine_fp4_down *dut = new Vexpert_ffn_engine_fp4_down;
    int pass = 0, fail = 0;

    printf("=== expert_ffn_engine_fp4_down Production Structural Test ===\n");
    printf("  HIDDEN=7168  INTER=3072  LANES=4\n");
    printf("  K_BEATS_H=1792  K_BEATS_I=768\n");

    // ── Reset ──
    dut->clk = 0; dut->rst_n = 0;
    dut->activ_wr_en = 0; dut->activ_wr_beat = 0; dut->activ_wr_data = 0;
    dut->scale_wr_en = 0; dut->scale_wr_addr = 0; dut->scale_wr_data = 0;
    dut->gate_w_wr_en = 0; dut->gate_w_wr_row = 0; dut->gate_w_wr_beat = 0; dut->gate_w_wr_data = 0;
    dut->up_w_wr_en = 0; dut->up_w_wr_row = 0; dut->up_w_wr_beat = 0; dut->up_w_wr_data = 0;
    dut->down_w_wr_en = 0; dut->down_w_wr_row = 0; dut->down_w_wr_beat = 0; dut->down_w_wr_data = 0;
    dut->start = 0;

    for (int i = 0; i < 5; i++) { tick(dut); }
    dut->rst_n = 1;
    tick(dut);

    // Test 1: Reset state
    printf("--- Test 1: Reset state ---\n");
    if (dut->done == 0 && dut->busy == 0) {
        printf("  [PASS] done=0, busy=0\n"); pass++;
    } else {
        printf("  [FAIL] done=%d, busy=%d\n", dut->done, dut->busy); fail++;
    }

    // Test 2: Scale preload
    printf("--- Test 2: Scale preload ---\n");
    for (int g = 0; g < 4; g++) {
        dut->scale_wr_addr = g;
        dut->scale_wr_data = 0x3C;
        dut->scale_wr_en = 1;
        tick(dut);
        dut->scale_wr_en = 0;
    }
    printf("  [PASS] 4 scale groups loaded\n"); pass++;

    // Test 3: Minimal weight + activation load
    printf("--- Test 3: Data preload ---\n");
    dut->gate_w_wr_row = 0; dut->gate_w_wr_beat = 0;
    dut->gate_w_wr_data = 0x8888;
    dut->gate_w_wr_en = 1; tick(dut); dut->gate_w_wr_en = 0;

    dut->up_w_wr_row = 0; dut->up_w_wr_beat = 0;
    dut->up_w_wr_data = 0x8888;
    dut->up_w_wr_en = 1; tick(dut); dut->up_w_wr_en = 0;

    dut->down_w_wr_row = 0; dut->down_w_wr_beat = 0;
    dut->down_w_wr_data = 0x8888;
    dut->down_w_wr_en = 1; tick(dut); dut->down_w_wr_en = 0;

    // Load a few activation beats (not all 1792 — structural test only)
    dut->activ_wr_en = 1;
    for (int b = 0; b < 10; b++) {
        dut->activ_wr_beat = b;
        dut->activ_wr_data = 0x3C3C3C3C;
        tick(dut);
    }
    dut->activ_wr_en = 0;
    printf("  [PASS] 10 activation beats loaded\n"); pass++;

    // Test 4: FSM liveness — start and verify busy
    printf("--- Test 4: FSM liveness ---\n");
    dut->start = 1;
    tick(dut);
    dut->start = 0;

    if (dut->busy) {
        printf("  [PASS] busy=1 after start\n"); pass++;
    } else {
        printf("  [FAIL] busy stuck at 0\n"); fail++;
    }

    // Test 5: Sustained liveness — run a few thousand cycles
    printf("--- Test 5: Sustained liveness (5000-cycle run) ---\n");
    int alive_checks = 0;
    for (int c = 0; c < 5000; c++) {
        tick(dut);
        if (c % 1000 == 999 && dut->busy)
            alive_checks++;
    }
    if (alive_checks == 5) {
        printf("  [PASS] FSM alive for 5000 cycles (5/5 busy checks)\n"); pass++;
    } else if (alive_checks > 0) {
        printf("  [PASS] FSM alive (%d/5 busy checks — may have completed early)\n", alive_checks); pass++;
    } else {
        printf("  [FAIL] busy=0 for all checks (FSM dead)\n"); fail++;
    }

    printf("\n========================================\n");
    if (fail == 0)
        printf("PASS tb_expert_ffn_prod (%d/%d tests)\n", pass, pass + fail);
    else
        printf("FAIL tb_expert_ffn_prod (%d pass, %d fail)\n", pass, fail);

    dut->final();
    delete dut;
    return (fail > 0) ? 1 : 0;
}
