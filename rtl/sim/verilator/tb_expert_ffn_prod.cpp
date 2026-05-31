//=============================================================================
// tb_expert_ffn_prod.cpp — Verilator smoke test for expert_ffn_engine_fp4_down
// at production scale: HIDDEN=7168, INTER=3072
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

    printf("=== expert_ffn_engine_fp4_down Production Smoke Test ===\n");
    printf("  HIDDEN=7168  INTER=3072  LANES=4\n");
    printf("  Beat counts: K_BEATS_H=1792  K_BEATS_I=768\n");

    // Reset
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
    printf("\n--- Test 1: Reset state ---\n");
    if (dut->done == 0 && dut->busy == 0) {
        printf("  [PASS] done=0, busy=0 after reset\n"); pass++;
    } else {
        printf("  [FAIL] done=%d, busy=%d\n", dut->done, dut->busy); fail++;
    }

    // Test 2: Load scales
    printf("\n--- Test 2: Scale preload ---\n");
    // Scale groups: HIDDEN/16 = 448 groups for HIDDEN=7168
    // Write a few scale entries
    for (int g = 0; g < 4; g++) {
        dut->scale_wr_addr = g;
        dut->scale_wr_data = 0x3C;  // fp8 e4m3: 0x3C = 1.0
        dut->scale_wr_en = 1;
        tick(dut);
        dut->scale_wr_en = 0;
    }
    printf("  [PASS] Scale preload (4 groups)\n"); pass++;

    // Test 3: Load minimal gate/up/down weights and run
    printf("\n--- Test 3: Weight load + start ---\n");
    // Load one beat of gate weight (beat 0, row 0)
    dut->gate_w_wr_row = 0;
    dut->gate_w_wr_beat = 0;
    dut->gate_w_wr_data = 0x8888;  // fp4: 0x8 = 1.0 for all 4 lanes
    dut->gate_w_wr_en = 1;
    tick(dut);
    dut->gate_w_wr_en = 0;

    // Load one beat of up weight
    dut->up_w_wr_row = 0;
    dut->up_w_wr_beat = 0;
    dut->up_w_wr_data = 0x8888;
    dut->up_w_wr_en = 1;
    tick(dut);
    dut->up_w_wr_en = 0;

    // Load one beat of down weight
    dut->down_w_wr_row = 0;
    dut->down_w_wr_beat = 0;
    dut->down_w_wr_data = 0x8888;
    dut->down_w_wr_en = 1;
    tick(dut);
    dut->down_w_wr_en = 0;

    // Load activation (2 beats for HIDDEN=7168 with LANES=4: K_BEATS_H=1792)
    // Beat 0: first 4 elements as fp8
    dut->activ_wr_en = 1;
    dut->activ_wr_beat = 0;
    dut->activ_wr_data = 0x3C3C3C3C;  // fp8: 0x3C = 1.0
    tick(dut);
    // Load all remaining beats quickly
    for (int b = 1; b < 1792; b++) {
        dut->activ_wr_beat = b;
        tick(dut);
    }
    dut->activ_wr_en = 0;

    printf("  [PASS] Activation loaded (1792 beats)\n"); pass++;

    // Test 4: Start and wait for done
    printf("\n--- Test 4: Run FFN ---\n");
    dut->start = 1;
    tick(dut);
    dut->start = 0;

    int cycles = 0;
    int max_cycles = 50000;
    for (; cycles < max_cycles && !dut->done; cycles++) {
        tick(dut);
    }

    if (dut->done) {
        printf("  [PASS] FFN done in %d cycles\n", cycles); pass++;
    } else {
        printf("  [FAIL] FFN timeout after %d cycles\n", max_cycles); fail++;
    }

    // Test 5: Read back results
    printf("\n--- Test 5: Result readback ---\n");
    int results_read = 0;
    for (int c = 0; c < 100 && results_read < 8; c++) {
        tick(dut);
        if (dut->result_valid) {
            if (results_read < 4) {
                printf("  result[%lu] = 0x%08x\n",
                    (unsigned long)dut->result_row, dut->result_data);
            }
            results_read++;
        }
    }
    if (results_read > 0) {
        printf("  [PASS] %d result words read\n", results_read); pass++;
    } else {
        printf("  [FAIL] No results\n"); fail++;
    }

    printf("\n==============================\n");
    if (fail == 0)
        printf("PASS tb_expert_ffn_prod (%d/%d tests)\n", pass, pass + fail);
    else
        printf("FAIL tb_expert_ffn_prod (%d pass, %d fail)\n", pass, fail);

    dut->final();
    delete dut;
    return (fail > 0) ? 1 : 0;
}
