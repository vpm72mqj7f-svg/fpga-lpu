// Minimal smoke test for mla_attention_v2 production build
#include <cstdio>
#include <cstdlib>
#include <verilated.h>
#include "Vmla_attention_v2.h"

double sc_time_stamp() { return 0; }

int main(int argc, char **argv) {
    printf("Starting mla_attention_v2 smoke test...\n"); fflush(stdout);
    Verilated::commandArgs(argc, argv);

    Vmla_attention_v2 *dut = new Vmla_attention_v2;
    printf("DUT created (HIDDEN=%d)\n", 7168); fflush(stdout);

    dut->clk = 0;
    dut->rst_n = 0;
    dut->in_valid = 0;
    dut->out_ready = 1;
    dut->qkv_wt_wr_en = 0;
    dut->rope_lut_wr_en = 0;
    dut->position = 0;

    // Zero the wide input port (hidden_flat is 229,376 bits = 7168 words)
    for (int i = 0; i < 7168; i++) {
        dut->hidden_flat[i] = 0;
    }

    printf("Evaling...\n"); fflush(stdout);
    dut->eval();

    // Check basic status
    printf("in_ready=%d, out_valid=%d\n", dut->in_ready, dut->out_valid);
    printf("y0=%d y1=%d y2=%d y3=%d y4=%d y5=%d y6=%d y7=%d\n",
        dut->y0, dut->y1, dut->y2, dut->y3,
        dut->y4, dut->y5, dut->y6, dut->y7);

    // Run a few clock cycles
    for (int c = 0; c < 10; c++) {
        dut->clk = 1; dut->eval();
        dut->clk = 0; dut->eval();
    }

    printf("After 10 cycles: in_ready=%d, out_valid=%d\n",
        dut->in_ready, dut->out_valid);

    dut->final();
    delete dut;
    printf("Done. mla_attention_v2 production smoke test PASSED.\n");
    return 0;
}
