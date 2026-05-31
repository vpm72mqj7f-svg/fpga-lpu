// Minimal smoke test for mla_kv_cache production build
#include <cstdio>
#include <cstdlib>
#include <verilated.h>
#include "Vmla_kv_cache.h"

double sc_time_stamp() { return 0; }

int main(int argc, char **argv) {
    printf("Starting...\n"); fflush(stdout);
    Verilated::commandArgs(argc, argv);

    Vmla_kv_cache *dut = new Vmla_kv_cache;
    printf("DUT created\n"); fflush(stdout);

    dut->clk = 0;
    dut->rst_n = 0;
    dut->wr_en = 0;
    dut->rd_en = 0;
    dut->rd_addr = 0;
    for (int i = 0; i < 512; i++) {
        dut->K_latent_flat[i] = 0;
        dut->V_latent_flat[i] = 0;
    }

    printf("Evaling...\n"); fflush(stdout);
    dut->eval();
    printf("Empty=%d, fill=%d, full=%d\n", dut->empty, dut->fill_count, dut->full);

    dut->final();
    delete dut;
    printf("Done.\n");
    return 0;
}
