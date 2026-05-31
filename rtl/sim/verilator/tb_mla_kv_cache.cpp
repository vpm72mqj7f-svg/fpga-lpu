//=============================================================================
// tb_mla_kv_cache.cpp — Verilator C++ testbench for mla_kv_cache (production)
//
// Production: NUM_SLOTS=4096, K_LATENT=512, V_LATENT=512
//
// Key timing for mla_kv_cache:
//   Write: set data + wr_en=1 at clk=0, tick (posedge commits), wr_en=0
//   Read:  set rd_addr + rd_en=1 at clk=0, tick (posedge latches rd_K_flat)
//   The syncram read is combinational; K_q settles on same eval as rd_addr.
//=============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <verilated.h>
#include "Vmla_kv_cache.h"

double sc_time_stamp() { return 0; }

static const int K_LATENT  = 512;
static const int V_LATENT  = 512;
static const int NUM_SLOTS = 4096;
static const int K_WORDS   = K_LATENT;

static void set_K(Vmla_kv_cache *dut, int base) {
    for (int i = 0; i < K_WORDS; i++)
        dut->K_latent_flat[i] = (uint32_t)(base + i);
}
static void set_V(Vmla_kv_cache *dut, int base) {
    for (int i = 0; i < V_LATENT; i++)
        dut->V_latent_flat[i] = (uint32_t)(base + i);
}
static void zero_KV(Vmla_kv_cache *dut) {
    for (int i = 0; i < K_WORDS; i++) {
        dut->K_latent_flat[i] = 0;
        dut->V_latent_flat[i] = 0;
    }
}

static void tick(Vmla_kv_cache *dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    dut->clk = 0; dut->eval();
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);

    Vmla_kv_cache *dut = new Vmla_kv_cache;
    int pass = 0, fail = 0;

    printf("=== mla_kv_cache Production Test ===\n");
    printf("  NUM_SLOTS=%d  K=%dx32=%dbit  V=%dx32=%dbit\n",
           NUM_SLOTS, K_LATENT, K_LATENT*32, V_LATENT, V_LATENT*32);

    // Reset
    dut->clk = 0; dut->rst_n = 0;
    dut->wr_en = 0; dut->rd_en = 0; dut->rd_addr = 0;
    zero_KV(dut);
    for (int i = 0; i < 5; i++) { tick(dut); }
    dut->rst_n = 1;
    tick(dut);

    printf("\n--- Test 1: Reset state ---\n");
    if (dut->empty == 1 && dut->fill_count == 0) {
        printf("  [PASS] empty=1, fill_count=0\n"); pass++;
    } else {
        printf("  [FAIL] empty=%d, fill_count=%d\n", dut->empty, dut->fill_count); fail++;
    }

    printf("\n--- Test 2: Single write + readback ---\n");
    zero_KV(dut);
    set_K(dut, 0x100);
    set_V(dut, 0x200);
    dut->wr_en = 1;
    tick(dut);   // write slot 0 at posedge
    dut->wr_en = 0;

    // Read slot 0: set rd_addr first, let combinational read settle (same eval)
    dut->rd_addr = 0;
    dut->rd_en = 1;
    tick(dut);   // latch rd_K/V at posedge
    dut->rd_en = 0;

    if (dut->rd_valid != 1) {
        printf("  [FAIL] rd_valid=%d\n", dut->rd_valid); fail++;
    } else {
        int mism = 0;
        for (int i = 0; i < K_WORDS && mism < 3; i++) {
            uint32_t exp = (uint32_t)(0x100 + i);
            if (dut->rd_K_flat[i] != exp) {
                printf("  K[%d]=0x%08x exp=0x%08x\n", i, dut->rd_K_flat[i], exp);
                mism++;
            }
        }
        if (mism == 0) {
            printf("  [PASS] rd_valid=1, data correct\n"); pass++;
        } else {
            printf("  [FAIL] %d K-mismatches\n", mism); fail++;
        }
    }

    // Reset to clear wr_ptr back to 0
    dut->rst_n = 0;
    for (int i = 0; i < 3; i++) { tick(dut); }
    dut->rst_n = 1;
    tick(dut);

    printf("\n--- Test 3: Ring buffer fill 4096 + readback ---\n");
    for (int s = 0; s < NUM_SLOTS; s++) {
        set_K(dut, s);
        set_V(dut, s + 0x10000);
        dut->wr_en = 1;
        tick(dut);
        dut->wr_en = 0;
        if ((s + 1) % 1024 == 0)
            printf("  Wrote %d / %d slots...\n", s + 1, NUM_SLOTS);
    }

    if (dut->full == 1 && dut->fill_count == NUM_SLOTS) {
        printf("  [PASS] full=1, fill_count=%d\n", dut->fill_count); pass++;
    } else {
        printf("  [FAIL] full=%d, fill_count=%d\n", dut->full, dut->fill_count); fail++;
    }

    // Read last slot: first wait a cycle for addr to settle
    dut->rd_en = 0;
    dut->rd_addr = NUM_SLOTS - 1;
    tick(dut);   // addr propagates through syncram
    dut->rd_en = 1;
    tick(dut);   // latch
    dut->rd_en = 0;

    if (dut->rd_valid == 1) {
        int mism = 0;
        for (int i = 0; i < K_WORDS && mism < 5; i++) {
            uint32_t exp = (uint32_t)(NUM_SLOTS - 1 + i);
            if (dut->rd_K_flat[i] != exp) {
                printf("  K[%d]=0x%08x exp=0x%08x\n", i, dut->rd_K_flat[i], exp);
                mism++;
            }
        }
        if (mism == 0) {
            printf("  [PASS] Slot %d readback correct\n", NUM_SLOTS - 1); pass++;
        } else {
            printf("  [FAIL] %d mismatches in slot %d\n", mism, NUM_SLOTS - 1); fail++;
        }
    } else {
        printf("  [FAIL] Slot %d rd_valid=0\n", NUM_SLOTS - 1); fail++;
    }

    printf("\n--- Test 4: Verify slot range (5 random slots) ---\n");
    int test_slots[] = {0, 1024, 2048, 3072, 4095};
    int slot_bad = 0;
    for (int si = 0; si < 5; si++) {
        int s = test_slots[si];
        dut->rd_en = 0;
        dut->rd_addr = s;
        tick(dut);
        dut->rd_en = 1;
        tick(dut);
        dut->rd_en = 0;
        if (dut->rd_valid != 1) {
            printf("  [FAIL] Slot %d rd_valid=0\n", s);
            slot_bad++;
        } else if (dut->rd_K_flat[0] != (uint32_t)s) {
            printf("  [FAIL] Slot %d K[0]=0x%08x exp=0x%08x\n", s, dut->rd_K_flat[0], s);
            slot_bad++;
        }
    }
    if (slot_bad == 0) {
        printf("  [PASS] All 5 sampled slots have correct data\n"); pass++;
    } else {
        printf("  [FAIL] %d slot mismatches\n", slot_bad); fail++;
    }

    printf("\n==============================\n");
    if (fail == 0)
        printf("PASS tb_mla_kv_cache (%d/%d tests)\n", pass, pass + fail);
    else
        printf("FAIL tb_mla_kv_cache (%d pass, %d fail)\n", pass, fail);

    dut->final();
    delete dut;
    return (fail > 0) ? 1 : 0;
}
