// tb_ffn_axi.cpp — Verilator testbench for FFN + HBM2 AXI
#include "Vv2_lite_ffn_engine.h"
#include "verilated.h"
#include <cstdio>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vv2_lite_ffn_engine* top = new Vv2_lite_ffn_engine;

    top->clk = 0;
    top->rst_n = 0;
    top->mode_prefill = 0;
    top->pcie_rx_valid = 0;
    top->pcie_tx_ready = 1;
    top->m_axi_arready = 1;
    top->m_axi_rvalid = 0;
    top->m_axi_rlast = 0;
    top->m_axi_rresp = 0;
    for (int w = 0; w < 8; w++) top->m_axi_rdata[w] = 0;

    printf("TB_FFN_AXI: Starting\n");

    // Reset
    for (int i = 0; i < 5; i++) { top->clk=0; top->eval(); top->clk=1; top->eval(); }
    top->rst_n = 1;

    for (int cyc = 0; cyc < 2000; cyc++) {
        top->clk = 0; top->eval();

        // AXI R response — respond when RREADY is high (independent of arvalid)
        if (top->m_axi_rready) {
            top->m_axi_rvalid = 1;
            for (int w = 0; w < 8; w++) top->m_axi_rdata[w] = 0xA5A5A5A5;
            top->m_axi_rresp = 0;
            top->m_axi_rlast = 1;
        } else {
            top->m_axi_rvalid = 0;
            top->m_axi_rlast = 0;
        }

        top->clk = 1; top->eval();

        if (cyc == 3) printf("cyc=%d arvalid=%08x arready=%d rvalid=%d busy=%d arst=%d\n",
            cyc, top->m_axi_arvalid, top->m_axi_arready, top->m_axi_rvalid,
            top->busy, top->dbg_sub_fsm);
        if (top->perf_token_cnt > 0) {
            printf("PASS: cyc=%d perf=%u busy=%d done=%d\n",
                   cyc, top->perf_token_cnt, top->busy, top->done);
            delete top; return 0;
        }
    }
    printf("WARN: arvalid=%08x busy=%d arst=%d\n",
           top->m_axi_arvalid, top->busy, top->dbg_sub_fsm);
    delete top; return 1;
}
