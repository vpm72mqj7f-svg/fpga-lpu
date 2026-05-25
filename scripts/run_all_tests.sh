#!/bin/bash
# run_all_tests.sh — comprehensive RTL test suite for fpgalpu
#
# Compiles and runs all Icarus testbenches, reports PASS/FAIL summary.
# Usage: bash scripts/run_all_tests.sh

set -e

IVL=/c/iverilog/bin/iverilog
VVP=/c/iverilog/bin/vvp
ROOT=$(dirname "$0")/..
BUILD=$ROOT/rtl/sim/build
INC="-I $ROOT/rtl/include"

mkdir -p $BUILD

PASS=0
FAIL=0
TOTAL=0

run_test() {
    local name=$1
    local vvp=$BUILD/${name}.vvp
    shift
    TOTAL=$((TOTAL + 1))
    echo -n "[$TOTAL] $name ... "
    local err_out
    err_out=$($IVL -g2012 $INC -o "$vvp" "$@" 2>&1)
    if [ $? -ne 0 ]; then
        echo "COMPILE ERROR"
        FAIL=$((FAIL + 1))
        return 1
    fi
    local sim_out
    sim_out=$($VVP "$vvp" 2>&1)
    if echo "$sim_out" | grep -q "PASS"; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== fpgalpu RTL Test Suite ==="
echo ""

# DSP
# tb_fp4_mac: requires tb_golden_pkg.sv (missing) — skip
run_test tb_fp4_scale_reader $ROOT/rtl/dsp/fp4_scale_reader.sv $ROOT/rtl/sim/tb_fp4_scale_reader.sv
run_test tb_fp4_systolic_tile $ROOT/rtl/dsp/fp4_mac.sv $ROOT/rtl/dsp/fp4_systolic_tile.sv $ROOT/rtl/sim/tb_fp4_systolic_tile.sv
run_test tb_fp4_scaled_tile  $ROOT/rtl/dsp/fp4_mac.sv $ROOT/rtl/dsp/fp4_systolic_tile.sv $ROOT/rtl/dsp/fp4_scale_reader.sv $ROOT/rtl/dsp/fp4_scaled_tile.sv $ROOT/rtl/sim/tb_fp4_scaled_tile.sv
run_test tb_fp4_systolic_array $ROOT/rtl/dsp/fp4_mac.sv $ROOT/rtl/dsp/fp4_systolic_tile.sv $ROOT/rtl/dsp/fp4_scale_reader.sv $ROOT/rtl/dsp/fp4_scaled_tile.sv $ROOT/rtl/dsp/fp4_systolic_array.sv $ROOT/rtl/sim/tb_fp4_systolic_array.sv
run_test tb_fp4_linear_engine $ROOT/rtl/dsp/fp4_mac.sv $ROOT/rtl/dsp/fp4_systolic_tile.sv $ROOT/rtl/dsp/fp4_scale_reader.sv $ROOT/rtl/dsp/fp4_scaled_tile.sv $ROOT/rtl/dsp/fp4_systolic_array.sv $ROOT/rtl/dsp/fp4_linear_engine.sv $ROOT/rtl/sim/tb_fp4_linear_engine.sv

# Activation
run_test tb_silu_q12_lut $ROOT/rtl/activation/silu_q12_lut.sv $ROOT/rtl/sim/tb_silu_q12_lut.sv
run_test tb_rms_norm $ROOT/rtl/activation/rms_norm.sv $ROOT/rtl/sim/tb_rms_norm.sv

# MoE
run_test tb_router_topk $ROOT/rtl/moe/router_topk.sv $ROOT/rtl/sim/tb_router_topk.sv
# tb_expert_ffn_engine: references old expert_ffn_engine module (skip)
run_test tb_expert_ffn_engine_fp4_down $ROOT/rtl/dsp/fp4_mac.sv $ROOT/rtl/dsp/fp4_systolic_tile.sv $ROOT/rtl/dsp/fp4_scale_reader.sv $ROOT/rtl/dsp/fp4_scaled_tile.sv $ROOT/rtl/dsp/fp4_systolic_array.sv $ROOT/rtl/dsp/fp4_linear_engine.sv $ROOT/rtl/activation/silu_q12_lut.sv $ROOT/rtl/activation/q12_to_fp8_e4m3.sv $ROOT/rtl/moe/expert_ffn_engine_fp4_down.sv $ROOT/rtl/sim/tb_expert_ffn_engine_fp4_down.sv

# Attention
run_test tb_mla_attention $ROOT/rtl/attention/mla_attention.sv $ROOT/rtl/sim/tb_mla_attention.sv
run_test tb_mla_qkv $ROOT/rtl/attention/mla_qkv_proj.sv $ROOT/rtl/attention/mla_rope.sv $ROOT/rtl/attention/mla_kv_cache.sv $ROOT/rtl/sim/tb_mla_qkv.sv
run_test tb_mla_attention_v2 $ROOT/rtl/attention/mla_qkv_proj.sv $ROOT/rtl/attention/mla_rope.sv $ROOT/rtl/attention/mla_kv_cache.sv $ROOT/rtl/attention/mla_attention_v2.sv $ROOT/rtl/sim/tb_mla_attention_v2.sv

# Engram
run_test tb_lookup_engine $ROOT/rtl/engram/hash_unit.sv $ROOT/rtl/engram/sram_cache.sv $ROOT/rtl/engram/lookup_engine.sv $ROOT/rtl/sim/tb_lookup_engine.sv

# Layer
run_test tb_mhc_mixer $ROOT/rtl/layer/mhc_mixer.sv $ROOT/rtl/sim/tb_mhc_mixer.sv
run_test tb_full_transformer_layer $ROOT/rtl/dsp/fp4_mac.sv $ROOT/rtl/dsp/fp4_systolic_tile.sv $ROOT/rtl/dsp/fp4_scale_reader.sv $ROOT/rtl/dsp/fp4_scaled_tile.sv $ROOT/rtl/dsp/fp4_systolic_array.sv $ROOT/rtl/dsp/fp4_linear_engine.sv $ROOT/rtl/activation/silu_q12_lut.sv $ROOT/rtl/activation/q12_to_fp8_e4m3.sv $ROOT/rtl/activation/rms_norm.sv $ROOT/rtl/moe/router_topk.sv $ROOT/rtl/moe/expert_ffn_engine_fp4_down.sv $ROOT/rtl/attention/mla_attention.sv $ROOT/rtl/layer/full_transformer_layer.sv $ROOT/rtl/sim/tb_full_transformer_layer.sv

# Head
run_test tb_mtp_head $ROOT/rtl/head/mtp_head.sv $ROOT/rtl/sim/tb_mtp_head.sv

# Chip
run_test tb_c2c_ring $ROOT/rtl/chip/c2c_node.sv $ROOT/rtl/sim/tb_c2c_ring.sv
run_test tb_kv_dma $ROOT/rtl/chip/kv_dma_engine.sv $ROOT/rtl/sim/tb_kv_dma.sv

echo ""
echo "=== Results: $PASS PASS, $FAIL FAIL, $TOTAL TOTAL ==="

if [ $FAIL -eq 0 ]; then
    echo "ALL TESTS PASS"
    exit 0
else
    echo "SOME TESTS FAILED"
    exit 1
fi
