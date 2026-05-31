#!/bin/bash
# build_prod_modules.sh — Build production-scale modules for V3.4 verification
# Usage: bash build_prod_modules.sh <module>  where module = rms_norm | router_topk | expert_ffn
set -e

MODULE=$1
if [ -z "$MODULE" ]; then
    echo "Usage: $0 <rms_norm|router_topk|expert_ffn>" >&2
    exit 1
fi

export PATH=/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH
HERE=$(dirname "$(readlink -f "$0")")
cd "$HERE"

I_FLAGS="-I../../include -I../../sim"
DEFINES="-DFPGA_LPU_PRODUCTION"
VL_WARN="-Wno-WIDTH -Wno-LITENDIAN"
VL_EXTRA="--replication-limit 32768"

rm -rf "obj_dir_${MODULE}"
mkdir -p "obj_dir_${MODULE}"

case "$MODULE" in
    rms_norm)
        echo "=== Building rms_norm @ HIDDEN=7168 ==="
        TOP="rms_norm"
        TB="tb_rms_norm.cpp"
        RTL="../../activation/rms_norm.sv"
        GFLAGS="-GHIDDEN=7168"
        ;;
    router_topk)
        echo "=== Building router_topk @ EXPERTS=384 HIDDEN=7168 ==="
        TOP="router_topk"
        TB="tb_router_topk_prod.cpp"
        RTL="../../moe/router_topk.sv ../../sim/altera_mult_add.sv"
        GFLAGS="-GEXPERTS=384 -GHIDDEN=7168"
        ;;
    expert_ffn)
        echo "=== Building expert_ffn_engine_fp4_down @ HIDDEN=7168 INTER=3072 ==="
        TOP="expert_ffn_engine_fp4_down"
        TB="tb_expert_ffn_prod.cpp"
        RTL="../../moe/expert_ffn_engine_fp4_down.sv \
             ../../activation/silu_q12_lut.sv \
             ../../activation/q12_to_fp8_e4m3.sv \
             ../../dsp/fp4_linear_engine.sv \
             ../../dsp/fp4_systolic_array.sv \
             ../../dsp/fp4_scaled_tile.sv \
             ../../dsp/fp4_scale_reader.sv \
             ../../dsp/fp4_systolic_tile.sv \
             ../../dsp/fp4_mac.sv \
             ../../sim/altera_mult_add.sv \
             ../../sim/altera_syncram.sv"
        GFLAGS="-GHIDDEN=7168 -GINTER=3072"
        VL_EXTRA="$VL_EXTRA -I../../include"
        ;;
    *)
        echo "Unknown module: $MODULE" >&2; exit 1
        ;;
esac

echo "  Verilator --cc ..."
verilator --cc $I_FLAGS $DEFINES $VL_WARN $VL_EXTRA \
    $RTL $GFLAGS \
    --top-module $TOP --exe $TB \
    -Mdir "obj_dir_${MODULE}"

cd "obj_dir_${MODULE}"

# Generate combined source (bypass verilator_includer MSYS2 path bug)
COMBINED="V${TOP}__ALL.cpp"
echo "// Combined Verilator sources" > $COMBINED
for f in V${TOP}*.cpp; do
    [ "$f" = "$COMBINED" ] && continue
    echo "#include \"$f\"" >> $COMBINED
done

# Compile flags
VL_INC=/c/msys64/mingw64/share/verilator/include
VL_VLT=$VL_INC/vltstd
CXXFLAGS="-Os -faligned-new -fcf-protection=none"
WARNFLAGS="-Wno-bool-operation -Wno-int-in-bool-context -Wno-shadow \
           -Wno-sign-compare -Wno-subobject-linkage -Wno-tautological-compare \
           -Wno-uninitialized -Wno-unused-but-set-parameter \
           -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-variable"
DEFS="-DVERILATOR=1 -DVM_COVERAGE=0 -DVM_SC=0 -DVM_TIMING=0 \
      -DVM_TRACE=0 -DVM_TRACE_FST=0 -DVM_TRACE_VCD=0 -DVM_TRACE_SAIF=0 \
      -D_GLIBCXX_USE_CXX11_ABI=0"
FLAGS="$CXXFLAGS $WARNFLAGS $DEFS -I. -I$VL_INC -I$VL_VLT"

echo "  Compiling..."
g++ $FLAGS -c ../$TB -o tb_main.o
g++ $FLAGS -c $VL_INC/verilated.cpp -o verilated.o
g++ $FLAGS -c $VL_INC/verilated_threads.cpp -o verilated_threads.o
g++ $FLAGS -c $COMBINED -o combined.o

echo "  Linking..."
g++ tb_main.o verilated.o verilated_threads.o combined.o \
    -pthread -lpthread -latomic -o $TOP

echo "=== Build complete: obj_dir_${MODULE}/$TOP ==="
echo "  Running test..."
./$TOP
echo "=== Done ==="
