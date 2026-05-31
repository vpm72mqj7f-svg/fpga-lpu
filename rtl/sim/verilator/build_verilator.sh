#!/bin/bash
# build_verilator.sh — Reusable Verilator build script for FPGA LPU
# Usage: bash build_verilator.sh <top_module> <tb_cpp> <rtl_dir1/rtl_src1> [rtl_src2 ...]
#
# Prerequisites (MSYS2):
#   pacman -S mingw-w64-x86_64-verilator perl make mingw-w64-x86_64-gcc
#
# Key insight: _GLIBCXX_USE_CXX11_ABI=0 avoids string ABI mismatch
# between MSYS2-packaged verilated headers and GCC 16.
set -e

TOP_MODULE=$1
TB_CPP=$2
shift 2
RTL_SOURCES="$@"

if [ -z "$TOP_MODULE" ] || [ -z "$TB_CPP" ] || [ -z "$RTL_SOURCES" ]; then
    echo "Usage: $0 <top_module> <tb_cpp> <rtl_src1> [rtl_src2 ...]" >&2
    exit 1
fi

export PATH=/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH

HERE=$(dirname "$(readlink -f "$0")")
cd "$HERE"

echo "=== Verilator build: $TOP_MODULE ==="

# Step 1: Verilator --cc
rm -rf obj_dir
I_FLAGS="-I../../include"
SRC_ARGS=""
for src in $RTL_SOURCES; do
    SRC_ARGS="$SRC_ARGS $src"
done

verilator --cc $I_FLAGS $SRC_ARGS \
    --top-module $TOP_MODULE --exe $TB_CPP \
    -Wno-WIDTH -Wno-LITENDIAN

cd obj_dir

# Step 2: Generate combined source (bypass verilator_includer MSYS2 path bug)
COMBINED="${TOP_MODULE}__ALL.cpp"
echo "// Combined Verilator sources" > $COMBINED
for f in ${TOP_MODULE}*.cpp; do
    [ "$f" = "$COMBINED" ] && continue
    echo "#include \"$f\"" >> $COMBINED
done

# Step 3: Compile flags
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

# Step 4: Compile
echo "Compiling..."
g++ $FLAGS -c ../$TB_CPP -o tb_main.o
g++ $FLAGS -c $VL_INC/verilated.cpp -o verilated.o
g++ $FLAGS -c $VL_INC/verilated_threads.cpp -o verilated_threads.o
g++ $FLAGS -c $COMBINED -o combined.o

# Step 5: Link
echo "Linking..."
g++ tb_main.o verilated.o verilated_threads.o combined.o \
    -pthread -lpthread -latomic -o $TOP_MODULE

echo "=== Build complete: obj_dir/$TOP_MODULE ==="
