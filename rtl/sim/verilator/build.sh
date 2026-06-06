#!/bin/bash
export PATH=/mingw64/bin:/usr/bin:$PATH
cd /d/workspace/fpgalpu/rtl/sim/verilator
rm -rf obj_dir

echo "=== Step 1: Verilator --cc ==="
verilator --cc -I../../include ../../activation/rms_norm.sv \
    --top-module rms_norm --exe tb_rms_norm.cpp \
    -Wno-WIDTH -Wno-LITENDIAN

if [ $? -ne 0 ]; then echo "Verilator failed"; exit 1; fi

cd obj_dir

echo "=== Step 2: Generate combined source ==="
echo "// Combined Verilator sources" > Vrms_norm__ALL.cpp
for f in Vrms_norm.cpp Vrms_norm__Syms__Slow.cpp \
         Vrms_norm___024root__0.cpp Vrms_norm___024root__Slow.cpp \
         Vrms_norm___024root__0__Slow.cpp; do
    echo "#include \"$f\"" >> Vrms_norm__ALL.cpp
done

VL_INC=/mingw64/share/verilator/include
VL_VLT=$VL_INC/vltstd

CXXFLAGS="-Os -faligned-new -fcf-protection=none"
WARN="-Wno-bool-operation -Wno-int-in-bool-context -Wno-shadow \
      -Wno-sign-compare -Wno-subobject-linkage -Wno-tautological-compare \
      -Wno-uninitialized -Wno-unused-but-set-parameter \
      -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-variable"
DEFS="-DVERILATOR=1 -DVM_COVERAGE=0 -DVM_SC=0 -DVM_TIMING=0 \
      -DVM_TRACE=0 -DVM_TRACE_FST=0 -DVM_TRACE_VCD=0 -DVM_TRACE_SAIF=0"

FLAGS="$CXXFLAGS $WARN $DEFS -I. -I$VL_INC -I$VL_VLT"

echo "=== Step 3: Compile ==="
set -x
g++ $FLAGS -c ../tb_rms_norm.cpp -o tb_main.o || exit 1
g++ $FLAGS -c $VL_INC/verilated.cpp -o verilated.o || exit 1
g++ $FLAGS -c $VL_INC/verilated_threads.cpp -o verilated_threads.o || exit 1
g++ $FLAGS -c Vrms_norm__ALL.cpp -o combined.o || exit 1

echo "=== Step 4: Link ==="
g++ tb_main.o verilated.o verilated_threads.o combined.o \
    -pthread -lpthread -latomic -o Vrms_norm || exit 1

echo "=== Build complete: obj_dir/Vrms_norm ==="
ls -la Vrms_norm
