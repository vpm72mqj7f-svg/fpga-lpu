#!/usr/bin/env bash
#
# build.sh — Build libcpu_prefill.so from cpu_prefill.c
#
# Usage:
#   bash build.sh            # auto-detect backend
#   bash build.sh amx        # force Intel AMX (requires Granite Rapids+)
#   bash build.sh avx512     # force AVX-512 BF16 (requires EPYC Turin / Xeon SP)
#   bash build.sh scalar     # portable scalar fallback (no SIMD required)
#
# Output: build/libcpu_prefill.so
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
SRC="$SCRIPT_DIR/cpu_prefill.c"
OUT="$BUILD_DIR/libcpu_prefill.so"

mkdir -p "$BUILD_DIR"

BACKEND="${1:-auto}"

# ── Backend-specific compile flags ──────────────────────────────────
case "$BACKEND" in
    amx)
        # Intel AMX: Granite Rapids+
        CFLAGS="$CFLAGS -O3 -march=graniterapids -mamx-tile -mamx-int8 -mamx-bf16"
        ;;
    avx512)
        # AVX-512 BF16: EPYC Turin, Xeon SP Gen 4+
        CFLAGS="$CFLAGS -O3 -mavx512bf16 -mavx512f -mavx512vl"
        ;;
    scalar)
        # Portable: no SIMD
        CFLAGS="$CFLAGS -O3"
        ;;
    auto|*)
        # Try to detect and enable available extensions
        CFLAGS="$CFLAGS -O3"
        if gcc -march=native -Q --help=target 2>/dev/null | grep -q 'AMX-TILE.*enabled'; then
            echo "[build.sh] Detected AMX support (Granite Rapids)"
            CFLAGS="$CFLAGS -march=native -mamx-tile -mamx-int8 -mamx-bf16"
        elif gcc -march=native -Q --help=target 2>/dev/null | grep -q 'AVX512BF16.*enabled'; then
            echo "[build.sh] Detected AVX-512 BF16 support"
            CFLAGS="$CFLAGS -march=native -mavx512bf16"
        else
            echo "[build.sh] No AMX/AVX-512 BF16 detected, building scalar fallback"
        fi
        ;;
esac

CFLAGS="$CFLAGS -std=c11 -fPIC -Wall -Wextra"

echo "[build.sh] Compiling cpu_prefill.c → $OUT"
echo "[build.sh] CFLAGS=$CFLAGS"

# Compile to shared library
gcc $CFLAGS -shared -o "$OUT" "$SRC" -lpthread -lm

if [ -f "$OUT" ]; then
    echo "[build.sh] SUCCESS: $OUT built"
    ls -lh "$OUT"
else
    echo "[build.sh] ERROR: compilation failed"
    exit 1
fi
