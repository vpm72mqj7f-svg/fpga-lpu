#!/bin/bash
# Pre-build pin-port consistency check
# Usage: bash check_pins.sh
# Verifies all top-level ports have QSF pin assignments, and vice versa.
# MUST PASS before any quartus build.
BDIR="/home/ic-server31/bringup/v2_lite_full"

cd "$BDIR" 2>/dev/null || { echo "ERROR: $BDIR not found"; exit 1; }

# Extract ports from top-level module
grep -E '^\s*(input|output|inout)' v2_lite_full_top.sv | \
    tr ',' '\n' | tr ';' '\n' | \
    sed 's/^\s*//; s/\s*$//' | \
    grep -v '^\s*$' | grep -v '^input\|^output\|^inout\|^wire\|^reg\|^\[' | \
    sed 's/\[.*//' | sort -u > /tmp/_ports.txt

# Extract pin assignments from QSF
grep 'set_location_assignment' v2_lite_full.qsf | \
    grep -oP '\-to\s+"?([^"\s]+)"?' | sed 's/-to //; s/"//g' | \
    sed 's/\[.*//' | sort -u > /tmp/_pins.txt

echo "Ports: $(wc -l < /tmp/_ports.txt) | Pins: $(wc -l < /tmp/_pins.txt)"

UNPINNED=$(comm -23 /tmp/_ports.txt /tmp/_pins.txt | grep -v '^m2u_bridge\|^tg[0-7]\|^ffn_')
UNPORTED=$(comm -13 /tmp/_ports.txt /tmp/_pins.txt)

if [ -n "$UNPINNED" ]; then
    echo "PORTS WITHOUT PIN ASSIGNMENTS:"
    echo "$UNPINNED"
    exit 1
fi

if [ -n "$UNPORTED" ]; then
    echo "PINS WITHOUT PORT CONNECTIONS:"
    echo "$UNPORTED"
    exit 1
fi

echo "Pin-port check PASSED"
exit 0
