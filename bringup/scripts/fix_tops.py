#!/usr/bin/env python3
"""Fix heredoc-mangled Verilog syntax in v2_lite_top and v4_flash_top"""
import sys, re, glob

for proj in ['v2_lite', 'v4_flash']:
    path = f'/home/ic-server31/bringup/{proj}/rtl/{proj}_top.sv'
    try:
        with open(path) as f:
            txt = f.read()
        # Fix: 1b0 -> 1'b0, 1b1 -> 1'b1
        txt = txt.replace("1b0", "1'b0")
        txt = txt.replace("1b1", "1'b1")
        txt = txt.replace("0b0", "1'b0")  # in case of 0b0
        # Fix: 8"(i) or 8\"(i) -> 8'(i)
        txt = re.sub(r'8\\?"\(', "8'(", txt)
        txt = re.sub(r'8\\?"\)', "8')", txt)
        # Fix: logic[$clog2 -> logic [$clog2 (heredoc eats backslash)
        txt = txt.replace('logic[$clog2', 'logic [$clog2')
        txt = txt.replace('logic[3:0]', 'logic [3:0]')
        with open(path, 'w') as f:
            f.write(txt)
        print(f"{proj}: fixed")
    except Exception as e:
        print(f"{proj}: {e}")
