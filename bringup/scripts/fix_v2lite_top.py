#!/usr/bin/env python3
"""Replace altera_iopll + stratix10_reset_release with simple bypass in v2_lite_top.sv"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "v2_lite_top.sv"
with open(path) as f:
    lines = f.readlines()

# Find and replace lines ~75-101 (IOPLL + Reset Release)
new_block = """    // PLL bypass for synthesis (replace with altera_iopll IP in production)
    assign clk_500m = clk_100m; assign clk_250m = clk_100m;
    logic [7:0] pll_cnt = 0;
    always_ff @(posedge clk_100m) if (pll_cnt < 8'd255) pll_cnt <= pll_cnt + 1;
    assign pll_locked = (pll_cnt == 8'd255);
    assign reset_release = 1'b1;
"""
# Find the start (altera_iopll comment) and end (endmodule or next section)
start = end = -1
for i, line in enumerate(lines):
    if "Intel Stratix 10 I/O PLL" in line or "altera_iopll #" in line:
        start = i - 1  # include the comment line
    if start > 0 and "ninit_done (reset_release)" in line:
        end = i + 1  # include this line
        break

if start > 0 and end > start:
    lines[start:end+1] = [new_block]
    with open(path, 'w') as f:
        f.writelines(lines)
    print(f"Fixed {path}: lines {start+1}-{end+1} replaced")
else:
    print(f"Could not find IOPLL block (start={start}, end={end})")
    for i, line in enumerate(lines):
        if "iopll" in line.lower() or "reset_release" in line.lower():
            print(f"  {i+1}: {line.rstrip()}")
