#!/usr/bin/env python3
"""Generate golden vectors for tb_expert_ffn_engine_fp4_down.sv.

Matches the RTL approximation:
  fp4_mac product = (fp4_q12 * fp8_q8 * scale_q8) >> 8
  silu_q12_lut piecewise interpolation
  q12_to_fp8_e4m3 threshold encoder
"""

import os

HIDDEN = 8
INTER = 4
LANES = 4
K_BEATS = 2

FP4_LUT = [0, 4, 8, 12, 16, 24, 32, 48]


def fp4_q12(code):
    mag = code & 0x7
    sign = (code >> 3) & 1
    v = FP4_LUT[mag]
    return -v if sign and mag != 0 else v


def fp8_q8(code):
    sign = (code >> 7) & 1
    exp = (code >> 3) & 0xf
    mant = code & 0x7
    if exp == 0:
        mag = mant // 2
    elif exp < 2:
        mag = (8 + mant) >> 1
    else:
        mag = (8 + mant) << (exp - 2)
        mag = min(mag, 2047)
    return -mag if sign and mag else mag


def mac_product(w, a, s=0x38):
    return (fp4_q12(w) * fp8_q8(a) * fp8_q8(s)) >> 8


def linear(weights_rows, activ_beats, scales=(0x38, 0x38)):
    out = []
    for row in weights_rows:
        acc = 0
        for beat_idx, beat in enumerate(row):
            for lane in range(LANES):
                w = beat[lane]
                a = activ_beats[beat_idx][lane]
                s = scales[beat_idx]
                acc += mac_product(w, a, s)
        out.append(acc & 0xffffffff)
    return out


def silu_q12(x):
    # x is already a signed Python int (callers convert 32-bit unsigned first)
    knots = [
        (-32768, -11), (-16384, -295), (-8192, -976), (-4096, -1102),
        (0, 0), (4096, 2994), (8192, 7215), (16384, 16089), (32768, 32768),
    ]
    if x <= knots[0][0]:
        y = knots[0][1]
    elif x >= knots[-1][0]:
        y = x
    else:
        y = 0
        for (x0, y0), (x1, y1) in zip(knots[:-1], knots[1:]):
            if x0 <= x < x1:
                y = y0 + ((x - x0) * (y1 - y0)) // (x1 - x0)
                break
    return y


def q12_to_fp8(x):
    sign = x < 0
    ax = -x if sign else x
    if ax < 512: code = 0x00
    elif ax < 1536: code = 0x28
    elif ax < 2560: code = 0x30
    elif ax < 3584: code = 0x34
    elif ax < 5120: code = 0x38
    elif ax < 7168: code = 0x3c
    elif ax < 10240: code = 0x40
    elif ax < 14336: code = 0x44
    elif ax < 20480: code = 0x48
    elif ax < 28672: code = 0x4c
    else: code = 0x50
    return code | (0x80 if sign and code else 0)


def ffn_case(activ_beats, gate_weights, up_weights, down_weights, scales=(0x38, 0x38)):
    gate = linear(gate_weights, activ_beats, scales)
    up = linear(up_weights, activ_beats, scales)
    mid_fp8 = []
    mid_q12 = []
    for g, u in zip(gate, up):
        if g & 0x80000000: g_s = g - (1 << 32)
        else: g_s = g
        if u & 0x80000000: u_s = u - (1 << 32)
        else: u_s = u
        m = (silu_q12(g_s) * u_s) >> 12
        mid_q12.append(m)
        mid_fp8.append(q12_to_fp8(m))
    # Down uses one beat of 4 intermediate values.
    down_out = linear([[row] for row in down_weights], [mid_fp8], (0x38,))
    return gate, up, mid_q12, mid_fp8, down_out


def pack_values(vals, width):
    # element 0 in low bits -> emit reversed concatenation
    return "{" + ", ".join(f"{width}'h{v & ((1<<width)-1):0{(width+3)//4}x}" for v in reversed(vals)) + "}"


def flatten_beats(beats):
    vals = []
    for beat in beats:
        vals.extend(beat)
    return vals


def flatten_weight_rows(rows):
    vals = []
    for row in rows:
        vals.extend(flatten_beats(row))
    return vals


def case_defs(prefix, activ, gate_w, up_w, down_w, expected):
    lines = []
    lines.append(f"    localparam logic [{K_BEATS*LANES*8}-1:0] {prefix}_ACT_PACK = {pack_values(flatten_beats(activ), 8)};")
    lines.append(f"    localparam logic [{INTER*K_BEATS*LANES*4}-1:0] {prefix}_GATE_W_PACK = {pack_values(flatten_weight_rows(gate_w), 4)};")
    lines.append(f"    localparam logic [{INTER*K_BEATS*LANES*4}-1:0] {prefix}_UP_W_PACK = {pack_values(flatten_weight_rows(up_w), 4)};")
    lines.append(f"    localparam logic [{HIDDEN*LANES*4}-1:0] {prefix}_DOWN_W_PACK = {pack_values(flatten_beats(down_w), 4)};")
    lines.append(f"    localparam logic [{HIDDEN*32}-1:0] {prefix}_EXPECTED_PACK = {pack_values(expected, 32)};")
    return lines


def main():
    # Case 0: positive identity (expected first 4 outputs = 0x0c00)
    act0 = [[0x38]*4, [0x38]*4]
    gate0 = [[[0x1]*4, [0x0]*4] for _ in range(INTER)]
    up0 = [[[0x1]*4, [0x0]*4] for _ in range(INTER)]
    down0 = []
    for r in range(HIDDEN):
        row = [0x0]*4
        if r < INTER:
            row[r] = 0x4
        down0.append(row)
    _, _, _, _, exp0 = ffn_case(act0, gate0, up0, down0)

    # Case 1: mixed signs with genuinely different rows
    # activ beat0: [+1, -1, 0, 0], beat1: all zero
    act1 = [[0x38, 0xB8, 0x00, 0x00], [0x00]*4]
    # gate: row0→+1, row1→-1, row2→-1, row3→+1
    gate1 = [
        [[0x4, 0x0, 0x0, 0x0],  # row0 beat0
         [0x0]*4],                 # row0 beat1
        [[0xC, 0x0, 0x0, 0x0],  # row1: -1 × +1 = -1
         [0x0]*4],
        [[0x0, 0x4, 0x0, 0x0],  # row2: +1 × -1 = -1
         [0x0]*4],
        [[0x0, 0xC, 0x0, 0x0],  # row3: -1 × -1 = +1
         [0x0]*4],
    ]
    # up: row0→+1, row1→+1, row2→-1, row3→-1
    up1 = [
        [[0x4, 0x0, 0x0, 0x0], [0x0]*4],  # row0: +1
        [[0x4, 0x0, 0x0, 0x0], [0x0]*4],  # row1: +1
        [[0x0, 0xC, 0x0, 0x0], [0x0]*4],  # row2: -1 × -1 = +1? wait negative
        [[0x0, 0x4, 0x0, 0x0], [0x0]*4],  # row3: +1 × -1 = -1
    ]
    # down identity rows 0-3
    down1 = down0
    _, _, mid1_q12, mid1_fp8, exp1 = ffn_case(act1, gate1, up1, down1)

    lines = []
    lines.append("// AUTO-GENERATED by scripts/simulation/gen_ffn_tb_vectors.py")
    lines.append("package tb_ffn_golden_pkg;")
    lines.append(f"    localparam int HIDDEN = {HIDDEN};")
    lines.append(f"    localparam int INTER = {INTER};")
    lines.append(f"    localparam int LANES = {LANES};")
    lines.append(f"    localparam int K_BEATS = {K_BEATS};")
    lines.append("")
    lines.extend(case_defs("C0", act0, gate0, up0, down0, exp0))
    lines.append("")
    lines.extend(case_defs("C1", act1, gate1, up1, down1, exp1))
    lines.append("endpackage")

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "rtl", "sim", "tb_ffn_golden_pkg.sv"))
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print(f"Case0 expected: {[hex(x) for x in exp0]}")
    print(f"Case1 expected: {[hex(x) for x in exp1]}")


if __name__ == "__main__":
    main()
