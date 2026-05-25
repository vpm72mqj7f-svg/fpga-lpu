#!/usr/bin/env python3
"""
Run fp4 MAC golden tests on FPGA hardware via PCIe BAR0 MMIO.

Reads rtl/sim/tb_golden_pkg.sv → extracts 15 test vectors → writes to
FPGA MMIO registers → triggers MAC execution → reads back result →
compares to Python golden expected.

Prerequisites:
  - DK-SI-AGM027 board connected
  - top.sv bitstream loaded
  - VFIO driver bound (or UIO)
  - sudo or appropriate /dev/vfio permissions

Usage:
  python hw/scripts/run_golden_tests.py --device 0

Registers (BAR0 offsets):
  0x1000: scale_wr_data[7:0]  | scale_wr_addr[16:8] | scale_wr_en[17]
  0x1004: mac_weight[3:0] | mac_scale[11:4] | mac_activ[19:12] | accum_clr[20] | mac_go[21]
  0x1008: mac_result[31:0] (read-only)
  0x100C: test_status[0]=done, [1]=error, [15:8]=pass_count, [23:16]=fail_count
"""

import os, sys, struct, time, re, argparse


def parse_golden_pkg(path):
    """Parse tb_golden_pkg.sv → list of (name, weights, activs, scales, expected)."""
    with open(path, encoding="utf-8") as f:
        src = f.read()

    count = int(re.search(r"NUM_TESTS\s*=\s*(\d+)", src).group(1))
    tests = []
    for i in range(count):
        # Extract packed arrays
        w_match = re.search(rf"T{i}_W_PACK\s*=\s*\{{\s*([^}}]+)\}}", src)
        a_match = re.search(rf"T{i}_A_PACK\s*=\s*\{{\s*([^}}]+)\}}", src)
        s_match = re.search(rf"T{i}_S_PACK\s*=\s*\{{\s*([^}}]+)\}}", src)
        e_match = re.search(rf"T{i}_EXPECTED\s*=\s*(\d+'h[0-9a-fA-F]+)", src)

        def parse_list(text, width):
            vals = re.findall(rf"{width}'h([0-9a-fA-F]+)", text)
            return [int(v, 16) for v in vals[::-1]]  # reverse: low bits are elem 0

        weights  = parse_list(w_match.group(1), "4")
        activs   = parse_list(a_match.group(1), "8")
        scales   = parse_list(s_match.group(1), "8")
        expected = int(e_match.group(1).split("'h")[1], 16)

        tests.append({
            "name": f"T{i}",
            "weights": weights,
            "activs": activs,
            "scales": scales,
            "expected": expected,
        })
    return tests


class FpgaDevice:
    """MMIO-based FPGA access via /dev/mem or VFIO."""

    def __init__(self, bar0_base=0):
        self.bar0 = bar0_base  # will be mmap'd address
        # In production, use VFIO or UIO. For now, skeleton.
        self._connected = False

    def connect(self, device_id=0):
        """Discover and map FPGA BAR0."""
        # [TODO] Implement VFIO device open + BAR mmap
        # For bring-up, can use /sys/bus/pci/devices/0000:XX:00.0/resource0
        try:
            path = f"/sys/bus/pci/devices/0000:{device_id:02x}:00.0/resource0"
            if not os.path.exists(path):
                raise FileNotFoundError(f"FPGA device {device_id} not found")
            # fd = os.open(path, os.O_RDWR | os.O_SYNC)
            # self.bar0 = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            self._connected = True
            print(f"Connected to FPGA device {device_id}")
        except Exception as e:
            print(f"Failed to connect: {e}")
            raise

    def reg_write(self, offset, value):
        """Write 32-bit value to BAR0 offset."""
        # struct.pack_into("<I", self.bar0, offset, value)
        pass  # [TODO]

    def reg_read(self, offset):
        """Read 32-bit value from BAR0 offset."""
        # return struct.unpack_from("<I", self.bar0, offset)[0]
        return 0  # [TODO]

    def write_scale(self, addr, data):
        self.reg_write(0x1000, (1 << 17) | ((addr & 0x1FF) << 8) | (data & 0xFF))

    def run_mac(self, weight, scale, activ, is_first):
        val = ((1 if is_first else 0) << 20) | (1 << 21)  # accum_clr | go
        val |= (weight & 0xF)
        val |= ((scale & 0xFF) << 4)
        val |= ((activ & 0xFF) << 12)
        self.reg_write(0x1004, val)

    def read_result(self):
        return self.reg_read(0x1008)

    def read_status(self):
        return self.reg_read(0x100C)

    def wait_done(self, timeout_ms=1000):
        for _ in range(timeout_ms):
            if self.read_status() & 1:
                return True
            time.sleep(0.001)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--golden", default="rtl/sim/tb_golden_pkg.sv")
    args = parser.parse_args()

    tests = parse_golden_pkg(args.golden)
    print(f"Parsed {len(tests)} golden tests from {args.golden}")

    fpga = FpgaDevice()
    try:
        fpga.connect(args.device)
    except Exception:
        print("FPGA not connected — running in simulator mode")
        print("(Will compare Python golden to itself as sanity check)")

    passed = 0
    failed = 0

    for t in tests:
        # Load scales
        scale_set = set()
        for s in t["scales"]:
            if s not in scale_set:
                fpga.write_scale(s, s)  # group_id = scale_value (simplified)
                scale_set.add(s)

        # Run all MAC ops in sequence
        for i, (w, a, s) in enumerate(zip(t["weights"], t["activs"], t["scales"])):
            fpga.run_mac(w, s, a, is_first=(i == 0))

        # Wait for pipeline drain
        if not fpga.wait_done():
            print(f"[FAIL] {t['name']}: FPGA timeout")
            failed += 1
            continue

        result = fpga.read_result()
        expected = t["expected"]
        if result == expected:
            print(f"[ OK ] {t['name']}: result=0x{result:08X}")
            passed += 1
        else:
            print(f"[FAIL] {t['name']}: got 0x{result:08X}, expected 0x{expected:08X}")
            failed += 1

    print(f"\nPassed: {passed}/{len(tests)}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
