# =============================================================================
# program_board.tcl — JTAG Programming Script
#
# Usage:
#   quartus_pgm -t scripts/program_board.tcl
#
# Programs the FPGA via on-board USB-Blaster II (default JTAG chain).
# =============================================================================

set sof_file "output_files/bringup.sof"

# -----------------------------------------------------------------------------
# Detect JTAG cable
# -----------------------------------------------------------------------------
puts "Detecting JTAG cables..."
set cables [get_hardware_names]
if {[llength $cables] == 0} {
    puts "ERROR: No hardware detected!"
    puts "  - Check USB cable connection to J15 (Micro-USB)"
    puts "  - Check board power (blue LED D4 should be ON)"
    puts "  - Install Intel FPGA Download Cable II driver"
    qexit -error
}

puts "Found hardware: $cables"
set cable [lindex $cables 0]

# -----------------------------------------------------------------------------
# Open Programmer
# -----------------------------------------------------------------------------
puts "Opening programmer..."
programmer_open

# -----------------------------------------------------------------------------
# Detect JTAG chain
# -----------------------------------------------------------------------------
puts "Detecting JTAG chain devices..."
if {[catch {
    device_detect
} result]} {
    puts "ERROR: JTAG chain detection failed!"
    puts $result
    programmer_close
    qexit -error
}

# List devices in chain
puts "\nJTAG Chain:"
foreach device [get_device_names -hardware_name $cable] {
    puts "  $device"
}

# -----------------------------------------------------------------------------
# Program the Stratix 10 MX FPGA
# -----------------------------------------------------------------------------
if {![file exists $sof_file]} begin
    puts "ERROR: SOF file not found: $sof_file"
    puts "Run build_quartus.tcl first."
    programmer_close
    qexit -error
end

puts "\nProgramming FPGA with: $sof_file"

if {[catch {
    # The Stratix 10 device is typically the last device in the JTAG chain
    # (after MAX10 System Controller, MAX10 Power Manager)
    set devices [get_device_names -hardware_name $cable]
    set s10_index [expr {[llength $devices] - 1}]
    set s10_device [lindex $devices $s10_index]

    puts "Target device: $s10_device"

    # Configure
    device_program -hardware_name $cable \
                   -device_name $s10_device \
                   -programming_file $sof_file
} result]} {
    puts "ERROR: Programming failed!"
    puts $result
    # Try reducing JTAG clock frequency
    puts "\nTrying with slower JTAG clock (16 MHz)..."
    catch { jtagconfig --setparam 1 JtagClock 16M }
    puts "Retry: quartus_pgm -t scripts/program_board.tcl"
    programmer_close
    qexit -error
}

puts "\n=============================================="
puts " Programming Complete!"
puts "=============================================="
puts "Check:"
puts "  - CONFIG_DONE LED (D14) should be ON"
puts "  - CvP_DONE LED (D16) if using CvP"
puts "  - User LEDs should show bring-up status"
puts "    LED0: PLL heartbeat (blink)"
puts "    LED1: FFN busy"
puts "    LED2: FFN done pulse"
puts "    LED3: Pass/Fail"

programmer_close
puts "Done."
