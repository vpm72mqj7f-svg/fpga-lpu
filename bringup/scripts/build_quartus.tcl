# =============================================================================
# build_quartus.tcl — Quartus Prime Pro Compilation Script
#
# Usage:
#   quartus_sh -t scripts/build_quartus.tcl
#
# Performs: Synthesis → Fitter → Assembler → Timing Analysis
# Output:   output_files/bringup.sof
# =============================================================================

# Project settings
set project_name   "bringup"
set project_dir    [file dirname [info script]]/..
set top_module     "bringup_top"
set device_family  "Stratix 10"
set device_part    "1SM21BHU2F53E1VG"

# Change to project directory
cd $project_dir

# -----------------------------------------------------------------------------
# Open/Create Project
# -----------------------------------------------------------------------------
if {![project_exists $project_name]} {
    puts "Creating new Quartus project: $project_name"
    project_new -overwrite $project_name
} else {
    puts "Opening existing project: $project_name"
    project_open $project_name
}

# -----------------------------------------------------------------------------
# Run Synthesis
# -----------------------------------------------------------------------------
puts "\n=============================================="
puts " Stage 1/3: Analysis & Synthesis"
puts "=============================================="

if {[catch {
    execute_flow -analysis_and_synthesis
} result]} {
    puts "ERROR: Synthesis failed!"
    puts $result
    qexit -error
}

# -----------------------------------------------------------------------------
# Run Fitter (Place & Route)
# -----------------------------------------------------------------------------
puts "\n=============================================="
puts " Stage 2/3: Fitter (Place & Route)"
puts "=============================================="

if {[catch {
    execute_flow -fit
} result]} {
    puts "ERROR: Fitter failed!"
    puts $result
    qexit -error
}

# -----------------------------------------------------------------------------
# Run Assembler (Generate .sof)
# -----------------------------------------------------------------------------
puts "\n=============================================="
puts " Stage 3/3: Assembler (.sof generation)"
puts "=============================================="

if {[catch {
    execute_flow -assembly
} result]} {
    puts "ERROR: Assembler failed!"
    puts $result
    qexit -error
}

# -----------------------------------------------------------------------------
# Timing Analysis
# -----------------------------------------------------------------------------
puts "\n=============================================="
puts " Timing Analysis"
puts "=============================================="

if {[catch {
    execute_flow -sta
} result]} {
    puts "WARNING: Timing analysis failed (may be OK for bring-up)"
    puts $result
}

# -----------------------------------------------------------------------------
# Report Summary
# -----------------------------------------------------------------------------
puts "\n=============================================="
puts " Build Complete!"
puts "=============================================="

# Load reports
set fitter_summary [quartus_fit --read_summary -c $project_name 2>/dev/null]
if {[llength $fitter_summary] > 0} {
    puts "Fitter Summary:"
    puts "  ALMs used:      [lindex $fitter_summary 5]"
    puts "  Registers:      [lindex $fitter_summary 7]"
    puts "  M20K blocks:    [lindex $fitter_summary 9]"
    puts "  DSP blocks:     [lindex $fitter_summary 11]"
    puts "  Fmax (MHz):     [lindex $fitter_summary 13]"
}

puts ""
puts "Output files:"
puts "  output_files/$project_name.sof"
puts "  output_files/$project_name.sof.rpt"

project_close
puts "Done."
