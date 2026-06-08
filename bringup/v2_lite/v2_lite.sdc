create_clock -name {hbm_0_example_design_pll_ref_clk_clk} -period 10.000 -waveform { 0.000 5.000 } [get_ports { hbm_0_example_design_pll_ref_clk_clk }]	
create_clock -name {core_clk_iopll_ref_clk_clk} -period 10.000 -waveform { 0.000 5.000 } [get_ports { core_clk_iopll_ref_clk_clk }]	


#derive_pll_clocks
#not supported in S10 device family