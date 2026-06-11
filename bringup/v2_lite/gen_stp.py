#!/usr/bin/env python3
"""Generate correct Intel SignalTap session .stp file."""
signals = [
    'led[3]','led[2]','led[1]','led[0]',
    'dbg_ffn_state[3]','dbg_ffn_state[2]','dbg_ffn_state[1]','dbg_ffn_state[0]',
    'dbg_ffn_busy','dbg_ffn_done','dbg_ffn_pass',
    'dbg_ffn_td0[7]','dbg_ffn_td0[6]','dbg_ffn_td0[5]','dbg_ffn_td0[4]',
    'dbg_ffn_td0[3]','dbg_ffn_td0[2]','dbg_ffn_td0[1]','dbg_ffn_td0[0]',
    'dbg_ffn_td1[7]','dbg_ffn_td1[6]','dbg_ffn_td1[5]','dbg_ffn_td1[4]',
    'dbg_ffn_td1[3]','dbg_ffn_td1[2]','dbg_ffn_td1[1]','dbg_ffn_td1[0]',
    'dbg_ffn_arvalid','dbg_ffn_arready',
    'dbg_ffn_rx_v','dbg_ffn_rx_r','dbg_ffn_tx_v','dbg_ffn_tx_r',
    'dbg_hbm_tg_pass',
    'dbg_pcie_pll',
    'dbg_pcie_pll_bank[15]','dbg_pcie_pll_bank[14]','dbg_pcie_pll_bank[13]','dbg_pcie_pll_bank[12]',
    'dbg_pcie_pll_bank[11]','dbg_pcie_pll_bank[10]','dbg_pcie_pll_bank[9]','dbg_pcie_pll_bank[8]',
    'dbg_pcie_pll_bank[7]','dbg_pcie_pll_bank[6]','dbg_pcie_pll_bank[5]','dbg_pcie_pll_bank[4]',
    'dbg_pcie_pll_bank[3]','dbg_pcie_pll_bank[2]','dbg_pcie_pll_bank[1]','dbg_pcie_pll_bank[0]',
]

n = len(signals)
sig_entries = '\n'.join([f'          <wire name="{s}" tap_mode="classic" type="unknown"/>' for s in signals])

stp = f'''<?xml version="1.0" encoding="UTF-8"?>
<session sof_file="">
  <display_tree gui_logging_enabled="0">
    <display_branch instance="auto_signaltap_0" signal_set="default" trigger="default"/>
  </display_tree>
  <instance enabled="true" entity_name="sld_signaltap" is_auto_node="yes" is_expanded="true" name="auto_signaltap_0" source_file="sld_signaltap.vhd">
    <node_ip_info instance_id="0" mfg_id="110" node_id="0" version="6"/>
    <position_info>
      <single attribute="active tab" value="1"/>
    </position_info>
    <signal_set global_temp="1" name="default">
      <clock name="core_clk_iopll_ref_clk_clk" polarity="posedge" tap_mode="classic"/>
      <config pipeline_level="0" ram_type="AUTO" reserved_data_nodes="0" reserved_storage_qualifier_nodes="0" reserved_trigger_nodes="0" sample_depth="4096" trigger_in_enable="no" trigger_out_enable="no"/>
      <top_entity/>
      <signal_vec>
        <trigger_input_vec>
{sig_entries}
        </trigger_input_vec>
        <data_input_vec>
{sig_entries}
        </data_input_vec>
        <storage_qualifier_input_vec>
{sig_entries}
        </storage_qualifier_input_vec>
      </signal_vec>
      <trigger attribute_mem_mode="false" gap_record="true" global_temp="1" name="default" position="pre" power_up_trigger_mode="false" record_data_gap="true" segment_size="64" storage_mode="off" storage_qualifier_disabled="no" storage_qualifier_port_is_pin="true" storage_qualifier_port_name="auto_stp_external_storage_qualifier" storage_qualifier_port_tap_mode="classic" trigger_type="circular">
        <power_up_trigger position="pre" storage_qualifier_disabled="no"/>
        <events use_custom_flow_control="no">
          <level enabled="yes" name="condition1" type="basic">
            <power_up enabled="yes">
            </power_up>
            <op_node/>
          </level>
        </events>
        <storage_qualifier_events>
          <transitional>{"1" * n}
            <pwr_up_transitional>{"1" * n}</pwr_up_transitional>
          </transitional>
        </storage_qualifier_events>
      </trigger>
    </signal_set>
  </instance>
  <mnemonics/>
  <global_info>
    <single attribute="active instance" value="0"/>
  </global_info>
</session>'''

import sys
out_path = sys.argv[1] if len(sys.argv) > 1 else 'v2_lite_full.stp'
with open(out_path, 'w') as f:
    f.write(stp)
print(f'STP written: {out_path} ({n} signals)')
