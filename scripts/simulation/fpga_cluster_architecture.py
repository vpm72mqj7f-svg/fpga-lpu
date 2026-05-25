"""
fpga_cluster_architecture.py — Multi-Server FPGA Inference Cluster Model
=========================================================================

Designs and models the FPGA cluster architecture for scaling beyond a
single 4U server. Covers four parallelism strategies, rack topology,
interconnect options, and cluster-level performance.

Architecture Principles
-----------------------
1. Data Parallel is primary: each server = independent inference unit
2. Model Parallel is the upgrade path for larger future models
3. No external switch dependency (unlike GPU clusters)
4. Physical isolation = multi-tenancy via server assignment

Cluster Scale Levels
--------------------
  Level 1: Single 4U server   (8 cards × 4 chips = 32 FPGA)  — base unit
  Level 2: Single rack        (up to 10 servers, 42U)         — enterprise
  Level 3: Multi-rack         (up to 100 servers)             — SaaS/cloud
  Level 4: Edge/multi-site    (geographically distributed)     — CDN-like

Usage:
  python scripts/simulation/fpga_cluster_architecture.py
  python scripts/simulation/fpga_cluster_architecture.py --servers 10 --strategy hybrid
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum, auto
from collections import defaultdict
import math
import sys
import argparse


# ============================================================================
# Base Server Specs (from fpga_4chip_pipeline.py results + proposal)
# ============================================================================

@dataclass
class ServerSpec:
    """Single 4U FPGA inference server."""
    name: str = "FPGA Inference Server"
    rack_units: int = 4
    cards_per_server: int = 8
    chips_per_card: int = 4
    total_chips: int = 32

    # Performance (from fpga_4chip_pipeline.py results)
    batch1_throughput_tps: float = 875          # batch=1 decode
    pipelined_throughput_tps: float = 23_104    # ideal deep pipeline
    avg_token_latency_ms: float = 1.14
    avg_per_layer_us: float = 18.7

    # KV Cache capacity (per server)
    hbm_total_gb: float = 32 * 32              # 32 chips × 32 GB = 1 TB
    kv_cache_per_token_bytes: int = 576        # FP8
    max_concurrent_sessions: int = 32          # practical limit (1/chip)

    # Power
    power_per_card_w: float = 550              # 4 chips + VRM + cooling
    power_per_server_w: float = 5_300          # 8 cards + CPU + fans

    # Physical
    weight_gb: float = 24 * 8                  # ~192 GB fp4 weights per server
    server_cost_rmb: float = 1_200_000         # including FPGA cards

    # Ports
    pcie_slots: int = 8                        # x16 slots
    management_nic: str = "1/10 GbE"           # BMC + management
    data_plane_ports: int = 0                  # optional high-speed uplinks


# ============================================================================
# Interconnect Options
# ============================================================================

class InterconnectType(Enum):
    NONE = auto()              # Data parallel, no cross-server data plane
    ETHERNET_100GBE = auto()   # 100 GbE RoCE v2
    ETHERNET_200GBE = auto()   # 200 GbE RoCE v2
    ETHERNET_400GBE = auto()   # 400 GbE RoCE v2
    INFINIBAND_NDR = auto()    # 400 Gbps InfiniBand NDR
    INFINIBAND_XDR = auto()    # 800 Gbps InfiniBand XDR
    PCIE_FABRIC = auto()       # External PCIe 5.0 switch fabric
    CUSTOM_FIBER = auto()      # Direct F-Tile SerDes over fiber


@dataclass
class InterconnectSpec:
    """Specification for a server-to-server interconnect."""
    ic_type: InterconnectType
    bandwidth_gbps: float
    latency_us: float          # one-way, including switch hops
    cost_per_port_rmb: float
    power_per_port_w: float
    protocol: str
    max_distance_m: float
    requires_switch: bool
    switch_cost_rmb: float     # per switch, if required
    notes: str


# Interconnect options catalog
INTERCONNECTS = {
    InterconnectType.NONE: InterconnectSpec(
        InterconnectType.NONE, 0, 0, 0, 0, "None",
        0, False, 0, "Data parallel only, no cross-server data path"
    ),
    InterconnectType.ETHERNET_200GBE: InterconnectSpec(
        InterconnectType.ETHERNET_200GBE, 200, 1.5, 8_000, 15,
        "RoCE v2 / UDP", 100, True, 80_000,
        "QSFP56, single-mode fiber. Standard ToR switch."
    ),
    InterconnectType.ETHERNET_400GBE: InterconnectSpec(
        InterconnectType.ETHERNET_400GBE, 400, 1.2, 15_000, 20,
        "RoCE v2 / UDP", 100, True, 150_000,
        "QSFP-DD, single-mode fiber. 25.6T switch."
    ),
    InterconnectType.INFINIBAND_NDR: InterconnectSpec(
        InterconnectType.INFINIBAND_NDR, 400, 0.8, 20_000, 18,
        "IB Verbs / RDMA", 150, True, 200_000,
        "NVIDIA Quantum-2, 40-port NDR switch. Lowest latency."
    ),
    InterconnectType.CUSTOM_FIBER: InterconnectSpec(
        InterconnectType.CUSTOM_FIBER, 256, 0.3, 5_000, 8,
        "Raw C2C SerDes frames", 500, False, 0,
        "Direct F-Tile → QSFP28 optical. No switch. Dual Ring topology."
    ),
}


# ============================================================================
# Parallelism Strategies
# ============================================================================

class ParallelismStrategy(Enum):
    DATA_PARALLEL = auto()       # Each server = full model replica
    EXPERT_PARALLEL = auto()     # Split 384 experts across N servers
    PIPELINE_PARALLEL = auto()   # Split 61 layers across N servers
    HYBRID = auto()              # Expert parallel within rack, data parallel across racks


@dataclass
class ClusterConfig:
    """Configuration for a multi-server FPGA cluster."""
    num_servers: int = 1
    strategy: ParallelismStrategy = ParallelismStrategy.DATA_PARALLEL
    interconnect: InterconnectType = InterconnectType.NONE
    rack_layout: str = "single"  # "single", "multi-rack", "distributed"

    # Derived
    servers_per_rack: int = 0
    num_racks: int = 0
    total_fpga_chips: int = 0
    total_hbm_tb: float = 0.0

    def __post_init__(self):
        self.total_fpga_chips = self.num_servers * 32
        self.total_hbm_tb = self.total_fpga_chips * 32 / 1024

        if self.rack_layout == "single":
            # 42U rack: ~4U per server + 2U ToR switch + 2U UPS margin
            self.servers_per_rack = min(self.num_servers, 9)  # 9×4=36U + 2U switch = 38U
            self.num_racks = math.ceil(self.num_servers / 9)
        elif self.rack_layout == "high-density":
            self.servers_per_rack = min(self.num_servers, 10)  # 10×4=40U tight
            self.num_racks = math.ceil(self.num_servers / 10)
        else:
            self.servers_per_rack = self.num_servers
            self.num_racks = 1


# ============================================================================
# Cluster Performance Model
# ============================================================================

class ClusterPerformanceModel:
    """Models throughput, latency, and efficiency for FPGA clusters."""

    def __init__(self, server_spec: ServerSpec = ServerSpec()):
        self.server = server_spec

    def analyze_data_parallel(self, config: ClusterConfig) -> Dict:
        """
        Data Parallel: each server runs the full model independently.
        Throughput scales linearly. No cross-server communication needed.
        """

        n = config.num_servers

        # Aggregate throughput (with load balancer overhead)
        lb_efficiency = 1.0 if n <= 4 else 0.98 if n <= 8 else 0.95
        aggregate_tps = n * self.server.batch1_throughput_tps * lb_efficiency

        # Pipelined throughput
        aggregate_pipelined_tps = n * self.server.pipelined_throughput_tps * lb_efficiency

        # Per-request latency unchanged (each request hits one server)
        p50_latency_ms = self.server.avg_token_latency_ms
        p99_latency_ms = self.server.avg_token_latency_ms * 1.5  # queueing

        # Concurrent sessions
        max_sessions = n * self.server.max_concurrent_sessions

        # KV Cache capacity
        total_kv_cache_tokens = n * (self.server.hbm_total_gb * 1e9) // self.server.kv_cache_per_token_bytes

        # Power
        total_power_kw = n * self.server.power_per_server_w / 1000

        # Cost
        total_cost_rmb = n * self.server.server_cost_rmb
        cost_per_tps = total_cost_rmb / aggregate_tps if aggregate_tps > 0 else float('inf')

        # Fault tolerance
        fault_domain = "per-server"
        degraded_throughput_tps = (n - 1) * self.server.batch1_throughput_tps * lb_efficiency
        availability_pct = (1 - (1/8760)) * 100  # ~99.99% per server, <1h downtime/yr

        return {
            'strategy': 'Data Parallel',
            'servers': n,
            'aggregate_tps': aggregate_tps,
            'aggregate_pipelined_tps': aggregate_pipelined_tps,
            'p50_latency_ms': p50_latency_ms,
            'p99_latency_ms': p99_latency_ms,
            'max_sessions': max_sessions,
            'total_kv_cache_tokens': total_kv_cache_tokens,
            'total_power_kw': total_power_kw,
            'total_cost_rmb': total_cost_rmb,
            'cost_per_tps': cost_per_tps,
            'fault_domain': fault_domain,
            'degraded_throughput_tps': degraded_throughput_tps,
            'availability_pct': availability_pct,
            'cross_server_bandwidth_gbps': 0,
            'cross_server_latency_us': 0,
        }

    def analyze_expert_parallel(self, config: ClusterConfig) -> Dict:
        """
        Expert Parallel: split 384 experts across N servers.
        Each server hosts 384/N experts. MoE dispatch goes cross-server.

        Trade-off: reduces per-server HBM (fewer experts to store) but
        adds cross-server all-to-all for every MoE layer.
        """

        n = config.num_servers
        experts_per_server = 384 // n

        # Per-server per-layer timing
        # Attention + shared expert: same as single server (always local)
        local_dsp_us = 6.72 + 2.98 + 0.25  # MLA + Shared + Router ≈ 10.0 μs

        # Expert hit probability per server
        # With 384/N experts per server, P(hit per expert) = (384/N) / 384 = 1/N
        p_hit_per_expert = 1.0 / n
        # Top-6 selections, probability at least 1 hits this server
        # Expected local experts = 6/N
        expected_local = 6.0 / n

        # For expert parallel to make sense, most experts are remote
        # Each remote expert requires: C2C dispatch (local) + cross-server + remote compute + return
        ic_spec = INTERCONNECTS.get(config.interconnect, INTERCONNECTS[InterconnectType.NONE])
        cross_server_us = ic_spec.latency_us

        # HBM time for local experts
        local_hbm_us = expected_local * 33.0 / (920 / 1024)  # ~36.7 μs per expert

        # DSP time: local expert DSP = expected_local * 5.97 μs
        local_expert_dsp_us = expected_local * 5.97

        # Remote expert: cross-server dispatch + remote HBM + remote DSP + reduce
        # Each remote expert = 2 × cross_server_us + 36.7 + 5.97
        remote_experts = 6.0 - expected_local
        remote_time_us = remote_experts * (2 * cross_server_us + 36.7 + 5.97)

        # Total per-layer
        per_layer_us = local_dsp_us + max(local_hbm_us, local_expert_dsp_us) + remote_time_us
        per_layer_us += 0.5  # C2C reduce aggregation

        # Token latency
        token_latency_us = per_layer_us * 61
        throughput_tps = 1e6 / per_layer_us if per_layer_us > 0 else 0

        # Aggregate cluster throughput
        aggregate_tps = n * throughput_tps  # Each server processes different tokens

        # Cross-server bandwidth
        # Each layer: (6 - expected_local) × 7168 B dispatch + same for reduce
        cross_server_bytes_per_layer = remote_experts * 2 * 7168
        cross_server_gbps = cross_server_bytes_per_layer * 8 / (per_layer_us / 1e6) / 1e9

        return {
            'strategy': 'Expert Parallel',
            'servers': n,
            'experts_per_server': experts_per_server,
            'expected_local_experts': expected_local,
            'per_layer_us': per_layer_us,
            'token_latency_us': token_latency_us,
            'throughput_tps_per_server': throughput_tps,
            'aggregate_tps': aggregate_tps,
            'cross_server_gbps': cross_server_gbps,
            'cross_server_us': cross_server_us,
            'interconnect': config.interconnect.name,
            'hbm_per_server_gb': experts_per_server * 33 * 0.5 / 1024 + 15,  # ~expert weights + attention
        }

    def analyze_pipeline_parallel(self, config: ClusterConfig) -> Dict:
        """
        Pipeline Parallel: split 61 layers across N servers.
        Each server handles 61/N consecutive layers.
        Hidden state (7168 B) passed between servers via interconnect.

        Reduces per-server HBM (fewer layers × weight) but adds
        pipeline bubbles and cross-server forwarding latency.
        """

        n = config.num_servers
        layers_per_server = math.ceil(61 / n)
        ic_spec = INTERCONNECTS.get(config.interconnect, INTERCONNECTS[InterconnectType.NONE])

        # Per-layer time (local to each server — all experts for those layers are local)
        # Actually: experts still distributed! Need expert parallel too for this to work.
        # Simplified model: assume all experts for assigned layers are local
        per_layer_us = 18.7  # from single-server model

        # Pipeline forward between servers: 7168 B over interconnect
        forward_latency_us = ic_spec.latency_us + (7168 * 8 / ic_spec.bandwidth_gbps) / 1000
        forward_latency_us = max(forward_latency_us, 0.4)  # minimum ~400ns

        # Pipeline bubble: first token fills pipeline (N-1 bubbles)
        # Steady state: tokens flow at per-layer rate + forward latency
        # Bottleneck: slowest server × its layers + N-1 forward hops per token

        server_time_us = layers_per_server * per_layer_us
        # Each token needs N-1 cross-server forwards
        total_forward_us = (n - 1) * forward_latency_us
        token_latency_us = 61 * per_layer_us + total_forward_us

        # Throughput: limited by bottleneck server (all servers process in parallel)
        # Pipeline rate = 1 / (per_layer_us + forward_latency_us/(layers_per_server))
        effective_per_layer_us = per_layer_us + forward_latency_us / layers_per_server
        throughput_tps = 1e6 / effective_per_layer_us

        # Aggregate: tokens flow through the pipeline — it's a single pipeline
        # NOT n × throughput — all servers work on the same tokens
        aggregate_tps = throughput_tps

        # Cross-server bandwidth
        # Each token: (n-1) forwards × 7168 B / token_latency_us
        bps = (n - 1) * 7168 * 8 / (token_latency_us / 1e6)
        cross_server_gbps = bps / 1e9

        return {
            'strategy': 'Pipeline Parallel',
            'servers': n,
            'layers_per_server': layers_per_server,
            'per_layer_us': per_layer_us,
            'forward_latency_us': forward_latency_us,
            'token_latency_us': token_latency_us,
            'effective_per_layer_us': effective_per_layer_us,
            'throughput_tps': throughput_tps,
            'aggregate_tps': aggregate_tps,
            'cross_server_gbps': cross_server_gbps,
            'pipeline_bubble_pct': (total_forward_us / token_latency_us * 100) if token_latency_us > 0 else 0,
            'hbm_per_server_gb': layers_per_server * (33 * 12 + 100) / 1024,  # rough
        }

    def analyze_hybrid(self, config: ClusterConfig) -> Dict:
        """
        Hybrid: Data Parallel across racks × Expert/Pipeline within rack.

        Rack-level: expert or pipeline parallel for large model fit.
        Cross-rack: data parallel for throughput scaling.

        This is the most flexible architecture for:
        - Future models too large for 1 server
        - Mixed workloads (some tenants need full model, some share)
        """

        servers_per_rack = min(config.num_servers, 9)
        num_racks = max(1, config.num_servers // servers_per_rack)

        # Within rack: expert parallel (servers share the model)
        rack_config = ClusterConfig(
            num_servers=servers_per_rack,
            strategy=ParallelismStrategy.EXPERT_PARALLEL,
            interconnect=config.interconnect,
        )
        rack_result = self.analyze_expert_parallel(rack_config)

        # Across racks: data parallel
        rack_throughput = rack_result['aggregate_tps']
        aggregate_tps = num_racks * rack_throughput

        return {
            'strategy': 'Hybrid (Expert-in-Rack, Data-cross-Rack)',
            'servers': config.num_servers,
            'servers_per_rack': servers_per_rack,
            'num_racks': num_racks,
            'within_rack_strategy': 'Expert Parallel',
            'cross_rack_strategy': 'Data Parallel',
            'per_rack_throughput_tps': rack_throughput,
            'aggregate_tps': aggregate_tps,
            'per_rack_cross_server_gbps': rack_result['cross_server_gbps'],
            'cross_rack_bandwidth_gbps': 0,  # no data path between racks
            'token_latency_ms': rack_result['token_latency_us'] / 1000,
            'max_sessions': num_racks * 32,
        }


# ============================================================================
# Rack Topology Design
# ============================================================================

@dataclass
class RackDesign:
    """Physical rack layout for FPGA inference cluster."""
    total_rack_units: int = 42
    used_rack_units: int = 0
    servers: int = 0
    switches: int = 0
    pdus: int = 2  # redundant power
    total_power_kw: float = 0.0
    total_weight_kg: float = 0.0


def design_rack(config: ClusterConfig, server_spec: ServerSpec) -> RackDesign:
    """Design physical rack layout for a cluster configuration."""
    rack = RackDesign()

    # 4U per server
    servers_in_rack = min(config.num_servers, (42 - 4) // 4)  # 2U switch + 2U margin
    servers_in_rack = min(servers_in_rack, 9)  # practical max

    rack.servers = servers_in_rack
    rack.used_rack_units += servers_in_rack * server_spec.rack_units

    # Switch (if needed)
    ic_spec = INTERCONNECTS.get(config.interconnect)
    if ic_spec and ic_spec.requires_switch:
        rack.switches = 1
        rack.used_rack_units += 2  # 2U for ToR switch

    # PDU
    rack.used_rack_units += 1  # vertical PDUs

    # Power
    rack.total_power_kw = servers_in_rack * server_spec.power_per_server_w / 1000
    if ic_spec and ic_spec.requires_switch:
        rack.total_power_kw += 0.5  # switch power

    # Weight estimate
    rack.total_weight_kg = servers_in_rack * 45 + (20 if rack.switches > 0 else 0)

    return rack


# ============================================================================
# Cluster-Level Fault Tolerance
# ============================================================================

class ClusterFaultModel:
    """Fault tolerance and high availability model for FPGA clusters."""

    def __init__(self, server_spec: ServerSpec):
        self.server = server_spec

    def analyze(self, config: ClusterConfig) -> Dict:
        n = config.num_servers
        strategy = config.strategy

        # Per-server MTBF (AGM 039-F, from proposal §6.6)
        chip_mtbf_h = 50_000   # per chip
        # 32 chips per server
        server_mtbf_h = chip_mtbf_h / 32  # ~1,562 hours

        # Recovery times
        chip_recovery_ms = 100      # chip-level self-healing (C2C re-route)
        card_recovery_h = 4         # manual card replacement
        server_recovery_h = 2       # server reboot/replace

        if strategy == ParallelismStrategy.DATA_PARALLEL:
            # Single server failure: LB removes it, others take over
            # Degraded throughput: (n-1)/n
            degraded_pct = (n - 1) / n * 100
            downtime_per_failure_h = server_recovery_h
            # Cluster availability
            cluster_mtbf_h = server_mtbf_h / n
            annual_downtime_h = 8760 * downtime_per_failure_h / cluster_mtbf_h if cluster_mtbf_h > 0 else 0
            availability = (1 - annual_downtime_h / 8760) * 100

        elif strategy in (ParallelismStrategy.EXPERT_PARALLEL,
                          ParallelismStrategy.PIPELINE_PARALLEL):
            # Any server failure stalls the entire pipeline
            degraded_pct = 0  # complete stall until recovery (or reconfiguration)
            # But chip-level self-healing can prevent many server failures
            availability = 99.97  # from proposal §6.6.5

        else:
            degraded_pct = (n - 1) / n * 100
            availability = 99.95

        return {
            'server_mtbf_h': server_mtbf_h,
            'cluster_mtbf_h': server_mtbf_h / n if n > 0 else float('inf'),
            'degraded_throughput_pct': degraded_pct,
            'estimated_availability_pct': availability,
            'chip_recovery_ms': chip_recovery_ms,
            'annual_downtime_estimate_h': (100 - availability) / 100 * 8760,
        }


# ============================================================================
# Cost Model
# ============================================================================

def cluster_cost_model(config: ClusterConfig, server_spec: ServerSpec) -> Dict:
    """Total cost of ownership model for FPGA cluster."""

    n = config.num_servers
    ic_spec = INTERCONNECTS.get(config.interconnect)

    # Capital expenditure
    server_cost = n * server_spec.server_cost_rmb

    # Interconnect cost
    interconnect_cost = 0
    if ic_spec and ic_spec.requires_switch:
        num_switches = math.ceil(n / 32)  # 32-port switch
        interconnect_cost = n * ic_spec.cost_per_port_rmb + num_switches * ic_spec.switch_cost_rmb
    elif ic_spec and not ic_spec.requires_switch:
        interconnect_cost = n * ic_spec.cost_per_port_rmb

    # Rack infrastructure
    num_racks = math.ceil(n / 9)
    rack_cost = num_racks * 15_000  # PDU, cabling, rails

    total_capex = server_cost + interconnect_cost + rack_cost

    # Annual operating expenditure
    power_kw = n * server_spec.power_per_server_w / 1000
    if ic_spec and ic_spec.requires_switch:
        power_kw += math.ceil(n / 32) * 0.5  # switch power
    annual_power_cost = power_kw * 8760 * 0.8  # 0.8 RMB/kWh

    # Cooling (PUE 1.3 for air-cooled DC)
    annual_cooling_cost = annual_power_cost * 0.3

    # Maintenance (5% of capex)
    annual_maintenance = total_capex * 0.05

    annual_opex = annual_power_cost + annual_cooling_cost + annual_maintenance

    return {
        'servers': n,
        'total_capex_rmb': total_capex,
        'capex_per_server_rmb': total_capex / n if n > 0 else 0,
        'interconnect_cost_rmb': interconnect_cost,
        'rack_infrastructure_rmb': rack_cost,
        'total_power_kw': power_kw,
        'annual_power_cost_rmb': annual_power_cost,
        'annual_cooling_cost_rmb': annual_cooling_cost,
        'annual_maintenance_rmb': annual_maintenance,
        'annual_opex_rmb': annual_opex,
        'opex_as_pct_of_capex': annual_opex / total_capex * 100 if total_capex > 0 else 0,
        'tco_3yr_rmb': total_capex + 3 * annual_opex,
        'tco_5yr_rmb': total_capex + 5 * annual_opex,
    }


# ============================================================================
# Main Analysis
# ============================================================================

def print_cluster_analysis():
    """Run comprehensive cluster architecture analysis."""

    server = ServerSpec()

    print()
    print("=" * 79)
    print("   FPGA Inference Cluster Architecture")
    print("   Multi-Server Scale-Out Design")
    print("=" * 79)
    print()
    print(f"   Base unit: 4U server, 8 cards x 4 AGM 039-F = 32 chips")
    print(f"   Single server: {server.batch1_throughput_tps:.0f} tok/s (batch=1)")
    print(f"   Single server: {server.pipelined_throughput_tps:.0f} tok/s (ideal pipeline)")
    print(f"   Single server power: {server.power_per_server_w/1000:.1f} kW")
    print()

    model = ClusterPerformanceModel()

    # ==========================================================================
    # Strategy 1: Data Parallel (Primary)
    # ==========================================================================
    print("=" * 79)
    print("   STRATEGY 1: Data Parallel (Recommended)")
    print("=" * 79)
    print()
    print("   Each server = independent full-model replica.")
    print("   Load balancer distributes requests. No cross-server data plane.")
    print("   Throughput scales linearly. Fault isolation per server.")
    print()

    print(f"   {'Servers':>8s} {'Agg TPS':>10s} {'Sessions':>10s} "
          f"{'Power(kW)':>10s} {'Cost(M RMB)':>12s} {'Cost/TPS':>10s} {'Avail%':>8s}")
    print(f"   {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")

    for n in [1, 2, 4, 8, 10, 20, 50, 100]:
        config = ClusterConfig(num_servers=n, strategy=ParallelismStrategy.DATA_PARALLEL)
        r = model.analyze_data_parallel(config)
        print(f"   {n:>8d} {r['aggregate_tps']:>10.0f} {r['max_sessions']:>10d} "
              f"{r['total_power_kw']:>10.1f} {r['total_cost_rmb']/1e6:>11.2f} "
              f"{r['cost_per_tps']:>10.0f} {r['availability_pct']:>7.2f}%")

    print()
    print("   Observations:")
    print("   - Throughput scales near-linearly (95-100% efficient)")
    print("   - No switch, no cross-server cables, no RoCE IP")
    print("   - Physical tenant isolation: assign servers, not GPU MIG slices")
    print("   - Server failure = ~12.5% throughput drop (8->7 servers)")
    print()

    # ==========================================================================
    # Strategy 2: Expert Parallel
    # ==========================================================================
    print("=" * 79)
    print("   STRATEGY 2: Expert Parallel (for oversize models)")
    print("=" * 79)
    print()
    print("   Split 384 experts across N servers. MoE dispatch crosses servers.")
    print("   Required when a future model has >384 experts or >32GB HBM per chip.")
    print()

    print(f"   {'Servers':>8s} {'Exp/Srv':>8s} {'Local Exp':>10s} "
          f"{'Layer(us)':>10s} {'TPS/Srv':>10s} {'Agg TPS':>10s} {'X-Srv BW':>10s}")
    print(f"   {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for n in [2, 4, 8]:
        for ic_name in [InterconnectType.CUSTOM_FIBER, InterconnectType.ETHERNET_200GBE]:
            config = ClusterConfig(
                num_servers=n,
                strategy=ParallelismStrategy.EXPERT_PARALLEL,
                interconnect=ic_name,
            )
            r = model.analyze_expert_parallel(config)
            print(f"   {n:>8d} {r['experts_per_server']:>8d} {r['expected_local_experts']:>9.1f} "
                  f"{r['per_layer_us']:>10.1f} {r['throughput_tps_per_server']:>10.0f} "
                  f"{r['aggregate_tps']:>10.0f} {r['cross_server_gbps']:>9.1f} Gbps  [{ic_name.name}]")

    print()
    print("   Observations:")
    print("   - Expert Parallel REDUCES throughput (not increases it)")
    print("   - Cross-server dispatch ~0.3-1.5 us dominates per-layer time")
    print("   - Only advantageous when model CANNOT fit in one server")
    print("   - Custom Fiber (direct F-Tile) has latency advantage over Ethernet")
    print("   - Use case: DeepSeek V5 with 768 experts or 16K hidden dim")
    print()

    # ==========================================================================
    # Strategy 3: Pipeline Parallel
    # ==========================================================================
    print("=" * 79)
    print("   STRATEGY 3: Pipeline Parallel (for deep models)")
    print("=" * 79)
    print()
    print("   Split 61 layers across N servers. Hidden state forwarded between servers.")
    print("   Reduces per-server HBM for layer weights. Adds pipeline bubbles.")
    print()

    print(f"   {'Servers':>8s} {'Layers/Srv':>10s} {'Fwd(us)':>8s} "
          f"{'Latency(us)':>12s} {'TPS':>8s} {'Bubble%':>8s}")
    print(f"   {'-'*8} {'-'*10} {'-'*8} {'-'*12} {'-'*8} {'-'*8}")

    for n in [2, 4, 8]:
        for ic_name in [InterconnectType.CUSTOM_FIBER, InterconnectType.ETHERNET_200GBE]:
            config = ClusterConfig(
                num_servers=n,
                strategy=ParallelismStrategy.PIPELINE_PARALLEL,
                interconnect=ic_name,
            )
            r = model.analyze_pipeline_parallel(config)
            print(f"   {n:>8d} {r['layers_per_server']:>10d} {r['forward_latency_us']:>8.2f} "
                  f"{r['token_latency_us']:>12.0f} {r['throughput_tps']:>8.0f} "
                  f"{r['pipeline_bubble_pct']:>7.1f}%  [{ic_name.name}]")

    print()
    print("   Observations:")
    print("   - Pipeline parallel reduces throughput due to forward latency")
    print("   - Bubbles from N-1 cross-server forwards per token")
    print("   - Best for very deep models (100+ layers) where HBM is limiting")
    print("   - Current 61-layer model fits in 1 server -- this is future-proofing")
    print()

    # ==========================================================================
    # Strategy Comparison
    # ==========================================================================
    print("=" * 79)
    print("   Strategy Comparison Summary")
    print("=" * 79)
    print()

    n_compare = 4  # Compare with 4 servers
    strategies = {
        'Data Parallel': model.analyze_data_parallel(
            ClusterConfig(n_compare, ParallelismStrategy.DATA_PARALLEL)),
        'Expert Parallel (Fiber)': model.analyze_expert_parallel(
            ClusterConfig(n_compare, ParallelismStrategy.EXPERT_PARALLEL,
                          InterconnectType.CUSTOM_FIBER)),
        'Expert Parallel (200GbE)': model.analyze_expert_parallel(
            ClusterConfig(n_compare, ParallelismStrategy.EXPERT_PARALLEL,
                          InterconnectType.ETHERNET_200GBE)),
        'Pipeline Parallel (Fiber)': model.analyze_pipeline_parallel(
            ClusterConfig(n_compare, ParallelismStrategy.PIPELINE_PARALLEL,
                          InterconnectType.CUSTOM_FIBER)),
    }

    print(f"   {'Strategy':<25s} {'TPS':>10s} {'Latency':>10s} "
          f"{'X-Srv BW':>10s} {'HBM/Srv':>10s}")
    print(f"   {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in strategies.items():
        tps = r.get('aggregate_tps', r.get('throughput_tps', 0))
        lat = r.get('token_latency_us', r.get('p50_latency_ms', 0) * 1000)
        if 'p50_latency_ms' in r:
            lat = r['p50_latency_ms'] * 1000
        bw = r.get('cross_server_gbps', r.get('cross_server_bandwidth_gbps', 0))
        hbm = r.get('hbm_per_server_gb', server.hbm_total_gb / 1024)
        print(f"   {name:<25s} {tps:>10.0f} {lat:>10.0f} {bw:>10.1f} {hbm:>10.1f}")

    print()

    # ==========================================================================
    # Rack Design
    # ==========================================================================
    print("=" * 79)
    print("   Rack-Level Physical Design")
    print("=" * 79)
    print()

    for n_servers in [4, 9, 10]:
        config = ClusterConfig(num_servers=n_servers)
        rack = design_rack(config, server)
        print(f"   {n_servers} servers in rack:")
        print(f"     Used: {rack.used_rack_units}U / {rack.total_rack_units}U")
        print(f"     Power: {rack.total_power_kw:.1f} kW (need {'3-phase 32A' if rack.total_power_kw > 10 else 'single-phase 32A'})")
        print(f"     Weight: {rack.total_weight_kg:.0f} kg")
        print(f"     Switches: {rack.switches}")

    print()
    print("   Rack layout (9 servers, Data Parallel):")
    print("   +-----------------------------------------+")
    print("   |  42U  |                                  |")
    print("   |  41U  |  Server 9   (4U)                 |")
    print("   |  ...  |  ...                             |")
    print("   |  10U  |  Server 2   (4U)                 |")
    print("   |   6U  |  Server 1   (4U)                 |")
    print("   |   2U  |  Management Switch (1/10 GbE)     |")
    print("   |   0U  |  PDU (rear)                      |")
    print("   +-----------------------------------------+")
    print()

    # ==========================================================================
    # TCO Analysis
    # ==========================================================================
    print("=" * 79)
    print("   Total Cost of Ownership (TCO) -- Data Parallel")
    print("=" * 79)
    print()

    print(f"   {'Servers':>8s} {'CAPEX(M)':>10s} {'OPEX/yr(M)':>12s} "
          f"{'TCO 3yr(M)':>12s} {'TCO 5yr(M)':>12s} {'TCO/tps-yr':>12s}")
    print(f"   {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for n in [1, 4, 9, 20, 50, 100]:
        config = ClusterConfig(num_servers=n, strategy=ParallelismStrategy.DATA_PARALLEL)
        cost = cluster_cost_model(config, server)
        perf = model.analyze_data_parallel(config)
        tco_per_tps_yr = cost['tco_5yr_rmb'] / perf['aggregate_tps'] / 5 if perf['aggregate_tps'] > 0 else 0
        print(f"   {n:>8d} {cost['total_capex_rmb']/1e6:>9.2f} {cost['annual_opex_rmb']/1e6:>11.2f} "
              f"{cost['tco_3yr_rmb']/1e6:>11.2f} {cost['tco_5yr_rmb']/1e6:>11.2f} "
              f"{tco_per_tps_yr:>11.2f}")

    print()
    print("   Comparison benchmarks:")
    print("   - H100 8-GPU server: ~14M RMB capex, ~1,200 tok/s -> 2,333 RMB/tps/yr")
    print("   - FPGA 1-server:     ~1.2M RMB capex, ~875 tok/s  -> 274 RMB/tps/yr")
    print("   - FPGA 100-server:   ~87M RMB capex, ~83,000 tok/s -> 210 RMB/tps/yr")
    print()

    # ==========================================================================
    # Cluster Topology Diagrams
    # ==========================================================================
    print("=" * 79)
    print("   Cluster Topology Options")
    print("=" * 79)
    print()

    print("   [A] Data Parallel -- No cross-server data plane")
    print()
    print("        Client Requests")
    print("             |")
    print("      +------+------+")
    print("      |  L4/L7 LB   |  (nginx/HAProxy/envoy, 1GbE mgmt)")
    print("      +------+------+")
    print("      +------+------+")
    print("      |      |      |")
    print("   +--v--+ +-v---+ +-v----+")
    print("   | Svr1| |Svr2 | |SvrN |  (each = 4U, 32 FPGA, full model)")
    print("   | 875 | |875  | |875  |  tok/s")
    print("   +-----+ +-----+ +-----+")
    print("   No cross-server data path -> no switch, no RoCE, no fiber")
    print()

    print("   [B] Expert Parallel -- Custom Fiber Dual Ring")
    print()
    print("        +----------------------------------+")
    print("        |         Fiber Ring A              |")
    print("        |   Svr1 <----------> Svr2         |")
    print("        |     v                    v        |")
    print("        |   Svr4 <----------> Svr3         |")
    print("        +----------------------------------+")
    print()
    print("        F-Tile SerDes -> QSFP28 optical -> 256 Gbps/link")
    print("        Same C2C protocol as intra-card, extended over fiber")
    print("        ~300 ns hop latency (vs 50 ns on-PCB)")
    print("        No switch needed (Dual Ring redundancy)")
    print()

    print("   [C] Expert Parallel -- Switched Ethernet")
    print()
    print("      +------+  +------+  +------+  +------+")
    print("      | Svr1 |  | Svr2 |  | Svr3 |  | Svr4 |")
    print("      +--+---+  +--+---+  +--+---+  +--+---+")
    print("         | 200GbE | 200GbE | 200GbE | 200GbE")
    print("         +--------+--------+--------+")
    print("                  |   ToR  |")
    print("                  | Switch |  (25.6T, 32x400G)")
    print("                  +--------+")
    print("        Standard RoCE v2, ~1.2 us latency")
    print()

    # ==========================================================================
    # Recommendations
    # ==========================================================================
    print("=" * 79)
    print("   Architecture Recommendations")
    print("=" * 79)
    print()
    print("   1. PRIMARY (Data Parallel) -- for current DeepSeek V4 Pro (1.6T params)")
    print("      - Deploy N independent 4U servers")
    print("      - L4/L7 load balancer for request distribution")
    print("      - No cross-server interconnect needed")
    print("      - Linear throughput scaling: N x 875 tok/s")
    print("      - Fault isolation: server failure -> LB removes it")
    print("      - Multi-tenancy: assign servers to tenants, physical isolation")
    print()
    print("   2. UPGRADE PATH (Hybrid) -- for future larger models (V5, 3T+ params)")
    print("      - Phase 1: Add F-Tile fiber ports to server design")
    print("        (4x F-Tile lane per server -> QSFP28 cage, <5K RMB BOM)")
    print("      - Phase 2: Within-rack Expert Parallel via Custom Fiber")
    print("        (servers in same rack share expert pool)")
    print("      - Phase 3: Cross-rack Data Parallel")
    print("        (each rack = one logical inference unit)")
    print()
    print("   3. NOT RECOMMENDED (for current model):")
    print("      - InfiniBand (too expensive, too complex, supply-constrained)")
    print("      - Pure Pipeline Parallel (bubbles hurt throughput)")
    print("      - External PCIe Fabric (distance limited, vendor lock-in)")
    print()
    print("   4. KEY ARCHITECTURAL INSIGHT:")
    print("      Unlike GPU clusters that NEED InfiniBand/RoCE for tensor")
    print("      parallelism, FPGA's fp4 compression means the full model fits")
    print("      in ONE server. The cluster exists for THROUGHPUT, not CAPACITY.")
    print("      This fundamentally changes the interconnect requirements --")
    print("      from high-bandwidth all-to-all (GPU) to zero-data-path (FPGA).")
    print()

    print("=" * 79)
    print("   End of Cluster Architecture Analysis")
    print("=" * 79)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FPGA Multi-Server Cluster Architecture Analysis"
    )
    parser.add_argument('--servers', type=int, default=0,
                        help='Number of servers (0 = run all analyses)')
    parser.add_argument('--strategy', type=str, default='data',
                        choices=['data', 'expert', 'pipeline', 'hybrid'],
                        help='Parallelism strategy')
    args = parser.parse_args()

    print_cluster_analysis()
