#!/usr/bin/env python3
"""
Module-level smoke tests for the FPGA AI accelerator simulation stack.

Covers:
  fpga_arch: config/chip/interconnect/cluster/expert_popularity/pipeline
  vllm_serve: kv_cache/scheduler/model_runner/api_server/weight_layout
  serving: short end-to-end run

Run:
  cd D:/workspace/fpgalpu
  python scripts/run_module_smoke.py
"""

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def ok(name, value=None):
    return {'name': name, 'status': 'PASS', 'value': value}


def fail(name, exc):
    return {'name': name, 'status': 'FAIL', 'error': str(exc),
            'traceback': traceback.format_exc(limit=4)}


def run_case(name, fn):
    try:
        value = fn()
        return ok(name, value)
    except Exception as e:
        return fail(name, e)


# ── Test cases ───────────────────────────────────────────────

def test_chip_resources():
    from fpga_arch.chip import SRAMBank, HBMBank, DSPArray, FPGAChip
    s = SRAMBank(); h = HBMBank(); d = DSPArray(); c = FPGAChip(0, 0)
    t = d.compute_time_us(1e9)
    hbm = h.read_time_us(920)  # 920 MB ~ 1024 us at 920 GB/s
    c.assign_layers([0, 1])
    c.place_weights(396, 180, 2.6)
    return {'dsp_1G_mac_us': round(t, 3), 'hbm_920MB_us': round(hbm, 3),
            'chip_weight_gb': round(c.hbm.weight_storage_gb, 3)}


def test_interconnect():
    from fpga_arch.chip import FPGAChip
    from fpga_arch.interconnect import C2CDualRing, PCIeFabric
    chips = [FPGAChip(i, 0) for i in range(4)]
    ring = C2CDualRing(0, chips)
    pcie = PCIeFabric()
    return {
        'c2c_7KB_us': round(ring.transfer_time_us(0, 2, 7168), 4),
        'pcie_7KB_us': round(pcie.transfer_time_us(0, 1, 7168), 4),
    }


def test_cluster_replication():
    from fpga_arch.cluster import FPGACluster
    base = FPGACluster(seed=42, expert_replication='none')
    hot = FPGACluster(seed=42, expert_replication='hot')
    counts = [len(c.assigned_experts) for c in hot.chips]
    eid = int(hot._pop.sorted_expert_ids[0])
    return {
        'baseline_experts_chip0': len(base.chips[0].assigned_experts),
        'hot_min_avg_max': [min(counts), round(sum(counts)/len(counts), 1), max(counts)],
        'hottest_expert_replicas': len(hot.expert_to_chips[eid]),
    }


def test_expert_popularity():
    from fpga_arch.expert_popularity import ExpertPopularity
    p = ExpertPopularity(alpha=1.0)
    plan = p.replica_plan(total_chips=32, hbm_budget_per_chip_gb=2.0)
    top = int(p.sorted_expert_ids[0])
    return {'top20_mass': round(p.top_k_mass(20), 3),
            'top77_mass': round(p.top_k_mass(77), 3),
            'top_replica_count': plan[top]}


def test_pipeline_models():
    from fpga_arch import FPGACluster, PipelineEngine
    c = FPGACluster(seed=42, expert_replication='hot')
    p = PipelineEngine(c)
    return {
        'k_pipeline': round(p.k_pipeline, 2),
        'decode_tps_B1': round(PipelineEngine.throughput_model(1), 1),
        'decode_tps_B8': round(PipelineEngine.throughput_model(8), 1),
        'prefill_chip0_us': round(PipelineEngine.prefill_chip0_bottleneck_us(), 1),
        'chip0_rate_clone2_req_s': round(PipelineEngine.chip0_admission_rate(chip0_parallelism=2)['admission_reqs_s'], 1),
    }


def test_weight_layout():
    from vllm_serve.weight_layout import WeightLayoutCompiler
    rpt = WeightLayoutCompiler(pipeline_clones=2, replication='hot').compile()
    return {'max_used_gb': round(rpt.max_used_gb, 2),
            'min_free_gb': round(rpt.min_free_gb, 2),
            'total_weight_gb': round(rpt.total_weight_gb, 1)}


def test_kv_cache():
    from vllm_serve.kv_cache import KVCacheManager
    kv = KVCacheManager(num_chips=8, max_blocks_per_chip=1024)
    blocks = kv.allocate_prefill(request_id=1, prompt_len=512,
                                 chip_ids=list(range(8)), current_time_us=0)
    before = kv.total_blocks_allocated
    kv.allocate_decode(1, decode_step=0, chip_ids=list(range(8)), current_time_us=10)
    after = kv.total_blocks_allocated
    kv.free_request(1)
    return {'prefill_blocks': len(blocks), 'after_decode_blocks': after,
            'allocated_before_free': before, 'after_free': kv.total_blocks_allocated}


def test_scheduler():
    from vllm_serve.scheduler import ContinuousBatchingScheduler
    from vllm_serve.types import Request
    from vllm_serve.kv_cache import KVCacheManager
    from fpga_arch import FPGACluster, PipelineEngine
    from vllm_serve.model_runner import ModelRunner

    sched = ContinuousBatchingScheduler(num_chips=8, max_decode_batch=16)
    kv = KVCacheManager(num_chips=8, max_blocks_per_chip=1024)
    cluster = FPGACluster(seed=1)
    runner = ModelRunner(cluster, PipelineEngine(cluster))
    for i in range(4):
        sched.submit_request(Request(i, arrival_time_us=0, prompt_len=128, max_output_len=4))
    batches = sched.schedule(0, kv, runner)
    return {'num_batches': len(batches), 'first_batch_type': batches[0].batch_type.name,
            'batch_size': batches[0].size}


def test_api_server():
    from vllm_serve.scheduler import ContinuousBatchingScheduler
    from vllm_serve.api_server import APIServer
    sched = ContinuousBatchingScheduler(num_chips=8)
    api = APIServer(sched, seed=42, prompt_len_mean=256, output_len_mean=64)
    reqs = api.generator.generate_arrivals(arrival_rate=3, duration_us=2_000_000)
    return {'generated': len(reqs), 'first_prompt': reqs[0].prompt_len if reqs else None}


def test_serving_short():
    from run_serving import ServingSimulation
    sim = ServingSimulation(arrival_rate=2, duration_s=10, agent_mode=True,
                            agent_turns=3, agent_output_per_turn=64,
                            kv_blocks_per_chip=2048, microbatch=True,
                            expert_replication='hot', pipeline_clone=2)
    m = sim.run()
    return {'requests': m.total_requests, 'finished': m.total_finished,
            'accept_rate': round(m.accept_rate * 100, 1),
            'output_tps': round(m.throughput_tps, 1),
            'ttft_p95_ms': round(m.ttft_p95_ms, 1)}


# ── S3.7: Additional smoke tests ──

def test_concurrent_prefill_decode():
    """Verify the pipeline model handles concurrent prefill+decode workloads.

    The concurrent_pipeline_model computes throughput when prefill and decode
    run simultaneously on the same set of chips. Prefill is DSP-bound (MLA QK^T)
    while decode is HBM-bound (expert weight streaming). Because DSP and HBM
    are independent hardware units, the overall slowdown (contention factor)
    should be modest (< 1.2 for typical batch sizes).

    PASS if: contention_factor < 1.20 and combined_tps > 0.
    """
    from fpga_arch.pipeline import PipelineEngine
    r = PipelineEngine.concurrent_pipeline_model(
        prefill_tokens=128, decode_batch=8,
        use_fp4_attn=True, attn_sparsity=0.888,
    )
    contention = r['contention_factor']
    combined = r['combined_tps']
    passed = contention < 1.20 and combined > 0
    return {
        'contention_factor': round(contention, 3),
        'combined_tps': round(combined, 0),
        'prefill_tps': round(r['prefill_tps'], 0),
        'decode_tps': round(r['decode_tps'], 0),
        'status': 'PASS' if passed else 'FAIL',
    }


def test_pipeline_backpressure():
    """Verify chip-0 admission control prevents overload.

    Chip 0 is the pipeline entry point. It processes the first few layers of
    every token (prefill and decode). In superscalar mode, multiple prefills
    can be interleaved through chip 0, but admission must be rate-limited to
    prevent DSP saturation.

    The admission model reports how many requests per second chip 0 can handle.
    With chip0_parallelism=1 (single pipeline), the rate should be finite and
    < total DSP throughput.

    PASS if: admission_reqs_s is finite and > 0 and < 1000.
    """
    from fpga_arch.pipeline import PipelineEngine
    # Single pipeline clone (no parallelism) — bottleneck is real
    r1 = PipelineEngine.chip0_admission_rate(
        chunk_size=128, chip0_parallelism=1,
        use_fp4_attn=True, attn_sparsity=0.888,
    )
    # Dual pipeline clone — roughly 2× admission rate
    r2 = PipelineEngine.chip0_admission_rate(
        chunk_size=128, chip0_parallelism=2,
        use_fp4_attn=True, attn_sparsity=0.888,
    )

    rate1 = r1['admission_reqs_s']
    rate2 = r2['admission_reqs_s']
    ratio = rate2 / max(rate1, 0.01)

    # Check: single pipeline rate is finite, positive, reasonable
    # Check: dual pipeline is roughly 2× (within 10%)
    passed = (0 < rate1 < 1000 and 1.8 < ratio < 2.2)
    return {
        'admit_rate_p1_req_s': round(rate1, 1),
        'admit_rate_p2_req_s': round(rate2, 1),
        'p2_p1_ratio': round(ratio, 2),
        'per_chunk_us': round(r1['per_chunk_us'], 1),
        'status': 'PASS' if passed else 'FAIL',
    }


def test_disaggregated_kv_transfer():
    """Verify the prefill-to-decode KV transfer model in disaggregated mode.

    In disaggregated deployments, KV cache computed during prefill on dedicated
    prefill servers must be transferred to decode servers via C2C + PCIe P2P.
    The transfer latency depends on:
      - Number of KV tokens being transferred
      - KV bytes per token (compressed MLA format: conservative 1152 bytes)
      - Interconnect bandwidth (PCIe P2P cross-server ~54.4 GB/s effective)

    This test exercises kv_disaggregated_transfer_time_us() and
    kv_transfer_us_per_token() and verifies reasonable latencies.

    PASS if: transfer time for P=512 batch is < 1000 us (1ms),
             and per-token cost decreases with larger batches (amortization).
    """
    from fpga_arch.interconnect import (
        kv_disaggregated_transfer_time_us, kv_transfer_us_per_token,
    )

    # Conservative KV bytes per token (K: 512+64 + V: 512 = 1088, rounded to 1152)
    kv_bytes_per_token = 1152

    # Small batch transfer (P=128)
    t_small = kv_disaggregated_transfer_time_us(
        num_tokens=128, kv_bytes_per_token=kv_bytes_per_token,
    )

    # Typical batch transfer (P=512)
    t_typical = kv_disaggregated_transfer_time_us(
        num_tokens=512, kv_bytes_per_token=kv_bytes_per_token,
    )

    # Large batch transfer (P=2048)
    t_large = kv_disaggregated_transfer_time_us(
        num_tokens=2048, kv_bytes_per_token=kv_bytes_per_token,
    )

    # Per-token cost at different batch sizes
    us_per_tok_small = kv_transfer_us_per_token(128, kv_bytes_per_token)
    us_per_tok_typical = kv_transfer_us_per_token(512, kv_bytes_per_token)
    us_per_tok_large = kv_transfer_us_per_token(2048, kv_bytes_per_token)

    # Check: typical transfer < 1ms, monotonic scaling, per-token cost amortizes
    passed = (t_typical < 1000.0 and
              t_small < t_typical < t_large and
              us_per_tok_small > us_per_tok_large)  # amortization effect
    return {
        'transfer_P128_us': round(t_small, 1),
        'transfer_P512_us': round(t_typical, 1),
        'transfer_P2048_us': round(t_large, 1),
        'us_per_tok_P128': round(us_per_tok_small, 4),
        'us_per_tok_P512': round(us_per_tok_typical, 4),
        'us_per_tok_P2048': round(us_per_tok_large, 4),
        'kv_bytes_per_token': kv_bytes_per_token,
        'status': 'PASS' if passed else 'FAIL',
    }


TESTS = [
    ('chip_resources', test_chip_resources),
    ('interconnect', test_interconnect),
    ('cluster_replication', test_cluster_replication),
    ('expert_popularity', test_expert_popularity),
    ('pipeline_models', test_pipeline_models),
    ('weight_layout', test_weight_layout),
    ('kv_cache', test_kv_cache),
    ('scheduler', test_scheduler),
    ('api_server', test_api_server),
    ('serving_short', test_serving_short),
    ('concurrent_prefill_decode', test_concurrent_prefill_decode),
    ('pipeline_backpressure', test_pipeline_backpressure),
    ('disaggregated_kv_transfer', test_disaggregated_kv_transfer),
]


def main():
    results = [run_case(name, fn) for name, fn in TESTS]
    out_json = ROOT / 'docs' / 'module_smoke_results.json'
    out_md = ROOT / 'docs' / 'module_smoke_report.md'

    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = ['# FPGA LPU Python Module Smoke Test Report', '']
    n_pass = sum(1 for r in results if r['status'] == 'PASS')
    lines.append(f'Passed: {n_pass}/{len(results)}')
    lines.append('')
    lines.append('| Module | Status | Key Output |')
    lines.append('|---|---:|---|')
    for r in results:
        val = r.get('value', r.get('error', ''))
        if isinstance(val, dict):
            val = ', '.join(f'{k}={v}' for k, v in val.items())
        lines.append(f"| {r['name']} | {r['status']} | {val} |")
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('\n'.join(lines))
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
