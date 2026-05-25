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
