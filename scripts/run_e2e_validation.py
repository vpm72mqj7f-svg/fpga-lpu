"""End-to-end validation: 18 configs × workloads.

Configs (cumulative stack):
  baseline:  KV=4096, no microbatch, no replication, no cloning
  +D:        KV=22528 (default in code now)
  +D+C:      + --microbatch (but it's a no-op for autoregressive decode, so really:
                              just min_decode_batch removal which is implicit when
                              we don't override; baseline path uses old defaults)
  +D+C+A:    + --expert-replication hot
  +D+C+A+PC2: + --pipeline-clone 2
  +D+C+A+PC4: + --pipeline-clone 4

Workloads:
  chat:    arrival=2, prompt=512, output=256, non-agent
  agent:   arrival=4, agent, P_init~512, delta=256, output=512/turn, 10 turns
  burst:   arrival=20, prompt=1024, output=1024, non-agent
"""
import subprocess
import json
import re
import sys
import os
import time

# Force baseline KV by overriding the constant via CLI (we changed default to 22528)
# We use --kv-blocks-per-chip to control this.

WORKLOADS = {
    'chat':   {'arrival': 2,  'output': 256,  'agent': False, 'prompt': 512},
    'agent':  {'arrival': 4,  'output': 512,  'agent': True,  'prompt': 512},
    'burst':  {'arrival': 20, 'output': 1024, 'agent': False, 'prompt': 1024},
}

CONFIGS = [
    {'name': 'baseline',  'kv': 4096,  'microbatch': False, 'rep': 'none', 'clone': 1},
    {'name': '+D',        'kv': 22528, 'microbatch': False, 'rep': 'none', 'clone': 1},
    {'name': '+D+C',      'kv': 22528, 'microbatch': True,  'rep': 'none', 'clone': 1},
    {'name': '+D+C+A',    'kv': 22528, 'microbatch': True,  'rep': 'hot',  'clone': 1},
    {'name': '+all+PC2',  'kv': 22528, 'microbatch': True,  'rep': 'hot',  'clone': 2},
    {'name': '+all+PC4',  'kv': 22528, 'microbatch': True,  'rep': 'hot',  'clone': 4},
]


def parse_metrics(stdout: str) -> dict:
    """Extract key metrics from run_serving stdout."""
    metrics = {}
    patterns = {
        'accept_rate':    r'Accept rate:\s+([\d.]+)%',
        'output_tps':     r'Output TPS:\s+([\d,.]+)\s+tok/s',
        'ttft_p50':       r'TTFT P50:\s+([\d,.]+)\s+ms',
        'ttft_p95':       r'TTFT P95:\s+([\d,.]+)\s+ms',
        'tpot_p50':       r'TPOT P50:\s+([\d.]+)\s+ms',
        'avg_active':     r'Avg active:\s+([\d.]+)',
        'avg_batch':      r'Avg batch size:\s+([\d.]+)',
        'prefill_admit':  r'Prefill admission:\s+([\d.]+)\s+req/s',
        'k_pipeline':     r'K_pipeline:\s+([\d.]+)',
        'ttft_sla':       r'TTFT SLA compliance:\s+([\d.]+)%',
    }
    for k, p in patterns.items():
        m = re.search(p, stdout)
        if m:
            try:
                metrics[k] = float(m.group(1).replace(',', ''))
            except ValueError:
                metrics[k] = None
    return metrics


def run_one(workload: str, cfg: dict, duration: int = 90) -> dict:
    w = WORKLOADS[workload]
    args = [
        sys.executable, 'scripts/run_serving.py',
        '--duration', str(duration),
        '--arrival-rate', str(w['arrival']),
        '--output-len-mean', str(w['output']),
        '--prompt-len-mean', str(w['prompt']),
        '--kv-blocks-per-chip', str(cfg['kv']),
        '--expert-replication', cfg['rep'],
        '--pipeline-clone', str(cfg['clone']),
    ]
    if w['agent']:
        args += ['--agent', '--agent-output-per-turn', str(w['output'])]
    if cfg['microbatch']:
        args += ['--microbatch']

    t0 = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=600)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {'error': proc.stderr[:200], 'elapsed_s': elapsed}
    m = parse_metrics(proc.stdout)
    m['elapsed_s'] = round(elapsed, 1)
    return m


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results = {}
    total = len(WORKLOADS) * len(CONFIGS)
    i = 0
    for w_name in WORKLOADS:
        for cfg in CONFIGS:
            i += 1
            key = f'{w_name} | {cfg["name"]}'
            print(f'[{i:>2}/{total}] {key}', flush=True)
            m = run_one(w_name, cfg, duration=90)
            results[key] = m
            print(f'         tps={m.get("output_tps","?")}, '
                  f'accept={m.get("accept_rate","?")}%, '
                  f'B={m.get("avg_batch","?")}, '
                  f'active={m.get("avg_active","?")}, '
                  f'TTFT_p95={m.get("ttft_p95","?")}ms '
                  f'({m.get("elapsed_s","?")}s)', flush=True)
    with open('docs/_e2e_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nResults saved to docs/_e2e_results.json')


if __name__ == '__main__':
    main()
