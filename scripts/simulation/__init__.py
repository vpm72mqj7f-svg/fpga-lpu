"""
DeepSeek V4 Pro FPGA Inference — Python Functional Simulation.

Before the FPGA development board arrives, validate:
  1. fp4 precision vs BF16 baseline
  2. HBM bandwidth under MoE access patterns
  3. Single-layer latency estimation

Usage:
  python run_all.py            — all 3 experiments
  python experiment_1_fp4_precision.py
  python experiment_2_hbm_bandwidth.py
  python experiment_3_layer_latency.py
"""
