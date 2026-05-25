# FPGA LPU Python Module Smoke Test Report

Passed: 10/10

| Module | Status | Key Output |
|---|---:|---|
| chip_resources | PASS | dsp_1G_mac_us=90.334, hbm_920MB_us=1024.0, chip_weight_gb=0.565 |
| interconnect | PASS | c2c_7KB_us=0.551, pcie_7KB_us=0.896 |
| cluster_replication | PASS | baseline_experts_chip0=12, hot_min_avg_max=[12, 14.3, 16], hottest_expert_replicas=8 |
| expert_popularity | PASS | top20_mass=0.551, top77_mass=0.755, top_replica_count=8 |
| pipeline_models | PASS | k_pipeline=23.08, decode_tps_B1=724.3, decode_tps_B8=4489.7, prefill_chip0_us=13480.3, chip0_rate_clone2_req_s=37.1 |
| weight_layout | PASS | max_used_gb=23.29, min_free_gb=8.71, total_weight_gb=744.5 |
| kv_cache | PASS | prefill_blocks=40, after_decode_blocks=40, allocated_before_free=32, after_free=0 |
| scheduler | PASS | num_batches=1, first_batch_type=PREFILL, batch_size=4 |
| api_server | PASS | generated=6, first_prompt=181 |
| serving_short | PASS | requests=49, finished=44, accept_rate=89.8, output_tps=402.3, ttft_p95_ms=523.9 |
