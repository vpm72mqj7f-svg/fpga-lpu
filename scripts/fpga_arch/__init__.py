"""
fpga_arch — FPGA hardware architecture model for the 32-chip inference cluster.

Exports:
  - config: unified hardware constants (single source of truth)
  - chip: FPGAChip with SRAMBank, HBMBank, DSPArray, KV block management
  - interconnect: C2CDualRing + PCIeFabric
  - cluster: FPGACluster (32-chip assembly, layer/expert assignment, weight placement)
  - pipeline: 10-stage PipelineEngine with execute_batch() and throughput_model()
"""

from .config import *
from .chip import FPGAChip, SRAMBank, HBMBank, DSPArray, KVBlock
from .interconnect import C2CDualRing, PCIeFabric, C2CMessageType, C2CMessage
from .cluster import FPGACluster, ClusterStats
from .expert_popularity import ExpertPopularity
from .pipeline import (
    PipelineEngine, PipelineStage,
    WeightPrefetchStage, MLAAttentionStage, AttnNormStage,
    MoERouterStage, MoEDispatchStage, SharedExpertStage,
    RoutedExpertStage, MoEReduceStage, FFNNormStage,
    PipelineForwardStage,
    StageType, StageTiming, LayerTiming, TokenTrace, BatchResult,
    MACBreakdown, C2CContentionModel, ExpertHitPath,
    simulate_pipeline, detailed_layer_timing, detailed_stage_timing,
    print_pipeline_result, PipelineSimResult,
)
