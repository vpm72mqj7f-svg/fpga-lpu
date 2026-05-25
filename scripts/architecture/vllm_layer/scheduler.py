"""
Scheduler — Continuous Batching 调度器 (vLLM 核心).

每 iteration:
  1. 从等待队列选请求 (不超过 max_num_seqs, 不超过 max_batch_tokens)
  2. 新请求 → prefill (一次性处理整个 prompt)
  3. 进行中请求 → decode (每步生成 1 token)
  4. 完成的请求移出

这是 vLLM 的 iteration-level scheduler 的简化实现.
与 FPGA 无关 — 100% 复用 vLLM 逻辑.
"""

from dataclasses import dataclass, field
from enum import Enum
from .. import config
from .api_server import APIServer, GenerationRequest, RequestState
from .kv_cache import KVManager, KVBlockTable


class StepType(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class ScheduledBatch:
    """一个调度 batch 的内容."""
    step_type: StepType
    requests: list[GenerationRequest]     # 本 batch 的请求
    num_tokens: int                       # 总 token 数 (prefill=prompt长度, decode=请求数)
    seq_ids: list[int]                    # 序列 ID 列表

    # KV cache 信息
    kv_blocks_to_swap_in: list[int] = field(default_factory=list)
    kv_blocks_to_allocate: int = 0

    # Expert routing (模拟: 从 MoE Router 获取的 expert 分配)
    # 实际由 Model Runner 在推理时获取, 此处预留
    expert_ids_per_layer: list[list[int]] = field(default_factory=list)


class Scheduler:
    """Continuous Batching 调度器.

    逻辑完全复用 vLLM, 不依赖 FPGA 硬件.
    """

    def __init__(self, api_server: APIServer, kv_manager: KVManager):
        self.api = api_server
        self.kv = kv_manager

        self.max_seqs = config.SW_MAX_NUM_SEQS
        self.max_tokens = config.SW_MAX_BATCH_TOKENS
        self.block_size = config.SW_BLOCK_SIZE

        # 运行中的序列
        self.running: dict[int, GenerationRequest] = {}
        self._next_seq_id = 0

        # 统计
        self.total_steps = 0
        self.total_prefill_steps = 0
        self.total_decode_steps = 0

    def schedule(self) -> ScheduledBatch | None:
        """一次调度决策. 返回要执行的 batch, 或 None 表示空闲."""
        self.total_steps += 1

        # ── Phase 1: 等待队列中是否有新请求? ──
        waiting = self.api.waiting
        can_add = self.max_seqs - len(self.running)

        new_requests = []
        if waiting and can_add > 0:
            # 取尽量多的新请求
            for req in waiting[:can_add]:
                if len(req.prompt_tokens) <= self.max_tokens:
                    new_requests.append(req)
                    req.state = RequestState.PREFILL
                    self.running[self._next_seq_id] = req
                    self._next_seq_id += 1

        # ── Phase 2: 决定是 prefill 还是 decode step ──
        # 有 prefill 请求? → prefill step (可以合并多个 prefill)
        prefill_reqs = [r for r in self.running.values()
                        if r.state == RequestState.PREFILL]

        if prefill_reqs:
            # Prefill: 处理新请求的 prompt
            total_tokens = sum(len(r.prompt_tokens) for r in prefill_reqs)
            # Token 数不能超过 max_tokens
            if total_tokens > self.max_tokens:
                # 分批: 只取前几个请求
                prefill_reqs = prefill_reqs[:max(1, self.max_tokens // max(
                    len(r.prompt_tokens) for r in prefill_reqs))]
                total_tokens = sum(len(r.prompt_tokens) for r in prefill_reqs)

            self.total_prefill_steps += 1
            return ScheduledBatch(
                step_type=StepType.PREFILL,
                requests=prefill_reqs,
                num_tokens=total_tokens,
                seq_ids=[sid for sid, r in self.running.items()
                         if r in prefill_reqs],
            )

        # ── Phase 3: Decode step ──
        decode_reqs = [r for r in self.running.values()
                       if r.state == RequestState.DECODE]

        if not decode_reqs:
            # 将 prefill 完成的请求转为 decode
            # (实际在 Model Runner 完成后转换, 此处处理边界)
            return None

        self.total_decode_steps += 1
        return ScheduledBatch(
            step_type=StepType.DECODE,
            requests=decode_reqs,
            num_tokens=len(decode_reqs),  # decode: 每请求 1 token
            seq_ids=[sid for sid, r in self.running.items()
                     if r in decode_reqs],
        )

    def mark_step_complete(self, batch: ScheduledBatch, new_tokens: list[int]):
        """标记一个调度步完成, 更新请求状态.

        new_tokens: 每个请求新生成的 token (prefill 时为空, decode 时有 1 个)
        """
        import time

        for i, req in enumerate(batch.requests):
            if batch.step_type == StepType.PREFILL:
                # Prefill 完成 → 转为 decode
                req.state = RequestState.DECODE
                req.first_token_time = time.time()
            elif batch.step_type == StepType.DECODE:
                # Decode 完成 → 记录新 token
                if i < len(new_tokens):
                    req.output_tokens.append(new_tokens[i])

        # 检查完成的请求
        for sid, req in list(self.running.items()):
            if req.is_finished:
                req.state = RequestState.FINISHED
                req.finish_time = time.time()
                self.api.completed.append(req)
                del self.running[sid]

    @property
    def stats(self) -> dict:
        return {
            'total_steps': self.total_steps,
            'prefill_steps': self.total_prefill_steps,
            'decode_steps': self.total_decode_steps,
            'running_seqs': len(self.running),
            **self.api.stats,
        }
