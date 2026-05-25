"""
API Server — OpenAI 兼容接口 (模拟).

接收 /v1/chat/completions 请求, 转化为内部 Request 对象, 放入调度队列.
与 vLLM 的 api_server.py 接口一致, 此处只保留核心数据流.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
import uuid


class RequestState(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class GenerationRequest:
    """一个推理请求."""
    request_id: str
    prompt: str
    prompt_tokens: list[int]        # tokenized prompt
    max_tokens: int = 256
    temperature: float = 0.7

    # 状态
    state: RequestState = RequestState.WAITING
    output_tokens: list[int] = field(default_factory=list)
    arrival_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0

    @property
    def is_finished(self) -> bool:
        return (len(self.output_tokens) >= self.max_tokens or
                self.state == RequestState.FINISHED)

    @property
    def ttft_ms(self) -> float:
        """Time To First Token (ms)."""
        if self.first_token_time > 0:
            return (self.first_token_time - self.arrival_time) * 1000
        return 0.0


class APIServer:
    """模拟 vLLM API Server.

    只做两件事:
      1. 接收请求, tokenize, 放入队列
      2. 返回完成结果
    """

    def __init__(self):
        self.requests: list[GenerationRequest] = []
        self.completed: list[GenerationRequest] = []
        self._token_counter = 0

    def submit(self, prompt: str, max_tokens: int = 256) -> GenerationRequest:
        """提交请求. 模拟 tokenize 过程."""
        # 伪 tokenize: 按空格分词, 每词 = 1 token
        words = prompt.split()
        prompt_tokens = list(range(self._token_counter,
                                   self._token_counter + len(words)))
        self._token_counter += len(words)

        req = GenerationRequest(
            request_id=str(uuid.uuid4())[:8],
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            arrival_time=time.time(),
        )
        self.requests.append(req)
        return req

    def poll_completed(self) -> list[GenerationRequest]:
        """获取已完成的请求."""
        done = [r for r in self.completed]
        self.completed.clear()
        return done

    @property
    def waiting(self) -> list[GenerationRequest]:
        return [r for r in self.requests if r.state == RequestState.WAITING]

    @property
    def active(self) -> list[GenerationRequest]:
        return [r for r in self.requests
                if r.state in (RequestState.PREFILL, RequestState.DECODE)]

    @property
    def stats(self) -> dict:
        active = self.active
        waiting = self.waiting
        done = self.completed
        return {
            'waiting': len(waiting),
            'active': len(active),
            'completed': len(done),
            'prefill': len([r for r in active if r.state == RequestState.PREFILL]),
            'decode': len([r for r in active if r.state == RequestState.DECODE]),
        }
