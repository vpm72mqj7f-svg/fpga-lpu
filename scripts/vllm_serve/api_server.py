"""
vllm_serve/api_server.py — Request generator with Poisson arrival process.

Generates synthetic inference requests for simulation.
Models DeepSeek V4 Pro workload characteristics.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np

from .types import Request
from .config import (
    PROMPT_LEN_MEAN, PROMPT_LEN_MIN, PROMPT_LEN_MAX,
    OUTPUT_LEN_MEAN, OUTPUT_LEN_MAX,
)


class RequestGenerator:
    """Generates synthetic inference requests with configurable distributions.

    Models:
      - Arrival: Poisson process (configurable rate λ)
      - Prompt length: truncated normal (mean 512, min 16, max 4096)
      - Output length: truncated normal (mean 256, min 1, max 2048)
    """

    def __init__(self, seed: int = 42,
                 prompt_len_mean: int = PROMPT_LEN_MEAN,
                 output_len_mean: int = OUTPUT_LEN_MEAN):
        self.rng = np.random.RandomState(seed)
        self._request_counter = 0
        self.prompt_len_mean = prompt_len_mean
        self.output_len_mean = output_len_mean

        # Arrival statistics
        self.total_generated = 0

    def generate_arrivals(self, arrival_rate: float,
                          duration_us: float) -> List[Request]:
        """Generate requests that arrive during [0, duration_us].

        Args:
            arrival_rate: mean requests per second (Poisson λ)
            duration_us: time window in microseconds

        Returns list of Request objects with arrival_time_us set.
        """
        duration_s = duration_us / 1e6
        # Poisson: number of arrivals in duration
        n_arrivals = self.rng.poisson(arrival_rate * duration_s)

        # Uniform arrival times within window
        arrival_times = self.rng.uniform(0, duration_us, n_arrivals)
        arrival_times.sort()

        requests = []
        for arr_time in arrival_times:
            req = self._generate_request(arrival_time_us=arr_time)
            requests.append(req)

        self.total_generated += len(requests)
        return requests

    def _generate_request(self, arrival_time_us: float) -> Request:
        """Generate a single request with random prompt/output lengths."""
        prompt_len = int(np.clip(
            self.rng.normal(self.prompt_len_mean, self.prompt_len_mean * 0.5),
            PROMPT_LEN_MIN, PROMPT_LEN_MAX
        ))
        max_output = int(np.clip(
            self.rng.normal(self.output_len_mean, self.output_len_mean * 0.5),
            1, OUTPUT_LEN_MAX
        ))

        req = Request(
            request_id=self._request_counter,
            arrival_time_us=arrival_time_us,
            prompt_len=prompt_len,
            max_output_len=max_output,
        )
        self._request_counter += 1
        return req

    def reset(self):
        """Reset request counter (for repeated simulations)."""
        self._request_counter = 0
        self.total_generated = 0


class APIServer:
    """Simulated API server frontend.

    Wraps RequestGenerator with submission to scheduler.
    Models an OpenAI-compatible API endpoint.
    """

    def __init__(self, scheduler, seed: int = 42,
                 prompt_len_mean: int = PROMPT_LEN_MEAN,
                 output_len_mean: int = OUTPUT_LEN_MEAN):
        self.scheduler = scheduler
        self.generator = RequestGenerator(seed=seed,
            prompt_len_mean=prompt_len_mean,
            output_len_mean=output_len_mean)

    def generate_and_submit(self, arrival_rate: float,
                            duration_us: float) -> List[Request]:
        """Generate requests for a time window and submit to scheduler."""
        requests = self.generator.generate_arrivals(arrival_rate, duration_us)
        for req in requests:
            self.scheduler.submit_request(req)
        return requests

    def submit_manual(self, prompt_len: int, max_output: int,
                      arrival_time_us: float,
                      output_len_mean: int = OUTPUT_LEN_MEAN) -> Request:
        """Manually submit a single request (for testing)."""
        req = Request(
            request_id=self.generator._request_counter,
            arrival_time_us=arrival_time_us,
            prompt_len=prompt_len,
            max_output_len=max_output,
        )
        self.generator._request_counter += 1
        self.scheduler.submit_request(req)
        return req
