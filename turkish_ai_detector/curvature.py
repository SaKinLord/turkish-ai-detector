"""Language-model curvature scoring.

This module owns the expensive model interaction. The rest of the package can
be tested without loading a model by passing a lightweight scorer object into
the public pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from . import defaults


@dataclass(frozen=True)
class ModelRuntime:
    model_name: str = defaults.MODEL_NAME
    token_limit: int = defaults.TOKEN_LIMIT
    samples: int = defaults.TOKEN_SAMPLE_COUNT
    sampling_seed: int | None = defaults.TOKEN_SAMPLING_SEED
    trust_remote_code: bool = True


class CurvatureScorer:
    """Compute Fast-DetectGPT-style scores for text segments."""

    def __init__(self, runtime: ModelRuntime | None = None, lazy: bool = True) -> None:
        self.runtime = runtime or ModelRuntime()
        self.tokenizer: Any = None
        self.model: Any = None
        if not lazy:
            self.load()

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.runtime.model_name,
            trust_remote_code=self.runtime.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.runtime.model_name,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=self.runtime.trust_remote_code,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _sampling_generator(self, torch: Any, device: Any, text: str) -> Any:
        if self.runtime.sampling_seed is None:
            return None

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        text_seed = int.from_bytes(digest[:8], "big")
        seed = (int(self.runtime.sampling_seed) + text_seed) % (2**63 - 1)
        try:
            generator = torch.Generator(device=device)
        except (RuntimeError, TypeError):
            generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def score(self, text: str) -> float:
        import torch

        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.runtime.token_limit,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}

        input_ids = encoded["input_ids"]
        if input_ids.shape[-1] < 2:
            return 0.0

        with torch.no_grad():
            output = self.model(**encoded)
            logits = output.logits[:, :-1, :]

        target_ids = input_ids[:, 1:]
        log_distribution = torch.nn.functional.log_softmax(logits, dim=-1)
        original_log_values = log_distribution.gather(
            -1,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)
        original_average = original_log_values.mean().item()

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
        generator = self._sampling_generator(torch, device, text)

        sampled_averages: list[float] = []
        for _ in range(self.runtime.samples):
            sampled_ids = torch.multinomial(
                flat_probabilities,
                1,
                generator=generator,
            ).reshape(target_ids.shape)
            sampled_log_values = log_distribution.gather(
                -1,
                sampled_ids.unsqueeze(-1),
            ).squeeze(-1)
            sampled_averages.append(sampled_log_values.mean().item())

        return original_average - (sum(sampled_averages) / len(sampled_averages))
