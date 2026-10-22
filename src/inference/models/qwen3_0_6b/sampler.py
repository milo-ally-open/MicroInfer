"""Qwen3 greedy／temperature sampling 與 logits processor。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class Qwen3Sampler:
    """為各 request 獨立套用重複懲罰與取樣選項。"""

    def sample(
        self,
        logits: torch.Tensor,
        *,
        prompt_tokens: Sequence[int],
        generated_tokens: Sequence[int],
        options: dict[str, Any],
    ) -> int:
        scores = logits.float().clone()
        seen = set(prompt_tokens) | set(generated_tokens)
        penalty = float(options.get("repetition_penalty", 1.0))
        if penalty != 1.0 and seen:
            indices = torch.tensor(sorted(seen), dtype=torch.long, device=scores.device)
            selected = scores[indices]
            scores[indices] = torch.where(selected < 0, selected * penalty, selected / penalty)
        self._apply_no_repeat_ngram(
            scores,
            [*prompt_tokens, *generated_tokens],
            int(options.get("no_repeat_ngram_size", 0)),
        )
        temperature = float(options.get("temperature", 0.0))
        if temperature <= 0:
            return int(torch.argmax(scores).item())
        scores /= temperature
        top_k = int(options.get("top_k", 0))
        if top_k > 0 and top_k < scores.numel():
            cutoff = torch.topk(scores, top_k).values[-1]
            scores[scores < cutoff] = -torch.inf
        probabilities = torch.softmax(scores, dim=-1)
        top_p = float(options.get("top_p", 1.0))
        if 0 < top_p < 1:
            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            remove = torch.cumsum(sorted_probabilities, dim=-1) > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            probabilities[sorted_indices[remove]] = 0
        min_p = options.get("min_p")
        if min_p is not None and float(min_p) > 0:
            probabilities[probabilities < probabilities.max() * float(min_p)] = 0
        total = probabilities.sum()
        if not torch.isfinite(total) or total <= 0:
            return int(torch.argmax(scores).item())
        return int(torch.multinomial(probabilities / total, num_samples=1).item())

    @staticmethod
    def _apply_no_repeat_ngram(scores: torch.Tensor, tokens: list[int], ngram_size: int) -> None:
        if ngram_size < 1 or len(tokens) < ngram_size - 1:
            return
        prefix = tuple(tokens[-(ngram_size - 1):]) if ngram_size > 1 else ()
        banned = {
            tokens[index + ngram_size - 1]
            for index in range(len(tokens) - ngram_size + 1)
            if ngram_size == 1 or tuple(tokens[index:index + ngram_size - 1]) == prefix
        }
        if banned:
            scores[list(banned)] = -torch.inf

