"""以 CPU 假模型驗證 Mage-VL Stepwise Runner 的批次 Cache 邏輯。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:  # 模型環境提供 PyTorch；無此相依時保留無權重測試套件可執行。
    torch = None

if torch is not None:
    import sys
    from pathlib import Path

    INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "inference"
    sys.path.insert(0, str(INFERENCE_ROOT))
    from models.mage_vl_4b.components.stepwise_generation import MageVLStepwiseGeneration


@unittest.skipIf(torch is None, "PyTorch is only installed in the model runtime environment.")
class StepwiseBatchTests(unittest.TestCase):
    class FakeCache:
        def __init__(self, ddp_cache_data=None) -> None:
            self.layers = [SimpleNamespace(keys=key, values=value) for key, value in ddp_cache_data or ()]

        def get_seq_length(self) -> int:
            return int(self.layers[0].keys.shape[-2])

    class FakeModel:
        def __init__(self) -> None:
            self.generation_config = SimpleNamespace(eos_token_id=99)
            self.config = SimpleNamespace(text_config=None)
            self.forward_calls = 0
            self.batch_sizes: list[int] = []

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            use_cache=True,
            cache_position=None,
            is_first_iteration=False,
            **kwargs,
        ):
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }

        def __call__(self, input_ids, attention_mask, past_key_values=None, use_cache=True, return_dict=True):
            self.forward_calls += 1
            self.batch_sizes.append(int(input_ids.shape[0]))
            batch_size, sequence_length = input_ids.shape
            logits = torch.full((batch_size, sequence_length, 128), -100.0)
            next_ids = (input_ids[:, -1] + 10) % 128
            logits[torch.arange(batch_size), -1, next_ids] = 100.0
            current = input_ids[:, None, :, None].float()
            if past_key_values is not None:
                previous = past_key_values.layers[0].keys
                current = torch.cat([previous, current], dim=-2)
            cache = StepwiseBatchTests.FakeCache([(current, current.clone())])
            return SimpleNamespace(logits=logits, past_key_values=cache)

    class FakeProcessor:
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=99)

        class Tokenizer:
            @staticmethod
            def decode(token_ids, **kwargs):
                return ",".join(str(token_id) for token_id in token_ids)

        tokenizer = Tokenizer()

    def setUp(self) -> None:
        self.model = self.FakeModel()
        self.runner = MageVLStepwiseGeneration()
        self.runner.bind(self.model, self.FakeProcessor())

    def _begin(self, request_id: str, token_ids: list[int]) -> None:
        ids = torch.tensor([token_ids], dtype=torch.long)
        self.runner.begin(request_id, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, {
            "max_new_tokens": 3,
            "use_cache": True,
        })

    def test_compatible_prefill_and_decode_each_use_one_forward(self) -> None:
        self._begin("a", [1, 2])
        self._begin("b", [3, 4])

        prefill = self.runner.execute_batch((("a", True), ("b", True)))
        self.assertEqual(self.model.forward_calls, 1)
        self.assertEqual(self.model.batch_sizes, [2])
        self.assertEqual([item["text"] for item in prefill], ["12", "14"])
        self.assertEqual(self.runner._cache_length(self.runner._state("a")["past_key_values"]), 2)
        self.assertEqual(self.runner._cache_length(self.runner._state("b")["past_key_values"]), 2)

        decoded = self.runner.execute_batch((("a", False), ("b", False)))
        self.assertEqual(self.model.forward_calls, 2)
        self.assertEqual(self.model.batch_sizes, [2, 2])
        self.assertEqual([item["text"] for item in decoded], ["22", "24"])
        self.assertEqual(self.runner._cache_length(self.runner._state("a")["past_key_values"]), 3)
        self.assertEqual(self.runner._cache_length(self.runner._state("b")["past_key_values"]), 3)

    def test_prefill_removes_left_padding_from_each_cache(self) -> None:
        self._begin("long", [1, 2, 3])
        self._begin("short", [4, 5])

        results = self.runner.execute_batch((("long", True), ("short", True)))
        self.assertEqual(self.model.forward_calls, 1)
        self.assertEqual(self.model.batch_sizes, [2])
        self.assertEqual([item["text"] for item in results], ["13", "15"])
        self.assertEqual(self.runner._cache_length(self.runner._state("long")["past_key_values"]), 3)
        self.assertEqual(self.runner._cache_length(self.runner._state("short")["past_key_values"]), 2)
        self.assertEqual(self.runner._state("short")["model_inputs"]["attention_mask"].tolist(), [[1, 1]])


if __name__ == "__main__":
    unittest.main()
