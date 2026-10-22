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
    from engine.contracts import ExecutionPhase, ScheduledWorkItem
    from models.mage_vl_4b.components.stepwise_generation import MageVLStepwiseGeneration
    from models.mage_vl_4b.components.multimodal_model import build_position_ids


@unittest.skipIf(torch is None, "PyTorch is only installed in the model runtime environment.")
class StepwiseBatchTests(unittest.TestCase):
    def test_incremental_position_ids_match_current_query_length(self) -> None:
        decode_embeds = torch.zeros((1, 1, 16))
        full_attention_mask = torch.ones((1, 117), dtype=torch.long)

        decode_positions = build_position_ids(decode_embeds, full_attention_mask)

        self.assertEqual(tuple(decode_positions.shape), (1, 1))
        self.assertEqual(decode_positions.tolist(), [[116]])

        chunk_embeds = torch.zeros((1, 3, 16))
        chunk_positions = build_position_ids(chunk_embeds, full_attention_mask)
        self.assertEqual(chunk_positions.tolist(), [[114, 115, 116]])

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
            self.sequence_lengths: list[int] = []

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
                "pixel_values": kwargs.get("pixel_values"),
            }

        def __call__(
            self, input_ids, attention_mask, past_key_values=None, use_cache=True,
            return_dict=True, pixel_values=None,
        ):
            self.forward_calls += 1
            self.batch_sizes.append(int(input_ids.shape[0]))
            batch_size, sequence_length = input_ids.shape
            self.sequence_lengths.append(sequence_length)
            logits = torch.full((batch_size, sequence_length, 128), -100.0)
            next_ids = (input_ids[:, -1] + 10) % 128
            logits[torch.arange(batch_size), -1, next_ids] = 100.0
            current = input_ids[:, None, :, None].float()
            if pixel_values is not None and past_key_values is None:
                visual_positions = torch.zeros((batch_size, 1, 2, 1), dtype=current.dtype)
                current = torch.cat([current, visual_positions], dim=-2)
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

        prefill = self.runner.execute_work_batch((
            ScheduledWorkItem("a", ExecutionPhase.PREFILL, 2),
            ScheduledWorkItem("b", ExecutionPhase.PREFILL, 2),
        ))
        self.assertEqual(self.model.forward_calls, 1)
        self.assertEqual(self.model.batch_sizes, [2])
        self.assertEqual([item["text"] for item in prefill], ["12", "14"])
        self.assertEqual(self.runner.cache_manager.length("a"), 2)
        self.assertEqual(self.runner.cache_manager.length("b"), 2)

        decoded = self.runner.execute_work_batch((
            ScheduledWorkItem("a", ExecutionPhase.DECODE, 1),
            ScheduledWorkItem("b", ExecutionPhase.DECODE, 1),
        ))
        self.assertEqual(self.model.forward_calls, 2)
        self.assertEqual(self.model.batch_sizes, [2, 2])
        self.assertEqual([item["text"] for item in decoded], ["22", "24"])
        self.assertEqual(self.runner.cache_manager.length("a"), 3)
        self.assertEqual(self.runner.cache_manager.length("b"), 3)

    def test_mismatched_prefill_lengths_use_safe_individual_forwards(self) -> None:
        self._begin("long", [1, 2, 3])
        self._begin("short", [4, 5])

        results = self.runner.execute_work_batch((
            ScheduledWorkItem("long", ExecutionPhase.PREFILL, 3),
            ScheduledWorkItem("short", ExecutionPhase.PREFILL, 2),
        ))
        self.assertEqual(self.model.forward_calls, 2)
        self.assertEqual(self.model.batch_sizes, [1, 1])
        self.assertEqual([item["text"] for item in results], ["13", "15"])
        self.assertEqual(self.runner.cache_manager.length("long"), 3)
        self.assertEqual(self.runner.cache_manager.length("short"), 2)

    def test_chunked_prefill_forwards_only_the_scheduled_input_slice(self) -> None:
        self._begin("chunked", [1, 2, 3, 4])

        first = self.runner.execute_work_batch((
            ScheduledWorkItem("chunked", ExecutionPhase.PREFILL, 2),
        ))
        self.assertEqual(first[0]["prompt_tokens_processed"], 2)
        self.assertEqual(self.runner.cache_manager.length("chunked"), 2)

        second = self.runner.execute_work_batch((
            ScheduledWorkItem("chunked", ExecutionPhase.PREFILL, 2, prompt_offset=2),
        ))
        self.assertEqual(second[0]["prompt_tokens_processed"], 4)
        self.assertEqual(self.model.sequence_lengths, [2, 2])
        self.assertEqual(self.runner.cache_manager.length("chunked"), 4)

    def test_video_prefill_is_indivisible_and_tracks_expanded_cache_length(self) -> None:
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        self.runner.begin("video", {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.ones((1, 1)),
        }, {"max_new_tokens": 3, "use_cache": True})

        self.assertEqual(self.runner.prefill_min_chunk_tokens("video"), 4)
        result = self.runner.execute_work_batch((
            ScheduledWorkItem("video", ExecutionPhase.PREFILL, 4),
        ))
        self.assertEqual(result[0]["prompt_tokens_processed"], 4)
        self.assertEqual(self.runner.cache_manager.length("video"), 6)

        decoded = self.runner.execute_work_batch((
            ScheduledWorkItem("video", ExecutionPhase.DECODE, 1),
        ))
        self.assertEqual(decoded[0]["generated_tokens"], 1)
        self.assertEqual(self.model.sequence_lengths, [4, 1])
        self.assertEqual(self.runner.cache_manager.length("video"), 7)


if __name__ == "__main__":
    unittest.main()
