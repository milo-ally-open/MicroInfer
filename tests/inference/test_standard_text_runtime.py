"""驗證標準文字 Runner 的步進 Cache、取消、輸入處理與無權重預檢。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_ROOT = ROOT / "src" / "inference"
sys.path.insert(0, str(INFERENCE_ROOT))

import torch

from api.default_chat_processor import DefaultChatInputProcessor
from engine.contracts import ExecutionPhase, PreparedChatInput, ScheduledWorkItem
from engine.errors import InferenceError
from engine.standard_text import StandardTextModelAdapter, StandardTextModelRunner


class FakePlatform:
    name = "fake"

    def resolve_device(self, requested: str) -> torch.device:
        return torch.device("cpu")

    def gpu_status(self, device: torch.device) -> dict[str, object]:
        return {"device": str(device)}

    def peak_memory(self, device: torch.device) -> dict[str, int]:
        return {"peak_allocated": 0, "peak_reserved": 0}

    def reset_peak_memory(self, device: torch.device) -> None:
        return None

    def release_cached_memory(self) -> None:
        return None


class FakeCausalModel:
    def __init__(self, next_tokens: list[int]) -> None:
        self.config = SimpleNamespace(
            is_encoder_decoder=False,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=8,
            eos_token_id=9,
        )
        self.generation_config = SimpleNamespace(eos_token_id=9)
        self.next_tokens = list(next_tokens)
        self.calls: list[dict[str, object]] = []
        self.parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.parameter

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        attention_mask: torch.Tensor,
        past_key_values: list[int] | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        prior = list(past_key_values or [])
        self.calls.append(
            {
                "input_ids": input_ids.detach().cpu().reshape(-1).tolist(),
                "past": prior,
                "cache_position": cache_position.detach().cpu().tolist() if cache_position is not None else None,
                "attention_mask": attention_mask.detach().cpu().tolist(),
                "use_cache": use_cache,
                "return_dict": return_dict,
            }
        )
        cache = prior + input_ids.detach().cpu().reshape(-1).tolist()
        logits = torch.full((1, input_ids.shape[-1], 12), -10.0)
        logits[0, -1, self.next_tokens.pop(0)] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


class StandardTextRunnerTests(unittest.TestCase):
    def _runner(self, next_tokens: list[int]) -> tuple[StandardTextModelRunner, FakeCausalModel]:
        runner = StandardTextModelRunner(FakePlatform())
        runner.model = FakeCausalModel(next_tokens)
        runner.model_config = runner.model.config
        runner.device = torch.device("cpu")
        runner._forward_parameters = {"cache_position"}
        return runner, runner.model

    @staticmethod
    def _options(max_new_tokens: int = 2) -> dict[str, object]:
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "eos_token_ids": [9],
        }

    def test_batch_keeps_independent_caches_and_finishes_each_request(self) -> None:
        runner, model = self._runner([7, 8, 9, 10])
        for request_id, prompt in (("a", [1, 2]), ("b", [3, 4])):
            runner.begin_generation(request_id, {"input_ids": prompt}, self._options())
            runner.bind_generation_cache(request_id, (0,))

        prefill = runner.execute_batch(
            (
                ScheduledWorkItem("a", ExecutionPhase.PREFILL, 2),
                ScheduledWorkItem("b", ExecutionPhase.PREFILL, 2),
            )
        )
        self.assertEqual([result.token_id for result in prefill], [7, 8])
        decode = runner.execute_batch(
            (
                ScheduledWorkItem("a", ExecutionPhase.DECODE, 1),
                ScheduledWorkItem("b", ExecutionPhase.DECODE, 1),
            )
        )
        self.assertTrue(decode[0].finished)
        self.assertEqual(decode[0].finish_reason, "stop")
        self.assertTrue(decode[1].finished)
        self.assertEqual(decode[1].finish_reason, "length")
        self.assertNotEqual(model.calls[2]["past"], model.calls[3]["past"])
        self.assertEqual(runner.cache_snapshot()["active_requests"], 2)

        first_ids, first_reason = runner.finish_generation("a")
        second_ids, second_reason = runner.finish_generation("b")
        self.assertEqual(first_ids.tolist(), [7, 9])
        self.assertEqual(second_ids.tolist(), [8, 10])
        self.assertEqual((first_reason, second_reason), ("stop", "length"))
        self.assertEqual(runner.cache_snapshot()["active_requests"], 0)

    def test_chunked_prefill_advances_cache_positions_and_cancel_releases_state(self) -> None:
        runner, model = self._runner([4, 5])
        runner.begin_generation("chunked", {"input_ids": [1, 2, 3, 4]}, self._options(1))
        runner.bind_generation_cache("chunked", (0,))
        first = runner.execute_batch((ScheduledWorkItem("chunked", ExecutionPhase.PREFILL, 2),))[0]
        self.assertIsNone(first.token_id)
        self.assertEqual(first.prompt_tokens_processed, 2)
        second = runner.execute_batch(
            (ScheduledWorkItem("chunked", ExecutionPhase.PREFILL, 2, prompt_offset=2),)
        )[0]
        self.assertEqual(second.token_id, 5)
        self.assertEqual(model.calls[0]["cache_position"], [0, 1])
        self.assertEqual(model.calls[1]["cache_position"], [2, 3])
        runner.cancel_generation("chunked")
        self.assertEqual(runner.cache_snapshot(), {"active_requests": 0, "populated_caches": 0, "cached_tokens": 0})

    def test_default_adapter_preserves_token_ids_and_engine_lifecycle(self) -> None:
        runner, _ = self._runner([6])
        adapter = StandardTextModelAdapter(runner)
        prepared = PreparedChatInput("qwen3-0.6b", (1, 2), self._options(1))
        with patch.dict(os.environ, {"INFERENCE_MODEL_ALIAS": "qwen3-0.6b"}):
            descriptor = adapter.prepare_chat_generation("req", {}, prepared)
        self.assertEqual(descriptor["prompt_tokens"], 2)
        adapter.bind_generation_cache("req", (0,))
        result = adapter.execute_batch((ScheduledWorkItem("req", ExecutionPhase.PREFILL, 2),))[0]
        self.assertTrue(result.finished)
        inference_result = adapter.finish_chat_generation("req")
        self.assertEqual(inference_result.token_ids, (6,))
        self.assertEqual(inference_result.completion_tokens, 1)
        adapter.cancel_generation("req")


class FakeTokenizer:
    chat_template = "test-template"
    eos_token_id = 9
    return_mapping = False

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool, return_dict: bool):
        self.messages = messages
        self.assertions = (tokenize, add_generation_prompt, return_dict)
        if self.return_mapping:
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        return [1, 2, 3]

    def decode(self, token_ids, **kwargs):
        return " ".join(str(token_id) for token_id in token_ids)


class DefaultInputProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = DefaultChatInputProcessor()
        self.processor.model_alias = "standard-text"
        self.processor.tokenizer = FakeTokenizer()
        self.processor.config = {
            "generation": {"max_new_tokens": 8, "temperature": 0.0},
            "api": {"max_context_tokens": 32},
        }

    def test_prepares_chat_template_input_and_model_generation_defaults(self) -> None:
        prepared = self.processor.prepare_chat({"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(prepared.model_alias, "standard-text")
        self.assertEqual(prepared.input_ids, (1, 2, 3))
        self.assertEqual(prepared.sampling_params["max_new_tokens"], 8)
        self.assertEqual(prepared.sampling_params["eos_token_ids"], [9])
        self.assertEqual(self.processor.tokenizer.assertions, (True, True, False))

    def test_extracts_input_ids_when_tokenizer_returns_batch_encoding(self) -> None:
        self.processor.tokenizer.return_mapping = True
        prepared = self.processor.prepare_chat({"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(prepared.input_ids, (1, 2, 3))

    def test_rejects_images_and_unsupported_beam_search_before_worker(self) -> None:
        with self.assertRaises(InferenceError):
            self.processor.prepare_chat(
                {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}
            )
        with self.assertRaises(InferenceError):
            self.processor.prepare_chat({"messages": [{"role": "user", "content": "hello"}], "num_beams": 2})


class StandardProfilePreflightTests(unittest.TestCase):
    def test_standard_profile_uses_ast_defaults_without_importing_model_runtime(self) -> None:
        import launcher

        registration = {
            "integration_profile": "standard",
            "backend": "standard_text",
            "inference_config": "src/inference/models/qwen3_0_6b/configs/inference.yml",
            "network_config": "src/inference/models/qwen3_0_6b/configs/network.yml",
            "api": {
                "model_id": "example-text-model",
                "capabilities": {
                    "chat_inputs": ["text"],
                    "streaming": True,
                    "output_mode": "token_ids",
                    "video_upload": False,
                    "live_video": False,
                },
            },
        }
        with patch("launcher.importlib.import_module", side_effect=AssertionError("preflight imported a model")):
            result = launcher._validate_model_registration("example-text-model", registration, "")
        self.assertEqual(result["profile"], "standard")
        self.assertEqual(result["output_mode"], "token_ids")

    def test_standard_profile_rejects_multimodal_or_custom_cache_capabilities(self) -> None:
        import launcher

        registration = {
            "integration_profile": "standard",
            "backend": "standard_text",
            "inference_config": "src/inference/models/qwen3_0_6b/configs/inference.yml",
            "network_config": "src/inference/models/qwen3_0_6b/configs/network.yml",
            "api": {
                "model_id": "example-text-model",
                "capabilities": {
                    "chat_inputs": ["text", "image"],
                    "streaming": True,
                    "output_mode": "token_ids",
                    "video_upload": False,
                    "live_video": False,
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "standard profile supports text chat"):
            launcher._validate_model_registration("example-text-model", registration, "")


if __name__ == "__main__":
    unittest.main()
