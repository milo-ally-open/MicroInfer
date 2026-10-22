"""以合成權重驗證 Qwen3 原生計算圖、變長批次及實體 KV page。"""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "inference"
sys.path.insert(0, str(INFERENCE_ROOT))

from models.qwen3_0_6b.compatibility.hf_config import verify_snapshot_revision
from engine.contracts import ExecutionPhase, ScheduledWorkItem

try:
    import torch
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - 依賴精簡的開發環境略過此組測試
    torch = None
    save_file = None


if torch is not None and save_file is not None:
    from models.qwen3_0_6b.compatibility.safetensors_loader import load_safetensors
    from models.qwen3_0_6b.model.input_batch import InputBatch
    from models.qwen3_0_6b.model.paged_kv_cache import PagedKVCache
    from models.qwen3_0_6b.model.qwen3_model import Qwen3ForCausalLM


class Qwen3SnapshotMetadataTests(unittest.TestCase):
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"

    def _snapshot(self, root: Path, *, bad_revision: str | None = None) -> Path:
        metadata = root / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True)
        assets = (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        )
        for name in assets:
            (root / name).touch()
            commit = bad_revision if name == "model.safetensors" and bad_revision else self.revision
            (metadata / f"{name}.metadata").write_text(f"{commit}\netag\n0\n", encoding="utf-8")
        return root

    def test_accepts_only_complete_assets_from_the_pinned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._snapshot(Path(directory))
            verify_snapshot_revision(snapshot, self.revision)

    def test_rejects_snapshot_assets_from_a_different_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._snapshot(Path(directory), bad_revision="0" * 40)
            with self.assertRaisesRegex(ValueError, "belongs to revision"):
                verify_snapshot_revision(snapshot, self.revision)


@unittest.skipIf(torch is None or save_file is None, "Qwen3 synthetic runtime tests require PyTorch and Safetensors.")
class Qwen3NativeRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = SimpleNamespace(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
        self.model = Qwen3ForCausalLM(self.config, dtype=torch.float32, device="cpu").eval()
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.normal_(mean=0.0, std=0.02)
            self.model.lm_head.weight.copy_(self.model.model.embed_tokens.weight)
        self.model.tie_weights()

    def _cache(self) -> PagedKVCache:
        return PagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_blocks=8,
            block_size_tokens=4,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def test_input_batch_pads_and_validates_scheduled_metadata(self) -> None:
        items = (
            ScheduledWorkItem("short", ExecutionPhase.PREFILL, 2),
            ScheduledWorkItem("long", ExecutionPhase.PREFILL, 3),
        )
        batch = InputBatch.build(
            items,
            (torch.tensor([11, 12]), torch.tensor([21, 22, 23])),
            (2, 3),
            (0, 0),
            ((0,), (1, 2)),
            device=torch.device("cpu"),
        )
        self.assertEqual(batch.request_ids, ("short", "long"))
        self.assertEqual(batch.input_ids.tolist(), [[11, 12, 0], [21, 22, 23]])
        self.assertEqual(batch.token_lengths, (2, 3))
        self.assertEqual(batch.block_tables, ((0,), (1, 2)))
        with self.assertRaisesRegex(ValueError, "scheduled token budgets"):
            InputBatch.build(
                (ScheduledWorkItem("bad", ExecutionPhase.PREFILL, 3),),
                (torch.tensor([1, 2]),),
                (2,),
                (0,),
                ((0,),),
                device=torch.device("cpu"),
            )

    def test_safetensors_mapping_loads_all_synthetic_model_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()
            }
            save_file(state, str(snapshot / "model.safetensors"))
            target = Qwen3ForCausalLM(self.config, dtype=torch.float32, device="cpu")

            parameter_count = load_safetensors(target, snapshot)

        self.assertGreater(parameter_count, 0)
        self.assertIs(target.model.embed_tokens.weight, target.lm_head.weight)
        self.assertTrue(
            torch.equal(target.model.embed_tokens.weight, self.model.model.embed_tokens.weight)
        )

    def test_variable_length_prefill_and_decode_match_individual_forwards(self) -> None:
        batch_cache = self._cache()
        batch_cache.bind("short", (0, 1))
        batch_cache.bind("long", (2, 3))
        batch_inputs = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])

        with torch.inference_mode():
            batch_prefill = self.model.forward_batch(
                batch_inputs,
                token_lengths=(3, 5),
                position_offsets=(0, 0),
                request_ids=("short", "long"),
                block_tables=((0, 1), (2, 3)),
                cache=batch_cache,
            )

        single_caches = {"short": self._cache(), "long": self._cache()}
        single_prompts = {"short": (1, 2, 3), "long": (4, 5, 6, 7, 8)}
        expected_prefill = {}
        with torch.inference_mode():
            for index, request_id in enumerate(("short", "long")):
                single_cache = single_caches[request_id]
                single_cache.bind(request_id, (0, 1))
                prompt = torch.tensor([single_prompts[request_id]])
                expected_prefill[request_id] = self.model.forward_batch(
                    prompt,
                    token_lengths=(prompt.shape[1],),
                    position_offsets=(0,),
                    request_ids=(request_id,),
                    block_tables=((0, 1),),
                    cache=single_cache,
                )[0]
                self.assertTrue(torch.allclose(batch_prefill[index], expected_prefill[request_id], atol=1e-6, rtol=1e-5))

            batch_decode = self.model.forward_batch(
                torch.tensor([[9], [10]]),
                token_lengths=(1, 1),
                position_offsets=(3, 5),
                request_ids=("short", "long"),
                block_tables=((0, 1), (2, 3)),
                cache=batch_cache,
            )
            for index, request_id in enumerate(("short", "long")):
                prompt_length = len(single_prompts[request_id])
                expected_decode = self.model.forward_batch(
                    torch.tensor([[9 if request_id == "short" else 10]]),
                    token_lengths=(1,),
                    position_offsets=(prompt_length,),
                    request_ids=(request_id,),
                    block_tables=((0, 1),),
                    cache=single_caches[request_id],
                )[0]
                self.assertTrue(torch.allclose(batch_decode[index], expected_decode, atol=1e-6, rtol=1e-5))

    def test_physical_page_ownership_is_isolated_and_released(self) -> None:
        cache = self._cache()
        cache.bind("first", (0, 1))
        cache.bind("second", (2, 3))
        with self.assertRaises(RuntimeError):
            cache.bind("overlap", (3, 4))

        cache.release("first")
        self.assertEqual(cache.snapshot()["used_blocks"], 2)
        cache.release("second")
        self.assertEqual(cache.snapshot()["used_blocks"], 0)
        self.assertEqual(cache.snapshot()["free_blocks"], cache.num_blocks)


if __name__ == "__main__":
    unittest.main()
