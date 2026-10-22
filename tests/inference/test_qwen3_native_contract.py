"""以標準函式庫保護 Qwen3 三層模型執行契約。"""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "src/inference"
QWEN = INFERENCE / "models/qwen3_0_6b"


class Qwen3NativeContractTests(unittest.TestCase):
    def test_ecosystem_compatibility_is_separate_from_native_forward(self) -> None:
        config = (QWEN / "compatibility/hf_config.py").read_text(encoding="utf-8")
        tokenizer = (QWEN / "compatibility/hf_tokenizer.py").read_text(encoding="utf-8")
        loader = (QWEN / "compatibility/safetensors_loader.py").read_text(encoding="utf-8")
        runner = (QWEN / "model_runner.py").read_text(encoding="utf-8")
        self.assertIn("AutoConfig.from_pretrained", config)
        self.assertIn('trust_remote_code=False', config)
        self.assertIn("verify_snapshot_revision(snapshot, revision)", config)
        self.assertIn("AutoTokenizer.from_pretrained", tokenizer)
        self.assertIn("load_state_dict(state, strict=True)", loader)
        self.assertNotIn("AutoModelForCausalLM", runner)
        self.assertNotIn(".generate(", runner)

    def test_native_model_has_explicit_compute_and_physical_cache_modules(self) -> None:
        expected = (
            "model/qwen3_model.py",
            "model/qwen3_attention.py",
            "model/qwen3_layers.py",
            "model/rotary_embedding.py",
            "model/paged_kv_cache.py",
            "sampler.py",
        )
        for relative in expected:
            with self.subTest(module=relative):
                self.assertTrue((QWEN / relative).is_file())
        runner = (QWEN / "model_runner.py").read_text(encoding="utf-8")
        cache = (QWEN / "model/paged_kv_cache.py").read_text(encoding="utf-8")
        model = (QWEN / "model/qwen3_model.py").read_text(encoding="utf-8")
        self.assertIn("self.model.forward_batch", runner)
        self.assertIn("torch.empty(", cache)
        self.assertIn("cache.write(", (QWEN / "model/qwen3_attention.py").read_text(encoding="utf-8"))
        self.assertIn("enable_gqa=True", (QWEN / "model/qwen3_attention.py").read_text(encoding="utf-8"))
        self.assertIn("def tie_weights(", model)

    def test_engine_and_worker_share_page_capacity_and_release_order(self) -> None:
        yaml = (QWEN / "configs/inference.yml").read_text(encoding="utf-8")
        engine = (INFERENCE / "engine/engine.py").read_text(encoding="utf-8")
        runner = (QWEN / "model_runner.py").read_text(encoding="utf-8")
        cancel_context = engine[engine.index("def _cancel_context("):engine.index("def _cancel_worker_generation(")]
        self.assertRegex(yaml, r"(?m)^\s*block_size_tokens:\s*16\s*$")
        self.assertRegex(yaml, r"(?m)^\s*num_blocks:\s*1024\s*$")
        self.assertIn('"physical_blocks"', runner)
        self.assertIn("Engine KV block count does not match", engine)
        self.assertLess(
            cancel_context.index("self._cancel_worker_generation(context.request_id)"),
            cancel_context.index("self.kv_cache.release(context.request_id)"),
        )

    def test_chat_route_and_health_keep_stream_contract_and_expose_runtime_state(self) -> None:
        route = (INFERENCE / "api/routes/chat_completions.py").read_text(encoding="utf-8")
        schema = (INFERENCE / "api/schemas.py").read_text(encoding="utf-8")
        self.assertIn('if event.kind == "text"', route)
        self.assertIn('"data: [DONE]\\n\\n"', route)
        self.assertIn('message["type"] == "http.disconnect"', route)
        self.assertIn("CancelScope(shield=True)", route)
        self.assertIn('getattr(stream, "cancel", None)', route)
        for field in ("scheduler", "kv_cache", "worker_kv_cache", "gpu_peak_memory"):
            with self.subTest(field=field):
                self.assertIn(f"{field}: dict[str, int]", schema)


if __name__ == "__main__":
    unittest.main()
