"""以標準函式庫保護推論 API 契約及分層邊界。"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "src/inference"


class InferenceSourceContractTests(unittest.TestCase):
    def test_launcher_cli_surface_and_model_paths_remain_stable(self) -> None:
        launcher = (INFERENCE / "launcher.py").read_text(encoding="utf-8")
        registry = (INFERENCE / "model_registry.yaml").read_text(encoding="utf-8")
        inference_config = (INFERENCE / "models/mage_vl_4b/configs/inference.yml").read_text(encoding="utf-8")
        self.assertIn('commands.add_parser("list"', launcher)
        self.assertIn('commands.add_parser("start"', launcher)
        for option in ('"--model"', '"--inference-config"', '"--network-config"', '"--media-config"', '"--host"', '"--port"', '"--cuda-visible-devices"'):
            with self.subTest(option=option):
                self.assertIn(option, launcher)
        self.assertRegex(registry, r"(?m)^default_model:\s*mage-vl-4b\s*$")
        self.assertRegex(registry, r"(?m)^\s+mage-vl-4b:\s*$")
        self.assertIn("inference_config: src/inference/models/mage_vl_4b/configs/inference.yml", registry)
        self.assertIn("network_config: src/inference/models/mage_vl_4b/configs/network.yml", registry)
        self.assertIn("media_config: src/inference/models/mage_vl_4b/configs/mediamtx.yml", registry)
        self.assertIn("service_entrypoint: src/inference/models/mage_vl_4b/app.py", registry)
        self.assertFalse((INFERENCE / "mage_vl_4b").exists())
        self.assertRegex(inference_config, r"(?m)^\s+id:\s*microsoft/Mage-VL\s*$")
        self.assertIn("revision: d88b153285f1633a61b2f693c59c8576693af185", inference_config)

    def test_all_existing_routes_remain_in_api_server(self) -> None:
        expected = {
            ("GET", "/health"),
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/videos/understand"),
            ("POST", "/v1/live/sessions"),
            ("GET", "/v1/live/sessions/{session_id}/events"),
            ("DELETE", "/v1/live/sessions/{session_id}"),
            ("POST", "/v1/live/mediamtx/auth"),
        }
        routes: set[tuple[str, str]] = set()
        for path in (INFERENCE / "api").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                        method = decorator.func.attr.upper()
                        if method in {"GET", "POST", "DELETE"} and decorator.args and isinstance(decorator.args[0], ast.Constant):
                            routes.add((method, str(decorator.args[0].value)))
        self.assertEqual(routes, expected)

    def test_api_engine_worker_and_model_runner_are_separate(self) -> None:
        api_sources = [path.read_text(encoding="utf-8") for path in (INFERENCE / "api").rglob("*.py")]
        api_text = "\n".join(api_sources)
        self.assertIn('media_type="text/event-stream"', api_text)
        self.assertIn('"data: [DONE]\\n\\n"', api_text)
        for forbidden in ("AutoModelForCausalLM", "torch.cuda", "TextIteratorStreamer"):
            with self.subTest(api_symbol=forbidden):
                self.assertNotIn(forbidden, api_text)
        runner = (INFERENCE / "models/mage_vl_4b/model_runner.py").read_text(encoding="utf-8")
        loader = (INFERENCE / "models/mage_vl_4b/components/model_loader.py").read_text(encoding="utf-8")
        stepwise = (INFERENCE / "models/mage_vl_4b/components/stepwise_generation.py").read_text(encoding="utf-8")
        self.assertLess(runner.index("validate_configuration"), runner.index("load_mage_vl()"))
        self.assertIn("MageVLForConditionalGeneration.from_pretrained", loader)
        self.assertIn("AutoProcessor.from_pretrained", loader)
        self.assertIn("MAGE_VL_SOURCE_REVISION", loader)
        self.assertIn("self.model.generate", runner)
        self.assertIn("MageVLStepwiseGeneration", runner)
        engine = (INFERENCE / "engine/engine.py").read_text(encoding="utf-8")
        scheduler = (INFERENCE / "engine/scheduler.py").read_text(encoding="utf-8")
        worker = (INFERENCE / "engine/worker.py").read_text(encoding="utf-8")
        adapter = (INFERENCE / "models/mage_vl_4b/adapter.py").read_text(encoding="utf-8")
        self.assertIn("self.worker.start()", engine)
        self.assertIn("self.worker.invoke(prepare_operation", engine)
        self.assertIn("self.worker.execute_batch(batch.work_items)", engine)
        self.assertIn("def _dispatch_batches(", engine)
        self.assertIn("TokenScheduler", engine)
        self.assertIn("schedule_batch", scheduler)
        self.assertIn("decode_generation_step", adapter)
        self.assertIn("class ModelWorker(Protocol)", worker)
        self.assertIn("class InProcessModelWorker", worker)
        self.assertIn("class MultiprocessModelWorker", worker)
        model_sources = [path.read_text(encoding="utf-8") for path in (INFERENCE / "models").rglob("*.py")]
        engine_sources = [path.read_text(encoding="utf-8") for path in (INFERENCE / "engine").rglob("*.py")]
        for source in model_sources + engine_sources:
            self.assertNotIn("from fastapi", source)
            self.assertNotIn("StreamingResponse", source)

    def test_api_engine_core_and_worker_use_three_process_boundary(self) -> None:
        app = (INFERENCE / "models/mage_vl_4b/app.py").read_text(encoding="utf-8")
        core_factory = (INFERENCE / "models/mage_vl_4b/engine_core.py").read_text(encoding="utf-8")
        client = (INFERENCE / "engine/client.py").read_text(encoding="utf-8")
        chat_route = (INFERENCE / "api/routes/chat_completions.py").read_text(encoding="utf-8")
        self.assertIn("EngineClient(", app)
        self.assertIn("models.mage_vl_4b.engine_core:create_engine_core", app)
        self.assertNotIn("InferenceEngine(", app)
        self.assertIn("InferenceEngine(", core_factory)
        self.assertIn("MultiprocessModelWorker(", core_factory)
        self.assertIn('multiprocessing.get_context("spawn")', client)
        self.assertIn('context.Pipe(duplex=True)', client)
        self.assertIn("engine.register_request(request_id)", client)
        self.assertIn("engine.unregister_request(request_id)", client)
        self.assertIn('"data: [DONE]\\n\\n"', chat_route)
        self.assertIn('if event.kind == "text"', chat_route)

    def test_routes_call_engine_without_model_or_session_manager_access(self) -> None:
        route_paths = [path for path in (INFERENCE / "api/routes").glob("*.py") if path.name != "__init__.py"]
        route_sources = [path.read_text(encoding="utf-8") for path in route_paths]
        for source in route_sources:
            for forbidden in ("inference_service", "LiveSessionManager", "MageVLModelAdapter", "model_runner"):
                with self.subTest(symbol=forbidden):
                    self.assertNotIn(forbidden, source)
            self.assertIn("app.state.engine", source)

    def test_token_scheduler_kv_cache_and_worker_are_implemented(self) -> None:
        engine = (INFERENCE / "engine/engine.py").read_text(encoding="utf-8")
        scheduler = (INFERENCE / "engine/scheduler.py").read_text(encoding="utf-8")
        cache = (INFERENCE / "engine/kv_cache.py").read_text(encoding="utf-8")
        worker = (INFERENCE / "engine/worker.py").read_text(encoding="utf-8")
        runner = (INFERENCE / "models/mage_vl_4b/model_runner.py").read_text(encoding="utf-8")
        stepwise = (INFERENCE / "models/mage_vl_4b/components/stepwise_generation.py").read_text(encoding="utf-8")
        self.assertIn("class TokenScheduler", scheduler)
        self.assertIn("max_num_scheduled_tokens", scheduler)
        self.assertIn("schedule_batch()", engine)
        self.assertIn("complete_batch", engine + scheduler)
        self.assertIn("class KVCacheManager", cache)
        self.assertIn("self.kv_cache.reserve", engine)
        self.assertIn("self.kv_cache.release", engine)
        self.assertIn("class MultiprocessModelWorker", worker)
        self.assertIn("multiprocessing.get_context(\"spawn\")", worker)
        self.assertIn("def _step(", stepwise)
        self.assertIn("past_key_values", stepwise)

    def test_model_package_owns_its_configs_and_dependency_manifest(self) -> None:
        model_package = INFERENCE / "models/mage_vl_4b"
        self.assertTrue((model_package / "app.py").is_file())
        self.assertTrue((model_package / "configs/inference.yml").is_file())
        self.assertTrue((model_package / "requirements/requirements.txt").is_file())

    def test_new_model_onboarding_reuses_shared_launcher_api_and_engine(self) -> None:
        launcher = (INFERENCE / "launcher.py").read_text(encoding="utf-8")
        app = (INFERENCE / "models/mage_vl_4b/app.py").read_text(encoding="utf-8")
        engine_core = (INFERENCE / "models/mage_vl_4b/engine_core.py").read_text(encoding="utf-8")
        adapter = (INFERENCE / "models/mage_vl_4b/adapter.py").read_text(encoding="utf-8")
        readme = (INFERENCE / "README.md").read_text(encoding="utf-8")
        self.assertIn("sys.path.insert(0, str(INFERENCE_ROOT))", launcher)
        self.assertNotIn("mage-vl-4b", launcher)
        self.assertIn("from api.app import create_app", app)
        self.assertIn("app = create_app(", app)
        self.assertIn("EngineClient(", app)
        self.assertNotIn("InferenceEngine(", app)
        self.assertNotIn("MultiprocessModelWorker(", app)
        self.assertIn("def create_engine_core(", engine_core)
        self.assertIn("InferenceEngine(", engine_core)
        self.assertIn("MultiprocessModelWorker(", engine_core)
        self.assertIn("def create_model_worker(", engine_core)
        for method in (
            "prepare_chat_generation",
            "prepare_video_generation",
            "prepare_live_frames_generation",
            "finish_chat_generation",
            "finish_live_generation",
        ):
            with self.subTest(service_method=method):
                self.assertIn(f"def {method}(", adapter)
        self.assertFalse((INFERENCE / "models/mage_vl_4b/service.py").exists())
        self.assertFalse((INFERENCE / "models/mage_vl_4b/state.py").exists())
        self.assertIn("新模型可沿用共用 route／Scheduler／KV Manager", readme)

    def test_batch_runner_entry_preserves_incremental_sse_text(self) -> None:
        engine = (INFERENCE / "engine/engine.py").read_text(encoding="utf-8")
        worker = (INFERENCE / "engine/worker.py").read_text(encoding="utf-8")
        adapter = (INFERENCE / "models/mage_vl_4b/adapter.py").read_text(encoding="utf-8")
        runner = (INFERENCE / "models/mage_vl_4b/model_runner.py").read_text(encoding="utf-8")
        stepwise = (INFERENCE / "models/mage_vl_4b/components/stepwise_generation.py").read_text(encoding="utf-8")
        cache_manager = (INFERENCE / "models/mage_vl_4b/components/cache_manager.py").read_text(encoding="utf-8")
        adapter_source = (INFERENCE / "models/mage_vl_4b/adapter.py").read_text(encoding="utf-8")
        route = (INFERENCE / "api/routes/chat_completions.py").read_text(encoding="utf-8")
        self.assertIn("self.worker.execute_batch(batch.work_items)", engine)
        self.assertIn("def execute_batch(self, requests", worker)
        self.assertIn("def execute_batch(self, requests", adapter)
        self.assertIn("def execute_batch(self, requests", runner)
        self.assertIn("self._stepwise.execute_batch(", runner)
        self.assertIn("def _batch_forward(", stepwise)
        self.assertIn("past_key_values = self._combine_caches(", stepwise)
        self.assertIn("torch.cat(padded_ids, dim=0)", stepwise)
        self.assertIn("self.model(**prepared, return_dict=True)", stepwise)
        self.assertIn("self._split_cache(output_cache", stepwise)
        self.assertIn("request_id, split_caches[index], expected_length=expected_length", stepwise)
        self.assertIn("self.cache_manager.release(request_id)", stepwise)
        self.assertIn("self.cache_manager.clear()", stepwise)
        self.assertIn("expected_length=expected_cache_length", stepwise)
        self.assertIn("class RequestCacheManager:", cache_manager)
        self.assertIn("expected_length is not None and length != expected_length", cache_manager)
        self.assertIn('"worker_kv_cache": self.model_runner.kv_cache_snapshot()', adapter_source)
        self.assertIn('elif event.kind == "text":', route)
        self.assertIn('delta={"content": event.text}', route)

    def test_src_and_inference_roots_are_not_python_packages(self) -> None:
        self.assertFalse((ROOT / "src/__init__.py").exists())
        self.assertFalse((INFERENCE / "__init__.py").exists())
        self.assertFalse((INFERENCE / "models/__init__.py").exists())
        for path in INFERENCE.rglob("*.py"):
            with self.subTest(path=path):
                self.assertNotIn("src.inference", path.read_text(encoding="utf-8"))

    def test_request_limits_and_video_sse_contract_markers_remain(self) -> None:
        app_source = (INFERENCE / "api/app.py").read_text(encoding="utf-8")
        config = (INFERENCE / "models/mage_vl_4b/configs/inference.yml").read_text(encoding="utf-8")
        adapter = (INFERENCE / "models/mage_vl_4b/adapter.py").read_text(encoding="utf-8")
        for name, value in {
            "max_chat_request_bytes": 15728640,
            "max_live_request_bytes": 65536,
            "max_video_request_bytes": 210763776,
            "max_video_bytes": 209715200,
        }.items():
            with self.subTest(limit=name):
                self.assertRegex(config, rf"(?m)^\s*{re.escape(name)}:\s*{value}\s*$")
        self.assertIn("Content-Length is required", app_source)
        self.assertIn('response["object"] = "magevl4b.video.understanding"', (INFERENCE / "api/serializers.py").read_text(encoding="utf-8"))
        self.assertIn("result = InferenceResult", adapter)
        self.assertIn("/v1/live/mediamtx/auth", (INFERENCE / "api/routes/live_sessions.py").read_text(encoding="utf-8"))

    def test_live_sse_and_gateway_boundary_remain_separate(self) -> None:
        live_route = (INFERENCE / "api/routes/live_sessions.py").read_text(encoding="utf-8")
        self.assertIn('"X-Accel-Buffering": "no"', live_route)
        self.assertIn('"last-event-id"', live_route)
        for path in (INFERENCE / "api").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for gateway_auth_symbol in ("APIKeyHeader", "HTTPBearer", "Authorization"):
                with self.subTest(path=path, symbol=gateway_auth_symbol):
                    self.assertNotIn(gateway_auth_symbol, source)

    def test_inference_tests_do_not_import_heavy_model_runtime(self) -> None:
        imported = set()
        for path in (ROOT / "tests/inference").glob("test_*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
        self.assertTrue(imported.isdisjoint({"torch", "transformers", "huggingface_hub"}))


if __name__ == "__main__":
    unittest.main()
