"""選擇性地以本機固定 snapshot 驗證標準文字預設路徑；不啟動 API 或常駐服務。"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_ROOT = ROOT / "src" / "inference"
sys.path.insert(0, str(INFERENCE_ROOT))

try:
    import torch
    import yaml
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional local snapshot environment
    torch = None
    yaml = None
    AutoTokenizer = None


if torch is not None:
    from api.default_chat_processor import DefaultChatInputProcessor
    from engine.contracts import ExecutionPhase, PreparedChatInput, ScheduledWorkItem
    from engine.standard_text import StandardTextModelAdapter, StandardTextModelRunner


class CPUOnlyPlatform:
    name = "cpu-smoke"

    def resolve_device(self, requested: str) -> Any:
        return torch.device("cpu")

    def gpu_status(self, device: Any) -> dict[str, Any]:
        return {"device": str(device)}

    def peak_memory(self, device: Any) -> dict[str, int]:
        return {"peak_allocated": 0, "peak_reserved": 0}

    def reset_peak_memory(self, device: Any) -> None:
        return None

    def release_cached_memory(self) -> None:
        return None


@unittest.skipUnless(os.environ.get("RUN_STANDARD_TEXT_SNAPSHOT") == "1", "set RUN_STANDARD_TEXT_SNAPSHOT=1 to load a local snapshot")
class StandardTextSnapshotSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if torch is None or yaml is None or AutoTokenizer is None:
            raise unittest.SkipTest("PyTorch, PyYAML and Transformers are required for snapshot smoke.")
        device_name = os.environ.get("STANDARD_TEXT_DEVICE", "cpu").strip()
        if device_name != "cpu" and not device_name.startswith("cuda:"):
            raise ValueError("STANDARD_TEXT_DEVICE must be 'cpu' or a CUDA device such as 'cuda:0'.")
        platform: Any = CPUOnlyPlatform()
        if device_name.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise unittest.SkipTest("CUDA is unavailable in the selected Python environment.")
            device = torch.device(device_name)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            minimum_free_mb = int(os.environ.get("STANDARD_TEXT_MIN_FREE_GPU_MB", "3072"))
            if free_bytes < minimum_free_mb * 1024 * 1024:
                raise unittest.SkipTest(
                    f"CUDA device has {free_bytes // (1024 * 1024)} MiB free; "
                    f"smoke requires at least {minimum_free_mb} MiB."
                )
            from engine.platform import TorchCudaPlatform

            platform = TorchCudaPlatform()
        config_value = os.environ.get("STANDARD_TEXT_INFERENCE_CONFIG")
        if not config_value:
            raise unittest.SkipTest("set STANDARD_TEXT_INFERENCE_CONFIG to an explicit inference YAML path")
        config_path = Path(config_value).resolve()
        source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        local_path = source_config.get("runtime", {}).get("local_model_path")
        if local_path is None:
            raise ValueError("The selected snapshot smoke config must set runtime.local_model_path.")
        model_path = Path(os.path.expandvars(os.path.expanduser(str(local_path))))
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Local model snapshot is incomplete: {model_path}")
        source_config["model"]["dtype"] = os.environ.get(
            "STANDARD_TEXT_DTYPE", "float16" if device_name.startswith("cuda:") else "float32"
        )
        source_config["model"]["device_map"] = None
        source_config["model"]["device"] = device_name
        if device_name.startswith("cuda:"):
            source_config["runtime"]["platform"] = "torch_cuda"
        source_config["runtime"]["local_files_only"] = True
        source_config.setdefault("generation", {})["max_new_tokens"] = 2
        engine = source_config.setdefault("engine", {})
        engine["max_num_seqs"] = 1
        engine["max_waiting_requests"] = 1
        engine["max_num_scheduled_tokens"] = min(int(engine.get("max_num_scheduled_tokens", 128)), 128)
        engine["prefill_chunk_size_tokens"] = min(int(engine.get("prefill_chunk_size_tokens", 64)), 64)
        cls._temporary = tempfile.TemporaryDirectory(prefix="standard-text-snapshot-")
        cls.config_path = Path(cls._temporary.name) / "inference.yml"
        cls.config_path.write_text(yaml.safe_dump(source_config, allow_unicode=True), encoding="utf-8")
        cls._old_config = os.environ.get("INFERENCE_CONFIG")
        cls._old_alias = os.environ.get("INFERENCE_MODEL_ALIAS")
        os.environ["INFERENCE_CONFIG"] = str(cls.config_path)
        os.environ["INFERENCE_MODEL_ALIAS"] = str(os.environ.get("STANDARD_TEXT_MODEL_ALIAS", "qwen3-0.6b"))
        cls.processor = DefaultChatInputProcessor()
        cls.runner = StandardTextModelRunner(platform)
        try:
            cls.processor.start()
            cls.runner.load()
            cls.adapter = StandardTextModelAdapter(cls.runner)
        except BaseException:
            cls.processor.close()
            cls.runner.close()
            cls._restore_environment()
            cls._temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "processor"):
            cls.processor.close()
        if hasattr(cls, "runner"):
            cls.runner.close()
        cls._restore_environment()
        if hasattr(cls, "_temporary"):
            cls._temporary.cleanup()

    @classmethod
    def _restore_environment(cls) -> None:
        if getattr(cls, "_old_config", None) is None:
            os.environ.pop("INFERENCE_CONFIG", None)
        else:
            os.environ["INFERENCE_CONFIG"] = cls._old_config
        if getattr(cls, "_old_alias", None) is None:
            os.environ.pop("INFERENCE_MODEL_ALIAS", None)
        else:
            os.environ["INFERENCE_MODEL_ALIAS"] = cls._old_alias

    def test_fixed_snapshot_prefill_decode_cache_and_text_reconstruction(self) -> None:
        prepared = self.processor.prepare_chat(
            {
                "messages": [{"role": "user", "content": "Reply with one short word."}],
                "max_tokens": 2,
                "temperature": 0.0,
            }
        )
        self.assertIsInstance(prepared, PreparedChatInput)
        request_id = "snapshot-smoke"
        descriptor = self.adapter.prepare_chat_generation(request_id, {}, prepared)
        block_size = int(source_cache_block_size(self.config_path))
        block_count = math.ceil(
            (descriptor["prompt_tokens"] + descriptor["max_new_tokens"]) / block_size
        )
        self.adapter.bind_generation_cache(request_id, tuple(range(block_count)))
        result = self.adapter.execute_batch(
            (ScheduledWorkItem(request_id, ExecutionPhase.PREFILL, descriptor["prompt_tokens"]),)
        )[0]
        steps = 1
        while not result.finished:
            result = self.adapter.execute_batch(
                (ScheduledWorkItem(request_id, ExecutionPhase.DECODE, 1),)
            )[0]
            steps += 1
            self.assertLessEqual(steps, 2)
        self.assertGreaterEqual(steps, 1)
        output = self.adapter.finish_chat_generation(request_id)
        stream_processor = self.processor.create_stream_output_processor()
        streamed = "".join(stream_processor.push(token_id) for token_id in output.token_ids)
        streamed += stream_processor.finish()
        self.assertEqual(streamed, self.processor.decode_tokens(output.token_ids))
        self.assertEqual(self.runner.cache_snapshot()["active_requests"], 0)

    def test_cancel_releases_snapshot_request_cache_state(self) -> None:
        prepared = self.processor.prepare_chat(
            {"messages": [{"role": "user", "content": "Reply briefly."}], "max_tokens": 2}
        )
        request_id = "snapshot-cancel"
        descriptor = self.adapter.prepare_chat_generation(request_id, {}, prepared)
        block_size = int(source_cache_block_size(self.config_path))
        block_count = math.ceil(
            (descriptor["prompt_tokens"] + descriptor["max_new_tokens"]) / block_size
        )
        self.adapter.bind_generation_cache(request_id, tuple(range(block_count)))
        self.adapter.execute_batch(
            (ScheduledWorkItem(request_id, ExecutionPhase.PREFILL, descriptor["prompt_tokens"]),)
        )
        self.adapter.cancel_generation(request_id)
        self.assertEqual(self.runner.cache_snapshot()["active_requests"], 0)


def source_cache_block_size(config_path: Path) -> int:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return int(data.get("engine", {}).get("kv_cache", {}).get("block_size_tokens", 16))


if __name__ == "__main__":
    unittest.main()
