"""驗證 Model Worker 子程序啟動、RPC、串流及關閉。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "inference"))

from engine.worker import MultiprocessModelWorker


class MultiprocessWorkerTests(unittest.TestCase):
    def test_worker_starts_in_child_and_executes_ipc_calls(self) -> None:
        worker = MultiprocessModelWorker(
            "worker_process_fixture:create_worker",
            startup_timeout_seconds=10,
            request_timeout_seconds=10,
        )
        worker.start()
        try:
            self.assertIsNotNone(worker.pid)
            self.assertNotEqual(worker.pid, __import__("os").getpid())
            self.assertEqual(worker.health_info()["backend"], "process-test")
            self.assertEqual(worker.invoke("echo", 1, value="two"), {
                "operation": "echo", "args": (1,), "kwargs": {"value": "two"}
            })
            events = list(worker.stream("events"))
            self.assertEqual([event["index"] for event in events], [1, 2])
            self.assertEqual(worker.gpu_status()["gpu"], "child")
        finally:
            worker.close()
        self.assertIsNone(worker.pid)


if __name__ == "__main__":
    unittest.main()
