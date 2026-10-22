"""驗證 Engine KV block 預留、容量拒絕及 block id 回收。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "inference"))

from engine.kv_cache import KVCacheManager


class KVCacheManagerTests(unittest.TestCase):
    def test_reservation_maps_and_recycles_logical_blocks(self) -> None:
        manager = KVCacheManager(block_size_tokens=4, capacity_tokens=16, bytes_per_token=10)
        first = manager.reserve("first", prompt_tokens=5, max_output_tokens=2)
        self.assertIsNotNone(first)
        self.assertEqual(first.block_ids, (0, 1))
        self.assertEqual(first.estimated_bytes, 70)
        self.assertIsNone(manager.reserve("too-large", prompt_tokens=5, max_output_tokens=8))
        manager.release("first")
        reused = manager.reserve("second", prompt_tokens=5, max_output_tokens=2)
        self.assertEqual(reused.block_ids, first.block_ids)
        manager.release("second")
        self.assertEqual(manager.snapshot()["used_blocks"], 0)

if __name__ == "__main__":
    unittest.main()
