"""驗證 Engine KV block 預留、容量拒絕及 block id 回收。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "inference"))

from engine.kv_cache import KVCacheManager, KVCacheRequest


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

    def test_batch_reservation_is_atomic_and_returns_one_block_table_per_request(self) -> None:
        manager = KVCacheManager(block_size_tokens=4, capacity_tokens=8, bytes_per_token=2)
        rejected = manager.reserve_batch([
            KVCacheRequest("a", prompt_tokens=3, max_output_tokens=1),
            KVCacheRequest("b", prompt_tokens=3, max_output_tokens=1),
            KVCacheRequest("c", prompt_tokens=5, max_output_tokens=1),
        ])
        self.assertIsNone(rejected)
        self.assertEqual(manager.snapshot()["used_blocks"], 0)

        accepted = manager.reserve_batch([
            KVCacheRequest("a", prompt_tokens=3, max_output_tokens=1),
            KVCacheRequest("b", prompt_tokens=3, max_output_tokens=1),
        ])
        self.assertEqual(accepted["a"].block_ids, (0,))
        self.assertEqual(accepted["b"].block_ids, (1,))
        self.assertEqual(accepted["b"].estimated_bytes, 8)
        manager.release("a")
        manager.release("b")

    def test_batch_reservation_rejects_duplicate_ids_before_allocating(self) -> None:
        manager = KVCacheManager(block_size_tokens=4, capacity_tokens=16)
        with self.assertRaises(ValueError):
            manager.reserve_batch([
                KVCacheRequest("same", prompt_tokens=2, max_output_tokens=1),
                KVCacheRequest("same", prompt_tokens=2, max_output_tokens=1),
            ])
        self.assertEqual(manager.snapshot()["used_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
