"""驗證 Worker 端實際 Cache 物件登錄及釋放。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "inference"
sys.path.insert(0, str(INFERENCE_ROOT))

from models.mage_vl_4b.components.cache_manager import RequestCacheManager


class FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class RequestCacheManagerTests(unittest.TestCase):
    def test_tracks_cache_object_length_and_logical_lease_until_release(self) -> None:
        manager = RequestCacheManager()
        manager.register("request-a")
        manager.bind_blocks("request-a", (3, 4))
        cache = FakeCache(7)

        self.assertEqual(manager.update("request-a", cache), 7)
        self.assertIs(manager.get("request-a"), cache)
        self.assertEqual(manager.length("request-a"), 7)
        self.assertEqual(manager.snapshot(), {
            "active_requests": 1,
            "populated_caches": 1,
            "cached_tokens": 7,
            "logical_blocks": 2,
        })

        manager.release("request-a")
        self.assertEqual(manager.snapshot(), {
            "active_requests": 0,
            "populated_caches": 0,
            "cached_tokens": 0,
            "logical_blocks": 0,
        })

    def test_clear_releases_all_request_references(self) -> None:
        manager = RequestCacheManager()
        manager.register("request-a")
        manager.register("request-b")
        manager.update("request-a", FakeCache(2))
        manager.clear()

        self.assertEqual(manager.snapshot()["active_requests"], 0)

    def test_rejects_cache_length_mismatch_without_replacing_existing_cache(self) -> None:
        manager = RequestCacheManager()
        manager.register("request-a")
        original = FakeCache(4)
        manager.update("request-a", original, expected_length=4)

        with self.assertRaisesRegex(RuntimeError, "Cache length mismatch"):
            manager.update("request-a", FakeCache(6), expected_length=5)

        self.assertIs(manager.get("request-a"), original)
        self.assertEqual(manager.length("request-a"), 4)


if __name__ == "__main__":
    unittest.main()
