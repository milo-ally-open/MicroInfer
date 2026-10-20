"""驗證性能報告統計邏輯，不連接 Gateway 或推論服務。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from performance_baseline import build_report, distribution, percentile  # noqa: E402


class PerformanceReportTests(unittest.TestCase):
    def test_nearest_rank_percentiles(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(values, 0.50), 2.0)
        self.assertEqual(percentile(values, 0.95), 4.0)
        self.assertIsNone(percentile([], 0.95))

    def test_report_includes_errors_latency_ttft_and_upload(self) -> None:
        report = build_report(
            [
                {"kind": "chat_sse", "success": True, "latency_seconds": 2.0, "sse_ttft_seconds": 0.4},
                {"kind": "video_upload", "success": True, "latency_seconds": 5.0, "video_upload_seconds": 1.2},
                {"kind": "chat_sse", "success": False, "latency_seconds": 0.1, "error_class": "GatewayAPIError"},
            ],
            elapsed=7.0,
            mode="mixed",
            concurrency=2,
        )
        self.assertEqual(report["request_count"], 3)
        self.assertEqual(report["error_rate"], 1 / 3)
        self.assertEqual(report["sse_time_to_first_token_seconds"]["p50"], 0.4)
        self.assertEqual(report["video_upload_completion_seconds"]["p50"], 1.2)
        self.assertIsNone(report["sla"])

    def test_empty_distribution_is_explicit(self) -> None:
        self.assertIsNone(distribution([])["p95"])


if __name__ == "__main__":
    unittest.main()
