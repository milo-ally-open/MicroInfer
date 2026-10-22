"""以序列快照驗證批次排程計畫，不載入模型或 CUDA。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

INFERENCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "inference"
sys.path.insert(0, str(INFERENCE_ROOT))

from engine.batch_planner import SequenceProgress, plan_batch


class BatchPlannerTests(unittest.TestCase):
    def test_decode_is_scheduled_before_prefill_and_budget_is_shared(self) -> None:
        plan = plan_batch(
            [
                SequenceProgress("decode-a", "decode", 10, 10, 2),
                SequenceProgress("prefill-a", "prefill", 12, 0),
                SequenceProgress("decode-b", "decode", 8, 8, 1),
            ],
            max_num_seqs=3,
            max_num_scheduled_tokens=5,
        )

        self.assertEqual(
            [(item.request_id, item.token_budget) for item in plan.sequences],
            [("decode-a", 1), ("decode-b", 1), ("prefill-a", 3)],
        )
        self.assertEqual(plan.scheduled_tokens, 5)
        self.assertFalse(plan.sequences[-1].final_prefill_chunk)

    def test_prefill_chunks_keep_offsets_and_mark_last_chunk(self) -> None:
        first = plan_batch(
            [SequenceProgress("request-a", "prefill", 7, 0)],
            max_num_seqs=2,
            max_num_scheduled_tokens=4,
            prefill_chunk_size=3,
        )
        second = plan_batch(
            [SequenceProgress("request-a", "prefill", 7, 3)],
            max_num_seqs=2,
            max_num_scheduled_tokens=4,
            prefill_chunk_size=3,
        )
        third = plan_batch(
            [SequenceProgress("request-a", "prefill", 7, 6)],
            max_num_seqs=2,
            max_num_scheduled_tokens=4,
            prefill_chunk_size=3,
        )

        self.assertEqual([first.sequences[0].prompt_offset, second.sequences[0].prompt_offset], [0, 3])
        self.assertFalse(first.sequences[0].final_prefill_chunk)
        self.assertTrue(third.sequences[0].final_prefill_chunk)
        self.assertEqual(third.sequences[0].token_budget, 1)

    def test_cursor_rotates_when_batch_sequence_limit_is_reached(self) -> None:
        progress = [SequenceProgress(f"request-{index}", "decode", 1, 1) for index in range(4)]
        first = plan_batch(progress, max_num_seqs=2, max_num_scheduled_tokens=8, cursor=0)
        second = plan_batch(progress, max_num_seqs=2, max_num_scheduled_tokens=8, cursor=first.next_cursor)

        self.assertEqual([item.request_id for item in first.sequences], ["request-0", "request-1"])
        self.assertEqual([item.request_id for item in second.sequences], ["request-2", "request-3"])

    def test_decode_and_prefill_requests_are_unique_in_each_batch(self) -> None:
        plan = plan_batch(
            [
                SequenceProgress("request-a", "prefill", 5, 1),
                SequenceProgress("request-b", "decode", 3, 3),
            ],
            max_num_seqs=4,
            max_num_scheduled_tokens=5,
        )
        request_ids = [item.request_id for item in plan.sequences]

        self.assertEqual(len(request_ids), len(set(request_ids)))
        self.assertLessEqual(plan.scheduled_tokens, 5)

    def test_visual_prefix_is_kept_in_first_chunk_and_mixed_with_decode(self) -> None:
        plan = plan_batch(
            [
                SequenceProgress("decode", "decode", 8, 8, 1),
                SequenceProgress(
                    "visual-prefill", "prefill", 12, 0,
                    prefill_min_chunk_tokens=5,
                ),
                SequenceProgress("text-prefill", "prefill", 10, 0),
            ],
            max_num_seqs=3,
            max_num_scheduled_tokens=8,
            prefill_chunk_size=3,
        )

        self.assertEqual(
            [(item.request_id, item.phase, item.prompt_offset, item.token_budget)
             for item in plan.sequences],
            [
                ("visual-prefill", "prefill", 0, 5),
                ("decode", "decode", 0, 1),
                ("text-prefill", "prefill", 0, 2),
            ],
        )
        self.assertEqual(plan.scheduled_tokens, 8)

    def test_invalid_progress_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            plan_batch(
                [SequenceProgress("request-a", "prefill", 4, 4)],
                max_num_seqs=1,
                max_num_scheduled_tokens=4,
            )

        with self.assertRaises(ValueError):
            plan_batch(
                [SequenceProgress("request-a", "other", 4, 4)],
                max_num_seqs=1,
                max_num_scheduled_tokens=4,
            )


if __name__ == "__main__":
    unittest.main()
