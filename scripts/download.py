"""以明確指定的 Hugging Face 模型、revision 與目標目錄下載完整快照。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """建立通用模型快照下載器的命令列參數。"""
    parser = argparse.ArgumentParser(
        description="下載 Hugging Face Hub 上指定 revision 的完整模型快照。"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face 模型識別碼，例如 microsoft/Mage-VL。",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="要下載的 branch、tag 或 commit SHA；建議使用固定 commit SHA。",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        type=Path,
        help="模型檔案的明確下載目標目錄。",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="只下載符合的檔案模式；可重複指定。未指定時下載完整快照。",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="略過符合的檔案模式；可重複指定。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="並行下載的工作數，預設為 8。",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="讀取 Hugging Face 存取權杖的環境變數名稱，預設為 HF_TOKEN。",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="忽略本機已快取檔案並重新下載。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_workers < 1:
        parser.error("--max-workers 必須大於或等於 1。")

    destination = args.local_dir.expanduser().resolve()
    token = os.environ.get(args.token_env) or None

    try:
        from huggingface_hub import snapshot_download

        downloaded_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            local_dir=str(destination),
            allow_patterns=args.allow_pattern,
            ignore_patterns=args.ignore_pattern,
            max_workers=args.max_workers,
            token=token,
            force_download=args.force_download,
        )
    except Exception as exc:
        parser.exit(2, f"下載失敗：{exc}\n")

    print(f"repo_id: {args.repo_id}")
    print(f"revision: {args.revision}")
    print(f"local_dir: {downloaded_path}")
    print("download_status: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
