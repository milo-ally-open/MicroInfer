"""以明确指定的 Hugging Face 模型、revision 与目标目录下载完整快照。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """建立通用模型快照下载器的命令列参数。"""
    parser = argparse.ArgumentParser(
        description="下载 Hugging Face Hub 上指定 revision 的完整模型快照。"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face 模型识别码，例如 microsoft/Mage-VL。",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="要下载的 branch、tag 或 commit SHA；建议使用固定 commit SHA。",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        type=Path,
        help="模型档案的明确下载目标目录。",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="只下载符合的档案模式；可重复指定。未指定时下载完整快照。",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="略过符合的档案模式；可重复指定。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="并行下载的工作数，预设为 8。",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="读取 Hugging Face 存取权杖的环境变数名称，预设为 HF_TOKEN。",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="忽略本机已快取档案并重新下载。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_workers < 1:
        parser.error("--max-workers 必须大于或等于 1。")

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
        parser.exit(2, f"下载失败：{exc}\n")

    print(f"repo_id: {args.repo_id}")
    print(f"revision: {args.revision}")
    print(f"local_dir: {downloaded_path}")
    print("download_status: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
