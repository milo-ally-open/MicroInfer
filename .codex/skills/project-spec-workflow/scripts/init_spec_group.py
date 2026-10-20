#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path


FILES_BY_MODE = {
    "light": ("proposal.md",),
    "standard": ("proposal.md", "spec.md", "design.md", "tasks.md"),
    "high-risk": ("proposal.md", "spec.md", "design.md", "tasks.md"),
    "bug": ("proposal.md", "spec.md", "design.md", "tasks.md"),
    "prototype": ("proposal.md", "tasks.md"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Create a specification group from this skill's templates."
    )
    parser.add_argument("--root", default=".", help="repository root used to discover a documentation root (default: current directory)")
    parser.add_argument("--spec-root", help="specification directory that will contain the new group")
    parser.add_argument("--template-root", help="template directory (default: this skill's templates)")
    parser.add_argument("--name", required=True, help="new group name using lowercase letters, digits, hyphens, or underscores")
    parser.add_argument("--title", required=True, help="human-readable group title")
    parser.add_argument("--mode", choices=FILES_BY_MODE, default="standard")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9]+)?", args.name):
        parser.error("--name must be a simple lowercase directory name without path separators")

    root = Path(args.root).resolve()
    template_root = Path(args.template_root).resolve() if args.template_root else Path(__file__).resolve().parent.parent / "templates"
    if args.spec_root:
        spec_root = Path(args.spec_root).resolve()
    else:
        candidates = [root / "docs" / "Specification", root / "project" / "docs" / "Specification", root / "Specification"]
        matches = [path for path in candidates if path.is_dir()]
        if not matches:
            parser.error("could not discover a specification directory; pass --spec-root")
        if len(matches) > 1:
            parser.error("multiple specification directories found; pass --spec-root to choose one")
        spec_root = matches[0]
    if not spec_root.is_dir():
        parser.error("specification root does not exist: " + str(spec_root))
    if not template_root.is_dir():
        parser.error("template directory does not exist: " + str(template_root))

    destination = spec_root / args.name
    if destination.exists():
        parser.error("target group already exists; refusing to overwrite it")

    names = FILES_BY_MODE[args.mode]
    missing = [name for name in names if not (template_root / name).is_file()]
    if missing:
        parser.error("missing template(s): " + ", ".join(missing))

    destination.mkdir()
    try:
        for name in names:
            content = (template_root / name).read_text(encoding="utf-8")
            content = content.replace("<简短名称>", args.title)
            (destination / name).write_text(content, encoding="utf-8", newline="\n")
    except OSError as exc:
        print("Could not finish creating the new group: " + str(exc), file=sys.stderr)
        print("Partial files may remain at: " + str(destination), file=sys.stderr)
        return 1

    print("Created " + str(destination))
    for name in names:
        print("  " + name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
