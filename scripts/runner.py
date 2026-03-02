#!/usr/bin/env python3
"""통합 러너.

Usage:
    uv run scripts/runner.py --list                       # 목록 출력
    uv run scripts/runner.py 3.2.7_environment_variables  # 예제 실행
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def discover_examples() -> dict[str, Path]:
    """src/chapter_XX/X.Y.Z_*/main.py 패턴으로 예제를 탐색. 폴더명 → 경로 매핑."""
    examples = sorted(SRC.glob("chapter_*/*/main.py"))
    return {ex.parent.name: ex for ex in examples}


def chapter_num(main_py: Path) -> int:
    chapter_dir = main_py.parent.parent.name  # e.g. 'chapter_03'
    return int(chapter_dir.split("_")[1])


def cmd_list(examples: dict[str, Path]) -> None:
    current_chapter = -1
    for name, ex in examples.items():
        ch = chapter_num(ex)
        if ch != current_chapter:
            print(f"\nChapter {ch:02d}")
            current_chapter = ch
        print(f"  {name}")
    print(f"\nTotal: {len(examples)} examples")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="통합 러너",
        usage="uv run scripts/runner.py [example | --list]",
    )
    parser.add_argument(
        "example",
        nargs="?",
        help="실행할 예제 폴더명 (e.g. 3.2.7_environment_variables)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="예제 목록 출력",
    )
    args = parser.parse_args()

    examples = discover_examples()

    if args.list:
        cmd_list(examples)
        return

    if args.example:
        main_py = examples.get(args.example)
        if not main_py:
            print(f"No match for '{args.example}'")
            print("Available examples:")
            cmd_list(examples)
            sys.exit(1)
        print(f"Running {args.example}...\n")
        cp = subprocess.run(
            [sys.executable, str(main_py)],
            cwd=main_py.parent,
        )
        sys.exit(cp.returncode)

    parser.print_help()


if __name__ == "__main__":
    main()
