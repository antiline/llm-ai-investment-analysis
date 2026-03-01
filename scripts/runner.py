#!/usr/bin/env python3
"""통합 러너.

Usage:
    uv run scripts/runner.py 3.2.7          # 특정 섹션 실행
    uv run scripts/runner.py --list         # 목록 출력
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def discover_examples() -> list[Path]:
    """src/chapter_XX/X.Y.Z_*/main.py 패턴으로 예제를 탐색."""
    return sorted(SRC.glob("chapter_*/*/main.py"))


def section_id(main_py: Path) -> str:
    """main.py 경로에서 섹션 번호를 추출. e.g. '3.2.7'"""
    return main_py.parent.name.split("_")[0]


def section_label(main_py: Path) -> str:
    """main.py 경로에서 섹션 레이블을 추출."""
    return main_py.parent.name


def chapter_num(main_py: Path) -> int:
    """main.py 경로에서 챕터 번호를 추출."""
    return int(main_py.parent.parent.name.split("_")[1])


def fuzzy_match(query: str, examples: list[Path]) -> list[Path]:
    """섹션 번호 fuzzy match."""
    matched = []
    for ex in examples:
        sid = section_id(ex)
        label = section_label(ex)
        if sid == query or label == query:
            matched.append(ex)
        elif sid.startswith(query):
            matched.append(ex)
        elif query.replace(".", "") == sid.replace(".", ""):
            matched.append(ex)
    return matched


def run_example(main_py: Path) -> dict:
    """예제 하나를 실행하고 결과를 반환."""
    sid = section_id(main_py)
    label = section_label(main_py)

    start = time.time()
    try:
        cp = subprocess.run(
            [sys.executable, str(main_py)],
            cwd=main_py.parent,
            text=True,
            capture_output=True,
            timeout=120,
        )
        elapsed = time.time() - start
        return {
            "section": sid,
            "label": label,
            "status": "ok" if cp.returncode == 0 else "failed",
            "returncode": cp.returncode,
            "elapsed": round(elapsed, 2),
            "stdout": cp.stdout[:1000] if cp.stdout else "",
            "stderr": cp.stderr[:1000] if cp.stderr else "",
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "section": sid,
            "label": label,
            "status": "timeout",
            "returncode": -1,
            "elapsed": round(elapsed, 2),
            "stdout": "",
            "stderr": "Timeout after 120s",
        }


def print_result(result: dict) -> None:
    """실행 결과 한 건을 출력."""
    icon = {"ok": "\u2705", "failed": "\u274c", "timeout": "\u23f0"}.get(
        result["status"], "?"
    )
    print(f"  {icon} {result['section']} {result['label']} ({result['elapsed']}s)")
    if result["status"] != "ok":
        err = result["stderr"] or result["stdout"]
        if err:
            for line in err.strip().split("\n")[-3:]:
                print(f"     {line}")


def cmd_list(examples: list[Path]) -> None:
    """예제 목록을 출력."""
    current_chapter = -1
    for ex in examples:
        ch = chapter_num(ex)
        if ch != current_chapter:
            print(f"\nChapter {ch:02d}")
            current_chapter = ch
        print(f"  {section_id(ex):8s} {section_label(ex)}")
    print(f"\nTotal: {len(examples)} examples")


def cmd_run(targets: list[Path]) -> None:
    """대상 예제를 실행."""
    total = len(targets)
    ok = 0
    fail = 0

    print(f"Running {total} example(s)...\n")
    for i, ex in enumerate(targets, 1):
        print(f"[{i}/{total}] {section_id(ex)} {section_label(ex)}")
        result = run_example(ex)
        print_result(result)
        if result["status"] == "ok":
            ok += 1
        else:
            fail += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {ok} passed, {fail} failed, {total} total")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="통합 러너",
        usage="uv run scripts/runner.py [section | --list]",
    )
    parser.add_argument(
        "section",
        nargs="?",
        help="실행할 섹션 번호 (e.g. 3.2.7)",
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

    if args.section:
        targets = fuzzy_match(args.section, examples)
        if not targets:
            print(f"No match for '{args.section}'")
            print("Available sections:")
            cmd_list(examples)
            sys.exit(1)
        cmd_run(targets)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
