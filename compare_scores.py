#!/usr/bin/env python
"""Compare two replay output folders (replay_run.py --out) column by column.

    python compare_scores.py /tmp/replay_ref /tmp/replay_new [--rtol 0] [--atol 0]

Timestamp columns are ignored (wall-clock). Every other column must match
exactly (or within --rtol/--atol for floats). Exit code 1 on any difference.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

IGNORE_COLUMNS = {"timestamp", "analysis_timestamp", "watchdog_timestamp"}
CSV_FILES = ["scores.csv", "scores_z.csv", "regression_status_rt.csv", "fd_rt.csv"]


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _same(x: str, y: str, rtol: float, atol: float) -> bool:
    if x == y:
        return True
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return False
    if math.isnan(fx) and math.isnan(fy):
        return True
    return math.isclose(fx, fy, rel_tol=rtol, abs_tol=atol)


def compare_csv(ref: Path, new: Path, rtol: float, atol: float) -> list[str]:
    problems = []
    cols_r, rows_r = _read_csv(ref)
    cols_n, rows_n = _read_csv(new)
    if cols_r != cols_n:
        problems.append(f"{ref.name}: columns differ {cols_r} vs {cols_n}")
    if len(rows_r) != len(rows_n):
        problems.append(f"{ref.name}: {len(rows_r)} vs {len(rows_n)} rows")
    for i, (r, n) in enumerate(zip(rows_r, rows_n)):
        for col in cols_r:
            if col in IGNORE_COLUMNS or col not in n:
                continue
            if not _same(r[col], n[col], rtol, atol):
                problems.append(f"{ref.name} row {i} ({col}): {r[col]!r} vs {n[col]!r}")
                if len(problems) > 50:
                    return problems
    return problems


def compare_motion(ref: Path, new: Path, rtol: float, atol: float) -> list[str]:
    a = [l.split() for l in ref.read_text().splitlines() if l.strip()]
    b = [l.split() for l in new.read_text().splitlines() if l.strip()]
    if len(a) != len(b):
        return [f"motion_rt.1D: {len(a)} vs {len(b)} rows"]
    return [f"motion_rt.1D row {i}: {ra} vs {rb}" for i, (ra, rb) in enumerate(zip(a, b))
            if not all(_same(x, y, rtol, atol) for x, y in zip(ra, rb))][:50]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref", type=Path)
    p.add_argument("new", type=Path)
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--atol", type=float, default=0.0)
    a = p.parse_args()
    problems: list[str] = []
    for name in CSV_FILES:
        r, n = a.ref / name, a.new / name
        if r.exists() != n.exists():
            problems.append(f"{name}: present in only one folder")
        elif r.exists():
            problems.extend(compare_csv(r, n, a.rtol, a.atol))
    if (a.ref / "motion_rt.1D").exists() and (a.new / "motion_rt.1D").exists():
        problems.extend(compare_motion(a.ref / "motion_rt.1D", a.new / "motion_rt.1D", a.rtol, a.atol))
    if problems:
        print("\n".join(problems))
        print(f"\nDIFFERENT ({len(problems)} problem(s) listed)")
        return 1
    print("IDENTICAL (timestamp columns ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
