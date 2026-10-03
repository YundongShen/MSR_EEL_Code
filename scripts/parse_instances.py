"""
Parse SWE-bench JSONL instances into structured training data for Edit Entailment Learning.

Extracts four entity types per instance (all from JSONL, no repo needed):
  REQ  - requirement text (problem_statement)
  TEST - test function bodies (from test_files, filtered by fail_to_pass IDs)
  ORIG - original project code units (functions/classes from source_files)
  HUNK - gold patch hunks with surrounding source context

Output: data/processed/instances_lite.jsonl or instances_full.jsonl
        One line per instance, all entity types included.
Resumable: skips already-written instance_ids on restart.

Usage:
  # Experimental (Lite, 82 instances):
  python scripts/parse_instances.py \
      --input data/raw/swebench_instances.jsonl \
      --output data/processed/instances_lite.jsonl

  # Later (Full, 2294 instances):
  python scripts/parse_instances.py \
      --input data/raw/swebench_full_instances.jsonl \
      --output data/processed/instances_full.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Hunk parsing
# ---------------------------------------------------------------------------

def _parse_hunks(patch: str, source_files: dict[str, str]) -> list[dict]:
    """Split a unified diff into per-hunk records with source context."""
    hunks: list[dict] = []
    current_file: str | None = None
    current_lines: list[str] = []
    old_start: int | None = None

    for raw in patch.splitlines(keepends=True):
        if raw.startswith("--- "):
            m = re.match(r"^--- (?:a/)?(.+)", raw)
            current_file = m.group(1).strip() if m else None
            continue
        if raw.startswith("+++ "):
            continue
        if raw.startswith("@@"):
            if current_lines and current_file and old_start is not None:
                hunks.append(_build_hunk(
                    current_file, old_start, current_lines,
                    source_files, len(hunks),
                ))
            current_lines = [raw]
            m = re.search(r"@@ -(\d+)", raw)
            old_start = int(m.group(1)) if m else 1
        elif current_lines is not None:
            current_lines.append(raw)

    if current_lines and current_file and old_start is not None:
        hunks.append(_build_hunk(
            current_file, old_start, current_lines,
            source_files, len(hunks),
        ))
    return hunks


def _build_hunk(
    filepath: str,
    old_start: int,
    hunk_lines: list[str],
    source_files: dict[str, str],
    idx: int,
) -> dict:
    hunk_diff = "".join(hunk_lines)
    context_before: list[str] = []
    context_after: list[str] = []

    if filepath in source_files:
        src = source_files[filepath].splitlines()
        # context_before: 5 lines before the hunk in the old file (1-based → 0-based)
        before_end = max(0, old_start - 1)
        before_start = max(0, before_end - 5)
        context_before = src[before_start:before_end]

        # old lines consumed (context lines ' ' + removed lines '-')
        old_count = sum(
            1 for l in hunk_lines[1:]
            if l.startswith(" ") or l.startswith("-")
        )
        after_start = old_start - 1 + old_count
        context_after = src[after_start: after_start + 5]

    return {
        "hunk_id": idx,
        "filepath": filepath,
        "old_start_line": old_start,
        "hunk_diff": hunk_diff,
        "context_before": context_before,
        "context_after": context_after,
        "tier_label": None,  # filled in during coverage analysis
    }


# ---------------------------------------------------------------------------
# Source code unit extraction
# ---------------------------------------------------------------------------

def _extract_code_units(filepath: str, source: str) -> list[dict]:
    """Extract top-level and class-level functions/classes via AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    src_lines = source.splitlines()
    units = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not hasattr(node, "end_lineno"):
            continue
        start = node.lineno - 1
        end = node.end_lineno
        units.append({
            "filepath": filepath,
            "name": node.name,
            "kind": "class" if isinstance(node, ast.ClassDef) else "function",
            "start_line": node.lineno,
            "end_line": end,
            "code": "\n".join(src_lines[start:end]),
        })
    return units


# ---------------------------------------------------------------------------
# Test function extraction
# ---------------------------------------------------------------------------

def _extract_test_functions(
    test_files: dict[str, str],
    fail_to_pass_ids: list[str],
) -> list[dict]:
    """Extract test function bodies for the given fail_to_pass test IDs."""
    # Map filepath → set of bare function names we need
    needed: dict[str, set[str]] = {}
    for tid in fail_to_pass_ids:
        parts = tid.split("::")
        filepath = parts[0]
        # Strip parametrize suffix e.g. test_foo[case0] → test_foo
        func_name = parts[-1].split("[")[0]
        needed.setdefault(filepath, set()).add(func_name)

    results = []
    for filepath, source in test_files.items():
        if filepath not in needed:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        src_lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in needed[filepath]:
                continue
            if not hasattr(node, "end_lineno"):
                continue
            start = node.lineno - 1
            end = node.end_lineno
            results.append({
                "filepath": filepath,
                "function_name": node.name,
                "code": "\n".join(src_lines[start:end]),
            })
    return results


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def process_instance(inst: dict) -> dict:
    source_files: dict[str, str] = inst.get("source_files", {})
    test_files: dict[str, str] = inst.get("test_files", {})
    fail_to_pass: list[str] = inst.get("fail_to_pass", [])

    gold_hunks = _parse_hunks(inst.get("patch", ""), source_files)
    source_units = [
        unit
        for filepath, source in source_files.items()
        for unit in _extract_code_units(filepath, source)
    ]
    test_functions = _extract_test_functions(test_files, fail_to_pass)

    return {
        "instance_id": inst["instance_id"],
        "repo": inst["repo"],
        "requirement": inst["problem_statement"],
        "fail_to_pass_ids": fail_to_pass,
        "gold_hunks": gold_hunks,
        "source_units": source_units,
        "test_functions": test_functions,
        "test_patch": inst.get("test_patch", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: collect already-processed instance_ids
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} already done", flush=True)

    written = skipped = total = 0
    with open(args.input) as fin, open(out_path, "a") as fout:
        for lineno, raw in enumerate(fin, 1):
            if not raw.strip():
                continue
            try:
                inst = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skipping malformed line {lineno}: {exc}", flush=True)
                continue
            iid = inst["instance_id"]
            total += 1
            if iid in done_ids:
                skipped += 1
                continue
            record = process_instance(inst)
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            written += 1
            print(
                f"[{written + skipped}/{total}] {iid}"
                f"  hunks={len(record['gold_hunks'])}"
                f"  src_units={len(record['source_units'])}"
                f"  tests={len(record['test_functions'])}",
                flush=True,
            )

    print(f"Done. written={written} skipped={skipped} total={total}", flush=True)


if __name__ == "__main__":
    main()
