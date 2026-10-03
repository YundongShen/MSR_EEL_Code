"""Generate Tier-3 (scope-creep) hunks using Claude API.

For each SWE-bench instance:
  1. Build code context from source_files (gold patch file focus)
  2. Call Claude with unconstrained prompt: "Fix the bug. Make any changes necessary."
  3. Parse Claude's SEARCH/REPLACE response → unified diff
  4. Extract hunks that don't overlap with gold patch → Tier-3
  5. Save to tier3_hunks.jsonl (one line per instance with ≥1 Tier-3 hunk)

No repo cloning or test execution needed. Tier-3 is determined by diff comparison only.

Output format:
  {"instance_id": "...", "tier3_hunks": [
    {"filepath": "...", "hunk_diff": "...",
     "context_before": [...], "context_after": [...], "tier_label": 3}
  ]}

Usage:
  # Smoke test (5 sympy instances, ~$0.10):
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/generate_tier3.py --max 5 --repo-filter sympy/sympy

  # Full run via SLURM:
  sbatch scripts/generate_tier3.slurm
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.llm_client import LLMClient
from data.patch_generator import (
    _UNCONSTRAINED_SYSTEM,
    _CLAUDE_CODE_BUDGET,
    _relevant_context,
    _extract_diff_file_paths,
    _fuzzy_find,
)
from data.data_loader import DataSample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SR parsing — handles both standard (with FILE:) and compact (no FILE:) formats
# ---------------------------------------------------------------------------

# Standard format: <<<<<<< SEARCH / FILE: path / old / ======= / new / >>>>>>> REPLACE
_SR_WITH_FILE = re.compile(
    r"<<<<<<< SEARCH\s*\nFILE:\s*(.+?)\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
    re.DOTALL,
)
# Compact format (Haiku): <<<<<<< SEARCH / old / ======= / new / >>>>>>> REPLACE
_SR_NO_FILE = re.compile(
    r"<<<<<<< SEARCH\s*\n(?!FILE:)(.*?)=======\n(.*?)>>>>>>> REPLACE",
    re.DOTALL,
)


def _parse_sr_blocks_extended(
    text: str,
    source_files: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Parse SR blocks from LLM output, supporting both FILE: and no-FILE: formats.

    For blocks without a FILE: header (common in Haiku), the file is inferred by
    fuzzy-matching the SEARCH text against all source files.
    """
    blocks: list[tuple[str, str, str]] = []

    # Standard format (with FILE: header) — matches first, take precedence
    for m in _SR_WITH_FILE.finditer(text):
        blocks.append((m.group(1).strip(), m.group(2), m.group(3)))

    if blocks:
        return blocks

    # Fallback: compact format — infer file from content
    for m in _SR_NO_FILE.finditer(text):
        old_text = m.group(1)
        new_text = m.group(2)
        # Try to find which source file contains this SEARCH text
        best_file = None
        for filepath, src in source_files.items():
            if _fuzzy_find(src, old_text) is not None:
                best_file = filepath
                break
        if best_file:
            blocks.append((best_file, old_text, new_text))
        else:
            log.debug("No-FILE SR block: SEARCH text not found in any source file")

    return blocks


# ---------------------------------------------------------------------------
# SR application helpers
# ---------------------------------------------------------------------------

def _apply_sr_to_files(
    source_files: dict[str, str],
    blocks: list[tuple[str, str, str]],
) -> dict[str, str]:
    """Apply SR blocks, returning {filepath: new_content} for each modified file."""
    modified: dict[str, str] = {}
    for filepath, old_text, new_text in blocks:
        resolved = filepath
        if resolved not in source_files:
            candidates = [k for k in source_files if k.endswith(resolved)]
            if len(candidates) == 1:
                resolved = candidates[0]
            else:
                continue
        if resolved not in modified:
            modified[resolved] = source_files[resolved]
        idx = _fuzzy_find(modified[resolved], old_text)
        if idx is not None:
            modified[resolved] = (
                modified[resolved][:idx] + new_text + modified[resolved][idx + len(old_text):]
            )
    return modified


def _apply_sr_blocks_partial(
    source_files: dict[str, str],
    blocks: list[tuple[str, str, str]],
) -> str:
    """Apply SEARCH/REPLACE blocks, skipping any that don't match.

    Unlike the original _apply_sr_blocks, this does not abort on first failure.
    Returns a unified diff of all successfully applied blocks (may be empty).
    """
    import difflib

    parts: list[str] = []
    for filepath, old_text, new_text in blocks:
        # Resolve path
        if filepath not in source_files:
            candidates = [k for k in source_files if k.endswith(filepath)]
            if len(candidates) == 1:
                filepath = candidates[0]
            else:
                log.debug("SR: cannot resolve file '%s'", filepath)
                continue

        original = source_files[filepath]
        idx = _fuzzy_find(original, old_text)
        if idx is None:
            log.debug("SR: SEARCH block not found in '%s' (skipping)", filepath)
            continue

        modified = original[:idx] + new_text + original[idx + len(old_text):]
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
        ))
        if diff_lines:
            chunk = "".join(diff_lines)
            if not chunk.endswith("\n"):
                chunk += "\n"
            parts.append(chunk)

    return "".join(parts)


# ---------------------------------------------------------------------------
# Hunk parsing (self-contained, no dependency on parse_instances.py)
# ---------------------------------------------------------------------------

def _parse_hunks_simple(patch: str) -> list[dict]:
    """Split a unified diff into hunk records with file + start_line."""
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
                hunks.append({
                    "filepath": current_file,
                    "old_start": old_start,
                    "lines": list(current_lines),
                })
            current_lines = [raw]
            m = re.search(r"@@ -(\d+)", raw)
            old_start = int(m.group(1)) if m else 1
        elif current_lines is not None:
            current_lines.append(raw)

    if current_lines and current_file and old_start is not None:
        hunks.append({
            "filepath": current_file,
            "old_start": old_start,
            "lines": list(current_lines),
        })
    return hunks


def _is_extra_hunk(
    claude_file: str,
    claude_start: int,
    gold_hunks: list[dict],
    threshold: int = 20,
) -> bool:
    """Return True if this Claude hunk doesn't overlap any gold hunk.

    A hunk is considered extra (Tier-3) if:
    - It modifies a file not touched by gold at all, OR
    - It modifies a file in gold but the start line is >=threshold away from
      all gold hunks in that file.
    """
    same_file_gold = [gh for gh in gold_hunks if gh["filepath"] == claude_file]
    if not same_file_gold:
        return True  # different file entirely → definitely Tier-3
    return all(abs(gh["old_start"] - claude_start) >= threshold for gh in same_file_gold)


def _extract_context(
    source_files: dict[str, str],
    filepath: str,
    old_start: int,
    hunk_lines: list[str],
    ctx: int = 5,
) -> tuple[list[str], list[str]]:
    """Extract ctx lines before and after the hunk from source_files."""
    if filepath not in source_files:
        return [], []
    src = source_files[filepath].splitlines()
    before_end = max(0, old_start - 1)
    before_start = max(0, before_end - ctx)
    context_before = src[before_start:before_end]

    old_count = sum(
        1 for l in hunk_lines[1:]
        if l.startswith(" ") or l.startswith("-")
    )
    after_start = old_start - 1 + old_count
    context_after = src[after_start: after_start + ctx]
    return context_before, context_after


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def _build_user_prompt(inst: dict) -> str:
    """Build the user prompt, using the full Claude code budget for context."""
    source_files: dict[str, str] = inst.get("source_files", {})
    gold_diff: str = inst.get("patch", "")
    issue_text: str = inst.get("problem_statement", "")

    sample = DataSample(
        sample_id=inst.get("instance_id", ""),
        issue_text=issue_text,
        old_codebase=source_files,
        golden_diff=gold_diff,
        test_suite=inst.get("test_files", {}),
    )

    # Use the large Claude budget so entire relevant files are shown without gaps
    code_ctx = _relevant_context(sample, budget=_CLAUDE_CODE_BUDGET)
    gold_files = _extract_diff_file_paths(gold_diff)
    file_hint = (
        f"\nThe file(s) to modify are: {', '.join(gold_files)}\n"
        if gold_files else ""
    )
    return (
        f"## Issue\n{issue_text}\n"
        f"{file_hint}\n"
        f"## Codebase\n{code_ctx}\n\n"
        "Fix the issue using SEARCH/REPLACE blocks as instructed."
    )


def process_instance(inst: dict, llm: LLMClient) -> dict:
    """Process one instance. Returns a record dict with raw_response, claude_diff, tier3_hunks.

    Always returns the LLM response for caching, even when no Tier-3 hunks are extracted.
    Returns None only if we cannot attempt generation (no gold hunks, or LLM failure).
    """
    source_files: dict[str, str] = inst.get("source_files", {})
    gold_diff: str = inst.get("patch", "")

    gold_hunks = _parse_hunks_simple(gold_diff)
    if not gold_hunks:
        log.debug("%s: no gold hunks, skipping", inst.get("instance_id"))
        return None

    user = _build_user_prompt(inst)
    raw = llm.complete(_UNCONSTRAINED_SYSTEM, user)
    if raw is None:
        log.warning("%s: LLM returned None", inst.get("instance_id"))
        return None

    # Parse SR blocks → unified diff (extended parser handles Haiku's no-FILE: format)
    blocks = _parse_sr_blocks_extended(raw, source_files)
    claude_diff = _apply_sr_blocks_partial(source_files, blocks) if blocks else ""

    # Extract Tier-3 hunks (extra hunks not overlapping gold)
    tier3: list[dict] = []
    if claude_diff:
        for ch in _parse_hunks_simple(claude_diff):
            if not _is_extra_hunk(ch["filepath"], ch["old_start"], gold_hunks):
                continue
            hunk_diff = "".join(ch["lines"])
            ctx_before, ctx_after = _extract_context(
                source_files, ch["filepath"], ch["old_start"], ch["lines"]
            )
            tier3.append({
                "filepath": ch["filepath"],
                "old_start_line": ch["old_start"],
                "hunk_diff": hunk_diff,
                "context_before": ctx_before,
                "context_after": ctx_after,
                "tier_label": 3,
            })

    return {
        "raw_response": raw,
        "claude_diff": claude_diff,
        "modified_files": _apply_sr_to_files(source_files, blocks) if blocks else {},
        "tier3_hunks": tier3,
        "sr_blocks_found": len(blocks),
        "sr_blocks_applied": len([b for b in blocks if _fuzzy_find(
            source_files.get(b[0], source_files.get(
                next((k for k in source_files if k.endswith(b[0])), b[0]), ""
            )), b[1]
        ) is not None]) if blocks else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/raw/swebench_full_instances.jsonl",
        help="Raw SWE-bench JSONL with source_files and patch fields",
    )
    parser.add_argument(
        "--output", default="data/cache/tier3_hunks.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        help="Claude model ID",
    )
    parser.add_argument(
        "--max", type=int, default=None, dest="max_instances",
        help="Stop after this many instances (for smoke testing)",
    )
    parser.add_argument(
        "--repo-filter", default=None,
        help="Only process instances from this repo, e.g. sympy/sympy",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.3,
        help="Seconds to sleep between API calls (rate limiting)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Max tokens for LLM response (default 4096 to avoid cutoff)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default 0.7)",
    )
    parser.add_argument(
        "--provider", default="anthropic",
        help="LLM provider: anthropic or openai (for vLLM/Qwen)",
    )
    parser.add_argument(
        "--api-base", default=None,
        help="API base URL for OpenAI-compatible endpoints (vLLM)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cache file: stores raw LLM responses for every processed instance.
    # Named alongside the output file so they stay together.
    # Allows re-extracting Tier-3 with improved logic without re-calling the API.
    cache_path = out_path.with_suffix(".cache.jsonl")

    # Resume: collect already-written instance_ids from the main output.
    # Also index the cache so we can reuse responses without API calls.
    done_ids: set[str] = set()
    cached_responses: dict[str, dict] = {}  # instance_id → cached record

    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
    if cache_path.exists():
        with open(cache_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    cached_responses[rec["instance_id"]] = rec
                    done_ids.add(rec["instance_id"])
                except Exception:
                    pass
    if done_ids:
        log.info("Resuming: %d already processed (output=%d cache=%d)",
                 len(done_ids), len(done_ids) - len(cached_responses), len(cached_responses))

    llm = LLMClient(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=2,
        retry_delay=5.0,
    )

    processed = yielded = skipped = 0

    with open(args.input) as fin, \
         open(out_path, "a") as fout, \
         open(cache_path, "a") as fcache:

        for lineno, raw_line in enumerate(fin, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                inst = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", lineno, exc)
                continue

            iid = inst.get("instance_id", f"line_{lineno}")

            if args.repo_filter and inst.get("repo") != args.repo_filter:
                continue
            if iid in done_ids:
                skipped += 1
                continue
            if args.max_instances and processed >= args.max_instances:
                break

            processed += 1
            log.info("[%d] %s", processed, iid)

            result = process_instance(inst, llm)
            if result is None:
                log.info("  → skipped (no gold hunks or LLM failure)")
                continue

            # Build universal cache record — self-contained, reusable for future tasks.
            # Two logical layers in one record:
            #   "meta"       — SWE-bench instance identity, repo, problem statement
            #   "generation" — everything the LLM produced (raw + extracted)
            #   "annotations"— experiment-specific labels (Tier-3 hunks)
            import datetime
            cache_record = {
                # ── Universal layer (HuggingFace-compatible) ──────────────────
                "id": iid,
                "source": "swebench",
                "repo": inst.get("repo", ""),
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "source_files": inst.get("source_files", {}),
                "meta": {
                    "base_commit": inst.get("base_commit", ""),
                    "problem_statement": inst.get("problem_statement", ""),
                    "gold_patch": inst.get("patch", ""),
                    "fail_to_pass": inst.get("fail_to_pass", []),
                },
                "generation": {
                    "model": args.model,
                    "provider": args.provider,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "system_prompt": _UNCONSTRAINED_SYSTEM,
                    "raw_response": result["raw_response"],
                    "extracted_diff": result["claude_diff"],
                    "modified_files": result["modified_files"],
                    "sr_blocks_found": result["sr_blocks_found"],
                    "sr_blocks_applied": result["sr_blocks_applied"],
                },
                # ── Experiment-specific layer ─────────────────────────────────
                "annotations": {
                    "tier3_hunks": result["tier3_hunks"],
                    "has_scope_creep": bool(result["tier3_hunks"]),
                },
                # Flat alias for our pipeline's direct lookup
                "instance_id": iid,
                "tier3_hunks": result["tier3_hunks"],
            }
            fcache.write(json.dumps(cache_record) + "\n")
            fcache.flush()

            # Write Tier-3 hunks to main output only when present
            if result["tier3_hunks"]:
                record = {"instance_id": iid, "tier3_hunks": result["tier3_hunks"]}
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                yielded += 1
                log.info("  → %d Tier-3 hunk(s) saved  (SR: %d/%d applied)",
                         len(result["tier3_hunks"]),
                         result["sr_blocks_applied"], result["sr_blocks_found"])
            else:
                log.info("  → no Tier-3 hunks  (SR: %d/%d applied)",
                         result["sr_blocks_applied"], result["sr_blocks_found"])

            if args.sleep > 0:
                time.sleep(args.sleep)

    log.info(
        "Done. processed=%d  yielded=%d  skipped(resume)=%d  yield_rate=%.1f%%",
        processed, yielded, skipped,
        100.0 * yielded / processed if processed else 0,
    )
    log.info("Cache: %s", cache_path)


if __name__ == "__main__":
    main()
