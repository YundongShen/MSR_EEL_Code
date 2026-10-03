"""Build the candidate dataset (retained + non-retained hunks) from raw LLM generations.

Input : raw generations, one record per SWE-bench issue (data_pipeline/generate_candidates.py),
        the parsed instances (fail-to-pass ids) and the fixed train/val/test split.
Output:
  data/cache/llm_t12_hunks.jsonl    generated hunks that match the reference patch (Tier 1/2)
    {"instance_id", "llm_t12_hunks": [{filepath, old_start_line, hunk_diff,
                                        context_before, context_after, tier_label, sample}]}
  data/cache/tier3_hunks.jsonl      generated hunks that do not match it (Tier 3)
    {"instance_id", "tier3_hunks":   [{... same fields, tier_label = 3}]}
  data/cache/dataset_manifest.json  protocol, counts and evaluation-set sizes per split

Every generated hunk goes to exactly one of the two files, so retained and non-retained
candidates share one generation process and one diff renderer.  Several generation files
(independent samples of the same model) may be given; hunks are united per instance and
de-duplicated on (file, start line, diff text).

Usage:
    python data_pipeline/build_dataset.py
    python data_pipeline/build_dataset.py --generations data/cache/haiku_generations.jsonl \\
                                                  data/cache/haiku_generations_s1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.hunk_matching import (
    MATCH_TOLERANCE,
    extract_context,
    matches_reference,
    parse_hunks,
    retained_tier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(.*)$")


def _has_function_header(hunk_diff: str) -> bool:
    """True if the @@ line carries a function/class name (git-diff style)."""
    m = _HEADER.match(hunk_diff.split("\n", 1)[0])
    return bool(m and m.group(1).strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", nargs="+", default=["data/cache/haiku_generations.jsonl"],
                        help="Raw generation JSONL file(s); several = independent samples")
    parser.add_argument("--instances", default="data/processed/instances_full.jsonl")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--retained-out", default="data/cache/llm_t12_hunks.jsonl")
    parser.add_argument("--tier3-out", default="data/cache/tier3_hunks.jsonl")
    parser.add_argument("--manifest", default="data/cache/dataset_manifest.json")
    parser.add_argument("--tolerance", type=int, default=MATCH_TOLERANCE,
                        help="Max line distance between a generated and a reference hunk start")
    args = parser.parse_args()

    with open(args.splits) as fh:
        sp = json.load(fh)
    split_of: dict[str, str] = {}
    for key, name in (("train_ids", "train"), ("val_ids", "val"), ("test_ids", "test")):
        for iid in sp[key]:
            split_of[iid] = name

    log.info("Loading fail-to-pass ids from %s", args.instances)
    f2p: dict[str, list[str]] = {}
    with open(args.instances) as fh:
        for line in fh:
            rec = json.loads(line)
            f2p[rec["instance_id"]] = rec.get("fail_to_pass_ids", [])

    retained: dict[str, list[dict]] = defaultdict(list)
    tier3: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    order: dict[str, None] = {}
    protocol: dict = {}
    outcome: Counter = Counter()
    n_generations = 0

    for sample, path in enumerate(args.generations):
        log.info("Reading generations [%d] %s", sample, path)
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                iid = rec["instance_id"]
                gen = rec.get("generation") or {}
                n_generations += 1
                if not protocol:
                    protocol = {
                        "model": gen.get("model"),
                        "provider": gen.get("provider"),
                        "temperature": gen.get("temperature"),
                        "max_tokens": gen.get("max_tokens"),
                        "system_prompt_sha256": hashlib.sha256(
                            (gen.get("system_prompt") or "").encode()).hexdigest()[:16],
                    }
                order.setdefault(iid, None)

                reference_hunks = parse_hunks(rec["meta"]["gold_patch"] or "")
                generated_hunks = parse_hunks(gen.get("extracted_diff") or "")
                if not reference_hunks or not generated_hunks:
                    outcome["no reference hunks or no generated hunks"] += 1
                    continue
                outcome["usable generation"] += 1

                source_files = rec.get("source_files", {})
                for h in generated_hunks:
                    diff = "".join(h["lines"])
                    key = (iid, h["filepath"], h["old_start"], diff)
                    if key in seen:
                        continue
                    seen.add(key)
                    before, after = extract_context(source_files, h["filepath"], h["old_start"], h["lines"])
                    out = {
                        "filepath": h["filepath"],
                        "old_start_line": h["old_start"],
                        "hunk_diff": diff,
                        "context_before": before,
                        "context_after": after,
                        "sample": sample,
                    }
                    if matches_reference(h["filepath"], h["old_start"], reference_hunks, args.tolerance):
                        out["tier_label"] = retained_tier(h["filepath"], f2p.get(iid, []))
                        retained[iid].append(out)
                    else:
                        out["tier_label"] = 3
                        tier3[iid].append(out)

    for path, name, table in ((args.retained_out, "llm_t12_hunks", retained),
                              (args.tier3_out, "tier3_hunks", tier3)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for iid in order:
                if iid in table:
                    fh.write(json.dumps({"instance_id": iid, name: table[iid]}) + "\n")
        log.info("Wrote %d instances → %s", len(table), path)

    # ---- integrity: every hunk is in exactly one tier file ---------------------------------
    for iid in retained.keys() & tier3.keys():
        keys_r = {(h["filepath"], h["old_start_line"], h["hunk_diff"]) for h in retained[iid]}
        keys_t = {(h["filepath"], h["old_start_line"], h["hunk_diff"]) for h in tier3[iid]}
        assert not keys_r & keys_t, f"{iid}: a hunk is both retained and non-retained"

    # ---- manifest ------------------------------------------------------------------------
    per_split: dict[str, Counter] = defaultdict(Counter)
    for iid in order:
        s = split_of.get(iid, "?")
        c = per_split[s]
        r, t = retained.get(iid, []), tier3.get(iid, [])
        c["instances"] += 1
        c["with_retained"] += bool(r)
        c["with_tier3"] += bool(t)
        c["eligible (retained and tier3)"] += bool(r and t)
        c["eligible with a Tier-2 hunk"] += bool(r and t and any(h["tier_label"] == 2 for h in r))
        c["tier1_hunks"] += sum(h["tier_label"] == 1 for h in r)
        c["tier2_hunks"] += sum(h["tier_label"] == 2 for h in r)
        c["tier3_hunks"] += len(t)

    all_r = [h for v in retained.values() for h in v]
    all_t = [h for v in tier3.values() for h in v]
    parity = {
        "retained_function_header_rate": round(sum(_has_function_header(h["hunk_diff"]) for h in all_r) / max(1, len(all_r)), 4),
        "tier3_function_header_rate": round(sum(_has_function_header(h["hunk_diff"]) for h in all_t) / max(1, len(all_t)), 4),
    }
    if abs(parity["retained_function_header_rate"] - parity["tier3_function_header_rate"]) > 0.05:
        log.warning("Retained and Tier-3 hunks differ in diff format: %s", parity)

    manifest = {
        "generation_files": args.generations,
        "n_generation_records": n_generations,
        "generation_protocol": protocol,
        "match_tolerance_lines": args.tolerance,
        "tier_rule": "retained = same file, start line within tolerance of a reference hunk; "
                     "Tier 1 if the file is statically linked to a fail-to-pass test, else Tier 2; "
                     "non-retained = Tier 3",
        "generation_outcome": dict(outcome),
        "diff_format_parity": parity,
        "per_split": {s: dict(c) for s, c in sorted(per_split.items())},
    }
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Manifest → %s", args.manifest)
    for s, c in sorted(per_split.items()):
        log.info("  %-5s %s", s, dict(c))
    log.info("Diff-format parity (share of hunks with a function header): %s", parity)


if __name__ == "__main__":
    main()
