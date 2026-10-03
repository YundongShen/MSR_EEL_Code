"""Assign Tier 1 / Tier 2 labels to gold patch hunks.

Tier assignment (offline, no test execution required):

  Tier 1 — Hunk is in a file that is likely covered by at least one
            fail-to-pass test, inferred from import statements in the
            test files and directory co-location.

  Tier 2 — Gold hunk with no such inferred coverage link.
            These are "necessary but untested" changes — the hardest
            sub-task for any scope detection method.

Both tiers are gold changes; only Tier 3 (unconstrained extras) is drift.
The Tier 1 / Tier 2 split lets us measure whether the model can explain
changes that tests alone cannot justify.

Heuristic:
  A hunk file F is Tier 1 if any fail-to-pass test file T satisfies:
    (a) T imports a module that is a prefix of F's dotted module name, OR
    (b) F and T share the same package directory (co-location).
"""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_to_module(filepath: str) -> str:
    """Convert a/b/c.py → a.b.c (drop .py suffix)."""
    return Path(filepath).with_suffix("").as_posix().replace("/", ".")


def _extract_imports(source: str) -> set[str]:
    """Return all top-level module names referenced in import statements."""
    modules: set[str] = set()
    for m in re.finditer(
        r"^(?:from|import)\s+([\w.]+)", source, re.MULTILINE
    ):
        modules.add(m.group(1))
    return modules


def _package_dir(filepath: str) -> str:
    """Return the parent directory of filepath (normalised to forward slashes)."""
    return str(Path(filepath).parent).replace("\\", "/")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def label_tiers(
    instance: dict,
    raw_test_sources: dict[str, str] | None = None,
) -> list[int]:
    """Return a Tier label (1 or 2) for each hunk in ``instance['gold_hunks']``.

    Parameters
    ----------
    instance:
        A parsed instance dict as produced by ``scripts/parse_instances.py``.
        Must contain ``fail_to_pass_ids`` and ``gold_hunks``.
    raw_test_sources:
        Optional mapping filepath → full source of test files.
        If provided, import-based coverage is checked in addition to
        directory co-location (more accurate).

    Returns
    -------
    list[int] — one tier per hunk, in the same order as ``gold_hunks``.
    """
    fail_to_pass: list[str] = instance.get("fail_to_pass_ids", [])
    gold_hunks: list[dict] = instance.get("gold_hunks", [])

    # Collect test file paths referenced by fail_to_pass IDs
    test_paths: set[str] = {tid.split("::")[0] for tid in fail_to_pass}

    # Build set of (module, package_dir) pairs for test files
    test_modules: set[str] = set()
    test_dirs: set[str] = set()

    for tp in test_paths:
        test_dirs.add(_package_dir(tp))
        # If raw sources are available, extract imports for exact coverage
        if raw_test_sources and tp in raw_test_sources:
            test_modules |= _extract_imports(raw_test_sources[tp])

    # For each hunk decide tier
    tiers: list[int] = []
    for hunk in gold_hunks:
        filepath = hunk.get("filepath", "")
        if not filepath:
            tiers.append(2)
            continue

        hunk_module = _path_to_module(filepath)
        hunk_dir = _package_dir(filepath)

        covered = False

        # Check (a): import-based coverage
        if test_modules:
            covered = any(
                hunk_module == imp or hunk_module.startswith(imp + ".")
                for imp in test_modules
            )

        # Check (b): directory co-location
        # Test file lives in a 'tests/' subdirectory of the hunk's package dir
        if not covered:
            covered = any(
                hunk_dir in td or td.startswith(hunk_dir)
                for td in test_dirs
            )

        tiers.append(1 if covered else 2)

    return tiers


def label_instance_inplace(
    instance: dict,
    raw_test_sources: dict[str, str] | None = None,
) -> None:
    """Assign tier_label to each hunk in ``instance['gold_hunks']`` in-place."""
    tiers = label_tiers(instance, raw_test_sources)
    for hunk, tier in zip(instance.get("gold_hunks", []), tiers):
        hunk["tier_label"] = tier
