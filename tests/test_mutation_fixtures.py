"""Hermetic deterministic mutation-fixture evaluation (reliability Slice 11).

Static cases live under ``tests/fixtures/mutations/*.json``. Each case names a
source path/text plus either an ``edit_file`` or ``hashline`` op, and an
``expect`` block for final normalized text / content-tag / stale refusal.

This surface is tests/evaluation-only: it drives
``hashline.apply_patch`` and ``AgenticAdapter._tool_edit_file`` against known
fixtures. It does not touch RQGM anchors, evaluator registries, or routing.

Fixture schema (one JSON object per file)::

    {
      "name": "edit_tagged_success",
      "engine": "edit_file" | "hashline",
      "source_path": "pkg/sample.py",   # forward slashes; Windows-safe
      "source_text": "...",
      "edit_file": {
        "old_string": "...",
        "new_string": "...",
        "expected_tag": "live" | "stale" | null
      },
      "hashline": {
        "tag": "live" | "stale",
        "ops": "SWAP 1.=1:\\n+LINE_ONE\\n"
      },
      "expect": {
        "status": "ok" | "stale",
        "text": "...",
        "assert_result_tag": true
      }
    }

``expected_tag`` / ``hashline.tag`` sentinels:
- ``live`` — compute ``content_tag(source_text)`` at apply time
- ``stale`` — use a known-wrong tag (``FFFF``) so concurrency refuses
- ``null`` (edit_file only) — omit ``expected_tag`` (legacy untagged path)

To add a case: drop another ``*.json`` beside the existing fixtures; the
parametrized loader picks it up automatically.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from puppetmaster.hashline import (
    SnapshotStore,
    StaleTagError,
    apply_patch,
    content_tag,
    normalize_text,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "mutations"
_STALE_TAG = "FFFF"

def _load_cases() -> List[Dict[str, Any]]:
    if not _FIXTURES_DIR.is_dir():
        return []
    cases: List[Dict[str, Any]] = []
    for path in sorted(_FIXTURES_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise AssertionError(f"{path.name}: root must be an object")
        data["_fixture_file"] = path.name
        cases.append(data)
    return cases

def _canonical_rel(path: str) -> str:
    """Normalize fixture paths to forward-slash relatives (Windows-safe)."""
    text = str(path).strip().replace("\\", "/")
    if not text or text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise AssertionError(f"source_path must be a relative POSIX-ish path, got {path!r}")
    if ".." in Path(text).parts:
        raise AssertionError(f"source_path must not contain '..': {path!r}")
    return text

def _resolve_tag_sentinel(sentinel: Optional[str], source_text: str) -> Optional[str]:
    if sentinel is None:
        return None
    key = str(sentinel).strip().lower()
    if key == "live":
        return content_tag(source_text)
    if key == "stale":
        return _STALE_TAG
    # Allow an explicit 4-hex tag in fixtures when needed.
    raw = str(sentinel).strip().upper()
    if len(raw) == 4 and all(c in "0123456789ABCDEF" for c in raw):
        return raw
    raise AssertionError(f"unknown tag sentinel {sentinel!r}")

def _materialize(case: Dict[str, Any], cwd: Path) -> Path:
    rel = _canonical_rel(str(case["source_path"]))
    target = cwd / Path(*rel.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    # Use open(..., newline="\n") so Python 3.9 keeps LF on Windows too
    # (Path.write_text newline= arrived in 3.10).
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(str(case["source_text"]))
    return target

def _apply_edit_file(case: Dict[str, Any], cwd: Path) -> str:
    from puppetmaster.adapters.agentic import AgenticAdapter

    edit = case["edit_file"]
    args: Dict[str, Any] = {
        "path": _canonical_rel(str(case["source_path"])),
        "old_string": str(edit["old_string"]),
        "new_string": str(edit["new_string"]),
    }
    if "replace_all" in edit:
        args["replace_all"] = bool(edit["replace_all"])
    tag = _resolve_tag_sentinel(edit.get("expected_tag"), str(case["source_text"]))
    if tag is not None:
        args["expected_tag"] = tag
    return AgenticAdapter()._tool_edit_file(args, cwd)

def _apply_hashline(case: Dict[str, Any], cwd: Path) -> str:
    hl = case["hashline"]
    rel = _canonical_rel(str(case["source_path"]))
    tag = _resolve_tag_sentinel(hl.get("tag", "live"), str(case["source_text"]))
    assert tag is not None
    ops = str(hl["ops"]).lstrip("\n")
    if not ops.endswith("\n"):
        ops = ops + "\n"
    patch = f"[{rel}#{tag}]\n{ops}"
    result = apply_patch(cwd, patch, SnapshotStore())
    # Surface a stable success token for expect.status == ok assertions.
    return f"ok: {[sec.tag for sec in result.sections]}"

def _read_normalized(path: Path) -> str:
    return normalize_text(path.read_text(encoding="utf-8"))

class MutationFixtureDiscoveryTests(unittest.TestCase):
    def test_fixtures_present_and_named(self) -> None:
        cases = _load_cases()
        self.assertGreaterEqual(len(cases), 4, "expected core Slice 11 fixtures")
        names = {c["name"] for c in cases}
        required = {
            "edit_tagged_success",
            "edit_stale_tag_refusal",
            "edit_untagged_legacy",
            "hashline_tagged_success",
            "hashline_stale_refusal",
        }
        self.assertTrue(required.issubset(names), f"missing {required - names}")

class MutationFixtureEvalTests(unittest.TestCase):
    """Apply every static mutation fixture and assert deterministic outcomes."""

    def test_all_mutation_fixtures(self) -> None:
        cases = _load_cases()
        self.assertTrue(cases, f"no fixtures under {_FIXTURES_DIR}")
        for case in cases:
            with self.subTest(case=case.get("name", case.get("_fixture_file"))):
                self._run_case(case)

    def _run_case(self, case: Dict[str, Any]) -> None:
        engine = str(case["engine"])
        expect = case["expect"]
        status = str(expect["status"])
        expected_text = normalize_text(str(expect["text"]))

        with tempfile.TemporaryDirectory(prefix="pm-mutfix-") as tmp:
            cwd = Path(tmp)
            target = _materialize(case, cwd)
            before = _read_normalized(target)

            if engine == "edit_file":
                if status == "stale":
                    out = _apply_edit_file(case, cwd)
                    self.assertIn("error:", out, msg=out)
                    self.assertIn("StaleTagError", out, msg=out)
                    self.assertIn("stale expected_tag", out, msg=out)
                else:
                    out = _apply_edit_file(case, cwd)
                    self.assertTrue(out.startswith("edited "), msg=out)
            elif engine == "hashline":
                if status == "stale":
                    with self.assertRaises(StaleTagError):
                        _apply_hashline(case, cwd)
                    out = "stale"
                else:
                    out = _apply_hashline(case, cwd)
                    self.assertTrue(out.startswith("ok:"), msg=out)
            else:
                raise AssertionError(f"unknown engine {engine!r}")

            after = _read_normalized(target)
            self.assertEqual(after, expected_text)
            if status == "stale":
                self.assertEqual(after, before)
                self.assertEqual(after, normalize_text(str(case["source_text"])))
            elif expect.get("assert_result_tag"):
                result_tag = content_tag(after)
                self.assertEqual(result_tag, content_tag(expected_text))
                if engine == "edit_file":
                    rel = _canonical_rel(str(case["source_path"]))
                    self.assertIn(f"[{rel}#{result_tag}]", out)

if __name__ == "__main__":
    unittest.main()
