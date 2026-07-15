"""Unit tests for puppetmaster.hashline (OMP-inspired content-hash edits)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from puppetmaster.hashline import (
    SnapshotStore,
    StaleTagError,
    UnsupportedError,
    apply_patch,
    content_tag,
    format_numbered_read,
    hashline_enabled,
    normalize_text,
    parse_patch,
)


class ContentTagTests(unittest.TestCase):
    def test_tag_stability_and_normalization(self) -> None:
        a = "hello\nworld\n"
        b = "hello\r\nworld\r\n"
        c = "\ufeffhello\nworld\n"
        self.assertEqual(content_tag(a), content_tag(b))
        self.assertEqual(content_tag(a), content_tag(c))
        self.assertEqual(len(content_tag(a)), 4)
        self.assertRegex(content_tag(a), r"^[0-9A-F]{4}$")

    def test_trailing_whitespace_trimmed_for_hash(self) -> None:
        self.assertEqual(content_tag("hi  \n"), content_tag("hi\n"))


class SnapshotStoreTests(unittest.TestCase):
    def test_record_resolve_roundtrip(self) -> None:
        store = SnapshotStore()
        tag = store.record("a.py", "one\ntwo\n")
        self.assertEqual(store.resolve("a.py", tag), normalize_text("one\ntwo\n"))
        self.assertIsNone(store.resolve("a.py", "0000"))


class ParseAndApplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self.tmp.name)
        self.greet = self.cwd / "greet.py"
        self.greet.write_text(
            'def greet(name):\n    msg = "Hello, " + name\n    print(msg)\ngreet("world")\n',
            encoding="utf-8",
        )
        self.store = SnapshotStore()
        self.tag = self.store.record("greet.py", self.greet.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_format_numbered_read(self) -> None:
        out = format_numbered_read("greet.py", self.tag, ["a", "b"], start_line=1)
        self.assertTrue(out.startswith(f"[greet.py#{self.tag}]"))
        self.assertIn("1:a", out)
        self.assertIn("2:b", out)

    def test_swap_del_ins(self) -> None:
        patch = f"""[greet.py#{self.tag}]
INS.POST 1:
+    if not name: name = "stranger"
SWAP 2.=2:
+    greeting = "Hi"
+    msg = f"{{greeting}}, {{name}}"
DEL 3
"""
        # After INS.POST and SWAP/DEL using ORIGINAL numbers:
        # line 3 still means original print(msg)
        result = apply_patch(self.cwd, patch, self.store)
        self.assertEqual(result.sections[0].op, "update")
        text = self.greet.read_text(encoding="utf-8")
        self.assertIn('if not name: name = "stranger"', text)
        self.assertIn('greeting = "Hi"', text)
        self.assertNotIn("print(msg)", text)
        self.assertIn('greet("world")', text)

    def test_ins_head_tail(self) -> None:
        patch = f"""[greet.py#{self.tag}]
INS.HEAD:
+# header
INS.TAIL:
+# tail
"""
        apply_patch(self.cwd, patch, self.store)
        text = self.greet.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# header\n"))
        self.assertTrue(text.rstrip("\n").endswith("# tail"))

    def test_rem(self) -> None:
        patch = f"[greet.py#{self.tag}]\nREM\n"
        apply_patch(self.cwd, patch, self.store)
        self.assertFalse(self.greet.exists())

    def test_mv(self) -> None:
        patch = f"[greet.py#{self.tag}]\nMV greet_v2.py\n"
        result = apply_patch(self.cwd, patch, self.store)
        self.assertEqual(result.sections[0].op, "move")
        self.assertFalse(self.greet.exists())
        self.assertTrue((self.cwd / "greet_v2.py").exists())

    def test_stale_tag_rejected(self) -> None:
        patch = f"[greet.py#{self.tag}]\nDEL 4\n"
        self.greet.write_text("changed\n", encoding="utf-8")
        with self.assertRaises(StaleTagError):
            apply_patch(self.cwd, patch, self.store)
        self.assertEqual(self.greet.read_text(encoding="utf-8"), "changed\n")

    def test_blk_unsupported(self) -> None:
        with self.assertRaises(UnsupportedError):
            parse_patch(f"[greet.py#{self.tag}]\nSWAP.BLK 1:\n+x\n")

    def test_preflight_atomicity(self) -> None:
        other = self.cwd / "other.py"
        other.write_text("alpha\nbeta\n", encoding="utf-8")
        other_tag = self.store.record("other.py", other.read_text(encoding="utf-8"))
        # Second section has a stale/wrong tag — first must not be written.
        patch = f"""[greet.py#{self.tag}]
DEL 4
[other.py#FFFF]
DEL 1
"""
        before = self.greet.read_text(encoding="utf-8")
        with self.assertRaises(StaleTagError):
            apply_patch(self.cwd, patch, self.store)
        self.assertEqual(self.greet.read_text(encoding="utf-8"), before)
        self.assertEqual(other.read_text(encoding="utf-8"), "alpha\nbeta\n")
        # Valid other tag unused — ensure we didn't confuse stores.
        self.assertEqual(self.store.resolve("other.py", other_tag), "alpha\nbeta\n")

    def test_original_line_numbers_multi_hunk(self) -> None:
        # Both hunks refer to original lines; deleting 4 then swapping 2 must
        # still target original line 2, not a shifted index.
        patch = f"""[greet.py#{self.tag}]
DEL 4
SWAP 2.=2:
+    msg = "Hi"
"""
        apply_patch(self.cwd, patch, self.store)
        text = self.greet.read_text(encoding="utf-8")
        self.assertIn('msg = "Hi"', text)
        self.assertNotIn('greet("world")', text)


class KillSwitchTests(unittest.TestCase):
    def test_hashline_enabled_default_on(self) -> None:
        prev = os.environ.pop("PUPPETMASTER_HASHLINE", None)
        try:
            self.assertTrue(hashline_enabled())
            os.environ["PUPPETMASTER_HASHLINE"] = "0"
            self.assertFalse(hashline_enabled())
        finally:
            if prev is None:
                os.environ.pop("PUPPETMASTER_HASHLINE", None)
            else:
                os.environ["PUPPETMASTER_HASHLINE"] = prev


if __name__ == "__main__":
    unittest.main()
