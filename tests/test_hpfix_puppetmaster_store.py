from __future__ import annotations

import tempfile
import unittest

from puppetmaster.models import MemoryRecord
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore


def _memory(statement: str, memory_id: str) -> MemoryRecord:
    return MemoryRecord(
        scope="repo",
        statement=statement,
        evidence=[],
        source_artifacts=[],
        confidence=0.9,
        id=memory_id,
    )


class RetrieveMemoryPunctuationTests(unittest.TestCase):
    """Bug 38803feb-2: goal words with punctuation must match stored words."""

    def _ids(self, store_cls, query: str) -> list:
        with tempfile.TemporaryDirectory() as tmp:
            store = store_cls(tmp)
            store.init()
            store.promote_memory(_memory("zebra layer is stale", "m1"))
            store.promote_memory(_memory("worker crashes on bug", "m2"))
            return [m["id"] for m in store.retrieve_memory(query)]

    def test_json_store_matches_word_with_trailing_comma(self) -> None:
        self.assertEqual(self._ids(SwarmStore, "fix zebra, please"), ["m1"])

    def test_json_store_matches_word_with_trailing_period(self) -> None:
        self.assertEqual(self._ids(SwarmStore, "Fix the bug."), ["m2"])

    def test_sqlite_store_matches_word_with_punctuation(self) -> None:
        self.assertEqual(self._ids(SQLiteSwarmStore, "fix zebra, please"), ["m1"])
        self.assertEqual(self._ids(SQLiteSwarmStore, "Fix the bug."), ["m2"])

    def test_plain_words_keep_old_result(self) -> None:
        self.assertEqual(self._ids(SwarmStore, "fix zebra please"), ["m1"])
        self.assertEqual(self._ids(SQLiteSwarmStore, "fix zebra please"), ["m1"])

    def test_non_latin_word_with_punctuation_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SwarmStore(tmp)
            store.init()
            store.promote_memory(_memory("кэш устарел", "m3"))
            store.promote_memory(_memory("worker crashes on bug", "m2"))
            ids = [m["id"] for m in store.retrieve_memory("почини кэш.")]
        self.assertEqual(ids, ["m3"])

    def test_accented_word_keeps_whole_word(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SwarmStore(tmp)
            store.init()
            store.promote_memory(_memory("café menu is stale", "m4"))
            store.promote_memory(_memory("caf build flag", "m5"))
            ids = [m["id"] for m in store.retrieve_memory("café")]
        self.assertEqual(ids, ["m4"])

    def test_unrelated_non_latin_query_still_matches_nothing(self) -> None:
        # An ASCII-only split gives no terms here, and no terms returns
        # every memory, so this guards the term filter for other scripts.
        self.assertEqual(self._ids(SwarmStore, "ошибка"), [])
        self.assertEqual(self._ids(SQLiteSwarmStore, "ошибка"), [])

    def test_unrelated_query_still_matches_nothing(self) -> None:
        self.assertEqual(self._ids(SwarmStore, "giraffe, now."), [])
        self.assertEqual(self._ids(SQLiteSwarmStore, "giraffe, now."), [])


if __name__ == "__main__":
    unittest.main()
