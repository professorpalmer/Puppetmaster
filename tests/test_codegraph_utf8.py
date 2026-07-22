"""Focused tests for Windows CodeGraph UTF-8 hardening."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation


WARNING_MARK = "\u26a0"  # ⚠ — not representable in cp1252


class _Cp1252Console:
    """Text stream that mimics a legacy Windows console (cp1252, no .buffer)."""

    def __init__(self) -> None:
        self.encoding = "cp1252"
        self.errors = "strict"
        self.chunks: list[str] = []

    def reconfigure(self, *, encoding=None, errors=None) -> None:
        if encoding is not None:
            self.encoding = encoding
        if errors is not None:
            self.errors = errors

    def write(self, text: str) -> int:
        text.encode(self.encoding, errors=self.errors)
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        return None

    @property
    def value(self) -> str:
        return "".join(self.chunks)


class CodegraphUtf8PassthroughTests(unittest.TestCase):
    def test_passthrough_writes_unicode_to_cp1252_stdout(self) -> None:
        """Unicode CodeGraph markdown must not raise under a mocked cp1252 console."""
        from puppetmaster.cli import commands_codegraph as cg_cli

        payload = f"Blast radius {WARNING_MARK} no covering tests\n"
        args = SimpleNamespace(cg_args=["status"], cwd=None, timeout=0)
        cp1252_stdout = _Cp1252Console()
        # Baseline: bare write would raise on this stream.
        with self.assertRaises(UnicodeEncodeError):
            cp1252_stdout.write(payload)

        with patch.object(
            cg_cli.sys,
            "stdout",
            cp1252_stdout,
        ), patch(
            "puppetmaster.codegraph.run_codegraph_cli",
            return_value={
                "ok": True,
                "stdout": payload,
                "stderr": "",
                "returncode": 0,
            },
        ):
            rc = cg_cli._run_codegraph_passthrough(args)

        self.assertEqual(rc, 0)
        self.assertIn("Blast radius", cp1252_stdout.value)
        self.assertIn(WARNING_MARK, cp1252_stdout.value)


class CodegraphUtf8DecodeAndEnvTests(unittest.TestCase):
    def test_decode_stream_replaces_invalid_bytes(self) -> None:
        from puppetmaster.codegraph import _decode_stream

        # Invalid UTF-8 continuation byte sequence.
        self.assertEqual(_decode_stream(b"ok\xff\xfe"), "ok\ufffd\ufffd")
        self.assertEqual(_decode_stream(None), "")
        self.assertEqual(_decode_stream("already"), "already")

    def test_decode_stream_repair_module_replaces_invalid_bytes(self) -> None:
        from puppetmaster.codegraph_repair import _decode_stream

        self.assertEqual(_decode_stream(b"x\x80y"), "x\ufffdy")

    def test_nonpaging_env_includes_utf8_keys(self) -> None:
        from puppetmaster.codegraph import _nonpaging_env

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTHONUTF8", None)
            os.environ.pop("PYTHONIOENCODING", None)
            env = _nonpaging_env()
        self.assertEqual(env.get("PYTHONUTF8"), "1")
        self.assertEqual(env.get("PYTHONIOENCODING"), "utf-8")

    def test_nonpaging_env_preserves_explicit_utf8_overrides(self) -> None:
        from puppetmaster.codegraph import _nonpaging_env

        with patch.dict(
            os.environ,
            {"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp1252"},
            clear=False,
        ):
            env = _nonpaging_env()
        self.assertEqual(env.get("PYTHONUTF8"), "0")
        self.assertEqual(env.get("PYTHONIOENCODING"), "cp1252")


class CodegraphUtf8ProvisionTests(unittest.TestCase):
    def test_ensure_provisioned_passes_utf8_encoding_kwargs(self) -> None:
        from puppetmaster import codegraph as codegraph_mod

        codegraph_mod.reset_codegraph_provisioning_state()
        self.addCleanup(codegraph_mod.reset_codegraph_provisioning_state)

        which_calls = {"n": 0}
        seen_kwargs: list[dict] = []

        def fake_which(cmd):
            if cmd == "npx":
                return "/usr/local/bin/npx"
            if cmd == "npm":
                return "/usr/local/bin/npm"
            if cmd == codegraph_mod.CODEGRAPH_COMMAND:
                return "/usr/local/bin/codegraph" if which_calls["n"] else None
            return None

        def fake_run(cmd, **kwargs):
            seen_kwargs.append(kwargs)
            if cmd[:3] == ["/usr/local/bin/npm", "install", "-g"]:
                which_calls["n"] = 1
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch.object(
            codegraph_mod, "_cursor_codegraph_invocation", return_value=None
        ), patch(
            "puppetmaster.codegraph.shutil.which", side_effect=fake_which
        ), patch(
            "puppetmaster.codegraph.subprocess.run", side_effect=fake_run
        ), patch.dict(os.environ, {}, clear=False):
            for var in (
                "PUPPETMASTER_CODEGRAPH_NO_NPX",
                "PUPPETMASTER_CODEGRAPH_NO_GLOBAL_INSTALL",
                "PUPPETMASTER_CODEGRAPH_NODE",
                "PUPPETMASTER_CODEGRAPH_JS",
            ):
                os.environ.pop(var, None)
            self.assertTrue(codegraph_mod.ensure_codegraph_provisioned())

        self.assertTrue(seen_kwargs)
        for kwargs in seen_kwargs:
            self.assertEqual(kwargs.get("encoding"), "utf-8")
            self.assertEqual(kwargs.get("errors"), "replace")
            self.assertTrue(kwargs.get("text"))


class WriteConsoleTextTests(unittest.TestCase):
    def test_write_console_text_via_buffer_preserves_warning(self) -> None:
        from puppetmaster.cli.commands_codegraph import _write_console_text

        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        _write_console_text(stream, f"note {WARNING_MARK}\n")
        stream.flush()
        raw.seek(0)
        self.assertEqual(raw.read().decode("utf-8"), f"note {WARNING_MARK}\n")

    def test_write_console_text_without_buffer_does_not_raise(self) -> None:
        from puppetmaster.cli.commands_codegraph import _write_console_text

        stream = _Cp1252Console()
        with self.assertRaises(UnicodeEncodeError):
            stream.write(f"warn {WARNING_MARK}\n")
        _write_console_text(stream, f"warn {WARNING_MARK}\n")
        self.assertIn(WARNING_MARK, stream.value)


if __name__ == "__main__":
    unittest.main()
