"""Surface the wrong-state-dir warning where a human or agent will see it.

``default_state_dir()`` hashes the git root of the caller's cwd, so a session
opened at a non-git *wrapper* directory forked one logical project into two
state dirs: ``puppetmaster-c3177e6032c4`` (1 stale job, what the dashboard
showed) and ``Puppetmaster-b92145e840c8`` (24 jobs, where the real work went).

The detector for that split already exists (``puppetmaster.state_health``, see
tests/test_state_dir_identity.py) and doctor already reports it. The deeper
defect these tests close is that the *dashboard* stayed silent:
``dashboard_serves()`` only proves "the server on this port serves the state
dir I computed", never "that state dir is where this workspace's jobs go", so
``run_dashboard`` reported ``started=true`` with a cheerful "Started a new
dashboard for this project." while serving an empty store.

Three surfaces, three groups below: the MCP tool body an agent reads, the
stderr line a shell user reads, and the in-page banner a browser user reads.

A new file for the same reason as tests/test_dashboard_project_identity.py:
tests/test_puppetmaster.py is enormous and contended, so this workstream's
node harness is additive rather than an edit to a hot file.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import contextlib
import io
import json
import shutil
import subprocess
import threading
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import puppetmaster.dashboard as dash
import puppetmaster.state_health as state_health
from puppetmaster import state
from puppetmaster.dashboard import _PAGE_APP_JS, RENDERER_JS, make_handler, serve

# The exact note run_dashboard has always produced on the spawn path (the
# other is "Reused this project's own dashboard, already running."). Pinned as
# a constant so the "appended to, not replaced" contract is asserted against
# the literal text an agent has learned to read, not a paraphrase.
NOTE_STARTED = "Started a new dashboard for this project."

_JOB_EPOCH = 1_700_000_000.0


def _make_jobs(state_dir: Path, count: int, *, offset: float = 0.0) -> None:
    """Fake job history: ``<state_dir>/jobs/<id>`` dirs with fixed mtimes.

    The detector counts job *directories* rather than opening the store (it
    has to stay SQLite-free), so directories are the whole fixture.
    """
    jobs = state_dir / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        job = jobs / f"job_{state_dir.name[:8]}_{index:03d}"
        job.mkdir(exist_ok=True)
        stamp = _JOB_EPOCH + offset + index
        os.utime(job, (stamp, stamp))


@contextlib.contextmanager
def _chdir(target: Path):
    """Run the block with ``target`` as the process cwd.

    Not incidental: ``serve()`` and ``make_handler()`` diagnose against
    ``Path.cwd()``, because the dashboard process's cwd *is* the workspace
    whose jobs the user expects to see. Reproducing the incident therefore
    means actually standing in the wrapper directory — this repo's own root is
    a git checkout, so without the chdir the detector's "cwd is a git root"
    guard fires and every assertion below would pass vacuously.
    """
    previous = os.getcwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _fork_fixture(tmp: str, *, active_jobs: int, child_jobs: int):
    """The reported incident, as a filesystem: wrapper folder above a checkout.

    ``<tmp>/puppetmaster`` is not a git repo; ``<tmp>/puppetmaster/Puppetmaster``
    is. Yields ``(wrapper, child, active_dir, child_dir)`` with the app state
    root still patched, so callers can keep deriving paths inside the block —
    ``projects_root``/``project_state_dir_for``/``default_state_dir`` all
    resolve that root at *call* time, and a value derived after the block exits
    is silently computed against the real AppData root instead.

    ``_git_root`` is pinned to None rather than trusting the temp dir to sit
    outside a checkout: "this folder is not in a git repo" is the precondition
    of the whole incident, so it belongs in the fixture — and pinning it keeps
    the suite off a ``git`` subprocess. The base is ``resolve()``d so the
    detector's cwd-derivation check compares equal after ``_chdir``.
    """
    base = Path(tmp).resolve()
    with patch.object(
        state, "app_state_root", return_value=base / "app-state"
    ), patch.object(state, "_git_root", return_value=None):
        wrapper = base / "puppetmaster"
        child = wrapper / "Puppetmaster"
        child.mkdir(parents=True)
        (child / ".git").mkdir()
        active_dir = state.project_state_dir_for(wrapper)
        child_dir = state.project_state_dir_for(child)
        _make_jobs(active_dir, active_jobs)
        _make_jobs(child_dir, child_jobs, offset=100.0)
        yield wrapper, child, active_dir, child_dir


@contextlib.contextmanager
def _mocked_dashboard_lifecycle():
    """Spawn path, fully mocked — no real server, no real process.

    Mirrors tests/test_dashboard_background_identity.py's MCP cases: the
    own-runfile pre-check finds nothing tracked, the literal-port probe says
    "not ours" (``dashboard_serves`` False), the spawn is faked, and the
    child's runfile appears on the poll loop's read. Result: ``started=true``
    on port 8787 — the exact shape the reporting user saw.
    """
    spawned = MagicMock()
    spawned.pid = 7070
    spawned.poll.return_value = None
    child_runfile = {
        "pid": 7070,
        "host": "127.0.0.1",
        "port": 8787,
        "url": "http://127.0.0.1:8787/",
    }
    import puppetmaster.mcp_server as mcp

    with patch.object(
        dash, "read_dashboard_runfile", side_effect=[None, child_runfile]
    ), patch.object(
        dash, "dashboard_serves", side_effect=[False, True]
    ), patch.object(
        dash, "dashboard_identity", return_value=None
    ), patch(
        "subprocess.Popen", return_value=spawned
    ), patch.object(
        mcp, "_spawn_dashboard_server", return_value=spawned
    ):
        yield


def _run_dashboard(args: dict) -> dict:
    """``run_dashboard``'s parsed body, plus the raw result for ``isError``.

    Calls the handler directly rather than ``call_tool``: ``call_tool`` folds
    version-staleness nudges into the payload, and these tests assert on the
    exact contents of ``note``.
    """
    import puppetmaster.mcp_server as mcp

    result = mcp.run_dashboard(args)
    return {"result": result, "body": json.loads(result["content"][0]["text"])}


class McpDashboardWarningTests(unittest.TestCase):
    """The agent-facing surface: a structured warning plus an appended note."""

    def test_suspect_state_dir_warns_without_failing_the_start(self) -> None:
        with TemporaryDirectory() as tmp:
            with _fork_fixture(tmp, active_jobs=1, child_jobs=5) as (
                wrapper,
                _child,
                active_dir,
                child_dir,
            ):
                active_name, child_name = active_dir.name, child_dir.name
                with _mocked_dashboard_lifecycle():
                    run = _run_dashboard({"cwd": str(wrapper)})

        result, body = run["result"], run["body"]
        # The dashboard *did* start. This is advisory, not a failure: flipping
        # it to isError would make a working board look broken.
        self.assertFalse(result.get("isError"))
        self.assertTrue(body["started"])

        warnings = body["warnings"]
        self.assertEqual(warnings[0]["kind"], "state_dir_may_be_wrong")
        self.assertEqual(warnings[0]["state_dir_name"], active_name)
        self.assertEqual(warnings[0]["active_jobs"], 1)
        self.assertEqual(warnings[0]["candidate"], child_name)
        self.assertEqual(warnings[0]["candidate_jobs"], 5)

        # Appended, not replaced: the caller must still be able to tell a
        # fresh start from a reuse.
        self.assertIn(NOTE_STARTED, body["note"])
        # ...and the remedy has to name every escape hatch, because which one
        # applies depends on what the caller was actually trying to do.
        for needle in (
            "git root",
            "cwd",
            "state_dir",
            "all_projects=true",
            "python -m puppetmaster projects",
        ):
            self.assertIn(needle, body["note"], needle)

    def test_warning_and_note_carry_basenames_only(self) -> None:
        """No absolute path anywhere in the advisory text.

        ``body["state_dir"]`` legitimately carries the full path (it always
        has), so this is scoped to the two fields that are new: an MCP body is
        quoted verbatim into agent transcripts, where a home directory is
        gratuitous disclosure.
        """
        with TemporaryDirectory() as tmp:
            with _fork_fixture(tmp, active_jobs=1, child_jobs=5) as (
                wrapper,
                _child,
                active_dir,
                _child_dir,
            ):
                absolute, workspace = str(active_dir), str(wrapper)
                with _mocked_dashboard_lifecycle():
                    run = _run_dashboard({"cwd": str(wrapper)})

        body = run["body"]
        # ensure_ascii=False deliberately: the default would escape
        # `short_warning`'s em-dash to "—", whose backslash IS os.sep on
        # Windows and would make the separator assertion below fire on an
        # artefact of this test's own serialisation.
        advisory = body["note"] + json.dumps(body["warnings"], ensure_ascii=False)
        self.assertNotIn(absolute, advisory)
        self.assertNotIn(workspace, advisory)
        # "No separator at all" is the strongest form of "basenames only", and
        # it holds: the real advisory text contains no paths whatsoever.
        self.assertNotIn(os.sep, advisory)
        self.assertNotIn("/", advisory)

    def test_healthy_state_dir_stays_silent(self) -> None:
        """Anti-noise regression guard.

        Identical fixture to the suspect case with the job counts flipped, so
        the *only* difference is which dir holds the history. A busy active dir
        next to a nested checkout is the normal shape of a vendored/submodule
        tree and must produce no warnings key and a byte-identical note.
        """
        with TemporaryDirectory() as tmp:
            with _fork_fixture(tmp, active_jobs=10, child_jobs=5) as (
                wrapper,
                _child,
                _active_dir,
                _child_dir,
            ):
                with _mocked_dashboard_lifecycle():
                    run = _run_dashboard({"cwd": str(wrapper)})

        body = run["body"]
        self.assertNotIn("warnings", body)
        self.assertEqual(body["note"], NOTE_STARTED)

    def test_all_projects_board_never_warns(self) -> None:
        """An --all-projects board serves every project's store, so "you may
        be pointed at the wrong one" cannot be true of it — even on the exact
        filesystem that makes the single-project case suspect."""
        with TemporaryDirectory() as tmp:
            with _fork_fixture(tmp, active_jobs=1, child_jobs=5) as (
                wrapper,
                _child,
                _active_dir,
                _child_dir,
            ):
                with _mocked_dashboard_lifecycle():
                    run = _run_dashboard({"cwd": str(wrapper), "all_projects": True})

        body = run["body"]
        self.assertTrue(body["all_projects"])
        self.assertNotIn("warnings", body)
        self.assertEqual(body["note"], NOTE_STARTED)


class ServeStderrWarningTests(unittest.TestCase):
    """The shell surface: a line next to "Reading durable state from:".

    This is what the reporting user would have hit first — they started the
    board from a shell and got a confident "Reading durable state from: <the
    wrong dir>" with nothing on screen to contradict it.
    """

    def test_serve_warns_on_stderr_beside_the_state_dir_line(self) -> None:
        with TemporaryDirectory() as tmp:
            with _fork_fixture(tmp, active_jobs=1, child_jobs=5) as (
                wrapper,
                child,
                active_dir,
                _child_dir,
            ):
                err, out = io.StringIO(), io.StringIO()
                with _chdir(wrapper), patch.object(sys, "stderr", err):
                    with contextlib.redirect_stdout(out):
                        httpd = serve(
                            active_dir,
                            host="127.0.0.1",
                            port=0,
                            open_browser=False,
                            serve_forever=False,
                        )
                    httpd.server_close()
                names = (active_dir.name, child.name)
                absolute = str(active_dir)

        stderr_text = err.getvalue()
        self.assertIn("WARNING", stderr_text)
        for name in names:
            self.assertIn(name, stderr_text, name)
        # stdout still carries the line this one sits beside.
        self.assertIn("Reading durable state from:", out.getvalue())
        # Basenames only: this lands in dashboard.err.log, which users paste
        # into issues verbatim.
        self.assertNotIn(absolute, stderr_text)
        self.assertNotIn(os.sep, stderr_text)

    def test_state_dir_warning_returns_none_on_any_exception(self) -> None:
        """Everything after the bind in ``serve()`` sits inside an
        ``except BaseException: httpd.server_close(); raise``, so an unguarded
        diagnosis would convert a working dashboard into a failed start."""
        with TemporaryDirectory() as tmp:
            with patch.object(
                state_health, "diagnose_state_dir", side_effect=RuntimeError("boom")
            ):
                self.assertIsNone(dash.state_dir_warning(Path(tmp)))

    def test_state_dir_warning_skips_all_projects_and_none(self) -> None:
        with TemporaryDirectory() as tmp:
            with patch.object(
                state_health, "diagnose_state_dir", side_effect=AssertionError
            ) as probe:
                self.assertIsNone(dash.state_dir_warning(Path(tmp), all_projects=True))
                self.assertIsNone(dash.state_dir_warning(None))
            probe.assert_not_called()

    def test_a_raising_detector_still_lets_serve_bind(self) -> None:
        """End to end: the guard is only worth something if it is actually the
        thing standing between a raising detector and a dead dashboard."""
        with TemporaryDirectory() as tmp, patch.object(
            state_health, "diagnose_state_dir", side_effect=RuntimeError("boom")
        ):
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                httpd = serve(
                    state_dir,
                    host="127.0.0.1",
                    port=0,
                    open_browser=False,
                    serve_forever=False,
                )
            httpd.server_close()


class DiagnosticsEndpointTests(unittest.TestCase):
    """The browser surface, server side: a separate, cached, gated endpoint.

    Deliberately NOT on /api/meta: ``dashboard_identity`` reads at most 65536
    bytes under a one-second socket timeout and ``json.loads`` the result, so
    an oversized or slow body silently becomes None — which reads as "not my
    dashboard" and makes the CLI and MCP spawn duplicate servers. A filesystem
    walk behind that endpoint would risk the very bug it exists to prevent.
    """

    def _serve(self, state_dir: Path) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            httpd = serve(
                state_dir,
                backend="sqlite",
                host="127.0.0.1",
                port=0,
                open_browser=False,
                serve_forever=False,
            )
        # addCleanup is LIFO, so server_close must be registered FIRST for
        # shutdown to run before it (same order as tests/test_dashboard_ports.py).
        self.addCleanup(httpd.server_close)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        return httpd.server_address[1]

    @staticmethod
    def _get(port: int, path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as resp:
            return {"code": resp.status, "body": json.loads(resp.read().decode())}

    def test_unmanaged_state_dir_gets_a_200_with_no_diagnosis(self) -> None:
        """A temp dir outside the hashed ``projects/`` layout is a deliberate
        choice (``--state-dir``), not a fork, so the endpoint answers 200 with
        nothing to report rather than 404 or a guess."""
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            port = self._serve(state_dir)
            got = self._get(port, "/api/diagnostics")
        self.assertEqual(got["code"], 200)
        self.assertIsNone(got["body"]["diagnosis"])

    def test_a_raising_detector_cannot_break_the_reuse_contract(self) -> None:
        """The handler's blanket ``except`` turns any error into a 500, so the
        payload builder has to swallow — and the load-bearing half of this is
        that /api/meta, which reuse detection reads, keeps answering."""
        with TemporaryDirectory() as tmp, patch.object(
            state_health, "diagnose_state_dir", side_effect=RuntimeError("boom")
        ):
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            port = self._serve(state_dir)
            diagnostics = self._get(port, "/api/diagnostics")
            meta = self._get(port, "/api/meta")
        self.assertEqual(diagnostics["code"], 200)
        self.assertIsNone(diagnostics["body"]["diagnosis"])
        self.assertEqual(meta["code"], 200)
        self.assertEqual(meta["body"]["service"], "puppetmaster-dashboard")

    def _call(self, handler, peer: str) -> str:
        """Drive one request without a socket, so the peer address is ours to
        choose (same shape as tests/test_dashboard_project_aggregate.py)."""
        sent: list = []
        instance = handler.__new__(handler)
        instance.path = "/api/diagnostics"
        instance.client_address = (peer, 0)
        instance._send = lambda code, body, content_type: sent.append((code, body))
        instance.do_GET()
        self.assertEqual([code for code, _ in sent], [200])
        return sent[0][1].decode("utf-8")

    def test_non_loopback_peer_gets_a_boolean_and_nothing_else(self) -> None:
        """--mobile makes this endpoint reachable off-loopback, so it is gated
        exactly like /api/meta: a remote peer learns *that* something looks
        wrong, never a project basename and never a job count."""
        with TemporaryDirectory() as tmp:
            with _fork_fixture(tmp, active_jobs=1, child_jobs=5) as (
                wrapper,
                _child,
                active_dir,
                child_dir,
            ):
                handler = make_handler(lambda: None, state_dir=active_dir)
                with _chdir(wrapper):
                    remote = self._call(handler, "100.64.0.7")
                    local = self._call(handler, "127.0.0.1")
                secrets = (
                    active_dir.name,
                    child_dir.name,
                    str(active_dir),
                    str(wrapper),
                )
                active_name = active_dir.name

        payload = json.loads(remote)
        # Not vacuous: there really was something to withhold.
        self.assertTrue(payload["suspect"])
        self.assertIsNone(payload["diagnosis"])
        for secret in secrets:
            self.assertNotIn(secret, remote, secret)
        self.assertNotIn(os.sep, remote)
        # ...and the loopback caller on the SAME handler does get the detail,
        # which proves the gate withheld it rather than an inert diagnosis.
        self.assertIn(active_name, local)

    def test_the_diagnosis_is_ttl_cached_not_walked_per_request(self) -> None:
        """The page can be reloaded freely and /api/jobs polls every 1.5s next
        door; a directory walk per request is not an acceptable price for an
        advisory. One probe must serve every request inside the TTL."""
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            handler = make_handler(lambda: None, state_dir=state_dir)
            with patch.object(
                state_health,
                "diagnose_state_dir",
                wraps=state_health.diagnose_state_dir,
            ) as probe:
                for _ in range(5):
                    self._call(handler, "127.0.0.1")
        self.assertEqual(probe.call_count, 1)

    def test_all_projects_board_never_probes_the_filesystem(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            state_dir.mkdir()
            handler = make_handler(lambda: None, all_projects=True, state_dir=state_dir)
            with patch.object(
                state_health, "diagnose_state_dir", side_effect=AssertionError
            ) as probe:
                got = self._call(handler, "127.0.0.1")
            probe.assert_not_called()
        self.assertIsNone(json.loads(got)["diagnosis"])

    def test_state_dir_none_never_probes_the_filesystem(self) -> None:
        """``make_handler(lambda: None)`` with no state_dir is what several
        existing tests build; it must not be made to walk anything."""
        handler = make_handler(lambda: None)
        with patch.object(
            state_health, "diagnose_state_dir", side_effect=AssertionError
        ) as probe:
            got = self._call(handler, "127.0.0.1")
        probe.assert_not_called()
        self.assertIsNone(json.loads(got)["diagnosis"])


class BannerCallSiteTests(unittest.TestCase):
    """*Where* the banner is rendered from is the whole point of Part C."""

    @staticmethod
    def _load_index() -> str:
        start = _PAGE_APP_JS.index("async function loadIndex")
        return _PAGE_APP_JS[start : _PAGE_APP_JS.index("function rows(", start)]

    def test_banner_sits_after_the_jobs_heading(self) -> None:
        body = self._load_index()
        self.assertGreater(body.index("stateDirHint("), body.index("<h2>Jobs</h2>"))

    def test_banner_is_not_in_the_empty_state_branch(self) -> None:
        """The reported bad dir held exactly ONE job, so the empty state never
        rendered — a hint placed there would have missed the real incident."""
        body = self._load_index()
        self.assertEqual(body.count("stateDirHint("), 1)
        self.assertLess(body.index("stateDirHint("), body.index("if (!shown.length)"))

    def test_diagnostics_is_fetched_once_at_bootstrap_not_polled(self) -> None:
        for needle in (
            "/api/diagnostics",
            "async function loadDiagnostics",
            "loadDiagnostics();",
        ):
            self.assertIn(needle, _PAGE_APP_JS, needle)
        self.assertEqual(_PAGE_APP_JS.count("loadDiagnostics();"), 1)
        # Same seam as loadMeta(): after tick()'s definition, never inside it.
        self.assertGreater(
            _PAGE_APP_JS.index("loadDiagnostics();"),
            _PAGE_APP_JS.index("async function tick()"),
        )
        tick_region = _PAGE_APP_JS[
            _PAGE_APP_JS.index("async function tick()") : _PAGE_APP_JS.index(
                "async function loadDiagnostics"
            )
        ]
        self.assertNotIn("/api/diagnostics", tick_region)

    def test_load_diagnostics_owns_its_own_failure(self) -> None:
        start = _PAGE_APP_JS.index("async function loadDiagnostics")
        body = _PAGE_APP_JS[start : _PAGE_APP_JS.index("loadMeta();", start)]
        for needle in ("try {", "catch", "r.ok"):
            self.assertIn(needle, body, needle)


class StateDirHintPurityTests(unittest.TestCase):
    """RENDERER_JS runs verbatim under node, so stateDirHint may not touch a
    browser API — and the project basename it renders is filesystem-derived
    and unsanitised when --state-dir is explicit, so its trip through esc()
    needs real execution rather than a substring check on the source."""

    def test_state_dir_hint_is_pure_escapes_and_defers_to_live_rows(self) -> None:
        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")

        harness = RENDERER_JS + r"""
const assert = require("assert");

// Throwing getters, not deletions: a free `document` inside stateDirHint
// resolves through globalThis at CALL time and trips the getter.
const POISON = ["document", "window", "location", "fetch", "navigator", "localStorage"];
for (const name of POISON) {
  try {
    Object.defineProperty(globalThis, name, {
      configurable: true,
      get() { throw new Error("stateDirHint touched browser global: " + name); },
    });
  } catch (e) { /* non-configurable in this node build; the ones below suffice */ }
}
for (const name of ["document", "window"]) {
  assert.throws(() => globalThis[name], /touched browser global/, name + " not poisoned");
}

// Nothing to say -> nothing rendered. A healthy board, an --all-projects
// board and a remote peer all land here and must render identically.
assert.strictEqual(stateDirHint(null, 0), "");
assert.strictEqual(stateDirHint(undefined, 0), "");
assert.strictEqual(stateDirHint({}, 0), "");
assert.strictEqual(stateDirHint({suspect: false, message: "x"}, 0), "");

// The real incident: 1 job here, 24 over there.
const real = stateDirHint({
  suspect: true,
  jobs: 1,
  project: "puppetmaster-c3177e6032c4",
  candidate: "Puppetmaster-b92145e840c8",
  candidate_jobs: 24,
  message: "puppetmaster-c3177e6032c4 has 1 job(s) but nested repo Puppetmaster has 24",
}, 1);
assert.ok(real.includes("state-dir-hint"), real);
assert.ok(real.includes("puppetmaster projects"), real);
assert.ok(real.includes("Puppetmaster"), real);

// An unsanitised, filesystem-derived name reaches this string. It must be
// escaped, and the raw tag must be nowhere in the output.
const PAYLOAD = '<img src=x onerror="alert(1)">';
const attacked = stateDirHint({suspect: true, jobs: 1, message: "dir " + PAYLOAD}, 1);
assert.ok(!attacked.includes(PAYLOAD), "raw payload survived");
assert.ok(!attacked.includes("<img"), "raw tag survived");
assert.ok(!attacked.includes('onerror="'), "raw attribute survived");
assert.ok(attacked.includes("&lt;img"), "payload was not escaped: " + attacked);

// Staleness guard: the server caches the diagnosis for ~15s, so a board that
// is visibly busier than the "nearly empty" dir the detector complained about
// is not this incident. Drop the banner rather than contradict the rows.
const stale = {suspect: true, jobs: 1, message: "m"};
assert.strictEqual(stateDirHint(stale, 2), "");
assert.strictEqual(stateDirHint(stale, 24), "");
assert.notStrictEqual(stateDirHint(stale, 1), "");
assert.notStrictEqual(stateDirHint(stale, 0), "");
// A missing/garbage count must not silently suppress the banner.
assert.notStrictEqual(stateDirHint({suspect: true, message: "m"}, 5), "");

console.log("state-dir-hint-ok");
"""
        completed = subprocess.run(
            [node, "-e", harness],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("state-dir-hint-ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
