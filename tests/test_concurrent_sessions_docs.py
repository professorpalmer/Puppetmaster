"""Regression contract for concurrent-session operator documentation.

This deliberately checks the statements that prevent an operator from opening
the wrong project dashboard or running competing write jobs in one checkout.
Runtime behavior is covered by its focused state, dashboard, and worktree
tests; this file keeps the public operating contract connected to that behavior.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONCURRENT_SESSIONS = ROOT / "docs" / "CONCURRENT_SESSIONS.md"


def test_concurrent_sessions_operator_contract_is_discoverable_and_complete():
    guide = CONCURRENT_SESSIONS.read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    dashboard = (ROOT / "docs" / "DASHBOARD.md").read_text(encoding="utf-8")
    mobile = (ROOT / "docs" / "MOBILE.md").read_text(encoding="utf-8")

    assert "target workspace selects\nthe default state directory" in guide
    assert "`--state-dir` and `PUPPETMASTER_STATE_DIR` override" in guide
    assert "resolved from the launcher shell's current\ndirectory" in guide
    assert "dashboard --all-projects --background" in guide
    assert "URL printed by the\ncommand or returned by the MCP tool" in guide
    assert "Do not run two full-edit or in-place edit jobs against the same checkout" in guide
    assert "CONCURRENT_SESSIONS.md" in readme
    assert "CONCURRENT_SESSIONS.md" in docs_index
    assert "CONCURRENT_SESSIONS.md" in dashboard
    assert ":8787/" not in mobile
