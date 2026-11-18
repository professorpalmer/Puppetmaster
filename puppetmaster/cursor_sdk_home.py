"""Where ``@cursor/sdk`` lives so the cursor adapter survives reinstalls.

PyPI wheels cannot ship ``node_modules``, and ``uv tool upgrade`` / ``pipx
upgrade`` build a fresh environment, so an SDK bootstrapped next to the
installed package vanishes on every upgrade. The SDK home under the app state
root is version-independent instead.

Node resolves ESM bare specifiers from the importing file and ignores
``NODE_PATH``, so the runner has to execute *beside* the SDK it imports. When
only the SDK home has it, a byte-identical copy of the packaged runner is
placed there and run from that directory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from puppetmaster.state import app_state_root

PACKAGED_RUNNER = Path(__file__).resolve().with_name("cursor_sdk_runner.mjs")


def sdk_home() -> Path:
    return app_state_root() / "cursor-sdk"


def _sdk_beside(directory: Path) -> Optional[Path]:
    for ancestor in [directory, *directory.parents]:
        candidate = ancestor / "node_modules" / "@cursor" / "sdk"
        if _is_installed(candidate):
            return candidate
    return None


def _is_installed(candidate: Path) -> bool:
    # A partial npm install leaves the directory without package.json.
    return (candidate / "package.json").is_file()


def packaged_sdk() -> Optional[Path]:
    """The SDK Node finds walking up from the packaged runner, if any."""
    return _sdk_beside(PACKAGED_RUNNER.parent)


def home_sdk() -> Optional[Path]:
    candidate = sdk_home() / "node_modules" / "@cursor" / "sdk"
    return candidate if _is_installed(candidate) else None


def cursor_runner() -> Path:
    """The runner path from which Node can import ``@cursor/sdk``.

    Falls back to the packaged runner when neither location has the SDK, so
    the worker fails with the usual ``sdk_not_installed`` classification.
    """
    if packaged_sdk() is not None or home_sdk() is None:
        return PACKAGED_RUNNER
    source = PACKAGED_RUNNER.read_bytes()
    target = sdk_home() / PACKAGED_RUNNER.name
    try:
        if target.read_bytes() == source:
            return target
    except OSError:
        pass
    # Concurrent workers may each refresh the copy; os.replace keeps every
    # reader on a complete file.
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".runner-", suffix=".mjs")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(source)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target
