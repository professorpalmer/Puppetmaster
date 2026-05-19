"""Background launcher for `codegraph index`.

The MCP server spawns this script as a detached subprocess so that
indexing — which routinely takes minutes to hours on large repos — never
blocks the stdio transport. The launcher:

1. Re-acquires the global CodeGraph indexer lock (the parent released
   its handle after validating that no other indexer was running).
2. Runs ``codegraph index`` in the target workspace, streaming child
   stdout/stderr through this process so the MCP server's captured log
   files contain the indexer output.
3. Releases the lock on exit (success, failure, or signal) via an
   ``atexit`` hook and a finally block.

The launcher is intentionally minimal — no MCP plumbing, no Python
imports beyond stdlib + the codegraph helpers — so the indexer can keep
running even if the parent MCP server is later restarted by Cursor.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from puppetmaster.codegraph import (
    CODEGRAPH_COMMAND,
    CodegraphLock,
    CodegraphLockBusy,
)


def main(argv: Optional[list[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) < 2:
        sys.stderr.write(
            "usage: python -m puppetmaster.codegraph_index_runner <cwd> <lock_path>\n"
        )
        return 2
    target_cwd = args[0]
    lock_path = Path(args[1])

    lock = CodegraphLock(lock_path)
    try:
        lock.acquire()
    except CodegraphLockBusy as exc:
        sys.stderr.write(f"{exc}\n")
        return 75  # EX_TEMPFAIL — caller can retry once the holder finishes.

    atexit.register(lock.release)

    def _release_on_signal(signum: int, _frame) -> None:
        try:
            lock.release()
        finally:
            sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _release_on_signal)
    signal.signal(signal.SIGINT, _release_on_signal)

    try:
        completed = subprocess.run(
            [CODEGRAPH_COMMAND, "index"],
            cwd=target_cwd or None,
            env=os.environ.copy(),
            check=False,
        )
        return completed.returncode
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
