"""Spawn-safe helper for multiprocess run_id uniqueness checks.

Kept free of mcp_server / swarm_launch imports so ProcessPoolExecutor children
do not re-import the heavy MCP module graph during unittest discovery.
"""

from __future__ import annotations

FROZEN_MS = 1_785_429_501_672
FROZEN_PID = 74978


def spawn_new_run_id_batch(count: int) -> list[str]:
    from unittest.mock import patch

    from puppetmaster.run_id import new_run_id

    with patch("puppetmaster.run_id.time.time", return_value=FROZEN_MS / 1000.0), patch(
        "puppetmaster.run_id.os.getpid", return_value=FROZEN_PID
    ):
        return [new_run_id("mcp") for _ in range(count)]
