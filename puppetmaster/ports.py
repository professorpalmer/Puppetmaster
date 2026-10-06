"""Deterministic per-worktree port allocation (B1).

Parallel worktrees that each spin up a dev server / preview on a fixed port
(3000, 5173, 8000, …) collide the moment two workers run at once — the second
bind fails or, worse, two agents talk to the same server and cross-contaminate.

We avoid coordination entirely by deriving a stable, collision-resistant port
*block* from the resolved worktree path. The same worktree always maps to the
same block; different worktrees map to different blocks with overwhelming
probability. The worker (and any dev server / agent it launches) reads these env
vars instead of hardcoding a port.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

# A range above the common service ports (3000/5173/8000/8080) and *below* the
# Linux default ephemeral floor (32768) so a hinted port never collides with an
# OS-assigned ephemeral port. macOS's ephemeral range starts higher (49152), so
# staying under 32768 is the cross-OS-safe choice. Carved into fixed-size
# per-worktree blocks.
_PORT_RANGE_START = 10000
_PORT_RANGE_END = 32768
_BLOCK_SIZE = 8  # contiguous ports reserved per worktree

_SPAN = (_PORT_RANGE_END - _PORT_RANGE_START) // _BLOCK_SIZE


def worktree_port_base(worktree: Union[str, Path]) -> int:
    """A deterministic base port for ``worktree``, aligned to a block boundary.

    Stable across runs and processes because it depends only on the resolved
    path, so re-running in the same worktree reuses the same ports.
    """
    resolved = str(Path(worktree).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).digest()
    block = int.from_bytes(digest[:4], "big") % _SPAN
    return _PORT_RANGE_START + block * _BLOCK_SIZE


def worktree_port_env(worktree: Union[str, Path], *, count: int = _BLOCK_SIZE) -> dict[str, str]:
    """Env vars exposing a worktree's port block.

    Sets ``PUPPETMASTER_PORT_BASE``, ``PUPPETMASTER_PORT_0..N-1``, and a
    convenience ``PORT`` for the common single-server case.
    """
    base = worktree_port_base(worktree)
    count = max(1, min(count, _BLOCK_SIZE))
    env = {
        "PUPPETMASTER_PORT_BASE": str(base),
        "PUPPETMASTER_PORT_COUNT": str(count),
        "PORT": str(base),
    }
    for index in range(count):
        env[f"PUPPETMASTER_PORT_{index}"] = str(base + index)
    return env


def apply_worktree_ports(
    environment: dict, worktree: Union[str, Path], *, override_port: bool = False
) -> dict:
    """Inject per-worktree port env into ``environment`` in place and return it.

    A ``PORT`` the caller already pinned is preserved unless ``override_port`` is
    set, so an explicit user/job port always wins. Best-effort: a failure to
    compute ports never breaks the worker launch.
    """
    try:
        ports = worktree_port_env(worktree)
    except Exception:
        return environment
    for key, value in ports.items():
        if key == "PORT" and not override_port and environment.get("PORT"):
            continue
        environment[key] = value
    return environment

