"""Keep child processes from flashing console windows on Windows.

Puppetmaster's MCP server and worker runtimes run under console-less hosts
(Cursor's MCP client, Electron shells). On Windows, any console-subsystem
child (git, node, npm, python, pytest) spawned by a console-less parent
ALLOCATES A NEW VISIBLE CONSOLE unless CREATE_NO_WINDOW is passed. The
package has dozens of subprocess call sites across adapters, codegraph,
gates, and installers; rather than thread a creationflags kwarg through
every one (and every future one), this module installs a process-wide
default: every Popen gets CREATE_NO_WINDOW unless the caller explicitly
asked for a console (CREATE_NEW_CONSOLE) or full detachment
(DETACHED_PROCESS).

Grandchildren inherit the parent's hidden console, so worker subprocesses
that themselves spawn git/pytest stay silent as long as their interpreter
also imports this module (all Puppetmaster entry points install it).

Escape hatch: set PUPPETMASTER_SHOW_CONSOLES=1 to disable (debugging).
"""
from __future__ import annotations

import os
import subprocess

_CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
_DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)

_EXPLICIT_CONSOLE_FLAGS = _CREATE_NEW_CONSOLE | _DETACHED_PROCESS | _CREATE_NO_WINDOW

_original_popen_init = subprocess.Popen.__init__


def effective_creationflags(flags):
    """Add CREATE_NO_WINDOW unless the caller made an explicit console choice."""
    flags = flags or 0
    if flags & _EXPLICIT_CONSOLE_FLAGS:
        return flags
    return flags | _CREATE_NO_WINDOW


def _hidden_popen_init(self, *args, **kwargs):
    kwargs["creationflags"] = effective_creationflags(kwargs.get("creationflags", 0))
    _original_popen_init(self, *args, **kwargs)


def hide_child_consoles():
    """Install the hidden-console default. Returns True when active."""
    if os.name != "nt" or os.environ.get("PUPPETMASTER_SHOW_CONSOLES") == "1":
        return False
    if subprocess.Popen.__init__ is not _hidden_popen_init:
        subprocess.Popen.__init__ = _hidden_popen_init
    return True
