from __future__ import annotations

from puppetmaster.installers import (
    CLAUDE_NEXT_STEPS_GUIDANCE,
    CODEX_SANDBOX_GUIDANCE,
    CURSOR_NEXT_STEPS_GUIDANCE,
    HERMES_NEXT_STEPS_GUIDANCE,
)


_NOISY_LOG_EVENTS = {"task.lease_renewed", "run.heartbeat", "task.saved"}

_OPENAI_EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh")

_CODEX_EFFORT_LEVELS = ("low", "medium", "high")

_HERMES_EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh")

# Claude Code >= 2.1.204 exposes `--effort` with five levels (verified against
# the CLI's --help; older versions reject the flag, so effort variants require
# a current CLI).
_CLAUDE_CODE_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

_EFFORT_TOKEN_MULTIPLIERS = {
    "none": 0.7,
    "minimal": 0.7,
    "low": 0.7,
    "medium": 1.0,
    "high": 2.0,
    "xhigh": 3.0,
    # Anthropic documents max as unconstrained token spending; 4x is a
    # deliberately conservative cost-estimate anchor, not a cap.
    "max": 4.0,
}
