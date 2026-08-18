"""Turn public SWE-bench results into Puppetmaster role cards.

This module does one small job. It reads the public SWE-bench "Bash Only"
leaderboard and creates an ``implement`` scorecard that Puppetmaster already
knows how to import.

The leaderboard tests every model with mini-SWE-agent. Puppetmaster may run the
same model through Codex, Cursor, or another adapter. Those are not identical
setups, so this module never guesses the mapping. The caller must name the exact
leaderboard row and the exact Puppetmaster registry entry that should receive
it. The saved provenance keeps the original mini-SWE-agent source visible.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional


# SWE-bench publishes the website's leaderboard data in this repository. We
# read the JSON source directly instead of scraping the rendered web page.
_SWEBENCH_REPO = "SWE-bench/swe-bench.github.io"
_SWEBENCH_PATH = "data/leaderboards.json"
_COMMITS_URL = (
    f"https://api.github.com/repos/{_SWEBENCH_REPO}/commits"
    f"?path={_SWEBENCH_PATH}&per_page=1"
)
_RAW_URL = (
    "https://raw.githubusercontent.com/"
    f"{_SWEBENCH_REPO}/{{revision}}/{_SWEBENCH_PATH}"
)
_BENCHMARK_ID = "swe-bench-bash-only"
_AGENT = "mini-SWE-agent"
# The Bash Only board currently contains 500 tasks. Some leaderboard rows embed
# all 500 task results and some only publish the aggregate. In the latter case,
# this public board size is the honest sample count available to us.
_MIN_COMPARABLE_ROWS = 5
_BASH_ONLY_SAMPLE_COUNT = 500


@dataclass(frozen=True)
class RegistryModelMapping:
    """A deliberate link between one leaderboard row and one local model.

    ``leaderboard_name`` must match the row's published ``name`` exactly. The
    other three fields identify the existing Puppetmaster registry entry. This
    small amount of explicit setup prevents a similar-looking model name from
    receiving evidence that belongs to another model or adapter.
    """

    registry_id: str
    adapter: str
    adapter_model_name: str
    leaderboard_name: str

    def __post_init__(self) -> None:
        for field_name in (
            "registry_id",
            "adapter",
            "adapter_model_name",
            "leaderboard_name",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} must be non-empty")


JsonGetter = Callable[[str], Any]


def _stdlib_get_json(url: str) -> Any:
    """Read one trusted GitHub JSON document without adding a dependency."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "puppetmaster-swebench-import",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _bash_only_rows(payload: dict) -> list[dict]:
    """Return valid mini-SWE-agent rows from the Bash Only board.

    Other SWE-bench boards use different task sets or agent setups. Mixing them
    would make the comparison meaningless, so this function accepts only one
    named board and one named agent.
    """
    boards = payload.get("leaderboards") if isinstance(payload, dict) else None
    if not isinstance(boards, list):
        raise ValueError("SWE-bench payload has no leaderboards list")
    board = next(
        (item for item in boards if isinstance(item, dict) and item.get("name") == "bash-only"),
        None,
    )
    if board is None:
        raise ValueError("SWE-bench payload has no bash-only leaderboard")
    results = board.get("results")
    if not isinstance(results, list):
        raise ValueError("SWE-bench bash-only leaderboard has no results list")
    rows = [
        row
        for row in results
        if isinstance(row, dict)
        and row.get("agent") == _AGENT
        and isinstance(row.get("resolved"), (int, float))
        and not isinstance(row.get("resolved"), bool)
    ]
    if len(rows) < _MIN_COMPARABLE_ROWS:
        raise ValueError(
            f"SWE-bench bash-only leaderboard has only {len(rows)} comparable {_AGENT} rows; "
            f"need at least {_MIN_COMPARABLE_ROWS}"
        )
    return rows


def _classify_resolved_scale(rows: list[dict]) -> str:
    """Return ``percent`` or ``rate`` for one comparable resolved set.

    Values outside 0..100 are rejected. A set that contains both a percent
    score (``> 1``) and an exclusive unit-interval rate (``0 < value < 1``)
    is mixed and fails closed. ``0`` and ``1.0`` stay valid on a percent
    board and do not count as mixed.
    """
    values = []
    for row in rows:
        resolved = float(row["resolved"])
        if not 0.0 <= resolved <= 100.0:
            raise ValueError(
                f"SWE-bench row {str(row.get('name') or '')!r} resolved must be 0..100"
            )
        values.append(resolved)
    has_percent = any(value > 1.0 for value in values)
    has_unit_interval = any(0.0 < value < 1.0 for value in values)
    if has_percent and has_unit_interval:
        raise ValueError("SWE-bench comparable rows have mixed resolved scale")
    return "percent" if has_percent else "rate"


def _quality_from_resolved(resolved: float, resolved_scale: str) -> float:
    """Store quality on the unit interval without inventing a second scale."""
    if resolved_scale == "percent":
        return round(resolved / 100.0, 6)
    return round(resolved, 6)


def _percentile_capability(rows: list[dict], resolved: float) -> int:
    """Place one result from last to first on a 0-100 scale.

    Rank uses the published resolved values as-is. Scorecard ``quality`` is a
    separate scale-normalized copy. Puppetmaster's router also needs a 0-100
    capability value, so the MVP uses position within the comparable
    leaderboard group. A value of 80 means the submission ranks at or above
    roughly 80 percent of that group. It does not mean 80 percent of SWE-bench
    tasks passed.

    Ties receive the same rank. The exact source revision and comparison count
    are saved so the number can always be reproduced.
    """
    values = sorted(float(row["resolved"]) for row in rows)
    below_or_equal = sum(1 for value in values if value <= resolved)
    percentile = (below_or_equal - 1) / (len(values) - 1)
    return max(0, min(100, round(percentile * 100)))


def _mapped_row(rows: list[dict], mapping: RegistryModelMapping) -> dict:
    """Find the one row the caller named, or stop if the name is not unique."""
    matches = [row for row in rows if str(row.get("name") or "") == mapping.leaderboard_name]
    if not matches:
        raise ValueError(
            f"SWE-bench model mapping not found: {mapping.leaderboard_name!r} "
            f"for {mapping.registry_id}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"SWE-bench model mapping is ambiguous: {mapping.leaderboard_name!r} "
            f"matched {len(matches)} rows"
        )
    return matches[0]


def build_swebench_bash_only_bundle(
    payload: dict,
    *,
    mappings: Iterable[RegistryModelMapping],
    source_revision: str,
    published: str,
    source_url: str = "",
) -> dict:
    """Build an importable Puppetmaster bundle from one pinned leaderboard."""
    revision = str(source_revision or "").strip()
    published_at = str(published or "").strip()
    if not revision:
        raise ValueError("source_revision must be non-empty")
    if not published_at:
        raise ValueError("published must be non-empty")
    # First narrow the source to one board and one agent. Later, each mapping is
    # narrowed again to the target row's exact mini-SWE-agent version.
    rows = _bash_only_rows(payload)
    mapping_list = list(mappings)
    if not mapping_list:
        raise ValueError("at least one explicit model mapping is required")
    ids = [mapping.registry_id for mapping in mapping_list]
    if len(ids) != len(set(ids)):
        raise ValueError("registry model mappings must have unique registry_id values")

    pinned_url = source_url or _RAW_URL.format(revision=revision)
    entries: list[dict] = []
    for mapping in mapping_list:
        # The caller chooses the row. We do not normalize, guess, or choose the
        # newest model on their behalf.
        row = _mapped_row(rows, mapping)
        resolved = float(row["resolved"])
        for field_name in ("instance_cost", "instance_calls"):
            value = row.get(field_name)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) < 0.0
            ):
                raise ValueError(
                    f"SWE-bench row {mapping.leaderboard_name!r} "
                    f"{field_name} must be non-negative"
                )
        harness_version = str(row.get("mini-swe-agent_version") or "").strip()
        if not harness_version:
            raise ValueError(
                f"SWE-bench row {mapping.leaderboard_name!r} has no mini-SWE-agent version"
            )
        # mini-SWE-agent changes over time. Compare the model only with rows that
        # used the same version, otherwise the agent change would affect rank.
        comparable = [
            candidate
            for candidate in rows
            if str(candidate.get("mini-swe-agent_version") or "").strip()
            == harness_version
        ]
        if len(comparable) < _MIN_COMPARABLE_ROWS:
            raise ValueError(
                f"SWE-bench harness version {harness_version!r} has only "
                f"{len(comparable)} comparable rows; need at least {_MIN_COMPARABLE_ROWS}"
            )
        resolved_scale = _classify_resolved_scale(comparable)
        details = row.get("per_instance_details")
        if isinstance(details, dict) and details:
            # Prefer the task-level evidence when the row includes it.
            sample_count = len(details)
            sample_count_source = "embedded_per_instance_details"
        else:
            # Some public rows omit task details. The board still represents the
            # fixed 500-task Bash Only set, so record that source explicitly.
            sample_count = _BASH_ONLY_SAMPLE_COUNT
            sample_count_source = "bash_only_board_contract"
        capability = _percentile_capability(comparable, resolved)
        # Keep this evidence on the role card itself. A model may later combine
        # local evidence for one role with community evidence for another.
        provenance = {
            "source": "community_benchmark",
            "benchmark": _BENCHMARK_ID,
            "leaderboard": "bash-only",
            "agent": _AGENT,
            "harness_version": harness_version,
            "comparison_count": len(comparable),
            "source_url": pinned_url,
            "source_revision": revision,
            "raw_model_name": mapping.leaderboard_name,
            "capability_method": "leaderboard_percentile",
            "resolved_scale": resolved_scale,
            "sample_count_source": sample_count_source,
        }
        card = {
            "capability": capability,
            "quality": _quality_from_resolved(resolved, resolved_scale),
            "sample_count": sample_count,
            "last_calibrated": str(row.get("date") or published_at),
            "provenance": provenance,
        }
        if isinstance(row.get("instance_cost"), (int, float)):
            card["cost_per_task_usd"] = round(float(row["instance_cost"]), 6)
        if isinstance(row.get("instance_calls"), (int, float)):
            card["calls_per_task"] = round(float(row["instance_calls"]), 6)
        # ``adapter`` is copied from the caller's explicit mapping. This does not
        # claim that SWE-bench ran that adapter; the provenance above names the
        # mini-SWE-agent harness that actually produced the result.
        entries.append(
            {
                "id": mapping.registry_id,
                "adapter": mapping.adapter,
                "adapter_model_name": mapping.adapter_model_name,
                "role_scorecards": {"implement": card},
                "score_provenance": {
                    "source": "community_baseline",
                    "sample_count": sample_count,
                    "notes": (
                        f"{_BENCHMARK_ID} {_AGENT}; explicit mapping from "
                        f"{mapping.leaderboard_name!r}"
                    ),
                },
            }
        )

    return {
        "bundle_id": _BENCHMARK_ID,
        "version": revision[:12],
        "published": published_at,
        "adapter_scoped": True,
        "not_ground_truth": True,
        "methodology": [
            "SWE-bench Bash Only",
            f"fixed agent={_AGENT}",
            "capability=weak percentile rank among comparable submissions",
        ],
        "notes": (
            "Community model prior translated by explicit registry mapping. "
            "Review with import-baseline --dry-run before writing."
        ),
        "source_url": pinned_url,
        "entries": entries,
    }


def fetch_swebench_bash_only_bundle(
    mappings: Iterable[RegistryModelMapping],
    *,
    get_json: Optional[JsonGetter] = None,
    published: Optional[str] = None,
) -> dict:
    """Fetch the latest leaderboard revision and build a reproducible bundle.

    GitHub's default-branch URL can change between runs. We first ask which
    commit last changed the leaderboard file, then fetch the file through that
    immutable commit. The bundle stores both values, so another person can read
    the same bytes later.
    """
    getter = get_json or _stdlib_get_json
    commits = getter(_COMMITS_URL)
    if not isinstance(commits, list) or not commits or not isinstance(commits[0], dict):
        raise ValueError("SWE-bench commit lookup returned no revision")
    revision = str(commits[0].get("sha") or "").strip()
    if not revision:
        raise ValueError("SWE-bench commit lookup returned an empty revision")
    # Fetch through the commit SHA, not through ``main`` or ``master``.
    source_url = _RAW_URL.format(revision=revision)
    payload = getter(source_url)
    commit = commits[0].get("commit")
    committer = commit.get("committer") if isinstance(commit, dict) else None
    committed_at = committer.get("date") if isinstance(committer, dict) else None
    if published is None:
        # Use the source commit's date instead of today's date. Fetching the same
        # commit tomorrow should produce the same bundle.
        if not isinstance(committed_at, str) or len(committed_at) < 10:
            raise ValueError("SWE-bench commit lookup returned no publication date")
        published_at = committed_at[:10]
    else:
        published_at = published
    return build_swebench_bash_only_bundle(
        payload,
        mappings=mappings,
        source_revision=revision,
        published=published_at,
        source_url=source_url,
    )
