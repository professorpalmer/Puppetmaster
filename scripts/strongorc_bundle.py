"""Build the packaged StrongOrc observation bundle from the official board.

Usage:
    python3 scripts/strongorc_bundle.py [path/to/official.json]

The default source is the board data at
``~/Projects/puppetmaster-gg/strongorc/source/official.json``. The output is
``puppetmaster/baselines/strongorc-ranking-v1.json``.

Each board row is one model and effort on the OpenRouter channel. The worker
track becomes one observation per PM worker role, keyed to the registry id
``agentic/<openrouter id>``. The orchestrator track measures the parent and
has no PM worker role, so it is not imported. Contributor rows and the
``max`` effort (not a PM effort) are skipped.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puppetmaster.community_observations import WORKER_ROLES, parse_observation_bundle  # noqa: E402
from puppetmaster.swarm_reasoning import EFFORTS  # noqa: E402

DEFAULT_SOURCE = Path.home() / "Projects" / "puppetmaster-gg" / "strongorc" / "source" / "official.json"
OUTPUT = ROOT / "puppetmaster" / "baselines" / "strongorc-ranking-v1.json"
SKIPPED_TAGS = frozenset({"contrib"})


def wilson(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a rate over ``n`` attempts."""
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def build(source: Path) -> dict:
    board = json.loads(source.read_text(encoding="utf-8"))
    dataset = board["dataset"]
    per_track = int(dataset["attemptsPerRow"]) // 2
    published = str(dataset["generatedAt"])[:10]
    entries = []
    for row in sorted(board["results"], key=lambda r: (r["model"], r["effort"])):
        if row.get("tag") in SKIPPED_TAGS or row["effort"] not in EFFORTS:
            continue
        low, high = wilson(float(row["worker"]), per_track)
        for role in sorted(WORKER_ROLES):
            entries.append({
                "registry_id": f"agentic/{row['model']}",
                "adapter": "agentic",
                "role": role,
                "effort": row["effort"],
                "provider": dataset["channel"],
                "track": "worker",
                "bank": dataset["bank"],
                "harness": dataset["harnessVersion"],
                "pass_rate": round(float(row["worker"]), 6),
                "ci_low": round(low, 6),
                "ci_high": round(high, 6),
                "sample_count": per_track,
                "published": published,
            })
    bundle = {
        "bundle_id": "strongorc-ranking-v1",
        "not_ground_truth": True,
        "never_writes_capability_score": True,
        "notes": (
            "Official StrongOrc ranking-v1 worker-track rates (task-equalized strict "
            "pass) on the confined OpenRouter channel. Joins only agentic/<openrouter id> "
            "registry entries at the effort the worker runs. CI is a 95% Wilson interval "
            f"over {per_track} attempts per track. Built by scripts/strongorc_bundle.py."
        ),
        "source": "https://strongorc.puppetmaster.gg/",
        "bank": dataset["bank"],
        "bank_commitment": dataset["bankCommitment"],
        "harness": dataset["harnessVersion"],
        "generated_at": dataset["generatedAt"],
        "entries": entries,
    }
    parse_observation_bundle(bundle)
    return bundle


def main(argv: list[str]) -> int:
    source = Path(argv[1]).expanduser() if len(argv) > 1 else DEFAULT_SOURCE
    bundle = build(source)
    OUTPUT.write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
    models = sorted({entry["registry_id"] for entry in bundle["entries"]})
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(bundle['entries'])} entries, {len(models)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
