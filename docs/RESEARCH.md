# Durable Autoresearch

Coordinate claim / run / publish / verify experiment loops against Puppetmaster's durable SQLite (or file) store — not an ephemeral shared transcript.

## Why

Autoresearch needs **exclusive claims**, **replayable results**, and **verification by re-run**. Those are store problems, not chat-context problems. Lab verbs heartbeat the job so long-lived exploration is not marked `STALLED` by the liveness reaper.

## Artifacts

Research reuses existing `ArtifactType` values. Semantics live in `payload.research_kind`:

| `research_kind` | ArtifactType | Meaning |
| --- | --- | --- |
| `result` | `FINDING` | Harness metrics for a claimed fingerprint |
| `insight` | `FINDING` / announce `DECISION` | Lab notes |
| `hypothesis` | `DECISION` | Candidate to explore |
| `best` | `DECISION` | Leaderboard winner |
| `verification` | `VERIFICATION` | Re-run pass/fail (+ `DERIVED_FROM` edge) |

No `RESEARCH_*` enum members are added.

## Claim algorithm

1. `fingerprint = sha256(hypothesis \| harness_id \| canonical_config)`
2. Acquire lock `research-claim:{job_id}:{fingerprint}`
3. Dedup open `research-runner` tasks with the same fingerprint
4. `save_task` + `claim_task`; renew the lease while running
5. On publish: complete the task and release the lock

## CLI

```bash
python -m puppetmaster research init "Explore zlib ratios"
python -m puppetmaster research announce <job_id> "lab open"
python -m puppetmaster research claim <job_id> "level-6" --config '{"level":6,"seed":7,"size":4096}' --run
python -m puppetmaster research publish <job_id> --task-id <task> --run
python -m puppetmaster research verify <job_id> <artifact_id>
python -m puppetmaster research leaderboard <job_id>
python -m puppetmaster research think <job_id>          # zero-token artifact recall
python -m puppetmaster research demo                    # ToyCompressionHarness end-to-end
```

`research think` reads durable artifacts only — it does not spawn a nested LLM swarm.

## Toy harness + GPU later

`ToyCompressionHarness` is a deterministic CPU zlib / byte-entropy microbench (`bits_per_byte`, lower is better). A future GPU / nanochat trainer implements the same `ExperimentHarness` Protocol (`harness_id` + `run(config) -> metrics`); see the `GpuHarnessAdapter` docstring stub in `puppetmaster/research.py`. No GPU runtime or remote join is wired in v1.

## Demo brief

`python -m puppetmaster research demo` writes `examples/autoresearch-durable-brief.md` from durable artifacts.
