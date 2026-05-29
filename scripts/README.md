# Scripts

Helper scripts for the demo and its reproducible GIF.

| File | What it does |
|---|---|
| [`demo.sh`](demo.sh) | The 60-second tour — **no API keys required.** Routes a task mix by cost, fans out a swarm as independent processes, reads the stitched summary, and proves follow-up reads cost \$0.00. Safe to run on a clean machine. |
| [`demo.tape`](demo.tape) | [VHS](https://github.com/charmbracelet/vhs) "GIF-as-code" script that records `demo.sh` into [`docs/demo.gif`](../docs/demo.gif). |

## Run the demo

```bash
./scripts/demo.sh
```

## Regenerate the GIF

Requires [VHS](https://github.com/charmbracelet/vhs) (`brew install vhs`):

```bash
vhs scripts/demo.tape   # writes docs/demo.gif
```

The GIF is committed so the README renders it without a build step; regenerate only when the demo flow changes.
