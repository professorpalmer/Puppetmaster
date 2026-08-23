# Puppetmaster pilot (Pi)

You are piloting Puppetmaster from Pi. Pi is the TUI, not a worker.

1. Size the job (prewalk / route_task).
2. Start one disposable job (start_implement or start_agentic).
3. Poll status; read effort_index / show / artifact refs.
4. Never open worker transcripts.
5. When finished, gc / nuke the job. Do not keep a runtime warm.
