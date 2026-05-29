#!/usr/bin/env bash
#
# Puppetmaster 60-second demo — runs end-to-end with NO API keys (local +
# shell adapters only), so it works on a clean machine first try. This is the
# script the demo GIF records; see `demo.tape` to regenerate the GIF.
#
#   ./scripts/demo.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

STATE=${PUPPETMASTER_DEMO_STATE:-/tmp/puppetmaster-demo}
rm -rf "$STATE"; mkdir -p "$STATE"
export PUPPETMASTER_STATE_DIR="$STATE"
PM="python -m puppetmaster"

cyan='\033[1;36m'; dim='\033[2m'; off='\033[0m'
banner() { printf "\n${cyan}━━ %s ━━${off}\n" "$1"; }
pause() { sleep "${DEMO_PAUSE:-1.2}"; }

banner "1 / 4   Route each task to the cheapest model that can do it"
printf "${dim}\$ python -m bench.router_savings${off}\n"
python -m bench.router_savings | sed -n '/Per-task/,/Total savings/p'
pause

banner "2 / 4   Fan out a 6-role swarm — workers are independent OS processes"
printf "${dim}\$ puppetmaster run \"Investigate the auth flow\" --config examples/enterprise-workflow.json${off}\n"
$PM run "Investigate the auth flow and propose a fix" --config examples/enterprise-workflow.json
JOB=$($PM last)
pause

banner "3 / 4   Read the stitched summary — workers never shared a transcript"
printf "${dim}\$ puppetmaster show %s${off}\n" "$JOB"
$PM show "$JOB" | sed -n '1,28p'
pause

banner "4 / 4   Follow-up reads are FREE — 0 model calls, 0 tokens, \$0.00"
printf "${dim}\$ python -m bench.followup_cost --job-id %s --queries 5${off}\n" "$JOB"
python -m bench.followup_cost --job-id "$JOB" --state-dir "$STATE" --queries 5 \
  | sed -n '/Queries run/,/Cost:/p'
pause

banner "Watch it live in your browser (zero deps):"
printf "    ${cyan}python -m puppetmaster dashboard %s${off}\n\n" "$JOB"
