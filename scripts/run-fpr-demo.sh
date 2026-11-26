#!/usr/bin/env bash
# FPR demo, native: backend + dashboard, then rig-anchored simulated trajectories replayed through receiver -> NodeEngine -> backend.
#   ./scripts/run-fpr-demo.sh [contract=v7_node] [speed=12] [port=5080]     (Docker: docker compose --profile fpr-demo up --build backend-fpr fpr-replay)
set -euo pipefail
CONTRACT="${1:-v7_node}"; SPEED="${2:-12}"; PORT="${3:-5080}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python3}"
[ -f artifacts/dashboard/wwwroot/index.html ] || dotnet publish src/DigitalTwin.Dashboard -c Release -o artifacts/dashboard
dotnet build src/DigitalTwin.Backend -c Release --nologo -v q
( cd src/DigitalTwin.Backend/bin/Release/net10.0 && exec dotnet DigitalTwin.Backend.dll --Urls="http://0.0.0.0:$PORT" --Backend:Replay:Enabled=false \
    --Backend:PhysicsTwin=None --Backend:DashboardPath="$ROOT/artifacts/dashboard/wwwroot" --Backend:Replay:Source=node-live \
    "--Backend:MachineId=OH2-PUMP-01 [SIMULATION; rig-anchored replay]" ) &
BE=$!; trap 'kill $BE 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do curl -fs "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 && break; sleep 0.5; done
echo "== dashboard: http://localhost:$PORT/  contract=$CONTRACT [SIMULATION; rig-anchored replay]"
"$PY" scripts/fpr_replay.py --contract "$CONTRACT" --speed "$SPEED" --backend "http://127.0.0.1:$PORT" --loop
