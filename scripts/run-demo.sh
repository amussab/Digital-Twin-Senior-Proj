#!/usr/bin/env bash
# One command: publish the dashboard (once), build the backend, run it. One process, one LAN URL.
#   ./scripts/run-demo.sh [demo|twin-sim|synthetic] [rateHz] [bindAddress] [port]
set -euo pipefail
MODE="${1:-demo}"; RATE="${2:-10}"; BIND="${3:-0.0.0.0}"; PORT="${4:-5080}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ ! -f artifacts/dashboard/wwwroot/index.html ]; then
  echo "== publishing dashboard (Blazor WASM) =="; dotnet publish src/DigitalTwin.Dashboard -c Release -o artifacts/dashboard
fi
echo "== building backend =="; dotnet build src/DigitalTwin.Backend -c Release --nologo -v q
cd src/DigitalTwin.Backend/bin/Release/net10.0
echo "== dashboard: http://localhost:$PORT/  mode=$MODE rate=$RATE Hz =="
exec dotnet DigitalTwin.Backend.dll --Urls="http://$BIND:$PORT" --Backend:Replay:Source="$MODE" --Backend:Replay:RateHz="$RATE"
