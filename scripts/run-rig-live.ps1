<#
.SYNOPSIS  One command for the rig day: backend + dashboard (dotnet) + COE inference receiver (Python NodeEngine).
           Real node -> Wi-Fi COE-NODE -> http://<laptop>:8000/features -> NodeEngine -> POST /api/live/prediction -> dashboard.
.EXAMPLE   ./scripts/run-rig-live.ps1
.EXAMPLE   ./scripts/run-rig-live.ps1 -Contract v4_real -Baseline logs/baseline_1800.json
Stop with ./scripts/stop-rig-live.ps1. Logs: logs/backend.log, logs/receiver.log, logs/live_<time>.jsonl (every window + prediction).
#>
param(
  [string]$Contract = 'v3e_node',          # NODE_CONTRACT (v4_real after tomorrow's training)
  [string]$Baseline = '',                  # NODE_BASELINE json (loaded if present, saved when captured)
  [int]$BackendPort = 5080,
  [int]$NodePort = 8000,
  [string]$SettleS = '',                   # NODE_SETTLE_S override (default 120)
  [string]$BaselineS = '',                 # NODE_BASELINE_S override (default 60)
  [switch]$NoBrowser,
  [switch]$Republish
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
$logs = Join-Path $root 'logs'; New-Item -ItemType Directory -Force $logs | Out-Null
$py = 'C:\Users\lenovo\Documents\KFUPM\Term261\ICS414\Repo\DT-up2\AI-engine\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
if (Test-Path (Join-Path $logs 'rig.pids')) { Write-Host 'Already running? Run scripts/stop-rig-live.ps1 first.'; exit 1 }

if ($Republish -or -not (Test-Path 'artifacts/dashboard/wwwroot/index.html')) {
  Write-Host '== publishing dashboard (Blazor WASM, a few minutes) =='
  dotnet publish src/DigitalTwin.Dashboard -c Release -o artifacts/dashboard --nologo -v q
}
Write-Host '== building backend =='
dotnet build src/DigitalTwin.Backend -c Release --nologo -v q
$out = Join-Path $root 'src/DigitalTwin.Backend/bin/Release/net10.0'

# Backend: LAN-only (C5/I1 guard stays on), no replay, no physics twin (node sends features only), dashboard served from the same URL.
$dash = Join-Path $root 'artifacts/dashboard/wwwroot'
$be = Start-Process dotnet -PassThru -WindowStyle Minimized -WorkingDirectory $out `
  -ArgumentList @('DigitalTwin.Backend.dll', "--Urls=http://0.0.0.0:$BackendPort", '--Backend:Replay:Enabled=false',
                  '--Backend:PhysicsTwin=None', "--Backend:DashboardPath=$dash", '--Backend:Replay:Source=node-live') `
  -RedirectStandardOutput (Join-Path $logs 'backend.log') -RedirectStandardError (Join-Path $logs 'backend.err.log')

# Receiver
$env:NODE_CONTRACT = $Contract
if ($Baseline) { $env:NODE_BASELINE = (Resolve-Path -LiteralPath $Baseline -ErrorAction SilentlyContinue) ; if (-not $env:NODE_BASELINE) { $env:NODE_BASELINE = (Join-Path $root $Baseline) } }
if ($SettleS) { $env:NODE_SETTLE_S = $SettleS }
if ($BaselineS) { $env:NODE_BASELINE_S = $BaselineS }
$env:BACKEND_URL = "http://localhost:$BackendPort"
$env:RIG_LOG_DIR = $logs
$env:PYTHONUNBUFFERED = '1'
$rx = Start-Process $py -PassThru -WindowStyle Minimized -WorkingDirectory (Join-Path $root 'COE/inference') `
  -ArgumentList @('inference_receiver.py', '--port', $NodePort) `
  -RedirectStandardOutput (Join-Path $logs 'receiver.log') -RedirectStandardError (Join-Path $logs 'receiver.err.log')
"$($be.Id)`n$($rx.Id)" | Set-Content (Join-Path $logs 'rig.pids')

# wait for backend
$ok = $false
for ($i = 0; $i -lt 60 -and -not $ok; $i++) { Start-Sleep -Milliseconds 500; try { $ok = (Invoke-RestMethod "http://localhost:$BackendPort/api/health" -TimeoutSec 2).status -eq 'ok' } catch {} }
if (-not $ok) { Write-Warning 'backend did not answer; see logs/backend.log, logs/backend.err.log' }
$lan = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -match '^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)' } | Select-Object -First 1).IPAddress
Write-Host "Dashboard : http://localhost:$BackendPort/   (LAN: http://${lan}:$BackendPort/)"
Write-Host "Node POST : http://<this laptop on COE-NODE>:$NodePort/features   contract=$Contract"
Write-Host "Logs      : $logs  (backend.log, receiver.log, live_*.jsonl = every window + prediction)"
Write-Host "Latency   : http://localhost:$BackendPort/api/metrics  (live_* stages)"
Write-Host 'Stop      : ./scripts/stop-rig-live.ps1'
if (-not $NoBrowser) { Start-Process "http://localhost:$BackendPort/" }
