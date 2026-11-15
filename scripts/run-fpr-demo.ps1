<#
.SYNOPSIS  FPR demo, native (dotnet + python, no Docker): backend + dashboard, then the rig-anchored simulated trajectories are replayed through
           receiver -> NodeEngine (table time base) -> POST /api/live/prediction -> dashboard (class, stage, AI RUL, physics RUL).
.EXAMPLE   ./scripts/run-fpr-demo.ps1                      # contract v7_node, loops the three demo units
.EXAMPLE   ./scripts/run-fpr-demo.ps1 -Contract v7_node -Speed 8 -Once
Everything streamed is [SIMULATION; rig-anchored replay] and is labelled so on the dashboard. Stop: ./scripts/stop-rig-live.ps1 (same pid file).
Docker equivalent: docker compose --profile fpr-demo up --build backend-fpr fpr-replay
#>
param(
  [string]$Contract = 'v7_node',
  [double]$Speed = 12,
  [int]$BackendPort = 5080,
  [switch]$Once,
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
$dash = Join-Path $root 'artifacts/dashboard/wwwroot'
$be = Start-Process dotnet -PassThru -WindowStyle Minimized -WorkingDirectory $out `
  -ArgumentList @('DigitalTwin.Backend.dll', "--Urls=http://0.0.0.0:$BackendPort", '--Backend:Replay:Enabled=false', '--Backend:PhysicsTwin=None',
                  "--Backend:DashboardPath=$dash", '--Backend:Replay:Source=node-live', '--Backend:MachineId=OH2-PUMP-01 [SIMULATION; rig-anchored replay]') `
  -RedirectStandardOutput (Join-Path $logs 'backend.log') -RedirectStandardError (Join-Path $logs 'backend.err.log')
$ok = $false
for ($i = 0; $i -lt 120 -and -not $ok; $i++) { Start-Sleep -Milliseconds 500; try { $ok = (Invoke-RestMethod "http://127.0.0.1:$BackendPort/api/health" -TimeoutSec 2).status -eq 'ok' } catch {} }
if (-not $ok) { Write-Warning 'backend did not answer; see logs/backend.log' }
$args2 = @('scripts/fpr_replay.py', '--contract', $Contract, '--speed', $Speed, '--backend', "http://127.0.0.1:$BackendPort")
if ($Once) { $args2 += '--once' } else { $args2 += '--loop' }
$env:PYTHONUNBUFFERED = '1'
$rp = Start-Process $py -PassThru -WindowStyle Minimized -WorkingDirectory $root -ArgumentList $args2 `
  -RedirectStandardOutput (Join-Path $logs 'fpr_replay.log') -RedirectStandardError (Join-Path $logs 'fpr_replay.err.log')
"$($be.Id)`n$($rp.Id)" | Set-Content (Join-Path $logs 'rig.pids')
Write-Host "Dashboard : http://localhost:$BackendPort/   contract=$Contract   [SIMULATION; rig-anchored replay]"
Write-Host "Logs      : logs/backend.log, logs/fpr_replay.log;  latency: http://localhost:$BackendPort/api/metrics"
Write-Host 'Stop      : ./scripts/stop-rig-live.ps1'
if (-not $NoBrowser) { Start-Process "http://localhost:$BackendPort/" }
