<#
.SYNOPSIS  Backend (v3b contract, replay OFF, UDP :5005) + COE simulator sending a fault scenario over UDP, so the dashboard
           shows live data produced by the COE pipeline (raw samples -> BPF -> envelope -> FFT -> 152 B payload).
           Hardware day: stop the simulator, power the node; it sends the same 152-byte datagrams to the same port. Nothing else changes.
.EXAMPLE   ./scripts/run-up2-coe-demo.ps1
#>
param(
  [string]$Contract = 'AI-engine/models/model_contract_v3b.json',
  [double]$Rate = 2,
  [string]$Bind = '0.0.0.0',
  [int]$Port = 5080,
  [int]$UdpPort = 5005,
  [switch]$Republish
)
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root
if ($Republish -or -not (Test-Path 'artifacts/dashboard/wwwroot/index.html')) { dotnet publish src/DigitalTwin.Dashboard -c Release -o artifacts/dashboard }
dotnet build src/DigitalTwin.Backend -c Release --nologo -v q
$out = Join-Path $root 'src/DigitalTwin.Backend/bin/Release/net10.0'
$py = Join-Path $root 'AI-engine/.venv/Scripts/python.exe'; if (-not (Test-Path $py)) { $py = 'python' }
$be = Start-Process dotnet -WorkingDirectory $out -PassThru -ArgumentList @('DigitalTwin.Backend.dll', "--Urls=http://${Bind}:$Port",
  "--Backend:ModelContractPath=$Contract", '--Backend:Replay:Enabled=false', "--Backend:Udp:Port=$UdpPort", '--Backend:PhysicsTwin=None', '--Backend:MachineId=OH2-PUMP-01 [SYNTHETIC / SIMULATED SOURCE: COE sim]')
try {
  Start-Sleep -Seconds 6
  Write-Host "== dashboard: http://localhost:$Port/   COE simulator -> udp 127.0.0.1:$UdpPort at $Rate windows/s [SIMULATION] =="
  Set-Location (Join-Path $root 'AI-engine')
  & $py (Join-Path $root 'scripts/coe_fault_scenario.py') --udp "127.0.0.1:$UdpPort" --rate $Rate
} finally { if ($be -and -not $be.HasExited) { Stop-Process -Id $be.Id } }
