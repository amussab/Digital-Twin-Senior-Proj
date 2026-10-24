<#
.SYNOPSIS  UP2 demo: backend with the v3 model contract + replay source 'up2' (AI-engine/models/up2_demo_payloads.bin).
           Stream is [SYNTHETIC] synth_v3: healthy -> fault -> failure. Dashboard header shows "[SYNTHETIC / SIMULATED SOURCE]".
.EXAMPLE   ./scripts/run-up2-demo.ps1
.EXAMPLE   ./scripts/run-up2-demo.ps1 -Contract AI-engine/models/model_contract_v3b.json -RateHz 20
#>
param(
  [string]$Contract = 'AI-engine/models/model_contract_v3.json',
  [double]$RateHz = 10,
  [string]$Bind = '0.0.0.0',
  [int]$Port = 5080,
  [switch]$Republish
)
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root
if ($Republish -or -not (Test-Path 'artifacts/dashboard/wwwroot/index.html')) {
  Write-Host '== publishing dashboard (Blazor WASM, ~2 min first time) =='
  dotnet publish src/DigitalTwin.Dashboard -c Release -o artifacts/dashboard
}
Write-Host '== building backend =='
dotnet build src/DigitalTwin.Backend -c Release --nologo -v q
$out = Join-Path $root 'src/DigitalTwin.Backend/bin/Release/net10.0'
Set-Location $out
$lan = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -match '^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)' } | Select-Object -First 1).IPAddress
Write-Host "== dashboard: http://localhost:$Port/ (LAN: http://${lan}:$Port/)  contract=$Contract source=up2 rate=$RateHz Hz [SYNTHETIC] =="
dotnet DigitalTwin.Backend.dll --Urls="http://${Bind}:$Port" --Backend:ModelContractPath=$Contract `
  --Backend:Replay:Enabled=true --Backend:Replay:Source=up2 --Backend:Replay:RateHz=$RateHz --Backend:PhysicsTwin=None
