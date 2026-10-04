<#
.SYNOPSIS  One command: publish the dashboard (once), build the backend, run it. One process, one LAN URL.
.EXAMPLE   ./scripts/run-demo.ps1                    # replay of XJTU-SY public data (MEASURED), real ONNX models
.EXAMPLE   ./scripts/run-demo.ps1 -Mode twin-sim     # [SIMULATION] twin demo: physics RUL + residual populated
.EXAMPLE   ./scripts/run-demo.ps1 -Bind 192.168.1.50 # bind one LAN interface only (C5)
#>
param(
  [ValidateSet('demo','twin-sim','synthetic')][string]$Mode = 'demo',
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
Set-Location $out    # appsettings.json is read from the working directory; repo paths resolve by walking up
$lan = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -match '^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)' } | Select-Object -First 1).IPAddress
Write-Host "== dashboard:  http://localhost:$Port/   (LAN: http://${lan}:$Port/)  mode=$Mode rate=$RateHz Hz =="
dotnet DigitalTwin.Backend.dll --Urls="http://${Bind}:$Port" --Backend:Replay:Source=$Mode --Backend:Replay:RateHz=$RateHz
