<# Stops everything started by run-rig-live.ps1 and frees ports 8000 / 5080. #>
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pidFile = Join-Path $root 'logs/rig.pids'
if (Test-Path $pidFile) {
  foreach ($p in Get-Content $pidFile) { if ($p) { Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue } }
  Remove-Item $pidFile -Force
}
foreach ($port in 8000, 5080) {
  Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}
Write-Host 'Stopped (ports 8000 and 5080 freed).'
