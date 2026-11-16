<#
C1 live check (ME), one command, run on the COE laptop (the one that builds and flashes the node):
  1. switches the node to raw mode (LIVE_RAW),
  2. saves raw samples every 2 s and shows each housing's velocity in mm/s RMS, live,
  3. when you press a key, switches the node back to its normal LIVE firmware.
Easiest: double-click C1_check.bat in this folder.
#>
param([int]$Minutes = 20)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$stm = Join-Path $repo 'COE\firmware\stm32'
$ring = Join-Path $stm 'build\logs\ring'
$py = if (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { 'py' }

& $py -c "import numpy" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host 'Installing numpy...'; & $py -m pip install numpy }

Write-Host "`n[1/3] Switching the node to raw mode (LIVE_RAW). Takes a minute..." -ForegroundColor Cyan
& powershell -ExecutionPolicy Bypass -File (Join-Path $stm 'tools\flash.ps1') -Mode LIVE_RAW
if ($LASTEXITCODE -ne 0) { throw 'Could not flash LIVE_RAW. Ask COE (Abdullah).' }

New-Item -ItemType Directory -Force $ring | Out-Null
Remove-Item (Join-Path $ring 'snap_*.bin') -ErrorAction SilentlyContinue
$saver = Start-Process powershell -PassThru -WindowStyle Minimized -ArgumentList @(
    '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $stm 'tools\ring_snapshots.ps1'), '-Minutes', $Minutes)
try {
    Write-Host "`n[2/3] Live velocity. Run the rig at 1750 rpm, then 3500 rpm. Press any key when done.`n" -ForegroundColor Cyan
    & $py (Join-Path $PSScriptRoot 'c1_velocity.py') $ring
}
finally {
    if ($saver -and -not $saver.HasExited) { Stop-Process -Id $saver.Id -Force }
    Write-Host "`n[3/3] Switching the node back to normal (LIVE)..." -ForegroundColor Cyan
    & powershell -ExecutionPolicy Bypass -File (Join-Path $stm 'tools\flash.ps1') -Mode LIVE
    Write-Host 'Done. The node is back in normal mode.' -ForegroundColor Green
}
