<#
.SYNOPSIS
Report-only check before flashing: ST bundles, release ELFs + hashes, ST-LINK on USB. Flashes nothing.
#>
param([string]$Folder = (Join-Path (Split-Path -Parent $PSScriptRoot) 'release\rig_runtime_config'))
$bundles = Join-Path $env:LOCALAPPDATA 'stm32cube\bundles'
$fail = 0
function Report([bool]$ok, [string]$text) {
    if ($ok) { Write-Host "[ OK ] $text" -ForegroundColor Green } else { Write-Host "[FAIL] $text" -ForegroundColor Red; $script:fail++ }
}
foreach ($b in 'gnu-tools-for-stm32', 'cmake', 'ninja', 'programmer') {
    Report ([bool](Get-ChildItem (Join-Path $bundles $b) -Directory -ErrorAction SilentlyContinue)) "bundle $b"
}
$sums = Join-Path $Folder 'SHA256SUMS.txt'
if (Test-Path $sums) {
    foreach ($line in Get-Content $sums) {
        if ($line -notmatch '^([0-9a-fA-F]{64})\s+\*?(.+)$') { continue }
        $file = Join-Path $Folder $Matches[2].Trim()
        $ok = (Test-Path $file) -and ((Get-FileHash $file -Algorithm SHA256).Hash -eq $Matches[1].ToUpper())
        Report $ok "ELF hash matches: $($Matches[2].Trim())"
    }
} else { Report $false "SHA256SUMS.txt missing in $Folder" }
$stlink = Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue | Where-Object { $_.PNPDeviceID -match 'VID_0483' }
Report ([bool]$stlink) 'ST-LINK (USB VID_0483) detected'
$com = $stlink | Where-Object { $_.Name -match '\(COM\d+\)' } | Select-Object -First 1
if ($com) { Write-Host "       virtual COM port: $($com.Name)" }
Write-Host 'Reminder: 24 V ON before flashing (board is powered by it). This script flashed nothing.'
if ($fail) { Write-Host "$fail check(s) failed." -ForegroundColor Yellow; exit 1 } else { exit 0 }
