<#
.SYNOPSIS
While LIVE_RAW runs, save the 64-KB AD7606 sample ring (the last 136 ms of
raw samples, all 8 slots) every 2 seconds, read over the ST-LINK without
stopping the board. Files: build\logs\ring\snap_<clock>_<seconds since B1>s.bin

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\tools\ring_snapshots.ps1 -Minutes 10
#>
param([int]$Minutes = 10, [string]$OutDir)

. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
if (-not $OutDir) { $OutDir = Join-Path $LogDir 'ring' }
$elf = Get-Cm7Elf 'LIVE_RAW'
if (-not (Test-Path $elf)) { throw "Missing $elf; run tools\flash.ps1 -Mode LIVE_RAW first." }
$nm = & (Join-Path $GccBin 'arm-none-eabi-nm.exe') $elf
$ring = (($nm | Select-String ' ring$').ToString() -split ' ')[0]
$status = [Convert]::ToUInt32((($nm | Select-String ' ad7606_status$').ToString() -split ' ')[0], 16)
$framesAddr = '0x{0:X8}' -f ($status + 4) # ad7606_status.frames
New-Item -ItemType Directory -Force $OutDir | Out-Null
Write-Host "Saving ring snapshots to $OutDir for $Minutes min (only while acquisition runs)."
$end = (Get-Date).AddMinutes($Minutes)
$last = 0
while ((Get-Date) -lt $end) {
    $tmp = Join-Path $OutDir 'tmp.bin'
    $out = & $ProgrammerCli -c port=SWD mode=HOTPLUG -r32 $framesAddr 4 -u "0x$ring" 65536 $tmp 2>&1
    $line = $out | Select-String '^0x' | Select-Object -First 1
    if ($line) {
        $frames = [Convert]::ToUInt32(($line.ToString() -split ' : ')[1].Trim(), 16)
        if ($frames -ne $last -and $frames -ne 0) {
            $name = 'snap_{0:HHmmss}_{1:000.0}s.bin' -f (Get-Date), ($frames / 29998.5)
            Move-Item -Force $tmp (Join-Path $OutDir $name)
            Write-Host $name
            $last = $frames
        }
    }
    Start-Sleep -Milliseconds 1800
}
