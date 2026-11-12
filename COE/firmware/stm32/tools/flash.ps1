<#
.SYNOPSIS
Build one firmware mode, flash both cores through the ST-LINK and reset.

.DESCRIPTION
Switch the 24 V supply (ADC power) off before flashing: during reset the
STM32 pins float until the firmware holds them low again.

.EXAMPLE
.\tools\flash.ps1 -Mode REPLAY
.\tools\flash.ps1 -Mode LIVE_RAW -NoBuild
#>
param(
    [ValidateSet('REPLAY', 'LIVE_RAW', 'LIVE')][string]$Mode,
    [switch]$NoBuild
)

. "$PSScriptRoot\common.ps1"
if (-not $Mode) { $Mode = Get-SelectedMode }
if (-not $NoBuild) { & "$PSScriptRoot\build.ps1" -Mode $Mode }

foreach ($elf in @((Get-Cm4Elf), (Get-Cm7Elf $Mode))) {
    if (-not (Test-Path $elf)) { throw "Missing $elf; run tools\build.ps1 first." }
    Write-Host "Flashing $elf"
    $output = & $ProgrammerCli -c port=SWD mode=UR reset=HWrst -d $elf -v 2>&1
    if ($LASTEXITCODE -ne 0 -or -not ($output -match 'Download verified successfully')) {
        $output | Write-Host
        throw 'Flashing failed (is the board connected and no debugger session open?)'
    }
}
& $ProgrammerCli -c port=SWD mode=UR -hardRst | Out-Null
Write-Host "Flashed $Mode on both cores and reset the board."
