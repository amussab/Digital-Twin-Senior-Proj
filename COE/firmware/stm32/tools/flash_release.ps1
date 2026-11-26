<#
.SYNOPSIS
Flash the prebuilt ELFs in release\rig_runtime_config (CM4 + CM7 LIVE) through the ST-LINK. No build.

.DESCRIPTION
Switch the 24 V supply (ADC power) off first: during reset the STM32 pins float until the
firmware holds them low again. Same programmer commands as flash.ps1.
Afterwards: switch 24 V on, wait for the green LED, then  python training\node_config.py --query

.EXAMPLE
.\tools\flash_release.ps1
#>
param([string]$Folder = (Join-Path (Split-Path -Parent $PSScriptRoot) 'release\rig_runtime_config'))

. "$PSScriptRoot\common.ps1"
Write-Host 'REMINDER: 24 V supply ON + ST-LINK USB (verified 14 Nov: with 24 V off the target reads 0.02 V and is not found).' -ForegroundColor Yellow

# Check the files against the hash list first, so a damaged copy is never flashed.
$sums = Join-Path $Folder 'SHA256SUMS.txt'
if (-not (Test-Path $sums)) { throw "Missing $sums" }
foreach ($line in Get-Content $sums) {
    if ($line -notmatch '^([0-9a-fA-F]{64})\s+\*?(.+)$') { continue }
    $file = Join-Path $Folder $Matches[2].Trim()
    if (-not (Test-Path $file)) { throw "Missing $file" }
    if ((Get-FileHash $file -Algorithm SHA256).Hash -ne $Matches[1].ToUpper()) { throw "Hash mismatch: $file" }
}

foreach ($name in @('coe_firmware_CM4.elf', 'coe_firmware_CM7_LIVE.elf')) {
    $elf = Join-Path $Folder $name
    if (-not (Test-Path $elf)) { throw "Missing $elf" }
    Write-Host "Flashing $elf"
    $output = & $ProgrammerCli -c port=SWD mode=UR reset=HWrst -d $elf -v 2>&1
    if ($LASTEXITCODE -ne 0 -or -not ($output -match 'Download verified successfully')) {
        $output | Write-Host
        throw 'Flashing failed (is the board connected and no debugger session open?)'
    }
}
& $ProgrammerCli -c port=SWD mode=UR -hardRst | Out-Null
Write-Host 'Flashed LIVE (1800 RPM default, runtime RPM command) on both cores and reset the board.'
Write-Host 'Now: switch 24 V ON, wait for the green LED, then: python training\node_config.py --query' -ForegroundColor Yellow
