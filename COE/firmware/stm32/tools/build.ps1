<#
.SYNOPSIS
Build both cores for one firmware mode from the command line, into
%LOCALAPPDATA%\coe_firmware\cli-build-<id> (outside the project; one folder per project location).

.EXAMPLE
.\tools\build.ps1                 # mode from CM7/firmware_mode.cmake
.\tools\build.ps1 -Mode LIVE_RAW
#>
param([ValidateSet('REPLAY', 'LIVE_RAW', 'LIVE')][string]$Mode)

. "$PSScriptRoot\common.ps1"
if (-not $Mode) { $Mode = Get-SelectedMode }
$toolchain = Join-Path $ProjectRoot 'gcc-arm-none-eabi.cmake'

function Invoke-Build([string]$Source, [string]$Binary, [string[]]$Extra) {
    # CMake and the compiler report progress on stderr; that is not an error here.
    $ErrorActionPreference = 'Continue'
    $output = & cmake -S $Source -B $Binary -G Ninja --no-warn-unused-cli `
        "-DCMAKE_TOOLCHAIN_FILE=$toolchain" -DCMAKE_BUILD_TYPE=Debug @Extra 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { $output | Write-Host; throw "CMake configure failed for $Source" }
    $output = & cmake --build $Binary 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { $output | Write-Host; throw "Build failed for $Source" }
    $output | Where-Object { $_ -match 'warning:|FLASH:|RAM:|RAM_D1:' } | Write-Host
}

Write-Host "== CM4 (idle core)"
Invoke-Build (Join-Path $ProjectRoot 'CM4') (Join-Path $CliBuild 'CM4') @()
Write-Host "== CM7, mode $Mode"
Invoke-Build (Join-Path $ProjectRoot 'CM7') (Join-Path $CliBuild "CM7_$Mode") @("-DCOE_FIRMWARE_MODE_OVERRIDE=$Mode")
Write-Host "Built: $(Get-Cm7Elf $Mode)"
