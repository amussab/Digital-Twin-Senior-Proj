# Shared helpers for the tools scripts: find the ST tool bundles installed by
# the VS Code STM32 extension and the project paths. Dot-source this file.

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Bundles = Join-Path $env:LOCALAPPDATA 'stm32cube\bundles'
$Modes = @('REPLAY', 'LIVE_RAW', 'LIVE')

function Find-BundleBin([string]$Name) {
    $dir = Get-ChildItem (Join-Path $Bundles $Name) -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if (-not $dir) { throw "ST bundle '$Name' not found under $Bundles (install the STM32 VS Code extension)." }
    return (Join-Path $dir.FullName 'bin')
}

$GccBin = Find-BundleBin 'gnu-tools-for-stm32'
$CmakeBin = Find-BundleBin 'cmake'
$NinjaBin = Find-BundleBin 'ninja'
$ProgrammerCli = Join-Path (Find-BundleBin 'programmer') 'STM32_Programmer_CLI.exe'
$env:PATH = "$GccBin;$CmakeBin;$NinjaBin;$env:PATH"

# Command-line builds live outside the project, so VS Code's own build folders
# (CM4/build, CM7/build) are untouched and the STM32 extension does not pick
# them up as extra configurations. COM-port logs stay in build/logs (ignored).
# One build folder per project location: CMake refuses a build folder made for another copy.
$rootId = [BitConverter]::ToString([Security.Cryptography.MD5]::Create().ComputeHash(
    [Text.Encoding]::UTF8.GetBytes($ProjectRoot.ToLowerInvariant()))).Replace('-', '').Substring(0, 8)
$CliBuild = Join-Path $env:LOCALAPPDATA "coe_firmware\cli-build-$rootId"
$LogDir = Join-Path $ProjectRoot 'build\logs'

function Get-SelectedMode {
    $line = Select-String -Path (Join-Path $ProjectRoot 'CM7\firmware_mode.cmake') `
        -Pattern '^\s*set\(COE_FIRMWARE_MODE\s+(\w+)\)' | Select-Object -First 1
    return $line.Matches[0].Groups[1].Value
}

function Get-Cm7Elf([string]$Mode) { Join-Path $CliBuild "CM7_$Mode\coe_firmware_CM7.elf" }
function Get-Cm4Elf { Join-Path $CliBuild 'CM4\coe_firmware_CM4.elf' }

# The ST-LINK virtual COM port has USB vendor ID 0483.
function Find-StlinkPort {
    $device = Get-CimInstance Win32_PnPEntity |
        Where-Object { $_.PNPDeviceID -match 'VID_0483' -and $_.Name -match '\((COM\d+)\)' } |
        Select-Object -First 1
    if (-not $device) { throw 'ST-LINK COM port not found: is the STM32 USB cable plugged in?' }
    return ([regex]::Match($device.Name, 'COM\d+')).Value
}
