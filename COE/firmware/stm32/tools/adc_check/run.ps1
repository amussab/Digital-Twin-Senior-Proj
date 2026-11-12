<#
.SYNOPSIS
Build the standalone AD7606 check, flash it to the CM7 and show its output.

.DESCRIPTION
Replaces the CM7 firmware until you flash the project again, for example:
  powershell -ExecutionPolicy Bypass -File .\tools\flash.ps1 -Mode LIVE_RAW
The CM4 image is left as it is.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\tools\adc_check\run.ps1
#>
param([switch]$NoMonitor)

. "$PSScriptRoot\..\common.ps1"
$out = Join-Path $CliBuild 'adc_check'
New-Item -ItemType Directory -Force $out | Out-Null
$elf = Join-Path $out 'adc_check.elf'

$flags = @('-mcpu=cortex-m7', '-mfpu=fpv5-d16', '-mfloat-abi=hard', '-mthumb', '-O2', '-g',
    '-Wall', '-Wextra', '-DSTM32H755xx', '-DCORE_CM7', '-DUSE_PWR_DIRECT_SMPS_SUPPLY',
    '-DHSE_VALUE=8000000UL',
    "-I$ProjectRoot\Drivers\CMSIS\Device\ST\STM32H7xx\Include",
    "-I$ProjectRoot\Drivers\CMSIS\Include")
$sources = @("$PSScriptRoot\adc_check.c",
    "$ProjectRoot\Common\Src\system_stm32h7xx_dualcore_boot_cm4_cm7.c",
    "$ProjectRoot\CM7\Core\Startup\startup_stm32h755xx_CM7.s")
$ErrorActionPreference = 'Continue'
& (Join-Path $GccBin 'arm-none-eabi-gcc.exe') @flags @sources `
    "-T$ProjectRoot\CM7\stm32h755xx_flash_CM7.ld" '--specs=nano.specs' '--specs=nosys.specs' `
    '-Wl,--gc-sections' '-Wl,--print-memory-usage' -o $elf 2>&1 | ForEach-Object { "$_" } |
    Where-Object { $_ -match 'warning|error|FLASH:' -and $_ -notmatch 'is not implemented and will always fail' } |
    Write-Host
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$ErrorActionPreference = 'Stop'

$result = & $ProgrammerCli -c port=SWD mode=UR reset=HWrst -d $elf -v -hardRst 2>&1
if (-not ($result -match 'Download verified successfully')) { $result | Write-Host; throw 'Flashing failed' }
Write-Host 'ADC check flashed and running.'
if (-not $NoMonitor) { & "$PSScriptRoot\..\com.ps1" }
