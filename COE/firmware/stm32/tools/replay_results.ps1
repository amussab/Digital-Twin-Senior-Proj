<#
.SYNOPSIS
Read the REPLAY self-test results from the running board (no debugger needed)
and print PASS/FAIL for each one.

.DESCRIPTION
Flash REPLAY first (tools\flash.ps1 -Mode REPLAY) and wait until the green LED
blinks: the checks take a few seconds after reset. The board keeps running;
the values are read through the ST-LINK without stopping it.

.EXAMPLE
.\tools\replay_results.ps1
#>
param([int]$WaitSeconds = 8)

. "$PSScriptRoot\common.ps1"
$elf = Get-Cm7Elf 'REPLAY'
if (-not (Test-Path $elf)) { throw "Missing $elf; run tools\flash.ps1 -Mode REPLAY first." }

# Name, type, rule, description. Rules: true, =N, <=N, or 'info'.
$checks = @(
    @('SystemCoreClock', 'u32', '=400000000', 'Cortex-M7 clock'),
    @('processing_clock_source', 'u32', '=1', '1 = HSE 8 MHz from ST-LINK, 2 = HSI fallback'),
    @('processing_clock_measured_hz', 'u32', 'info', 'core clock measured against HSI (about 1%)'),
    @('demo_checks_passed', 'bool', 'true', 'time features'),
    @('demo_checked_channel_windows', 'u32', '=48', ''),
    @('demo_failed_check', 'u32', '=0', ''),
    @('bandpass_checks_passed', 'bool', 'true', 'band-pass filter'),
    @('bandpass_continuity_checks_passed', 'bool', 'true', ''),
    @('bandpass_checked_channel_runs', 'u32', '=16', ''),
    @('bandpass_failed_check', 'u32', '=0', ''),
    @('envelope_checks_passed', 'bool', 'true', 'envelope and decimation'),
    @('envelope_checked_channel_runs', 'u32', '=16', ''),
    @('envelope_checked_windows', 'u32', '=4', ''),
    @('envelope_failed_check', 'u32', '=0', ''),
    @('fft_checks_passed', 'bool', 'true', 'envelope FFT, all bins'),
    @('fft_checked_records', 'u32', '=9', ''),
    @('fft_failed_check', 'u32', '=0', ''),
    @('frequency_checks_passed', 'bool', 'true', 'bearing-band features'),
    @('frequency_checked_records', 'u32', '=18', ''),
    @('frequency_failed_check', 'u32', '=0', ''),
    @('payload_checks_passed', 'bool', 'true', '160-byte packet bytes'),
    @('payload_failed_check', 'u32', '=0', ''),
    @('pipeline_checks_passed', 'bool', 'true', 'streaming pipeline'),
    @('pipeline_checked_windows', 'u32', '=12', ''),
    @('pipeline_failed_check', 'u32', '=0', ''),
    @('pipeline_batch_checks_passed', 'bool', 'true', 'complete-window pipeline'),
    @('pipeline_batch_checked_windows', 'u32', '=12', ''),
    @('pipeline_batch_failed_check', 'u32', '=0', ''),
    @('pipeline_batch_fixed_shape_passed', 'bool', 'true', '32 features at both speeds'),
    @('pipeline_batch_deadline_met', 'bool', 'true', 'both window times <= 333 ms'),
    @('pipeline_batch_max_ms_1750', 'f32', '<=333', 'ms per window at 1750 RPM (saved: 91.03)'),
    @('pipeline_batch_max_ms_3600', 'f32', '<=333', 'ms per window at 3600 RPM (saved: 47.87)'),
    @('pipeline_max_block_time_us', 'f32', 'info', 'longest single block call, us'),
    @('communication_checks_passed', 'bool', 'true', 'packet decoder cases'),
    @('communication_checked_cases', 'u32', '=22', ''),
    @('communication_failed_check', 'u32', '=0', ''),
    @('feature_uart_ready', 'bool', 'true', 'USART2 to ESP32 running'),
    @('feature_uart_packets_sent', 'u32', 'info', '12 after start-up (more after B1)'),
    @('feature_uart_errors', 'u32', '=0', '')
)

# Symbol addresses from the ELF that was flashed.
$addresses = @{}
& (Join-Path $GccBin 'arm-none-eabi-nm.exe') $elf | ForEach-Object {
    $parts = $_ -split '\s+'
    if ($parts.Count -eq 3) { $addresses[$parts[2]] = $parts[0] }
}

if ($WaitSeconds -gt 0) {
    Write-Host "Waiting $WaitSeconds s for the start-up checks to finish..."
    Start-Sleep -Seconds $WaitSeconds
}

# One ST-LINK connection reads every value while the board keeps running.
$arguments = @('-c', 'port=SWD', 'mode=HOTPLUG')
foreach ($check in $checks) {
    if (-not $addresses.ContainsKey($check[0])) { throw "Symbol $($check[0]) not in $elf" }
    $arguments += @('-r8', "0x$($addresses[$check[0]])", '4')
}
$bytes = @{}
& $ProgrammerCli @arguments 2>&1 | ForEach-Object {
    if ($_ -match '^0x([0-9A-Fa-f]{8}) : ((?:[0-9A-Fa-f]{2} ?)+)') {
        $bytes[$Matches[1].ToLower()] = [byte[]]($Matches[2].Trim() -split ' ' | ForEach-Object { [Convert]::ToByte($_, 16) })
    }
}

$failures = 0
foreach ($check in $checks) {
    $name, $type, $rule, $note = $check
    $raw = $bytes[$addresses[$name].ToLower()]
    if (-not $raw) { throw "Could not read $name (is the board connected?)" }
    $value = switch ($type) {
        'bool' { [bool]$raw[0] }
        'u32' { [BitConverter]::ToUInt32($raw, 0) }
        'f32' { [Math]::Round([BitConverter]::ToSingle($raw, 0), 3) }
    }
    $ok = switch -Regex ($rule) {
        '^true$' { $value -eq $true }
        '^=(.+)$' { $value -eq [double]$Matches[1] }
        '^<=(.+)$' { $value -le [double]$Matches[1] }
        default { $null }
    }
    $status = if ($ok -eq $null) { 'info' } elseif ($ok) { 'PASS' } else { $failures++; 'FAIL' }
    '{0,-4}  {1,-36} {2,-12} {3}' -f $status, $name, $value, $note
}
Write-Host ''
if ($failures -eq 0) { Write-Host 'All replay checks passed.' -ForegroundColor Green }
else { Write-Host "$failures check(s) FAILED. A *_failed_check code names the case in CM7/TestData/demos." -ForegroundColor Red; exit 1 }
