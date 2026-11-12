<#
.SYNOPSIS
Show the STM32 messages from the ST-LINK virtual COM port (115200 8N1) and
save them to a log file. Stop with Ctrl+C, or after -Seconds.

.EXAMPLE
.\tools\com.ps1                      # until Ctrl+C, log in build\logs
.\tools\com.ps1 -Seconds 60 -Log run1.txt
#>
param([string]$Port, [int]$Seconds = 0, [string]$Log)

. "$PSScriptRoot\common.ps1"
if (-not $Port) { $Port = Find-StlinkPort }
if (-not $Log) {
    New-Item -ItemType Directory -Force $LogDir | Out-Null
    $Log = Join-Path $LogDir ("com_{0:yyyyMMdd_HHmmss}.txt" -f (Get-Date))
}

$serial = New-Object System.IO.Ports.SerialPort $Port, 115200, 'None', 8, 'One'
$serial.Open()
$writer = [System.IO.StreamWriter]::new($Log, $true)
Write-Host "Reading $Port; saving to $Log. Ctrl+C to stop."
$end = if ($Seconds -gt 0) { (Get-Date).AddSeconds($Seconds) } else { [DateTime]::MaxValue }
try {
    while ((Get-Date) -lt $end) {
        $text = $serial.ReadExisting()
        if ($text) {
            Write-Host -NoNewline $text
            $writer.Write($text)
            $writer.Flush()
        }
        Start-Sleep -Milliseconds 100
    }
} finally {
    $writer.Close()
    $serial.Close()
}
