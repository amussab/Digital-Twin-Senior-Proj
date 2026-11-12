<#
.SYNOPSIS
Open the ESP-IDF terminal for this project.
.DESCRIPTION
Use -OpenEditor to open the same project in a separate VS Code window.
#>
param([switch]$OpenEditor)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot
if ($OpenEditor) {
    & code.cmd --new-window $PSScriptRoot
} else {
    & 'C:\Espressif\eim.exe' --esp-idf-json-path 'C:\Users\USER2\.espressif\tools' shell v6.1
}
