<#
.SYNOPSIS
Rig-day preflight: checks everything needed to flash, record, train and stream. Report only:
it never flashes, never talks to the node, never changes settings.

.EXAMPLE
./scripts/preflight.ps1            # full check (~2 min, runs the AI tests)
./scripts/preflight.ps1 -Quick     # skip the test suites
./scripts/preflight.ps1 -Rehearse  # also replays real rig data through the live chain for ~40 s
#>
param([switch]$Quick, [switch]$Rehearse)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
$Py = 'C:\Users\lenovo\Documents\KFUPM\Term261\ICS414\Repo\DT-up2\AI-engine\.venv\Scripts\python.exe'
$DataDir = Join-Path (Split-Path -Parent $Root) 'rig_data'
$script:fails = 0; $script:warns = 0

function Ok([string]$m)   { Write-Host "  [ OK ] $m" -ForegroundColor Green }
function Bad([string]$m)  { Write-Host "  [FAIL] $m" -ForegroundColor Red; $script:fails++ }
function Warn([string]$m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow; $script:warns++ }
function Section([string]$m) { Write-Host "`n== $m" -ForegroundColor Cyan }

Set-Location $Root

Section 'Repository'
$branch = git rev-parse --abbrev-ref HEAD
if ($branch -eq 'fix/rig-new-bearing') { Ok "branch $branch @ $(git rev-parse --short HEAD)" } else { Warn "branch is '$branch' (expected fix/rig-new-bearing)" }
$dirty = git status --porcelain | Where-Object { $_ -notmatch '\.settings/|\.vscode/' }
if ($dirty) { Warn "uncommitted changes: $($dirty.Count) file(s)" } else { Ok 'working tree clean (VS Code settings ignored)' }

Section 'Firmware flash files + ST tools'
$rel = 'COE/firmware/stm32/release/rig_runtime_config'
if (Test-Path "$rel/SHA256SUMS.txt") {
    $okAll = $true
    foreach ($line in Get-Content "$rel/SHA256SUMS.txt") {
        if ($line -match '^([0-9a-f]{64})\s+\*?(.+)$') {
            $f = Join-Path $rel $Matches[2].Trim()
            if (-not (Test-Path $f)) { Bad "missing $f"; $okAll = $false; continue }
            $h = (Get-FileHash $f -Algorithm SHA256).Hash.ToLower()
            if ($h -ne $Matches[1]) { Bad "hash mismatch $f"; $okAll = $false }
        }
    }
    if ($okAll) { Ok "release ELFs present, SHA-256 verified ($rel)" }
} else { Bad "no release folder $rel" }
$b = Join-Path $env:LOCALAPPDATA 'stm32cube\bundles'
foreach ($n in 'gnu-tools-for-stm32', 'cmake', 'ninja', 'programmer') {
    if (Test-Path (Join-Path $b $n)) { Ok "ST bundle $n" } else { Bad "ST bundle $n missing (VS Code > Ctrl+Shift+P > Open Project STM32Cube Bundles Manager)" }
}
$st = Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPDeviceID -match 'VID_0483' -and $_.Name -match 'COM\d+' } | Select-Object -First 1
if ($st) { Ok "ST-LINK detected: $($st.Name)" } else { Warn 'ST-LINK USB not detected (plug the Nucleo ST-LINK USB before flashing / setting RPM)' }

Section 'Python / AI'
if (Test-Path $Py) { Ok "AI python $Py" } else { Bad "AI venv python missing: $Py" }
& $Py -c "import numpy, onnxruntime, xgboost, sklearn; print('imports ok')" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Ok 'numpy / onnxruntime / xgboost / sklearn import' } else { Bad 'AI python packages failed to import' }
foreach ($m in 'model_contract_v3e_node.json', 'v3e_node_cls.onnx') {
    if (Test-Path "AI-engine/models/$m") { Ok "model AI-engine/models/$m" } else { Bad "missing AI-engine/models/$m" }
}
Push-Location AI-engine
& $Py -c "from aiengine.node_adapter import NodeEngine; NodeEngine(); print('ok')" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Ok 'NodeEngine loads (default contract)' } else { Bad 'NodeEngine failed to load' }
Pop-Location
python COE/training/node_config.py --selftest 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Ok 'node_config.py selftest' } else { Bad 'node_config.py selftest failed (system python)' }
python --version 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Ok "system python for the recorder: $(python --version 2>&1)" } else { Bad 'system python not on PATH (recorder needs it)' }

Section 'Network / ports / firewall'
foreach ($p in 8000, 5080) {
    $l = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if ($l) { Bad "port $p in use by PID $($l[0].OwningProcess) (run scripts/stop-rig-live.ps1 or close the recorder)" } else { Ok "port $p free" }
}
$fw = netsh advfirewall firewall show rule name="COE node TCP 8000" 2>$null
if ($fw -match 'Enabled:\s+Yes') { Ok 'firewall rule "COE node TCP 8000"' } else { Warn 'no firewall rule for TCP 8000; admin terminal: netsh advfirewall firewall add rule name="COE node TCP 8000" dir=in action=allow protocol=TCP localport=8000 profile=any' }
$vpn = Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq 'Up' -and $_.InterfaceDescription -match 'VPN|TAP|WireGuard|Wintun|Cisco|Fortinet|Pulse|OpenVPN' }
if ($vpn) { Warn "VPN adapter up: $($vpn.Name) (it hid COE-NODE last time; disconnect it)" } else { Ok 'no VPN adapter up' }
$wifi = (netsh wlan show interfaces) -match '^\s+SSID\s+:' | Select-Object -First 1
if ($wifi -match 'COE-NODE') { Ok 'Wi-Fi on COE-NODE' } else { Warn "Wi-Fi not on COE-NODE yet ($($wifi -replace '\s+',' '))" }

Section 'Backend / dashboard'
dotnet --version 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Ok "dotnet $(dotnet --version)" } else { Bad 'dotnet missing' }
if (Test-Path 'artifacts/dashboard/wwwroot/index.html') { Ok 'dashboard published (artifacts/dashboard)' } else { Warn 'dashboard not published yet; first run-rig-live.ps1 takes ~3 min' }

Section 'Storage'
New-Item -ItemType Directory -Force $DataDir | Out-Null
$free = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
if ($free -gt 5) { Ok "C: free $free GB; rig data dir $DataDir" } else { Warn "C: only $free GB free" }

if (-not $Quick) {
    Section 'Test suites'
    Push-Location AI-engine
    $r = & $Py -m pytest tests -q 2>&1 | Select-Object -Last 1
    Pop-Location
    if ($r -match 'passed' -and $r -notmatch 'failed|error') { Ok "AI-engine: $r" } else { Bad "AI-engine: $r" }
    $r = & $Py -m pytest COE/training/tests COE/firmware/stm32/tools/tests -q 2>&1 | Select-Object -Last 1
    if ($r -match 'passed' -and $r -notmatch 'failed|error') { Ok "COE host tests: $r" } else { Bad "COE host tests: $r" }
    $r = dotnet test tests/DigitalTwin.Backend.Tests -c Release --nologo -v q 2>&1 | Select-String 'Passed!|Failed!' | Select-Object -Last 1
    if ("$r" -match 'Passed!') { Ok "backend: $r" } else { Bad "backend: $r" }
}

if ($Rehearse) {
    Section 'Rehearsal: replay real rig data through the live chain (~40 s)'
    & "$PSScriptRoot/run-rig-live.ps1" -NoBrowser -SettleS 5 -BaselineS 5 | Out-Null
    Start-Sleep -Seconds 8
    $src = Join-Path $DataDir 'probe_1800rpm_fw250_healthy.jsonl'
    & $Py COE/inference/replay_node.py $src --rpm-override 1800 --speed 4 --limit 120 2>&1 | Select-Object -Last 2 | ForEach-Object { Write-Host "    $_" }
    try {
        $live = Invoke-RestMethod -Uri 'http://127.0.0.1:5080/api/live/latest' -TimeoutSec 5
        if ($live) { Ok 'backend received live predictions (/api/live/latest)' } else { Bad 'backend has no live prediction' }
    } catch { Bad "backend /api/live/latest: $($_.Exception.Message)" }
    & "$PSScriptRoot/stop-rig-live.ps1" | Out-Null
    Ok 'live chain stopped'
}

Write-Host ''
if ($script:fails -eq 0) { Write-Host "PREFLIGHT PASSED ($script:warns warning(s))" -ForegroundColor Green }
else { Write-Host "PREFLIGHT FAILED: $script:fails failure(s), $script:warns warning(s)" -ForegroundColor Red }
exit $script:fails
