# Manual resume helper for the UP2 build. Re-launches Claude Code headless on the repo until
# .claude/UP2_STATE.md has no unchecked steps. Runs under your normal permission settings
# (no permission-skipping flags): approve prompts or pre-allow tools in .claude/settings.json.
# Usage:  powershell -File scripts/resume-up2.ps1 [-WaitMinutes 20]
param([int]$WaitMinutes = 20, [int]$MaxRounds = 40)
$repo  = Split-Path -Parent $PSScriptRoot
$state = Join-Path $repo ".claude\UP2_STATE.md"
$prompt = "Resume the UP2 prototype build. Read .claude/UP2_STATE.md, .claude/FACTS.md and .claude/rules/up2.md, then continue from the first unchecked step. Update the ledger after each step."
for ($i = 0; $i -lt $MaxRounds; $i++) {
  if (-not (Select-String -Path $state -Pattern '^- \[ \]' -Quiet)) { Write-Host "All steps done."; break }
  Push-Location $repo
  claude --continue -p $prompt
  Pop-Location
  Write-Host "Round $i ended (limit or completion). Sleeping $WaitMinutes min before resuming..."
  Start-Sleep -Seconds ($WaitMinutes * 60)
}
