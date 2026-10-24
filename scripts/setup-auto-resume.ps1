# ONE-TIME SETUP, run by the USER (not by Claude):  ! powershell -File scripts/setup-auto-resume.ps1
# 1) writes a narrow, repo-scoped allow-list to .claude/settings.local.json so headless resumes can
#    edit files / run tests / commit in THIS worktree only (no force-push, no push to main, no stash,
#    no hard reset).
# 2) registers a Windows scheduled task that runs scripts/resume-up2.ps1 every 30 min for 36 h.
#    The script exits immediately once .claude/UP2_STATE.md has no unchecked steps.
# Undo:  Unregister-ScheduledTask -TaskName UP2-Auto-Resume -Confirm:$false ; delete .claude/settings.local.json
$repo = Split-Path -Parent $PSScriptRoot
$settings = @'
{
  "permissions": {
    "allow": [
      "Read", "Glob", "Grep", "Edit", "Write", "Agent",
      "Bash(git status:*)", "Bash(git diff:*)", "Bash(git log:*)", "Bash(git add:*)",
      "Bash(git commit:*)", "Bash(git fetch:*)", "Bash(git push origin feature/up2-prototype)",
      "Bash(AI-engine/.venv/Scripts/python.exe:*)", "Bash(.venv/Scripts/python.exe:*)",
      "Bash(dotnet build:*)", "Bash(dotnet test:*)", "Bash(dotnet publish:*)", "Bash(dotnet run:*)",
      "PowerShell(git status:*)", "PowerShell(git diff:*)", "PowerShell(git log:*)", "PowerShell(git add:*)",
      "PowerShell(git commit:*)", "PowerShell(git push origin feature/up2-prototype)",
      "PowerShell(dotnet build:*)", "PowerShell(dotnet test:*)", "PowerShell(dotnet publish:*)", "PowerShell(dotnet run:*)"
    ],
    "deny": [
      "Bash(git push --force:*)", "Bash(git push origin main:*)", "Bash(git reset --hard:*)", "Bash(git stash:*)"
    ]
  }
}
'@
Set-Content -Path (Join-Path $repo ".claude\settings.local.json") -Value $settings -Encoding utf8
$s = Join-Path $repo "scripts\resume-up2.ps1"
$a = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$s`" -WaitMinutes 1 -MaxRounds 1"
$t = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Hours 36)
Register-ScheduledTask -TaskName "UP2-Auto-Resume" -Action $a -Trigger $t -Description "Resume ICS414 UP2 build from .claude/UP2_STATE.md" -Force | Select-Object TaskName, State
