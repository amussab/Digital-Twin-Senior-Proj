# Rule: git workflow

- Flow (updated 2026-11-05, team decision): `main` is the reference branch and always holds the
  latest finished work. Cut a `feature/<area>` branch from `origin/main`, push it, then merge it
  into `main` (pull request or plain merge) when the user asks. `dev` and `stage` are no longer
  used. Agents do not commit straight onto `main`: they commit on a feature branch, and update
  `main` only by merging that branch when the user says so.
- Branch ownership: `feature/dashboard` = Mussab (UI). Read only. Do not commit, rebase or
  force-push it. Integration happens by merging it INTO `feature/integration`, never the other
  way.
- Our branches: `feature/ai-engine` (models), `feature/backend` (ASP.NET Core + SignalR + ONNX),
  `feature/digital-twin` (physics twin), `feature/integration` (all parts wired together).
- Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`). One logical change per
  commit. End each message with the Co-Authored-By trailer the session specifies.
- Never commit: datasets (`AI-engine/data/`), checkpoints, `.venv`, or copyrighted PDFs (the
  Kumar et al. paper stays untracked). Trained ONNX models needed for the demo may be committed
  under `AI-engine/models/` (small, < 2 MB each) via an explicit gitignore exception.
- `git fetch` before branching. No `--force` on any shared branch.
