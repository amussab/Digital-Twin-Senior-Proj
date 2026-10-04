# Rule: git workflow

- Flow: `feature/<area>` → `dev` → `stage` → `main`. Cut feature branches from the current
  integration point (`origin/main` + the AI work). Never commit directly to main/stage/dev from
  an agent. The user/team merges.
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
