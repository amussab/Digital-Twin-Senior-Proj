---
paths:
  - "src/**"
  - "**/*.cs"
  - "**/*.razor"
  - "**/*.csproj"
---
# Rule: .NET backend / dashboard

- Target net10.0 (the SDK installed and what the dashboard uses).
- The dashboard's contract is `DashboardSnapshot` + `IDashboardDataSource` (feature/dashboard).
  The backend must produce exactly that shape. Do not change the teammate's files. Add a
  `SignalRDashboardDataSource` in our own branch instead.
- Read model metadata only from `model_contract.json`. Never hardcode Python constants or
  class order.
- C5 constraint: Kestrel binds to the LAN only (no public tunnel, no cloud calls). Document the
  binding and a firewall note.
