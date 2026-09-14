# Digital Twin Dashboard Frontend

Standalone Blazor WebAssembly dashboard using mock monitoring data. The UI depends on
`IDashboardDataSource`, not on the COE binary payload, so a future SignalR source can replace
the mock source without changing dashboard components.

## Run locally

1. Install the .NET 10 SDK.
2. From the repository root, run:

   ```bash
   dotnet restore
   dotnet run --project src/DigitalTwin.Dashboard
   ```

3. Open the local URL printed by the command.

## Backend integration point

Keep `DashboardSnapshot` as the frontend-facing message. Add a SignalR implementation of
`IDashboardDataSource`, then replace the mock registration in `Program.cs`.

Physics RUL and residual are nullable until the team resolves whether the physics model is an
independent estimator or an AI input.
