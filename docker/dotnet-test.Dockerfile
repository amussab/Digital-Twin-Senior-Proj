# Backend + twin unit tests (golden parity vs the Python engine, C5 LAN guard). Build context = repo root.
FROM mcr.microsoft.com/dotnet/sdk:10.0
WORKDIR /repo
COPY DigitalTwin.slnx DigitalTwin.Backend.slnx ./
COPY src/ src/
COPY tests/ tests/
COPY tools/ tools/
COPY AI-engine/models AI-engine/models
COPY AI-engine/reports/twin_golden.json AI-engine/reports/twin_golden.json
CMD ["sh", "-c", "dotnet test tests/DigitalTwin.Backend.Tests && dotnet test tests/DigitalTwin.Twin.Tests"]
