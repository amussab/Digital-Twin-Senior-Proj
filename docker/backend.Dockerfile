# Backend + published Blazor WASM dashboard + ONNX models, one process (same layout as scripts/run-demo.*).
# Build context = repo root:  docker compose build backend
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY DigitalTwin.slnx ./
COPY src/ src/
RUN dotnet publish src/DigitalTwin.Dashboard -c Release -o /out/dashboard \
 && dotnet publish src/DigitalTwin.Backend -c Release -o /out/backend

FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app
COPY --from=build /out/backend ./
COPY --from=build /out/dashboard/wwwroot ./artifacts/dashboard/wwwroot
COPY AI-engine/models ./AI-engine/models
# Fail the build if the ONNX Runtime native lib for this arch is missing or has unresolved shared-lib deps.
RUN set -e; arch=$(dpkg --print-architecture); case $arch in amd64) rid=linux-x64;; arm64) rid=linux-arm64;; *) echo "unsupported $arch"; exit 1;; esac;     lib=/app/runtimes/$rid/native/libonnxruntime.so; test -f "$lib";     if ldd "$lib" | grep -q "not found"; then ldd "$lib"; exit 1; fi; echo "onnxruntime native lib OK: $lib"
ENV Urls=http://0.0.0.0:5080 \
    DOTNET_gcServer=0 \
    DOTNET_EnableDiagnostics=0
EXPOSE 5080/tcp 5005/udp
# the aspnet:10.0 image ships a non-root user "app" (uid 1654)
USER app
ENTRYPOINT ["dotnet", "DigitalTwin.Backend.dll"]
