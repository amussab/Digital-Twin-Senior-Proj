using DigitalTwin.Dashboard;
using DigitalTwin.Dashboard.Services;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;

var builder = WebAssemblyHostBuilder.CreateDefault(args);

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");
// Data source switch (wwwroot/appsettings.json): "SignalR" (default) or "Mock".
// HubUrl empty = same origin as the page (dashboard served by the backend).
if (string.Equals(builder.Configuration["DataSource"], "Mock", StringComparison.OrdinalIgnoreCase))
{
    builder.Services.AddSingleton<IDashboardDataSource, MockDashboardDataSource>();
}
else
{
    var hub = builder.Configuration["HubUrl"];
    if (string.IsNullOrWhiteSpace(hub)) hub = new Uri(new Uri(builder.HostEnvironment.BaseAddress), "hubs/dashboard").ToString();
    builder.Services.AddSingleton<IDashboardDataSource>(new SignalRDashboardDataSource(hub));
}

await builder.Build().RunAsync();
