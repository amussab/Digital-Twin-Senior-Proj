using DigitalTwin.Dashboard.Models;

namespace DigitalTwin.Dashboard.Services;

public interface IDashboardDataSource
{
    IAsyncEnumerable<DashboardSnapshot> StreamAsync(CancellationToken cancellationToken);
}
