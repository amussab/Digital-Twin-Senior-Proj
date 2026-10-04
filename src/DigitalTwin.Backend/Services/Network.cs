using System.Net;
using System.Net.Sockets;

namespace DigitalTwin.Backend.Services;

/// <summary>C5 (LAN-only) helpers: private-range checks used by CORS and the remote-address guard.</summary>
public static class PrivateNetwork
{
    public static bool IsPrivate(IPAddress ip)
    {
        if (ip.IsIPv4MappedToIPv6) ip = ip.MapToIPv4();
        if (IPAddress.IsLoopback(ip)) return true;
        if (ip.AddressFamily == AddressFamily.InterNetworkV6)
            return ip.IsIPv6LinkLocal || ip.IsIPv6SiteLocal || (ip.GetAddressBytes()[0] & 0xFE) == 0xFC; // fe80::/10, fec0::/10, fc00::/7
        var b = ip.GetAddressBytes();
        return b[0] == 10 || (b[0] == 172 && b[1] >= 16 && b[1] <= 31) || (b[0] == 192 && b[1] == 168) || (b[0] == 169 && b[1] == 254);
    }

    public static bool IsPrivateOrigin(string origin)
    {
        if (!Uri.TryCreate(origin, UriKind.Absolute, out var u)) return false;
        if (u.IsLoopback || u.Host.EndsWith(".local", StringComparison.OrdinalIgnoreCase)) return true;
        return IPAddress.TryParse(u.Host.Trim('[', ']'), out var ip) && IsPrivate(ip);
    }
}

/// <summary>Rejects any request whose remote address is outside private/loopback ranges (defence in depth for C5).</summary>
public sealed class PrivateNetworkGuard
{
    private readonly RequestDelegate _next; private readonly ILogger<PrivateNetworkGuard> _log;
    public PrivateNetworkGuard(RequestDelegate next, ILogger<PrivateNetworkGuard> log) { _next = next; _log = log; }

    public Task Invoke(HttpContext ctx)
    {
        var ip = ctx.Connection.RemoteIpAddress; // null under TestServer
        if (ip is not null && !PrivateNetwork.IsPrivate(ip))
        {
            _log.LogWarning("C5: rejected non-LAN client {Ip}", ip);
            ctx.Response.StatusCode = StatusCodes.Status403Forbidden;
            return ctx.Response.WriteAsync("LAN clients only");
        }
        return _next(ctx);
    }
}
