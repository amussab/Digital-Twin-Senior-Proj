using System.Buffers.Binary;
using DigitalTwin.Backend.Models;

namespace DigitalTwin.Backend.Ingest;

/// <summary>
/// Little-endian codec for COE's fixed payload (AI-engine payload.py): rpm f32, 32 x f32 features,
/// 4 x f32 displacement, t20_ms u32 = 152 bytes. The FDR deck's 168-byte variant is unreconciled and rejected.
/// </summary>
public static class PayloadCodec
{
    public const int Size = 152;
    public const int FdrClaimedSize = 168;
    public const int FeatureCount = 32;
    public const int DisplacementCount = 4;

    public static bool TryDecode(ReadOnlySpan<byte> raw, out WindowPayload? window, out string error)
    {
        window = null;
        if (raw.Length != Size)
        {
            error = raw.Length == FdrClaimedSize
                ? "REJECTED 168-byte payload: this is the unreconciled FDR-deck variant; the backend implements the COE 152-byte layout only. Confirm the layout with COE."
                : $"REJECTED payload of {raw.Length} bytes, expected exactly {Size}.";
            return false;
        }
        var rpm = BinaryPrimitives.ReadSingleLittleEndian(raw);
        var feats = new float[FeatureCount];
        for (var i = 0; i < FeatureCount; i++) feats[i] = BinaryPrimitives.ReadSingleLittleEndian(raw[(4 + 4 * i)..]);
        var disp = new float[DisplacementCount];
        for (var i = 0; i < DisplacementCount; i++) disp[i] = BinaryPrimitives.ReadSingleLittleEndian(raw[(132 + 4 * i)..]);
        var t20 = BinaryPrimitives.ReadUInt32LittleEndian(raw[148..]);

        if (!float.IsFinite(rpm) || rpm <= 0) { error = $"REJECTED: invalid rpm {rpm}."; return false; }
        if (feats.Any(f => !float.IsFinite(f)) || disp.Any(f => !float.IsFinite(f)))
        { error = "REJECTED: non-finite value in features/displacements."; return false; }

        window = new WindowPayload(rpm, feats, disp, t20);
        error = "";
        return true;
    }

    public static byte[] Encode(WindowPayload w)
    {
        var buf = new byte[Size];
        BinaryPrimitives.WriteSingleLittleEndian(buf, w.Rpm);
        for (var i = 0; i < FeatureCount; i++) BinaryPrimitives.WriteSingleLittleEndian(buf.AsSpan(4 + 4 * i), w.Features[i]);
        for (var i = 0; i < DisplacementCount; i++) BinaryPrimitives.WriteSingleLittleEndian(buf.AsSpan(132 + 4 * i), w.Displacements[i]);
        BinaryPrimitives.WriteUInt32LittleEndian(buf.AsSpan(148), w.T20Ms);
        return buf;
    }
}
