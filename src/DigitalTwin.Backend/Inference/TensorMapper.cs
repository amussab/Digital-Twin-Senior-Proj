using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DigitalTwin.Backend.Inference;

/// <summary>
/// The ONLY place that knows how an engineered-feature history becomes ONNX input tensors. Written against the
/// AI-test/export_onnx.py contract (inputs: encoder_cont, encoder_cat, decoder_cont, decoder_cat, encoder_target,
/// target_scale; the graph may prune some). Pending: re-check against the per-bearing engine's final contract.
/// Column values are looked up BY NAME from the engineered row; unknown columns become 0 and are reported once
/// via <see cref="Unmapped"/> so a contract drift is visible rather than silent.
/// </summary>
public sealed class TensorMapper
{
    public HashSet<string> Unmapped { get; } = new();

    public List<NamedOnnxValue> Build(SubModelSpec spec, IReadOnlyList<Dictionary<string, double>> history,
        IReadOnlyDictionary<string, NodeMetadata> inputs, string modelName)
    {
        var enc = spec.EncoderLength; var dec = spec.DecoderLength;
        var rows = history.Skip(history.Count - enc).ToList();
        var feeds = new List<NamedOnnxValue>();
        foreach (var (name, meta) in inputs)
        {
            switch (name)
            {
                case "encoder_cont":
                    feeds.Add(Float(name, Cont(spec.EncoderContColumns, rows, enc, modelName), 1, enc, spec.EncoderContColumns.Count)); break;
                case "decoder_cont":
                    feeds.Add(Float(name, Cont(spec.EncoderContColumns, rows.TakeLast(1).ToList(), dec, modelName, hold: true), 1, dec, spec.EncoderContColumns.Count)); break;
                case "encoder_cat":
                    feeds.Add(Long(name, new long[enc * spec.EncoderCatColumns.Count], 1, enc, spec.EncoderCatColumns.Count)); break;
                case "decoder_cat":
                    feeds.Add(Long(name, new long[dec * spec.EncoderCatColumns.Count], 1, dec, spec.EncoderCatColumns.Count)); break;
                case "encoder_target":
                    if (meta.ElementType == typeof(long)) feeds.Add(Long(name, new long[enc], 1, enc));
                    else feeds.Add(Float(name, rows.Select(r => (float)r["hi"]).ToArray(), 1, enc));
                    break;
                case "target_scale":
                    var width = meta.Dimensions.Length > 1 && meta.Dimensions[1] > 0 ? meta.Dimensions[1] : 2;
                    var scale = new float[width]; if (width > 1) scale[1] = 1f; // identity normaliser (contract: no inverse transform)
                    feeds.Add(Float(name, scale, 1, width)); break;
                default:
                    throw new InvalidOperationException($"{modelName}: ONNX input '{name}' is not covered by TensorMapper; the contract/export changed.");
            }
        }
        return feeds;
    }

    private float[] Cont(IReadOnlyList<string> cols, IReadOnlyList<Dictionary<string, double>> rows, int len, string model, bool hold = false)
    {
        var a = new float[len * cols.Count];
        for (var t = 0; t < len; t++)
        {
            var row = hold ? rows[^1] : rows[t];
            for (var c = 0; c < cols.Count; c++)
            {
                if (row.TryGetValue(cols[c], out var v)) a[t * cols.Count + c] = (float)v;
                else if (cols[c] is "time_idx" or "relative_time_idx") a[t * cols.Count + c] = hold ? t : t - len;
                else Unmapped.Add($"{model}:{cols[c]}");
            }
        }
        return a;
    }

    private static NamedOnnxValue Float(string n, float[] d, params int[] dims) => NamedOnnxValue.CreateFromTensor(n, new DenseTensor<float>(d, dims));
    private static NamedOnnxValue Long(string n, long[] d, params int[] dims) => NamedOnnxValue.CreateFromTensor(n, new DenseTensor<long>(d, dims));
}
