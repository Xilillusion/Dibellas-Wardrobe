using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace SmpBridge;

internal static class SmpStatus
{
    public const int Ok = 0;
    public const int Argument = -1;
    public const int Io = -2;
    public const int Xml = -3;
    public const int NoDynamicBones = -4;
    public const int Internal = -5;
    public const int Unsupported = -6;
}

// Plain C ABI over SmpManager. Matrices cross the boundary as contiguous float32 buffers of 16 elements per matrix
// in the layout of a C-ordered column-vector 4x4 array (translation at elements 3, 7 and 11). Strings are
// NUL-terminated UTF-8. An exception escaping an [UnmanagedCallersOnly] method terminates the host process, so every
// entry point catches everything and reports through its return value or the log.
public static unsafe class SmpExports
{
    // Bumped whenever an exported signature or buffer layout changes.
    const int AbiVersion = 3;
    const int MaxCandidates = 100_000;
    const int KindKinematic = 0;
    const int KindDynamic = 1;

    static readonly object Gate = new();
    static SmpManager? manager;
    static string detachedLog = "";
    static int lastStepStatus;

    static SmpManager Manager => manager ??= new SmpManager();

    static string? Utf8(byte* p) => p == null ? null : Marshal.PtrToStringUTF8((IntPtr)p);

    static void Record(Exception ex)
    {
        var message = $"native error: {ex.GetType().Name}: {ex.Message}";
        if (manager != null)
            manager.Log(message);
        else
            detachedLog = message;
    }

    [UnmanagedCallersOnly(EntryPoint = "SmpAbiVersion", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SmpAbiVersion() => AbiVersion;

    // Supplies reference-pose world transforms (candidate bones and body collider nodes) before loading XML.
    [UnmanagedCallersOnly(EntryPoint = "SmpSetBindPose", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SmpSetBindPose(byte** utf8Names, float* matrices, int count)
    {
        if (utf8Names == null || matrices == null || count < 0 || count > MaxCandidates)
            return SmpStatus.Argument;
        try
        {
            lock (Gate)
            {
                var target = Manager;
                for (int i = 0; i < count; i++)
                {
                    var name = Utf8(utf8Names[i]);
                    if (string.IsNullOrEmpty(name))
                        continue;
                    var m = SmpMath.ReadColumnMajor(matrices + 16 * i);
                    if (SmpMath.IsFinite(m))
                        target.SetBindPose(name, m);
                }

            }
            return SmpStatus.Ok;
        }
        catch (Exception ex)
        {
            Record(ex);
            return SmpStatus.Internal;
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "SmpRegisterMesh", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SmpRegisterMesh(byte* sourceName, byte* shapeName, byte** boneNames, float* skinMatrices,
                    int boneCount, float* positions, int* indices, float* weights, int vertexCount, int* triangles, int triangleCount)
    {
        if (sourceName == null || shapeName == null || boneNames == null || skinMatrices == null ||
            positions == null || indices == null || weights == null || triangles == null ||
            boneCount <= 0 || boneCount > 4096 || vertexCount <= 0 || vertexCount > 1_000_000 ||
            triangleCount < 0 || triangleCount > 2_000_000)
            return SmpStatus.Argument;
        try
        {
            var source = Utf8(sourceName);
            var name = Utf8(shapeName);
            if (string.IsNullOrEmpty(source) || string.IsNullOrEmpty(name))
                return SmpStatus.Argument;
            var mesh = new SmpMeshSource
            {
                Source = source,
                Name = name,
                BoneNames = new string[boneCount],
                SkinMatrices = new Matrix4x4[boneCount],
                Positions = new Vector3[vertexCount],
                Indices = new int[vertexCount * 4],
                Weights = new float[vertexCount * 4],
                Triangles = new int[triangleCount * 3],
            };
            for (int i = 0; i < boneCount; i++)
            {
                var bone = Utf8(boneNames[i]);
                if (string.IsNullOrEmpty(bone))
                    return SmpStatus.Argument;
                mesh.BoneNames[i] = bone;
                mesh.SkinMatrices[i] = SmpMath.ReadColumnMajor(skinMatrices + i * 16);
                if (!SmpMath.IsFinite(mesh.SkinMatrices[i]))
                    return SmpStatus.Argument;
            }
            for (int i = 0; i < vertexCount; i++)
            {
                var position = new Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
                if (!float.IsFinite(position.X) || !float.IsFinite(position.Y) || !float.IsFinite(position.Z))
                    return SmpStatus.Argument;
                mesh.Positions[i] = position;
                for (int j = 0; j < 4; j++)
                {
                    int slot = i * 4 + j;
                    if (indices[slot] < 0 || indices[slot] >= boneCount ||
                        !float.IsFinite(weights[slot]) || weights[slot] < 0)
                        return SmpStatus.Argument;
                    mesh.Indices[slot] = indices[slot];
                    mesh.Weights[slot] = weights[slot];
                }
            }
            for (int i = 0; i < triangleCount * 3; i++)
            {
                if (triangles[i] < 0 || triangles[i] >= vertexCount)
                    return SmpStatus.Argument;
                mesh.Triangles[i] = triangles[i];
            }
            lock (Gate)
                Manager.RegisterMesh(mesh);
            return SmpStatus.Ok;
        }
        catch (Exception ex)
        {
            Record(ex);
            return SmpStatus.Internal;
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "SmpSetMeshScope", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SmpSetMeshScope(byte* sourceName)
    {
        if (sourceName == null)
            return SmpStatus.Argument;
        try
        {
            lock (Gate)
                Manager.MeshScope = Utf8(sourceName) ?? "";
            return SmpStatus.Ok;
        }
        catch (Exception ex)
        {
            Record(ex);
            return SmpStatus.Internal;
        }
    }

    // Loads one SMP XML into the world. Only bones named in `utf8CandidateBones` exist on the actor; anything else
    // the XML mentions is skipped. Returns 0 when at least one body was built, otherwise a negative status.
    [UnmanagedCallersOnly(EntryPoint = "SmpLoadXml", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SmpLoadXml(byte* utf8XmlPath, byte** utf8CandidateBones, int candidateCount)
    {
        if (utf8XmlPath == null || candidateCount < 0 || candidateCount > MaxCandidates ||
            (candidateCount > 0 && utf8CandidateBones == null))
            return SmpStatus.Argument;
        try
        {
            var path = Utf8(utf8XmlPath);
            if (string.IsNullOrEmpty(path))
                return SmpStatus.Argument;
            var names = new List<string>(candidateCount);
            for (int i = 0; i < candidateCount; i++)
            {
                var name = Utf8(utf8CandidateBones[i]);
                if (!string.IsNullOrEmpty(name))
                    names.Add(name);
            }
            lock (Gate)
            {
                var target = Manager;
                target.SetCandidates(names);
                return new SmpXmlLoader(target, Path.GetFileName(path)).Load(path);
            }
        }
        catch (Exception ex)
        {
            Record(ex);
            return SmpStatus.Internal;
        }
    }

    // kind 0: kinematic input slots, kind 1: dynamic output slots.
    [UnmanagedCallersOnly(EntryPoint = "SmpGetBodyCount", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SmpGetBodyCount(int kind)
    {
        try
        {
            lock (Gate)
            {
                if (manager == null)
                    return 0;
                return kind == KindKinematic ? manager.KinematicCount : kind == KindDynamic ? manager.DynamicCount : SmpStatus.Argument;
            }
        }
        catch (Exception ex)
        {
            Record(ex);
            return SmpStatus.Internal;
        }
    }

    // Copies the NUL-terminated UTF-8 name of a slot into `buffer`; returns its length in bytes, or a negative status.
    [UnmanagedCallersOnly(EntryPoint = "SmpGetBodyName", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SmpGetBodyName(int kind, int index, byte* buffer, int capacity)
    {
        if (buffer == null || capacity <= 0 || index < 0)
            return SmpStatus.Argument;
        try
        {
            lock (Gate)
            {
                if (manager == null)
                    return SmpStatus.Argument;
                string name;
                if (kind == KindKinematic && index < manager.KinematicCount)
                    name = manager.KinematicName(index);
                else if (kind == KindDynamic && index < manager.DynamicCount)
                    name = manager.DynamicName(index);
                else
                    return SmpStatus.Argument;
                return CopyUtf8(name, buffer, capacity, truncate: false);
            }

        }
        catch (Exception ex)
        {
            Record(ex);
            return SmpStatus.Internal;
        }
    }

    // Writes primitive kind, radius, height, half-extents and a column-vector world matrix (22 floats).
    [UnmanagedCallersOnly(EntryPoint = "SmpGetPrimitive", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SmpGetPrimitive(int kind, int index, float* output, int capacity)
    {
        if (output == null || capacity < 22)
            return SmpStatus.Argument;
        try
        {
            lock (Gate)
                return manager?.GetPrimitive(kind, index, output) ?? SmpStatus.Argument;
        }
        catch (Exception ex)
        {
            Record(ex);
            return SmpStatus.Internal;
        }
    }

    // Returns and clears the parse/build diagnostics as newline-separated UTF-8 text.
    [UnmanagedCallersOnly(EntryPoint = "SmpGetLog", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SmpGetLog(byte* buffer, int capacity)
    {
        if (buffer == null || capacity <= 0)
            return SmpStatus.Argument;
        try
        {
            lock (Gate)
            {
                var text = manager != null ? manager.DrainLog() : "";
                if (detachedLog.Length > 0)
                {
                    text = text.Length > 0 ? detachedLog + "\n" + text : detachedLog;
                    detachedLog = "";
                }
                return CopyUtf8(text, buffer, capacity, truncate: true);
            }
        }
        catch (Exception ex)
        {
            Record(ex);
            return SmpStatus.Internal;
        }
    }

    static int CopyUtf8(string text, byte* buffer, int capacity, bool truncate)
    {
        var span = new Span<byte>(buffer, capacity);
        int needed = Encoding.UTF8.GetByteCount(text);
        if (needed >= capacity)
        {
            if (!truncate)
                return SmpStatus.Argument;
            // Truncate on a character boundary so the text stays valid UTF-8.
            var encoder = Encoding.UTF8.GetEncoder();
            encoder.Convert(text.AsSpan(), span[..(capacity - 1)], true, out _, out int bytesUsed, out _);
            span[bytesUsed] = 0;
            return bytesUsed;
        }
        int written = Encoding.UTF8.GetBytes(text, span);
        span[written] = 0;
        return written;
    }

    // Advances the simulation by dt seconds. inKinematicMatrices holds kinematicCount matrices (the order reported by
    // SmpGetBodyName kind 0); the evaluated world transform of every dynamic bone is written to outDynamicMatrices
    // (kind 1 order).
    [UnmanagedCallersOnly(EntryPoint = "SmpStep", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static void SmpStep(float dt, float* inKinematicMatrices, int kinematicCount, float* outDynamicMatrices, int dynamicCount)
    {
        if (kinematicCount < 0 || dynamicCount < 0 ||
            (kinematicCount > 0 && inKinematicMatrices == null) ||
            (dynamicCount > 0 && outDynamicMatrices == null))
        {
            lastStepStatus = SmpStatus.Argument;
            return;
        }
        try
        {
            lock (Gate)
                lastStepStatus = manager?.Step(dt, inKinematicMatrices, kinematicCount, outDynamicMatrices, dynamicCount) ?? -1;
        }
        catch (Exception ex)
        {
            lastStepStatus = SmpStatus.Internal;
            Record(ex);
        }
    }

    // Status of the last SmpStep: 0 on success, -1 no world loaded, -2 buffer counts do not match the loaded bodies.
    [UnmanagedCallersOnly(EntryPoint = "SmpLastStepStatus", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static int SmpLastStepStatus() => lastStepStatus;

    // Zeroes velocities and returns every body to its reference pose; the next SmpStep re-anchors the cloth to the
    // kinematic transforms it receives.
    [UnmanagedCallersOnly(EntryPoint = "SmpReset", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static void SmpReset()
    {
        try
        {
            lock (Gate)
                manager?.Reset();
        }
        catch (Exception ex)
        {
            Record(ex);
        }
    }

    // Releases every native constraint, body, shape and the Bullet world. The library may be loaded again afterwards.
    [UnmanagedCallersOnly(EntryPoint = "SmpDestroy", CallConvs = new[] { typeof(CallConvCdecl) })]
    public static void SmpDestroy()
    {
        try
        {
            lock (Gate)
            {
                manager?.Dispose();
                manager = null;
                lastStepStatus = 0;
            }
        }
        catch (Exception ex)
        {
            detachedLog = $"native error: {ex.GetType().Name}: {ex.Message}";
            manager = null;
        }
    }
}
