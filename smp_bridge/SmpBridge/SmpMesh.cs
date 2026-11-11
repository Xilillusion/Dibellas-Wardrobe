using System.Numerics;
using BulletSharp;
using BVector = BulletSharp.Math.Vector3;

namespace SmpBridge;

internal sealed class SmpMeshSource
{
    public required string Source, Name;
    public required string[] BoneNames;
    public required Matrix4x4[] SkinMatrices;
    public required Vector3[] Positions;
    public required int[] Indices, Triangles;
    public required float[] Weights;
}

internal sealed class SmpMeshShape
{
    public required string System;
    public required SmpMeshSource Mesh;
    public required SmpBone[] Bones;
    public required bool Triangles;
    public float Margin = 1, Penetration = 1;
    public float[] Thresholds = Array.Empty<float>();
    public string Shared = "public";
    public string DisableTag = "";
    public int DisablePriority;
    public bool Disabled;
    public readonly HashSet<string> Tags = new(StringComparer.OrdinalIgnoreCase);
    public readonly HashSet<string> AllowTags = new(StringComparer.OrdinalIgnoreCase);
    public readonly HashSet<string> DenyTags = new(StringComparer.OrdinalIgnoreCase);
    public readonly HashSet<string> AllowBones = new(StringComparer.OrdinalIgnoreCase);
    public readonly HashSet<string> DenyBones = new(StringComparer.OrdinalIgnoreCase);
    public Vector3[] Positions = Array.Empty<Vector3>();
    public float[] Margins = Array.Empty<float>();
    public SmpMeshTree? VertexTree, FaceTree;
    public bool IsKinematic { get; private set; }
    Matrix4x4[] matrices = Array.Empty<Matrix4x4>();
    Action<int>? skinVertex;

    public void Build()
    {
        Positions = new Vector3[Mesh.Positions.Length];
        Margins = new float[Positions.Length];
        matrices = new Matrix4x4[Bones.Length];
        skinVertex = SkinVertex;
        IsKinematic = Bones.All(b => b.Kinematic);
        var vertices = Enumerable.Range(0, Positions.Length).Where(UsedVertex).ToArray();
        VertexTree = SmpMeshTree.Create(vertices, this, false);
        if (Triangles)
        {
            var faces = Enumerable.Range(0, Mesh.Triangles.Length / 3)
                .Where(f => UsedVertex(Mesh.Triangles[f * 3]) || UsedVertex(Mesh.Triangles[f * 3 + 1]) ||
                    UsedVertex(Mesh.Triangles[f * 3 + 2])).ToArray();
            FaceTree = SmpMeshTree.Create(faces, this, true);
        }
    }

    bool UsedVertex(int vertex)
    {
        for (int i = 0; i < 4; i++)
        {
            int slot = vertex * 4 + i;
            if (Mesh.Weights[slot] > Math.Max(float.Epsilon, Thresholds[Mesh.Indices[slot]]))
                return true;
        }
        return false;
    }

    public void Update()
    {
        for (int i = 0; i < Bones.Length; i++)
        {
            var bone = Bones[i];
            var current = bone.Kinematic ? bone.Current : bone.RigToLocal * SmpMath.FromBullet(bone.Body.WorldTransform);
            matrices[i] = Mesh.SkinMatrices[i] * current;
        }
        if (Positions.Length >= 4096 && Environment.ProcessorCount > 1)
            Parallel.For(0, Positions.Length, skinVertex!);
        else
            for (int v = 0; v < Positions.Length; v++)
                SkinVertex(v);
        VertexTree?.Update(this, false);
        FaceTree?.Update(this, true);
    }

    void SkinVertex(int v)
    {
        Vector3 position = Vector3.Zero;
        float margin = 0;
        for (int k = 0; k < 4; k++)
        {
            int slot = v * 4 + k;
            float weight = Mesh.Weights[slot];
            if (weight <= 0) continue;
            int bone = Mesh.Indices[slot];
            position += Vector3.Transform(Mesh.Positions[v], matrices[bone]) * weight;
            margin += weight * Bones[bone].MarginMultiplier;
        }
        Positions[v] = position;
        Margins[v] = margin;
    }

    public bool CanCollide(SmpMeshShape other)
    {
        if (Disabled || other.Disabled || (IsKinematic && other.IsKinematic) || Shared == "external")
            return false;
        if (Shared == "private" && System != other.System)
            return false;
        if (AllowTags.Count > 0)
        {
            foreach (var tag in other.Tags)
                if (AllowTags.Contains(tag)) return true;
            return false;
        }
        foreach (var tag in other.Tags)
            if (DenyTags.Contains(tag)) return false;
        return true;
    }

    public bool CanCollide(SmpBone bone) =>
        AllowBones.Count > 0 ? AllowBones.Contains(bone.Name) : !DenyBones.Contains(bone.Name);
}

internal sealed class SmpMeshTree
{
    public SmpMeshTree? Left, Right;
    public int Item;
    public Vector3 Min, Max;

    public static SmpMeshTree? Create(int[] items, SmpMeshShape shape, bool face)
    {
        if (items.Length == 0) return null;
        return Create(items, 0, items.Length, shape, face);
    }

    static SmpMeshTree Create(int[] items, int start, int count, SmpMeshShape shape, bool face)
    {
        if (count == 1) return new SmpMeshTree { Item = items[start] };
        Vector3 Center(int item)
        {
            if (!face) return shape.Mesh.Positions[item];
            int i = item * 3;
            return (shape.Mesh.Positions[shape.Mesh.Triangles[i]] +
                shape.Mesh.Positions[shape.Mesh.Triangles[i + 1]] +
                shape.Mesh.Positions[shape.Mesh.Triangles[i + 2]]) / 3;
        }
        var min = Center(items[start]);
        var max = min;
        for (int i = start + 1; i < start + count; i++)
        {
            var point = Center(items[i]);
            min = Vector3.Min(min, point);
            max = Vector3.Max(max, point);
        }
        var extent = max - min;
        int axis = extent.X >= extent.Y && extent.X >= extent.Z ? 0 : extent.Y >= extent.Z ? 1 : 2;
        Array.Sort(items, start, count, Comparer<int>.Create((a, b) =>
            SmpMath.Component(Center(a), axis).CompareTo(SmpMath.Component(Center(b), axis))));
        int half = count / 2;
        return new SmpMeshTree
        {
            Left = Create(items, start, half, shape, face),
            Right = Create(items, start + half, count - half, shape, face),
        };
    }

    public void Update(SmpMeshShape shape, bool face)
    {
        if (Left != null && Right != null)
        {
            Left.Update(shape, face);
            Right.Update(shape, face);
            Min = Vector3.Min(Left.Min, Right.Min);
            Max = Vector3.Max(Left.Max, Right.Max);
            return;
        }
        int a = face ? shape.Mesh.Triangles[Item * 3] : Item;
        var min = shape.Positions[a];
        var max = min;
        float margin = shape.Margins[a] * shape.Margin;
        if (face)
        {
            int b = shape.Mesh.Triangles[Item * 3 + 1], c = shape.Mesh.Triangles[Item * 3 + 2];
            min = Vector3.Min(min, Vector3.Min(shape.Positions[b], shape.Positions[c]));
            max = Vector3.Max(max, Vector3.Max(shape.Positions[b], shape.Positions[c]));
            float scale = (shape.Margins[a] + shape.Margins[b] + shape.Margins[c]) / 3;
            margin = scale * Math.Max(shape.Margin, Math.Abs(shape.Penetration));
        }
        Min = min - new Vector3(margin);
        Max = max + new Vector3(margin);
    }

    public bool Overlaps(SmpMeshTree other) =>
        Min.X <= other.Max.X && Max.X >= other.Min.X && Min.Y <= other.Max.Y &&
        Max.Y >= other.Min.Y && Min.Z <= other.Max.Z && Max.Z >= other.Min.Z;
}

internal sealed unsafe partial class SmpManager
{
    const float MeshEpsilon = 1.1920929e-7f;
    const int MaxMeshContacts = 512;
    readonly record struct MeshHit(int Vertex, int Other, Vector3 PointA, Vector3 PointB, Vector3 Normal, float Depth)
        : IComparable<MeshHit>
    {
        public int CompareTo(MeshHit other) => Depth.CompareTo(other.Depth);
    }
    struct MeshMerge
    {
        public Vector3 Normal, PointA, PointB;
        public float Weight;
        public SmpBone? A, B;
        public int Generation;
    }

    // At most two 512-hit traversals contribute 4 * 12 bone pairs per hit.
    const int MergeCapacity = 65536;
    readonly Stack<(SmpMeshTree, SmpMeshTree)> meshStack = new();
    readonly List<MeshHit> meshHits = new(MaxMeshContacts);
    readonly List<PersistentManifold> meshManifolds = new();
    ManifoldPoint? meshPoint;
    readonly HashSet<string> activeMeshTags = new(StringComparer.OrdinalIgnoreCase);
    readonly List<int> meshMergeIndices = new();
    MeshMerge[] meshMerges = Array.Empty<MeshMerge>();
    int meshMergeGeneration;

    void StepMeshes(float dt)
    {
        activeMeshTags.Clear();
        foreach (var shape in meshShapes)
            if (shape.DisableTag.Length == 0)
                activeMeshTags.UnionWith(shape.Tags);
        for (int i = 0; i < meshShapes.Count; i++)
        {
            var shape = meshShapes[i];
            shape.Disabled = shape.DisableTag.Length > 0 && activeMeshTags.Contains(shape.DisableTag);
            if (shape.DisableTag.Length == 0 || shape.Disabled) continue;
            for (int j = 0; j < meshShapes.Count; j++)
            {
                var other = meshShapes[j];
                if (i != j && other.DisableTag.Equals(shape.DisableTag, StringComparison.OrdinalIgnoreCase) &&
                    (other.DisablePriority > shape.DisablePriority ||
                     (other.DisablePriority == shape.DisablePriority && j < i)))
                {
                    shape.Disabled = true;
                    break;
                }
            }
        }

        int steps = Math.Min(MaxSubSteps, Math.Max(1, (int)Math.Ceiling(dt / FixedStep)));
        float substep = Math.Min(dt / steps, FixedStep);
        for (int step = 0; step < steps; step++)
        {
            foreach (var shape in meshShapes)
                if (!shape.Disabled && (step == 0 || !shape.IsKinematic)) shape.Update();
            int usedManifolds = 0;
            try
            {
                for (int i = 0; i < meshShapes.Count; i++)
                    for (int j = i + 1; j < meshShapes.Count; j++)
                    {
                        var a = meshShapes[i];
                        var b = meshShapes[j];
                        if (!a.CanCollide(b) || !b.CanCollide(a)) continue;
                        BeginMeshMerge();
                        if (b.Triangles)
                            Collect(a, b, true, false);
                        if (a.Triangles)
                            Collect(b, a, true, true);
                        if (!a.Triangles && !b.Triangles)
                            Collect(a, b, false, false);
                        foreach (int index in meshMergeIndices)
                        {
                            ref var merged = ref meshMerges[index];
                            if (merged.Weight <= MeshEpsilon) continue;
                            var normal = merged.Normal / merged.Weight;
                            var length = normal.Length();
                            if (length <= MeshEpsilon) continue;
                            var pointA = merged.PointA / merged.Weight;
                            var pointB = merged.PointB / merged.Weight;
                            var bodyA = merged.A!.Body;
                            var bodyB = merged.B!.Body;
                            var invA = SmpMath.Invert(SmpMath.FromBullet(bodyA.WorldTransform));
                            var invB = SmpMath.Invert(SmpMath.FromBullet(bodyB.WorldTransform));
                            var point = meshPoint ??= new ManifoldPoint(BVector.Zero, BVector.Zero, BVector.Zero, 0);
                            point.LocalPointA = SmpMath.ToBullet(Vector3.Transform(pointA, invA));
                            point.LocalPointB = SmpMath.ToBullet(Vector3.Transform(pointB, invB));
                            point.NormalWorldOnB = SmpMath.ToBullet(-normal / length);
                            point.Distance = -length;
                            point.PositionWorldOnA = SmpMath.ToBullet(pointA);
                            point.PositionWorldOnB = SmpMath.ToBullet(pointB);
                            point.CombinedFriction = bodyA.Friction * bodyB.Friction;
                            point.CombinedRestitution = bodyA.Restitution * bodyB.Restitution;
                            point.CombinedRollingFriction = bodyA.RollingFriction * bodyB.RollingFriction;
                            // Keep dispatcher-owned manifolds for reuse, but clear their contacts after every substep.
                            if (usedManifolds == meshManifolds.Count)
                                meshManifolds.Add(dispatcher!.GetNewManifold(bodyA, bodyB));
                            var manifold = meshManifolds[usedManifolds++];
                            manifold.SetBodies(bodyA, bodyB);
                            manifold.AddManifoldPoint(point, false);
                        }
                    }
                world!.StepSimulation(substep, 0, substep);
            }
            finally
            {
                for (int i = 0; i < usedManifolds; i++)
                    meshManifolds[i].ClearManifold();
            }
        }
    }

    void BeginMeshMerge()
    {
        meshMergeIndices.Clear();
        if (++meshMergeGeneration == int.MaxValue)
        {
            Array.Clear(meshMerges);
            meshMergeGeneration = 1;
        }
    }

    void ReleaseMeshManifolds()
    {
        foreach (var manifold in meshManifolds)
            dispatcher!.ReleaseManifold(manifold);
        meshManifolds.Clear();
    }

    void Collect(SmpMeshShape spheres, SmpMeshShape other, bool triangles, bool swap)
    {
        var otherTree = triangles ? other.FaceTree : other.VertexTree;
        if (spheres.VertexTree == null || otherTree == null || !spheres.VertexTree.Overlaps(otherTree)) return;
        meshStack.Clear();
        meshHits.Clear();
        meshStack.Push((spheres.VertexTree, otherTree));
        while (meshStack.Count > 0 && meshHits.Count < MaxMeshContacts)
        {
            var (a, b) = meshStack.Pop();
            if (!a.Overlaps(b)) continue;
            if (a.Left != null && a.Right != null)
            {
                meshStack.Push((a.Left, b)); meshStack.Push((a.Right, b));
            }
            else if (b.Left != null && b.Right != null)
            {
                meshStack.Push((a, b.Left)); meshStack.Push((a, b.Right));
            }
            else if (TryMeshHit(spheres, other, a.Item, b.Item, triangles, out var hit))
                meshHits.Add(hit);
        }
        meshHits.Sort();
        foreach (var hit in meshHits)
            MergeMeshHit(spheres, other, hit, triangles, swap);
    }

    static bool TryMeshHit(SmpMeshShape a, SmpMeshShape b, int vertex, int item, bool triangle, out MeshHit hit)
    {
        hit = default;
        var p = a.Positions[vertex];
        float r = a.Margins[vertex] * a.Margin;
        if (!triangle)
        {
            var delta = p - b.Positions[item];
            float length = delta.Length();
            float otherRadius = b.Margins[item] * b.Margin;
            float depth = length - r - otherRadius;
            if (depth >= -MeshEpsilon) return false;
            var normal = length > MeshEpsilon ? delta / length : Vector3.UnitX;
            hit = new(vertex, item, p - normal * r, b.Positions[item] + normal * otherRadius, normal, depth);
            return true;
        }
        int i0 = b.Mesh.Triangles[item * 3], i1 = b.Mesh.Triangles[item * 3 + 1], i2 = b.Mesh.Triangles[item * 3 + 2];
        var p0 = b.Positions[i0]; var p1 = b.Positions[i1]; var p2 = b.Positions[i2];
        var cross = Vector3.Cross(p1 - p0, p2 - p0);
        float area = cross.Length();
        if (area <= MeshEpsilon) return false;
        var normalPlate = cross / area;
        float scale = (b.Margins[i0] + b.Margins[i1] + b.Margins[i2]) / 3;
        float penetration = b.Penetration * scale;
        if (penetration < 0) { normalPlate = -normalPlate; penetration = -penetration; }
        float distance = Vector3.Dot(p - p0, normalPlate);
        var projection = p - normalPlate * distance;
        float radius = r + scale * b.Margin;
        if (penetration < MeshEpsilon)
        {
            if (distance < 0) { distance = -distance; normalPlate = -normalPlate; }
            if (distance >= radius) return false;
        }
        else if (distance >= radius || distance < -penetration) return false;
        var ap = projection - p0; var bp = projection - p1; var cp = projection - p2;
        float aa = Vector3.Cross(bp, cp).Length(), ab = Vector3.Cross(cp, ap).Length(), ac = Vector3.Cross(ap, bp).Length();
        if (ac + ab > area || ab + aa > area || aa + ac > area) return false;
        float depthPlate = distance - radius;
        if (depthPlate >= -MeshEpsilon) return false;
        hit = new(vertex, item, p - normalPlate * r, projection, normalPlate, depthPlate);
        return true;
    }

    void MergeMeshHit(SmpMeshShape a, SmpMeshShape b, MeshHit hit, bool triangle, bool swap)
    {
        Span<int> vertices = stackalloc int[3];
        int vertexCount = triangle ? 3 : 1;
        for (int i = 0; i < vertexCount; i++)
            vertices[i] = triangle ? b.Mesh.Triangles[hit.Other * 3 + i] : hit.Other;
        float flexibleA = Flexible(a, hit.Vertex), flexibleB = 0;
        for (int i = 0; i < vertexCount; i++) flexibleB += Flexible(b, vertices[i]) / vertexCount;
        float flexible = Math.Max(flexibleA, flexibleB);
        if (flexible <= MeshEpsilon) return;
        float weightedDepth = flexible * hit.Depth;
        float weight = weightedDepth * weightedDepth;
        if (weight <= 0) return;
        var weightedNormal = hit.Normal * (weightedDepth * weight);
        for (int i = 0; i < 4; i++)
        {
            int slotA = hit.Vertex * 4 + i;
            int indexA = a.Mesh.Indices[slotA];
            if (a.Mesh.Weights[slotA] <= a.Thresholds[indexA]) continue;
            var boneA = a.Bones[indexA];
            if (!b.CanCollide(boneA)) continue;
            for (int v = 0; v < vertexCount; v++)
                for (int j = 0; j < 4; j++)
                {
                    int slotB = vertices[v] * 4 + j;
                    int indexB = b.Mesh.Indices[slotB];
                    if (b.Mesh.Weights[slotB] <= b.Thresholds[indexB]) continue;
                    var boneB = b.Bones[indexB];
                    if (ReferenceEquals(boneA, boneB) || (boneA.Kinematic && boneB.Kinematic) ||
                        !a.CanCollide(boneB)) continue;
                    var first = swap ? boneB : boneA;
                    var second = swap ? boneA : boneB;
                    var normal = swap ? -weightedNormal : weightedNormal;
                    if (meshMerges.Length == 0) meshMerges = new MeshMerge[MergeCapacity];
                    uint hash = unchecked((uint)first.MeshId * 0x9E3779B1u ^ (uint)second.MeshId * 0x85EBCA77u);
                    int index = (int)(hash & (MergeCapacity - 1));
                    while (meshMerges[index].Generation == meshMergeGeneration &&
                        (!ReferenceEquals(meshMerges[index].A, first) || !ReferenceEquals(meshMerges[index].B, second)))
                        index = (index + 1) & (MergeCapacity - 1);
                    ref var cell = ref meshMerges[index];
                    if (cell.Generation != meshMergeGeneration)
                    {
                        cell = new MeshMerge { A = first, B = second, Generation = meshMergeGeneration };
                        meshMergeIndices.Add(index);
                    }
                    if (cell.Weight > MeshEpsilon && Vector3.Dot(cell.Normal, normal) < 0) continue;
                    cell.Normal += normal;
                    cell.PointA += (swap ? hit.PointB : hit.PointA) * weight;
                    cell.PointB += (swap ? hit.PointA : hit.PointB) * weight;
                    cell.Weight += weight;
                }
        }
    }

    static float Flexible(SmpMeshShape shape, int vertex)
    {
        float flexible = 0;
        for (int i = 0; i < 4; i++)
        {
            int slot = vertex * 4 + i;
            if (!shape.Bones[shape.Mesh.Indices[slot]].Kinematic)
                flexible += shape.Mesh.Weights[slot];
        }
        return flexible;
    }
}