using System.Numerics;
using BulletSharp;
using BVector = BulletSharp.Math.Vector3;

namespace SmpBridge;

internal sealed class SmpBone
{
    public required string Name;
    public required bool Kinematic;
    public required RigidBody Body;
    public required DefaultMotionState Motion;
    // Bone world transform at the reference pose, and the last one supplied for kinematic bones.
    public Matrix4x4 BindWorld = Matrix4x4.Identity;
    public Matrix4x4 Current = Matrix4x4.Identity;
    // centerOfMassTransform (identity in almost every XML) and its inverse: rigWorld = LocalToRig * boneWorld.
    public Matrix4x4 LocalToRig = Matrix4x4.Identity;
    public Matrix4x4 RigToLocal = Matrix4x4.Identity;
    public bool Collides;
    public PrimitiveSpec? Primitive;
    public float MarginMultiplier = 1f;
    public int MeshId;
    public readonly List<SmpBone> Links = new();
    // Nearest kinematic bone through the constraint graph; a reset keeps the bind-pose offset to it.
    public SmpBone? Anchor;
    public Matrix4x4 AnchorOffset = Matrix4x4.Identity;
}

// The Bullet world behind the exported entry points: one world shared by every loaded SMP system.
// Dynamic bones become rigid bodies stepped by Bullet; kinematic bones
// follow transforms supplied each frame and drag the dynamic bones through 6DOF spring constraints.
internal sealed unsafe partial class SmpManager : IDisposable
{
    public const float FixedStep = 1f / 240f;
    public const int MaxSubSteps = 4;
    // Bullet runs in Skyrim units: 9.8 m/s^2 at 1 unit = 1.425 cm.
    public const float Gravity = 9.8f / 0.01425f;
    // Cloth collides only with body obstacles, never with other cloth.
    const int ClothGroup = 1;
    const int BodyGroup = 2;
    const int MaxLogLines = 256;
    // A kinematic bone that moves further than this in one update is a pose jump (seek, loop wrap, teleport), not
    // motion: the cloth is re-anchored instead of being whipped across the gap, as FSMP does.
    const float JumpDistance = 20f;
    const float JumpSpeed = 1000f;
    // A dynamic bone further than this from its rest pose is treated as a diverged simulation.
    const float MaxDrift = 150f;

    DefaultCollisionConfiguration? configuration;
    CollisionDispatcher? dispatcher;
    DbvtBroadphase? broadphase;
    SequentialImpulseConstraintSolver? solver;
    DiscreteDynamicsWorld? world;
    EmptyShape? emptyShape;

    readonly Dictionary<string, SmpBone> bones = new(StringComparer.OrdinalIgnoreCase);
    readonly List<SmpBone> kinematic = new();
    readonly List<SmpBone> dynamic = new();
    readonly List<TypedConstraint> constraints = new();
    readonly List<CollisionShape> shapes = new();
    readonly List<SmpMeshSource> meshes = new();
    readonly List<SmpMeshShape> meshShapes = new();
    readonly Dictionary<string, Matrix4x4> bindPose = new(StringComparer.OrdinalIgnoreCase);
    readonly Dictionary<string, string> candidates = new(StringComparer.OrdinalIgnoreCase);
    readonly List<string> log = new();
    bool resetPending = true;
    public string MeshScope = "";

    public void RegisterMesh(SmpMeshSource mesh) => meshes.Add(mesh);
    public IEnumerable<SmpMeshSource> FindMeshes(string name) =>
        meshes.Where(m => m.Name.Equals(name, StringComparison.OrdinalIgnoreCase) &&
            (MeshScope.Length == 0 || m.Source.Equals(MeshScope, StringComparison.OrdinalIgnoreCase)));
    public void AddMeshShape(SmpMeshShape shape) => meshShapes.Add(shape);

    public int KinematicCount => kinematic.Count;
    public int DynamicCount => dynamic.Count;
    public string KinematicName(int index) => kinematic[index].Name;
    public string DynamicName(int index) => dynamic[index].Name;

    public void Log(string message)
    {
        if (log.Count < MaxLogLines)
            log.Add(message);
    }

    public string DrainLog()
    {
        var text = string.Join('\n', log);
        log.Clear();
        return text;
    }

    public void SetBindPose(string name, in Matrix4x4 world) => bindPose[name] = world;

    public void SetCandidates(IEnumerable<string> names)
    {
        candidates.Clear();
        foreach (var name in names)
            candidates.TryAdd(name, name);
    }

    public bool IsCandidate(string name) => candidates.ContainsKey(name);

    public SmpBone? Find(string name) => bones.TryGetValue(name, out var bone) ? bone : null;

    void EnsureWorld()
    {
        if (world != null)
            return;
        configuration = new DefaultCollisionConfiguration();
        dispatcher = new CollisionDispatcher(configuration);
        broadphase = new DbvtBroadphase();
        solver = new SequentialImpulseConstraintSolver();
        world = new DiscreteDynamicsWorld(dispatcher, broadphase, solver, configuration)
        {
            Gravity = new BVector(0f, 0f, -Gravity),
        };
        world.SolverInfo.NumIterations = 10;
        world.SimulationIslandManager.SplitIslands = false;
        emptyShape = new EmptyShape();
    }

    SmpBone BuildBone(string id, string name, bool isKinematic, CollisionShape shape, bool collides, BoneTemplate template, in Matrix4x4 bind)
    {
        var rigWorld = template.CenterOfMass * bind;
        var motion = new DefaultMotionState(SmpMath.ToBullet(rigWorld));
        var inertia = BVector.Zero;
        if (!isKinematic)
        {
            shape.CalculateLocalInertia(template.Mass, out inertia);
            if (!collides)
                inertia = SmpMath.ToBullet(template.Inertia);
        }
        using var info = new RigidBodyConstructionInfo(
            isKinematic ? 0f : template.Mass,
            motion,
            shape,
            inertia)
        {
            LinearDamping = template.LinearDamping,
            AngularDamping = template.AngularDamping,
            Friction = template.Friction,
            RollingFriction = template.RollingFriction,
            Restitution = template.Restitution,
        };
        var body = new RigidBody(info);
        if (isKinematic)
            body.CollisionFlags = (body.CollisionFlags & ~CollisionFlags.StaticObject) | CollisionFlags.KinematicObject;
        body.ActivationState = ActivationState.DisableDeactivation;

        var bone = new SmpBone
        {
            Name = name,
            Kinematic = isKinematic,
            Body = body,
            Motion = motion,
            BindWorld = bind,
            Current = bind,
            LocalToRig = template.CenterOfMass,
            RigToLocal = SmpMath.Invert(template.CenterOfMass),
            Collides = collides,
            Primitive = template.Shape,
            MarginMultiplier = template.MarginMultiplier,
            MeshId = bones.Count,
        };
        AddToWorld(bone);
        if (!isKinematic)
            body.Gravity = world!.Gravity * Math.Clamp(template.GravityFactor, 0f, 1f);
        bones[id] = bone;
        (isKinematic ? kinematic : dynamic).Add(bone);
        return bone;
    }

    void AddToWorld(SmpBone bone)
    {
        if (!bone.Collides)
            world!.AddRigidBody(bone.Body, 0, 0);
        else if (bone.Kinematic)
            world!.AddRigidBody(bone.Body, BodyGroup, ClothGroup);
        else
            world!.AddRigidBody(bone.Body, ClothGroup, BodyGroup);
    }

    public SmpBone? CreateBone(string id, string rawName, BoneTemplate template)
    {
        if (bones.TryGetValue(id, out var existing))
            return existing;
        if (!candidates.TryGetValue(rawName, out var name))
            return null;
        if (!bindPose.TryGetValue(name, out var bind))
        {
            Log($"bone {name}: no bind pose was supplied; skipped");
            return null;
        }

        EnsureWorld();
        bool isKinematic = template.MotionType is "kinematic" or "keyframed" ||
            (template.MotionType != "dynamic" && template.Mass == 0f);
        var shape = template.Shape == null ? emptyShape! : CreatePrimitive(template.Shape);
        return BuildBone(id, name, isKinematic, shape, template.Shape != null, template, bind);
    }

    CollisionShape CreatePrimitive(PrimitiveSpec spec)
    {
        CollisionShape shape = spec.Kind switch
        {
            PrimitiveKind.Capsule => new CapsuleShapeZ(spec.Radius, spec.Height),
            PrimitiveKind.Box => new BoxShape(SmpMath.ToBullet(spec.HalfExtents)),
            _ => new SphereShape(spec.Radius),
        };
        shapes.Add(shape);
        if (spec.Local == Matrix4x4.Identity)
            return shape;
        var compound = new CompoundShape();
        compound.AddChildShape(SmpMath.ToBullet(spec.Local), shape);
        shapes.Add(compound);
        return compound;
    }

    public bool AddShapeToAnchor(SmpBone bone, BoneTemplate template)
    {
        if (!bone.Kinematic || bone.Collides || template.Shape == null ||
            template.MotionType == "dynamic" ||
            (template.MotionType == "" && template.Mass > 0))
            return false;
        ReleaseMeshManifolds();
        world!.RemoveRigidBody(bone.Body);
        var spec = template.Shape;
        var adjusted = new PrimitiveSpec
        {
            Kind = spec.Kind, Radius = spec.Radius, Height = spec.Height, HalfExtents = spec.HalfExtents,
            Local = spec.Local * template.CenterOfMass * bone.RigToLocal,
        };
        bone.Body.CollisionShape = CreatePrimitive(adjusted);
        bone.Body.Friction = template.Friction;
        bone.Body.Restitution = template.Restitution;
        bone.Collides = true;
        bone.Primitive = adjusted;
        Place(bone, bone.Current);
        AddToWorld(bone);
        return true;
    }

    public int GetPrimitive(int kind, int index, float* output)
    {
        var list = kind == 0 ? kinematic : kind == 1 ? dynamic : null;
        if (list == null || index < 0 || index >= list.Count)
            return SmpStatus.Argument;
        var bone = list[index];
        var spec = bone.Primitive;
        if (spec == null)
            return 0;
        output[0] = (float)spec.Kind;
        output[1] = spec.Radius;
        output[2] = spec.Height;
        output[3] = spec.HalfExtents.X;
        output[4] = spec.HalfExtents.Y;
        output[5] = spec.HalfExtents.Z;
        var bodyWorld = bone.Kinematic ? bone.LocalToRig * bone.Current : SmpMath.FromBullet(bone.Body.WorldTransform);
        SmpMath.WriteColumnMajor(spec.Local * bodyWorld, output + 6);
        return 1;
    }

    public void AddGeneric(SmpBone a, SmpBone b, in Matrix4x4 frameA, in Matrix4x4 frameB, GenericTemplate template)
    {
        EnsureWorld();
        // Constraint frames are given relative to the bones; Bullet wants them relative to the rigid bodies.
        var joint = new Generic6DofSpring2Constraint(
            a.Body, b.Body,
            SmpMath.ToBullet(frameA * a.RigToLocal),
            SmpMath.ToBullet(frameB * b.RigToLocal),
            RotateOrder.XYZ);
        for (int i = 0; i < 6; i++)
            joint.EnableSpring(i, true);
        ApplyGeneric(joint, template);
        world!.AddConstraint(joint, true);
        constraints.Add(joint);
        a.Links.Add(b);
        b.Links.Add(a);
    }

    static void ApplyGeneric(Generic6DofSpring2Constraint joint, GenericTemplate t)
    {
        joint.LinearLowerLimit = SmpMath.ToBullet(t.LinearLower);
        joint.LinearUpperLimit = SmpMath.ToBullet(t.LinearUpper);
        joint.AngularLowerLimit = SmpMath.ToBullet(t.AngularLower);
        joint.AngularUpperLimit = SmpMath.ToBullet(t.AngularUpper);
        for (int i = 0; i < 3; i++)
        {
            float linearEquilibrium = SmpMath.Component(t.LinearEquilibrium, i);
            float angularEquilibrium = SmpMath.Component(t.AngularEquilibrium, i);
            joint.SetStiffness(i, SmpMath.Component(t.LinearStiffness, i), t.LinearStiffnessLimited);
            joint.SetStiffness(i + 3, SmpMath.Component(t.AngularStiffness, i), t.AngularStiffnessLimited);
            joint.SetDamping(i, SmpMath.Component(t.LinearDamping, i), t.SpringDampingLimited);
            joint.SetDamping(i + 3, SmpMath.Component(t.AngularDamping, i), t.SpringDampingLimited);
            joint.SetEquilibriumPoint(i, linearEquilibrium);
            joint.SetEquilibriumPoint(i + 3, angularEquilibrium);
            joint.EnableSpring(i, t.LinearSprings);
            joint.EnableSpring(i + 3, t.AngularSprings);
            joint.EnableMotor(i, t.LinearMotors);
            joint.EnableMotor(i + 3, t.AngularMotors);
            joint.SetServo(i, t.LinearServo);
            joint.SetServo(i + 3, t.AngularServo);
            joint.SetServoTarget(i, linearEquilibrium);
            joint.SetServoTarget(i + 3, angularEquilibrium);
            joint.SetTargetVelocity(i, SmpMath.Component(t.LinearTargetVelocity, i));
            joint.SetTargetVelocity(i + 3, SmpMath.Component(t.AngularTargetVelocity, i));
            joint.SetMaxMotorForce(i, SmpMath.Component(t.LinearMaxMotorForce, i));
            joint.SetMaxMotorForce(i + 3, SmpMath.Component(t.AngularMaxMotorForce, i));
            joint.SetBounce(i + 3, SmpMath.Component(t.AngularBounce, i));
            joint.SetParam(ConstraintParam.Erp, t.MotorErp, i);
            joint.SetParam(ConstraintParam.Cfm, t.MotorCfm, i);
            joint.SetParam(ConstraintParam.StopErp, t.StopErp, i);
            joint.SetParam(ConstraintParam.StopCfm, t.StopCfm, i);
        }
    }

    // Called once a system is fully read: every dynamic bone remembers its nearest kinematic bone so a reset can
    // re-pose it rigidly relative to where that anchor currently is.
    public void FinishLoad()
    {
        var seen = new HashSet<SmpBone>();
        var queue = new Queue<SmpBone>();
        foreach (var bone in dynamic)
        {
            if (bone.Anchor != null)
                continue;
            seen.Clear();
            queue.Clear();
            seen.Add(bone);
            queue.Enqueue(bone);
            while (queue.Count > 0 && bone.Anchor == null)
            {
                var at = queue.Dequeue();
                foreach (var next in at.Links)
                {
                    if (!seen.Add(next))
                        continue;
                    if (next.Kinematic)
                    {
                        bone.Anchor = next;
                        bone.AnchorOffset = bone.BindWorld * SmpMath.Invert(next.BindWorld);
                        break;
                    }
                    queue.Enqueue(next);
                }
            }
        }
        resetPending = true;
    }

    void Place(SmpBone bone, in Matrix4x4 boneWorld)
    {
        var rig = SmpMath.ToBullet(bone.LocalToRig * boneWorld);
        var body = bone.Body;
        body.WorldTransform = rig;
        body.InterpolationWorldTransform = rig;
        body.LinearVelocity = BVector.Zero;
        body.AngularVelocity = BVector.Zero;
        body.InterpolationLinearVelocity = BVector.Zero;
        body.InterpolationAngularVelocity = BVector.Zero;
        body.ClearForces();
        bone.Motion.GraphicsWorldTrans = rig;
    }

    Matrix4x4 RestPose(SmpBone bone) =>
        bone.Anchor != null ? bone.AnchorOffset * bone.Anchor.Current : bone.BindWorld;

    // Snaps every body back to its reference pose with zero velocity. The pending flag makes the next step repeat
    // this against the kinematic transforms it receives, so a seek or loop never whips cloth across the gap.
    public void Reset()
    {
        if (world == null)
            return;
        foreach (var bone in kinematic)
        {
            bone.Current = bone.BindWorld;
            Place(bone, bone.BindWorld);
        }
        foreach (var bone in dynamic)
            Place(bone, bone.BindWorld);
        resetPending = true;
    }

    void ApplyPendingReset()
    {
        foreach (var bone in kinematic)
            Place(bone, bone.Current);
        foreach (var bone in dynamic)
            Place(bone, RestPose(bone));
        resetPending = false;
    }

    // Returns 0 on success, a negative code when the buffers do not match the loaded systems.
    public int Step(float dt, float* kinematicIn, int kinematicCount, float* dynamicOut, int dynamicCount)
    {
        if (world == null)
            return -1;
        if (kinematicCount != kinematic.Count || dynamicCount != dynamic.Count)
            return -2;
        if (!float.IsFinite(dt) || dt < 0f)
            return -3;

        float jumpLimit = JumpDistance + JumpSpeed * dt;
        float jumpLimitSquared = jumpLimit * jumpLimit;
        for (int i = 0; i < kinematicCount; i++)
        {
            var bone = kinematic[i];
            var m = SmpMath.ReadColumnMajor(kinematicIn + 16 * i);
            if (!SmpMath.IsFinite(m))
                return -3;
            if (Vector3.DistanceSquared(bone.Current.Translation, m.Translation) > jumpLimitSquared)
                resetPending = true;
            bone.Current = m;
            bone.Motion.GraphicsWorldTrans = SmpMath.ToBullet(bone.LocalToRig * m);
        }

        if (resetPending)
            ApplyPendingReset();
        if (dt > 0f)
        {
            if (meshShapes.Count == 0)
                world.StepSimulation(dt, MaxSubSteps, FixedStep);
            else
                StepMeshes(dt);
        }
        else
        {
            foreach (var bone in kinematic)
            {
                var transform = SmpMath.ToBullet(bone.LocalToRig * bone.Current);
                bone.Body.WorldTransform = transform;
                bone.Body.InterpolationWorldTransform = transform;
                world.UpdateSingleAabb(bone.Body);
            }
        }

        bool diverged = false;
        for (int i = 0; i < dynamicCount; i++)
        {
            var bone = dynamic[i];
            var m = bone.RigToLocal * SmpMath.FromBullet(bone.Motion.GraphicsWorldTrans);
            var rest = RestPose(bone);
            if (!SmpMath.IsFinite(m) || Vector3.DistanceSquared(m.Translation, rest.Translation) > MaxDrift * MaxDrift)
            {
                diverged = true;
                m = rest;
            }
            SmpMath.WriteColumnMajor(m, dynamicOut + 16 * i);
        }
        if (diverged)
        {
            ApplyPendingReset();
            resetPending = true;
        }
        return 0;
    }

    public void Dispose()
    {
        ReleaseMeshManifolds();
        meshPoint?.Dispose();
        meshPoint = null;
        if (world != null)
        {
            foreach (var joint in constraints)
                world.RemoveConstraint(joint);
            foreach (var bone in bones.Values)
                world.RemoveRigidBody(bone.Body);
        }
        foreach (var joint in constraints)
            joint.Dispose();
        foreach (var bone in bones.Values)
        {
            bone.Body.Dispose();
            bone.Motion.Dispose();
        }
        for (int i = shapes.Count - 1; i >= 0; i--)
            shapes[i].Dispose();
        emptyShape?.Dispose();
        world?.Dispose();
        solver?.Dispose();
        broadphase?.Dispose();
        dispatcher?.Dispose();
        configuration?.Dispose();

        constraints.Clear();
        meshShapes.Clear();
        meshes.Clear();
        bones.Clear();
        kinematic.Clear();
        dynamic.Clear();
        shapes.Clear();
        world = null;
        emptyShape = null;
    }
}
