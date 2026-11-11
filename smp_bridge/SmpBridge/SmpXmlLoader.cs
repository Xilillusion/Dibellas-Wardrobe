using System.Globalization;
using System.Numerics;
using System.Xml;
using System.Xml.Linq;

namespace SmpBridge;

// Reads one HDT-SMP physics XML in document order, as FSMP does: a <bone> takes whatever the unnamed default
// template holds at the point it is created, and templates stack through `extends`.
//
// Supported: <bone> / <rigid-body> (+ -default templates, optional <shape> primitive), <generic-constraint>
// (+ default templates, <constraint-group>) and skinned vertex and triangle collision shapes.
// Stiffspring and conetwist constraints are not simulated and are reported in the log.
internal sealed class SmpXmlLoader
{
    sealed class ParseException(string message) : Exception(message);

    const long MaxFileBytes = 8L * 1024 * 1024;

    readonly SmpManager manager;
    readonly string system;
    readonly Dictionary<string, BoneTemplate> boneTemplates = new(StringComparer.OrdinalIgnoreCase);
    readonly Dictionary<string, GenericTemplate> genericTemplates = new(StringComparer.OrdinalIgnoreCase);
    readonly HashSet<string> reported = new();
    int createdBodies;
    int skippedConstraints;
    int unsupportedConstraints;

    public SmpXmlLoader(SmpManager manager, string systemName)
    {
        this.manager = manager;
        system = systemName;
    }

    public int Load(string path)
    {
        XDocument document;
        try
        {
            var info = new FileInfo(path);
            if (!info.Exists)
            {
                manager.Log($"{system}: file not found");
                return SmpStatus.Io;
            }
            if (info.Length > MaxFileBytes)
            {
                manager.Log($"{system}: file is larger than {MaxFileBytes} bytes");
                return SmpStatus.Io;
            }
            var settings = new XmlReaderSettings { DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null };
            using var stream = File.OpenRead(path);
            using var reader = XmlReader.Create(stream, settings);
            document = XDocument.Load(reader, LoadOptions.SetLineInfo);
        }
        catch (XmlException ex)
        {
            manager.Log($"{system}: XML is not well formed (line {ex.LineNumber}): {ex.Message}");
            return SmpStatus.Xml;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            manager.Log($"{system}: {ex.Message}");
            return SmpStatus.Io;
        }

        var root = document.Root;
        if (root == null || root.Name.LocalName != "system")
        {
            manager.Log($"{system}: root element is not <system>");
            return SmpStatus.Xml;
        }
        if (root.Descendants().Any(e => e.Name.LocalName is "pattern" or "pattern-default"))
        {
            manager.Log($"{system}: <pattern> macros are not expanded; no physics built for this file");
            return SmpStatus.Unsupported;
        }

        try
        {
            ValidateElements(root);
            ReadSystem(root);
        }

        catch (ParseException ex)
        {
            manager.Log($"{system}: {ex.Message}");
            return SmpStatus.Xml;
        }

        manager.FinishLoad();
        if (skippedConstraints > 0)
            manager.Log($"{system}: {skippedConstraints} constraint(s) skipped (a bone is missing on this actor or both are kinematic)");
        if (unsupportedConstraints > 0)
            manager.Log($"{system}: {unsupportedConstraints} stiffspring/conetwist constraint(s) ignored; only generic constraints are simulated");
        if (createdBodies == 0)
        {
            manager.Log($"{system}: no new rigid body in the XML matches this actor");
            return SmpStatus.NoDynamicBones;
        }
        return SmpStatus.Ok;
    }

    void ValidateElements(XElement root)
    {
        foreach (var e in root.Descendants())
        {
            switch (Name(e))
            {
                case "bone" or "rigid-body" or "bone-default" or "rigid-body-default":
                    ReadBoneTemplate(e, new BoneTemplate());
                    break;
                case "generic-constraint" or "generic-constraint-default":
                    ReadGenericTemplate(e, new GenericTemplate());
                    break;
            }
        }
    }

    void Warn(XElement at, string message)
    {
        var line = at is IXmlLineInfo info && info.HasLineInfo() ? $" (line {info.LineNumber})" : "";
        var text = $"{system}{line}: {message}";
        if (reported.Add(text))
            manager.Log(text);
    }

    // ---- element helpers ----

    static string Name(XElement e) => e.Name.LocalName;
    static string Text(XElement e) => e.Nodes().OfType<XText>().FirstOrDefault()?.Value ?? "";
    static string Optional(XElement e, string name) => e.Attribute(name)?.Value ?? "";

    static string Line(XElement e) => e is IXmlLineInfo info && info.HasLineInfo() ? $"line {info.LineNumber}" : "unknown line";

    static bool TryFloat(string text, out float value)
    {
        var s = text.Trim();
        var comma = s.IndexOf(',');
        if (comma >= 0)
            s = s[..comma] + "." + s[(comma + 1)..];
        if (s.StartsWith('+'))
            s = s[1..];
        return float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out value) && float.IsFinite(value);
    }

    static float Float(string text, XElement at) =>
        TryFloat(text, out var value) ? value : throw new ParseException($"<{Name(at)}> at {Line(at)} is not a number: '{text.Trim()}'");

    static float Float(XElement e) => Float(Text(e), e);

    static bool Bool(XElement e) => Text(e).Trim() switch
    {
        "true" or "1" => true,
        "false" or "0" => false,
        var other => throw new ParseException($"<{Name(e)}> at {Line(e)} is not a boolean: '{other}'"),
    };

    static string Attribute(XElement e, string name) =>
        e.Attribute(name)?.Value ?? throw new ParseException($"<{Name(e)}> at {Line(e)} is missing attribute {name}");

    static Vector3 Vector(XElement e) =>
        new(Float(Attribute(e, "x"), e), Float(Attribute(e, "y"), e), Float(Attribute(e, "z"), e));

    // <centerOfMassTransform> / <frameInA> ...: a rotation (<basis> quaternion or <basis-axis-angle>) and an <origin>.
    static Matrix4x4 Transform(XElement e)
    {
        var rotation = Quaternion.Identity;
        var origin = Vector3.Zero;
        foreach (var child in e.Elements())
        {
            switch (Name(child))
            {
                case "basis" or "rotation":
                    {
                        var q = new Quaternion(
                            Float(Attribute(child, "x"), child), Float(Attribute(child, "y"), child),
                            Float(Attribute(child, "z"), child), Float(Attribute(child, "w"), child));
                        if (q.LengthSquared() < 1e-12f)
                            throw new ParseException($"zero rotation at {Line(child)}");
                        rotation = Quaternion.Normalize(q);
                        break;
                    }
                case "basis-axis-angle":
                    {
                        var axis = Vector(child);
                        var angle = Float(Attribute(child, "angle"), child);
                        if (axis.LengthSquared() < 1e-12f)
                        {
                            axis = Vector3.UnitX;
                            angle = 0;
                        }
                        rotation = Quaternion.CreateFromAxisAngle(Vector3.Normalize(axis), angle);
                        break;
                    }
                case "origin" or "translation":
                    origin = Vector(child);
                    break;
            }
        }
        var m = Matrix4x4.CreateFromQuaternion(rotation);
        m.Translation = origin;
        return m;
    }

    // ---- templates ----

    BoneTemplate BoneDefault(string name) =>
        boneTemplates.TryGetValue(name, out var found) ? found
        : boneTemplates.TryGetValue("", out var unnamed) ? unnamed
        : boneTemplates[""] = new BoneTemplate();

    GenericTemplate GenericDefault(string name) =>
        genericTemplates.TryGetValue(name, out var found) ? found
        : genericTemplates.TryGetValue("", out var unnamed) ? unnamed
        : genericTemplates[""] = new GenericTemplate();

    void ReadBoneTemplate(XElement e, BoneTemplate t)
    {
        foreach (var c in e.Elements())
        {
            switch (Name(c))
            {
                case "mass": t.Mass = Float(c); break;
                case "motion-type":
                    t.MotionType = Text(c).Trim().ToLowerInvariant();
                    if (t.MotionType is not ("dynamic" or "kinematic" or "keyframed"))
                        throw new ParseException($"invalid motion-type '{t.MotionType}' at {Line(c)}");
                    break;
                case "inertia": t.Inertia = Vector(c); break;
                case "centerOfMassTransform": t.CenterOfMass = Transform(c); break;
                case "translation" or "origin":
                    t.CenterOfMass.Translation = Vector(c);
                    break;
                case "rotation":
                    var q = new Quaternion(Float(Attribute(c, "x"), c), Float(Attribute(c, "y"), c),
                        Float(Attribute(c, "z"), c), Float(Attribute(c, "w"), c));
                    if (q.LengthSquared() < 1e-12f)
                        throw new ParseException($"zero rotation at {Line(c)}");
                    var translation = t.CenterOfMass.Translation;
                    t.CenterOfMass = Matrix4x4.CreateFromQuaternion(Quaternion.Normalize(q));
                    t.CenterOfMass.Translation = translation;
                    break;
                case "linearDamping": t.LinearDamping = Float(c); break;
                case "angularDamping": t.AngularDamping = Float(c); break;
                case "friction": t.Friction = Float(c); break;
                case "rollingFriction": t.RollingFriction = Float(c); break;
                case "restitution": t.Restitution = Float(c); break;
                case "margin-multiplier": t.MarginMultiplier = Float(c); break;
                case "gravity-factor": t.GravityFactor = Math.Clamp(Float(c), 0f, 1f); break;
                case "shape": t.Shape = ReadPrimitive(c); break;
                case "wind-factor" or "collision-filter" or "can-collide-with-bone" or "no-collide-with-bone": break;
                default: Warn(c, $"unknown element <{Name(c)}> skipped"); break;
            }
        }
    }

    // <shape type="sphere|capsule|box"> with the parameters as attributes or child elements. Also accepts the
    // primitive as the shape's only child element: <shape><capsule radius=".." height=".."/></shape>.
    PrimitiveSpec? ReadPrimitive(XElement e)
    {
        var kindElement = e;
        var kindName = Optional(e, "type");
        if (kindName.Length == 0)
        {
            var child = e.Elements().FirstOrDefault(c => Name(c) is "sphere" or "capsule" or "box");
            if (child == null)
            {
                throw new ParseException($"<shape> at {Line(e)} has no supported type");
            }
            kindElement = child;
            kindName = Name(child);
        }

        float Param(string name)
        {
            var attr = kindElement.Attribute(name)?.Value ?? e.Attribute(name)?.Value;
            if (attr != null)
                return Float(attr, kindElement);
            var child = kindElement.Elements().Concat(e.Elements()).FirstOrDefault(c => Name(c) == name);
            return child != null ? Float(child) : throw new ParseException($"<shape> at {Line(e)} is missing {name}");
        }

        var spec = new PrimitiveSpec();
        spec.Local = Transform(kindElement);
        if (kindElement != e)
            spec.Local *= Transform(e);
        switch (kindName.Trim().ToLowerInvariant())
        {
            case "sphere":
                spec.Kind = PrimitiveKind.Sphere;
                spec.Radius = Param("radius");
                break;
            case "capsule":
                spec.Kind = PrimitiveKind.Capsule;
                spec.Radius = Param("radius");
                var p0 = kindElement.Elements().Concat(e.Elements()).FirstOrDefault(c => Name(c) == "p0");
                var p1 = kindElement.Elements().Concat(e.Elements()).FirstOrDefault(c => Name(c) == "p1");
                if ((p0 == null) != (p1 == null))
                    throw new ParseException($"capsule at {Line(e)} requires both p0 and p1");
                if (p0 != null && p1 != null)
                {
                    var start = Vector(p0);
                    var end = Vector(p1);
                    var delta = end - start;
                    spec.Height = delta.Length();
                    var direction = spec.Height > 1e-6f ? delta / spec.Height : Vector3.UnitZ;
                    var dot = Math.Clamp(Vector3.Dot(Vector3.UnitZ, direction), -1f, 1f);
                    var axis = Vector3.Cross(Vector3.UnitZ, direction);
                    var rotation = axis.LengthSquared() < 1e-12f
                        ? (dot < 0 ? Quaternion.CreateFromAxisAngle(Vector3.UnitX, MathF.PI) : Quaternion.Identity)
                        : Quaternion.CreateFromAxisAngle(Vector3.Normalize(axis), MathF.Acos(dot));
                    var segment = Matrix4x4.CreateFromQuaternion(rotation);
                    segment.Translation = (start + end) * 0.5f;
                    spec.Local = segment * spec.Local;
                }
                else
                {
                    var totalHeight = Param("height");
                    if (totalHeight < 2 * spec.Radius)
                        throw new ParseException($"capsule height at {Line(e)} is smaller than its diameter");
                    spec.Height = totalHeight - 2 * spec.Radius;
                }
                break;
            case "box":
                spec.Kind = PrimitiveKind.Box;
                var half = kindElement.Elements().Concat(e.Elements()).FirstOrDefault(c => Name(c) is "half-extents" or "halfExtents");
                if (half == null)
                    throw new ParseException($"box at {Line(e)} requires half-extents");
                spec.HalfExtents = Vector(half);
                break;
            default:
                throw new ParseException($"unknown shape type '{kindName}' at {Line(e)}");
        }
        if ((spec.Kind != PrimitiveKind.Box && spec.Radius <= 0) ||
            (spec.Kind == PrimitiveKind.Capsule && spec.Height < 0) ||
            (spec.Kind == PrimitiveKind.Box && (spec.HalfExtents.X <= 0 || spec.HalfExtents.Y <= 0 || spec.HalfExtents.Z <= 0)))
            throw new ParseException($"invalid shape dimensions at {Line(e)}");
        return spec;
    }

    void ReadGenericTemplate(XElement e, GenericTemplate t)
    {
        foreach (var c in e.Elements())
        {
            if (ReadFrame(c, t))
                continue;
            switch (Name(c))
            {
                case "enableLinearSprings": t.LinearSprings = Bool(c); break;
                case "enableAngularSprings": t.AngularSprings = Bool(c); break;
                case "linearStiffnessLimited": t.LinearStiffnessLimited = Bool(c); break;
                case "angularStiffnessLimited": t.AngularStiffnessLimited = Bool(c); break;
                case "springDampingLimited": t.SpringDampingLimited = Bool(c); break;
                case "linearNonHookeanDamping": t.LinearNonHookeanDamping = Vector(c); break;
                case "angularNonHookeanDamping": t.AngularNonHookeanDamping = Vector(c); break;
                case "linearNonHookeanStiffness": t.LinearNonHookeanStiffness = Vector(c); break;
                case "angularNonHookeanStiffness": t.AngularNonHookeanStiffness = Vector(c); break;
                case "linearMotors": t.LinearMotors = Bool(c); break;
                case "angularMotors": t.AngularMotors = Bool(c); break;
                case "linearServoMotors": t.LinearServo = Bool(c); break;
                case "angularServoMotors": t.AngularServo = Bool(c); break;
                case "linearTargetVelocity": t.LinearTargetVelocity = Vector(c); break;
                case "angularTargetVelocity": t.AngularTargetVelocity = Vector(c); break;
                case "linearMaxMotorForce": t.LinearMaxMotorForce = Vector(c); break;
                case "angularMaxMotorForce": t.AngularMaxMotorForce = Vector(c); break;
                case "stopERP": t.StopErp = Float(c); break;
                case "stopCFM": t.StopCfm = Float(c); break;
                case "motorERP": t.MotorErp = Float(c); break;
                case "motorCFM": t.MotorCfm = Float(c); break;
                case "useLinearReferenceFrameA": t.UseLinearReferenceFrameA = Bool(c); break;
                case "linearLowerLimit": t.LinearLower = Vector(c); break;
                case "linearUpperLimit": t.LinearUpper = Vector(c); break;
                case "angularLowerLimit": t.AngularLower = Vector(c); break;
                case "angularUpperLimit": t.AngularUpper = Vector(c); break;
                case "linearStiffness": t.LinearStiffness = Vector(c); break;
                case "angularStiffness": t.AngularStiffness = Vector(c); break;
                case "linearDamping": t.LinearDamping = Vector(c); break;
                case "angularDamping": t.AngularDamping = Vector(c); break;
                case "linearEquilibrium": t.LinearEquilibrium = Vector(c); break;
                case "angularEquilibrium": t.AngularEquilibrium = Vector(c); break;
                case "linearBounce": _ = Vector(c); break;
                case "angularBounce": t.AngularBounce = Vector(c); break;
                default: Warn(c, $"unknown element <{Name(c)}> skipped"); break;
            }
        }
    }

    bool ReadFrame(XElement c, GenericTemplate t)
    {
        switch (Name(c))
        {
            case "frameInA":
                t.FrameType = FrameType.InA;
                t.Frame = Transform(c);
                return true;
            case "frameInB":
                t.FrameType = FrameType.InB;
                t.Frame = Transform(c);
                return true;
            case "frameInLerp":
                t.FrameType = FrameType.InLerp;
                foreach (var child in c.Elements())
                {
                    if (Name(child) == "translationLerp") t.TranslationLerp = Float(child);
                    else if (Name(child) == "rotationLerp") t.RotationLerp = Float(child);
                    else Warn(child, $"unknown element <{Name(child)}> skipped");
                }
                return true;
            default:
                return false;
        }
    }

    // ---- document ----

    void ReadSystem(XElement root)
    {
        foreach (var e in root.Elements())
        {
            switch (Name(e))
            {
                case "bone" or "rigid-body":
                    ReadBone(e);
                    break;
                case "bone-default" or "rigid-body-default":
                    {
                        var template = BoneDefault(Optional(e, "extends")).Clone();
                        ReadBoneTemplate(e, template);
                        boneTemplates[Optional(e, "name")] = template;
                        break;
                    }
                case "per-vertex-shape" or "per-triangle-shape":
                    ReadMeshShape(e, Name(e) == "per-triangle-shape");
                    break;
                case "constraint-group":
                    foreach (var child in e.Elements())
                        if (!ReadConstraintElement(child))
                            Warn(child, $"unknown element <{Name(child)}> skipped");
                    break;
                case "shape":
                    break;
                default:
                    if (!ReadConstraintElement(e))
                        Warn(e, $"unknown element <{Name(e)}> skipped");
                    break;
            }
        }
    }

    void ReadMeshShape(XElement e, bool triangles)
    {
        var name = Attribute(e, "name");
        var sources = manager.FindMeshes(name).ToArray();
        if (sources.Length == 0)
        {
            Warn(e, $"shape {name}: no matching skinned NIF mesh; collision shape skipped");
            return;
        }
        foreach (var source in sources)
        {
            var bones = new SmpBone[source.BoneNames.Length];
            bool missing = false;
            for (int i = 0; i < bones.Length; i++)
            {
                var boneName = source.BoneNames[i];
                var bone = manager.Find(boneName) ?? CreateBone(boneName, boneName, "", null, e);
                if (bone == null)
                {
                    Warn(e, $"shape {name}: skin bone {boneName} is missing; collision shape skipped");
                    missing = true;
                    break;
                }
                bones[i] = bone;
            }
            if (missing) continue;
            var shape = new SmpMeshShape
            {
                System = system + "|" + manager.MeshScope,
                Mesh = source,
                Bones = bones,
                Triangles = triangles,
                Thresholds = new float[bones.Length],
            };
            foreach (var child in e.Elements())
            {
                var text = Text(child).Trim();
                switch (Name(child))
                {
                    case "margin":
                        shape.Margin = Float(child);
                        if (shape.Margin < 0) throw new ParseException($"negative margin at {Line(child)}");
                        break;
                    case "penetration" or "prenetration" when triangles:
                        shape.Penetration = Float(child);
                        break;
                    case "shared":
                        if (text is not ("public" or "private" or "internal" or "external"))
                            throw new ParseException($"invalid shared value '{text}' at {Line(child)}");
                        shape.Shared = text;
                        break;
                    case "tag": shape.Tags.Add(text); break;
                    case "can-collide-with-tag": shape.AllowTags.Add(text); break;
                    case "no-collide-with-tag": shape.DenyTags.Add(text); break;
                    case "can-collide-with-bone": shape.AllowBones.Add(text); break;
                    case "no-collide-with-bone": shape.DenyBones.Add(text); break;
                    case "disable-tag": shape.DisableTag = text; break;
                    case "disable-priority":
                        if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out shape.DisablePriority))
                            throw new ParseException($"invalid disable priority at {Line(child)}");
                        break;
                    case "weight-threshold":
                        var target = Attribute(child, "bone");
                        var threshold = Float(child);
                        for (int i = 0; i < bones.Length; i++)
                            if (bones[i].Name.Equals(target, StringComparison.OrdinalIgnoreCase))
                                shape.Thresholds[i] = threshold;
                        break;
                    case "priority": Warn(child, "<priority> is deprecated and ignored"); break;
                    default: Warn(child, $"unknown element <{Name(child)}> skipped"); break;
                }
            }
            shape.Build();
            manager.AddMeshShape(shape);
            createdBodies++;
        }
    }

    bool ReadConstraintElement(XElement e)
    {
        switch (Name(e))
        {
            case "generic-constraint":
                ReadGeneric(e);
                return true;
            case "generic-constraint-default":
                {
                    var template = GenericDefault(Optional(e, "extends")).Clone();
                    ReadGenericTemplate(e, template);
                    genericTemplates[Optional(e, "name")] = template;
                    return true;
                }
            case "stiffspring-constraint" or "conetwist-constraint":
                unsupportedConstraints++;
                return true;
            case "stiffspring-constraint-default" or "conetwist-constraint-default":
                return true;
            default:
                return false;
        }
    }

    void ReadBone(XElement e)
    {
        var name = Attribute(e, "name");
        var target = Optional(e, "bone");
        if (target.Length == 0)
            target = name;
        var existing = manager.Find(name);
        if (existing != null)
        {
            var template = BoneDefault(Optional(e, "template")).Clone();
            ReadBoneTemplate(e, template);
            if (manager.AddShapeToAnchor(existing, template))
                createdBodies++;
            else if (!existing.Kinematic || template.Shape != null)
                Warn(e, $"bone {name} already exists; second definition skipped");
            return;
        }
        CreateBone(name, target, Optional(e, "template"), e, e);
    }

    SmpBone? CreateBone(string name, string target, string templateName, XElement? read, XElement at)
    {
        if (!manager.IsCandidate(target))
        {
            Warn(at, $"node {target} is not a candidate on this actor; no bone created");
            return null;
        }
        var template = BoneDefault(templateName).Clone();
        if (read != null)
            ReadBoneTemplate(read, template);
        if (template.Mass < 0 || (template.MotionType == "dynamic" && template.Mass <= 0))
            throw new ParseException($"bone {name}: dynamic mass must be positive");
        var bone = manager.CreateBone(name, target, template);
        if (bone != null)
        {
            createdBodies++;
        }
        return bone;
    }

    bool FindBones(XElement e, out SmpBone a, out SmpBone b)
    {
        var nameA = Attribute(e, "bodyA");
        var nameB = Attribute(e, "bodyB");
        a = b = null!;
        var first = manager.Find(nameA);
        var second = manager.Find(nameB);
        if (first == null || second == null || ReferenceEquals(first, second) || (first.Kinematic && second.Kinematic))
        {
            skippedConstraints++;
            return false;
        }
        a = first;
        b = second;
        return true;
    }

    // Resolves the constraint frame in A's and B's local space from the bones' reference transforms.
    static void CalcFrame(GenericTemplate t, in Matrix4x4 trA, in Matrix4x4 trB, out Matrix4x4 frameA, out Matrix4x4 frameB)
    {
        Matrix4x4 world;
        switch (t.FrameType)
        {
            case FrameType.InA:
                frameA = t.Frame;
                world = t.Frame * trA;
                frameB = SmpMath.Rigid(world * SmpMath.Invert(trB));
                break;
            case FrameType.InB:
                frameB = t.Frame;
                world = t.Frame * trB;
                frameA = SmpMath.Rigid(world * SmpMath.Invert(trA));
                break;
            default:
                var q = Quaternion.Slerp(
                    Quaternion.CreateFromRotationMatrix(trA), Quaternion.CreateFromRotationMatrix(trB), t.RotationLerp);
                world = Matrix4x4.CreateFromQuaternion(Quaternion.Normalize(q));
                world.Translation = Vector3.Lerp(trA.Translation, trB.Translation, t.TranslationLerp);
                frameA = SmpMath.Rigid(world * SmpMath.Invert(trA));
                frameB = SmpMath.Rigid(world * SmpMath.Invert(trB));
                break;
        }
    }

    void ReadGeneric(XElement e)
    {
        if (!FindBones(e, out var a, out var b))
            return;
        var template = GenericDefault(Optional(e, "template")).Clone();
        ReadGenericTemplate(e, template);
        CalcFrame(template, a.Current, b.Current, out var frameA, out var frameB);
        // useLinearReferenceFrameA swaps which bone is Bullet's body A.
        if (template.UseLinearReferenceFrameA)
        {
            (a, b) = (b, a);
            (frameA, frameB) = (frameB, frameA);
        }
        if (template.LinearNonHookeanDamping != Vector3.Zero || template.AngularNonHookeanDamping != Vector3.Zero ||
            template.LinearNonHookeanStiffness != Vector3.Zero || template.AngularNonHookeanStiffness != Vector3.Zero)
            Warn(e, "non-Hookean spring terms need FSMP's modified Bullet and are not simulated; the spring stays linear");
        manager.AddGeneric(a, b, frameA, frameB, template);
    }
}
