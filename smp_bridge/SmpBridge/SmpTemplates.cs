using System.Numerics;

namespace SmpBridge;

internal enum PrimitiveKind { Sphere, Capsule, Box }

internal sealed class PrimitiveSpec
{
    public PrimitiveKind Kind;
    public float Radius;
    public float Height;
    public Vector3 HalfExtents;
    public Matrix4x4 Local = Matrix4x4.Identity;
}

internal sealed class BoneTemplate
{
    public float Mass;
    public string MotionType = "";
    public float LinearDamping;
    public float AngularDamping;
    public float Friction = 0.5f;
    public float RollingFriction;
    public float Restitution;
    public float MarginMultiplier = 1f;
    public float GravityFactor = 1f;
    public Vector3 Inertia;
    public Matrix4x4 CenterOfMass = Matrix4x4.Identity;
    public PrimitiveSpec? Shape;

    public BoneTemplate Clone() => (BoneTemplate)MemberwiseClone();
}

internal enum FrameType { InA, InB, InLerp }

// <generic-constraint> parameters. Defaults match FSMP: limits with lower > upper leave the axis free.
internal sealed class GenericTemplate
{
    public FrameType FrameType = FrameType.InB;
    public bool UseLinearReferenceFrameA;
    public Matrix4x4 Frame = Matrix4x4.Identity;
    public float TranslationLerp;
    public float RotationLerp;

    public Vector3 LinearLower = Vector3.One;
    public Vector3 LinearUpper = -Vector3.One;
    public Vector3 AngularLower = Vector3.One;
    public Vector3 AngularUpper = -Vector3.One;
    public Vector3 LinearStiffness, AngularStiffness;
    public Vector3 LinearDamping, AngularDamping;
    public Vector3 LinearEquilibrium, AngularEquilibrium;
    public Vector3 AngularBounce;
    public bool LinearSprings = true, AngularSprings = true;
    public bool LinearStiffnessLimited = true, AngularStiffnessLimited = true, SpringDampingLimited = true;
    public bool LinearMotors, AngularMotors, LinearServo, AngularServo;
    public Vector3 LinearTargetVelocity, AngularTargetVelocity;
    public Vector3 LinearMaxMotorForce, AngularMaxMotorForce;
    public Vector3 LinearNonHookeanDamping, AngularNonHookeanDamping;
    public Vector3 LinearNonHookeanStiffness, AngularNonHookeanStiffness;
    public float MotorErp = 0.9f, MotorCfm, StopErp = 0.2f, StopCfm;

    public GenericTemplate Clone() => (GenericTemplate)MemberwiseClone();
}
