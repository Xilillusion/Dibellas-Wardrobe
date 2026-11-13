using System.Numerics;
using BMatrix = BulletSharp.Math.Matrix;
using BVector = BulletSharp.Math.Vector3;

namespace SmpBridge;

// All transforms inside the engine are row-vector matrices (v' = v * M, translation in M41..M43), the layout shared
// by System.Numerics and BulletSharp. Composition "a then b" is therefore a * b.
//
// The Python side exchanges column-vector 4x4 float32 arrays (v' = M * v, translation in elements 3, 7 and 11 of the
// C-ordered buffer). Reading such a buffer as a row-vector matrix is a transpose, folded into the accessors below so
// that neither side has to copy or transpose per frame.
internal static unsafe class SmpMath
{
    public static Matrix4x4 ReadColumnMajor(float* p) => new(
        p[0], p[4], p[8], p[12],
        p[1], p[5], p[9], p[13],
        p[2], p[6], p[10], p[14],
        p[3], p[7], p[11], p[15]);

    public static void WriteColumnMajor(in Matrix4x4 m, float* p)
    {
        p[0] = m.M11; p[4] = m.M12; p[8] = m.M13; p[12] = m.M14;
        p[1] = m.M21; p[5] = m.M22; p[9] = m.M23; p[13] = m.M24;
        p[2] = m.M31; p[6] = m.M32; p[10] = m.M33; p[14] = m.M34;
        p[3] = m.M41; p[7] = m.M42; p[11] = m.M43; p[15] = m.M44;
    }

    public static BMatrix ToBullet(in Matrix4x4 m) => new(
        m.M11, m.M12, m.M13, m.M14,
        m.M21, m.M22, m.M23, m.M24,
        m.M31, m.M32, m.M33, m.M34,
        m.M41, m.M42, m.M43, m.M44);

    public static Matrix4x4 FromBullet(in BMatrix m) => new(
        m.M11, m.M12, m.M13, m.M14,
        m.M21, m.M22, m.M23, m.M24,
        m.M31, m.M32, m.M33, m.M34,
        m.M41, m.M42, m.M43, m.M44);

    public static BVector ToBullet(Vector3 v) => new(v.X, v.Y, v.Z);

    public static Matrix4x4 Invert(in Matrix4x4 m) => Matrix4x4.Invert(m, out var inverse) ? inverse : Matrix4x4.Identity;

    // Rotation and translation only: scale and shear are dropped, as FSMP does when it hands a pose to Bullet.
    public static Matrix4x4 Rigid(in Matrix4x4 m)
    {
        var q = Quaternion.CreateFromRotationMatrix(m);
        q = q.LengthSquared() > 1e-12f ? Quaternion.Normalize(q) : Quaternion.Identity;
        var rigid = Matrix4x4.CreateFromQuaternion(q);
        rigid.Translation = m.Translation;
        return rigid;
    }

    public static bool IsFinite(in Matrix4x4 m) =>
        float.IsFinite(m.M11) && float.IsFinite(m.M12) && float.IsFinite(m.M13) &&
        float.IsFinite(m.M21) && float.IsFinite(m.M22) && float.IsFinite(m.M23) &&
        float.IsFinite(m.M31) && float.IsFinite(m.M32) && float.IsFinite(m.M33) &&
        float.IsFinite(m.M41) && float.IsFinite(m.M42) && float.IsFinite(m.M43);

    public static float Component(Vector3 v, int index) => index switch { 0 => v.X, 1 => v.Y, _ => v.Z };
}
