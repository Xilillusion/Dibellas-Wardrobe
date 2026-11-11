"""
Headless HKX Animation & Skeleton Pose Engine for Dibella's Wardrobe.

Provides zero-Blender-dependency (`bpy` / `mathutils` free) loading of `skeleton_female.nif`
and Skyrim `.hkx` animations (`hkaSplineCompressedAnimation` / `hkaInterleavedUncompressedAnimation`),
real-time keyframe interpolation (quaternion Slerp + translation/scale Lerp), and hierarchical
bone pose matrix evaluation for GPU Linear Blend Skinning (LBS).
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("dibellas_wardrobe.anim_engine")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NIFLY_DIR = _REPO_ROOT / "io_scene_nifly"
_HKX_DIR = _NIFLY_DIR / "hkx"
_RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", _REPO_ROOT))

# Import the bundled headless NIF and HKX modules directly.
if str(_NIFLY_DIR) not in sys.path:
    sys.path.insert(0, str(_NIFLY_DIR))
if str(_HKX_DIR) not in sys.path:
    sys.path.insert(0, str(_HKX_DIR))

import anim_fo4  # type: ignore[import-not-found]  # noqa: E402
import anim_skyrim  # type: ignore[import-not-found]  # noqa: E402
from pyn.pynifly import NifFile  # type: ignore[import-not-found]  # noqa: E402

DEFAULT_SKELETON_PATH = _RESOURCE_ROOT / "assets" / "skeleton_female.nif"

# Canonical Skyrim SE / XPMSSE human Havok bone table (indices 0..98 = vanilla 99 bones,
# indices 99..125 = XPMSSE extended bones). Used when an HKX animation omits explicit
# track-to-bone binding indices or annotation track names.
SKYRIM_HAVOK_BONES: Tuple[str, ...] = (
    "NPC Root [Root]",
    "x_NPC LookNode [Look]",
    "x_NPC Translate [Pos ]",
    "x_NPC Rotate [Rot ]",
    "NPC COM [COM ]",
    "NPC Pelvis [Pelv]",
    "NPC L Thigh [LThg]",
    "NPC L Calf [LClf]",
    "NPC L Foot [Lft ]",
    "NPC R Thigh [RThg]",
    "NPC R Calf [RClf]",
    "NPC R Foot [Rft ]",
    "SkirtRBone01",
    "SkirtRBone02",
    "SkirtRBone03",
    "SkirtLBone01",
    "SkirtLBone02",
    "SkirtLBone03",
    "SkirtBBone01",
    "SkirtBBone02",
    "SkirtBBone03",
    "SkirtFBone01",
    "SkirtFBone02",
    "SkirtFBone03",
    "NPC Spine [Spn0]",
    "NPC Spine1 [Spn1]",
    "NPC Spine2 [Spn2]",
    "NPC L Clavicle [LClv]",
    "NPC L UpperArm [LUar]",
    "NPC L Forearm [LLar]",
    "NPC R Clavicle [RClv]",
    "NPC R UpperArm [RUar]",
    "NPC R Forearm [RLar]",
    "AnimObjectA",
    "AnimObjectB",
    "NPC Neck [Neck]",
    "NPC Head [Head]",
    "NPCEyeBone",
    "NPC L Hand [LHnd]",
    "NPC R Hand [RHnd]",
    "AnimObjectL",
    "AnimObjectR",
    "Shield",
    "Weapon",
    "NPC L Pauldron",
    "NPC R Pauldron",
    "MagicEffectsNode",
    "NPC L MagicNode [LMag]",
    "NPC R MagicNode [RMag]",
    "NPC Head MagicNode [Hmag]",
    "NPC L Toe0 [LToe]",
    "NPC R Toe0 [RToe]",
    "NPC L ForearmTwist1 [LLt1]",
    "NPC L ForearmTwist2 [LLt2]",
    "NPC L UpperarmTwist1 [LUt1]",
    "NPC L UpperarmTwist2 [LUt2]",
    "NPC R ForearmTwist1 [RLt1]",
    "NPC R ForearmTwist2 [RLt2]",
    "NPC R UpperarmTwist1 [RUt1]",
    "NPC R UpperarmTwist2 [RUt2]",
    "Quiver",
    "WeaponAxe",
    "WeaponBack",
    "WeaponBow",
    "WeaponDagger",
    "WeaponMace",
    "WeaponSword",
    "NPC L Finger00 [LF00]",
    "NPC L Finger01 [LF01]",
    "NPC L Finger02 [LF02]",
    "NPC L Finger10 [LF10]",
    "NPC L Finger11 [LF11]",
    "NPC L Finger12 [LF12]",
    "NPC L Finger20 [LF20]",
    "NPC L Finger21 [LF21]",
    "NPC L Finger22 [LF22]",
    "NPC L Finger30 [LF30]",
    "NPC L Finger31 [LF31]",
    "NPC L Finger32 [LF32]",
    "NPC L Finger40 [LF40]",
    "NPC L Finger41 [LF41]",
    "NPC L Finger42 [LF42]",
    "NPC R Finger00 [RF00]",
    "NPC R Finger01 [RF01]",
    "NPC R Finger02 [RF02]",
    "NPC R Finger10 [RF10]",
    "NPC R Finger11 [RF11]",
    "NPC R Finger12 [RF12]",
    "NPC R Finger20 [RF20]",
    "NPC R Finger21 [RF21]",
    "NPC R Finger22 [RF22]",
    "NPC R Finger30 [RF30]",
    "NPC R Finger31 [RF31]",
    "NPC R Finger32 [RF32]",
    "NPC R Finger40 [RF40]",
    "NPC R Finger41 [RF41]",
    "NPC R Finger42 [RF42]",
    "Camera3rd [Cam3]",
    "Camera Control",
    "NPC L Item01",
    "NPC L Item02",
    "NPC L Item03",
    "NPC R Item01",
    "NPC R Item02",
    "NPC R Item03",
    "LeftWing1",
    "LeftWing2",
    "LeftWing3",
    "LeftWing4",
    "LeftWing5",
    "RightWing1",
    "RightWing2",
    "RightWing3",
    "RightWing4",
    "RightWing5",
    "Belly",
    "NPC Anus Deep1",
    "NPC RB Anus1",
    "NPC LB Anus1",
    "NPC LT Anus1",
    "NPC RT Anus1",
    "NPC Anus Deep2",
    "NPC RB Anus2",
    "NPC LB Anus2",
    "NPC LT Anus2",
    "NPC RT Anus2",
)

# Supplemental XPMSSE / CBBE 3BA bone nodes not present in the base 621-node skeleton_female.nif
_SUPPLEMENTAL_3BA_NODES: Tuple[
    Tuple[str, str, Tuple[float, float, float], Tuple[Tuple[float, float, float], ...]],
    ...,
] = (
    ("x_NPC LookNode [Look]", "NPC Root [Root]", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("x_NPC Translate [Pos ]", "NPC Root [Root]", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("x_NPC Rotate [Rot ]", "NPC Root [Root]", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("L Breast00", "CME L PreBreast", (1.66853, -1.306802, -3.335846), ((1.0, 0.0, 0.0), (0.0, 0.991035, -0.133615), (0.0, 0.133615, 0.991035))),
    ("L Breast01", "L Breast00", (-1.120057, 3.090759, -0.538383), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("L Breast02", "L Breast01", (-1.33002, 3.67019, -0.52449), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("L Breast03", "L Breast02", (-1.314233, 3.626562, 0.583305), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("L Breast04", "L Breast03", (-1.314236, 3.626561, 0.583298), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("R Breast00", "CME R PreBreast", (-1.66855, -1.30679, -3.335846), ((1.0, 0.0, 0.0), (0.0, 0.991035, -0.133614), (0.0, 0.133614, 0.991035))),
    ("R Breast01", "R Breast00", (1.120055, 3.090774, -0.538383), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("R Breast02", "R Breast01", (1.330017, 3.670176, -0.524536), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("R Breast03", "R Breast02", (1.314238, 3.626567, 0.583328), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("R Breast04", "R Breast03", (1.314241, 3.62657, 0.583313), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("Anal", "CME Pelvis [Pelv]", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("NPC Anus Deep1", "Anal", (0.0, -4.618244, 1.628693), ((0.080711, -0.993214, -0.083735), (-0.995932, -0.083738, 0.033283), (-0.040069, 0.080709, -0.995932))),
    ("NPC Anus Deep2", "NPC Anus Deep1", (0.288759, 0.0, 0.924042), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("NPC LT Anus1", "Anal", (0.0, -5.757988, -0.103302), ((0.0, -0.993747, -0.11165), (-0.501634, 0.096586, -0.859671), (0.86508, 0.056008, -0.498497))),
    ("NPC LT Anus2", "NPC LT Anus1", (-0.75322, -0.048635, 0.432873), ((1.0, 0.0, 0.0), (0.0, 0.999978, -0.006794), (0.0, 0.006794, 0.999977))),
    ("NPC RT Anus1", "Anal", (0.0, -5.065755, -0.103302), ((0.0, -0.993747, 0.111651), (-0.504754, -0.096384, -0.857866), (0.863263, -0.056355, -0.501598))),
    ("NPC RT Anus2", "NPC RT Anus1", (-0.75322, 0.048634, 0.432854), ((1.0, 0.0, 0.0), (0.0, 0.999977, 0.006796), (0.0, -0.006796, 0.999977))),
    ("NPC LB Anus1", "Anal", (-0.750808, -5.412185, -0.103745), ((0.0, -0.990514, -0.137409), (0.0, 0.137409, -0.990514), (1.0, 0.0, 0.0))),
    ("NPC LB Anus2", "NPC LB Anus1", (-0.753212, -0.05986, 0.431488), ((1.0, 0.0, 0.0), (0.0, 0.999948, -0.010152), (0.0, 0.010152, 0.999948))),
    ("NPC RB Anus1", "Anal", (0.750812, -5.412186, -0.103745), ((0.0, -0.990514, 0.13741), (0.0, -0.13741, -0.990514), (1.0, 0.0, 0.0))),
    ("NPC RB Anus2", "NPC RB Anus1", (-0.753212, 0.059857, 0.431465), ((1.0, 0.0, 0.0), (0.0, 0.999949, 0.010153), (0.0, -0.010153, 0.999949))),
    ("NPC L Item01", "NPC Root [Root]", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("NPC L Item02", "NPC L Item01", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("NPC L Item03", "NPC L Item02", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("NPC R Item01", "NPC Root [Root]", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("NPC R Item02", "NPC R Item01", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("NPC R Item03", "NPC R Item02", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ("CamSpin", "Camera3rd [Cam3]", (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
)


# ------------------------------------------------------------------
# Pure NumPy Quaternion & Transform Utilities (xyzw convention)
# ------------------------------------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize an `[x, y, z, w]` quaternion."""
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two `[x, y, z, w]` quaternions (`q1 * q2`)."""
    x1, y1, z1, w1 = float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])
    x2, y2, z2, w2 = float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two `[x, y, z, w]` unit quaternions."""
    if alpha <= 0.0:
        return q0
    if alpha >= 1.0:
        return q1

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        blended = q0 + alpha * (q1 - q0)
        return quat_normalize(blended)

    theta_0 = math.acos(min(1.0, max(-1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    if abs(sin_theta_0) < 1e-6:
        return q0

    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quat_normalize(s0 * q0 + s1 * q1)


def quat_to_mat3(q: np.ndarray) -> np.ndarray:
    """Convert an `[x, y, z, w]` unit quaternion into a `(3, 3)` float32 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x * x + y * y + z * z + w * w
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def mat3_to_quat(m: np.ndarray) -> np.ndarray:
    """Convert a `(3, 3)` rotation matrix into an `[x, y, z, w]` unit quaternion."""
    m00, m01, m02 = float(m[0, 0]), float(m[0, 1]), float(m[0, 2])
    m10, m11, m12 = float(m[1, 0]), float(m[1, 1]), float(m[1, 2])
    m20, m21, m22 = float(m[2, 0]), float(m[2, 1]), float(m[2, 2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w], dtype=np.float32))


# ------------------------------------------------------------------
# Skeleton & Animation Data Structures
# ------------------------------------------------------------------

@dataclass
class SkeletonNode:
    """Single node in the character skeleton hierarchy."""

    name: str
    index: int
    parent_index: int
    parent_name: Optional[str]
    local_translation: np.ndarray  # (3,) float32
    local_rotation: np.ndarray     # (3, 3) float32
    local_quat: np.ndarray         # (4,) float32 [x, y, z, w]
    local_scale: float
    local_matrix: np.ndarray       # (4, 4) float32
    global_bind_matrix: np.ndarray # (4, 4) float32
    inverse_bind_matrix: np.ndarray  # (4, 4) float32


@dataclass
class BoneTrack:
    """Decompressed keyframe arrays for a single bone track."""

    bone_name: str
    bone_index: int
    translations: Optional[np.ndarray] = None  # (F, 3) float32
    rotations: Optional[np.ndarray] = None     # (F, 4) float32 [x, y, z, w]
    scales: Optional[np.ndarray] = None        # (F, 3) float32


@dataclass
class LoadedAnimation:
    """Parsed HKX animation clip ready for real-time evaluation."""

    filepath: Path
    duration: float
    num_frames: int
    num_tracks: int
    frame_duration: float
    fps: float
    blend_hint: int
    tracks_by_bone_index: Dict[int, BoneTrack] = field(default_factory=dict)
    annotations: List[Tuple[float, str]] = field(default_factory=list)


class AnimEngine:
    """
    Headless skeleton and `.hkx` animation evaluation engine.

    Loads `skeleton_female.nif` (plus supplemental 3BA/XPMSSE bone nodes), parses Skyrim
    `.hkx` animation clips, and evaluates `{bone_name: 4x4_global_matrix}` at any timestamp `t`.
    """

    def __init__(self, skeleton_path: Optional[Path | str] = None) -> None:
        self.skeleton_path: Optional[Path] = None
        self.nodes: List[SkeletonNode] = []
        self.node_by_name: Dict[str, SkeletonNode] = {}
        self._casefold_to_name: Dict[str, str] = {}
        self.bind_pose_matrices: Dict[str, np.ndarray] = {}
        self.active_animation: Optional[LoadedAnimation] = None
        self._fallback_parent_cache: Dict[str, Optional[str]] = {}

        target_skel = Path(skeleton_path) if skeleton_path else DEFAULT_SKELETON_PATH
        if target_skel.is_file():
            self.load_skeleton(target_skel)

    def load_skeleton(self, skeleton_path: Path | str = DEFAULT_SKELETON_PATH) -> Dict[str, np.ndarray]:
        """
        Load base bone nodes from `skeleton_female.nif` and compute bind-pose matrices.
        Returns `{bone_name: 4x4_global_bind_matrix}`.
        """
        path = Path(skeleton_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Skeleton NIF not found: {path}")

        raw_nodes: Dict[str, Tuple[Optional[str], np.ndarray, np.ndarray, float]] = {}
        with NifFile(str(path)) as nif:
            for name, node in nif.nodes.items():
                parent_name = node.parent.name if node.parent is not None else None
                t = np.array(
                    [
                        float(node.transform.translation[0]),
                        float(node.transform.translation[1]),
                        float(node.transform.translation[2]),
                    ],
                    dtype=np.float32,
                )
                r = np.array(
                    [
                        list(node.transform.rotation[0]),
                        list(node.transform.rotation[1]),
                        list(node.transform.rotation[2]),
                    ],
                    dtype=np.float32,
                )
                s = float(node.transform.scale)
                raw_nodes[name] = (parent_name, t, r, s)

        # Add supplemental XPMSSE / 3BA nodes if not present in the base skeleton NIF
        for name, parent_name, t_tuple, r_tuple in _SUPPLEMENTAL_3BA_NODES:
            if name not in raw_nodes and parent_name in raw_nodes:
                t = np.array(t_tuple, dtype=np.float32)
                r = np.array(r_tuple, dtype=np.float32)
                raw_nodes[name] = (parent_name, t, r, 1.0)

        # Topologically sort nodes so every parent precedes its children
        ordered_names: List[str] = []
        visited: set[str] = set()

        def visit(node_name: str) -> None:
            if node_name in visited or node_name not in raw_nodes:
                return
            visited.add(node_name)
            p_name = raw_nodes[node_name][0]
            if p_name and p_name in raw_nodes:
                visit(p_name)
            ordered_names.append(node_name)

        for name in raw_nodes:
            visit(name)

        self.skeleton_path = path
        self.nodes.clear()
        self.node_by_name.clear()
        self._casefold_to_name.clear()
        self.bind_pose_matrices.clear()
        self._fallback_parent_cache.clear()

        name_to_idx: Dict[str, int] = {}
        for idx, name in enumerate(ordered_names):
            parent_name, t, r, s = raw_nodes[name]
            parent_idx = name_to_idx.get(parent_name, -1) if parent_name else -1

            local_mat = np.eye(4, dtype=np.float32)
            local_mat[:3, :3] = r
            local_mat[:3, 3] = t
            local_q = mat3_to_quat(r)

            if parent_idx >= 0:
                global_mat = self.nodes[parent_idx].global_bind_matrix @ local_mat
            else:
                global_mat = local_mat.copy()

            try:
                inv_bind = np.linalg.inv(global_mat).astype(np.float32)
            except np.linalg.LinAlgError:
                inv_bind = np.eye(4, dtype=np.float32)

            s_node = SkeletonNode(
                name=name,
                index=idx,
                parent_index=parent_idx,
                parent_name=parent_name if parent_idx >= 0 else None,
                local_translation=t,
                local_rotation=r,
                local_quat=local_q,
                local_scale=s,
                local_matrix=local_mat,
                global_bind_matrix=global_mat,
                inverse_bind_matrix=inv_bind,
            )
            self.nodes.append(s_node)
            self.node_by_name[name] = s_node
            self._casefold_to_name[name.casefold()] = name
            self.bind_pose_matrices[name] = global_mat
            name_to_idx[name] = idx

        return dict(self.bind_pose_matrices)

    def resolve_bone_name(self, name: str) -> Optional[str]:
        """Resolve `name` (case-insensitively) to a canonical skeleton node name."""
        if name in self.node_by_name:
            return name
        return self._casefold_to_name.get(name.casefold())

    def find_fallback_ancestor(
        self, bone_name: str, bind_position: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        For a custom outfit/SMP bone not present in `skeleton_female.nif`, determine the
        most appropriate skeleton bone to inherit motion from.
        """
        resolved = self.resolve_bone_name(bone_name)
        if resolved is not None:
            return resolved
        if bone_name in self._fallback_parent_cache:
            return self._fallback_parent_cache[bone_name]

        low = bone_name.casefold()
        candidate: Optional[str] = None
        if "breast" in low:
            candidate = "CME L PreBreast" if ("l " in low or low.startswith("l")) else "CME R PreBreast"
        elif any(tok in low for tok in ("skirt", "pelv", "butt", "anus", "vagina", "clit", " labia")):
            candidate = "NPC Pelvis [Pelv]"
        elif any(tok in low for tok in ("hair", "head", "ear", "hat", "crown", "bang", "ponytail")):
            candidate = "NPC Head [Head]"
        elif "cape" in low or "cloak" in low or "wing" in low or " ribbon" in low:
            candidate = "NPC Spine2 [Spn2]"
        elif "spine" in low or "belly" in low:
            candidate = "NPC Spine1 [Spn1]"

        if candidate and candidate in self.node_by_name:
            self._fallback_parent_cache[bone_name] = candidate
            return candidate

        # Nearest major body bone by bind-pose position if provided
        if bind_position is not None and self.nodes:
            best_name = "NPC COM [COM ]" if "NPC COM [COM ]" in self.node_by_name else self.nodes[0].name
            best_dist = float("inf")
            for node in self.nodes:
                if not node.name.startswith("NPC "):
                    continue
                dist = float(np.linalg.norm(node.global_bind_matrix[:3, 3] - bind_position))
                if dist < best_dist:
                    best_dist = dist
                    best_name = node.name
            self._fallback_parent_cache[bone_name] = best_name
            return best_name

        fallback = "NPC COM [COM ]" if "NPC COM [COM ]" in self.node_by_name else (
            self.nodes[0].name if self.nodes else None
        )
        self._fallback_parent_cache[bone_name] = fallback
        return fallback

    def _map_animation_track_names(self, anim_data: anim_fo4.AnimationData) -> List[str]:
        """Resolve Havok track indices to skeleton bone names."""
        indices = list(anim_data.track_to_bone_indices or [])
        if indices:
            return [
                SKYRIM_HAVOK_BONES[idx] if 0 <= idx < len(SKYRIM_HAVOK_BONES) else ""
                for idx in indices
            ]

        if anim_data.bone_names and any(bool(name) for name in anim_data.bone_names):
            return list(anim_data.bone_names)

        return [
            SKYRIM_HAVOK_BONES[i] if i < len(SKYRIM_HAVOK_BONES) else ""
            for i in range(anim_data.num_tracks)
        ]

    def load_animation(self, hkx_path: Path | str) -> LoadedAnimation:
        """
        Parse a Skyrim `.hkx` animation file and prepare its keyframe tracks for evaluation.
        """
        if not self.nodes:
            self.load_skeleton(DEFAULT_SKELETON_PATH)

        path = Path(hkx_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"HKX animation file not found: {path}")

        try:
            anim_data = anim_skyrim.load_skyrim_animation(str(path))
        except Exception as skyrim_exc:
            try:
                anim_data = anim_fo4.load_fo4_animation(str(path))
            except Exception:
                raise ValueError(f"Could not parse HKX animation '{path.name}': {skyrim_exc}") from skyrim_exc

        if anim_data is None or anim_data.num_tracks <= 0 or not anim_data.tracks:
            raise ValueError(f"'{path.name}' does not contain readable animation tracks.")

        raw_track_names = self._map_animation_track_names(anim_data)
        tracks_by_bone_index: Dict[int, BoneTrack] = {}

        for track_idx, raw_track in enumerate(anim_data.tracks):
            if track_idx >= len(raw_track_names):
                break
            raw_name = raw_track_names[track_idx]
            if not raw_name:
                continue
            resolved_name = self.resolve_bone_name(raw_name)
            if resolved_name is None:
                continue
            node = self.node_by_name[resolved_name]

            trans_arr: Optional[np.ndarray] = None
            if raw_track.translations:
                trans_arr = np.asarray(raw_track.translations, dtype=np.float32)

            rot_arr: Optional[np.ndarray] = None
            if raw_track.rotations:
                rot_arr = np.asarray(raw_track.rotations, dtype=np.float32)
                norms = np.linalg.norm(rot_arr, axis=1, keepdims=True)
                valid = norms[:, 0] > 1e-8
                rot_arr[valid] /= norms[valid]
                rot_arr[~valid] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            scale_arr: Optional[np.ndarray] = None
            if raw_track.scales:
                scale_arr = np.asarray(raw_track.scales, dtype=np.float32)

            tracks_by_bone_index[node.index] = BoneTrack(
                bone_name=resolved_name,
                bone_index=node.index,
                translations=trans_arr,
                rotations=rot_arr,
                scales=scale_arr,
            )

        frame_duration = float(anim_data.frame_duration) if anim_data.frame_duration > 0 else (1.0 / 30.0)
        num_frames = max(1, int(anim_data.num_frames))
        duration = float(anim_data.duration)
        if duration <= 0.0 and num_frames > 1:
            duration = (num_frames - 1) * frame_duration
        fps = 1.0 / frame_duration if frame_duration > 0 else 30.0

        annotations = [
            (float(ann.time), str(ann.text))
            for ann in getattr(anim_data, "annotations", [])
            if getattr(ann, "text", "")
        ]

        loaded = LoadedAnimation(
            filepath=path,
            duration=max(0.0, duration),
            num_frames=num_frames,
            num_tracks=int(anim_data.num_tracks),
            frame_duration=frame_duration,
            fps=fps,
            blend_hint=int(getattr(anim_data, "blend_hint", 0)),
            tracks_by_bone_index=tracks_by_bone_index,
            annotations=annotations,
        )
        self.active_animation = loaded
        return loaded

    def clear_animation(self) -> None:
        """Unload the active animation and return to bind pose."""
        self.active_animation = None

    def get_bind_pose(self) -> Dict[str, np.ndarray]:
        """Return `{bone_name: 4x4_global_bind_matrix}` for the loaded skeleton."""
        return dict(self.bind_pose_matrices)

    @staticmethod
    def _sample_vec3_track(samples: np.ndarray, frame_pos: float) -> np.ndarray:
        count = len(samples)
        if count == 1 or frame_pos <= 0.0:
            return samples[0]
        max_idx = count - 1
        if frame_pos >= max_idx:
            return samples[max_idx]
        idx0 = int(frame_pos)
        idx1 = min(max_idx, idx0 + 1)
        alpha = float(frame_pos - idx0)
        if alpha <= 1e-6:
            return samples[idx0]
        return (1.0 - alpha) * samples[idx0] + alpha * samples[idx1]

    @staticmethod
    def _sample_quat_track(samples: np.ndarray, frame_pos: float) -> np.ndarray:
        count = len(samples)
        if count == 1 or frame_pos <= 0.0:
            return samples[0]
        max_idx = count - 1
        if frame_pos >= max_idx:
            return samples[max_idx]
        idx0 = int(frame_pos)
        idx1 = min(max_idx, idx0 + 1)
        alpha = float(frame_pos - idx0)
        if alpha <= 1e-6:
            return samples[idx0]
        return quat_slerp(samples[idx0], samples[idx1], alpha)

    def evaluate_pose(self, current_time: float) -> Dict[str, np.ndarray]:
        """
        Sample the active animation at `current_time` (seconds), accumulate bone transforms
        down the skeleton hierarchy, and return `{bone_name: 4x4_global_matrix}`.
        """
        if not self.nodes:
            self.load_skeleton(DEFAULT_SKELETON_PATH)

        anim = self.active_animation
        if anim is None:
            return dict(self.bind_pose_matrices)

        if anim.duration > 0.0 and anim.num_frames > 1:
            clamped_t = max(0.0, min(float(current_time), anim.duration))
            frame_pos = (clamped_t / anim.duration) * float(anim.num_frames - 1)
        else:
            frame_pos = 0.0

        is_additive = anim.blend_hint != 0
        tracks = anim.tracks_by_bone_index
        num_nodes = len(self.nodes)
        global_poses: List[np.ndarray] = [None] * num_nodes  # type: ignore[list-item]
        pose_dict: Dict[str, np.ndarray] = {}

        for node in self.nodes:
            track = tracks.get(node.index)
            if track is None:
                local_mat = node.local_matrix
            else:
                # Rotation (Slerp)
                if track.rotations is not None and len(track.rotations) > 0:
                    q_sample = self._sample_quat_track(track.rotations, frame_pos)
                    if is_additive:
                        q_final = quat_mul(node.local_quat, q_sample)
                    else:
                        q_final = q_sample
                    rot_mat = quat_to_mat3(q_final)
                else:
                    rot_mat = node.local_rotation

                # Translation (Lerp)
                if track.translations is not None and len(track.translations) > 0:
                    t_sample = self._sample_vec3_track(track.translations, frame_pos)
                    if is_additive:
                        t_final = node.local_translation + (node.local_rotation @ t_sample)
                    else:
                        t_final = t_sample
                else:
                    t_final = node.local_translation

                local_mat = np.eye(4, dtype=np.float32)
                local_mat[:3, :3] = rot_mat
                local_mat[:3, 3] = t_final

            if node.parent_index >= 0:
                global_mat = global_poses[node.parent_index] @ local_mat
            else:
                global_mat = local_mat

            global_poses[node.index] = global_mat
            pose_dict[node.name] = global_mat

        return pose_dict


# ------------------------------------------------------------------
# Module-level Convenience API
# ------------------------------------------------------------------

_default_engine: Optional[AnimEngine] = None


def get_default_engine() -> AnimEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = AnimEngine()
    return _default_engine


def load_skeleton(skeleton_path: str | Path = DEFAULT_SKELETON_PATH) -> Dict[str, np.ndarray]:
    """Load base nodes from `skeleton_female.nif` using the shared headless `AnimEngine`."""
    return get_default_engine().load_skeleton(skeleton_path)


def load_animation(hkx_path: str | Path) -> LoadedAnimation:
    """Parse target `.hkx` animation using the shared headless `AnimEngine`."""
    return get_default_engine().load_animation(hkx_path)


def evaluate_pose(current_time: float) -> Dict[str, np.ndarray]:
    """Evaluate `{bone_name: 4x4_global_matrix}` at `current_time` seconds."""
    return get_default_engine().evaluate_pose(current_time)
