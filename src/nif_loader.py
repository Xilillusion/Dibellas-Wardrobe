"""
NIF Mesh & Material Parser for Dibella's Wardrobe.

Wraps `io_scene_nifly.pyn.pynifly` (`NiflyDLL.dll`) to extract geometry (vertices, normals,
UVs, vertex colors, tangents, bitangents, triangles), `BSLightingShaderProperty` /
`BSEffectShaderProperty` parameters and texture slots, and `NiAlphaProperty` transparency
settings.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure bundled io_scene_nifly is importable without importing its top-level Blender __init__.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NIFLY_DIR = _REPO_ROOT / "io_scene_nifly"
if str(_NIFLY_DIR) not in sys.path:
    sys.path.insert(0, str(_NIFLY_DIR))

from pyn.nifdefs import (  # type: ignore[import-not-found]  # noqa: E402
    BSLSPShaderType,
    PynBufferTypes,
    ShaderFlags1,
    ShaderFlags2,
    VERTEX_WEIGHT_PAIR,
)
from pyn.niflydll import nifly  # type: ignore[import-not-found]  # noqa: E402
from pyn.pynifly import NifFile  # type: ignore[import-not-found]  # noqa: E402

from .texture_resolver import ResolvedTexture, TextureResolver  # noqa: E402

log = logging.getLogger("dibellas_wardrobe.nif_loader")

_VW_DTYPE = np.dtype([("vertex", "<u2"), ("_pad", "<u2"), ("weight", "<f4")])
MAX_SHADER_BONES = 256


@dataclass
class AlphaPropertyData:
    """Extracted NiAlphaProperty transparency and cutout settings."""

    has_alpha_property: bool = False
    flags: int = 0
    alpha_blend: bool = False
    alpha_test: bool = False
    threshold: int = 128
    source_blend_mode: int = 6  # GL_SRC_ALPHA
    dst_blend_mode: int = 7     # GL_ONE_MINUS_SRC_ALPHA

    @property
    def shader_alpha_threshold(self) -> float:
        """Threshold in [0.0, 1.0] for default.frag, or -1.0 when alpha testing is disabled."""
        if self.has_alpha_property and self.alpha_test:
            return float(self.threshold) / 255.0
        return -1.0


@dataclass
class ShaderMaterialData:
    """Extracted BSLightingShaderProperty / BSEffectShaderProperty attributes & textures."""

    block_name: str = "BSLightingShaderProperty"
    bslsp_shader_type: int = 0
    bslsp_shader_type_name: str = "Default"
    shader_flags_1: int = 0
    shader_flags_2: int = 0

    uv_offset: Tuple[float, float] = (0.0, 0.0)
    uv_scale: Tuple[float, float] = (1.0, 1.0)
    specular_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    specular_strength: float = 1.0
    shininess: float = 32.0
    env_reflection: float = 1.0
    emissive_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_multiple: float = 1.0
    tint_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    alpha: float = 1.0
    rimlight_power: float = 2.0
    softlighting: float = 0.3

    # Feature flags matching Outfit Studio's default.frag uniforms
    b_normal_map: bool = True
    b_model_space: bool = False
    b_cubemap: bool = False
    b_env_mask: bool = False
    b_specular: bool = True
    b_emissive: bool = False
    b_backlight: bool = False
    b_rimlight: bool = False
    b_softlight: bool = False
    b_glowmap: bool = False
    b_tint_color: bool = False
    b_face_tint: bool = False
    b_double_sided: bool = False
    b_vertex_colors: bool = True
    b_vertex_alpha: bool = False

    raw_textures: Dict[str, str] = field(default_factory=dict)
    resolved_textures: Dict[str, ResolvedTexture] = field(default_factory=dict)
    default_resolved_textures: Dict[str, ResolvedTexture] = field(default_factory=dict)
    default_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ShapeMeshData:
    """Complete geometry and material payload for a single NIF shape."""

    name: str
    block_name: str
    positions: np.ndarray    # (V, 3) float32
    normals: np.ndarray      # (V, 3) float32
    uvs: np.ndarray          # (V, 2) float32
    tangents: np.ndarray     # (V, 3) float32
    bitangents: np.ndarray   # (V, 3) float32
    colors: np.ndarray       # (V, 4) float32
    triangles: np.ndarray    # (T, 3) uint32
    is_skinned: bool
    bone_names: List[str]
    shader: ShaderMaterialData
    alpha_prop: AlphaPropertyData
    bbox_min: np.ndarray     # (3,) float32
    bbox_max: np.ndarray     # (3,) float32
    bone_indices: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.int32)
    )  # (V, 4) int32
    bone_weights: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.float32)
    )  # (V, 4) float32
    bone_skin_matrices: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4, 4), dtype=np.float32)
    )  # (B, 4, 4) float32 inverse-bind (skin_to_bone) matrices


@dataclass
class LoadedNifData:
    """Container for all shapes extracted from a single `.nif` file."""

    filepath: Path
    game: str
    root_name: str
    shapes: List[ShapeMeshData]
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    # Path stored in the root's "HDT Skinned Mesh Physics Object" NiStringExtraData ("" when absent).
    smp_xml_ref: str = ""
    # Non-shape nodes: name -> (parent name or None, 4x4 local transform).
    nodes: Dict[str, Tuple[Optional[str], np.ndarray]] = field(default_factory=dict)


SMP_EXTRA_DATA_NAME = "HDT Skinned Mesh Physics Object"


def _extract_smp_xml_ref(nif) -> str:
    """Return the physics XML path embedded by an HDT-SMP outfit, or an empty string."""
    try:
        for extra in nif.root.extra_data():
            if getattr(extra, "name", "") == SMP_EXTRA_DATA_NAME:
                return str(getattr(extra, "string_data", "") or "").strip()
    except Exception as exc:
        log.debug("Could not read SMP extra data: %s", exc)
    return ""


def _extract_node_tree(nif, shape_names: set) -> Dict[str, Tuple[Optional[str], np.ndarray]]:
    """Collect every non-shape node with its parent name and 4x4 local transform."""
    tree: Dict[str, Tuple[Optional[str], np.ndarray]] = {}
    try:
        for name, node in nif.nodes.items():
            if name in shape_names:
                continue
            parent = node.parent
            xf = node.transform
            local = np.eye(4, dtype=np.float32)
            local[:3, :3] = [list(xf.rotation[0]), list(xf.rotation[1]), list(xf.rotation[2])]
            local[:3, 3] = [float(xf.translation[0]), float(xf.translation[1]), float(xf.translation[2])]
            tree[name] = (parent.name if parent is not None else None, local)
    except Exception as exc:
        log.debug("Could not read node tree: %s", exc)
    return tree


def compute_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute smooth area-weighted vertex normals if the NIF shape has no normals."""
    normals = np.zeros_like(positions, dtype=np.float32)
    if len(triangles) == 0 or len(positions) == 0:
        normals[:, 2] = 1.0
        return normals

    v0 = positions[triangles[:, 0]]
    v1 = positions[triangles[:, 1]]
    v2 = positions[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-8
    normals[valid] /= lengths[valid]
    normals[~valid, 2] = 1.0
    return normals


def compute_tangents_bitangents(
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-vertex tangent and bitangent vectors aligned with UV coordinates
    for tangent-space normal mapping (Lengyel's method, vectorized with NumPy).
    """
    num_verts = len(positions)
    tangents = np.zeros((num_verts, 3), dtype=np.float32)
    bitangents = np.zeros((num_verts, 3), dtype=np.float32)

    if len(triangles) == 0 or num_verts == 0:
        tangents[:, 0] = 1.0
        bitangents[:, 1] = 1.0
        return tangents, bitangents

    i0 = triangles[:, 0]
    i1 = triangles[:, 1]
    i2 = triangles[:, 2]

    p0, p1, p2 = positions[i0], positions[i1], positions[i2]
    w0, w1, w2 = uvs[i0], uvs[i1], uvs[i2]

    dp1 = p1 - p0
    dp2 = p2 - p0
    duv1 = w1 - w0
    duv2 = w2 - w0

    denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    valid = np.abs(denom) > 1e-8
    r = np.zeros_like(denom, dtype=np.float32)
    r[valid] = 1.0 / denom[valid]

    sdir = (duv2[:, 1:2] * dp1 - duv1[:, 1:2] * dp2) * r[:, None]
    tdir = (duv1[:, 0:1] * dp2 - duv2[:, 0:1] * dp1) * r[:, None]

    np.add.at(tangents, i0, sdir)
    np.add.at(tangents, i1, sdir)
    np.add.at(tangents, i2, sdir)

    np.add.at(bitangents, i0, tdir)
    np.add.at(bitangents, i1, tdir)
    np.add.at(bitangents, i2, tdir)

    # Gram-Schmidt orthogonalize tangent against normal
    n_dot_t = np.sum(normals * tangents, axis=1, keepdims=True)
    tangents = tangents - normals * n_dot_t
    t_len = np.linalg.norm(tangents, axis=1, keepdims=True)
    t_valid = t_len[:, 0] > 1e-6
    tangents[t_valid] /= t_len[t_valid]

    # Fallback orthonormal tangent for degenerate UV vertices
    if not np.all(t_valid):
        inv_mask = ~t_valid
        n_inv = normals[inv_mask]
        helper = np.where(
            np.abs(n_inv[:, :1]) < 0.9,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        fallback_t = np.cross(n_inv, helper)
        fb_len = np.linalg.norm(fallback_t, axis=1, keepdims=True)
        fb_len = np.maximum(fb_len, 1e-6)
        tangents[inv_mask] = fallback_t / fb_len

    # Orthogonalize bitangent and preserve handedness
    cross_nt = np.cross(normals, tangents)
    handedness = np.where(np.sum(cross_nt * bitangents, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    bitangents = cross_nt * handedness
    b_len = np.linalg.norm(bitangents, axis=1, keepdims=True)
    b_valid = b_len[:, 0] > 1e-6
    bitangents[b_valid] /= b_len[b_valid]

    return tangents, bitangents


def _apply_shape_transform(
    positions: np.ndarray, normals: np.ndarray, shape
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the shape's local TransformBuf if it has a non-identity rotation, scale, or translation.
    In Skyrim SE/AE, bind-pose outfit shapes share the root origin; applying `shape.transform`
    aligns any offset static/accessory sub-shapes while keeping standard bind-pose shapes intact.
    """
    try:
        xf = shape.transform
        scale = float(xf.scale) if xf.scale != 0.0 else 1.0
        rot = np.array(
            [
                [xf.rotation[0][0], xf.rotation[0][1], xf.rotation[0][2]],
                [xf.rotation[1][0], xf.rotation[1][1], xf.rotation[1][2]],
                [xf.rotation[2][0], xf.rotation[2][1], xf.rotation[2][2]],
            ],
            dtype=np.float32,
        )
        trans = np.array(
            [xf.translation[0], xf.translation[1], xf.translation[2]],
            dtype=np.float32,
        )

        is_identity = (
            abs(scale - 1.0) < 1e-5
            and np.allclose(rot, np.eye(3, dtype=np.float32), atol=1e-5)
            and np.allclose(trans, 0.0, atol=1e-5)
        )
        if is_identity:
            return positions, normals

        # Transform positions: P' = P * R^T * scale + T
        positions = (positions @ rot.T) * scale + trans
        normals = normals @ rot.T
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = norm_len[:, 0] > 1e-8
        normals[valid] /= norm_len[valid]
    except Exception as exc:
        log.debug("Could not apply transform for shape %s: %s", getattr(shape, "name", "?"), exc)

    return positions, normals


def _extract_shader_data(
    shape, nif_path: Path, resolver: TextureResolver
) -> ShaderMaterialData:
    """Extract shader properties and resolve all relevant `.dds` textures for `shape`."""
    mat = ShaderMaterialData()
    try:
        shader = shape.shader
        if shader is None:
            raise ValueError("Shape has no shader object")
        props = shader.properties
        mat.block_name = getattr(shape, "shader_block_name", "") or "BSLightingShaderProperty"
    except Exception as exc:
        log.warning("Shape '%s' has no readable shader (%s); using defaults.", shape.name, exc)
        mat.resolved_textures["Diffuse"] = resolver.load_or_fallback("Diffuse", "", nif_path)
        mat.resolved_textures["Normal"] = resolver.load_or_fallback("Normal", "", nif_path)
        mat.default_resolved_textures = dict(mat.resolved_textures)
        mat.default_flags = {
            "b_env_mask": mat.b_env_mask,
            "b_glowmap": mat.b_glowmap,
            "b_emissive": mat.b_emissive,
            "b_cubemap": mat.b_cubemap,
        }
        return mat

    # Raw texture dictionary from pynifly
    try:
        raw_tex = dict(shape.textures)
    except Exception:
        raw_tex = {}
    mat.raw_textures = {k: v for k, v in raw_tex.items() if isinstance(v, str)}

    is_lighting = props.bufType == PynBufferTypes.BSLightingShaderPropertyBufType
    is_effect = props.bufType == PynBufferTypes.BSEffectShaderPropertyBufType

    mat.shader_flags_1 = int(props.Shader_Flags_1)
    mat.shader_flags_2 = int(props.Shader_Flags_2)
    mat.uv_offset = (float(props.UV_Offset_U), float(props.UV_Offset_V))
    mat.uv_scale = (
        float(props.UV_Scale_U) if props.UV_Scale_U != 0.0 else 1.0,
        float(props.UV_Scale_V) if props.UV_Scale_V != 0.0 else 1.0,
    )

    if is_lighting:
        st_val = int(props.Shader_Type)
        mat.bslsp_shader_type = st_val
        try:
            mat.bslsp_shader_type_name = BSLSPShaderType(st_val).name
        except ValueError:
            mat.bslsp_shader_type_name = f"Type_{st_val}"

        mat.specular_color = (
            float(props.Spec_Color[0]),
            float(props.Spec_Color[1]),
            float(props.Spec_Color[2]),
        )
        mat.specular_strength = float(props.Spec_Str)
        mat.shininess = max(1.0, float(props.Glossiness))
        mat.env_reflection = float(props.Env_Map_Scale)
        mat.emissive_color = (
            float(props.Emissive_Color[0]),
            float(props.Emissive_Color[1]),
            float(props.Emissive_Color[2]),
        )
        mat.emissive_multiple = float(props.Emissive_Mult)
        mat.alpha = float(props.Alpha)
        mat.rimlight_power = max(0.1, float(props.Rim_Light_Power))
        mat.softlighting = float(props.Soft_Lighting)

        # Skin Tint (5) or Hair Tint (6) or Facegen RGB Tint flag
        if st_val == BSLSPShaderType.Skin_Tint:
            mat.b_tint_color = True
            mat.tint_color = (
                float(props.skinTintColor[0]),
                float(props.skinTintColor[1]),
                float(props.skinTintColor[2]),
            )
        elif st_val == BSLSPShaderType.Hair_Tint:
            mat.b_tint_color = True
            mat.tint_color = (
                float(props.hairTintColor[0]),
                float(props.hairTintColor[1]),
                float(props.hairTintColor[2]),
            )

        mat.b_model_space = bool(props.shaderflags1_test(ShaderFlags1.MODEL_SPACE_NORMALS))
        mat.b_specular = bool(props.shaderflags1_test(ShaderFlags1.SPECULAR)) or not mat.b_model_space
        mat.b_cubemap = bool(
            props.shaderflags1_test(ShaderFlags1.ENVIRONMENT_MAPPING)
            or props.shaderflags1_test(ShaderFlags1.EYE_ENVIRONMENT_MAPPING)
            or st_val == BSLSPShaderType.Environment_Map
        )
        mat.b_emissive = bool(
            props.shaderflags1_test(ShaderFlags1.OWN_EMIT)
            or props.shaderflags2_test(ShaderFlags2.GLOW_MAP)
            or st_val == BSLSPShaderType.Glow_Shader
        )
        mat.b_glowmap = bool(
            props.shaderflags2_test(ShaderFlags2.GLOW_MAP) and mat.raw_textures.get("Glow")
        )
        mat.b_softlight = bool(props.shaderflags2_test(ShaderFlags2.SOFT_LIGHTING))
        mat.b_rimlight = bool(props.shaderflags2_test(ShaderFlags2.RIM_LIGHTING))
        mat.b_backlight = bool(props.shaderflags2_test(ShaderFlags2.BACK_LIGHTING))
        mat.b_double_sided = bool(props.shaderflags2_test(ShaderFlags2.DOUBLE_SIDED))
        mat.b_vertex_colors = bool(props.shaderflags2_test(ShaderFlags2.VERTEX_COLORS))
        mat.b_vertex_alpha = bool(props.shaderflags1_test(ShaderFlags1.VERTEX_ALPHA))

    elif is_effect:
        mat.bslsp_shader_type_name = "BSEffectShaderProperty"
        mat.alpha = float(props.baseColor[3]) if props.baseColor[3] > 0.0 else 1.0
        mat.emissive_color = (
            float(props.baseColor[0]),
            float(props.baseColor[1]),
            float(props.baseColor[2]),
        )
        mat.emissive_multiple = float(props.baseColorScale) if props.baseColorScale > 0.0 else 1.0
        mat.b_emissive = True
        mat.b_double_sided = bool(props.shaderflags2_test(ShaderFlags2.DOUBLE_SIDED))
        mat.b_vertex_colors = bool(props.shaderflags2_test(ShaderFlags2.VERTEX_COLORS))
        mat.b_vertex_alpha = bool(props.shaderflags1_test(ShaderFlags1.VERTEX_ALPHA))

    # Resolve Diffuse and Normal (always bound to units 0 and 1)
    diff_raw = mat.raw_textures.get("Diffuse", "")
    norm_raw = mat.raw_textures.get("Normal", "")
    mat.resolved_textures["Diffuse"] = resolver.load_or_fallback("Diffuse", diff_raw, nif_path)
    mat.resolved_textures["Normal"] = resolver.load_or_fallback("Normal", norm_raw, nif_path)
    mat.b_normal_map = True

    # Resolve optional EnvMap / EnvMask (Slots 4 & 5)
    env_map_raw = mat.raw_textures.get("EnvMap", "")
    env_mask_raw = mat.raw_textures.get("EnvMask", "") or mat.raw_textures.get("EnvMapMask", "")
    if mat.b_cubemap or env_map_raw:
        mat.b_cubemap = True
        if env_map_raw:
            mat.resolved_textures["EnvMap"] = resolver.load_or_fallback("EnvMap", env_map_raw, nif_path)
    if env_mask_raw:
        mat.resolved_textures["EnvMask"] = resolver.load_or_fallback("EnvMask", env_mask_raw, nif_path)
        mat.b_env_mask = True
    else:
        mat.b_env_mask = False

    # Resolve optional Specular / Backlight / Lightmask / Glow
    spec_raw = mat.raw_textures.get("Specular", "")
    if spec_raw:
        mat.resolved_textures["Specular"] = resolver.load_or_fallback("Specular", spec_raw, nif_path)

    glow_raw = mat.raw_textures.get("Glow", "")
    if glow_raw:
        mat.resolved_textures["Glow"] = resolver.load_or_fallback("Glow", glow_raw, nif_path)

    soft_raw = mat.raw_textures.get("SoftLighting", "") or mat.raw_textures.get("RimLighting", "")
    if soft_raw:
        mat.resolved_textures["Lightmask"] = resolver.load_or_fallback("Lightmask", soft_raw, nif_path)

    mat.default_resolved_textures = dict(mat.resolved_textures)
    mat.default_flags = {
        "b_env_mask": mat.b_env_mask,
        "b_glowmap": mat.b_glowmap,
        "b_emissive": mat.b_emissive,
        "b_cubemap": mat.b_cubemap,
    }
    return mat


def _extract_alpha_property(shape) -> AlphaPropertyData:
    """Extract NiAlphaProperty settings from `shape` if present."""
    alpha_data = AlphaPropertyData()
    try:
        if shape.has_alpha_property and shape.alpha_property is not None:
            ap_props = shape.alpha_property.properties
            alpha_data.has_alpha_property = True
            alpha_data.flags = int(ap_props.flags)
            alpha_data.alpha_blend = bool(ap_props.alpha_blend)
            alpha_data.alpha_test = bool(ap_props.alpha_test)
            alpha_data.threshold = int(ap_props.threshold)
            alpha_data.source_blend_mode = int(ap_props.source_blend_mode)
            alpha_data.dst_blend_mode = int(ap_props.dst_blend_mode)
    except Exception as exc:
        log.debug("Failed reading NiAlphaProperty on %s: %s", getattr(shape, "name", "?"), exc)
    return alpha_data


def _extract_skin_data(
    nif, shape, num_verts: int
) -> Tuple[bool, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract unique bone names, `(B, 4, 4)` inverse-bind (`skin_to_bone`) matrices,
    and per-vertex top-4 `(V, 4)` bone indices and normalized weights for GPU LBS.
    """
    default_indices = np.zeros((num_verts, 4), dtype=np.int32)
    default_weights = np.zeros((num_verts, 4), dtype=np.float32)
    default_weights[:, 0] = 1.0
    default_mats = np.zeros((0, 4, 4), dtype=np.float32)

    if not bool(getattr(shape, "has_skin_instance", False)) or num_verts <= 0:
        return False, [], default_indices, default_weights, default_mats

    try:
        raw_names = list(shape.bone_names)
        raw_ids = list(shape.bone_ids)
    except Exception as exc:
        log.debug("Failed reading bone list on shape %s: %s", getattr(shape, "name", "?"), exc)
        return False, [], default_indices, default_weights, default_mats

    if not raw_names or not raw_ids:
        return False, [], default_indices, default_weights, default_mats

    unique_names: List[str] = []
    id_to_unique_idx: Dict[int, int] = {}
    for nm, bid in zip(raw_names, raw_ids):
        if bid not in id_to_unique_idx:
            if len(unique_names) >= MAX_SHADER_BONES:
                continue
            id_to_unique_idx[bid] = len(unique_names)
            unique_names.append(nm)

    num_bones = len(unique_names)
    if num_bones == 0:
        return False, [], default_indices, default_weights, default_mats

    # Extract per-bone skin_to_bone inverse-bind 4x4 matrices
    skin_mats = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], num_bones, axis=0)
    for b_idx, b_name in enumerate(unique_names):
        try:
            s2b = shape.get_shape_skin_to_bone(b_name)
            if s2b is not None:
                rot = np.array(
                    [
                        list(s2b.rotation[0]),
                        list(s2b.rotation[1]),
                        list(s2b.rotation[2]),
                    ],
                    dtype=np.float32,
                )
                skin_mats[b_idx, :3, :3] = rot
                skin_mats[b_idx, :3, 3] = [
                    float(s2b.translation[0]),
                    float(s2b.translation[1]),
                    float(s2b.translation[2]),
                ]
        except Exception:
            pass

    # If the shape itself had a non-identity local transform applied to `positions`,
    # fold its inverse into skin_mats so vertices in NIF root space transform properly.
    try:
        xf = shape.transform
        scale = float(xf.scale) if xf.scale != 0.0 else 1.0
        rot = np.array(
            [list(xf.rotation[0]), list(xf.rotation[1]), list(xf.rotation[2])],
            dtype=np.float32,
        )
        trans = np.array(
            [float(xf.translation[0]), float(xf.translation[1]), float(xf.translation[2])],
            dtype=np.float32,
        )
        is_identity = (
            abs(scale - 1.0) < 1e-5
            and np.allclose(rot, np.eye(3, dtype=np.float32), atol=1e-5)
            and np.allclose(trans, 0.0, atol=1e-5)
        )
        if not is_identity:
            sh_mat = np.eye(4, dtype=np.float32)
            sh_mat[:3, :3] = rot * scale
            sh_mat[:3, 3] = trans
            sh_inv = np.linalg.inv(sh_mat).astype(np.float32)
            skin_mats = skin_mats @ sh_inv[None, :, :]
    except Exception:
        pass

    # Fast ctypes + NumPy extraction of per-vertex weights across all partition bone slots
    v_list: List[np.ndarray] = []
    b_list: List[np.ndarray] = []
    w_list: List[np.ndarray] = []
    for pos, bid in enumerate(raw_ids):
        u_idx = id_to_unique_idx.get(bid)
        if u_idx is None:
            continue
        try:
            cnt = int(nifly.getShapeBoneWeightsCount(nif._handle, shape._handle, pos))
            if cnt <= 0:
                continue
            buf = (VERTEX_WEIGHT_PAIR * cnt)()
            nifly.getShapeBoneWeights(nif._handle, shape._handle, pos, buf, cnt)
            arr = np.frombuffer(buf, dtype=_VW_DTYPE, count=cnt)
            v_idx = arr["vertex"].astype(np.int32)
            w_val = arr["weight"].copy()
            valid = (v_idx >= 0) & (v_idx < num_verts) & (w_val > 1e-5)
            if np.any(valid):
                valid_count = int(np.count_nonzero(valid))
                v_list.append(v_idx[valid])
                b_list.append(np.full(valid_count, u_idx, dtype=np.int32))
                w_list.append(w_val[valid])
        except Exception:
            continue

    if not v_list:
        return False, unique_names, default_indices, default_weights, skin_mats

    all_v = np.concatenate(v_list)
    all_b = np.concatenate(b_list)
    all_w = np.concatenate(w_list)

    # Aggregate weights for the same (vertex, unique_bone) pair across partitions
    vb_key = all_v.astype(np.int64) * num_bones + all_b.astype(np.int64)
    uniq_vb, inv_idx = np.unique(vb_key, return_inverse=True)
    sum_w = np.bincount(inv_idx, weights=all_w).astype(np.float32)
    u_v = (uniq_vb // num_bones).astype(np.int32)
    u_b = (uniq_vb % num_bones).astype(np.int32)

    # Sort by vertex index ascending, weight descending, and keep top 4 per vertex
    order = np.lexsort((-sum_w, u_v))
    u_v = u_v[order]
    u_b = u_b[order]
    sum_w = sum_w[order]

    first_idx = np.unique(u_v, return_index=True)[1]
    group_id = np.searchsorted(first_idx, np.arange(len(u_v)), side="right") - 1
    rank = np.arange(len(u_v)) - first_idx[group_id]
    keep = rank < 4

    bone_indices = np.zeros((num_verts, 4), dtype=np.int32)
    bone_weights = np.zeros((num_verts, 4), dtype=np.float32)
    bone_indices[u_v[keep], rank[keep]] = u_b[keep]
    bone_weights[u_v[keep], rank[keep]] = sum_w[keep]

    w_sum = bone_weights.sum(axis=1, keepdims=True)
    has_w = w_sum[:, 0] > 1e-6
    bone_weights[has_w] /= w_sum[has_w]
    bone_weights[~has_w, 0] = 1.0

    return (
        True,
        unique_names,
        np.ascontiguousarray(bone_indices, dtype=np.int32),
        np.ascontiguousarray(bone_weights, dtype=np.float32),
        np.ascontiguousarray(skin_mats, dtype=np.float32),
    )


class NifLoader:
    """Loads `.nif` files via `pynifly` and resolves their textures via `TextureResolver`."""

    def __init__(self, resolver: Optional[TextureResolver] = None) -> None:
        self.resolver = resolver or TextureResolver()

    def load_nif(self, nif_path: Path | str) -> LoadedNifData:
        """Parse a `.nif` file and return all renderable `ShapeMeshData` blocks."""
        path = Path(nif_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"NIF file not found: {path}")

        with NifFile(str(path)) as nif:
            game = getattr(nif, "game", "SKYRIMSE") or "SKYRIMSE"
            root_name = getattr(nif, "rootName", "Scene Root") or "Scene Root"

            shapes_out: List[ShapeMeshData] = []
            global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
            global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

            for shape in nif.shapes:
                raw_verts = shape.verts
                raw_tris = shape.tris
                if not raw_verts or not raw_tris:
                    continue

                positions = np.asarray(raw_verts, dtype=np.float32)
                triangles = np.asarray(raw_tris, dtype=np.uint32)
                num_verts = len(positions)

                # Normals
                raw_normals = shape.normals
                if raw_normals and len(raw_normals) == num_verts:
                    normals = np.asarray(raw_normals, dtype=np.float32)
                else:
                    normals = compute_normals(positions, triangles)

                # Apply local shape transform if present
                positions, normals = _apply_shape_transform(positions, normals, shape)

                # UVs
                try:
                    raw_uvs = shape.uvs
                except Exception:
                    raw_uvs = None
                if raw_uvs and len(raw_uvs) == num_verts:
                    uvs = np.asarray(raw_uvs, dtype=np.float32)
                else:
                    uvs = np.zeros((num_verts, 2), dtype=np.float32)

                # Tangents and bitangents
                tangents, bitangents = compute_tangents_bitangents(positions, normals, uvs, triangles)

                # Vertex colors
                try:
                    raw_colors = shape.colors
                except Exception:
                    raw_colors = None
                if raw_colors and len(raw_colors) == num_verts:
                    colors = np.asarray(raw_colors, dtype=np.float32)
                else:
                    colors = np.ones((num_verts, 4), dtype=np.float32)

                shader_data = _extract_shader_data(shape, path, self.resolver)
                if not shader_data.b_vertex_colors:
                    colors = np.ones((num_verts, 4), dtype=np.float32)
                elif not shader_data.b_vertex_alpha:
                    colors[:, 3] = 1.0

                alpha_data = _extract_alpha_property(shape)

                (
                    is_skinned,
                    bone_names,
                    bone_indices,
                    bone_weights,
                    bone_skin_matrices,
                ) = _extract_skin_data(nif, shape, num_verts)

                s_min = positions.min(axis=0)
                s_max = positions.max(axis=0)
                global_min = np.minimum(global_min, s_min)
                global_max = np.maximum(global_max, s_max)

                shapes_out.append(
                    ShapeMeshData(
                        name=shape.name or shape.blockname,
                        block_name=shape.blockname,
                        positions=np.ascontiguousarray(positions, dtype=np.float32),
                        normals=np.ascontiguousarray(normals, dtype=np.float32),
                        uvs=np.ascontiguousarray(uvs, dtype=np.float32),
                        tangents=np.ascontiguousarray(tangents, dtype=np.float32),
                        bitangents=np.ascontiguousarray(bitangents, dtype=np.float32),
                        colors=np.ascontiguousarray(colors, dtype=np.float32),
                        triangles=np.ascontiguousarray(triangles, dtype=np.uint32),
                        is_skinned=is_skinned,
                        bone_names=bone_names,
                        shader=shader_data,
                        alpha_prop=alpha_data,
                        bbox_min=s_min,
                        bbox_max=s_max,
                        bone_indices=bone_indices,
                        bone_weights=bone_weights,
                        bone_skin_matrices=bone_skin_matrices,
                    )
                )

            if not np.all(np.isfinite(global_min)):
                global_min = np.zeros(3, dtype=np.float32)
                global_max = np.ones(3, dtype=np.float32)

            shape_names = {s.name for s in shapes_out}
            return LoadedNifData(
                filepath=path,
                game=game,
                root_name=root_name,
                shapes=shapes_out,
                bbox_min=global_min,
                bbox_max=global_max,
                smp_xml_ref=_extract_smp_xml_ref(nif),
                nodes=_extract_node_tree(nif, shape_names),
            )
