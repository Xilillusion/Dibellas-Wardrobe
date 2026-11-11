"""
HDT-SMP cloth physics bridge: a zero-copy ``ctypes`` binding to the Native AOT ``SmpBridge.dll``.

``SmpBridge.dll`` (``smp_bridge/SmpBridge``, C# + BulletSharp compiled with Native AOT) steps a Bullet world with
6DOF spring constraints parsed from an outfit's SMP XML. It is a plain C ABI loaded with ``ctypes.CDLL``, so it needs
no managed runtime, and per-frame data moves through preallocated contiguous float32 NumPy buffers whose raw
pointers are handed to the native side: a step allocates nothing on either side of the boundary.

Per-frame pipeline (see ``SmpBridgeSolver.update``):

1. kinematic bone matrices from the animated skeleton are copied into one buffer;
2. ``SmpStep`` advances Bullet and writes every dynamic cloth bone's world matrix into a second buffer;
3. those matrices are merged into the pose palette next to the skeleton and CBPC bones, ready for GPU skinning.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .anim_engine import AnimEngine
from .cbpc_manager import CbpcManager
if TYPE_CHECKING:
    from .nif_loader import ShapeMeshData

log = logging.getLogger("dibellas_wardrobe.smp_bridge")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", _REPO_ROOT))

DLL_NAME = "SmpBridge.dll"
BULLET_DLL_NAME = "libbulletc.dll"
ENV_DIR = "SMP_BRIDGE_DIR"
PUBLISH_DIR = _REPO_ROOT / "smp_bridge" / "SmpBridge" / "bin" / "Release" / "net8.0" / "win-x64" / "publish"
# Must equal SmpExports.AbiVersion in SmpBridge.cs.
ABI_VERSION = 3

KIND_KINEMATIC = 0
KIND_DYNAMIC = 1

STATUS_OK = 0
STATUS_NO_BODIES = -4
_NAME_BUFFER = 512
_LOG_BUFFER = 64 * 1024

_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_INT_P = ctypes.POINTER(ctypes.c_int)
_CHAR_PP = ctypes.POINTER(ctypes.c_char_p)


class SmpBridgeError(RuntimeError):
    """The native library is missing, incompatible, or rejected a call."""


def native_search_dirs(directory: Optional[Path | str] = None) -> List[Path]:
    dirs: List[Path] = []
    if directory:
        dirs.append(Path(directory))
    env = os.environ.get(ENV_DIR)
    if env:
        dirs.append(Path(env))
    dirs.append(_RESOURCE_ROOT / "smp_bridge")
    dirs.append(PUBLISH_DIR)
    return dirs


def find_native_dir(directory: Optional[Path | str] = None) -> Optional[Path]:
    for candidate in native_search_dirs(directory):
        if (candidate / DLL_NAME).is_file() and (candidate / BULLET_DLL_NAME).is_file():
            return candidate.resolve()
    return None


class SmpNative:
    """Typed ``ctypes.CDLL`` wrapper around the exported ``Smp*`` entry points."""

    def __init__(self, directory: Optional[Path | str] = None) -> None:
        found = find_native_dir(directory)
        if found is None:
            searched = ", ".join(str(p) for p in native_search_dirs(directory))
            raise SmpBridgeError(
                f"{DLL_NAME} and {BULLET_DLL_NAME} were not found (searched: {searched}). "
                "Build them with: dotnet publish smp_bridge/SmpBridge -c Release -r win-x64"
            )
        self.directory = found
        self._dll_dir = os.add_dll_directory(str(found)) if hasattr(os, "add_dll_directory") else None
        try:
            # Native AOT resolves libbulletc by name; loading it first by full path guarantees it is found.
            self._bullet = ctypes.CDLL(str(found / BULLET_DLL_NAME))
            self._lib = ctypes.CDLL(str(found / DLL_NAME))
            self._declare()
        except (OSError, AttributeError) as exc:
            self.close()
            raise SmpBridgeError(f"Could not load {DLL_NAME} from {found}: {exc}") from exc
        version = self._lib.SmpAbiVersion()
        if version != ABI_VERSION:
            self.close()
            raise SmpBridgeError(f"{DLL_NAME} ABI version {version} does not match expected {ABI_VERSION}")
        self._name_buffer = ctypes.create_string_buffer(_NAME_BUFFER)
        self._log_buffer = ctypes.create_string_buffer(_LOG_BUFFER)

    def _declare(self) -> None:
        lib = self._lib
        lib.SmpAbiVersion.argtypes = []
        lib.SmpAbiVersion.restype = ctypes.c_int
        lib.SmpSetBindPose.argtypes = [_CHAR_PP, _FLOAT_P, ctypes.c_int]
        lib.SmpSetBindPose.restype = ctypes.c_int
        lib.SmpRegisterMesh.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, _CHAR_PP, _FLOAT_P, ctypes.c_int,
            _FLOAT_P, _INT_P, _FLOAT_P, ctypes.c_int, _INT_P, ctypes.c_int,
        ]
        lib.SmpRegisterMesh.restype = ctypes.c_int
        lib.SmpSetMeshScope.argtypes = [ctypes.c_char_p]
        lib.SmpSetMeshScope.restype = ctypes.c_int
        lib.SmpLoadXml.argtypes = [ctypes.c_char_p, _CHAR_PP, ctypes.c_int]
        lib.SmpLoadXml.restype = ctypes.c_int
        lib.SmpGetBodyCount.argtypes = [ctypes.c_int]
        lib.SmpGetBodyCount.restype = ctypes.c_int
        lib.SmpGetBodyName.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.SmpGetBodyName.restype = ctypes.c_int
        lib.SmpGetPrimitive.argtypes = [ctypes.c_int, ctypes.c_int, _FLOAT_P, ctypes.c_int]
        lib.SmpGetPrimitive.restype = ctypes.c_int
        lib.SmpGetLog.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.SmpGetLog.restype = ctypes.c_int
        lib.SmpStep.argtypes = [ctypes.c_float, _FLOAT_P, ctypes.c_int, _FLOAT_P, ctypes.c_int]
        lib.SmpStep.restype = None
        lib.SmpLastStepStatus.argtypes = []
        lib.SmpLastStepStatus.restype = ctypes.c_int
        lib.SmpReset.argtypes = []
        lib.SmpReset.restype = None
        lib.SmpDestroy.argtypes = []
        lib.SmpDestroy.restype = None

    @staticmethod
    def pointer(array: np.ndarray) -> "ctypes._Pointer[ctypes.c_float]":
        """Raw float pointer into ``array`` without copying; the array must stay alive and contiguous."""
        if array.dtype != np.float32 or not array.flags["C_CONTIGUOUS"]:
            raise SmpBridgeError("native buffers must be C-contiguous float32 arrays")
        return array.ctypes.data_as(_FLOAT_P)

    @staticmethod
    def _names(names: Sequence[str]):
        return (ctypes.c_char_p * len(names))(*[n.encode("utf-8") for n in names])

    def set_bind_pose(self, names: Sequence[str], matrices: np.ndarray) -> int:
        matrices = np.ascontiguousarray(matrices, dtype=np.float32).reshape(-1, 4, 4)
        if len(names) != len(matrices):
            raise SmpBridgeError("bind pose names and matrices differ in length")
        return self._lib.SmpSetBindPose(self._names(names), self.pointer(matrices), len(names))

    def load_xml(self, xml_path: Path | str, candidates: Sequence[str]) -> int:
        return self._lib.SmpLoadXml(str(xml_path).encode("utf-8"), self._names(candidates), len(candidates))

    def register_mesh(self, source: str, shape: "ShapeMeshData") -> int:
        bones = shape.bone_names
        positions = np.ascontiguousarray(shape.positions, dtype=np.float32).reshape(-1, 3)
        matrices = np.ascontiguousarray(shape.bone_skin_matrices, dtype=np.float32).reshape(-1, 4, 4)
        weights = np.ascontiguousarray(shape.bone_weights, dtype=np.float32).reshape(-1, 4)
        indices = np.ascontiguousarray(shape.bone_indices, dtype=np.int32).reshape(-1, 4)
        triangles = np.ascontiguousarray(shape.triangles, dtype=np.int32).reshape(-1, 3)
        if len(matrices) != len(bones) or len(weights) != len(positions) or len(indices) != len(positions):
            raise SmpBridgeError(f"shape {shape.name}: inconsistent skin data")
        return self._lib.SmpRegisterMesh(
            source.encode("utf-8"), shape.name.encode("utf-8"), self._names(bones),
            self.pointer(matrices), len(bones), self.pointer(positions), indices.ctypes.data_as(_INT_P),
            self.pointer(weights), len(positions), triangles.ctypes.data_as(_INT_P), len(triangles),
        )

    def set_mesh_scope(self, source: str) -> int:
        return self._lib.SmpSetMeshScope(source.encode("utf-8"))

    def body_count(self, kind: int) -> int:
        return self._lib.SmpGetBodyCount(kind)

    def body_names(self, kind: int) -> List[str]:
        names: List[str] = []
        for index in range(max(0, self.body_count(kind))):
            length = self._lib.SmpGetBodyName(kind, index, self._name_buffer, _NAME_BUFFER)
            if length < 0:
                raise SmpBridgeError(f"SmpGetBodyName({kind}, {index}) failed with status {length}")
            names.append(self._name_buffer.raw[:length].decode("utf-8"))
        return names

    def drain_log(self) -> List[str]:
        length = self._lib.SmpGetLog(self._log_buffer, _LOG_BUFFER)
        if length <= 0:
            return []
        return self._log_buffer.raw[:length].decode("utf-8", "replace").splitlines()

    def primitives(self) -> List[np.ndarray]:
        result = []
        for kind in (KIND_KINEMATIC, KIND_DYNAMIC):
            for index in range(self.body_count(kind)):
                data = np.zeros(22, dtype=np.float32)
                status = self._lib.SmpGetPrimitive(kind, index, self.pointer(data), len(data))
                if status < 0:
                    raise SmpBridgeError(f"SmpGetPrimitive({kind}, {index}) failed with status {status}")
                if status:
                    result.append(data)
        return result

    def step(self, dt: float, kinematic_ptr, kinematic_count: int, dynamic_ptr, dynamic_count: int) -> int:
        self._lib.SmpStep(dt, kinematic_ptr, kinematic_count, dynamic_ptr, dynamic_count)
        return self._lib.SmpLastStepStatus()

    def reset(self) -> None:
        self._lib.SmpReset()

    def destroy(self) -> None:
        self._lib.SmpDestroy()

    def close(self) -> None:
        lib = getattr(self, "_lib", None)
        if lib is not None:
            try:
                lib.SmpDestroy()
            except (OSError, AttributeError):
                pass
        if self._dll_dir is not None:
            self._dll_dir.close()
            self._dll_dir = None


_native: Optional[SmpNative] = None
_native_error: Optional[str] = None


def get_native(directory: Optional[Path | str] = None) -> SmpNative:
    """Process-wide binding; the native library keeps a single Bullet world, so it is loaded once."""
    global _native, _native_error
    if _native is None:
        try:
            _native = SmpNative(directory)
        except SmpBridgeError as exc:
            _native_error = str(exc)
            raise
    return _native


# ----------------------------------------------------------------------
# Asset resolution
# ----------------------------------------------------------------------

def _safe_relative(ref: str) -> Optional[PurePosixPath]:
    """The XML reference as a relative POSIX path, or None if it is absolute or climbs out of the mod tree."""
    cleaned = ref.strip().strip('"').replace("\\", "/")
    if not cleaned:
        return None
    path = PurePosixPath(cleaned)
    if path.is_absolute() or (len(cleaned) > 1 and cleaned[1] == ":") or ".." in path.parts:
        return None
    return path


def resolve_smp_xml(nif_path: Path | str, xml_ref: str = "") -> Optional[Path]:
    """Find the physics XML for ``nif_path``.

    1. The path in the NIF's ``HDT Skinned Mesh Physics Object`` ``NiStringExtraData`` (``xml_ref``), relative to
       the mod root: tried against every ancestor folder of the NIF, then next to the NIF by file name.
    2. A co-located ``[nif_name].xml``.
    """
    nif = Path(nif_path)
    rel = _safe_relative(xml_ref) if xml_ref else None
    if rel is not None:
        for folder in (nif.parent, *nif.parent.parents):
            candidate = folder.joinpath(*rel.parts)
            if candidate.is_file():
                return candidate.resolve()
        sibling = nif.parent / rel.name
        if sibling.is_file():
            return sibling.resolve()
    colocated = nif.with_suffix(".xml")
    if colocated.is_file():
        return colocated.resolve()
    return None


def primitive_debug_lines(primitives: Sequence[np.ndarray], segments: int = 24) -> np.ndarray:
    """Return world-space line segments for the exact native primitive dimensions."""
    lines: List[Tuple[float, float, float]] = []
    for data in primitives:
        kind, radius, height, x, y, z = data[:6]
        points: List[Tuple[float, float, float]] = []

        def line(a, b):
            points.extend((a, b))

        if int(kind) == 2:
            corners = [(sx * x, sy * y, sz * z) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            for i, a in enumerate(corners):
                for bit in (1, 2, 4):
                    j = i ^ bit
                    if i < j:
                        line(a, corners[j])
        else:
            def circle(axis: int, center_z: float, ring_radius: float) -> None:
                for i in range(segments):
                    a, b = 2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments
                    def point(t):
                        c, s = ring_radius * math.cos(t), ring_radius * math.sin(t)
                        return (c, s, center_z) if axis == 2 else ((c, 0, center_z + s) if axis == 1 else (0, c, center_z + s))
                    line(point(a), point(b))

            if int(kind) == 0:
                for axis in range(3):
                    circle(axis, 0, radius)
            else:
                half = height / 2
                circle(2, -half, radius)
                circle(2, half, radius)
                for axis in (0, 1):
                    for side in (-1, 1):
                        def point(angle):
                            radial = radius * math.cos(angle)
                            zz = side * (half + radius * math.sin(angle))
                            return (radial, 0, zz) if axis == 0 else (0, radial, zz)
                        for i in range(segments // 2):
                            a = math.pi * i / (segments // 2)
                            b = math.pi * (i + 1) / (segments // 2)
                            line(point(a), point(b))
                for angle in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
                    px, py = radius * math.cos(angle), radius * math.sin(angle)
                    line((px, py, -half), (px, py, half))

        vertices = np.asarray(points, dtype=np.float32)
        matrix = data[6:].reshape(4, 4)
        world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        lines.extend(map(tuple, world))
    return np.asarray(lines, dtype=np.float32).reshape(-1, 3)


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------

@dataclass
class _NifSource:
    path: Path
    xml_ref: str
    nodes: Dict[str, Tuple[Optional[str], np.ndarray]]
    shapes: Sequence["ShapeMeshData"]


@dataclass
class _OutfitBone:
    name: str
    parent: str
    local: np.ndarray  # (4, 4) float32, relative to ``parent``
    slot: int


class SmpBridgeSolver:
    """Drives SMP cloth for the visible outfits and merges the result into the skinning palette.

    Outfit-only bones (cloth chains such as ``cocoa02``, absent from the skeleton) always get a world matrix in the
    palette, following their NIF parent chain; bones SMP simulates are then overwritten with the Bullet result.
    """

    def __init__(
        self,
        anim_engine: AnimEngine,
        cbpc: Optional[CbpcManager] = None,
        native_dir: Optional[Path | str] = None,
    ) -> None:
        self.anim_engine = anim_engine
        self.cbpc = cbpc
        self.enabled = True
        self.diagnostics: List[str] = []
        self._native_dir = native_dir
        self._sources: Dict[str, _NifSource] = {}
        self._visible: FrozenSet[str] = frozenset()
        self._collision_xmls: Tuple[Path, ...] = ()
        self._signature: Optional[tuple] = None
        self._failed = False

        self._outfit: List[_OutfitBone] = []
        self._outfit_index: Dict[str, _OutfitBone] = {}
        self._bind_cache: Optional[Dict[str, np.ndarray]] = None
        self._rest = np.zeros((0, 4, 4), dtype=np.float32)
        self._rest_views: List[np.ndarray] = []

        self._kin_names: List[str] = []
        self._dyn_names: List[str] = []
        self._kin = np.zeros((0, 4, 4), dtype=np.float32)
        self._dyn = np.zeros((0, 4, 4), dtype=np.float32)
        self._kin_ptr = None
        self._dyn_ptr = None
        self._dyn_views: List[np.ndarray] = []
        self._second_pass: List[_OutfitBone] = []
        self._dyn_outfit: Dict[str, int] = {}

    # -- state ---------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when the native library can be loaded."""
        try:
            get_native(self._native_dir)
        except SmpBridgeError:
            return False
        return True

    @property
    def error(self) -> Optional[str]:
        return _native_error

    @property
    def has_physics(self) -> bool:
        return bool(self._dyn_names) and not self._failed

    @property
    def is_active(self) -> bool:
        return self.enabled and self.has_physics

    @property
    def dynamic_bones(self) -> List[str]:
        return list(self._dyn_names)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self._release_world()

    # -- NIF sources -----------------------------------------------------

    def add_source(
        self,
        key: str,
        nif_path: Path | str,
        xml_ref: str,
        nodes: Dict[str, Tuple[Optional[str], np.ndarray]],
        shapes: Sequence["ShapeMeshData"] = (),
    ) -> None:
        self._sources[key] = _NifSource(Path(nif_path), xml_ref, dict(nodes), tuple(shapes))
        self._rebuild_outfit_order()

    def remove_source(self, key: str) -> None:
        if self._sources.pop(key, None) is not None:
            self._rebuild_outfit_order()

    def has_outfit_bone(self, name: str) -> bool:
        return name in self._outfit_index

    def outfit_bone_matches_skin(self, name: str, skin_to_bone: np.ndarray, tolerance: float = 0.05) -> bool:
        """True when ``bind_world(name) @ skin_to_bone`` is the identity, i.e. the NIF's node tree reproduces the
        shape's bind pose and the bone can be skinned through its own matrix instead of a skeleton ancestor."""
        if name not in self._outfit_index:
            return False
        if self._bind_cache is None:
            self._bind_cache = self._bind_pose()
        product = self._bind_cache[name] @ np.asarray(skin_to_bone, dtype=np.float32)
        return bool(np.abs(product - np.eye(4, dtype=np.float32)).max() <= tolerance)

    def _rebuild_outfit_order(self) -> None:
        """Order every non-skeleton NIF node parents-first, keeping only those that chain up to a skeleton bone."""
        merged: Dict[str, Tuple[Optional[str], np.ndarray]] = {}
        for source in self._sources.values():
            for name, entry in source.nodes.items():
                if self.anim_engine.resolve_bone_name(name) is None:
                    merged.setdefault(name, entry)

        ordered: List[_OutfitBone] = []
        placed: Dict[str, str] = {}
        pending = dict(merged)
        progressed = True
        while pending and progressed:
            progressed = False
            for name, (parent, local) in list(pending.items()):
                if parent is None:
                    del pending[name]
                    progressed = True
                    continue
                skeleton_parent = self.anim_engine.resolve_bone_name(parent)
                resolved = skeleton_parent if skeleton_parent is not None else (parent if parent in placed else None)
                if resolved is None:
                    continue
                ordered.append(_OutfitBone(name, resolved, np.asarray(local, dtype=np.float32), len(ordered)))
                placed[name] = resolved
                del pending[name]
                progressed = True

        self._outfit = ordered
        self._outfit_index = {bone.name: bone for bone in ordered}
        self._rest = np.zeros((len(ordered), 4, 4), dtype=np.float32)
        self._rest_views = [self._rest[i] for i in range(len(ordered))]
        self._bind_cache = None
        self._signature = None

    def _bind_pose(self) -> Dict[str, np.ndarray]:
        bind = dict(self.anim_engine.bind_pose_matrices)
        for bone in self._outfit:
            bind[bone.name] = bind[bone.parent] @ bone.local
        return bind

    def _rest_pass(self, pose: Dict[str, np.ndarray]) -> None:
        for bone, view in zip(self._outfit, self._rest_views):
            np.matmul(pose[bone.parent], bone.local, out=view)
            pose[bone.name] = view

    # -- native world ----------------------------------------------------

    def _release_world(self) -> None:
        if _native is not None:
            _native.destroy()
        self._kin_names = []
        self._dyn_names = []
        self._kin = np.zeros((0, 4, 4), dtype=np.float32)
        self._dyn = np.zeros((0, 4, 4), dtype=np.float32)
        self._kin_ptr = self._dyn_ptr = None
        self._dyn_views = []
        self._second_pass = []
        self._dyn_outfit = {}

    def set_visible_sources(self, keys: Iterable[str]) -> None:
        self._visible = frozenset(k for k in keys if k in self._sources)

    def set_collision_xmls(self, paths: Iterable[Path | str]) -> None:
        self._collision_xmls = tuple(Path(path) for path in paths)

    def refresh(self) -> bool:
        """Rebuild the native world if the visible outfits, XMLs or enable flag changed.

        Returns True when the set of simulated bones changed.
        """
        signature = (self.enabled, self._visible, self._collision_xmls)
        if signature == self._signature:
            return False
        self._signature = signature
        before = list(self._dyn_names)
        self._release_world()
        self._failed = False
        if self.enabled and self._visible:
            self._build()
        return before != self._dyn_names

    def _build(self) -> None:
        self.diagnostics = []
        try:
            native = get_native(self._native_dir)
        except SmpBridgeError as exc:
            self._warn(str(exc))
            return

        xml_files: Dict[Tuple[str, str], Path] = {}
        for path in self._collision_xmls:
            if not path.is_file():
                self._warn(f"collision XML '{path}' was not found")
                continue
            xml = path.resolve()
            xml_files.setdefault((str(xml).casefold(), ""), xml)
        for key in sorted(self._visible):
            source = self._sources[key]
            xml = resolve_smp_xml(source.path, source.xml_ref)
            if xml is None:
                if source.xml_ref:
                    self._warn(f"{source.path.name}: physics XML '{source.xml_ref}' was not found")
                continue
            xml_files.setdefault((str(xml).casefold(), key), xml)
        if not xml_files:
            return

        bind = self._bind_pose()
        candidates = list(bind)

        native.destroy()
        names = list(bind)
        status = native.set_bind_pose(names, np.stack([bind[n] for n in names]).astype(np.float32))
        if status != STATUS_OK:
            self._warn(f"SmpSetBindPose failed with status {status}")
            native.destroy()
            return

        for key in sorted(self._visible):
            for shape in self._sources[key].shapes:
                if not shape.is_skinned or not shape.bone_names or len(shape.positions) == 0:
                    continue
                try:
                    status = native.register_mesh(key, shape)
                except (SmpBridgeError, ValueError) as exc:
                    self._warn(f"{key}: cannot register collision mesh {shape.name}: {exc}")
                    continue
                if status != STATUS_OK:
                    self._warn(f"{key}: collision mesh {shape.name} rejected with status {status}")

        loaded = 0
        for (_, scope), xml in xml_files.items():
            status = native.set_mesh_scope(scope)
            if status != STATUS_OK:
                self._warn(f"SmpSetMeshScope failed with status {status}")
                native.destroy()
                return
            status = native.load_xml(xml, candidates)
            if status == STATUS_OK:
                loaded += 1
            self.diagnostics.extend(native.drain_log())
            if status != STATUS_OK:
                log.info("SMP XML %s was not loaded (status %d)", xml.name, status)
                if status != STATUS_NO_BODIES:
                    self._warn(f"SMP XML {xml.name} failed with status {status}; physics world discarded")
                    native.destroy()
                    return
        for line in self.diagnostics:
            log.info("SMP: %s", line)
        if loaded == 0:
            native.destroy()
            return

        kin_names = native.body_names(KIND_KINEMATIC)
        dyn_names = native.body_names(KIND_DYNAMIC)
        if not dyn_names:
            native.destroy()
            return

        self._kin_names = kin_names
        self._dyn_names = dyn_names
        self._kin = np.ascontiguousarray(
            np.stack([bind[n] for n in kin_names]) if kin_names else np.zeros((0, 4, 4)), dtype=np.float32
        )
        self._dyn = np.ascontiguousarray(np.stack([bind[n] for n in dyn_names]), dtype=np.float32)
        self._kin_ptr = native.pointer(self._kin)
        self._dyn_ptr = native.pointer(self._dyn)
        self._dyn_views = [self._dyn[i] for i in range(len(dyn_names))]
        self._dyn_outfit = {n: i for i, n in enumerate(dyn_names) if n in self._outfit_index}

        # Outfit bones that must be recomputed after the step: the simulated ones and their descendants.
        simulated = set(dyn_names)
        affected: set = set()
        second: List[_OutfitBone] = []
        for bone in self._outfit:
            if bone.name in simulated or bone.parent in simulated or bone.parent in affected:
                affected.add(bone.name)
                second.append(bone)
        self._second_pass = second

    def _warn(self, message: str) -> None:
        self.diagnostics.append(message)
        log.warning("SMP: %s", message)

    # -- per-frame ---------------------------------------------------------

    def reset(self) -> None:
        """Snap the cloth back to its reference pose with zero velocity (seek, loop wrap, new clip)."""
        if self.has_physics and _native is not None:
            _native.reset()

    def collision_debug_lines(self) -> np.ndarray:
        if not self.is_active or _native is None:
            return np.zeros((0, 3), dtype=np.float32)
        return primitive_debug_lines(_native.primitives())

    def update(
        self, pose: Dict[str, np.ndarray], dt: float, kinematic_pose: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Add outfit-bone matrices to ``pose`` (modified in place) and, when active, step SMP and merge it in."""
        shared = {}
        if self.cbpc is not None and self.cbpc.is_active:
            active = {name.casefold() for name in self.cbpc.active_bones}
            shared = {name: pose[name] for name in self._dyn_names if name.casefold() in active and name in pose}
        if self._outfit:
            self._rest_pass(pose)
            pose.update(shared)
        if not self.is_active or _native is None:
            return pose

        if kinematic_pose is not None:
            kinematic_pose = dict(kinematic_pose)
            for bone in self._outfit:
                kinematic_pose[bone.name] = kinematic_pose[bone.parent] @ bone.local
        else:
            kinematic_pose = pose
        kin = self._kin
        for i, name in enumerate(self._kin_names):
            matrix = kinematic_pose.get(name)
            if matrix is None:
                self._warn(f"Animated pose is missing SMP kinematic bone {name}; cloth physics disabled")
                self._failed = True
                return pose
            kin[i] = matrix
        status = _native.step(dt, self._kin_ptr, len(self._kin_names), self._dyn_ptr, len(self._dyn_names))
        if status != STATUS_OK:
            self._warn(f"SmpStep failed with status {status}; cloth physics disabled until the outfits change")
            self._failed = True
            return pose

        for name, view in zip(self._dyn_names, self._dyn_views):
            if name not in shared:
                pose[name] = view
        for bone in self._second_pass:
            if bone.name in self._dyn_outfit:
                continue
            view = self._rest_views[bone.slot]
            np.matmul(pose[bone.parent], bone.local, out=view)
            pose[bone.name] = view
        return pose

    def close(self) -> None:
        self._release_world()
        self._signature = None
