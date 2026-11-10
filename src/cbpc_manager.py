"""
CBPC physics manager: bone filtering, gravity control and per-frame pose adjustment.

Active physical bones = (bones bound to the loaded mesh(es)) ∩ (bones in the compiled `assets/CBPConfig.json`),
restricted to bones the animation engine can drive (``AnimEngine.resolve_bone_name`` covers the
skeleton nodes, including the XPMSSE / Havok bone table ``anim_engine.SKYRIM_HAVOK_BONES``), so
every particle has an animated target. Missing bones are skipped silently.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .anim_engine import AnimEngine
from .cbpc_solver import CbpcSolver, ColliderShapes, ParticleParams
from .config import SOFTNESS_GROUPS

log = logging.getLogger("dibellas_wardrobe.cbpc_manager")

_RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
CBPC_CONFIG_PATH = _RESOURCE_ROOT / "assets" / "CBPConfig.json"
# Breast gravity slider: 1.0 is the compiled config (More Gravity), 0.0 is Armor.
GRAVITY_MIN = 0.0
GRAVITY_MAX = 3.0
DEFAULT_GRAVITY = 1.0 / 3.0
SOFTNESS_MIN = 0.0
SOFTNESS_MAX = 5.0


def is_breast(config_key: str) -> bool:
    return "breast" in normalize_node_key(config_key)

BoneParams = Dict[str, float]

# Config bones whose NIF name differs beyond the NPC prefix / [tag] suffix
_BONE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "VaginaB1": ("VaginaB",),
    "Clitoral1": ("Clitoral",),
}


def normalize_node_key(name: str) -> str:
    """Comparison key for bone names: drops ``@``, ``[Tag]`` and ``NPC`` prefix, case-folded."""
    s = name.strip().lstrip("@").strip()
    s = re.sub(r"\s*\[[^\]]*\]", "", s).strip()
    s = re.sub(r"^npc\s+", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).casefold()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.exception("Failed to load CBPC config %s", path)
        return {}


class CbpcManager:
    def __init__(self, anim_engine: AnimEngine, config_path: Optional[Path | str] = None) -> None:
        self.anim_engine = anim_engine
        data = _load_json(Path(config_path) if config_path else CBPC_CONFIG_PATH)
        self.bones: Dict[str, BoneParams] = data.get("bones", {})
        collision = data.get("collision", {})
        self.collision_affected: Dict[str, List[str]] = collision.get("affected", {})
        self.collision_nodes: Dict[str, dict] = collision.get("colliders", {})
        self.collider_nodes = set(collision.get("collider_nodes", ()))
        if self.collision_nodes and "collider_nodes" not in collision:
            log.error("CBPC collision config %s is missing collider_nodes", config_path or CBPC_CONFIG_PATH)
        self.enabled: bool = True
        self.gravity: float = DEFAULT_GRAVITY
        self.softness: Dict[str, float] = dict.fromkeys(SOFTNESS_GROUPS, 1.0)
        self.solver = CbpcSolver()
        self._mesh_bones: Dict[str, Tuple[str, ...]] = {}
        self._particle_keys: Dict[str, str] = {}  # pose bone name -> canonical config key

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def active_bones(self) -> List[str]:
        return list(self.solver.names)

    @property
    def has_physics(self) -> bool:
        return len(self.solver.names) > 0

    @property
    def is_active(self) -> bool:
        return self.enabled and self.has_physics

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def set_gravity(self, factor: float) -> None:
        """Adjust the breast rest position without changing spring dynamics."""
        self.gravity = max(GRAVITY_MIN, min(GRAVITY_MAX, float(factor)))
        self.solver.update_params(self._particle_params())

    def set_softness(self, group: str, factor: float) -> None:
        """Adjust one group's soft-body response without rebuilding particles."""
        if group not in self.softness:
            raise ValueError(f"Unknown CBPC softness group: {group}")
        self.softness[group] = max(SOFTNESS_MIN, min(SOFTNESS_MAX, float(factor)))
        self.solver.update_params(self._particle_params())

    # ------------------------------------------------------------------
    # Mesh bone intersection
    # ------------------------------------------------------------------

    def set_mesh_bones(self, key: str, bone_names: Iterable[str]) -> None:
        """Register the bones bound to a loaded mesh (``key`` identifies the NIF)."""
        self._mesh_bones[key] = tuple(dict.fromkeys(bone_names))
        self._rebuild()

    def remove_mesh_bones(self, key: str) -> None:
        if self._mesh_bones.pop(key, None) is not None:
            self._rebuild()

    def set_mesh_bones_bulk(self, sources: Dict[str, Iterable[str]]) -> None:
        """Replace all registered mesh bone sets at once (single rebuild)."""
        self._mesh_bones = {k: tuple(dict.fromkeys(v)) for k, v in sources.items()}
        self._rebuild()

    def _rebuild(self) -> None:
        mesh_index: Dict[str, List[str]] = {}
        for names in self._mesh_bones.values():
            for name in names:
                bucket = mesh_index.setdefault(normalize_node_key(name), [])
                if name not in bucket:
                    bucket.append(name)

        particle_keys: Dict[str, str] = {}
        for cfg_key in self.bones:
            for cand in (cfg_key, *_BONE_ALIASES.get(cfg_key, ())):
                for mesh_name in mesh_index.get(normalize_node_key(cand), ()):
                    pose_name = self.anim_engine.resolve_bone_name(mesh_name)
                    if pose_name is not None and pose_name not in particle_keys:
                        particle_keys[pose_name] = cfg_key
        self._particle_keys = particle_keys

        self.solver.set_particles(self._particle_params())
        self._build_colliders()
        log.debug("CBPC active bones: %s", self.solver.names)

    def _particle_params(self) -> List[ParticleParams]:
        out: List[ParticleParams] = []
        for pose_name, cfg_key in self._particle_keys.items():
            p = self.bones.get(cfg_key, {})
            factor = self.gravity if is_breast(cfg_key) else DEFAULT_GRAVITY
            gravity = p.get("gravity", 0.0) * factor
            key = normalize_node_key(cfg_key)
            group = ("breasts" if is_breast(cfg_key) else
                     "butt" if "butt" in key else
                     "belly" if key == "belly" else "other")
            out.append(
                ParticleParams(
                    name=pose_name,
                    stiffness=p.get("stiffness", 0.02),
                    damping=p.get("damping", 0.03),
                    gravity=gravity,
                    maxoffset=p.get("maxoffset", 2.0),
                    collision_maxoffset=p.get("collisionMaxoffset"),
                    amplitude=p.get("amplitude", 1.0),
                    softness=self.softness[group],
                    gravity_as_offset=group == "breasts",
                )
            )
        return out

    def _build_colliders(self) -> None:
        if not self.has_physics:
            self.solver.set_colliders([])
            return

        def make_shapes(node_name: str, raw: List[dict]) -> ColliderShapes:
            p1 = np.array([s["p1"] for s in raw], dtype=np.float64)
            r1 = np.array([s["r1"] for s in raw], dtype=np.float64)
            p2 = np.array([s["p2"] or s["p1"] for s in raw], dtype=np.float64)
            r2 = np.array([s["r2"] if s["p2"] else s["r1"] for s in raw], dtype=np.float64)
            return ColliderShapes(node_name, p1, r1, p2, r2)

        affected_shapes: Dict[str, ColliderShapes] = {}
        for pose_name, config_name in self._particle_keys.items():
            key = normalize_node_key(config_name)
            raw = self.collision_nodes.get(key, {}).get("shapes", [])
            if key in self.collision_affected and raw:
                affected_shapes[pose_name] = make_shapes(pose_name, raw)
        self.solver.set_affected_shapes(affected_shapes)

        by_norm: Dict[str, List[str]] = {}
        for node in self.anim_engine.nodes:
            by_norm.setdefault(normalize_node_key(node.name), []).append(node.name)

        shapes: List[ColliderShapes] = []
        keys: List[str] = []
        for key, collider in self.collision_nodes.items():
            if key not in self.collider_nodes:
                continue
            raw = collider.get("shapes", [])
            if not raw:
                continue
            node_name = self.anim_engine.resolve_bone_name(collider["name"])
            if node_name is None:
                candidates = by_norm.get(key, ())
                node_name = candidates[0] if candidates else None
            if node_name is not None:
                shapes.append(make_shapes(node_name, raw))
                keys.append(key)
        # AffectedNodes entries list self-colliders to ignore, not allowed colliders.
        mask = np.zeros((len(self.solver.names), len(shapes)), dtype=bool)
        for i, pose_name in enumerate(self.solver.names):
            own = normalize_node_key(self._particle_keys[pose_name])
            ignored = self.collision_affected.get(own)
            if ignored is None:
                continue
            for j, key in enumerate(keys):
                if key != own and key not in ignored:
                    mask[i, j] = True
        self.solver.set_colliders(shapes, mask)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _anim_positions(self, pose: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([pose[n][:3, 3] for n in self.solver.names], dtype=np.float64).reshape(-1, 3)

    def reset(self, pose: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Snap particles to the animated pose and zero velocities (loops, scrubbing, new clip)."""
        if not self.has_physics:
            return
        if pose is not None and all(n in pose for n in self.solver.names):
            self.solver.reset(self._anim_positions(pose))
        else:
            self.solver.reset()

    def update(self, pose: Dict[str, np.ndarray], dt: float) -> Dict[str, np.ndarray]:
        """Step physics and return the pose palette with dynamic-bone deltas blended in."""
        if not self.is_active or not all(n in pose for n in self.solver.names):
            return pose
        deltas = self.solver.step(self._anim_positions(pose), dt, pose)
        return self.solver.apply_to_pose(pose, deltas)
