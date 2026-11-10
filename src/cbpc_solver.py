"""
Pure NumPy CBPC particle solver (spring-mass, semi-implicit Euler) with hard offset clamping,
sphere/capsule collision push-out and skinning-delta reconstruction.

Config values are CBPC-native (tiny); they are converted to simulation units with the scale
constants below (mass is 1, lengths are Skyrim units):

* spring constant   k = stiffness * STIFFNESS_SCALE
* damping           d = damping   * DAMPING_SCALE
* gravity force     F = [0, 0, -gravityBias * GRAVITY_SCALE] for non-breast particles
* breast gravity    a static downward offset, independent of spring dynamics
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Tuned for natural 3BA flesh dynamics without artificial rigidification
STIFFNESS_SCALE = 8_500.0  # k ~ 160-180 (natural 2-3 Hz wobble frequency)
DAMPING_SCALE = 150.0  # Under-damped (allows 1-2 smooth elastic oscillations)
GRAVITY_SCALE = 12_000.0  # Natural idle sag (~0.8 to 1.5 Skyrim units)
BREAST_REST_GRAVITY_SCALE = 1.2  # Match the previous equilibrium at default softness.
MIN_STIFFNESS = 1e-3
FIXED_DT = 1.0 / 240.0
MAX_SUBSTEPS = 8
_EPS = 1e-9

_CHAIN_RE = re.compile(r"^([lr]) (breast|butt)")
_BREAST_SEGMENT_RE = re.compile(r"^([lr]) breast0?([123])$")
_ABDOMEN_COLLIDERS = frozenset(("belly", "pelvis", "spine", "spine0", "spine1", "spine2"))
# Give distal segments extra drop while bounding the added separation at high gravity.
_SAG_GRADIENT = 2.1
_MAX_SEGMENT_SAG = 2.0


def _normalize(name: str) -> str:
    """Lower-case node key without ``NPC`` prefix or ``[Tag]`` suffix."""
    s = re.sub(r"\s*\[[^\]]*\]", "", name.strip().lstrip("@")).strip()
    s = re.sub(r"^npc\s+", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).casefold()


def _chain_of(key: str) -> Optional[Tuple[str, str]]:
    """(side, part) for breast / butt chains, e.g. ``l breast02`` -> ``("l", "breast")``."""
    m = _CHAIN_RE.match(key)
    return (m.group(1), m.group(2)) if m else None


@dataclass
class ParticleParams:
    """Solver parameters for one dynamic bone (``name`` is the animated pose / skeleton node name)."""

    name: str
    stiffness: float = 0.02
    damping: float = 0.03
    gravity: float = 0.0
    maxoffset: float = 2.0
    collision_maxoffset: Optional[float] = None
    amplitude: float = 1.0
    mass: float = 1.0
    softness: float = 1.0
    gravity_as_offset: bool = False


@dataclass
class ColliderShapes:
    """Collision shapes of one animated collider node, in that node's local space."""

    node: str
    p1: np.ndarray  # (S, 3)
    r1: np.ndarray  # (S,)
    p2: np.ndarray  # (S, 3) (== p1 for spheres)
    r2: np.ndarray  # (S,)


def translation_matrices(deltas: np.ndarray) -> np.ndarray:
    """Batch of 4x4 translation matrices for an (N, 3) array of deltas."""
    mats = np.repeat(np.eye(4, dtype=np.float32)[None], len(deltas), axis=0)
    mats[:, :3, 3] = deltas
    return mats


class CbpcSolver:
    def __init__(self) -> None:
        self.names: List[str] = []
        self._index: Dict[str, int] = {}
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.anim = np.zeros((0, 3))
        self._active = np.zeros(0, dtype=bool)
        self._k = np.zeros(0)
        self._d = np.zeros(0)
        self._g = np.zeros(0)
        self._rest_offset = np.zeros((0, 3))
        self._m = np.ones(0)
        self._maxoff = np.zeros(0)
        self._colmax = np.zeros(0)
        self._initialized = np.zeros(0, dtype=bool)
        self._breast_chains: List[List[Tuple[int, int]]] = []
        self._breast_pairs: List[Tuple[int, int]] = []
        self._collision_contact = np.zeros(0, dtype=bool)

        self._colliders: List[ColliderShapes] = []
        self._shape_mask = np.zeros((0, 0), dtype=bool)  # (N, S) True = shape may push particle
        self._shape_owner = np.zeros(0, dtype=np.int64)
        self._shape_p1 = np.zeros((0, 3))
        self._shape_p2 = np.zeros((0, 3))
        self._shape_r1 = np.zeros(0)
        self._shape_r2 = np.zeros(0)
        self._collision_rest_dist = np.zeros((0, 0))
        self._affected_owner = np.zeros(0, dtype=np.int64)
        self._affected_p1 = np.zeros((0, 3))
        self._affected_p2 = np.zeros((0, 3))
        self._affected_r1 = np.zeros(0)
        self._affected_r2 = np.zeros(0)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_particles(self, params: Sequence[ParticleParams]) -> None:
        """Replace the particle set; particles that persist keep their position / velocity."""
        old = {
            n: (self.pos[i].copy(), self.vel[i].copy(), self._initialized[i])
            for n, i in self._index.items()
        }
        self.names = [p.name for p in params]
        self._index = {n: i for i, n in enumerate(self.names)}
        n = len(params)
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.anim = np.zeros((n, 3))
        self._initialized = np.zeros(n, dtype=bool)
        chains: Dict[str, List[Tuple[int, int]]] = {}
        for i, name in enumerate(self.names):
            match = _BREAST_SEGMENT_RE.fullmatch(_normalize(name))
            if match:
                chains.setdefault(match.group(1), []).append((int(match.group(2)), i))
        self._breast_chains = [sorted(chain) for chain in chains.values()]
        self._collision_contact = np.zeros(n, dtype=bool)
        for i, name in enumerate(self.names):
            if name in old:
                self.pos[i], self.vel[i], self._initialized[i] = old[name]
        self.update_params(params)
        self.set_affected_shapes({})
        self.set_colliders(self._colliders)

    def update_params(self, params: Sequence[ParticleParams]) -> None:
        """Re-apply parameters with decoupled softness scaling, preserving particle state."""
        by_name = {p.name: p for p in params}
        n = len(self.names)
        self._k = np.zeros(n)
        self._d = np.zeros(n)
        self._g = np.zeros(n)
        self._rest_offset = np.zeros((n, 3))
        self._m = np.ones(n)
        self._maxoff = np.zeros(n)
        self._colmax = np.zeros(n)
        self._active = np.zeros(n, dtype=bool)
        self._breast_pairs = []
        for i, name in enumerate(self.names):
            p = by_name[name]
            s = max(p.softness, 0.0)
            self._active[i] = s > 0.0
            k_mult = 1.0 / (0.3 + 0.7 * s) if s > 0 else 10.0
            d_mult = 1.0 / (0.5 + 0.5 * s) if s > 0 else 10.0
            g_mult = 0.8 + 0.4 * s
            limit_amp = 0.6 + 0.6 * s
            self._k[i] = max(p.stiffness, MIN_STIFFNESS) * STIFFNESS_SCALE * k_mult
            self._d[i] = max(p.damping, 0.0) * DAMPING_SCALE * d_mult
            if p.gravity_as_offset:
                base_k = max(p.stiffness, MIN_STIFFNESS) * STIFFNESS_SCALE
                self._rest_offset[i, 2] = -max(p.gravity, 0.0) * GRAVITY_SCALE * BREAST_REST_GRAVITY_SCALE / base_k
            else:
                self._g[i] = p.gravity * GRAVITY_SCALE * g_mult
            self._m[i] = max(p.mass, _EPS)
            amp = p.amplitude * limit_amp
            self._maxoff[i] = max(p.maxoffset * amp, 0.0)
            col = p.collision_maxoffset if p.collision_maxoffset else p.maxoffset * 1.5
            self._colmax[i] = max(col * amp, self._maxoff[i])

        for chain in self._breast_chains:
            for (prev_segment, parent), (segment, child) in zip(chain, chain[1:]):
                if not (by_name[self.names[parent]].gravity_as_offset and by_name[self.names[child]].gravity_as_offset):
                    continue
                self._breast_pairs.append((parent, child))
                sag = max(-self._rest_offset[parent, 2], 0.0)
                increment = min(sag * _SAG_GRADIENT, _MAX_SEGMENT_SAG) * (segment - prev_segment)
                self._rest_offset[child, 2] = min(self._rest_offset[child, 2], -sag - increment)
                self._k[child] = min(self._k[child], self._k[parent])
                self._maxoff[child] = max(self._maxoff[child], self._colmax[parent])
                self._colmax[child] = max(self._colmax[child], self._maxoff[child])

    def set_colliders(
        self, colliders: Sequence[ColliderShapes], mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Register collider nodes. ``mask`` is an (N_particles, N_collider_nodes) bool array
        (True = that collider node may push that particle); default is all-True.
        """
        self._colliders = list(colliders)
        owners: List[int] = []
        p1, p2, r1, r2 = [], [], [], []
        for ci, c in enumerate(self._colliders):
            owners.extend([ci] * len(c.r1))
            p1.append(c.p1)
            p2.append(c.p2)
            r1.append(c.r1)
            r2.append(c.r2)
        self._shape_owner = np.asarray(owners, dtype=np.int64)
        self._shape_p1 = np.concatenate(p1) if p1 else np.zeros((0, 3))
        self._shape_p2 = np.concatenate(p2) if p2 else np.zeros((0, 3))
        self._shape_r1 = np.concatenate(r1) if r1 else np.zeros(0)
        self._shape_r2 = np.concatenate(r2) if r2 else np.zeros(0)
        n = len(self.names)
        if mask is None or np.shape(mask) != (n, len(self._colliders)):
            mask = np.ones((n, len(self._colliders)), dtype=bool)
        mask = np.asarray(mask, dtype=bool) & self._chain_compatibility()
        self._shape_mask = mask[:, self._shape_owner] if len(owners) else np.zeros((n, 0), dtype=bool)
        self._collision_rest_dist = np.full((len(self._affected_owner), len(owners)), np.nan)

    def set_affected_shapes(self, shapes: Dict[str, ColliderShapes]) -> None:
        """Set particle-local collision geometry; particles without shapes remain points."""
        owners, p1, p2, r1, r2 = [], [], [], [], []
        for i, name in enumerate(self.names):
            shape = shapes.get(name)
            if shape is None or len(shape.r1) == 0:
                owners.append(i)
                p1.append(np.zeros((1, 3)))
                p2.append(np.zeros((1, 3)))
                r1.append(np.zeros(1))
                r2.append(np.zeros(1))
            else:
                owners.extend([i] * len(shape.r1))
                p1.append(shape.p1)
                p2.append(shape.p2)
                r1.append(shape.r1)
                r2.append(shape.r2)
        self._affected_owner = np.asarray(owners, dtype=np.int64)
        self._affected_p1 = np.concatenate(p1) if p1 else np.zeros((0, 3))
        self._affected_p2 = np.concatenate(p2) if p2 else np.zeros((0, 3))
        self._affected_r1 = np.concatenate(r1) if r1 else np.zeros(0)
        self._affected_r2 = np.concatenate(r2) if r2 else np.zeros(0)
        self._collision_rest_dist = np.full((len(owners), len(self._shape_owner)), np.nan)

    def _chain_compatibility(self) -> np.ndarray:
        """(N, C) bool: False where a collider is the particle itself or belongs to its own chain."""
        keys = [_normalize(c.node) for c in self._colliders]
        compat = np.ones((len(self.names), len(keys)), dtype=bool)
        for i, name in enumerate(self.names):
            own = _normalize(name)
            own_chain = _chain_of(own)
            breast = _BREAST_SEGMENT_RE.fullmatch(own)
            for j, key in enumerate(keys):
                if (key == own or (own_chain is not None and _chain_of(key) == own_chain)
                    or (breast is not None and breast.group(2) == "3" and key in _ABDOMEN_COLLIDERS)):
                    compat[i, j] = False
        return compat

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def reset(self, anim_positions: Optional[np.ndarray] = None) -> None:
        """Snap particles to their animated positions and zero all velocities."""
        if anim_positions is not None:
            self.anim = np.asarray(anim_positions, dtype=np.float64).reshape(-1, 3).copy()
        self.pos = self.anim.copy()
        self.vel = np.zeros_like(self.pos)
        self._initialized[:] = True
        self._collision_rest_dist.fill(np.nan)

    def step(
        self,
        anim_positions: np.ndarray,
        dt: float,
        collider_world: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Advance the simulation by ``dt`` seconds against animated targets ``anim_positions`` (N, 3).
        ``collider_world`` maps collider node name -> animated 4x4 world matrix.
        Returns world displacements including any static rest offsets.
        """
        n = len(self.names)
        if n == 0:
            return np.zeros((0, 3))
        self.anim = np.asarray(anim_positions, dtype=np.float64).reshape(n, 3)

        fresh = ~self._initialized
        if fresh.any():
            self.pos[fresh] = self.anim[fresh]
            self.vel[fresh] = 0.0
            self._initialized[fresh] = True

        inactive = ~self._active
        self.pos[inactive] = self.anim[inactive]
        self.vel[inactive] = 0.0
        if not self._active.any():
            return self._rest_offset.copy()

        shapes = self._world_shapes(collider_world)
        remaining = max(0.0, float(dt))
        substeps = min(MAX_SUBSTEPS, int(np.ceil(remaining / FIXED_DT - 1e-9)))
        if substeps > 0:
            h = remaining / substeps
            for _ in range(substeps):
                self._substep(h, shapes)
        else:
            self._apply_collision_and_clamp(shapes)
        return self.pos - self.anim + self._rest_offset

    def _substep(self, h: float, shapes) -> None:
        d = self.pos - self.anim
        f_spring = -self._k[:, None] * d  # F = -k (P - P_anim)
        f_damp = -self._d[:, None] * self.vel  # F = -d V
        f_grav = np.zeros_like(self.pos)
        f_grav[:, 2] = -self._g  # F = [0, 0, -g_bias]
        total = f_spring + f_damp + f_grav
        self.vel += (total / self._m[:, None]) * h
        self.pos += self.vel * h
        self.pos[~self._active] = self.anim[~self._active]
        self.vel[~self._active] = 0.0
        self._apply_collision_and_clamp(shapes)

    def _apply_collision_and_clamp(self, shapes) -> None:
        self._clamp(self._maxoff)
        self._collision_contact[:] = False
        if self._collide(shapes):
            self._clamp(self._colmax)
        if self._follow_breast_chain() and self._collide(shapes):
            self._clamp(self._colmax)

    def _follow_breast_chain(self) -> bool:
        moved = False
        for parent, child in self._breast_pairs:
            # External contact takes precedence over chain following.
            if not (self._active[parent] and self._active[child]) or self._collision_contact[child]:
                continue
            parent_delta = self.pos[parent, 2] - self.anim[parent, 2]
            child_delta = self.pos[child, 2] - self.anim[child, 2]
            if child_delta <= parent_delta:
                continue
            limit = self._colmax[child] if self._collision_contact[parent] else self._maxoff[child]
            vertical = max(-limit, min(parent_delta, limit))
            self.pos[child, 2] = self.anim[child, 2] + vertical
            moved = True
            horizontal = self.pos[child, :2] - self.anim[child, :2]
            length = np.linalg.norm(horizontal)
            available = np.sqrt(max(limit * limit - vertical * vertical, 0.0))
            if length > available:
                self.pos[child, :2] = self.anim[child, :2] + horizontal * (available / length)
            self.vel[child, 2] = min(self.vel[child, 2], self.vel[parent, 2])
        return moved

    def _clamp(self, limit: np.ndarray) -> None:
        disp = self.pos - self.anim
        dist = np.linalg.norm(disp, axis=1)
        over = dist > limit
        if not over.any():
            return
        normal = disp[over] / dist[over, None]
        self.pos[over] = self.anim[over] + normal * limit[over, None]
        # Drop outward velocity so particles do not keep pressing into the limit
        outward = np.einsum("ij,ij->i", self.vel[over], normal)
        self.vel[over] -= np.maximum(outward, 0.0)[:, None] * normal

    def _world_shapes(self, collider_world: Optional[Dict[str, np.ndarray]]):
        if collider_world is None or len(self._shape_owner) == 0:
            return None
        s = len(self._shape_owner)
        mats = np.zeros((len(self._colliders), 4, 4))
        present = np.zeros(len(self._colliders), dtype=bool)
        for ci, c in enumerate(self._colliders):
            m = collider_world.get(c.node)
            if m is not None:
                mats[ci] = m
                present[ci] = True
        m_s = mats[self._shape_owner]
        ones = np.ones((s, 1))
        a = np.einsum("sij,sj->si", m_s, np.concatenate([self._shape_p1, ones], axis=1))[:, :3]
        b = np.einsum("sij,sj->si", m_s, np.concatenate([self._shape_p2, ones], axis=1))[:, :3]
        affected_mats = np.repeat(np.eye(4)[None], len(self.names), axis=0)
        affected_mats[:, :3, 3] = self.anim
        for i, name in enumerate(self.names):
            if name in collider_world:
                affected_mats[i] = collider_world[name]
        mats = affected_mats[self._affected_owner]
        ones = np.ones((len(mats), 1))
        affected_a = np.einsum("sij,sj->si", mats, np.concatenate([self._affected_p1, ones], axis=1))[:, :3]
        affected_b = np.einsum("sij,sj->si", mats, np.concatenate([self._affected_p2, ones], axis=1))[:, :3]
        return a, b, present[self._shape_owner], affected_a, affected_b

    @staticmethod
    def _closest_segments(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray):
        u, v = p2 - p1, q2 - q1
        w = p1[:, None, :] - q1[None, :, :]
        aa = np.einsum("ij,ij->i", u, u)[:, None]
        bb = u @ v.T
        cc = np.einsum("ij,ij->i", v, v)[None, :]
        dd = np.einsum("nsi,ni->ns", w, u)
        ee = np.einsum("nsi,si->ns", w, v)
        denom = aa * cc - bb * bb
        s = np.where(denom > _EPS, (bb * ee - cc * dd) / np.maximum(denom, _EPS), 0.0)
        s = np.clip(s, 0.0, 1.0)
        t = np.clip((bb * s + ee) / np.maximum(cc, _EPS), 0.0, 1.0)
        s = np.clip((bb * t - dd) / np.maximum(aa, _EPS), 0.0, 1.0)
        t = np.clip((bb * s + ee) / np.maximum(cc, _EPS), 0.0, 1.0)
        difference = w + s[..., None] * u[:, None, :] - t[..., None] * v[None, :, :]
        return difference, s, t

    def _collide(self, shapes) -> bool:
        """Push particles out of overlapping shapes along the contact normal."""
        if shapes is None:
            return False
        a, b, present, affected_a, affected_b = shapes
        if not present.any():
            return False
        owner = self._affected_owner
        rest_a = affected_a + self._rest_offset[owner]
        rest_b = affected_b + self._rest_offset[owner]
        delta = (self.pos - self.anim)[owner]
        diff, s, t = self._closest_segments(rest_a + delta, rest_b + delta, a, b)
        radius = (self._affected_r1[:, None] + s * (self._affected_r2 - self._affected_r1)[:, None]
                  + self._shape_r1[None] + t * (self._shape_r2 - self._shape_r1)[None])
        # Preserve initial overlap, but do not shrink the collision radius as a collider approaches.
        fresh = np.isnan(self._collision_rest_dist) & present[None, :]
        if fresh.any():
            rest_diff, _, _ = self._closest_segments(rest_a, rest_b, a, b)
            rest_dist = np.linalg.norm(rest_diff, axis=2)
            self._collision_rest_dist[fresh] = rest_dist[fresh]
        radius = np.minimum(radius, self._collision_rest_dist)
        dist = np.linalg.norm(diff, axis=2)
        pen = radius - dist
        active = (pen > 0.0) & self._shape_mask[owner] & present[None, :] & self._active[owner, None]
        if not active.any():
            return False
        np.logical_or.at(self._collision_contact, owner, active.any(axis=1))
        normal = diff / np.maximum(dist, _EPS)[..., None]
        normal[dist < _EPS] = (0.0, 1.0, 0.0)
        pushes = np.where(active[..., None], normal * pen[..., None], 0.0).sum(axis=1)
        np.add.at(self.pos, owner, pushes)
        return True

    # ------------------------------------------------------------------
    # Skinning
    # ------------------------------------------------------------------

    def apply_to_pose(
        self, pose: Dict[str, np.ndarray], deltas: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Blend per-bone delta translation matrices into the skinning pose palette:
        ``M' = T(delta) @ M``. Returns a new dict; ``pose`` is not modified.
        """
        out = dict(pose)
        present = [i for i, name in enumerate(self.names) if name in pose]
        if not present:
            return out
        t_mats = translation_matrices(deltas[present])
        for row, i in enumerate(present):
            out[self.names[i]] = t_mats[row] @ pose[self.names[i]]
        return out
