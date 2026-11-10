import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.anim_engine import AnimEngine
from src.cbpc_manager import CBPC_CONFIG_PATH, GRAVITY_MAX, CbpcManager
from src.cbpc_solver import CbpcSolver, ColliderShapes, ParticleParams

ENGINE = AnimEngine()
MESH_BONES = ["NPC Root [Root]", "L Breast01", "L Breast02", "NPC L Butt", "NPC R Butt", "Bogus Bone"]


def test_flat_config_with_gravity():
    mgr = CbpcManager(ENGINE)
    data = json.loads(CBPC_CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(data) == {"bones", "collision"}
    assert mgr.bones["NPC L Breast01"]["gravity"] > mgr.bones["NPC L Breast03"]["gravity"] > 0
    assert mgr.bones["NPC L Butt"]["gravity"] > 0 and mgr.bones["NPC Belly"]["gravity"] > 0

def test_bone_intersection_skips_missing():
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", ["Bogus Bone"])
    assert not mgr.has_physics
    mgr.set_mesh_bones("body", MESH_BONES)
    assert set(mgr.active_bones) == {"L Breast01", "L Breast02", "NPC L Butt", "NPC R Butt"}
    pose = ENGINE.get_bind_pose()
    assert mgr.update(pose, 1 / 60) is not pose
    mgr.remove_mesh_bones("body")
    assert not mgr.has_physics and mgr.update(pose, 1 / 60) is pose


def test_gravity_slider_scales_breasts_only():
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", MESH_BONES)
    pose = ENGINE.get_bind_pose()
    mgr.reset(pose)
    pos_before = mgr.solver.pos.copy()
    gravity = lambda name: mgr.solver._g[mgr.solver.names.index(name)]
    offset = lambda name: mgr.solver._rest_offset[mgr.solver.names.index(name), 2]
    butt = gravity("NPC L Butt")
    k, d = mgr.solver._k.copy(), mgr.solver._d.copy()
    mgr.set_gravity(0.0)  # Armor: no gravity
    assert offset("L Breast01") == 0.0 and gravity("NPC L Butt") == butt
    mgr.set_gravity(1.0)  # More Gravity: compiled config
    full = offset("L Breast01")
    assert full < 0 and gravity("L Breast01") == 0.0
    mgr.set_gravity(2.0)
    assert np.isclose(offset("L Breast01"), 2 * full)
    mgr.set_gravity(9.0)  # clamped
    assert mgr.gravity == GRAVITY_MAX and gravity("NPC L Butt") == butt
    assert np.array_equal(pos_before, mgr.solver.pos)
    assert np.array_equal(mgr.solver._k, k) and np.array_equal(mgr.solver._d, d)


@pytest.mark.parametrize("softness", [0.0, 0.05, 1.0, 2.0, 5.0])
def test_breast_gravity_is_an_immediate_offset_independent_of_softness(softness):
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", ["L Breast01", "NPC L Butt"])
    mgr.set_softness("breasts", softness)
    pose = ENGINE.get_bind_pose()
    mgr.set_gravity(0.0)
    mgr.reset(pose)
    baseline = mgr.update(pose, 0.0)
    positions = mgr.solver.pos.copy()
    velocities = mgr.solver.vel.copy()
    stiffness = mgr.solver._k.copy()
    damping = mgr.solver._d.copy()
    mgr.set_gravity(3.0)
    heavy = mgr.update(pose, 0.0)
    diff = heavy["L Breast01"][:3, 3] - baseline["L Breast01"][:3, 3]
    assert diff[2] == pytest.approx(mgr.solver._rest_offset[0, 2], abs=1e-5)
    assert diff[2] < -0.1 and np.allclose(diff[:2], 0)
    assert np.allclose(heavy["NPC L Butt"], baseline["NPC L Butt"])
    assert np.array_equal(mgr.solver.pos, positions)
    assert np.array_equal(mgr.solver.vel, velocities)
    assert np.array_equal(mgr.solver._k, stiffness)
    assert np.array_equal(mgr.solver._d, damping)


def test_gravity_does_not_change_breast_oscillation():
    outputs = []
    for gravity in (0.0, 5.0):
        mgr = CbpcManager(ENGINE)
        mgr.set_mesh_bones("body", ["L Breast01"])
        mgr.set_gravity(gravity)
        mgr.set_softness("breasts", 2.0)
        pose = {name: mat.copy() for name, mat in ENGINE.get_bind_pose().items()}
        mgr.reset(pose)
        # Isolate the spring response from collider geometry.
        mgr.solver.set_colliders([])
        trajectory = []
        initial = pose["L Breast01"][:3, 3].copy()
        for frame in range(90):
            pose["L Breast01"][:3, 3] = initial + (np.sin(frame / 8), 0, np.cos(frame / 7))
            result = mgr.update(pose, 1 / 60)
            trajectory.append(result["L Breast01"][:3, 3].copy() - pose["L Breast01"][:3, 3])
        outputs.append((np.array(trajectory), mgr.solver.vel.copy(), mgr.solver._rest_offset.copy()))
    assert np.allclose(outputs[0][0] - outputs[0][2], outputs[1][0] - outputs[1][2], atol=1e-5)
    assert np.allclose(outputs[0][1], outputs[1][1])


def test_breast_offset_affects_collision_geometry():
    s = CbpcSolver()
    s.set_particles([ParticleParams("a", stiffness=0.02, gravity=0.01,
                                    gravity_as_offset=True, maxoffset=10.0)])
    s.set_colliders([ColliderShapes("c", np.zeros((1, 3)), np.array([3.0]),
                                    np.zeros((1, 3)), np.array([3.0]))])
    target = np.array([[4.0, 0.0, 1.0]])
    s.reset(target)
    s.pos[0, 0] = 1.0
    physical_before = s.pos + s._rest_offset
    assert s._collide(s._world_shapes({"c": np.eye(4)}))
    physical_after = s.pos + s._rest_offset
    assert np.linalg.norm(physical_after[0]) == pytest.approx(3.0)
    assert np.linalg.norm(physical_before[0]) < 3.0


@pytest.mark.parametrize("factor", [1 / 3, 2 / 3, 1.0, 5.0])
def test_real_breast_chains_sag_in_order_and_have_headroom(factor):
    mgr = CbpcManager(ENGINE)
    names = [f"{side} Breast0{segment}" for side in ("L", "R") for segment in (1, 2, 3)]
    mgr.set_mesh_bones("body", names)
    mgr.set_gravity(factor)
    pose = ENGINE.get_bind_pose()
    mgr.reset(pose)
    palette = mgr.update(pose, 0)
    solver = mgr.solver
    for side in ("L", "R"):
        indices = [solver.names.index(f"{side} Breast0{n}") for n in (1, 2, 3)]
        drops = -solver._rest_offset[indices, 2]
        assert 0 < drops[0] < drops[1] < drops[2]
        assert solver._k[indices[2]] <= solver._k[indices[0]]
        assert solver._maxoff[indices[2]] >= solver._colmax[indices[1]]
        if factor >= 2 / 3:
            assert palette[f"{side} Breast03"][2, 3] <= palette[f"{side} Breast02"][2, 3]
        assert all(np.isfinite(palette[f"{side} Breast0{n}"]).all() for n in (1, 2, 3))


def test_breast_chain_constraint_follows_downward_bounce():
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", ["L Breast01", "L Breast02", "L Breast03"])
    mgr.set_gravity(1.0)
    pose = ENGINE.get_bind_pose()
    mgr.reset(pose)
    s = mgr.solver
    s.set_colliders([])
    parent, mid, tip = [s.names.index(f"L Breast0{i}") for i in (1, 2, 3)]
    s.pos[parent, 2] -= 2.0
    s.vel[parent, 2] = -1.0
    result = s.step(mgr._anim_positions(pose), 0.0)
    assert result[tip, 2] <= result[mid, 2] + 1e-10
    assert result[mid, 2] <= result[parent, 2] + 1e-10
    assert s.vel[mid, 2] <= s.vel[parent, 2]
    assert s.vel[tip, 2] <= s.vel[mid, 2]
    assert np.all(np.linalg.norm(s.pos - s.anim, axis=1) <= s._maxoff + 1e-8)


def test_gravity_preserves_full_chain_dynamics_without_collisions():
    simulations = []
    for gravity in (0.0, 2 / 3, 5.0):
        mgr = CbpcManager(ENGINE)
        mgr.set_mesh_bones("body", ["L Breast01", "L Breast02", "L Breast03"])
        mgr.set_gravity(gravity)
        mgr.set_softness("breasts", 2.0)
        mgr.solver.set_colliders([])
        pose = ENGINE.get_bind_pose()
        mgr.reset(pose)
        target = mgr._anim_positions(pose)
        trace = []
        for frame in range(120):
            moved = target.copy()
            moved[:, 2] += np.sin(frame / 10) * 2.0
            moved[:, 0] += np.cos(frame / 12)
            mgr.solver.step(moved, 1 / 60)
            trace.append((mgr.solver.pos - mgr.solver.anim).copy())
        simulations.append(np.array(trace))
    assert np.allclose(simulations[0], simulations[1])
    assert np.allclose(simulations[0], simulations[2])


def test_breast_chain_external_collision_takes_priority():
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", ["L Breast01", "L Breast02", "L Breast03"])
    mgr.set_gravity(1.0)
    s = mgr.solver
    mgr.reset(ENGINE.get_bind_pose())
    mid, tip = [s.names.index(f"L Breast0{i}") for i in (2, 3)]
    s.pos[mid, 2] -= 1.0
    tip_before = s.pos[tip].copy()
    s._collision_contact[tip] = True
    s._follow_breast_chain()
    assert np.array_equal(s.pos[tip], tip_before)


def test_distal_breast_excludes_abdomen_but_retains_external_colliders():
    s = CbpcSolver()
    s.set_particles([ParticleParams(f"NPC L Breast0{n}") for n in (1, 2, 3)])
    names = ["NPC Belly", "NPC Pelvis [Pelv]", "NPC Spine1", "NPC L Hand [LHnd]", "NPC R Breast01"]
    shape = lambda name: ColliderShapes(name, np.zeros((1, 3)), np.array([3.0]),
                                         np.zeros((1, 3)), np.array([3.0]))
    s.set_colliders([shape(name) for name in names])
    assert s._shape_mask[:2, :3].all()
    assert not s._shape_mask[2, :3].any()
    assert s._shape_mask[:, 3:].all()

    s.set_colliders([shape("NPC Belly")])
    target = np.tile([4.0, 0.0, 0.0], (3, 1))
    s.reset(target)
    s.pos[:] = (1.0, 0.0, 0.0)
    assert s._collide(s._world_shapes({"NPC Belly": np.eye(4)}))
    assert np.allclose(s.pos[:2, 0], 3.0)
    assert s.pos[2, 0] == 1.0


def test_incomplete_breast_chain_does_not_create_particles():
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", ["L Breast01", "L Breast03"])
    mgr.set_gravity(1.0)
    assert mgr.active_bones == ["L Breast01", "L Breast03"]
    assert mgr.solver._rest_offset[1, 2] <= mgr.solver._rest_offset[0, 2]
    mgr.set_mesh_bones("body", ["L Breast03"])
    assert mgr.active_bones == ["L Breast03"]
    assert mgr.solver._rest_offset[0, 2] < 0


def test_negative_breast_gravity_cannot_lift():
    s = CbpcSolver()
    s.set_particles([ParticleParams("L Breast01", gravity=-1.0, gravity_as_offset=True)])
    assert s._rest_offset[0, 2] == 0.0


def test_solver_spring_clamp_and_reset():
    s = CbpcSolver()
    s.set_particles([ParticleParams("a", stiffness=0.02, damping=0.05, gravity=0.0, maxoffset=1.0)])
    s.reset(np.zeros((1, 3)))
    d = s.step(np.array([[10.0, 0, 0]]), 1 / 60)
    assert np.linalg.norm(d[0]) <= 1.2 + 1e-9  # hard clamp at default softness
    for _ in range(600):
        d = s.step(np.array([[10.0, 0, 0]]), 1 / 60)
    assert np.linalg.norm(d[0]) < 1e-2  # settles on the animated target
    s.step(np.array([[10.0, 0, 5.0]]), 1 / 60)
    s.reset(np.array([[10.0, 0, 5.0]]))
    assert np.allclose(s.pos, [[10.0, 0, 5.0]]) and not s.vel.any()


def test_solver_gravity_sags():
    s = CbpcSolver()
    s.set_particles([ParticleParams("a", stiffness=0.02, damping=0.1, gravity=0.05, maxoffset=5.0)])
    s.reset(np.zeros((1, 3)))
    for _ in range(600):
        d = s.step(np.zeros((1, 3)), 1 / 60)
    assert d[0, 2] < -0.1 and abs(d[0, 0]) < 1e-6


def test_collision_push_out_and_mask():
    shape = ColliderShapes("c", np.array([[0.0, 0, 0]]), np.array([3.0]), np.array([[0.0, 0, 0]]), np.array([3.0]))
    pose = {"c": np.eye(4)}
    for allowed in (True, False):
        s = CbpcSolver()
        s.set_particles([ParticleParams("a", stiffness=0.02, maxoffset=5.0, collision_maxoffset=8.0)])
        s.set_colliders([shape], np.array([[allowed]]))
        s.reset(np.array([[4.0, 0, 0]]))
        s.pos[0] = (1.0, 0, 0)  # newly penetrating the sphere
        s.step(np.array([[4.0, 0, 0]]), 1 / 60, pose)
        assert (np.linalg.norm(s.pos[0]) >= 3.0 - 1e-6) == allowed


def test_rest_overlap_does_not_push():
    shape = ColliderShapes("c", np.zeros((1, 3)), np.array([3.0]), np.zeros((1, 3)), np.array([3.0]))
    s = CbpcSolver()
    s.set_particles([ParticleParams("a", stiffness=0.02, maxoffset=5.0)])
    s.set_colliders([shape], np.array([[True]]))
    s.reset(np.array([[1.0, 0, 0]]))
    d = s.step(np.array([[1.0, 0, 0]]), 1 / 60, {"c": np.eye(4)})
    assert np.linalg.norm(d) < 1e-9


def test_collision_data_compiled_in_json():
    mgr = CbpcManager(ENGINE)
    assert "l butt" in mgr.collision_affected and "l breast01" in mgr.collision_affected["l butt"]
    shapes = mgr.collision_nodes["l rearthigh"]["shapes"]
    assert shapes[0]["p2"] is not None and shapes[0]["r1"] > 0


def test_compiled_collision_exclusions_allow_hands_not_breast_colliders():
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", ["L Breast01", "L Breast03", "NPC L Butt", "NPC Belly", "VaginaB1"])
    s = mgr.solver

    def allowed(particle):
        i = s.names.index(particle)
        return {c.node for j, c in enumerate(s._colliders)
                if np.any(s._shape_mask[i] & (s._shape_owner == j))}

    for breast in ("L Breast01", "L Breast03"):
        nodes = allowed(breast)
        assert "NPC L Hand [LHnd]" in nodes and "NPC R Forearm [RLar]" in nodes
        assert "L Breast01" not in nodes and "R Breast01" not in nodes
    assert "NPC L Thigh [LThg]" not in allowed("NPC L Butt")
    assert "NPC L Thigh [LThg]" in allowed("VaginaB1")
    assert not allowed("NPC Belly")
    assert not any(c.node in ("NPC Belly", "NPC Pelvis [Pelv]") for c in s._colliders)
    assert sum(c.node == "L Breast01" for c in s._colliders) == 1
    assert not any(c.node == "NPC L Breast01" for c in s._colliders)


def test_compiled_hand_collider_pushes_breast_when_it_approaches():
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", ["L Breast01"])
    solver = mgr.solver
    collider = next(c for c in solver._colliders if c.node == "NPC L Hand [LHnd]")
    target = np.array([[0.0, 0.0, 0.0]])
    solver.reset(target)

    def hand_at(x):
        matrix = np.eye(4)
        matrix[:3, 3] = np.array([x, 0, 0]) - collider.p1[0]
        return {collider.node: matrix}

    solver.step(target, 0, hand_at(20.0))
    out = solver.step(target, 0, hand_at(1.0))
    assert solver._collision_contact[0]
    assert out[0, 0] < -0.5


def test_compiled_breast_shape_collides_at_its_surface():
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", ["L Breast01"])
    s = mgr.solver
    collider = next(c for c in s._colliders if c.node == "NPC L Hand [LHnd]")
    target = np.zeros((1, 3))
    s.reset(target)
    center = s._affected_p1[0] + s._rest_offset[0]
    hand = np.eye(4)
    hand[:3, 3] = center + [20.0, 0.0, 0.0] - collider.p1[0]
    s.step(target, 0, {collider.node: hand})
    hand[:3, 3] = center + [s._affected_r1[0] + collider.r1[0] - 0.5, 0.0, 0.0] - collider.p1[0]
    result = s.step(target, 0, {collider.node: hand})
    assert result[0, 0] < -0.4
    assert s._collision_contact[0]


def test_moving_collider_pushes_animated_target_and_rest_overlap_is_stable():
    s = CbpcSolver()
    s.set_particles([ParticleParams("a", maxoffset=10.0)])
    s.set_colliders([ColliderShapes("hand", np.zeros((1, 3)), np.array([3.0]),
                                    np.zeros((1, 3)), np.array([3.0]))])
    target = np.array([[0.0, 0.0, 0.0]])
    s.reset(target)
    hand = np.eye(4)
    hand[0, 3] = 4.0
    assert np.allclose(s.step(target, 0, {"hand": hand}), 0)
    target[0, 0] = 1.0
    hand[0, 3] = 2.0
    delta = s.step(target, 0, {"hand": hand})
    assert delta[0, 0] == pytest.approx(-2.0)

    s.reset(np.array([[1.0, 0, 0]]))
    hand[0, 3] = 0.0
    assert np.allclose(s.step(np.array([[1.0, 0, 0]]), 0, {"hand": hand}), 0)
    assert np.allclose(s.step(np.array([[1.0, 0, 0]]), 0, {"hand": hand}), 0)


def test_affected_sphere_collides_before_hand_reaches_bone_center():
    s = CbpcSolver()
    s.set_particles([ParticleParams("a", maxoffset=10.0)])
    s.set_affected_shapes({"a": ColliderShapes("a", np.array([[0.0, 2.0, 0.0]]), np.array([3.0]),
                                                np.array([[0.0, 2.0, 0.0]]), np.array([3.0]))})
    s.set_colliders([ColliderShapes("hand", np.zeros((1, 3)), np.array([1.0]),
                                    np.zeros((1, 3)), np.array([1.0]))])
    target = np.array([[0.0, 0.0, 0.0]])
    s.reset(target)
    hand = np.eye(4)
    hand[:3, 3] = (10.0, 2.0, 0.0)
    s.step(target, 0, {"hand": hand})
    hand[0, 3] = 3.0
    result = s.step(target, 0, {"hand": hand})
    assert result[0, 0] == pytest.approx(-1.0)


def test_affected_and_collider_capsules_resolve_overlap():
    s = CbpcSolver()
    s.set_particles([ParticleParams("a", maxoffset=10.0)])
    s.set_affected_shapes({"a": ColliderShapes("a", np.array([[-2.0, 0, 0]]), np.array([1.0]),
                                                np.array([[2.0, 0, 0]]), np.array([1.0]))})
    s.set_colliders([ColliderShapes("hand", np.array([[0.0, -2.0, 0]]), np.array([1.0]),
                                    np.array([[0.0, 2.0, 0]]), np.array([1.0]))])
    target = np.zeros((1, 3))
    s.reset(target)
    hand = np.eye(4)
    hand[2, 3] = 4.0
    s.step(target, 0, {"hand": hand})
    hand[2, 3] = 1.0
    assert s.step(target, 0, {"hand": hand})[0, 2] == pytest.approx(-1.0)


def test_late_collider_uses_first_present_pose_for_rest_overlap():
    s = CbpcSolver()
    s.set_particles([ParticleParams("a", maxoffset=10.0)])
    s.set_colliders([ColliderShapes("hand", np.zeros((1, 3)), np.array([3.0]),
                                    np.zeros((1, 3)), np.array([3.0]))])
    target = np.array([[1.0, 0.0, 0.0]])
    s.reset(target)
    assert np.allclose(s.step(target, 0, {}), 0)
    assert np.allclose(s.step(target, 0, {"hand": np.eye(4)}), 0)
    moved = np.eye(4)
    moved[0, 3] = 0.5
    assert s.step(target, 0, {"hand": moved})[0, 0] == pytest.approx(0.5)


def test_collision_rest_geometry_is_cached_until_reset_or_reconfiguration(monkeypatch):
    s = CbpcSolver()
    s.set_particles([ParticleParams("a", maxoffset=10.0)])
    shape = lambda name: ColliderShapes(name, np.zeros((1, 3)), np.array([3.0]),
                                         np.zeros((1, 3)), np.array([3.0]))
    s.set_colliders([shape("hand"), shape("late")])
    target = np.array([[4.0, 0.0, 0.0]])
    s.reset(target)
    original = s._closest_segments
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(s, "_closest_segments", counted)
    hand = np.eye(4)
    s.step(target, 0, {"hand": hand})
    assert len(calls) == 2
    s.step(target, 0, {"hand": hand})
    assert len(calls) == 3
    s.step(target, 0, {"hand": hand, "late": hand})
    assert len(calls) == 5
    s.step(target, 0, {"hand": hand, "late": hand})
    assert len(calls) == 6
    s.reset(target)
    s.step(target, 0, {"hand": hand})
    assert len(calls) == 8
    s.set_affected_shapes({"a": shape("a")})
    s.step(target, 0, {"hand": hand})
    assert len(calls) == 10
    s.set_colliders([shape("hand")])
    s.step(target, 0, {"hand": hand})
    assert len(calls) == 12

def test_same_chain_colliders_excluded():
    def shape(node):
        return ColliderShapes(node, np.zeros((1, 3)), np.array([3.0]), np.zeros((1, 3)), np.array([3.0]))

    s = CbpcSolver()
    s.set_particles([ParticleParams("NPC L Breast01"), ParticleParams("L Breast02"), ParticleParams("NPC R Butt")])
    s.set_colliders([shape("L Breast01"), shape("NPC L Breast"), shape("R Breast01"), shape("NPC L Thigh [LThg]"), shape("NPC R Butt"), shape("NPC L Butt")])
    mask = s._shape_mask  # one shape per collider node
    assert not mask[0, :2].any() and mask[0, 2:4].all()  # own-side breast chain ignored
    assert not mask[1, :2].any() and mask[1, 2]
    assert not mask[2, 4] and mask[2, 5] and mask[2, 3]  # own butt ignored, opposite side pushes


def test_collision_keeps_tangential_velocity():
    s = CbpcSolver()
    s.set_particles([ParticleParams("a", stiffness=0.02, maxoffset=9.0, collision_maxoffset=9.0)])
    s.set_colliders([ColliderShapes("c", np.zeros((1, 3)), np.array([3.0]), np.zeros((1, 3)), np.array([3.0]))], np.array([[True]]))
    s.reset(np.array([[4.0, 0, 0]]))
    s.pos[0] = (2.0, 0, 0)
    s.vel[0] = (-5.0, 7.0, 0)
    s._collide(s._world_shapes({"c": np.eye(4)}))
    assert s.vel[0, 1] == 7.0 and s.vel[0, 0] == -5.0

def test_overlay_states():
    from PySide6.QtWidgets import QApplication, QWidget
    from src.cbpc_manager import GRAVITY_MIN
    from src.cbpc_overlay import CbpcOverlay, STEPS_PER_UNIT

    app = QApplication.instance() or QApplication([])
    host = QWidget()
    host.resize(600, 400)
    ov = CbpcOverlay(host, GRAVITY_MIN, GRAVITY_MAX)
    ov.sync(True, False, 1.0)
    assert not ov.checkbox.isEnabled() and not ov.checkbox.isChecked() and not ov.slider.isEnabled()
    ov.sync(True, True, 1.0, 4)
    assert ov.checkbox.isChecked() and ov.slider.isEnabled()
    assert ov.slider.minimum() == 0 and ov.slider.maximum() == round(GRAVITY_MAX * STEPS_PER_UNIT)
    assert ov.slider.value() == 30 and ov.label.text() == "1.00x"
    ov.reposition()
    assert ov.x() + ov.width() < 600 and ov.y() + ov.height() < 400 and ov.x() > 100
    seen = []
    ov.gravityChanged.connect(seen.append)
    ov.slider.setValue(0)
    assert seen == [0.0] and ov.label.text() == "0.00x"
    assert set(ov.softness_sliders) == {"breasts", "butt", "belly"}
    assert all(s.minimum() == 0 and s.maximum() == 100 for s in ov.softness_sliders.values())
    seen_softness = []
    ov.softnessChanged.connect(lambda group, value: seen_softness.append((group, value)))
    for group, value in (("breasts", 100), ("butt", 40), ("belly", 0)):
        ov.softness_sliders[group].setValue(value)
    assert seen_softness == [("breasts", 5.0), ("butt", 2.0), ("belly", 0.0)]
    assert ov.softness_labels["breasts"].text() == "5.00x"
    values = {"breasts": 0.2, "butt": 2.0, "belly": 3.0}
    ov.sync(True, True, 1.0, 4, values)
    assert [ov.softness_sliders[g].value() for g in values] == [4, 40, 60]
    assert len(seen_softness) == 3
    ov.sync(False, True, 1.0, 4, values)
    assert all(not slider.isEnabled() for slider in ov.softness_sliders.values())
    assert app is not None

@pytest.mark.parametrize(
    "softness,k_mult,d_mult,g_mult,limit_amp",
    [
        (0.0, 10.0, 10.0, 0.8, 0.6),
        (0.05, 1 / 0.335, 1 / 0.525, 0.82, 0.63),
        (1.0, 1.0, 1.0, 1.2, 1.2),
        (2.0, 1 / 1.7, 1 / 1.5, 1.6, 1.8),
        (5.0, 1 / 3.8, 1 / 3.0, 2.8, 3.6),
    ],
)
def test_softness_scales_parameters_and_limits(softness, k_mult, d_mult, g_mult, limit_amp):
    from src.cbpc_solver import DAMPING_SCALE, GRAVITY_SCALE, STIFFNESS_SCALE

    params = [
        ParticleParams("a", stiffness=0.02, damping=0.03, gravity=0.01,
                       maxoffset=2.0, collision_maxoffset=4.0, amplitude=0.5, mass=2.0, softness=softness),
        ParticleParams("b", maxoffset=2.0, amplitude=0.5, softness=softness),
    ]
    solver = CbpcSolver()
    solver.set_particles(params)
    assert solver._k[0] == pytest.approx(0.02 * STIFFNESS_SCALE * k_mult)
    assert solver._d[0] == pytest.approx(0.03 * DAMPING_SCALE * d_mult)
    assert solver._g[0] == pytest.approx(0.01 * GRAVITY_SCALE * g_mult)
    assert solver._m[0] == 2.0
    assert solver._maxoff == pytest.approx([limit_amp, limit_amp])
    assert solver._colmax == pytest.approx([2 * limit_amp, 1.5 * limit_amp])


def test_softness_updates_manager_and_zero_disables_motion():
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", MESH_BONES + ["R Breast01", "NPC Belly", "NPC L FrontThigh"])
    s = mgr.solver
    k, d, g = s._k.copy(), s._d.copy(), s._g.copy()
    pos = s.pos.copy()
    breast = np.array(["Breast" in n for n in s.names])
    assert sum(breast) == 3 and "NPC Belly" in s.names and "NPC L FrontThigh" in s.names
    mgr.set_softness("breasts", 2.0)
    assert np.allclose(s._k[breast], k[breast] / 1.7)
    assert np.all(s._g[breast] == 0)
    assert np.allclose(s._d[breast], d[breast] / 1.5) and np.array_equal(s.pos, pos)
    assert np.array_equal(s._k[~breast], k[~breast])
    mgr.set_softness("butt", 99.0)
    assert mgr.softness["butt"] == 5.0
    for name in ("NPC L Butt", "NPC R Butt"):
        i = s.names.index(name)
        assert s._k[i] == pytest.approx(k[i] / 3.8)
    mgr.set_softness("belly", 0.0)
    pose = ENGINE.get_bind_pose()
    mgr.reset(pose)
    s.pos += 1.0
    s.vel += 1.0
    mgr.update(pose, 1 / 60)
    i = s.names.index("NPC Belly")
    assert np.allclose(s.pos[i], s.anim[i]) and not s.vel[i].any()
    assert np.any(np.linalg.norm(s.pos[breast] - s.anim[breast], axis=1) > 0)
    assert s._k[s.names.index("NPC L FrontThigh")] == k[s.names.index("NPC L FrontThigh")]
    with pytest.raises(ValueError):
        mgr.set_softness("unknown", 1.0)


def test_softness_changes_sag_and_damping_ratio():
    sags, damping_ratios = [], []
    for softness in (0.2, 1.0, 2.0, 5.0):
        s = CbpcSolver()
        s.set_particles([ParticleParams("a", stiffness=0.02, damping=0.1, gravity=0.01, maxoffset=100.0, softness=softness)])
        s.reset(np.zeros((1, 3)))
        for _ in range(1200):
            s.step(np.zeros((1, 3)), 1 / 60)
        sags.append(-s.pos[0, 2])
        damping_ratios.append(s._d[0] / (2 * np.sqrt(s._k[0] * s._m[0])))
        assert sags[-1] == pytest.approx(s._g[0] / s._k[0], rel=0.02)
    assert all(a < b for a, b in zip(sags, sags[1:]))
    assert all(a > b for a, b in zip(damping_ratios, damping_ratios[1:]))


@pytest.mark.parametrize("group,affected", [
    ("breasts", {"L Breast01", "L Breast02", "R Breast01"}),
    ("butt", {"NPC L Butt", "NPC R Butt"}),
    ("belly", {"NPC Belly"}),
])
def test_each_softness_group_changes_only_its_own_particles(group, affected):
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", MESH_BONES + ["R Breast01", "NPC Belly", "NPC L FrontThigh"])
    before = np.column_stack((mgr.solver._k, mgr.solver._d, mgr.solver._g,
                              mgr.solver._maxoff, mgr.solver._colmax))
    mgr.set_softness(group, 2.0)
    after = np.column_stack((mgr.solver._k, mgr.solver._d, mgr.solver._g,
                             mgr.solver._maxoff, mgr.solver._colmax))
    changed = {name for name, old, new in zip(mgr.solver.names, before, after)
               if not np.allclose(old, new)}
    assert changed == affected


def test_zero_softness_particles_ignore_collisions_while_others_move():
    s = CbpcSolver()
    s.set_particles([
        ParticleParams("off", softness=0.0, gravity=0.1, maxoffset=10),
        ParticleParams("on", softness=1.0, gravity=0.1, maxoffset=10),
    ])
    s.set_colliders([ColliderShapes("collider", np.zeros((1, 3)), np.array([3.0]),
                                    np.zeros((1, 3)), np.array([3.0]))])
    target = np.array([[4.0, 0, 0], [4.0, 0, 0]])
    s.reset(target)
    s.pos[:] = (1.0, 0, 0)
    s.vel[:] = (1.0, 0, 0)
    delta = s.step(target, 1 / 60, {"collider": np.eye(4)})
    assert np.array_equal(delta[0], np.zeros(3)) and not s.vel[0].any()
    assert np.linalg.norm(delta[1]) > 0
    s.step(target, 0, {"collider": np.eye(4)})
    assert np.array_equal(s.pos[0], target[0])


def test_softness_changes_response_to_movement():
    positions = []
    for softness in (0.2, 1.0, 5.0):
        s = CbpcSolver()
        s.set_particles([ParticleParams("a", stiffness=0.02, damping=0.03, maxoffset=10, softness=softness)])
        s.reset(np.zeros((1, 3)))
        for _ in range(6):
            s.step(np.array([[1.0, 0.0, 0.0]]), 1 / 60)
        positions.append(s.pos[0, 0])
    assert positions[0] > positions[1] > positions[2]


def test_softness_extremes_remain_finite():
    mgr = CbpcManager(ENGINE)
    mgr.set_mesh_bones("body", MESH_BONES)
    pose = ENGINE.get_bind_pose()
    for softness in (0.05, 5.0):
        for group in ("breasts", "butt", "belly", "other"):
            mgr.set_softness(group, softness)
        mgr.reset(pose)
        moving = {name: matrix.copy() for name, matrix in pose.items()}
        for frame in range(120):
            for name in mgr.active_bones:
                moving[name][:3, 3] = pose[name][:3, 3] + (np.sin(frame / 10), 0, 0)
            mgr.update(moving, 0.1)
            assert np.isfinite(mgr.solver.pos).all() and np.isfinite(mgr.solver.vel).all()


def test_cbpc_settings_roundtrip_in_config(tmp_path):
    from src.config import AppConfig

    path = tmp_path / "config.json"
    cfg = AppConfig()
    values = {"breasts": 3.5, "butt": 0.0, "belly": 1.5, "other": 1.0}
    cfg.cbpc.enabled, cfg.cbpc.gravity, cfg.cbpc.softness = False, 2.0, values
    cfg.save(path)
    loaded = AppConfig.load(path)
    assert (loaded.cbpc.enabled, loaded.cbpc.gravity, loaded.cbpc.softness) == (False, 2.0, values)
    path.write_text('{"cbpc": {"gravity": 99, "softness": "x", "enabled": 1}}', encoding="utf-8")
    bad = AppConfig.load(path)
    assert bad.cbpc.gravity == 5.0 and all(v == 1.0 for v in bad.cbpc.softness.values()) and bad.cbpc.enabled is True
    path.write_text('{"cbpc": {"bounce": 5.0}}', encoding="utf-8")
    migrated = AppConfig.load(path)
    assert all(v == 0.2 for v in migrated.cbpc.softness.values())
    migrated.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))["cbpc"]
    assert data["softness"] == dict.fromkeys(values, 0.2) and "bounce" not in data
    path.write_text('{"cbpc": {"bounce": 0.0}}', encoding="utf-8")
    assert all(v == 0.0 for v in AppConfig.load(path).cbpc.softness.values())
    path.write_text('{"cbpc": {"bounce": 2.0, "softness": 3.0}}', encoding="utf-8")
    assert all(v == 3.0 for v in AppConfig.load(path).cbpc.softness.values())
    path.write_text('{"cbpc": {"softness": {"breasts": 2, "butt": -1, "belly": 9}}}', encoding="utf-8")
    assert AppConfig.load(path).cbpc.softness == {"breasts": 2.0, "butt": 0.0, "belly": 5.0, "other": 1.0}