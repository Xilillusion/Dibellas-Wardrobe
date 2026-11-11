import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.anim_engine import AnimEngine
from src.smp_bridge import (
    DLL_NAME,
    KIND_DYNAMIC,
    KIND_KINEMATIC,
    SmpBridgeSolver,
    SmpNative,
    find_native_dir,
    get_native,
    primitive_debug_lines,
    resolve_smp_xml,
)

ENGINE = AnimEngine()
ANCHOR = "NPC Pelvis [Pelv]"
needs_native = pytest.mark.skipif(find_native_dir() is None, reason=f"{DLL_NAME} has not been built")

SKIRT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<system>
  <per-triangle-shape name="VirtualGround"><margin>1</margin></per-triangle-shape>
  <bone name="NPC Pelvis [Pelv]"/>
  <bone-default>
    <mass>0.2</mass>
    <shape type="sphere" radius="0.5"/>
    <inertia x="2" y="2" z="2"/>
    <centerOfMassTransform><basis x="0" y="0" z="0" w="1"/><origin x="0" y="0" z="0"/></centerOfMassTransform>
    <linearDamping>0.6</linearDamping>
    <angularDamping>0.8</angularDamping>
    <friction>0.1</friction>
  </bone-default>
  <bone name="skirt1"/>
  <bone name="skirt2"/>
  <bone name="skirt3"/>
  <generic-constraint-default>
    <frameInB><basis x="0" y="0" z="0" w="1"/><origin x="0" y="0" z="0"/></frameInB>
    <useLinearReferenceFrameA>false</useLinearReferenceFrameA>
    <linearLowerLimit x="0" y="0" z="0"/>
    <linearUpperLimit x="0" y="0" z="0"/>
    <angularLowerLimit x="-0.7" y="-0.7" z="-0.7"/>
    <angularUpperLimit x="0.7" y="0.7" z="0.7"/>
    <linearStiffness x="0" y="0" z="0"/>
    <angularStiffness x="30" y="30" z="30"/>
    <linearDamping x="1" y="1" z="1"/>
    <angularDamping x="2" y="2" z="2"/>
    <linearEquilibrium x="0" y="0" z="0"/>
    <angularEquilibrium x="0" y="0" z="0"/>
  </generic-constraint-default>
  <generic-constraint bodyA="skirt1" bodyB="NPC Pelvis [Pelv]"/>
  <generic-constraint bodyA="skirt2" bodyB="skirt1"/>
  <generic-constraint bodyA="skirt3" bodyB="skirt2"/>
</system>
"""


def _local(z):
    m = np.eye(4, dtype=np.float32)
    m[2, 3] = z
    return m


NODES = {
    "skirt1": (ANCHOR, _local(-6.0)),
    "skirt2": ("skirt1", _local(-6.0)),
    "skirt3": ("skirt2", _local(-6.0)),
}


class StubCbpc:
    """The slice of CbpcManager the bridge reads."""

    def __init__(self, active=(), colliders=()):
        self.is_active = bool(active)
        self.active_bones = list(active)
        self._colliders = list(colliders)

    def collider_specs(self):
        return self._colliders


def make_outfit(tmp_path, xml=SKIRT_XML, ref="meshes\\mod\\skirt.xml"):
    nif_dir = tmp_path / "meshes" / "mod" / "outfits"
    nif_dir.mkdir(parents=True)
    (tmp_path / "meshes" / "mod" / "skirt.xml").write_text(xml, encoding="utf-8")
    nif = nif_dir / "skirt_0.nif"
    nif.touch()
    return nif, ref


def make_solver(tmp_path, cbpc=None, xml=SKIRT_XML):
    nif, ref = make_outfit(tmp_path, xml)
    solver = SmpBridgeSolver(ENGINE, cbpc)
    solver.add_source("skirt", nif, ref, NODES)
    solver.set_visible_sources(["skirt"])
    solver.refresh()
    return solver


def pose_with_anchor_offset(offset=(0.0, 0.0, 0.0)):
    pose = ENGINE.get_bind_pose()
    moved = pose[ANCHOR].copy()
    moved[:3, 3] += offset
    pose[ANCHOR] = moved
    return pose


def simulate(solver, frames, offset_fn, dt=1 / 60):
    last = None
    for i in range(frames):
        last = solver.update(pose_with_anchor_offset(offset_fn(i * dt)), dt)
    return last


def world(solver, pose, name):
    return np.array(pose[name][:3, 3], dtype=np.float64)


@pytest.fixture(autouse=True)
def release_native():
    yield
    if find_native_dir() is not None:
        get_native().destroy()


@needs_native
@pytest.mark.parametrize("dt,status", [(0, 0), (1 / 240, 0), (-1, -3), (float("nan"), -3), (float("inf"), -3)])
def test_native_step_validates_timestep(tmp_path, dt, status):
    solver = make_solver(tmp_path)
    assert solver.has_physics
    native = get_native()
    assert native.step(dt, solver._kin_ptr, len(solver._kin_names), solver._dyn_ptr, len(solver._dyn_names)) == status


@needs_native
@pytest.mark.parametrize("bounce,status", [('x="1" y="2" z="3"', 0), ('x="1" y="invalid" z="3"', -3)])
def test_linear_bounce_xml_is_still_validated(tmp_path, bounce, status):
    native = get_native()
    names = ["ground", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(names), axis=0)
    assert native.set_bind_pose(names, bind) == 0
    xml = tmp_path / "bounce.xml"
    xml.write_text(
        '<system><bone name="ground"/><bone name="cloth"><mass>1</mass></bone>'
        f'<generic-constraint bodyA="ground" bodyB="cloth"><linearBounce {bounce}/>'
        '</generic-constraint></system>',
        encoding="utf-8",
    )
    assert native.load_xml(xml, names) == status


def test_xml_resolution_prefers_extra_data_then_colocated(tmp_path):
    nif, ref = make_outfit(tmp_path)
    expected = (tmp_path / "meshes" / "mod" / "skirt.xml").resolve()
    assert resolve_smp_xml(nif, ref) == expected
    assert resolve_smp_xml(nif, "meshes/mod/skirt.xml") == expected
    assert resolve_smp_xml(nif, "") is None
    colocated = nif.with_suffix(".xml")
    colocated.write_text("<system/>", encoding="utf-8")
    assert resolve_smp_xml(nif, "") == colocated.resolve()
    assert resolve_smp_xml(nif, "meshes/mod/missing.xml") == colocated.resolve()
    assert resolve_smp_xml(nif, ref) == expected


def test_xml_reference_cannot_escape_the_mod_tree(tmp_path):
    nif, _ = make_outfit(tmp_path)
    outside = tmp_path.parent / "outside_smp.xml"
    outside.write_text("<system/>", encoding="utf-8")
    try:
        assert resolve_smp_xml(nif, "..\\..\\..\\..\\outside_smp.xml") is None
        assert resolve_smp_xml(nif, str(outside)) is None
    finally:
        outside.unlink()


def _mesh(name, bone, positions, triangles):
    count = len(positions)
    return SimpleNamespace(
        name=name, bone_names=[bone], positions=np.array(positions, dtype=np.float32),
        bone_skin_matrices=np.eye(4, dtype=np.float32).reshape(1, 4, 4),
        bone_indices=np.zeros((count, 4), dtype=np.int32),
        bone_weights=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (count, 1)),
        triangles=np.array(triangles, dtype=np.int32).reshape(-1, 3), is_skinned=True,
    )


@needs_native
@pytest.mark.parametrize("blocked", [False, True])
def test_skinned_vertex_collides_with_xml_triangle_plate(tmp_path, blocked):
    native = get_native()
    native.destroy()
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    bind[1, 2, 3] = 3
    names = ["ground", "cloth"]
    assert native.set_bind_pose(names, bind) == 0
    plate = _mesh("plate", "ground", [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    point = _mesh("point", "cloth", [[0, 0, 0]], [])
    assert native.register_mesh("actor", plate) == 0
    assert native.register_mesh("actor", point) == 0
    assert native.set_mesh_scope("actor") == 0
    xml = tmp_path / "mesh.xml"
    xml.write_text(
        '<system><bone name="ground"/><bone name="cloth"><mass>1</mass>'
        '<inertia x="1" y="1" z="1"/><linearDamping>0.5</linearDamping></bone>'
        '<per-triangle-shape name="plate"><margin>0.1</margin><penetration>1</penetration>'
        '<tag>body</tag></per-triangle-shape>'
        '<per-vertex-shape name="point"><margin>1</margin>' +
        ('<no-collide-with-tag>body</no-collide-with-tag>' if blocked else '') +
        '</per-vertex-shape></system>', encoding="utf-8"
    )
    assert native.load_xml(xml, names) == 0
    assert not any("not supported" in line for line in native.drain_log())
    kin = np.ascontiguousarray(bind[:1])
    dyn = np.ascontiguousarray(bind[1:])
    for _ in range(60):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    if blocked:
        assert dyn[0, 2, 3] < 0
    else:
        assert dyn[0, 2, 3] > 0.5


@needs_native
@pytest.mark.parametrize("count", [64, 4096])
def test_mesh_skinning_collides_on_serial_and_parallel_paths(tmp_path, count):
    native = get_native()
    native.destroy()
    names = ["ground", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    bind[1, 2, 3] = 0.3
    assert native.set_bind_pose(names, bind) == 0
    plate = _mesh("plate", "ground", [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    points = _mesh("point", "cloth", np.zeros((count, 3), dtype=np.float32), [])
    assert native.register_mesh("actor", plate) == 0
    assert native.register_mesh("actor", points) == 0
    assert native.set_mesh_scope("actor") == 0
    xml = tmp_path / "large.xml"
    xml.write_text(
        '<system><bone name="ground"/><bone name="cloth"><mass>1</mass><gravity-factor>0</gravity-factor>'
        '<inertia x="1" y="1" z="1"/></bone>'
        '<per-triangle-shape name="plate"><margin>0.1</margin><penetration>1</penetration>'
        '</per-triangle-shape><per-vertex-shape name="point"><margin>1</margin>'
        '</per-vertex-shape></system>', encoding="utf-8"
    )
    assert native.load_xml(xml, names) == 0
    kin = np.ascontiguousarray(bind[:1])
    dyn = np.ascontiguousarray(bind[1:])
    for _ in range(10):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    assert dyn[0, 2, 3] > 0.35


@needs_native
@pytest.mark.parametrize("enabled_priority", [1, 3])
def test_disable_tag_priority_selects_only_highest_shape(tmp_path, enabled_priority):
    native = get_native()
    native.destroy()
    names = ["ground", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    bind[1, 2, 3] = 0.3
    assert native.set_bind_pose(names, bind) == 0
    for name in ("near", "far"):
        height = 0 if name == "near" else 20
        assert native.register_mesh(
            "actor", _mesh(name, "ground", [[-10, -10, height], [10, -10, height], [0, 10, height]], [[0, 1, 2]])
        ) == 0
    assert native.register_mesh("actor", _mesh("point", "cloth", [[0, 0, 0]], [])) == 0
    assert native.set_mesh_scope("actor") == 0
    xml = tmp_path / "priority.xml"
    xml.write_text(
        '<system><bone name="ground"/><bone name="cloth"><mass>1</mass><gravity-factor>0</gravity-factor>'
        '<inertia x="1" y="1" z="1"/></bone>'
        f'<per-triangle-shape name="near"><margin>0.1</margin><penetration>1</penetration>'
        f'<disable-tag>group</disable-tag><disable-priority>{enabled_priority}</disable-priority></per-triangle-shape>'
        '<per-triangle-shape name="far"><margin>0.1</margin><penetration>1</penetration>'
        '<disable-tag>group</disable-tag><disable-priority>2</disable-priority></per-triangle-shape>'
        '<per-vertex-shape name="point"><margin>1</margin></per-vertex-shape></system>', encoding="utf-8"
    )
    assert native.load_xml(xml, names) == 0
    kin = np.ascontiguousarray(bind[:1])
    dyn = np.ascontiguousarray(bind[1:])
    for _ in range(10):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    if enabled_priority > 2:
        assert dyn[0, 2, 3] > 0.35
    else:
        assert abs(dyn[0, 2, 3] - 0.3) < 1e-5


@needs_native
def test_mesh_manifolds_survive_later_kinematic_shape_loading(tmp_path):
    native = get_native()
    native.destroy()
    names = ["ground", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    bind[1, 2, 3] = 0.3
    assert native.set_bind_pose(names, bind) == 0
    assert native.register_mesh(
        "actor", _mesh("plate", "ground", [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    ) == 0
    assert native.register_mesh("actor", _mesh("point", "cloth", [[0, 0, 0]], [])) == 0
    assert native.set_mesh_scope("actor") == 0
    mesh_xml = tmp_path / "mesh.xml"
    mesh_xml.write_text(
        '<system><bone name="ground"/><bone name="cloth"><mass>1</mass><gravity-factor>0</gravity-factor>'
        '<inertia x="1" y="1" z="1"/></bone>'
        '<per-triangle-shape name="plate"><margin>0.1</margin><penetration>1</penetration>'
        '</per-triangle-shape><per-vertex-shape name="point"><margin>1</margin>'
        '</per-vertex-shape></system>', encoding="utf-8"
    )
    assert native.load_xml(mesh_xml, names) == 0
    kin = np.ascontiguousarray(bind[:1])
    dyn = np.ascontiguousarray(bind[1:])
    for _ in range(10):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    assert dyn[0, 2, 3] > 0.35
    primitive_xml = tmp_path / "primitive.xml"
    primitive_xml.write_text(
        '<system><bone name="ground"><shape type="sphere"><radius>1</radius></shape></bone></system>',
        encoding="utf-8"
    )
    assert native.load_xml(primitive_xml, names) == 0
    native.reset()
    for _ in range(10):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    assert dyn[0, 2, 3] > 0.35


@needs_native
@pytest.mark.parametrize(
    "penetration,z,threshold,expected",
    [(1, 0.3, 0, 1), (-1, 0.3, 0, -1), (0, -0.3, 0, -1), (1, 0.3, 1, 0)],
)
def test_triangle_penetration_direction_and_weight_threshold(tmp_path, penetration, z, threshold, expected):
    native = get_native()
    native.destroy()
    names = ["ground", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    bind[1, 2, 3] = z
    assert native.set_bind_pose(names, bind) == 0
    assert native.register_mesh(
        "actor", _mesh("plate", "ground", [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    ) == 0
    assert native.register_mesh("actor", _mesh("point", "cloth", [[0, 0, 0]], [])) == 0
    assert native.set_mesh_scope("actor") == 0
    xml = tmp_path / "direction.xml"
    xml.write_text(
        '<system><bone name="ground"/><bone name="cloth"><mass>1</mass><gravity-factor>0</gravity-factor>'
        '<inertia x="1" y="1" z="1"/></bone>'
        f'<per-triangle-shape name="plate"><margin>0.1</margin><penetration>{penetration}</penetration>'
        '<weight-threshold bone="ground">0</weight-threshold></per-triangle-shape>'
        f'<per-vertex-shape name="point"><margin>1</margin><weight-threshold bone="cloth">{threshold}'
        '</weight-threshold></per-vertex-shape></system>', encoding="utf-8"
    )
    assert native.load_xml(xml, names) == 0
    kin = np.ascontiguousarray(bind[:1])
    dyn = np.ascontiguousarray(bind[1:])
    for _ in range(10):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    displacement = dyn[0, 2, 3] - z
    if expected:
        assert displacement * expected > 0.05
    else:
        assert abs(displacement) < 1e-5


@needs_native
def test_solver_registers_meshes_from_visible_nif(tmp_path):
    xml = SKIRT_XML.replace(
        '<per-triangle-shape name="VirtualGround"><margin>1</margin></per-triangle-shape>',
        '<per-triangle-shape name="plate"><margin>0.1</margin></per-triangle-shape>'
        '<per-vertex-shape name="point"><margin>1</margin></per-vertex-shape>'
    )
    nif, ref = make_outfit(tmp_path, xml)
    shapes = [
        _mesh("plate", ANCHOR, [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]]),
        _mesh("point", "skirt1", [[0, 0, 0]], []),
    ]
    solver = SmpBridgeSolver(ENGINE)
    solver.add_source("skirt", nif, ref, NODES, shapes)
    solver.set_visible_sources(["skirt"])
    solver.refresh()
    assert solver.has_physics
    assert not any("collision shape skipped" in line for line in solver.diagnostics)
    solver.update(pose_with_anchor_offset(), 1 / 60)
    assert solver.is_active


@needs_native
def test_unrelated_invalid_mesh_does_not_disable_solver(tmp_path):
    nif, ref = make_outfit(tmp_path, SKIRT_XML)
    invalid = _mesh("unrelated", ANCHOR, [[0, 0, 0]], [])
    invalid.bone_weights = np.zeros((0, 4), dtype=np.float32)
    solver = SmpBridgeSolver(ENGINE)
    solver.add_source("skirt", nif, ref, NODES, [invalid])
    solver.set_visible_sources(["skirt"])
    solver.refresh()
    assert solver.has_physics
    assert any("cannot register collision mesh unrelated" in line for line in solver.diagnostics)


@needs_native
def test_triangle_penetration_broadphase_uses_bone_margin_multiplier(tmp_path):
    native = get_native()
    native.destroy()
    names = ["ground", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    bind[1, 2, 3] = -1.5
    assert native.set_bind_pose(names, bind) == 0
    assert native.register_mesh(
        "actor", _mesh("plate", "ground", [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    ) == 0
    assert native.register_mesh("actor", _mesh("point", "cloth", [[0, 0, 0]], [])) == 0
    assert native.set_mesh_scope("actor") == 0
    xml = tmp_path / "multiplier.xml"
    xml.write_text(
        '<system><bone name="ground"><margin-multiplier>2</margin-multiplier></bone>'
        '<bone name="cloth"><mass>1</mass><gravity-factor>0</gravity-factor>'
        '<inertia x="1" y="1" z="1"/></bone>'
        '<per-triangle-shape name="plate"><margin>0.1</margin><penetration>1</penetration>'
        '</per-triangle-shape><per-vertex-shape name="point"><margin>0.1</margin>'
        '</per-vertex-shape></system>', encoding="utf-8"
    )
    assert native.load_xml(xml, names) == 0
    kin = np.ascontiguousarray(bind[:1])
    dyn = np.ascontiguousarray(bind[1:])
    for _ in range(10):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    assert dyn[0, 2, 3] > -1.45


@needs_native
@pytest.mark.parametrize("reverse", [False, True])
def test_two_triangle_shapes_collide_in_both_orders(tmp_path, reverse):
    native = get_native()
    native.destroy()
    names = ["ground", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    bind[1, 2, 3] = 0.3
    assert native.set_bind_pose(names, bind) == 0
    assert native.register_mesh(
        "actor", _mesh("plate", "ground", [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    ) == 0
    assert native.register_mesh(
        "actor", _mesh("clothmesh", "cloth", [[-0.1, -0.1, 0], [0.1, -0.1, 0], [0, 0.1, 0]], [[0, 1, 2]])
    ) == 0
    assert native.set_mesh_scope("actor") == 0
    shapes = [
        '<per-triangle-shape name="plate"><margin>0.1</margin><penetration>1</penetration></per-triangle-shape>',
        '<per-triangle-shape name="clothmesh"><margin>1</margin></per-triangle-shape>',
    ]
    if reverse:
        shapes.reverse()
    xml = tmp_path / "triangles.xml"
    xml.write_text(
        '<system><bone name="ground"/><bone name="cloth"><mass>1</mass><gravity-factor>0</gravity-factor>'
        '<inertia x="1" y="1" z="1"/></bone>' + ''.join(shapes) + '</system>', encoding="utf-8"
    )
    assert native.load_xml(xml, names) == 0
    kin = np.ascontiguousarray(bind[:1])
    dyn = np.ascontiguousarray(bind[1:])
    for _ in range(10):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    assert dyn[0, 2, 3] > 0.35


@needs_native
def test_triangle_skin_matrix_and_rotated_kinematic_bone(tmp_path):
    native = get_native()
    native.destroy()
    names = ["ground", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    bind[0, :3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    bind[1, 1, 3] = -2.3
    assert native.set_bind_pose(names, bind) == 0
    plate = _mesh("plate", "ground", [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    plate.bone_skin_matrices[0, 2, 3] = 2
    assert native.register_mesh("actor", plate) == 0
    assert native.register_mesh("actor", _mesh("point", "cloth", [[0, 0, 0]], [])) == 0
    assert native.set_mesh_scope("actor") == 0
    xml = tmp_path / "rotated.xml"
    xml.write_text(
        '<system><bone name="ground"/><bone name="cloth"><mass>1</mass><gravity-factor>0</gravity-factor>'
        '<inertia x="1" y="1" z="1"/></bone>'
        '<per-triangle-shape name="plate"><margin>0.1</margin><penetration>1</penetration>'
        '</per-triangle-shape><per-vertex-shape name="point"><margin>1</margin>'
        '</per-vertex-shape></system>', encoding="utf-8"
    )
    assert native.load_xml(xml, names) == 0
    kin = np.ascontiguousarray(bind[:1])
    dyn = np.ascontiguousarray(bind[1:])
    for _ in range(10):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    assert dyn[0, 1, 3] < -2.35


@needs_native
@pytest.mark.parametrize("scope,collides", [("near", True), ("far", False), ("", True)])
def test_mesh_scope_selects_named_source(tmp_path, scope, collides):
    native = get_native()
    native.destroy()
    names = ["ground", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    bind[1, 2, 3] = 0.3
    assert native.set_bind_pose(names, bind) == 0
    assert native.register_mesh(
        "near", _mesh("plate", "ground", [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    ) == 0
    assert native.register_mesh(
        "far", _mesh("plate", "ground", [[-10, -10, 20], [10, -10, 20], [0, 10, 20]], [[0, 1, 2]])
    ) == 0
    for source in ("near", "far"):
        assert native.register_mesh(source, _mesh("point", "cloth", [[0, 0, 0]], [])) == 0
    assert native.set_mesh_scope(scope) == 0
    xml = tmp_path / "scope.xml"
    xml.write_text(
        '<system><bone name="ground"/><bone name="cloth"><mass>1</mass><gravity-factor>0</gravity-factor>'
        '<inertia x="1" y="1" z="1"/></bone>'
        '<per-triangle-shape name="plate"><margin>0.1</margin><penetration>1</penetration>'
        '</per-triangle-shape><per-vertex-shape name="point"><margin>1</margin>'
        '</per-vertex-shape></system>', encoding="utf-8"
    )
    assert native.load_xml(xml, names) == 0
    kin = np.ascontiguousarray(bind[:1])
    dyn = np.ascontiguousarray(bind[1:])
    for _ in range(10):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    assert (dyn[0, 2, 3] > 0.35) == collides


@needs_native
def test_triangle_mesh_tracks_kinematic_animation(tmp_path):
    native = get_native()
    native.destroy()
    names = ["ground", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    bind[1, 2, 3] = 2.5
    assert native.set_bind_pose(names, bind) == 0
    assert native.register_mesh(
        "actor", _mesh("plate", "ground", [[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    ) == 0
    assert native.register_mesh("actor", _mesh("point", "cloth", [[0, 0, 0]], [])) == 0
    assert native.set_mesh_scope("actor") == 0
    xml = tmp_path / "animated.xml"
    xml.write_text(
        '<system><bone name="ground"/><bone name="cloth"><mass>1</mass><gravity-factor>0</gravity-factor>'
        '<inertia x="1" y="1" z="1"/></bone>'
        '<per-triangle-shape name="plate"><margin>0.1</margin><penetration>0</penetration></per-triangle-shape>'
        '<per-vertex-shape name="point"><margin>1</margin></per-vertex-shape></system>', encoding="utf-8"
    )
    assert native.load_xml(xml, names) == 0
    kin = np.ascontiguousarray(bind[:1])
    dyn = np.ascontiguousarray(bind[1:])
    assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    kin[0, 2, 3] = 2
    for _ in range(10):
        assert native.step(1 / 240, native.pointer(kin), 1, native.pointer(dyn), 1) == 0
    assert dyn[0, 2, 3] > 2.55


@needs_native
def test_exports_declare_exact_ctypes_signatures():
    native = SmpNative()
    lib = native._lib
    assert lib.SmpAbiVersion() == 3
    p_float = native.pointer(np.zeros((1, 4, 4), dtype=np.float32)).__class__
    import ctypes

    assert lib.SmpStep.restype is None
    assert lib.SmpStep.argtypes == [ctypes.c_float, p_float, ctypes.c_int, p_float, ctypes.c_int]
    assert lib.SmpLoadXml.restype == ctypes.c_int
    assert lib.SmpLoadXml.argtypes[0] is ctypes.c_char_p and lib.SmpLoadXml.argtypes[2] is ctypes.c_int
    assert lib.SmpReset.argtypes == [] and lib.SmpDestroy.argtypes == []
    with pytest.raises(Exception):
        native.pointer(np.zeros((1, 4, 4), dtype=np.float64))
    native.close()


@needs_native
def test_native_dependency_is_deployed_next_to_the_library():
    directory = find_native_dir()
    assert (directory / "SmpBridge.dll").is_file() and (directory / "libbulletc.dll").is_file()


@needs_native
def test_load_builds_dynamic_chain_and_kinematic_anchor(tmp_path):
    solver = make_solver(tmp_path)
    assert solver.has_physics
    assert solver.dynamic_bones == ["skirt1", "skirt2", "skirt3"]
    native = get_native()
    assert ANCHOR in native.body_names(KIND_KINEMATIC)
    assert native.body_names(KIND_DYNAMIC) == ["skirt1", "skirt2", "skirt3"]


@needs_native
def test_bones_missing_from_the_actor_are_skipped_without_error(tmp_path):
    xml = SKIRT_XML.replace('<bone name="skirt3"/>', '<bone name="skirt3"/><bone name="NoSuchBone"/>')
    xml = xml.replace(
        '</system>', '<generic-constraint bodyA="NoSuchBone" bodyB="skirt3"/></system>'
    )
    solver = make_solver(tmp_path, xml=xml)
    assert solver.dynamic_bones == ["skirt1", "skirt2", "skirt3"]
    assert any("NoSuchBone" in line for line in solver.diagnostics)


@needs_native
def test_hanging_chain_stays_at_rest_and_follows_the_anchor(tmp_path):
    solver = make_solver(tmp_path)
    pose = simulate(solver, 120, lambda t: (0, 0, 0))
    anchor = world(solver, pose, ANCHOR)
    still = {n: world(solver, pose, n) for n in solver.dynamic_bones}
    for _ in range(60):
        pose = solver.update(pose_with_anchor_offset(), 1 / 60)
    for name in solver.dynamic_bones:
        assert np.allclose(world(solver, pose, name), still[name], atol=0.05)

    # Translating the anchor drags the whole chain with it; the damped pendulum then settles within ~10 s.
    pose = simulate(solver, 720, lambda t: (min(t, 0.5) * 20.0, 0.0, 0.0))
    shift = world(solver, pose, ANCHOR) - anchor
    for name in solver.dynamic_bones:
        assert np.allclose(world(solver, pose, name) - still[name], shift, atol=0.2), name


@needs_native
def test_cloth_swings_then_settles_back(tmp_path):
    solver = make_solver(tmp_path)
    still = simulate(solver, 120, lambda t: (0, 0, 0))
    tip_rest = world(solver, still, "skirt3").copy()
    peak = 0.0
    for i in range(90):
        t = i / 60
        pose = solver.update(pose_with_anchor_offset((30.0 * np.sin(t * 8.0), 0.0, 0.0)), 1 / 60)
        peak = max(peak, float(np.linalg.norm(world(solver, pose, "skirt3") - world(solver, pose, ANCHOR) - (tip_rest - world(solver, still, ANCHOR)))))
    assert peak > 0.5, "cloth should lag behind a shaken anchor"
    pose = simulate(solver, 600, lambda t: (0, 0, 0))
    assert np.allclose(world(solver, pose, "skirt3"), tip_rest, atol=0.3)


@needs_native
def test_reset_returns_cloth_to_the_anchored_rest_pose_with_zero_velocity(tmp_path):
    solver = make_solver(tmp_path)
    rest = simulate(solver, 120, lambda t: (0, 0, 0))
    tip_rest = world(solver, rest, "skirt3").copy()
    shaken = simulate(solver, 30, lambda t: (40.0 * np.sin(t * 15.0), 0.0, 0.0))
    assert np.linalg.norm(world(solver, shaken, "skirt3") - tip_rest) > 1.0

    # A seek jumps the anchor far away; reset must carry the cloth along instead of whipping it across the gap.
    solver.reset()
    jumped = solver.update(pose_with_anchor_offset((500.0, 0.0, 0.0)), 0.0)
    expected = tip_rest + np.array([500.0, 0.0, 0.0])
    assert np.allclose(world(solver, jumped, "skirt3"), expected, atol=0.2)
    after = solver.update(pose_with_anchor_offset((500.0, 0.0, 0.0)), 1 / 60)
    assert np.allclose(world(solver, after, "skirt3"), expected, atol=0.5)


@needs_native
def test_violent_motion_stays_finite_and_bounded(tmp_path):
    solver = make_solver(tmp_path)
    rng = np.random.default_rng(7)
    for _ in range(400):
        offset = rng.uniform(-80, 80, 3)
        pose = solver.update(pose_with_anchor_offset(offset), 1 / 60)
        for name in solver.dynamic_bones:
            matrix = pose[name]
            assert np.isfinite(matrix).all()
            assert np.linalg.norm(matrix[:3, 3] - pose[ANCHOR][:3, 3]) < 100.0


@needs_native
def test_pose_jump_without_reset_reanchors_instead_of_whipping(tmp_path):
    solver = make_solver(tmp_path)
    rest = simulate(solver, 120, lambda t: (0, 0, 0))
    tip_rest = world(solver, rest, "skirt3").copy()
    jumped = solver.update(pose_with_anchor_offset((400.0, 0.0, 0.0)), 1 / 60)
    assert np.allclose(world(solver, jumped, "skirt3"), tip_rest + [400.0, 0.0, 0.0], atol=0.5)


@needs_native
def test_cbpc_shared_dynamic_bone_keeps_smp_chain_and_visual_precedence(tmp_path):
    cbpc = StubCbpc(active=["skirt2"])
    solver = make_solver(tmp_path, cbpc=cbpc)
    assert solver.dynamic_bones == ["skirt1", "skirt2", "skirt3"]
    assert not any("skipped" in line and "constraint" in line for line in solver.diagnostics)
    assert not any("already driven by CBPC" in line for line in solver.diagnostics)

    pose = pose_with_anchor_offset()
    cbpc_pose = np.eye(4, dtype=np.float32)
    cbpc_pose[:3, 3] = [100, 101, 102]
    pose["skirt2"] = cbpc_pose.copy()
    result = solver.update(pose, 1 / 60)
    assert np.array_equal(result["skirt2"], cbpc_pose)
    assert np.shares_memory(result["skirt3"], solver._dyn)

    cbpc.is_active = False
    assert not solver.refresh()
    assert solver.dynamic_bones == ["skirt1", "skirt2", "skirt3"]
    result = solver.update(pose_with_anchor_offset(), 1 / 60)
    assert np.shares_memory(result["skirt2"], solver._dyn)


@needs_native
def test_eight_constraints_stay_connected_with_cbpc_shared_bones(tmp_path):
    bodies = "".join(f'<bone name="chain{i}"/>' for i in range(8))
    constraints = "".join(
        f'<generic-constraint bodyA="chain{i}" bodyB="{ANCHOR if i == 0 else f"chain{i - 1}"}"/>'
        for i in range(8)
    )
    xml = SKIRT_XML.replace('<bone name="skirt1"/>', bodies)
    xml = xml.replace('<bone name="skirt2"/>', "").replace('<bone name="skirt3"/>', "")
    xml = xml[:xml.index('  <generic-constraint bodyA="skirt1"')] + constraints + "</system>"
    nodes = {
        f"chain{i}": (ANCHOR if i == 0 else f"chain{i - 1}", _local(-3))
        for i in range(8)
    }
    nif, ref = make_outfit(tmp_path, xml)
    solver = SmpBridgeSolver(ENGINE, StubCbpc(active=["chain2", "chain5"]))
    solver.add_source("skirt", nif, ref, nodes)
    solver.set_visible_sources(["skirt"])
    solver.refresh()
    assert solver.dynamic_bones == list(nodes)
    assert not any("constraint(s) skipped" in line for line in solver.diagnostics)
    solver.reset()
    pose = solver.update(pose_with_anchor_offset((400, 0, 0)), 0)
    for i in range(8):
        expected = pose[ANCHOR] @ np.linalg.matrix_power(_local(-3), i + 1)
        if i not in (2, 5):
            assert np.allclose(pose[f"chain{i}"], expected, atol=1e-4)
        assert np.allclose(solver._dyn[i], expected, atol=1e-4)


@needs_native
def test_body_collider_deflects_cloth(tmp_path):
    collider_node = "NPC L Hand [LHnd]"
    xml = SKIRT_XML.replace('<bone name="NPC Pelvis [Pelv]"/>', (
        '<bone name="NPC Pelvis [Pelv]"/>'
        f'<rigid-body name="{collider_node}"><shape type="sphere" radius="6"/></rigid-body>'
    ))
    solver = make_solver(tmp_path, cbpc=StubCbpc(active=[collider_node, "skirt2"]), xml=xml)
    assert collider_node in get_native().body_names(KIND_KINEMATIC)
    assert "skirt2" in solver.dynamic_bones

    def run(with_hand):
        solver.reset()
        pose = None
        for i in range(180):
            pose = pose_with_anchor_offset()
            if with_hand:
                # Park the sphere against the hanging tip so it presses on the cloth.
                tip = np.array([0.0, 0.0, 0.0])
                if i > 0:
                    tip = rest_tip
                hand = np.eye(4, dtype=np.float32)
                hand[:3, 3] = tip + np.array([4.0, 0.0, 0.0])
                pose[collider_node] = hand
            pose = solver.update(pose, 1 / 60)
        return world(solver, pose, "skirt3")

    solver.reset()
    settled = simulate(solver, 120, lambda t: (0, 0, 0))
    rest_tip = world(solver, settled, "skirt3").copy()
    free = run(False)
    pushed = run(True)
    assert np.linalg.norm(pushed - free) > 0.2


@needs_native
def test_cbpc_shapes_are_not_registered(tmp_path):
    shape = [{"p1": [0, 0, 0], "r1": 6, "p2": None, "r2": 0}]
    node = "NPC L Hand [LHnd]"
    solver = make_solver(tmp_path, cbpc=StubCbpc(colliders=[("hand", node, shape)]))
    assert solver.has_physics
    assert node not in get_native().body_names(KIND_KINEMATIC)


@needs_native
def test_separate_collision_xml_and_capsule_transform(tmp_path):
    nif, ref = make_outfit(tmp_path)
    collision = tmp_path / "Collision.xml"
    collision.write_text("""<system>
      <rigid-body name="thigh collider" bone="NPC L Hand [LHnd]">
        <motion-type>keyframed</motion-type>
        <translation x="1" y="2" z="3"/>
        <shape type="capsule" radius="2"><p0 x="0" y="0" z="0"/><p1 x="4" y="0" z="0"/></shape>
      </rigid-body>
    </system>""", encoding="utf-8")
    solver = SmpBridgeSolver(ENGINE)
    solver.add_source("skirt", nif, ref, NODES)
    solver.set_collision_xmls([collision])
    solver.set_visible_sources(["skirt"])
    solver.refresh()
    assert solver.has_physics
    pose = pose_with_anchor_offset()
    hand = np.eye(4, dtype=np.float32)
    hand[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    hand[:3, 3] = [10, 20, 30]
    pose["NPC L Hand [LHnd]"] = hand
    solver.update(pose, 0)
    capsule = next(p for p in get_native().primitives() if p[0] == 1)
    assert np.allclose(capsule[1:3], [2, 4])
    matrix = capsule[6:].reshape(4, 4)
    assert np.allclose(matrix[:3, 3], [8, 23, 33], atol=1e-5)
    assert np.allclose(matrix[:3, 2], [0, 1, 0], atol=1e-5)
    lines = primitive_debug_lines([capsule])
    assert np.allclose(lines.min(axis=0), [6, 19, 31], atol=1e-5)
    assert np.allclose(lines.max(axis=0), [10, 27, 35], atol=1e-5)


@needs_native
def test_kinematic_collider_on_cbpc_bone_uses_animated_pose(tmp_path):
    node = "NPC L Hand [LHnd]"
    xml = SKIRT_XML.replace('<bone name="NPC Pelvis [Pelv]"/>', (
        '<bone name="NPC Pelvis [Pelv]"/>'
        f'<rigid-body name="{node}"><shape type="sphere" radius="2"/></rigid-body>'
    ))
    solver = make_solver(tmp_path, cbpc=StubCbpc(active=[node]), xml=xml)
    assert node in get_native().body_names(KIND_KINEMATIC)
    animated = pose_with_anchor_offset()
    deformed = dict(animated)
    deformed[node] = animated[node].copy()
    deformed[node][0, 3] += 50
    solver.update(deformed, 0, animated)
    sphere = next(p for p in get_native().primitives() if p[1] == 2)
    assert np.allclose(sphere[6:].reshape(4, 4), animated[node], atol=1e-5)


@needs_native
def test_total_capsule_height_and_explicit_box_dimensions(tmp_path):
    xml = SKIRT_XML.replace('<bone name="NPC Pelvis [Pelv]"/>', (
        '<bone name="NPC Pelvis [Pelv]"/>'
        '<rigid-body name="capsule" bone="NPC L Hand [LHnd]">'
        '<motion-type>kinematic</motion-type><mass>2</mass><shape type="capsule" radius="2" height="10"/></rigid-body>'
        '<rigid-body name="box" bone="NPC R Hand [RHnd]">'
        '<shape type="box"><half-extents x="1" y="2" z="3"/></shape></rigid-body>'
    ))
    solver = make_solver(tmp_path, xml=xml)
    assert solver.has_physics
    primitives = get_native().primitives()
    capsule = next(p for p in primitives if p[0] == 1)
    box = next(p for p in primitives if p[0] == 2)
    assert np.allclose(capsule[1:3], [2, 6])
    assert np.allclose(box[3:6], [1, 2, 3])
    assert len(primitive_debug_lines([box])) == 24


@needs_native
def test_kinematic_only_xml_succeeds_and_bad_file_does_not_create_partial_bodies(tmp_path):
    native = get_native()
    native.destroy()
    bind = ENGINE.get_bind_pose()
    names = [ANCHOR, "NPC L Hand [LHnd]"]
    assert native.set_bind_pose(names, np.stack([bind[name] for name in names])) == 0
    valid = tmp_path / "valid.xml"
    valid.write_text(f'<system><rigid-body name="{ANCHOR}"><shape type="sphere" radius="1"/>'
                     '</rigid-body></system>', encoding="utf-8")
    assert native.load_xml(valid, names) == 0
    invalid = tmp_path / "invalid.xml"
    invalid.write_text(f'<system><rigid-body name="{names[1]}"><shape type="sphere" radius="2"/>'
                       '</rigid-body><rigid-body name="other"><shape type="box"/></rigid-body></system>',
                       encoding="utf-8")
    assert native.load_xml(invalid, names) == -3
    assert native.body_names(KIND_KINEMATIC) == [ANCHOR]


@needs_native
def test_missing_or_nonfinite_kinematic_pose_is_reported(tmp_path):
    solver = make_solver(tmp_path)
    pose = pose_with_anchor_offset()
    pose[ANCHOR] = pose[ANCHOR].copy()
    pose[ANCHOR][0, 3] = np.nan
    solver.update(pose, 1 / 60)
    assert not solver.is_active
    assert any("status -3" in line for line in solver.diagnostics)


@needs_native
def test_cloth_sphere_does_not_fall_through_touching_obstacles(tmp_path):
    native = get_native()
    native.destroy()
    names = ["left", "right", "cloth"]
    bind = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
    bind[0, 0, 3], bind[1, 0, 3], bind[2, 2, 3] = -3, 3, 20
    assert native.set_bind_pose(names, bind) == 0
    xml = tmp_path / "seam.xml"
    xml.write_text("""<system>
      <rigid-body name="left"><shape type="sphere" radius="3"/></rigid-body>
      <rigid-body name="right"><shape type="sphere" radius="3"/></rigid-body>
      <rigid-body name="cloth"><mass>1</mass><shape type="sphere" radius="1"/>
        <linearDamping>0.5</linearDamping></rigid-body>
    </system>""", encoding="utf-8")
    assert native.load_xml(xml, names) == 0
    kin = np.ascontiguousarray(bind[:2])
    dyn = np.ascontiguousarray(bind[2:])
    for _ in range(120):
        assert native.step(1 / 60, native.pointer(kin), 2, native.pointer(dyn), 1) == 0
    assert dyn[0, 2, 3] > 1.0


@needs_native
@pytest.mark.parametrize("shape", [
    '<shape type="sphere"/>',
    '<shape type="capsule" radius="1"><p0 x="0" y="0" z="0"/></shape>',
    '<shape type="box"/>',
    '<shape type="unknown"/>',
])
def test_invalid_shapes_are_reported_without_fallback(tmp_path, shape):
    xml = f'<system><rigid-body name="{ANCHOR}">{shape}</rigid-body></system>'
    solver = make_solver(tmp_path, xml=xml)
    assert not solver.has_physics
    assert solver.diagnostics


@needs_native
def test_step_streams_through_fixed_buffers_without_reallocating(tmp_path):
    solver = make_solver(tmp_path)
    kin, dyn = solver._kin, solver._dyn
    kin_address, dyn_address = kin.ctypes.data, dyn.ctypes.data
    for i in range(30):
        pose = solver.update(pose_with_anchor_offset((i * 0.1, 0.0, 0.0)), 1 / 60)
    assert solver._kin is kin and solver._dyn is dyn
    assert kin.ctypes.data == kin_address and dyn.ctypes.data == dyn_address
    assert kin.dtype == np.float32 and kin.flags["C_CONTIGUOUS"]
    assert np.shares_memory(pose["skirt3"], dyn)
    assert ctypes_address(solver._kin_ptr) == kin_address
    assert ctypes_address(solver._dyn_ptr) == dyn_address


def ctypes_address(pointer):
    import ctypes

    return ctypes.cast(pointer, ctypes.c_void_p).value


@needs_native
def test_step_is_fast_enough_for_realtime(tmp_path):
    solver = make_solver(tmp_path)
    poses = [pose_with_anchor_offset((np.sin(i * 0.1) * 5.0, 0.0, 0.0)) for i in range(240)]
    started = time.perf_counter()
    for pose in poses:
        solver.update(pose, 1 / 60)
    per_frame_ms = (time.perf_counter() - started) / len(poses) * 1000
    assert per_frame_ms < 4.0, f"{per_frame_ms:.2f} ms per SMP frame"


@needs_native
def test_disabling_returns_cloth_to_its_skeleton_parent_chain(tmp_path):
    solver = make_solver(tmp_path)
    assert solver.is_active
    solver.set_enabled(False)
    assert not solver.is_active and not solver.has_physics
    solver.refresh()
    pose = solver.update(pose_with_anchor_offset((10.0, 0.0, 0.0)), 1 / 60)
    expected = pose[ANCHOR] @ _local(-6.0) @ _local(-6.0) @ _local(-6.0)
    assert np.allclose(pose["skirt3"], expected, atol=1e-4)
    solver.set_enabled(True)
    solver.refresh()
    assert solver.is_active


@needs_native
def test_outfit_bones_follow_their_parent_chain_without_xml(tmp_path):
    nif = tmp_path / "plain.nif"
    nif.touch()
    solver = SmpBridgeSolver(ENGINE, None)
    solver.add_source("plain", nif, "", NODES)
    solver.set_visible_sources(["plain"])
    solver.refresh()
    assert not solver.has_physics
    assert solver.has_outfit_bone("skirt2") and not solver.has_outfit_bone(ANCHOR)
    pose = solver.update(pose_with_anchor_offset((3.0, 0.0, 0.0)), 1 / 60)
    assert np.allclose(pose["skirt2"], pose[ANCHOR] @ _local(-6.0) @ _local(-6.0), atol=1e-4)


@needs_native
def test_outfit_bone_requires_matching_skin_matrix(tmp_path):
    nif = tmp_path / "plain.nif"
    nif.touch()
    solver = SmpBridgeSolver(ENGINE, None)
    solver.add_source("plain", nif, "", NODES)
    bind = ENGINE.get_bind_pose()[ANCHOR] @ _local(-6.0)
    assert solver.outfit_bone_matches_skin("skirt1", np.linalg.inv(bind))
    assert not solver.outfit_bone_matches_skin("skirt1", np.eye(4, dtype=np.float32))
    assert not solver.outfit_bone_matches_skin("not_an_outfit_bone", np.eye(4, dtype=np.float32))


@needs_native
def test_missing_xml_builds_no_world(tmp_path):
    nif = tmp_path / "bare.nif"
    nif.touch()
    solver = SmpBridgeSolver(ENGINE, None)
    solver.add_source("bare", nif, "meshes\\nope.xml", NODES)
    solver.set_visible_sources(["bare"])
    solver.refresh()
    assert not solver.has_physics
    assert any("nope.xml" in line for line in solver.diagnostics)


@needs_native
def test_malformed_xml_is_reported_not_raised(tmp_path):
    solver = make_solver(tmp_path, xml="<system><bone name='skirt1'></system>")
    assert not solver.has_physics
    assert any("not well formed" in line for line in solver.diagnostics)


@needs_native
def test_xml_with_external_entities_is_rejected(tmp_path):
    xml = '<?xml version="1.0"?><!DOCTYPE system [<!ENTITY x SYSTEM "file:///C:/Windows/win.ini">]><system>&x;</system>'
    solver = make_solver(tmp_path, xml=xml)
    assert not solver.has_physics


def test_no_pythonnet_dependency_remains():
    root = Path(__file__).resolve().parent.parent
    needles = ("python" + "net", "import " + "clr", "from " + "clr")
    offenders = []
    for path in list((root / "src").glob("*.py")) + [root / "requirements.txt", root / "main.py"]:
        text = path.read_text(encoding="utf-8").lower()
        offenders += [str(path.name) for n in needles if n in text]
    assert offenders == []
