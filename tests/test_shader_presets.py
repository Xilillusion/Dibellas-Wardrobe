import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import glm
import moderngl
import numpy as np
import pytest
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtWidgets import QApplication

from src.config import AppConfig
from src.nif_loader import ShaderMaterialData, _resolve_preset_textures
from src.shader_presets import load_postprocess_profile
from src.texture_resolver import TextureResolver
from src.viewport import OutfitViewportWidget

SHADERS = Path(__file__).resolve().parent.parent / "shaders"
MATERIAL_SAMPLERS = (
    "u_material_normal", "u_material_env_mask", "u_material_subsurface", "u_material_flags"
)


def test_preset_texture_discovery_respects_authored_slots_and_companions(tmp_path, monkeypatch):
    texture_dir = tmp_path / "textures" / "armor"
    texture_dir.mkdir(parents=True)
    for name in ("armor_msn.dds", "armor_m.dds", "armor_sk.dds", "custom_n.dds"):
        (texture_dir / name).write_bytes(b"fixture")
    resolver = TextureResolver([tmp_path])
    monkeypatch.setattr(
        resolver, "decode_dds_file", lambda _path: np.ones((1, 1, 4), dtype=np.uint8)
    )
    mat = ShaderMaterialData(raw_textures={
        "Diffuse": r"textures\armor\armor.dds",
        "Normal": r"textures\armor\custom_n.dds",
    })
    _resolve_preset_textures(mat, resolver, tmp_path / "armor.nif")
    assert mat.resolved_textures["PresetNormal"].resolved_path.name == "custom_n.dds"
    assert mat.resolved_textures["PresetEnvMask"].resolved_path.name == "armor_m.dds"
    assert mat.resolved_textures["Subsurface"].resolved_path.name == "armor_sk.dds"

    mat.raw_textures.pop("Normal")
    mat.resolved_textures.clear()
    _resolve_preset_textures(mat, resolver, tmp_path / "armor.nif")
    assert mat.resolved_textures["PresetNormal"].resolved_path.name == "armor_msn.dds"
    mat.raw_textures["Diffuse"] = r"textures\armor\armor_d.dds"
    mat.resolved_textures.clear()
    _resolve_preset_textures(mat, resolver, tmp_path / "armor.nif")
    assert mat.resolved_textures["Subsurface"].resolved_path.name == "armor_sk.dds"
    assert mat.resolved_textures["PresetEnvMask"].resolved_path.name == "armor_m.dds"


def test_preset_texture_discovery_does_not_replace_missing_explicit_slot(tmp_path, monkeypatch):
    directory = tmp_path / "textures"
    directory.mkdir()
    (directory / "armor_n.dds").write_bytes(b"fixture")
    resolver = TextureResolver([tmp_path])
    monkeypatch.setattr(
        resolver, "decode_dds_file", lambda _path: np.ones((1, 1, 4), dtype=np.uint8)
    )
    mat = ShaderMaterialData(raw_textures={
        "Diffuse": "textures/armor.dds", "Normal": "textures/missing_n.dds"
    })
    _resolve_preset_textures(mat, resolver, tmp_path / "armor.nif")
    assert "PresetNormal" not in mat.resolved_textures


def test_shader_presets_discover_fragment_shaders_but_skip_builtin_stages(tmp_path):
    for filename in (
        "default.frag",
        "postprocess.frag",
        "realistic.frag",
        "future-look.frag",
        "ignored.vert",
    ):
        (tmp_path / filename).write_text("", encoding="utf-8")

    assert OutfitViewportWidget._discover_shader_presets(tmp_path) == (
        ("future-look", "Future Look"),
        ("realistic", "Realistic"),
    )


def test_builtin_anime_preset_is_discovered():
    assert ("anime", "Anime") in OutfitViewportWidget._discover_shader_presets()


def test_shader_preset_config_round_trips(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"shader_preset": "realistic"}), encoding="utf-8")

    config = AppConfig.load(path, supported_languages=["en"])
    assert config.shader_preset == "realistic"

    config.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["shader_preset"] == "realistic"


def test_anime_selection_round_trips(tmp_path):
    path = tmp_path / "config.json"
    config = AppConfig(language="en", shader_preset="anime")
    config.save(path)
    assert AppConfig.load(path, supported_languages=["en"]).shader_preset == "anime"


def test_legacy_shader_preset_selection_is_migrated(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"shader_preset": "preset_1"}), encoding="utf-8")

    config = AppConfig.load(path, supported_languages=["en"])
    assert config.shader_preset == "realistic"
    config.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["shader_preset"] == "realistic"


@pytest.mark.parametrize("uses_normals", [False, True])
def test_preset_normal_sampler_is_optional_and_bound_when_used(uses_normals):
    program = MagicMock()
    uniforms = {}
    program.__getitem__.side_effect = lambda name: uniforms.setdefault(name, MagicMock())
    program.__contains__.side_effect = lambda name: name == "u_scene_normal" and uses_normals
    widget = SimpleNamespace(
        ctx=MagicMock(),
        preset_prog=program,
        preset_uniforms={},
        preset_vao=MagicMock(),
        scene_depth=MagicMock(),
        scene_normal=MagicMock(),
    )
    input_color = MagicMock()
    OutfitViewportWidget._composite_shader_preset(
        widget, MagicMock(), input_color, glm.mat4(1.0), 8, 8
    )
    input_color.use.assert_called_once_with(location=0)
    widget.scene_normal.use.assert_called_once_with(location=2)
    if uses_normals:
        assert uniforms["u_scene_normal"].value == 2
    else:
        assert all(call.args != ("u_scene_normal",) for call in program.__getitem__.call_args_list)


def test_preset_material_samplers_are_bound_to_matching_textures():
    program = MagicMock()
    uniforms = {}
    program.__getitem__.side_effect = lambda name: uniforms.setdefault(name, MagicMock())
    names = MATERIAL_SAMPLERS
    program.__contains__.side_effect = lambda name: name in names
    textures = [MagicMock() for _ in names]
    widget = SimpleNamespace(
        ctx=MagicMock(), preset_prog=program, preset_vao=MagicMock(), preset_uniforms={},
        scene_depth=MagicMock(), scene_normal=MagicMock(),
        material_textures=textures,
    )
    OutfitViewportWidget._composite_shader_preset(
        widget, MagicMock(), MagicMock(), glm.mat4(1.0), 8, 8
    )
    for index, name in enumerate(names):
        assert uniforms[name].value == index + 3
        textures[index].use.assert_called_once_with(location=index + 3)


@pytest.fixture(scope="module")
def shader_renderer_factory():
    app = QApplication.instance() or QApplication([])
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    context = QOpenGLContext()
    context.setFormat(fmt)
    if not surface.isValid() or not context.create() or not context.makeCurrent(surface):
        pytest.skip("OpenGL 3.3 offscreen context is unavailable")

    ctx = moderngl.create_context(require=330)
    projection = glm.perspective(glm.radians(42.0), 1.0, 1.0, 1000.0)
    vertex_source = (SHADERS / "postprocess.vert").read_text(encoding="utf-8")
    programs = {}
    arrays = {}
    # Uniform grey studio, so reflections only scale with their weights.
    studio = ctx.texture_cube((4, 4), 4, bytes([153, 153, 153, 255]) * 16 * 6)

    def render(
        shader_name, color, depth, normals=None, materials=None, key_light=None, outputs=1,
        uniforms=None,
    ):
        if shader_name not in programs:
            program = ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=(SHADERS / f"{shader_name}.frag").read_text(encoding="utf-8"),
            )
            programs[shader_name] = program
            arrays[shader_name] = ctx.vertex_array(program, [])
            program["u_scene_color"].value = 0
            program["u_scene_depth"].value = 1
            program["u_scene_normal"].value = 2
            for index, name in enumerate(MATERIAL_SAMPLERS):
                if name in program:
                    program[name].value = index + 3
            program["u_inverse_projection"].write(
                bytes(glm.transpose(glm.inverse(projection)))
            )
            if "u_studio_cubemap" in program:
                program["u_studio_cubemap"].value = 7
            if "u_view_to_world" in program:
                program["u_view_to_world"].write(bytes(glm.mat3(1.0)))
        program = programs[shader_name]
        height, width = depth.shape
        scene = ctx.texture((width, height), 4, color.astype("f4").tobytes(), dtype="f4")
        depth_texture = ctx.texture(
            (width, height), 1, depth.astype("f4").tobytes(), dtype="f4"
        )
        if normals is None:
            normals = np.zeros((height, width, 4), dtype="f4")
        normal_texture = ctx.texture(
            (width, height), 4, normals.astype("f4").tobytes(), dtype="f4"
        )
        materials = materials or {}
        material_textures = []
        for name in MATERIAL_SAMPLERS:
            data = materials.get(name, np.zeros((height, width, 4), dtype="f4"))
            texture = ctx.texture((width, height), 4, data.astype("f4").tobytes(), dtype="f4")
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            material_textures.append(texture)
        targets = [ctx.texture((width, height), 4, dtype="f4") for _ in range(outputs)]
        fbo = ctx.framebuffer(color_attachments=targets)
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        scene.filter = (moderngl.LINEAR, moderngl.LINEAR)
        depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        normal_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        scene.use(location=0)
        depth_texture.use(location=1)
        normal_texture.use(location=2)
        for index, texture in enumerate(material_textures):
            texture.use(location=index + 3)
        studio.use(location=7)
        program["u_viewport_size"].value = (float(width), float(height))
        if "u_key_light" in program:
            program["u_key_light"].value = tuple(key_light) if key_light is not None else (0.0,) * 3
        for name, value in (uniforms or {}).items():
            program[name].value = value
        fbo.use()
        ctx.viewport = (0, 0, width, height)
        arrays[shader_name].render(moderngl.TRIANGLES, vertices=3)
        results = [
            np.frombuffer(
                fbo.read(components=4, dtype="f4", attachment=index), dtype="f4"
            ).reshape((height, width, 4)).copy()
            for index in range(outputs)
        ]
        fbo.release()
        for target in targets:
            target.release()
        for texture in material_textures:
            texture.release()
        depth_texture.release()
        normal_texture.release()
        scene.release()
        return results[0] if outputs == 1 else tuple(results)

    render.ctx = ctx
    yield render
    studio.release()
    for vao in arrays.values():
        vao.release()
    for program in programs.values():
        program.release()
    ctx.release()
    context.doneCurrent()
    app.processEvents()


@pytest.fixture(scope="module")
def preset_renderer(shader_renderer_factory):
    return lambda color, depth, normals=None, materials=None, key_light=None: (
        shader_renderer_factory("realistic", color, depth, normals, materials, key_light)
    )


@pytest.fixture(scope="module")
def anime_renderer(shader_renderer_factory):
    return lambda color, depth, normals=None, materials=None, key_light=None: (
        shader_renderer_factory("anime", color, depth, normals, materials, key_light)
    )


@pytest.fixture(scope="module")
def glossy_renderer(shader_renderer_factory):
    """Returns the (diffuse, specular + SSSS mask) outputs of the Glossy preset."""
    return lambda color, depth, normals=None, materials=None, key_light=None: (
        shader_renderer_factory("glossy", color, depth, normals, materials, key_light, outputs=2)
    )


FALLBACK_LIGHT = (-0.3537, 0.5558, 0.7680)
@pytest.mark.parametrize("gtao, ssss", [(False, False), (True, False), (False, True), (True, True)])
def test_postprocess_variants_compile_only_enabled_effects(shader_renderer_factory, gtao, ssss):
    ctx = shader_renderer_factory.ctx
    program, vertex_array = OutfitViewportWidget._compile_postprocess_variant(
        SimpleNamespace(ctx=ctx), gtao, ssss
    )
    try:
        assert ("u_ao_radius" in program) == gtao
        assert ("u_ssss_strength" in program) == ssss
        assert ("u_scene_depth" in program) == (gtao or ssss)
    finally:
        vertex_array.release()
        program.release()


def test_postprocess_copy_variant_passes_scene_through(shader_renderer_factory):
    ctx = shader_renderer_factory.ctx
    program, vertex_array = OutfitViewportWidget._compile_postprocess_variant(
        SimpleNamespace(ctx=ctx), False, False
    )
    rng = np.random.default_rng(7)
    color = rng.random((6, 6, 4), dtype="f4")
    scene = ctx.texture((6, 6), 4, color.tobytes(), dtype="f4")
    scene.filter = (moderngl.NEAREST, moderngl.NEAREST)
    output = ctx.texture((6, 6), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[output])
    try:
        scene.use(location=0)
        program["u_scene_color"].value = 0
        fbo.use()
        ctx.viewport = (0, 0, 6, 6)
        vertex_array.render(moderngl.TRIANGLES, vertices=3)
        result = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype="f4").reshape(6, 6, 4)
        assert np.allclose(result, color)
    finally:
        for resource in (fbo, output, scene, vertex_array, program):
            resource.release()


REALISTIC_EXPOSURE = 1.1


def _realistic_grade(value):
    """Realistic's final exposure and contrast curve for a neutral grey."""
    value *= REALISTIC_EXPOSURE
    return value * (1.0 + 0.40 * (value - 0.5) * (1.0 - value))


def _depth_at_distance(distance):
    return 1000.0 / 999.0 - 1000.0 / (999.0 * distance)


def test_preset_render_preserves_background_and_alpha(preset_renderer):
    color = np.full((32, 32, 4), 0.3, dtype="f4")
    color[:, :, 3] = 0.65
    depth = np.full((32, 32), _depth_at_distance(10.0), dtype="f4")
    depth[:4, :] = 1.0
    result = preset_renderer(color, depth)

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:4], color[:4], atol=1e-6)
    np.testing.assert_allclose(result[:, :, 3], color[:, :, 3], atol=1e-6)
    # Dark tones are deepened by the contrast curve after the exposure lift.
    assert result[16, 16, 0] < color[16, 16, 0] * REALISTIC_EXPOSURE
    np.testing.assert_allclose(result[16, 16, 0], _realistic_grade(0.3), atol=1e-5)
    np.testing.assert_allclose(result[16, 16, :3], result[16, 16, 0], atol=1e-6)


@pytest.mark.parametrize("distance", [10.0, 100.0])
def test_preset_render_adds_contact_shadows_without_darkening_flat_surfaces(
    preset_renderer, distance
):
    color = np.full((64, 64, 4), 0.5, dtype="f4")
    color[:, :, 3] = 1.0
    flat_depth = np.full((64, 64), _depth_at_distance(distance), dtype="f4")
    contact_depth = flat_depth.copy()
    contact_depth[28:37, 34:43] = _depth_at_distance(distance * 0.97)

    flat = preset_renderer(color, flat_depth)
    contact = preset_renderer(color, contact_depth)

    np.testing.assert_allclose(flat[20:44, 20:44, :3], _realistic_grade(0.5), atol=1e-5)
    assert contact[32, 33, 0] < flat[32, 33, 0] - 0.002
    np.testing.assert_allclose(contact[16, 16], flat[16, 16], atol=1e-5)


def test_preset_render_enhances_detail_without_crossing_depth_edges(preset_renderer):
    depth = np.full((32, 32), _depth_at_distance(10.0), dtype="f4")
    color = np.full((32, 32, 4), 0.4, dtype="f4")
    color[:, :, 3] = 1.0
    uniform = preset_renderer(color, depth)

    color[16, 16, :3] = 0.55
    detail = preset_renderer(color, depth)
    assert detail[16, 16, 0] > uniform[16, 16, 0]
    assert detail[16, 15, 0] < uniform[16, 15, 0]

    depth[:, 16:] = 1.0
    color[:, 16:, :3] = 1.0
    edge = preset_renderer(color, depth)
    np.testing.assert_allclose(edge[16, 15], uniform[16, 15], atol=1e-5)
    np.testing.assert_allclose(edge[16, 16], color[16, 16], atol=1e-5)


def test_preset_uses_shading_normals_and_falls_back_when_invalid(preset_renderer):
    color = np.full((64, 64, 4), 0.5, dtype="f4")
    color[:, :, 3] = 1.0
    depth = np.full((64, 64), _depth_at_distance(10.0), dtype="f4")
    depth[28:37, 34:43] = _depth_at_distance(9.7)
    normals = np.zeros((64, 64, 4), dtype="f4")
    fallback = preset_renderer(color, depth, normals)
    normals[:, :, :] = (0.9, 0.5, 0.8, 1.0)
    mapped = preset_renderer(color, depth, normals)
    assert abs(mapped[32, 33, 0] - fallback[32, 33, 0]) > 0.001

    normals[:, :, 3] = 0.0
    invalid = preset_renderer(color, depth, normals)
    np.testing.assert_allclose(invalid, fallback, atol=1e-6)


def test_normal_map_cavities_add_local_shading_without_flat_surface_darkening(preset_renderer):
    color = np.full((32, 32, 4), 0.5, dtype="f4")
    color[:, :, 3] = 1.0
    depth = np.full((32, 32), _depth_at_distance(10.0), dtype="f4")
    normals = np.empty((32, 32, 4), dtype="f4")
    normals[:, :, :] = (0.5, 0.5, 1.0, 1.0)
    flat = preset_renderer(color, depth, normals)

    normals[16, 17] = (0.3, 0.5, 0.958258, 1.0)
    normals[16, 15] = (0.7, 0.5, 0.958258, 1.0)
    normals[17, 16] = (0.5, 0.3, 0.958258, 1.0)
    normals[15, 16] = (0.5, 0.7, 0.958258, 1.0)
    concave = preset_renderer(color, depth, normals)
    assert 0.005 < flat[16, 16, 0] - concave[16, 16, 0] < 0.05
    np.testing.assert_allclose(concave[8, 8], flat[8, 8], atol=1e-5)

    normals[:, :, :] = (0.8, 0.5, 0.9, 1.0)
    tilted = preset_renderer(color, depth, normals)
    np.testing.assert_allclose(tilted[4:28, 4:28], flat[4:28, 4:28], atol=1e-5)


def test_normal_cavity_ignores_invalid_and_discontinuous_neighbors(preset_renderer):
    color = np.full((32, 32, 4), 0.5, dtype="f4")
    color[:, :, 3] = 1.0
    depth = np.full((32, 32), _depth_at_distance(10.0), dtype="f4")
    normals = np.empty((32, 32, 4), dtype="f4")
    normals[:, :, :] = (0.5, 0.5, 1.0, 1.0)
    baseline = preset_renderer(color, depth, normals)
    normals[16, 17] = (0.0, 0.5, 0.5, 1.0)
    normals[16, 15] = (1.0, 0.5, 0.5, 1.0)
    normals[17, 16] = (0.5, 0.0, 0.5, 1.0)
    normals[15, 16] = (0.5, 1.0, 0.5, 1.0)
    result = preset_renderer(color, depth, normals)
    np.testing.assert_allclose(result[16, 16], baseline[16, 16], atol=1e-5)
    normals[:, :, 3] = 0.0
    invalid = preset_renderer(color, depth, normals)
    np.testing.assert_allclose(invalid, baseline, atol=1e-5)


def test_surface_detail_respects_normal_discontinuities(preset_renderer):
    color = np.full((32, 32, 4), 0.35, dtype="f4")
    color[:, :, 3] = 1.0
    color[:, 16:, :3] = 0.65
    depth = np.full((32, 32), _depth_at_distance(10.0), dtype="f4")
    normals = np.empty((32, 32, 4), dtype="f4")
    normals[:, :, :] = (0.5, 0.5, 1.0, 1.0)
    same_surface = preset_renderer(color, depth, normals)
    normals[:, 16:, :] = (0.9, 0.5, 0.8, 1.0)
    hard_edge = preset_renderer(color, depth, normals)

    assert hard_edge[16, 15, 0] > same_surface[16, 15, 0] + 0.005
    assert hard_edge[16, 16, 0] < same_surface[16, 16, 0] - 0.005


_KEY_HALF_VECTOR = np.array([-0.3537, 0.5558, 0.7680 + 1.0])
_KEY_HALF_VECTOR /= np.linalg.norm(_KEY_HALF_VECTOR)


def _material_scene(size=48, value=0.45, normal=(0.0, 0.0, 1.0)):
    color = np.full((size, size, 4), value, dtype="f4")
    color[:, :, 3] = 1.0
    depth = np.full((size, size), _depth_at_distance(10.0), dtype="f4")
    normals = np.empty((size, size, 4), dtype="f4")
    normals[:, :, :3] = np.asarray(normal) * 0.5 + 0.5
    normals[:, :, 3] = 1.0
    return color, depth, normals


def _material_maps(size=48, normal=None, env_mask=None, subsurface=None, model_space=False):
    maps = {name: np.zeros((size, size, 4), dtype="f4") for name in MATERIAL_SAMPLERS}
    flags = maps["u_material_flags"]
    for channel, (name, value) in enumerate(
        (("u_material_normal", normal), ("u_material_env_mask", env_mask),
         ("u_material_subsurface", subsurface))
    ):
        if value is not None:
            maps[name][:, :] = value
            flags[:, :, channel] = 1.0
    flags[:, :, 3] = 1.0 if model_space and normal is not None else 0.0
    return maps


def test_realistic_ignores_material_samples_without_flags(preset_renderer):
    color, depth, normals = _material_scene(normal=_KEY_HALF_VECTOR)
    baseline = preset_renderer(color, depth, normals)
    maps = _material_maps(normal=(0.5, 0.5, 1.0, 1.0), env_mask=(1.0, 1.0, 1.0, 1.0),
                          subsurface=(1.0, 0.3, 0.2, 1.0))
    maps["u_material_flags"][:] = 0.0
    result = preset_renderer(color, depth, normals, maps)
    np.testing.assert_allclose(result, baseline, atol=1e-6)


def test_realistic_env_mask_adds_glossy_highlight_and_preserves_background(preset_renderer):
    color, depth, normals = _material_scene(normal=_KEY_HALF_VECTOR)
    depth[:4, :] = 1.0
    color[:, :, 3] = 0.8
    matte = preset_renderer(color, depth, normals, _material_maps(env_mask=(0.0, 0.0, 0.0, 1.0)))
    rough = preset_renderer(color, depth, normals, _material_maps(env_mask=(1.0, 0.1, 0.0, 1.0)))
    glossy = preset_renderer(color, depth, normals, _material_maps(env_mask=(1.0, 1.0, 0.0, 1.0)))

    assert np.isfinite(glossy).all()
    np.testing.assert_allclose(glossy[:4], color[:4], atol=1e-6)
    np.testing.assert_allclose(glossy[:, :, 3], color[:, :, 3], atol=1e-6)
    np.testing.assert_allclose(matte, preset_renderer(color, depth, normals), atol=1e-6)
    assert rough[24, 24, 0] > matte[24, 24, 0] + 0.003
    assert glossy[24, 24, 0] > rough[24, 24, 0] + 0.03
    np.testing.assert_allclose(glossy[24, 24, :3], glossy[24, 24, 0], atol=1e-5)


def test_realistic_complex_env_mask_metalness_tints_reflection(preset_renderer):
    color, depth, normals = _material_scene(normal=_KEY_HALF_VECTOR)
    color[:, :, :3] = (0.6, 0.25, 0.1)
    base = preset_renderer(color, depth, normals)
    dielectric = preset_renderer(
        color, depth, normals, _material_maps(env_mask=(1.0, 0.8, 0.0, 1.0))
    )
    metal = preset_renderer(color, depth, normals, _material_maps(env_mask=(1.0, 0.8, 1.0, 1.0)))
    dielectric_gain = dielectric[24, 24, :3] - base[24, 24, :3]
    metal_gain = metal[24, 24, :3] - base[24, 24, :3]

    assert metal_gain[0] > dielectric_gain[0]
    assert metal_gain[0] / metal_gain[2] > dielectric_gain[0] / dielectric_gain[2] * 1.5


def test_realistic_uses_tangent_normal_alpha_as_fallback_specular(preset_renderer):
    color, depth, normals = _material_scene(normal=_KEY_HALF_VECTOR)
    base = preset_renderer(color, depth, normals)
    tangent = preset_renderer(
        color, depth, normals, _material_maps(normal=(0.5, 0.5, 1.0, 1.0))
    )
    no_specular = preset_renderer(
        color, depth, normals, _material_maps(normal=(0.5, 0.5, 1.0, 0.0))
    )
    model_space = preset_renderer(
        color, depth, normals, _material_maps(normal=(0.5, 0.5, 1.0, 1.0), model_space=True)
    )
    assert tangent[24, 24, 0] > base[24, 24, 0] + 0.002
    np.testing.assert_allclose(no_specular, base, atol=1e-6)
    np.testing.assert_allclose(model_space, base, atol=1e-6)


def test_realistic_highlight_tightens_on_smooth_normals(preset_renderer):
    color, depth, normals = _material_scene(normal=_KEY_HALF_VECTOR)
    maps = _material_maps(env_mask=(1.0, 1.0, 0.0, 1.0))
    smooth = preset_renderer(color, depth, normals, maps)
    smooth_gain = smooth[24, 24, 1] - preset_renderer(color, depth, normals)[24, 24, 1]
    bumpy = normals.copy()
    tilt = np.array([0.6, 0.0, 0.0])
    for x in range(bumpy.shape[1]):
        direction = _KEY_HALF_VECTOR + (tilt if x % 2 else -tilt)
        bumpy[:, x, :3] = direction / np.linalg.norm(direction) * 0.5 + 0.5
    # The center pixel keeps its exact normal; only its neighbours disagree.
    bumpy[24, 24, :3] = _KEY_HALF_VECTOR * 0.5 + 0.5
    rough = preset_renderer(color, depth, bumpy, maps)
    rough_gain = rough[24, 24, 1] - preset_renderer(color, depth, bumpy)[24, 24, 1]
    assert smooth_gain > 0.05
    assert rough_gain < smooth_gain * 0.6


def test_realistic_subsurface_diffuses_lighting_and_tints_shadows(preset_renderer):
    color, depth, normals = _material_scene(value=0.6)
    color[:, 24:, :3] = 0.12
    base = preset_renderer(color, depth, normals)
    skin = preset_renderer(
        color, depth, normals, _material_maps(subsurface=(0.9, 0.35, 0.25, 1.0))
    )

    shadow = skin[20, 25, :3] - base[20, 25, :3]
    assert shadow[0] > 0.02
    assert shadow[0] > shadow[1] > shadow[2]
    assert shadow[0] - shadow[2] > 0.01
    assert skin[20, 22, 0] < base[20, 22, 0]
    assert abs(skin[20, 12, 0] - base[20, 12, 0]) < 1e-5


def test_realistic_subsurface_stays_on_its_own_surface(preset_renderer):
    color, depth, normals = _material_scene(value=0.12)
    color[:, :24, :3] = 0.6
    maps = _material_maps(subsurface=(0.9, 0.35, 0.25, 1.0))
    base = preset_renderer(color, depth, normals)
    maps["u_material_flags"][:, :24, 2] = 0.0
    no_bleed_from_other_material = preset_renderer(color, depth, normals, maps)
    maps["u_material_flags"][:, :, 2] = 1.0
    depth[:, :24] = _depth_at_distance(5.0)
    depth_edge_base = preset_renderer(color, depth, normals)
    no_bleed_across_depth = preset_renderer(color, depth, normals, maps)

    def gain(result, reference, x):
        return result[20, x, 0] - reference[20, x, 0]

    assert abs(gain(no_bleed_from_other_material, base, 25)
               - gain(no_bleed_from_other_material, base, 40)) < 0.004
    assert abs(gain(no_bleed_across_depth, depth_edge_base, 25)
               - gain(no_bleed_across_depth, depth_edge_base, 40)) < 0.004


def _luminance(pixel):
    return float(np.dot(pixel[:3], (0.2126, 0.7152, 0.0722)))


_FRONT_LIGHT = (0.0, 0.0, 1.0)


def _unit(*vector):
    return np.asarray(vector) / np.linalg.norm(vector)


def _skin_maps():
    return _material_maps(subsurface=(1.0, 0.3, 0.2, 1.0))


def test_anime_preserves_background_and_alpha(anime_renderer):
    color, depth, normals = _material_scene(value=0.5)
    depth[:4] = 1.0
    color[:, :, 3] = 0.6
    result = anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:4], color[:4], atol=1e-6)
    np.testing.assert_allclose(result[:, :, 3], 0.6, atol=1e-6)


def test_anime_shades_by_shape_with_crisp_cool_shadows(anime_renderer):
    shades = {}
    for name, normal in (("lit", _unit(0.3, 0.0, 1.0)), ("edge_lit", _unit(1.0, 0.0, 0.2)),
                         ("edge_dark", _unit(1.0, 0.0, -0.2)), ("dark", _unit(1.0, 0.0, -0.8))):
        color, depth, normals = _material_scene(value=0.5, normal=normal)
        # Feed a flat source so only the relighting differs between samples.
        shades[name] = anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)[24, 24]
    # A narrow terminator separates two tones; each tone stays flat.
    assert _luminance(shades["edge_dark"]) < _luminance(shades["edge_lit"]) * 0.97
    assert abs(_luminance(shades["dark"]) - _luminance(shades["edge_dark"])) < 0.03
    # Cloth shadows lean blue-violet instead of neutral grey.
    assert shades["dark"][2] > shades["dark"][0] + 0.04


def test_anime_flattens_lit_gradient(anime_renderer):
    lum = []
    for x, shade in ((0.2, 1.0), (0.75, 0.75)):
        color, depth, normals = _material_scene(value=0.6 * shade, normal=_unit(x, 0.0, 1.0))
        lum.append(_luminance(anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)[24, 24]))
    # A 25% Lambert falloff in the source is visibly compressed.
    assert lum[1] / lum[0] > 0.75 + 0.05


def test_anime_keeps_texture_detail_in_the_lit_tone(anime_renderer):
    color, depth, normals = _material_scene(value=0.5)
    color[:, 24:, :3] = (0.7, 0.4, 0.25)
    result = anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)
    assert result[10, 40, 0] > result[10, 40, 1] > result[10, 40, 2]
    assert result[10, 40, 0] > result[10, 10, 0]


def test_anime_form_shadow_ignores_isolated_normal_grain(anime_renderer):
    color, depth, normals = _material_scene(value=0.6)
    baseline = anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)
    normals[24, 24, :3] = _unit(1.0, 0.0, -0.4) * 0.5 + 0.5
    speck = anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)
    # Neither the speck nor its neighbours fall into shadow or get inked.
    assert _luminance(speck[24, 24]) > _luminance(baseline[24, 24]) * 0.9
    for y, x in ((24, 23), (24, 25), (23, 24), (25, 24)):
        assert _luminance(speck[y, x]) > _luminance(baseline[y, x]) * 0.9


def test_anime_skin_shadows_are_warm_and_terminator_glows(anime_renderer):
    color, depth, normals = _material_scene(value=0.5, normal=_unit(1.0, 0.0, -0.8))
    skin = anime_renderer(color, depth, normals, _skin_maps(), key_light=_FRONT_LIGHT)[24, 24]
    cloth = anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)[24, 24]
    assert skin[0] > skin[2] + 0.04 and cloth[2] > cloth[0]
    assert _luminance(skin) > _luminance(cloth)

    terminator = _unit(1.0, 0.0, -0.1)
    color, depth, normals = _material_scene(value=0.5, normal=terminator)
    skin = anime_renderer(color, depth, normals, _skin_maps(), key_light=_FRONT_LIGHT)[24, 24]
    cloth = anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)[24, 24]
    assert skin[0] / skin[2] > cloth[0] / cloth[2] + 0.3


def test_anime_ignores_material_samples_without_flags(anime_renderer):
    color, depth, normals = _material_scene(normal=_KEY_HALF_VECTOR)
    baseline = anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)
    maps = _material_maps(normal=(0.5, 0.5, 1.0, 1.0), env_mask=(1.0, 1.0, 1.0, 1.0),
                          subsurface=(1.0, 0.3, 0.2, 1.0))
    maps["u_material_flags"][:] = 0.0
    np.testing.assert_allclose(
        anime_renderer(color, depth, normals, maps, key_light=_FRONT_LIGHT), baseline, atol=1e-6
    )


def test_anime_env_mask_adds_crisp_highlight_only_near_the_light(anime_renderer):
    half = _unit(0.0, 0.0, 2.0)
    maps = _material_maps(env_mask=(0.8, 0.8, 0.8, 1.0))
    color, depth, facing = _material_scene(normal=half)
    base = anime_renderer(color, depth, facing, key_light=_FRONT_LIGHT)
    lit = anime_renderer(color, depth, facing, maps, key_light=_FRONT_LIGHT)
    assert lit[24, 24, 0] > base[24, 24, 0] + 0.13
    color, depth, away = _material_scene(normal=_unit(0.4, 0.0, 1.0))
    np.testing.assert_allclose(
        anime_renderer(color, depth, away, maps, key_light=_FRONT_LIGHT)[24, 24],
        anime_renderer(color, depth, away, key_light=_FRONT_LIGHT)[24, 24], atol=1e-6,
    )


@pytest.mark.parametrize("distance", [10.0, 100.0])
def test_anime_inks_silhouettes_and_depth_steps_with_coloured_lines(anime_renderer, distance):
    color = np.ones((48, 48, 4), dtype="f4")
    color[:, :, :3] = (0.7, 0.45, 0.3)
    depth = np.full((48, 48), _depth_at_distance(distance), dtype="f4")
    depth[:, :12] = 1.0
    depth[:, 24:] = _depth_at_distance(distance * 1.1)
    result = anime_renderer(color, depth, key_light=_FRONT_LIGHT)
    interior = result[20, 18]
    for x in (12, 23):
        line = result[20, x]
        assert _luminance(line) < _luminance(interior) * 0.45
        # Ink keeps the surface hue rather than being neutral black.
        assert line[0] > line[1] > line[2]
    # Steps get a single line on the nearer surface only.
    np.testing.assert_allclose(result[20, 24], result[20, 30], atol=1e-5)
    np.testing.assert_allclose(result[20, 11], color[20, 11], atol=1e-6)


@pytest.mark.parametrize("distance", [10.0, 100.0])
def test_anime_inks_folds_but_not_smooth_slopes(anime_renderer, distance):
    columns = np.arange(48)
    color, depth, normals = _material_scene(value=0.5)
    slope = distance * (1.0 + 0.4 * (columns - 24) / 48)
    depth[:] = _depth_at_distance(slope)[None, :]
    smooth = anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)
    assert np.ptp(_luminance_rows(smooth[20, 16:32])) < 1e-4
    fold = distance * (1.0 + 0.15 * np.abs(columns - 24) / 24)
    depth[:] = _depth_at_distance(fold)[None, :]
    folded = anime_renderer(color, depth, normals, key_light=_FRONT_LIGHT)
    assert _luminance(folded[20, 24]) < _luminance(folded[20, 20]) * 0.5
    np.testing.assert_allclose(folded[20, 20], smooth[20, 20], atol=1e-5)


def _luminance_rows(pixels):
    return pixels[:, :3] @ np.array([0.2126, 0.7152, 0.0722])


def test_anime_rim_light_brightens_silhouette_edges(anime_renderer):
    color, depth, normals = _material_scene(value=0.5)
    for x in range(12, 20):
        t = (x - 12) / 8
        normals[:, x, :3] = _unit(-(1.0 - t), 0.0, 0.3 + t) * 0.5 + 0.5
    closed = anime_renderer(color, depth, normals, key_light=_unit(-0.6, 0.0, 0.8))
    depth[:, :12] = 1.0
    open_edge = anime_renderer(color, depth, normals, key_light=_unit(-0.6, 0.0, 0.8))
    assert _luminance(open_edge[20, 13]) > _luminance(closed[20, 13]) + 0.05
    np.testing.assert_allclose(open_edge[20, 30], closed[20, 30], atol=1e-6)


def test_anime_without_scene_lighting_uses_fallback_light(anime_renderer):
    color, depth, normals = _material_scene(value=0.5, normal=_unit(0.8, -0.5, -0.2))
    unlit = anime_renderer(color, depth, normals)
    np.testing.assert_allclose(
        anime_renderer(color, depth, normals, key_light=(0.0, 0.0, 0.0)), unlit, atol=1e-6
    )
    # Facing away from the fallback light lands in the cool shadow tone.
    assert unlit[24, 24, 2] > unlit[24, 24, 0]
    assert np.isfinite(unlit).all()


def test_realistic_relights_toward_scene_key_light(preset_renderer):
    key = _unit(-0.6, 0.0, 0.8)
    shades = {}
    for name, normal in (("toward", key), ("away", _unit(0.8, 0.0, 0.6))):
        color, depth, normals = _material_scene(value=0.5, normal=normal)
        shades[name] = preset_renderer(color, depth, normals, key_light=key)[24, 24]
    assert _luminance(shades["toward"]) > 0.45
    assert _luminance(shades["toward"]) > _luminance(shades["away"]) + 0.12
    # The shadow side is filled by the cool sky ambient.
    assert shades["away"][2] > shades["away"][0]


def test_realistic_skin_terminator_stays_warm(preset_renderer):
    key = _unit(-0.6, 0.0, 0.8)
    color, depth, normals = _material_scene(value=0.5, normal=_unit(0.8, 0.0, 0.6))
    cloth = preset_renderer(color, depth, normals, key_light=key)[24, 24]
    skin = preset_renderer(color, depth, normals, _skin_maps(), key_light=key)[24, 24]
    assert _luminance(skin) > _luminance(cloth) + 0.1
    assert skin[0] > skin[2] + 0.03


def test_realistic_without_scene_lighting_keeps_baked_shading(preset_renderer):
    color, depth, normals = _material_scene(value=0.5, normal=_unit(0.8, 0.0, 0.6))
    np.testing.assert_allclose(
        preset_renderer(color, depth, normals, key_light=(0.0, 0.0, 0.0)),
        preset_renderer(color, depth, normals), atol=1e-6,
    )


def _glossy_pixel(glossy_renderer, normal, maps=None, key_light=_FRONT_LIGHT):
    color, depth, normals = _material_scene(value=0.5, normal=normal)
    diffuse, specular = glossy_renderer(color, depth, normals, maps, key_light=key_light)
    return diffuse[24, 24], specular[24, 24]


def _glossy_skin(normal_alpha=None, env_mask=None):
    normal = None if normal_alpha is None else (0.5, 0.5, 1.0, normal_alpha)
    return _material_maps(subsurface=(1.0, 0.3, 0.2, 1.0), normal=normal, env_mask=env_mask)


def _tilted(degrees):
    angle = np.radians(degrees)
    return (np.sin(angle), 0.0, np.cos(angle))


def test_glossy_preserves_background_and_splits_outputs(glossy_renderer):
    color, depth, normals = _material_scene(value=0.5)
    depth[:4] = 1.0
    color[:, :, 3] = 0.7
    diffuse, specular = glossy_renderer(color, depth, normals, _glossy_skin(), key_light=_FRONT_LIGHT)
    assert np.isfinite(diffuse).all() and np.isfinite(specular).all()
    np.testing.assert_allclose(diffuse[:4], color[:4], atol=1e-6)
    np.testing.assert_allclose(specular[:4], 0.0, atol=1e-6)
    np.testing.assert_allclose(diffuse[:, :, 3], 0.7, atol=1e-6)


def _glossy_cloth():
    return _material_maps(env_mask=(1.0, 1.0, 1.0, 1.0))


def test_glossy_oil_has_dual_lobe_specular(glossy_renderer):
    # Glossy cloth keeps its full normal map, so it shows the raw lobe shape.
    _, peak = _glossy_pixel(glossy_renderer, (0.0, 0.0, 1.0), _glossy_cloth())
    _, near = _glossy_pixel(glossy_renderer, _tilted(4.0), _glossy_cloth())
    _, sheen = _glossy_pixel(glossy_renderer, _tilted(10.0), _glossy_cloth())
    _, far = _glossy_pixel(glossy_renderer, _tilted(25.0), _glossy_cloth())
    # A crisp, over-bright wet pinpoint sits on a soft, wide oil sheen.
    assert peak[0] > 1.5
    assert near[0] < peak[0] * 0.3
    assert 0.03 < sheen[0] < near[0]
    assert far[0] < sheen[0]
    _, skin = _glossy_pixel(glossy_renderer, (0.0, 0.0, 1.0), _glossy_skin())
    assert skin[0] > 1.5 and skin[3] == pytest.approx(1.0)


def test_glossy_specular_never_reaches_the_diffuse_output(glossy_renderer):
    lit_diffuse, lit = _glossy_pixel(glossy_renderer, (0.0, 0.0, 1.0), _glossy_skin())
    dark_diffuse, dark = _glossy_pixel(
        glossy_renderer, (0.0, 0.0, 1.0), _glossy_skin(), key_light=(0.0, -1.0, 0.0)
    )
    assert lit[0] > dark[0] + 1.0
    assert _luminance(lit_diffuse) > _luminance(dark_diffuse)
    assert (lit_diffuse[:3] <= 1.0).all()


def test_glossy_diffuse_is_hentai3d_under_a_wet_film(glossy_renderer, hentai3d_renderer):
    color, depth, normals = _material_scene(value=0.5, normal=_tilted(30.0))

    def both(maps, key_light=_FRONT_LIGHT):
        glossy = glossy_renderer(color, depth, normals, maps, key_light)[0][24, 24, :3]
        hentai = hentai3d_renderer(color, depth, normals, maps, key_light)[0][24, 24, :3]
        return glossy, hentai

    # Plain cloth has no oil coverage: only the half warm tint differs.
    glossy, hentai = both(None)
    assert np.allclose(glossy / hentai, (1.02, 0.995, 0.98), atol=0.03)
    # Oiled skin is darker and richer than 3D Hentai's dry skin.
    glossy, hentai = both(_glossy_skin())
    assert _luminance(glossy) < _luminance(hentai) * 0.95
    assert glossy[0] / glossy[2] > hentai[0] / hentai[2]


def test_glossy_keeps_hentai3d_rim_and_tunable_uniforms(
    shader_renderer_factory, hentai3d_renderer,
):
    def glossy(normal, maps, key_light, **uniforms):
        color, depth, normals = _material_scene(value=0.5, normal=normal)
        return shader_renderer_factory(
            "glossy", color, depth, normals, maps, key_light, outputs=2,
            uniforms={"u_gloss_intensity": 1.0, "u_rim_intensity": 1.0, **uniforms},
        )[1][24, 24]

    contour, side = _unit(0.968, 0.0, 0.25), _unit(0.7, 0.0, 0.7)
    # Matte cloth gets no oil, only 3D Hentai's rim light.
    rim = glossy(contour, None, side)
    color, depth, normals = _material_scene(value=0.5, normal=contour)
    hentai = hentai3d_renderer(color, depth, normals, None, key_light=side)[1][24, 24]
    np.testing.assert_allclose(rim[:3], hentai[:3], rtol=1e-4, atol=1e-6)
    assert rim[0] > 0.05
    np.testing.assert_allclose(glossy(contour, None, side, u_rim_intensity=2.0)[:3],
                               rim[:3] * 2.0, rtol=1e-4)
    full = glossy((0.0, 0.0, 1.0), _glossy_cloth(), _FRONT_LIGHT)
    half = glossy((0.0, 0.0, 1.0), _glossy_cloth(), _FRONT_LIGHT, u_gloss_intensity=0.5)
    np.testing.assert_allclose(half[:3], full[:3] * 0.5, rtol=1e-3, atol=1e-5)


def test_glossy_specular_mask_has_a_floor_on_skin_only(glossy_renderer):
    _, masked_out = _glossy_pixel(glossy_renderer, (0.0, 0.0, 1.0), _glossy_skin(0.0))
    _, masked_in = _glossy_pixel(glossy_renderer, (0.0, 0.0, 1.0), _glossy_skin(1.0))
    assert masked_out[0] == pytest.approx(masked_in[0] * 0.6, rel=1e-3)

    _, plain_cloth = _glossy_pixel(glossy_renderer, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(plain_cloth, 0.0, atol=1e-6)
    _, glossy_cloth = _glossy_pixel(
        glossy_renderer, (0.0, 0.0, 1.0), _material_maps(env_mask=(1.0, 1.0, 1.0, 1.0))
    )
    assert glossy_cloth[0] > 1.5 and glossy_cloth[3] == 0.0


def test_glossy_fresnel_reflects_the_studio_at_grazing_angles(glossy_renderer):
    # A light in the surface plane isolates the cubemap reflection.
    side_light = (0.0, -1.0, 0.0)
    # Glossy cloth: skin would smooth the grazing normal back towards the geometry.
    _, facing = _glossy_pixel(glossy_renderer, (0.0, 0.0, 1.0), _glossy_cloth(), side_light)
    _, grazing = _glossy_pixel(glossy_renderer, _unit(0.95, 0.0, 0.31), _glossy_cloth(), side_light)
    assert grazing[0] > facing[0] * 1.8
    assert grazing[0] > 0.1


def test_glossy_shades_creases_and_dims_oil_inside_them(glossy_renderer):
    color = np.full((64, 64, 4), 0.5, dtype="f4")
    color[:, :, 3] = 1.0
    flat_depth = np.full((64, 64), _depth_at_distance(10.0), dtype="f4")
    contact_depth = flat_depth.copy()
    contact_depth[28:37, 34:43] = _depth_at_distance(9.7)
    maps = _material_maps(size=64, env_mask=(1.0, 1.0, 1.0, 1.0))
    flat_diffuse, flat_specular = glossy_renderer(color, flat_depth, None, maps, _FRONT_LIGHT)
    diffuse, specular = glossy_renderer(color, contact_depth, None, maps, _FRONT_LIGHT)
    assert diffuse[32, 33, 0] < flat_diffuse[32, 33, 0] - 0.002
    assert specular[32, 33, 0] < flat_specular[32, 33, 0] * 0.95
    # Flat areas away from the contact are untouched.
    np.testing.assert_allclose(diffuse[16, 16], flat_diffuse[16, 16], atol=1e-5)
    np.testing.assert_allclose(specular[16, 16], flat_specular[16, 16], atol=1e-5)


def test_glossy_silhouette_without_scene_normal_stays_finite_and_lit(glossy_renderer):
    color, depth, normals = _material_scene(value=0.5)
    normals[:] = 0.0
    maps = _material_maps(env_mask=(1.0, 1.0, 1.0, 1.0))
    _, inner = glossy_renderer(color, depth, normals, maps, _FRONT_LIGHT)
    depth[:, 25:] = 1.0
    diffuse, edge = glossy_renderer(color, depth, normals, maps, _FRONT_LIGHT)
    assert np.isfinite(diffuse).all() and np.isfinite(edge).all()
    # The edge pixel must not take a normal from the background behind it.
    np.testing.assert_allclose(edge[24, 24, :3], inner[24, 24, :3], rtol=0.05)


def test_glossy_smooths_skin_pores_under_highlights(glossy_renderer):
    def speck(maps):
        color, depth, normals = _material_scene(value=0.5)
        normals[24, 24, :3] = np.asarray(_tilted(8.0)) * 0.5 + 0.5
        return glossy_renderer(color, depth, normals, maps, key_light=_FRONT_LIGHT)[1][24, 24]

    skin = speck(_glossy_skin(env_mask=(1.0, 1.0, 1.0, 1.0)))
    cloth = speck(_material_maps(env_mask=(1.0, 1.0, 1.0, 1.0)))
    # The pore keeps more of the highlight on skin than on equally glossy cloth.
    assert skin[0] > cloth[0] * 1.5


def _split_composite(ctx, diffuse, specular, ssss):
    program, vertex_array = OutfitViewportWidget._compile_postprocess_variant(
        SimpleNamespace(ctx=ctx), False, ssss, True
    )
    height, width = diffuse.shape[:2]
    projection = glm.perspective(glm.radians(42.0), 1.0, 1.0, 1000.0)
    depth = np.full((height, width), _depth_at_distance(10.0), dtype="f4")
    textures = [
        ctx.texture((width, height), 4, diffuse.astype("f4").tobytes(), dtype="f4"),
        ctx.texture((width, height), 1, depth.tobytes(), dtype="f4"),
        ctx.texture((width, height), 4, specular.astype("f4").tobytes(), dtype="f4"),
    ]
    output = ctx.texture((width, height), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[output])
    try:
        for location, texture in enumerate(textures):
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            texture.use(location=location)
        program["u_scene_color"].value = 0
        program["u_scene_specular"].value = 2
        if ssss:
            program["u_scene_depth"].value = 1
            program["u_viewport_size"].value = (float(width), float(height))
            program["u_inverse_projection"].write(bytes(glm.transpose(glm.inverse(projection))))
            program["u_ssss_radius"].value = 3.0
            program["u_ssss_strength"].value = 1.0
            program["u_ssss_falloff_color"].value = (1.0, 1.0, 1.0)
        fbo.use()
        ctx.viewport = (0, 0, width, height)
        vertex_array.render(moderngl.TRIANGLES, vertices=3)
        return np.frombuffer(fbo.read(components=4, dtype="f4"), dtype="f4").reshape(
            height, width, 4
        ).copy()
    finally:
        for resource in (fbo, output, *textures, vertex_array, program):
            resource.release()


def test_split_composite_scatters_masked_diffuse_and_adds_sharp_specular(
    shader_renderer_factory
):
    ctx = shader_renderer_factory.ctx
    diffuse = np.zeros((32, 32, 4), dtype="f4")
    diffuse[:, :, 3] = 1.0
    diffuse[:, :16, :3] = (0.8, 0.2, 0.2)  # skin
    diffuse[:, 16:, :3] = (0.2, 0.2, 0.8)  # cloth
    diffuse[16, 8, :3] = 0.0
    specular = np.zeros((32, 32, 4), dtype="f4")
    specular[:, :16, 3] = 1.0
    specular[16, 8, :3] = 2.0

    result = _split_composite(ctx, diffuse, specular, ssss=True)
    # Masked skin is blurred, but the highlight it carries is added unblurred.
    assert result[16, 8, 0] > 2.3
    np.testing.assert_allclose(result[22, 10, :3], (0.8, 0.2, 0.2), atol=1e-6)
    assert result[16, 5, 0] < 0.8
    # Cloth is unmasked: untouched, and it does not bleed into the skin edge.
    np.testing.assert_allclose(result[:, 16:], diffuse[:, 16:], atol=1e-6)
    np.testing.assert_allclose(result[8, 15, :3], (0.8, 0.2, 0.2), atol=1e-6)

    without_ssss = _split_composite(ctx, diffuse, specular, ssss=False)
    expected = diffuse.copy()
    expected[:, :, :3] += specular[:, :, :3]
    np.testing.assert_allclose(without_ssss, expected, atol=1e-6)


@pytest.fixture
def standard_renderer(shader_renderer_factory):
    return lambda color, depth, normals=None, materials=None, key_light=None: (
        shader_renderer_factory("standard", color, depth, normals, materials, key_light)
    )


# Standard is a simplified Realistic: it keeps Realistic's lighting, contact
# shading, subsurface and specular behaviour, without the micro-detail passes.
@pytest.mark.parametrize("realistic_test", [
    test_preset_render_preserves_background_and_alpha,
    test_preset_uses_shading_normals_and_falls_back_when_invalid,
    test_normal_cavity_ignores_invalid_and_discontinuous_neighbors,
    test_realistic_ignores_material_samples_without_flags,
    test_realistic_env_mask_adds_glossy_highlight_and_preserves_background,
    test_realistic_complex_env_mask_metalness_tints_reflection,
    test_realistic_uses_tangent_normal_alpha_as_fallback_specular,
    test_realistic_subsurface_diffuses_lighting_and_tints_shadows,
    test_realistic_subsurface_stays_on_its_own_surface,
    test_realistic_relights_toward_scene_key_light,
    test_realistic_skin_terminator_stays_warm,
    test_realistic_without_scene_lighting_keeps_baked_shading,
], ids=lambda test: test.__name__.removeprefix("test_"))
def test_standard_matches_realistic_behaviour(standard_renderer, realistic_test):
    realistic_test(standard_renderer)


@pytest.mark.parametrize("distance", [10.0, 100.0])
def test_standard_adds_contact_shadows_without_darkening_flat_surfaces(standard_renderer, distance):
    test_preset_render_adds_contact_shadows_without_darkening_flat_surfaces(
        standard_renderer, distance
    )


def test_standard_is_discovered_with_realistic_postprocessing():
    assert ("standard", "Standard") in OutfitViewportWidget._discover_shader_presets()
    assert load_postprocess_profile(SHADERS / "standard.frag") == load_postprocess_profile(
        SHADERS / "realistic.frag"
    )


HENTAI3D_UNIFORMS = json.loads(
    (SHADERS / "hentai3d.json").read_text(encoding="utf-8")
)["uniforms"]


@pytest.fixture
def hentai3d_renderer(shader_renderer_factory):
    """Returns the (diffuse, gloss + rim + SSSS mask) outputs of the 3D Hentai preset."""
    def render(color, depth, normals=None, materials=None, key_light=None, **uniforms):
        return shader_renderer_factory(
            "hentai3d", color, depth, normals, materials, key_light, outputs=2,
            uniforms={**HENTAI3D_UNIFORMS, **uniforms},
        )
    return render


def _hentai3d_pixel(renderer, normal, maps=None, key_light=_FRONT_LIGHT, **uniforms):
    color, depth, normals = _material_scene(value=0.5, normal=normal)
    diffuse, specular = renderer(color, depth, normals, maps, key_light=key_light, **uniforms)
    return diffuse[24, 24], specular[24, 24]


def test_hentai3d_preserves_background_and_splits_outputs(hentai3d_renderer):
    color, depth, normals = _material_scene(value=0.5)
    depth[:4] = 1.0
    color[:, :, 3] = 0.7
    diffuse, specular = hentai3d_renderer(color, depth, normals, _glossy_skin(),
                                          key_light=_FRONT_LIGHT)
    assert np.isfinite(diffuse).all() and np.isfinite(specular).all()
    np.testing.assert_allclose(diffuse[:4], color[:4], atol=1e-6)
    np.testing.assert_allclose(specular[:4], 0.0, atol=1e-6)
    np.testing.assert_allclose(diffuse[:, :, 3], 0.7, atol=1e-6)
    assert specular[24, 24, 3] == pytest.approx(1.0)
    _, cloth = _hentai3d_pixel(hentai3d_renderer, (0.0, 0.0, 1.0))
    assert cloth[3] == pytest.approx(0.0)


def test_hentai3d_skin_is_glossy_with_a_tight_highlight(hentai3d_renderer):
    def gloss(degrees, maps=None, **uniforms):
        return _hentai3d_pixel(hentai3d_renderer, _tilted(degrees),
                               _glossy_skin() if maps is None else maps, **uniforms)[1][0]

    # Skin keeps 40% of the normal map's tilt, so 15 and 40 degrees act as 6 and 16.
    peak, near, far = gloss(0.0), gloss(15.0), gloss(40.0)
    assert peak > 0.3
    assert near < peak * 0.8
    assert 0.0 < far < near
    assert gloss(0.0, u_gloss_intensity=0.0) < peak * 0.05
    # Untextured cloth gets no highlight head-on.
    _, cloth = _hentai3d_pixel(hentai3d_renderer, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(cloth[:3], 0.0, atol=1e-6)


def test_hentai3d_rim_lights_lit_contours_brightest_on_skin(hentai3d_renderer):
    contour = _unit(0.968, 0.0, 0.25)
    side_light = _unit(0.7, 0.0, 0.7)

    def rim(maps=None, key_light=side_light, **uniforms):
        return _hentai3d_pixel(hentai3d_renderer, contour, maps, key_light, **uniforms)[1]

    cloth = rim()
    assert cloth[0] > 0.05 and cloth[0] > cloth[1]
    assert rim(key_light=_unit(-0.7, 0.0, 0.7))[0] < cloth[0] * 0.1
    np.testing.assert_allclose(rim(u_rim_intensity=0.0)[:3], 0.0, atol=1e-6)
    np.testing.assert_allclose(rim(u_rim_intensity=2.0)[:3], cloth[:3] * 2.0, rtol=1e-4)


def test_hentai3d_skin_is_pink_flushed_and_more_saturated_than_standard(
    hentai3d_renderer, standard_renderer,
):
    color, depth, normals = _material_scene(value=0.5)
    color[:, :, :3] = (0.6, 0.45, 0.4)
    maps = _glossy_skin()
    skin, _ = hentai3d_renderer(color, depth, normals, maps, key_light=_FRONT_LIGHT)
    standard = standard_renderer(color, depth, normals, maps, key_light=_FRONT_LIGHT)
    skin, standard = skin[24, 24], standard[24, 24]
    assert skin[0] / skin[1] > standard[0] / standard[1] * 1.05
    cloth, _ = hentai3d_renderer(color, depth, normals, None, key_light=_FRONT_LIGHT)
    standard_cloth = standard_renderer(color, depth, normals, None, key_light=_FRONT_LIGHT)
    cloth, standard_cloth = cloth[24, 24], standard_cloth[24, 24]
    assert cloth[0] - cloth[2] > standard_cloth[0] - standard_cloth[2]


def test_hentai3d_soft_ramp_steepens_the_terminator(hentai3d_renderer, standard_renderer):
    color, depth, normals = _material_scene(value=0.5)
    lit_side, shadow_side = _unit(0.968, 0.0, 0.25), _unit(0.9987, 0.0, -0.05)

    def step(renderer, unpack):
        lit = unpack(renderer(color, depth, normals, None, key_light=lit_side))
        dark = unpack(renderer(color, depth, normals, None, key_light=shadow_side))
        return _luminance(lit[24, 24]) - _luminance(dark[24, 24])

    assert step(hentai3d_renderer, lambda out: out[0]) > step(standard_renderer, lambda out: out) * 1.15


def test_hentai3d_is_discovered_with_label_uniforms_and_pink_ssss():
    assert ("hentai3d", "3D Hentai") in OutfitViewportWidget._discover_shader_presets()
    assert set(HENTAI3D_UNIFORMS) == {"u_gloss_intensity", "u_rim_intensity"}
    profile = load_postprocess_profile(SHADERS / "hentai3d.frag")
    standard = load_postprocess_profile(SHADERS / "standard.frag")
    assert profile.enable_gtao and profile.enable_ssss and profile.ssss_after_preset
    assert profile.sss_strength > standard.sss_strength


def test_mesh_shader_outputs_normal_mapped_normals():
    app = QApplication.instance() or QApplication([])
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    context = QOpenGLContext()
    context.setFormat(fmt)
    if not surface.isValid() or not context.create() or not context.makeCurrent(surface):
        pytest.skip("OpenGL 3.3 offscreen context is unavailable")

    ctx = moderngl.create_context(require=330)
    program = ctx.program(
        vertex_shader=(SHADERS / "default.vert").read_text(encoding="utf-8"),
        fragment_shader=(SHADERS / "default.frag").read_text(encoding="utf-8"),
    )
    vertex_shader = """
        #version 330
        out vec3 lightFrontal, lightDirectional0, lightDirectional1, lightDirectional2;
        out vec3 viewDir, n;
        out mat3 mv_tbn;
        out float maskFactor;
        out vec3 weightColor;
        out vec4 vColor;
        out vec2 vUV;
        void main() {
            vec2 positions[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
            gl_Position = vec4(positions[gl_VertexID], 0, 1);
            lightFrontal = lightDirectional0 = lightDirectional1 = lightDirectional2 = vec3(0, 0, 1);
            viewDir = n = vec3(0, 0, 1);
            mv_tbn = mat3(1);
            maskFactor = 1.0;
            weightColor = vec3(1);
            vColor = vec4(1);
            vUV = vec2(0.5);
        }
    """
    program.release()
    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=(SHADERS / "default.frag").read_text(encoding="utf-8"),
    )
    vao = ctx.vertex_array(program, [])
    output = ctx.texture((8, 8), 4, dtype="f4")
    normals = ctx.texture((8, 8), 4, dtype="f4")
    depth = ctx.depth_texture((8, 8))
    fbo = ctx.framebuffer([output, normals], depth)
    diffuse = ctx.texture((1, 1), 4, np.ones(4, dtype="f4").tobytes(), dtype="f4")
    normal_map = ctx.texture(
        (1, 1), 4, np.array([0.8, 0.5, 0.9, 1.0], dtype="f4").tobytes(), dtype="f4"
    )
    diffuse.use(location=0)
    normal_map.use(location=1)
    for name, value in {
        "texDiffuse": 0,
        "texNormal": 1,
        "bShowTexture": True,
        "bLightEnabled": True,
        "bNormalMap": True,
        "bModelSpace": False,
        "prop.uvScale": (1.0, 1.0),
        "prop.alpha": 1.0,
        "alphaThreshold": -1.0,
    }.items():
        if name in program:
            program[name].value = value
    program["mv_normalMatrix"].write(bytes(glm.mat3(1.0)))
    fbo.use()
    ctx.viewport = (0, 0, 8, 8)

    def read_normal():
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        vao.render(moderngl.TRIANGLES, vertices=3)
        return np.frombuffer(fbo.read(attachment=1, components=4, dtype="f4"), dtype="f4").reshape(
            (8, 8, 4)
        )[4, 4].copy()

    mapped = read_normal()
    np.testing.assert_allclose(mapped, (0.8, 0.5, 0.9, 1.0), atol=1e-5)
    program["bModelSpace"].value = True
    np.testing.assert_allclose(read_normal(), (0.2, 0.5, 0.9, 1.0), atol=1e-5)
    program["bModelSpace"].value = False
    normal_map.write(np.array([0.8, 0.5, 0.0, 1.0], dtype="f4").tobytes())
    np.testing.assert_allclose(read_normal(), (0.8, 0.5, 0.9, 1.0), atol=1e-5)
    program["bNormalMap"].value = False
    np.testing.assert_allclose(read_normal(), (0.5, 0.5, 1.0, 1.0), atol=1e-5)
    program["bLightEnabled"].value = False
    np.testing.assert_allclose(read_normal(), (0.5, 0.5, 1.0, 1.0), atol=1e-5)
    program["alphaThreshold"].value = 1.0
    np.testing.assert_allclose(read_normal(), (0.0, 0.0, 0.0, 0.0), atol=1e-5)

    program["alphaThreshold"].value = -1.0
    program["bLightEnabled"].value = True
    program["bNormalMap"].value = True
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    program["prop.alpha"].value = 0.25
    vao.render(moderngl.TRIANGLES, vertices=3)
    blended_normal = np.frombuffer(
        fbo.read(attachment=1, components=4, dtype="f4"), dtype="f4"
    ).reshape((8, 8, 4))[4, 4]
    np.testing.assert_allclose(blended_normal, (0.8, 0.5, 0.9, 1.0), atol=1e-5)
    fbo.color_mask = ((True, True, True, True), (False, False, False, False))
    program["bNormalMap"].value = False
    vao.render(moderngl.TRIANGLES, vertices=3)
    preserved = np.frombuffer(
        fbo.read(attachment=1, components=4, dtype="f4"), dtype="f4"
    ).reshape((8, 8, 4))[4, 4]
    np.testing.assert_allclose(preserved, (0.8, 0.5, 0.9, 1.0), atol=1e-5)
    fbo.color_mask = ((True, True, True, True), (True, True, True, True))
    ctx.disable(moderngl.BLEND)

    material_values = [
        np.array([0.2, 0.3, 0.4, 0.6], dtype="f4"),
        np.array([0.1, 0.5, 0.7, 0.9], dtype="f4"),
        np.array([0.8, 0.6, 0.4, 0.2], dtype="f4"),
    ]
    maps = [ctx.texture((1, 1), 4, value.tobytes(), dtype="f4") for value in material_values]
    for location, (name, texture) in enumerate(zip(
        ("texPresetNormal", "texPresetEnvMask", "texPresetSubsurface"), maps
    ), start=9):
        texture.use(location=location)
        program[name].value = location
    for name in ("bPresetHasNormal", "bPresetHasEnvMask", "bPresetHasSubsurface",
                 "bPresetNormalModelSpace"):
        program[name].value = True
    program["alphaThreshold"].value = -1.0
    fbo.depth_mask = True
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    ctx.enable(moderngl.DEPTH_TEST)
    vao.render(moderngl.TRIANGLES, vertices=3)
    before_depth = np.frombuffer(depth.read(), dtype="f4").copy()
    item = SimpleNamespace(shape_data=SimpleNamespace(
        alpha_prop=SimpleNamespace(alpha_test=False)
    ))
    widget = SimpleNamespace(
        ctx=ctx,
        scene_depth=depth,
        material_fbo=None,
        material_clear_fbo=None,
        material_textures=[],
        _set_uniform=lambda name, value: setattr(program[name], "value", value),
        _render_shape_item=lambda _item, wireframe: vao.render(moderngl.TRIANGLES, vertices=3),
    )
    OutfitViewportWidget._render_preset_materials(widget, [item], [], 8, 8)
    np.testing.assert_array_equal(np.frombuffer(depth.read(), dtype="f4"), before_depth)
    for index, value in enumerate(material_values + [np.ones(4, dtype="f4")]):
        pixels = np.frombuffer(
            widget.material_fbo.read(attachment=index, components=4, dtype="f4"), dtype="f4"
        ).reshape((8, 8, 4))
        np.testing.assert_allclose(pixels[4, 4], value, atol=1.0 / 255.0)
    program["bPresetHasNormal"].value = False
    program["bPresetHasEnvMask"].value = False
    program["bPresetHasSubsurface"].value = False
    OutfitViewportWidget._render_preset_materials(widget, [item], [], 8, 8)
    for index in range(4):
        pixels = np.frombuffer(
            widget.material_fbo.read(attachment=index, components=4, dtype="f4"), dtype="f4"
        ).reshape((8, 8, 4))
        np.testing.assert_allclose(pixels[4, 4], np.zeros(4), atol=1e-6)
    np.testing.assert_array_equal(np.frombuffer(depth.read(), dtype="f4"), before_depth)
    widget.material_fbo.release()
    widget.material_clear_fbo.release()
    for texture in widget.material_textures + maps:
        texture.release()

    vao.release()
    program.release()
    fbo.release()
    output.release()
    normals.release()
    depth.release()
    diffuse.release()
    normal_map.release()
    ctx.release()
    context.doneCurrent()
    app.processEvents()
