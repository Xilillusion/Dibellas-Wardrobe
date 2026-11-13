import glm
import moderngl
import numpy as np
import pytest

from test_shader_presets import (  # noqa: F401
    SHADERS,
    _material_maps,
    _material_scene,
    shader_renderer_factory,
)


@pytest.mark.parametrize("preset", ["standard", "realistic", "glossy", "vivid", "darkfantasy"])
@pytest.mark.parametrize("skin", [False, True])
def test_depth_slope_never_replaces_smooth_shading_normal(shader_renderer_factory, preset, skin):
    color, depth, normals = _material_scene(normal=(0.8, 0, 0.6))
    maps = _material_maps(env_mask=(0.7, 0.7, 0.7, 1))
    if skin:
        maps = _material_maps(env_mask=(0.7, 0.7, 0.7, 1), subsurface=(0.5, 0.3, 0.2, 1))
    outputs = 1 if preset in ("standard", "realistic") else 2
    results = []
    for slope in (-0.08, 0.08):
        projection = glm.perspective(glm.radians(42.0), 1.0, 1.0, 1000.0)
        x = (np.arange(48) + 0.5) / 48 * 2 - 1
        distance = 10.0 / (1.0 + slope * x * np.tan(np.radians(21.0)))
        z = -distance
        ndc = (projection[2][2] * z + projection[3][2]) / -z
        depth[:] = (ndc + 1) * 0.5
        result = shader_renderer_factory(
            preset, color, depth, normals, maps, key_light=(1, 0, 0), outputs=outputs,
        )
        results.append([image[24, 24] for image in result] if outputs == 2 else [result[24, 24]])
    # Depth taps can vary contact/scatter by less than a tenth of a percent.
    np.testing.assert_allclose(results[0], results[1], atol=5e-4, rtol=2e-4)


@pytest.mark.parametrize("preset", ["standard", "realistic", "glossy", "vivid", "darkfantasy"])
def test_missing_scene_normal_does_not_relight_from_depth(shader_renderer_factory, preset):
    color, depth, normals = _material_scene()
    normals[:] = 0
    outputs = 1 if preset in ("standard", "realistic") else 2
    results = []
    for light in ((1, 0, 0), (-1, 0, 0)):
        result = shader_renderer_factory(
            preset, color, depth, normals, key_light=light, outputs=outputs,
        )
        if outputs == 2:
            diffuse, specular = result
            np.testing.assert_allclose(specular[:, :, :3], 0, atol=1e-6)
        else:
            diffuse = result
        results.append(diffuse)
    np.testing.assert_allclose(results[0], results[1], atol=1e-6)


@pytest.mark.parametrize("preset", ["standard", "realistic", "glossy", "vivid", "darkfantasy"])
@pytest.mark.parametrize("normal", [(0.6, 0, 0.8), (0.0, 0.0, 1.0)])
@pytest.mark.parametrize("skin", [False, True])
def test_inverted_normals_are_flipped_without_flattening(shader_renderer_factory, preset, normal, skin):
    color, depth, normals = _material_scene(normal=normal)
    maps = _material_maps(env_mask=(0.7, 0.8, 0.0, 1))
    if skin:
        maps = _material_maps(env_mask=(0.7, 0.8, 0.0, 1), subsurface=(0.5, 0.3, 0.2, 1))
    outputs = 1 if preset in ("standard", "realistic") else 2
    expected = shader_renderer_factory(
        preset, color, depth, normals, maps, key_light=(0.6, 0, 0.8), outputs=outputs,
    )
    normals[:, :, :3] = 1.0 - normals[:, :, :3]
    actual = shader_renderer_factory(
        preset, color, depth, normals, maps, key_light=(0.6, 0, 0.8), outputs=outputs,
    )
    if outputs == 2:
        for correct, inverted in zip(expected, actual):
            np.testing.assert_allclose(inverted[24, 24], correct[24, 24], atol=1e-5)
    else:
        np.testing.assert_allclose(actual[24, 24], expected[24, 24], atol=1e-5)


@pytest.mark.parametrize("preset", ["standard", "realistic", "glossy", "vivid", "darkfantasy"])
@pytest.mark.parametrize("agreement", [0.0, 0.10, 0.149, -0.10])
def test_unreliable_normals_never_drive_key_relighting(
    shader_renderer_factory, preset, agreement,
):
    color, depth, normals = _material_scene(
        value=0.08, normal=(np.sqrt(1 - agreement * agreement), 0, agreement),
    )
    maps = _material_maps(env_mask=(1, 1, 1, 1))
    outputs = 1 if preset in ("standard", "realistic") else 2
    actual = shader_renderer_factory(
        preset, color, depth, normals, maps, key_light=(1, 0, 0), outputs=outputs,
    )
    normals[:, :, 3] = 0
    expected = shader_renderer_factory(
        preset, color, depth, normals, maps, key_light=(1, 0, 0), outputs=outputs,
    )
    if outputs == 2:
        for guarded, missing in zip(actual, expected):
            np.testing.assert_allclose(guarded[24, 24], missing[24, 24], atol=1e-5)
        diffuse, specular = actual
        np.testing.assert_allclose(specular[24, 24, :3], 0, atol=1e-6)
    else:
        diffuse = actual
        np.testing.assert_allclose(diffuse[24, 24], expected[24, 24], atol=1e-5)
    assert np.max(diffuse[24, 24, :3]) < 0.12


@pytest.mark.parametrize("preset", ["standard", "realistic", "glossy", "vivid", "darkfantasy"])
def test_normals_above_hemisphere_threshold_still_relight(shader_renderer_factory, preset):
    normal = (np.sqrt(1 - 0.151 ** 2), 0, 0.151)
    color, depth, normals = _material_scene(value=0.25, normal=normal)
    outputs = 1 if preset in ("standard", "realistic") else 2
    actual = shader_renderer_factory(
        preset, color, depth, normals, key_light=normal, outputs=outputs,
    )
    normals[:, :, 3] = 0
    missing = shader_renderer_factory(
        preset, color, depth, normals, key_light=normal, outputs=outputs,
    )
    if outputs == 2:
        actual, missing = actual[0], missing[0]
    assert np.max(np.abs(actual[24, 24, :3] - missing[24, 24, :3])) > 0.01


@pytest.mark.parametrize("preset", ["standard", "realistic", "glossy", "vivid", "darkfantasy"])
def test_smooth_normals_still_shade_without_depth_neighbours(shader_renderer_factory, preset):
    color, depth, normals = _material_scene()
    outputs = 1 if preset in ("standard", "realistic") else 2
    full = shader_renderer_factory(preset, color, depth, normals, key_light=(0, 0, 1), outputs=outputs)
    isolated_depth = np.ones_like(depth)
    isolated_depth[24, 24] = depth[24, 24]
    isolated = shader_renderer_factory(
        preset, color, isolated_depth, normals, key_light=(0, 0, 1), outputs=outputs,
    )
    if outputs == 2:
        for expected, actual in zip(full, isolated):
            np.testing.assert_allclose(actual[24, 24], expected[24, 24], atol=1e-5)
    elif preset == "realistic":
        # An isolated pixel has no surface-detail taps, but is still relit.
        relit = 0.45 * np.array([0.40 + 0.68, 0.40 + 0.66, 0.415 + 0.62]) * 1.1
        luminance = relit @ np.array([0.2126, 0.7152, 0.0722])
        expected = relit * (1 + 0.4 * (luminance - 0.5) * (1 - luminance))
        np.testing.assert_allclose(isolated[24, 24, :3], expected, atol=1e-5)
    else:
        np.testing.assert_allclose(isolated[24, 24], full[24, 24], atol=1e-5)


@pytest.mark.parametrize("lighting", [False, True])
@pytest.mark.parametrize("skinned", [False, True])
def test_mesh_unmapped_normals_are_smooth_and_view_space(shader_renderer_factory, lighting, skinned):
    ctx = shader_renderer_factory.ctx
    program = ctx.program(
        vertex_shader=(SHADERS / "default.vert").read_text(encoding="utf-8"),
        fragment_shader=(SHADERS / "default.frag").read_text(encoding="utf-8"),
    )
    rotation = glm.mat3(glm.rotate(glm.mat4(1), glm.radians(35.0), glm.vec3(0, 0, 1)))
    skin_matrix = glm.rotate(glm.mat4(1), glm.radians(25.0), glm.vec3(0, 0, 1)) if skinned else glm.mat4(1)
    positions = np.array([[-1, -1, 0], [3, -1, 0], [-1, 3, 0]], dtype="f4")
    positions = np.array([
        glm.vec3(glm.inverse(skin_matrix) * glm.vec4(*position, 1))
        for position in positions
    ], dtype="f4")
    smooth = np.array([[0, 0, 1], [0.6, 0, 0.8], [0, 0.6, 0.8]], dtype="f4")
    data = {
        "in_position": (positions, "3f"),
        "in_normal": (smooth, "3f"),
        "in_uv": (np.zeros((3, 2), dtype="f4"), "2f"),
        "in_tangent": (np.tile([1, 0, 0], (3, 1)).astype("f4"), "3f"),
        "in_bitangent": (np.tile([0, 1, 0], (3, 1)).astype("f4"), "3f"),
        "in_color": (np.ones((3, 4), dtype="f4"), "4f"),
        "in_bone_indices": (np.zeros((3, 4), dtype="i4"), "4i"),
        "in_bone_weights": (np.tile([1, 0, 0, 0], (3, 1)).astype("f4"), "4f"),
    }
    buffers = []
    attributes = []
    for name, (array, fmt) in data.items():
        if name in program:
            buffer = ctx.buffer(array.tobytes())
            buffers.append(buffer)
            attributes.append((buffer, fmt, name))
    vao = ctx.vertex_array(program, attributes)
    palette = ctx.buffer(bytes(glm.transpose(skin_matrix)) + bytes(glm.mat4(1)) * 255)
    program["BonePalette"].binding = 0
    palette.bind_to_uniform_block(0)
    for name in ("matProjection", "matModel", "matView"):
        program[name].write(bytes(glm.mat4(1)))
    program["mv_normalMatrix"].write(bytes(glm.transpose(rotation)))
    program["bSkinned"].value = skinned
    program["bLightEnabled"].value = lighting
    program["bNormalMap"].value = False
    program["prop.alpha"].value = 1
    program["alphaThreshold"].value = -1
    for name in ("frontal", "directional0", "directional1", "directional2"):
        program[f"{name}.direction"].value = (0, 0, 1)
        program[f"{name}.diffuse"].value = (0.25, 0.25, 0.25) if name == "frontal" else (0, 0, 0)
    program["ambient"].value = 0
    program["prop.shininess"].value = 1
    targets = [ctx.texture((16, 16), 4, dtype="f4") for _ in range(2)]
    fbo = ctx.framebuffer(targets)
    fbo.use()
    ctx.viewport = (0, 0, 16, 16)
    ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
    vao.render(moderngl.TRIANGLES, vertices=3)
    pixels = np.frombuffer(fbo.read(attachment=1, components=4, dtype="f4"), dtype="f4").reshape(16, 16, 4)
    colors = np.frombuffer(fbo.read(attachment=0, components=4, dtype="f4"), dtype="f4").reshape(16, 16, 4)
    for row, col in ((4, 4), (8, 8), (10, 5)):
        x, y = (col + 0.5) / 16, (row + 0.5) / 16
        barycentric = np.array([1 - x / 2 - y / 2, x / 2, y / 2])
        normal = rotation * glm.mat3(skin_matrix) * glm.vec3(*(barycentric @ smooth))
        expected = np.array(glm.normalize(normal)) * 0.5 + 0.5
        np.testing.assert_allclose(pixels[row, col, :3], expected, atol=1e-5)
        assert pixels[row, col, 3] == pytest.approx(1)
        if lighting:
            def tonemap(value):
                return ((value * (0.15 * value + 0.05) + 0.004) /
                        (value * (0.15 * value + 0.5) + 0.06)) - 0.02 / 0.3
            encoded_normal = pixels[row, col, :3] * 2 - 1
            expected_color = tonemap(max(encoded_normal[2], 0) * 0.25) / tonemap(1)
            np.testing.assert_allclose(colors[row, col, :3], expected_color, atol=1e-5)
    fbo.release()
    for target in targets:
        target.release()
    palette.release()
    vao.release()
    for buffer in buffers:
        buffer.release()
    program.release()
