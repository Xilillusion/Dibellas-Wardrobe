import json

import numpy as np
import pytest

from src.shader_presets import load_postprocess_profile
from src.viewport import OutfitViewportWidget
from test_shader_presets import (  # noqa: F401
    SHADERS,
    _material_maps,
    _material_scene,
    _unit,
    shader_renderer_factory,
)


@pytest.fixture
def copilot_renderer(shader_renderer_factory):
    def render(color, depth, normals=None, materials=None, key_light=None):
        return shader_renderer_factory(
            "copilot", color, depth, normals, materials, key_light, outputs=2,
        )
    return render


def pixel(renderer, rgb=(0.4, 0.4, 0.4), normal=(0, 0, 1), maps=None, light=(0, 0, 1)):
    color, depth, normals = _material_scene(normal=normal)
    color[:, :, :3] = rgb
    diffuse, specular = renderer(color, depth, normals, maps, light)
    return diffuse[24, 24], specular[24, 24]


def test_copilot_is_discovered_with_postprocess_only():
    shader = SHADERS / "copilot.frag"
    assert ("copilot", "Copilot") in OutfitViewportWidget._discover_shader_presets()
    assert set(json.loads(shader.with_suffix(".json").read_text(encoding="utf-8"))) == {
        "postprocess"
    }
    profile = load_postprocess_profile(shader)
    assert profile.enable_gtao and profile.gtao_intensity == pytest.approx(0.5)
    assert profile.enable_ssss and profile.ssss_after_preset
    assert profile.sss_strength == pytest.approx(0.22)


def test_copilot_background_alpha_and_skin_mask(copilot_renderer):
    color, depth, normals = _material_scene()
    color[:, :, 3] = 0.7
    depth[:4] = 1.0
    skin = _material_maps(subsurface=(0.5, 0.3, 0.2, 1.0))
    diffuse, specular = copilot_renderer(color, depth, normals, skin)
    assert np.isfinite(diffuse).all() and np.isfinite(specular).all()
    np.testing.assert_allclose(diffuse[:4], color[:4], atol=1e-6)
    np.testing.assert_allclose(specular[:4], 0.0, atol=1e-6)
    np.testing.assert_allclose(diffuse[:, :, 3], 0.7, atol=1e-6)
    assert specular[24, 24, 3] == pytest.approx(1.0)
    assert pixel(copilot_renderer)[1][3] == pytest.approx(0.0)


def test_copilot_luminous_warm_light_and_colored_shadows(copilot_renderer):
    lit, _ = pixel(copilot_renderer, light=(0, 0, 1))
    # The source already contains viewport lighting: a shadow receives 65% of
    # the front-lit input before the preset estimates its albedo.
    shadow, _ = pixel(copilot_renderer, rgb=(0.26, 0.26, 0.26), light=(0, 0, -1))
    assert lit[0] > lit[1] > lit[2]
    assert shadow[2] > shadow[1] > shadow[0] > 0.25
    assert np.mean(lit[:3]) > np.mean(shadow[:3]) * 1.1
    skin = _material_maps(subsurface=(0.5, 0.3, 0.2, 1.0))
    rosy, _ = pixel(copilot_renderer, rgb=(0.26, 0.26, 0.26),
                    maps=skin, light=(0, 0, -1))
    assert rosy[0] > shadow[0] and rosy[2] < shadow[2]


def test_copilot_material_highlights_and_matte_fallback(copilot_renderer):
    bright = _material_maps(env_mask=(1.0, 0.8, 0.0, 1.0))
    matte = _material_maps(env_mask=(0.0, 0.8, 0.0, 1.0))
    normal_mask = _material_maps(normal=(0.5, 0.5, 1.0, 1.0))
    _, gloss = pixel(copilot_renderer, maps=bright)
    _, no_gloss = pixel(copilot_renderer, maps=matte)
    _, normal_gloss = pixel(copilot_renderer, maps=normal_mask)
    _, off_axis = pixel(copilot_renderer, maps=bright, light=_unit(0.9, 0, 0.43589))
    assert gloss[0] > 0.25
    assert no_gloss[0] < 0.005
    assert normal_gloss[0] > 0.02
    assert off_axis[0] < gloss[0] * 0.1


def test_copilot_handles_missing_normals_and_opposite_light(copilot_renderer):
    color, depth, normals = _material_scene()
    normals[:, :, 3] = 0.0
    diffuse, specular = copilot_renderer(color, depth, normals, key_light=(0, 0, -1))
    assert np.isfinite(diffuse).all() and np.isfinite(specular).all()
    assert diffuse[24, 24, 0] > 0.0
    np.testing.assert_allclose(specular[24, 24, :3], 0.0, atol=1e-6)


def test_copilot_tints_metal_and_preserves_saturated_colors(copilot_renderer):
    metal = _material_maps(env_mask=(0.9, 0.8, 0.9, 1.0))
    dielectric = _material_maps(env_mask=(0.9, 0.8, 0.0, 1.0))
    rgb = (0.6, 0.3, 0.1)
    diffuse, shine = pixel(copilot_renderer, rgb=rgb, maps=metal)
    _, white_shine = pixel(copilot_renderer, rgb=rgb, maps=dielectric)
    assert diffuse[0] > diffuse[1] > diffuse[2] > 0
    assert shine[0] > shine[1] > shine[2]
    assert white_shine[2] > shine[2] * 2


def test_copilot_lit_rim_and_inactive_material_flags(copilot_renderer):
    contour = _unit(0.98, 0, 0.2)
    _, rim = pixel(copilot_renderer, normal=contour, light=(1, 0, 0))
    _, unlit = pixel(copilot_renderer, normal=contour, light=(-1, 0, 0))
    assert rim[2] > rim[1] > rim[0] > 0.02
    np.testing.assert_allclose(unlit[:3], 0.0, atol=1e-6)

    maps = _material_maps(env_mask=(1.0, 0.8, 0.0, 1.0),
                          subsurface=(0.5, 0.3, 0.2, 1.0))
    maps["u_material_flags"][:] = 0.0
    reference = pixel(copilot_renderer)
    ignored = pixel(copilot_renderer, maps=maps)
    np.testing.assert_allclose(ignored, reference, atol=1e-6)
