import json

import numpy as np
import pytest

from src.shader_presets import load_postprocess_profile
from src.viewport import OutfitViewportWidget
from test_shader_presets import (  # noqa: F401  (fixtures are used by name)
    SHADERS,
    _FRONT_LIGHT,
    _glossy_skin,
    _luminance,
    _material_maps,
    _material_scene,
    _unit,
    standard_renderer,
    shader_renderer_factory,
)

DARK_FANTASY_UNIFORMS = json.loads(
    (SHADERS / "darkfantasy.json").read_text(encoding="utf-8")
)["uniforms"]
RIM_COLOR = np.array([0.55, 0.72, 0.95])


@pytest.fixture
def dark_renderer(shader_renderer_factory):
    """Returns the (diffuse, highlights/reflections/rim + SSSS mask) outputs of Dark Fantasy."""
    def render(color, depth, normals=None, materials=None, key_light=None, **uniforms):
        return shader_renderer_factory(
            "darkfantasy", color, depth, normals, materials, key_light, outputs=2,
            uniforms={**DARK_FANTASY_UNIFORMS, **uniforms},
        )
    return render


def _pixel(renderer, normal=(0.0, 0.0, 1.0), maps=None, key_light=_FRONT_LIGHT, rgb=None,
           **uniforms):
    color, depth, normals = _material_scene(value=0.5, normal=normal)
    if rgb is not None:
        color[:, :, :3] = rgb
    diffuse, specular = renderer(color, depth, normals, maps, key_light=key_light, **uniforms)
    return diffuse[24, 24], specular[24, 24]


def _metal(red, gloss=0.6):
    return _material_maps(env_mask=(red, gloss, red, 1.0))


def test_dark_fantasy_is_discovered_with_uniforms_and_postprocessing():
    assert ("darkfantasy", "Dark Fantasy") in OutfitViewportWidget._discover_shader_presets()
    assert DARK_FANTASY_UNIFORMS == {
        "u_contrast_power": 1.35, "u_rim_intensity": 1.0, "u_metal_boost": 1.0,
    }
    profile = load_postprocess_profile(SHADERS / "darkfantasy.frag")
    assert profile.enable_gtao and profile.gtao_intensity == pytest.approx(1.35)
    assert profile.enable_ssss and profile.ssss_after_preset
    assert profile.sss_strength == pytest.approx(0.75)
    # 1.0 px is the smallest SSSS radius the post-process accepts.
    assert profile.sss_radius == pytest.approx(1.0)
    assert profile.sss_falloff_color == pytest.approx((0.75, 0.32, 0.25))


def test_dark_fantasy_preserves_background_and_splits_outputs(dark_renderer):
    color, depth, normals = _material_scene(value=0.5)
    depth[:4] = 1.0
    color[:, :, 3] = 0.7
    diffuse, specular = dark_renderer(color, depth, normals, _glossy_skin(), key_light=_FRONT_LIGHT)
    assert np.isfinite(diffuse).all() and np.isfinite(specular).all()
    np.testing.assert_allclose(diffuse[:4], color[:4], atol=1e-6)
    np.testing.assert_allclose(specular[:4], 0.0, atol=1e-6)
    np.testing.assert_allclose(diffuse[:, :, 3], 0.7, atol=1e-6)
    assert specular[24, 24, 3] == pytest.approx(1.0)
    assert _pixel(dark_renderer)[1][3] == pytest.approx(0.0)


def test_dark_fantasy_shadows_are_deep_and_cool_while_the_key_is_warm(
    dark_renderer, standard_renderer,
):
    lit, _ = _pixel(dark_renderer)
    shadow, _ = _pixel(dark_renderer, key_light=(0.0, -1.0, 0.0))
    color, depth, normals = _material_scene(value=0.5)
    base_lit = standard_renderer(color, depth, normals, None, key_light=_FRONT_LIGHT)[24, 24]
    base_shadow = standard_renderer(color, depth, normals, None, key_light=(0.0, -1.0, 0.0))[24, 24]
    assert _luminance(shadow) < _luminance(base_shadow) * 0.3
    assert _luminance(lit) / _luminance(shadow) > _luminance(base_lit) / _luminance(base_shadow) * 3
    assert shadow[2] > shadow[0]
    assert lit[0] > lit[2]


def test_dark_fantasy_contrast_power_drops_half_lit_form_into_shadow(dark_renderer):
    side = _unit(0.0, 0.8, 0.6)
    soft, _ = _pixel(dark_renderer, key_light=side, u_contrast_power=1.0)
    default, _ = _pixel(dark_renderer, key_light=side)
    hard, _ = _pixel(dark_renderer, key_light=side, u_contrast_power=2.5)
    assert _luminance(soft) > _luminance(default) > _luminance(hard)
    # Full-on light is unaffected by the exponent.
    np.testing.assert_allclose(
        _pixel(dark_renderer, u_contrast_power=1.0)[0], _pixel(dark_renderer, u_contrast_power=2.5)[0],
        atol=1e-5,
    )


def test_dark_fantasy_desaturates_albedo(dark_renderer, standard_renderer):
    rgb = (0.6, 0.3, 0.2)
    dark, _ = _pixel(dark_renderer, rgb=rgb)
    color, depth, normals = _material_scene(value=0.5)
    color[:, :, :3] = rgb
    standard = standard_renderer(color, depth, normals, None, key_light=_FRONT_LIGHT)[24, 24]
    assert dark[0] / dark[2] < standard[0] / standard[2] * 0.95


def test_dark_fantasy_metal_reflects_a_tinted_boosted_studio(dark_renderer):
    # A light in the surface plane isolates the studio reflection from the highlight.
    side = (0.0, -1.0, 0.0)
    rgb = (0.8, 0.5, 0.2)
    _, metal = _pixel(dark_renderer, maps=_metal(0.6), key_light=side, rgb=rgb)
    _, boosted = _pixel(dark_renderer, maps=_metal(0.6), key_light=side, rgb=rgb, u_metal_boost=2.0)
    _, leather = _pixel(dark_renderer, maps=_metal(0.29), key_light=side, rgb=rgb)
    assert metal[0] > metal[1] > metal[2]
    assert metal[0] > leather[0] * 3
    assert boosted[0] > metal[0] * 1.3
    # Below the metal threshold, the reflection is untinted.
    assert leather[0] == pytest.approx(leather[2], rel=0.05)
    # Metal boost only affects metal.
    _, leather_boosted = _pixel(dark_renderer, maps=_metal(0.29), key_light=side, rgb=rgb,
                                u_metal_boost=2.0)
    np.testing.assert_allclose(leather_boosted, leather, atol=1e-6)


def _expected_rim(normal, light=(0.0, 0.0, 1.0), boost=1.0, intensity=1.0):
    normal = np.asarray(normal, dtype="f8")
    boosted = np.array([0.0, 0.0, 1.0]) + (normal - np.array([0.0, 0.0, 1.0])) * boost
    boosted /= np.linalg.norm(boosted)
    n_dot_l = boosted @ np.asarray(light)
    t = np.clip((n_dot_l + 0.3) / 0.8, 0.0, 1.0)
    # The view ray through pixel (24, 24): half a pixel off-centre, 42 degree fov.
    offset = (24.5 / 48.0 * 2.0 - 1.0) * np.tan(np.radians(21.0))
    view = np.array([-offset, -offset, 1.0])
    view /= np.linalg.norm(view)
    rim = (1.0 - max(boosted @ view, 0.0)) ** 3 * t * t * (3.0 - 2.0 * t)
    return RIM_COLOR * intensity * rim


def test_dark_fantasy_moonlight_rim_uses_boosted_grit_normals(dark_renderer):
    tilted = _unit(np.sin(np.radians(60.0)), 0.0, np.cos(np.radians(60.0)))
    light = _unit(0.6, 0.0, 0.8)
    _, cloth = _pixel(dark_renderer, normal=tilted, key_light=light)
    np.testing.assert_allclose(
        cloth[:3], _expected_rim(tilted, light, boost=1.25), rtol=2e-3, atol=1e-5
    )
    assert cloth[2] > cloth[1] > cloth[0] > 0.05
    # Skin keeps its normal map as authored.
    _, skin = _pixel(dark_renderer, normal=tilted, maps=_glossy_skin(), key_light=light)
    np.testing.assert_allclose(
        skin[:3], _expected_rim(tilted, light, boost=1.0), rtol=2e-3, atol=1e-5
    )
    # Off on the side facing away from the key, and scaled by u_rim_intensity.
    _, away = _pixel(dark_renderer, normal=tilted, key_light=_unit(-0.95, 0.0, 0.3))
    assert away[2] < cloth[2] * 0.05
    _, doubled = _pixel(dark_renderer, normal=tilted, key_light=light, u_rim_intensity=2.0)
    np.testing.assert_allclose(doubled[:3], cloth[:3] * 2.0, rtol=1e-4)
    _, facing = _pixel(dark_renderer)
    np.testing.assert_allclose(facing[:3], 0.0, atol=1e-6)
