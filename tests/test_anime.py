import numpy as np
import pytest

from src.shader_presets import load_postprocess_profile
from src.viewport import OutfitViewportWidget
from test_shader_presets import (  # noqa: F401
    SHADERS,
    _material_maps,
    _material_scene,
    shader_renderer_factory,
)

ANIME_UNIFORMS = {
    "u_shadow_step": 0.0, "u_rim_intensity": 0.35, "u_normal_strength": 0.3,
    "u_saturation": 1.12, "u_gloss_intensity": 0.6,
}


@pytest.fixture
def anime_renderer(shader_renderer_factory):
    def render(color, depth, normals=None, materials=None, key_light=None, **uniforms):
        return shader_renderer_factory(
            "anime", color, depth, normals, materials, key_light, outputs=2,
            uniforms={**ANIME_UNIFORMS, **uniforms},
        )
    return render


def pixel(renderer, rgb=(0.4, 0.4, 0.4), normal=(0, 0, 1), maps=None,
          light=(0, 0, 0), **uniforms):
    color, depth, normals = _material_scene(normal=normal)
    color[:, :, :3] = rgb
    diffuse, specular = renderer(color, depth, normals, maps, key_light=light, **uniforms)
    return diffuse[24, 24, :3], specular[24, 24, :3]


def test_anime_preset_metadata():
    shader = SHADERS / "anime.frag"
    assert ("anime", "Anime") in OutfitViewportWidget._discover_shader_presets()
    profile = load_postprocess_profile(shader)
    assert profile.enable_gtao and profile.gtao_radius == 4.0
    assert profile.gtao_intensity == 0.55
    assert not profile.enable_ssss and profile.ssss_after_preset


def test_anime_background_and_alpha_are_preserved(anime_renderer):
    color, depth, normals = _material_scene()
    color[:, :, 3] = 0.7
    depth[:4] = 1.0
    diffuse, specular = anime_renderer(color, depth, normals)
    assert np.isfinite(diffuse).all() and np.isfinite(specular).all()
    np.testing.assert_allclose(diffuse[:4], color[:4], atol=1e-6)
    np.testing.assert_allclose(specular[:4], 0.0, atol=1e-6)
    np.testing.assert_allclose(diffuse[:, :, 3], 0.7, atol=1e-6)
    np.testing.assert_allclose(specular[:, :, 3], 0.0, atol=1e-6)


def test_anime_cel_ramp_and_skin_tint(anime_renderer):
    light = (1, 0, 0)
    skin = _material_maps(subsurface=(0.5, 0.2, 0.1, 1.0))
    shadow, _ = pixel(anime_renderer, light=light, u_shadow_step=-0.3, u_saturation=1.0)
    lit, _ = pixel(anime_renderer, light=light, u_shadow_step=0.3, u_saturation=1.0)
    rose, _ = pixel(anime_renderer, maps=skin, light=light, u_shadow_step=-0.3,
                    u_saturation=1.0)
    albedo = 0.4 / 0.65
    np.testing.assert_allclose(shadow, albedo * np.array([0.74, 0.78, 0.90]), atol=1e-4)
    np.testing.assert_allclose(rose, albedo * np.array([0.84, 0.72, 0.78]), atol=1e-4)
    np.testing.assert_allclose(lit, albedo * (np.array([1.04, 1.03, 1.05]) + 0.025), atol=1e-4)


def test_anime_saturation_adds_chroma_without_blowing_out_white(anime_renderer):
    rgb = (0.55, 0.3, 0.2)
    neutral, _ = pixel(anime_renderer, rgb=rgb, u_saturation=1.0)
    vivid, _ = pixel(anime_renderer, rgb=rgb, u_saturation=1.4)
    assert vivid[0] - vivid[2] > (neutral[0] - neutral[2]) * 1.25
    white, _ = pixel(anime_renderer, rgb=(0.75, 0.75, 0.75), u_saturation=2.0)
    reference, _ = pixel(anime_renderer, rgb=(0.75, 0.75, 0.75), u_saturation=1.0)
    np.testing.assert_allclose(white, reference, atol=1e-5)
    saturated, _ = pixel(anime_renderer, rgb=(1.0, 0.1, 0.0), u_saturation=3.0)
    assert np.isfinite(saturated).all() and np.all(saturated >= 0.0) and np.all(saturated <= 1.0)


def test_anime_highlights_are_tight_and_masked(anime_renderer):
    mask = _material_maps(env_mask=(1.0, 0.0, 0.0, 1.0))
    matte = _material_maps(env_mask=(0.0, 0.0, 0.0, 1.0))
    _, peak = pixel(anime_renderer, maps=mask, light=(0, 0, 1), u_rim_intensity=0)
    _, off_axis = pixel(anime_renderer, normal=(0.5, 0, 0.8660254), maps=mask,
                        light=(0, 0, 1), u_normal_strength=1.0, u_rim_intensity=0)
    _, no_gloss = pixel(anime_renderer, maps=matte, light=(0, 0, 1), u_rim_intensity=0)
    _, disabled = pixel(anime_renderer, maps=mask, light=(0, 0, 1), u_rim_intensity=0,
                        u_gloss_intensity=0)
    assert peak[0] > 0.25
    np.testing.assert_allclose(off_axis, 0.0, atol=1e-5)
    np.testing.assert_allclose(no_gloss, 0.0, atol=1e-5)
    np.testing.assert_allclose(disabled, 0.0, atol=1e-5)


def test_anime_normal_softening_and_rim(anime_renderer):
    tilted = (0.98, 0.0, 0.1989975)
    flat, _ = pixel(anime_renderer, u_normal_strength=0)
    softened, _ = pixel(anime_renderer, normal=tilted, u_normal_strength=0)
    np.testing.assert_allclose(flat, softened, atol=1e-5)
    _, no_rim = pixel(anime_renderer, normal=tilted, light=(1, 0, 0),
                      u_normal_strength=1, u_rim_intensity=0, u_gloss_intensity=0)
    _, rim = pixel(anime_renderer, normal=tilted, light=(1, 0, 0),
                   u_normal_strength=1, u_rim_intensity=0.35, u_gloss_intensity=0)
    np.testing.assert_allclose(no_rim, 0.0, atol=1e-5)
    assert rim[2] > rim[1] > rim[0] > 0.2
    shape, _ = pixel(anime_renderer, normal=tilted, light=(1, 0, 0),
                     u_normal_strength=0, u_shadow_step=-0.15)
    detail, _ = pixel(anime_renderer, normal=tilted, light=(1, 0, 0),
                      u_normal_strength=0.3, u_shadow_step=-0.15)
    assert detail[0] > shape[0] + 0.08


def test_anime_grazing_contours_stay_colored_and_soft(anime_renderer):
    contour, specular = pixel(
        anime_renderer, rgb=(0.5, 0.25, 0.15), normal=(0.99875, 0, 0.04998),
        light=(1, 0, 0), u_normal_strength=1, u_rim_intensity=0, u_gloss_intensity=0,
    )
    assert contour[0] > contour[1] > contour[2] > 0.05
    assert contour[0] > 0.28
    np.testing.assert_allclose(specular, 0.0, atol=1e-5)
