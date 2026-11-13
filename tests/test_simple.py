import json

import numpy as np
import pytest

from src.shader_presets import load_postprocess_profile
from src.config import AppConfig
from src.viewport import OutfitViewportWidget
from test_shader_presets import (  # noqa: F401
    SHADERS,
    _material_scene,
    shader_renderer_factory,
)


@pytest.fixture
def simple_renderer(shader_renderer_factory):
    def render(color, depth, normals=None, key_light=None):
        return shader_renderer_factory("simple", color, depth, normals, key_light=key_light)
    return render


def test_simple_is_discovered_and_matches_realistic_postprocessing():
    simple = SHADERS / "simple.frag"
    realistic = SHADERS / "realistic.frag"
    presets = OutfitViewportWidget._discover_shader_presets()
    assert ("simple", "Simple") in presets
    assert not any(name == "simp" for name, _ in presets)
    simple_data = json.loads(simple.with_suffix(".json").read_text(encoding="utf-8"))
    realistic_data = json.loads(realistic.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(simple_data) == {"postprocess"}
    assert simple_data == realistic_data
    profile = load_postprocess_profile(simple)
    assert profile == load_postprocess_profile(realistic)
    assert profile.enable_gtao and profile.enable_ssss and not profile.ssss_after_preset


def test_simp_selection_migrates_to_simple(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"shader_preset": "simp"}), encoding="utf-8")
    config = AppConfig.load(path, supported_languages=["en"])
    assert config.shader_preset == "simple"
    config.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["shader_preset"] == "simple"


def test_simple_preserves_background_and_alpha(simple_renderer):
    color, depth, normals = _material_scene()
    depth[:4] = 1.0
    color[:, :, 3] = 0.7
    output = simple_renderer(color, depth, normals, key_light=(0, 0, 1))
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output[:4], color[:4], atol=1e-6)
    np.testing.assert_allclose(output[:, :, 3], 0.7, atol=1e-6)


@pytest.mark.parametrize("rgb", [
    (0.5, 0.3, 0.2), (0.1, 0.1, 0.1), (0.95, 0.95, 0.95), (1.0, 0.05, 0.0),
])
def test_simple_matches_vivid_grade_with_lighting_disabled(
    simple_renderer, shader_renderer_factory, rgb,
):
    color, depth, normals = _material_scene(value=0.4)
    color[:, :, :3] = rgb
    invalid = normals.copy()
    invalid[:, :, 3] = 0.0
    output = simple_renderer(color, depth, invalid, key_light=(0, 0, 0))
    vivid = shader_renderer_factory("vivid", color, depth, invalid, key_light=(0, 0, 0))
    np.testing.assert_allclose(output[24, 24], vivid[24, 24], atol=1e-5)
    source = color[24, 24, :3].astype("f8")
    luma = np.dot(source, (0.2126, 0.7152, 0.0722))
    graded = np.maximum(luma + (source - luma) * 1.15, 0) * 1.12
    lum = np.dot(graded, (0.2126, 0.7152, 0.0722))
    expected = np.clip(graded * (1.0 + 0.45 * (lum - 0.5) * (1.0 - lum)), 0, 1)
    np.testing.assert_allclose(output[24, 24, :3], expected, atol=1e-5)


def test_simple_key_light_gently_warms_lit_surfaces(simple_renderer):
    color, depth, normals = _material_scene(value=0.4)
    lit = simple_renderer(color, depth, normals, key_light=(0, 0, 1))[24, 24]
    shadow = simple_renderer(color, depth, normals, key_light=(0, 0, -1))[24, 24]
    assert lit[0] > lit[1] > lit[2]
    assert shadow[2] > shadow[1] > shadow[0]
    assert lit[0] > shadow[0]
    assert lit[0] < shadow[0] * 1.08
    for actual, tint in ((lit, (1.02, 1.01, 1.0)), (shadow, (0.98, 0.99, 1.0))):
        source = np.full(3, 0.4) * tint
        luma = np.dot(source, (0.2126, 0.7152, 0.0722))
        graded = np.maximum(luma + (source - luma) * 1.15, 0) * 1.12
        lum = np.dot(graded, (0.2126, 0.7152, 0.0722))
        expected = np.clip(graded * (1.0 + 0.45 * (lum - 0.5) * (1.0 - lum)), 0, 1)
        np.testing.assert_allclose(actual[:3], expected, atol=1e-5)


def test_simple_invalid_normals_do_not_create_lighting_artifacts(simple_renderer):
    color, depth, normals = _material_scene(value=0.4)
    normals[:, :, 3] = 0.0
    lit = simple_renderer(color, depth, normals, key_light=(0, 0, 1))
    shadow = simple_renderer(color, depth, normals, key_light=(0, 0, -1))
    np.testing.assert_allclose(lit, shadow, atol=1e-6)
    assert np.isfinite(lit).all()
