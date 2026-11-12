import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import moderngl
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtWidgets import QApplication

from src.config import AppConfig
from src.smaa import DEFAULT_SMAA_QUALITY, SMAA_QUALITIES, SmaaPostProcess, smaa_shader_sources
from src.viewport import OutfitViewportWidget, ViewportRenderSettings


@pytest.fixture(scope="module")
def smaa_runner():
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
    smaa = SmaaPostProcess(ctx)

    def run(image, quality="high"):
        height, width = image.shape[:2]
        input_fbo = smaa.prepare(quality, width, height)
        rgba = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        smaa.input_color.write(rgba.tobytes())
        output = ctx.texture((width, height), 4, dtype="f1")
        output_fbo = ctx.framebuffer(color_attachments=[output])
        smaa.apply(output_fbo, quality, width, height)
        result = np.frombuffer(output_fbo.read(components=4), dtype=np.uint8)
        output_fbo.release()
        output.release()
        assert input_fbo is smaa.input_fbo
        return result.reshape((height, width, 4)).astype(np.float64) / 255.0

    yield run
    smaa.release()
    ctx.release()
    context.doneCurrent()
    app.processEvents()


def _sloped_edge(size=64, slope=1.0 / 8.0, base=20.0, vertical=False):
    """An aliased white-over-black edge and its exact per-pixel area coverage."""
    x = np.arange(size, dtype=np.float64)
    y = x[:, None]
    edge = base + (x[None, :] + 0.5) * slope
    coverage = np.clip(edge - y, 0.0, 1.0)
    aliased = (y + 0.5 < edge).astype(np.float64)
    if vertical:
        coverage, aliased = coverage.T, aliased.T
    image = np.empty((size, size, 4))
    image[..., :3] = aliased[..., None]
    image[..., 3] = 1.0
    return image, coverage


def test_shader_sources_select_quality_and_pass():
    vertex, fragment = smaa_shader_sources("ultra", 1)
    assert vertex.startswith("#version 330\n")
    assert "#define SMAA_PRESET_ULTRA" in vertex
    assert "#define SMAA_PASS 1" in fragment
    assert "#define SMAA_INCLUDE_PS 0" in vertex
    assert "#define SMAA_INCLUDE_VS 0" in fragment
    with pytest.raises(ValueError):
        smaa_shader_sources("extreme", 0)
    with pytest.raises(ValueError):
        smaa_shader_sources("high", 3)


@pytest.mark.parametrize("quality", SMAA_QUALITIES)
def test_every_quality_compiles_and_keeps_flat_images_unchanged(smaa_runner, quality):
    image = np.full((32, 32, 4), 0.4)
    image[..., 3] = 0.7
    result = smaa_runner(image, quality)
    np.testing.assert_allclose(result, np.round(image * 255.0) / 255.0, atol=1.0 / 255.0)


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("slope", [1.0 / 8.0, -1.0 / 8.0])
def test_smaa_approximates_true_edge_coverage(smaa_runner, vertical, slope):
    image, coverage = _sloped_edge(slope=slope, base=32.0 if slope < 0 else 20.0,
                                   vertical=vertical)
    result = smaa_runner(image)
    interior = (slice(8, -8), slice(8, -8))
    aliased_error = np.abs(image[..., 0] - coverage)[interior].mean()
    smaa_error = np.abs(result[..., 0] - coverage)[interior].mean()

    assert np.isfinite(result).all()
    # Correctly oriented lookup tables cut this error >10x; flipped ones barely help.
    assert smaa_error < aliased_error * 0.25
    np.testing.assert_allclose(result[..., 3], 1.0, atol=1.0 / 255.0)


def test_smaa_leaves_pixels_away_from_edges_untouched(smaa_runner):
    image, _coverage = _sloped_edge()
    result = smaa_runner(image)
    np.testing.assert_allclose(result[:12, :, :3], 1.0, atol=1.0 / 255.0)
    np.testing.assert_allclose(result[34:, :, :3], 0.0, atol=1.0 / 255.0)


def test_smaa_resizes_targets_and_clears_stale_edges(smaa_runner):
    image, _coverage = _sloped_edge()
    smaa_runner(image)
    flat = np.full((40, 24, 4), 0.25)
    flat[..., 3] = 1.0
    result = smaa_runner(flat)
    assert result.shape == (40, 24, 4)
    np.testing.assert_allclose(result[..., :3], 64.0 / 255.0, atol=1.0 / 255.0)

def test_smaa_config_round_trips_and_rejects_unknown_quality(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"effects": {
        "enable_smaa": False, "smaa_quality": "ultra",
        "enable_ssr": True, "ssr_max_steps": 64,
    }}), encoding="utf-8")
    config = AppConfig.load(path, supported_languages=["en"])
    assert (config.effects.enable_smaa, config.effects.smaa_quality) == (False, "ultra")

    config.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))["effects"]
    assert saved["smaa_quality"] == "ultra"
    assert not any(key.startswith("ssr") or key == "enable_ssr" for key in saved)

    path.write_text(json.dumps({"effects": {"smaa_quality": "extreme"}}), encoding="utf-8")
    effects = AppConfig.load(path, supported_languages=["en"]).effects
    assert (effects.enable_smaa, effects.smaa_quality) == (True, DEFAULT_SMAA_QUALITY)


def test_viewport_renders_without_smaa_when_setup_fails():
    viewport = SimpleNamespace(
        settings=ViewportRenderSettings(smaa_quality="ultra"),
        smaa=MagicMock(),
        _smaa_failed_qualities=set(),
    )
    viewport.smaa.prepare.side_effect = moderngl.Error("compile failed")

    assert OutfitViewportWidget._prepare_smaa(viewport, 64, 32) is None
    assert OutfitViewportWidget._prepare_smaa(viewport, 64, 32) is None
    viewport.smaa.prepare.assert_called_once_with("ultra", 64, 32)

    viewport.settings.smaa_quality = "high"
    viewport.smaa.prepare.side_effect = None
    assert OutfitViewportWidget._prepare_smaa(viewport, 64, 32) == "high"
    viewport.settings.enable_smaa = False
    assert OutfitViewportWidget._prepare_smaa(viewport, 64, 32) is None
