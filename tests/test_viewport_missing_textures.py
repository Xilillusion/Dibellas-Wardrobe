import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.nif_loader import AlphaPropertyData, ShaderMaterialData
from src.texture_resolver import ResolvedTexture, TextureResolver
from src.viewport import OutfitViewportWidget


@pytest.fixture
def viewport():
    app = QApplication.instance() or QApplication([])
    widget = OutfitViewportWidget()
    widget.ctx = MagicMock()
    widget.prog = MagicMock()
    widget.post_prog = MagicMock()
    widget.post_vao = MagicMock()
    widget.scene_fbo = MagicMock()
    widget._ensure_scene_target = MagicMock()
    widget._setup_global_shader_uniforms = MagicMock()
    widget._composite_postprocess = MagicMock()
    widget._render_shape_item = MagicMock()
    widget._current_pose_matrices = {}
    widget.settings.show_ground_grid = False
    yield widget
    widget.ctx = None
    widget.close()
    app.processEvents()


def shape(texture=None, *, alpha_test=False, alpha_blend=False):
    material = ShaderMaterialData()
    if texture is not None:
        material.resolved_textures["Diffuse"] = texture
    return SimpleNamespace(
        shape_data=SimpleNamespace(
            shader=material,
            alpha_prop=AlphaPropertyData(
                has_alpha_property=alpha_test or alpha_blend,
                alpha_test=alpha_test,
                alpha_blend=alpha_blend,
            ),
        )
    )


def valid_texture():
    return ResolvedTexture(
        slot="Diffuse",
        raw_nif_path="textures/example.dds",
        resolved_path=Path("example.dds"),
        is_fallback=False,
        width=1,
        height=1,
        rgba_data=np.full((1, 1, 4), 255, dtype=np.uint8),
    )


@pytest.mark.parametrize("show_textures", [True, False])
def test_missing_diffuse_is_absent_from_every_render_pass(viewport, show_textures):
    fallback = TextureResolver().load_or_fallback("Diffuse", "")
    unbound = shape()
    missing = shape(fallback, alpha_test=True, alpha_blend=True)
    opaque = shape(valid_texture())
    blended = shape(valid_texture(), alpha_blend=True)
    viewport._gpu_nifs = {"outfit": [unbound, missing, opaque, blended]}
    viewport.settings.show_textures = show_textures
    viewport.settings.wireframe_overlay = True

    viewport.paintGL()

    calls = viewport._render_shape_item.call_args_list
    assert [(call.args[0], call.kwargs["wireframe"]) for call in calls] == [
        (opaque, False), (blended, False), (opaque, True), (blended, True)
    ]


def test_assigning_and_resetting_diffuse_updates_visibility(viewport):
    fallback = TextureResolver().load_or_fallback("Diffuse", "textures/missing.dds")
    item = shape(fallback)
    item.shape_data.shader.default_resolved_textures = {"Diffuse": fallback}
    viewport._gpu_nifs = {"outfit": [item]}

    viewport.paintGL()
    viewport._render_shape_item.assert_not_called()

    context = viewport.ctx
    viewport.ctx = None
    viewport.update_nif_shape_texture("outfit", 0, "Diffuse", valid_texture())
    viewport.ctx = context
    viewport.paintGL()
    viewport._render_shape_item.assert_called_once_with(item, wireframe=False)

    viewport._render_shape_item.reset_mock()
    viewport.ctx = None
    viewport.reset_nif_textures("outfit")
    viewport.ctx = context
    viewport.paintGL()
    viewport._render_shape_item.assert_not_called()


def test_missing_and_unreadable_diffuse_use_transparent_fallback(tmp_path, caplog):
    resolver = TextureResolver([tmp_path])
    missing = resolver.load_or_fallback("Diffuse", "")
    assert missing.is_fallback
    assert np.all(missing.rgba_data[:, :, 3] == 0)

    texture = tmp_path / "textures" / "broken.dds"
    texture.parent.mkdir()
    texture.write_bytes(b"not a dds")
    unreadable = resolver.load_or_fallback("Diffuse", "textures/broken.dds")
    assert unreadable.is_fallback
    assert np.all(unreadable.rgba_data[:, :, 3] == 0)
    assert "Fallback used for Diffuse" in caplog.text
