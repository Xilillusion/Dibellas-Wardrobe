import json
import os
import struct
from unittest.mock import MagicMock

import glm
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.shader_presets import (
    NO_POSTPROCESS,
    PostProcessProfile,
    load_postprocess_profile,
    load_preset_label,
    load_preset_uniforms,
)
from src.viewport import _SHADER_DIR, OutfitViewportWidget


def write_metadata(tmp_path, payload):
    (tmp_path / "look.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path / "look.frag"


def test_preset_without_metadata_disables_postprocess(tmp_path):
    assert load_postprocess_profile(tmp_path / "look.frag") == NO_POSTPROCESS
    assert not NO_POSTPROCESS.enable_gtao and not NO_POSTPROCESS.enable_ssss


def test_profile_reads_and_clamps_values(tmp_path):
    fragment = write_metadata(tmp_path, {"postprocess": {
        "enable_gtao": True, "gtao_radius": 500, "gtao_intensity": 0.5,
        "enable_ssss": True, "sss_radius": 3, "sss_strength": -1,
        "sss_falloff_color": [2, 0.5, -1],
    }})
    profile = load_postprocess_profile(fragment)
    assert profile == PostProcessProfile(
        enable_gtao=True, gtao_radius=40.0, gtao_intensity=0.5,
        enable_ssss=True, sss_radius=3.0, sss_strength=0.0,
        sss_falloff_color=(1.0, 0.5, 0.0),
    )


@pytest.mark.parametrize("payload", [
    {"postprocess": {"enable_gtao": "yes"}},
    {"postprocess": {"ssss_after_preset": 1}},
    {"postprocess": {"gtao_radius": "far"}},
    {"postprocess": {"gtao_radius": float("nan")}},
    {"postprocess": {"sss_falloff_color": [1, 0]}},
    {"postprocess": {"ssao": True}},
    {"postprocess": []},
    [],
])
def test_invalid_profile_is_rejected(tmp_path, payload):
    fragment = write_metadata(tmp_path, payload)
    with pytest.raises(ValueError, match="look.json"):
        load_postprocess_profile(fragment)


def test_label_and_uniforms_are_optional(tmp_path):
    assert load_preset_label(tmp_path / "look.frag") is None
    assert load_preset_uniforms(tmp_path / "look.frag") == {}
    fragment = write_metadata(tmp_path, {"postprocess": {}})
    assert load_preset_label(fragment) is None
    assert load_preset_uniforms(fragment) == {}


def test_label_and_uniforms_are_read(tmp_path):
    fragment = write_metadata(tmp_path, {
        "label": "  Silk (Velvet / Sheen) ",
        "uniforms": {"u_sheen_intensity": 2, "u_tint": [1, 0.5, 0.25]},
    })
    assert load_preset_label(fragment) == "Silk (Velvet / Sheen)"
    assert load_preset_uniforms(fragment) == {
        "u_sheen_intensity": 2.0, "u_tint": (1.0, 0.5, 0.25),
    }
    # The post-process section is independent of them.
    assert load_postprocess_profile(fragment) == NO_POSTPROCESS


@pytest.mark.parametrize("label", ["", "   ", 3, "x" * 65])
def test_invalid_label_is_rejected(tmp_path, label):
    with pytest.raises(ValueError, match="look.json"):
        load_preset_label(write_metadata(tmp_path, {"label": label}))


@pytest.mark.parametrize("uniforms", [
    [], {"sheen": 1.0}, {"u_": 1.0}, {"u_a-b": 1.0}, {"u_key_light": [0, 0, 1]},
    {"u_x": "1"}, {"u_x": True}, {"u_x": float("inf")}, {"u_x": [1]},
    {"u_x": [1, 2, 3, 4, 5]}, {"u_x": [1, "2"]},
])
def test_invalid_uniforms_are_rejected(tmp_path, uniforms):
    with pytest.raises(ValueError, match="look.json"):
        load_preset_uniforms(write_metadata(tmp_path, {"uniforms": uniforms}))


def test_discovery_falls_back_to_file_name_for_bad_labels(tmp_path, caplog):
    (tmp_path / "soft_cloth.frag").write_text("", encoding="utf-8")
    (tmp_path / "soft_cloth.json").write_text(json.dumps({"label": 5}), encoding="utf-8")
    assert OutfitViewportWidget._discover_shader_presets(tmp_path) == (
        ("soft_cloth", "Soft Cloth"),
    )
    assert "soft_cloth" in caplog.text


def test_malformed_json_is_rejected(tmp_path):
    (tmp_path / "look.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="look.json"):
        load_postprocess_profile(tmp_path / "look.frag")


def test_builtin_presets_configure_postprocess():
    realistic = load_postprocess_profile(_SHADER_DIR / "realistic.frag")
    assert realistic.enable_gtao and realistic.enable_ssss
    assert 0.0 < realistic.gtao_intensity < 1.0
    assert 0.0 < realistic.sss_strength < 0.5
    anime = load_postprocess_profile(_SHADER_DIR / "anime.frag")
    assert anime.enable_gtao and anime.enable_ssss
    assert anime.gtao_intensity <= realistic.gtao_intensity
    assert anime.sss_strength <= realistic.sss_strength
    assert not realistic.ssss_after_preset and not anime.ssss_after_preset
    glossy = load_postprocess_profile(_SHADER_DIR / "glossy.frag")
    assert glossy.enable_gtao and glossy.enable_ssss and glossy.ssss_after_preset
    assert glossy.sss_strength > realistic.sss_strength


def test_profile_reads_ssss_after_preset(tmp_path):
    fragment = write_metadata(tmp_path, {"postprocess": {
        "enable_ssss": True, "ssss_after_preset": True,
    }})
    assert load_postprocess_profile(fragment) == PostProcessProfile(
        enable_ssss=True, ssss_after_preset=True
    )


def test_composite_postprocess_binds_profile_uniforms():
    program = MagicMock()
    uniforms = {}
    program.__getitem__.side_effect = lambda name: uniforms.setdefault(name, MagicMock())
    program.__contains__.side_effect = lambda name: True
    widget = MagicMock()
    widget._postprocess_variant.return_value = (program, MagicMock())
    profile = PostProcessProfile(
        enable_gtao=True, gtao_radius=6.0, gtao_intensity=0.4, gtao_bias=0.01,
        enable_ssss=True, sss_radius=3.0, sss_strength=0.2, sss_falloff_color=(1.0, 0.5, 0.4),
    )
    OutfitViewportWidget._composite_postprocess(
        widget, MagicMock(), glm.mat4(1.0), 8, 8, profile
    )
    widget._postprocess_variant.assert_called_once_with(profile)
    assert uniforms["u_ao_radius"].value == 6.0
    assert uniforms["u_ao_intensity"].value == 0.4
    assert uniforms["u_ao_bias"].value == 0.01
    assert uniforms["u_ssss_radius"].value == 3.0
    assert uniforms["u_ssss_strength"].value == 0.2
    assert uniforms["u_ssss_falloff_color"].value == (1.0, 0.5, 0.4)


def test_disabled_profile_uses_copy_program_without_compiling():
    widget = MagicMock()
    widget._failed_post_variants = set()
    variant = OutfitViewportWidget._postprocess_variant(widget, NO_POSTPROCESS)
    assert variant == (widget.post_prog, widget.post_vao)
    widget._compile_postprocess_variant.assert_not_called()


@pytest.fixture
def viewport():
    app = QApplication.instance() or QApplication([])
    widget = OutfitViewportWidget()
    widget.ctx = MagicMock()
    widget.prog = MagicMock()
    widget.post_prog = MagicMock()
    widget.post_vao = MagicMock()
    widget.scene_fbo = MagicMock()
    widget.scene_color = MagicMock()
    widget._ensure_scene_target = MagicMock()
    widget._ensure_selected_shader_preset = MagicMock()
    widget._setup_global_shader_uniforms = MagicMock()
    widget._composite_postprocess = MagicMock()
    widget._composite_shader_preset = MagicMock()
    widget._ensure_preset_target = MagicMock()
    widget._ensure_split_target = MagicMock()
    widget._render_shape_item = MagicMock()
    widget._current_pose_matrices = {}
    widget._gpu_nifs = {}
    widget.settings.show_ground_grid = False
    widget.settings.enable_smaa = False
    yield widget
    widget.ctx = None
    widget.close()
    app.processEvents()


def activate_preset(widget, profile):
    widget.preset_prog = MagicMock()
    widget.preset_prog.__contains__.side_effect = lambda name: False
    widget.preset_vao = MagicMock()
    widget.postprocess_profile = profile


def test_preset_without_effects_skips_postprocess_pass(viewport):
    activate_preset(viewport, PostProcessProfile())
    viewport.paintGL()
    viewport._composite_postprocess.assert_not_called()
    viewport._ensure_preset_target.assert_not_called()
    assert viewport._composite_shader_preset.call_args.args[1] is viewport.scene_color


@pytest.mark.parametrize("gtao, ssss", [(True, False), (False, True), (True, True)])
def test_preset_effects_run_postprocess_before_preset(viewport, gtao, ssss):
    profile = PostProcessProfile(enable_gtao=gtao, enable_ssss=ssss)
    activate_preset(viewport, profile)
    viewport.preset_fbo = MagicMock()
    viewport.preset_color = MagicMock()
    viewport.paintGL()
    viewport._ensure_preset_target.assert_called_once()
    args = viewport._composite_postprocess.call_args.args
    assert args[0] is viewport.preset_fbo and args[-1] is profile
    assert viewport._composite_shader_preset.call_args.args[1] is viewport.preset_color


@pytest.mark.parametrize("gtao", [True, False])
def test_split_preset_runs_gtao_before_and_ssss_after(viewport, gtao):
    profile = PostProcessProfile(enable_gtao=gtao, enable_ssss=True, ssss_after_preset=True)
    activate_preset(viewport, profile)
    viewport.preset_fbo = MagicMock()
    viewport.preset_color = MagicMock()
    viewport.split_fbo = MagicMock()
    viewport.paintGL()

    viewport._ensure_split_target.assert_called_once()
    viewport.split_fbo.clear.assert_called_once_with(0.0, 0.0, 0.0, 0.0)
    preset_args = viewport._composite_shader_preset.call_args.args
    assert preset_args[0] is viewport.split_fbo
    passes = [call.args for call in viewport._composite_postprocess.call_args_list]
    if gtao:
        before, after = passes
        assert before[0] is viewport.preset_fbo
        assert before[-1].enable_gtao
        assert not before[-1].enable_ssss and not before[-1].ssss_after_preset
        assert preset_args[1] is viewport.preset_color
    else:
        (after,) = passes
        viewport._ensure_preset_target.assert_not_called()
        assert preset_args[1] is viewport.scene_color
    assert after[0] is not viewport.split_fbo and after[0] is not viewport.preset_fbo
    assert after[-1] == PostProcessProfile(enable_ssss=True, ssss_after_preset=True)


def test_split_preset_without_ssss_only_adds_specular_after(viewport):
    profile = PostProcessProfile(enable_gtao=True, ssss_after_preset=True)
    activate_preset(viewport, profile)
    viewport.preset_fbo = MagicMock()
    viewport.preset_color = MagicMock()
    viewport.split_fbo = MagicMock()
    viewport.paintGL()
    assert viewport._composite_shader_preset.call_args.args[0] is viewport.split_fbo
    before, after = (call.args for call in viewport._composite_postprocess.call_args_list)
    assert before[-1] == PostProcessProfile(enable_gtao=True)
    assert after[-1] == PostProcessProfile(ssss_after_preset=True)


def test_split_composite_reads_preset_diffuse_and_specular():
    program = MagicMock()
    uniforms = {}
    program.__getitem__.side_effect = lambda name: uniforms.setdefault(name, MagicMock())
    program.__contains__.side_effect = lambda name: name == "u_scene_specular"
    widget = MagicMock()
    widget._postprocess_variant.return_value = (program, MagicMock())
    profile = PostProcessProfile(ssss_after_preset=True)
    OutfitViewportWidget._composite_postprocess(widget, MagicMock(), glm.mat4(1.0), 8, 8, profile)
    widget.split_diffuse.use.assert_called_once_with(location=0)
    widget.split_specular.use.assert_called_once_with(location=2)
    widget.scene_color.use.assert_not_called()
    assert uniforms["u_scene_specular"].value == 2


def test_split_variant_key_includes_after_preset():
    widget = MagicMock()
    widget._failed_post_variants = set()
    widget._post_variants = {}
    variant = (MagicMock(), MagicMock())
    widget._compile_postprocess_variant.return_value = variant
    profile = PostProcessProfile(ssss_after_preset=True)
    assert OutfitViewportWidget._postprocess_variant(widget, profile) == variant
    widget._compile_postprocess_variant.assert_called_once_with(False, False, True)


def test_no_preset_presents_scene_without_effects(viewport):
    viewport.paintGL()
    viewport._composite_shader_preset.assert_not_called()
    assert viewport._composite_postprocess.call_args.args[-1] is NO_POSTPROCESS


def test_failed_preset_resets_postprocess_profile(viewport):
    viewport.postprocess_profile = PostProcessProfile(enable_gtao=True)
    viewport._report_shader_preset_error("look", "broken")
    assert viewport.postprocess_profile is NO_POSTPROCESS


def test_selecting_preset_loads_its_profile_and_none_clears_it(viewport):
    del viewport._ensure_selected_shader_preset
    viewport.settings.shader_preset = "realistic"
    viewport._ensure_selected_shader_preset()
    assert viewport.postprocess_profile == load_postprocess_profile(
        _SHADER_DIR / "realistic.frag"
    )
    viewport.settings.shader_preset = "none"
    viewport._ensure_selected_shader_preset()
    assert viewport.postprocess_profile is NO_POSTPROCESS


def test_invalid_profile_keeps_preset_without_effects(viewport, monkeypatch, caplog):
    del viewport._ensure_selected_shader_preset

    def reject(_path):
        raise ValueError("realistic.json: bad")

    monkeypatch.setattr("src.viewport.load_postprocess_profile", reject)
    viewport.settings.shader_preset = "realistic"
    viewport._ensure_selected_shader_preset()
    assert viewport.preset_prog is not None
    assert viewport.postprocess_profile is NO_POSTPROCESS
    assert "realistic.json: bad" in caplog.text



def test_selecting_preset_loads_its_uniforms_and_none_clears_them(viewport):
    del viewport._ensure_selected_shader_preset
    viewport.settings.shader_preset = "glossy"
    viewport._ensure_selected_shader_preset()
    assert viewport.preset_uniforms == load_preset_uniforms(_SHADER_DIR / "glossy.frag")
    viewport.settings.shader_preset = "none"
    viewport._ensure_selected_shader_preset()
    assert viewport.preset_uniforms == {}


def test_invalid_uniforms_keep_preset_with_shader_defaults(viewport, monkeypatch, caplog):
    del viewport._ensure_selected_shader_preset

    def reject(_path):
         in caplog.text


def test_preset_receives_sidecar_uniforms_and_drops_mismatched_ones(caplog):
    program = MagicMock()
    uniforms = {}
    program.__getitem__.side_effect = lambda name: uniforms.setdefault(name, MagicMock())
    program.__contains__.side_effect = lambda name: name in ("u_sheen_intensity", "u_bad")
    bad = MagicMock()
    type(bad).value = property(lambda _self: 0.0, MagicMock(side_effect=struct.error("float")))
    uniforms["u_bad"] = bad
    widget = MagicMock(preset_prog=program, preset_uniforms={
        "u_sheen_intensity": 1.5, "u_bad": (1.0, 2.0), "u_unused": 3.0,
    })
    OutfitViewportWidget._composite_shader_preset(
        widget, MagicMock(), MagicMock(), glm.mat4(1.0), 8, 8
    )
    assert uniforms["u_sheen_intensity"].value == 1.5
    assert widget.preset_uniforms == {"u_sheen_intensity": 1.5, "u_unused": 3.0}
    assert "u_bad" in caplog.text


def test_preset_receives_view_space_key_light():
    program = MagicMock()
    uniforms = {}
    program.__getitem__.side_effect = lambda name: uniforms.setdefault(name, MagicMock())
    program.__contains__.side_effect = lambda name: name == "u_key_light"
    widget = MagicMock(preset_prog=program, _key_light_view=glm.vec3(0.0, 0.6, 0.8))
    OutfitViewportWidget._composite_shader_preset(
        widget, MagicMock(), MagicMock(), glm.mat4(1.0), 8, 8
    )
    assert uniforms["u_key_light"].value == pytest.approx((0.0, 0.6, 0.8))


def test_preset_receives_studio_cubemap_and_view_rotation():
    program = MagicMock()
    uniforms = {}
    program.__getitem__.side_effect = lambda name: uniforms.setdefault(name, MagicMock())
    program.__contains__.side_effect = lambda name: name in (
        "u_studio_cubemap", "u_view_to_world"
    )
    rotation = glm.mat3(0, 1, 0, -1, 0, 0, 0, 0, 1)
    widget = MagicMock(preset_prog=program, _view_to_world=rotation)
    OutfitViewportWidget._composite_shader_preset(
        widget, MagicMock(), MagicMock(), glm.mat4(1.0), 8, 8
    )
    widget.default_cubemap.use.assert_called_once_with(location=7)
    assert uniforms["u_studio_cubemap"].value == 7
    uniforms["u_view_to_world"].write.assert_called_once_with(
        bytes(glm.transpose(rotation))
    )


def test_view_to_world_turns_view_directions_into_world_space(viewport):
    del viewport._setup_global_shader_uniforms
    view = glm.lookAt(glm.vec3(0, -10, 0), glm.vec3(0), glm.vec3(0, 0, 1))
    viewport.prog = MagicMock()
    viewport.prog.__contains__.side_effect = lambda name: False
    viewport._setup_global_shader_uniforms(
        glm.mat4(1.0), view, glm.mat4(1.0), glm.mat4(1.0), glm.mat3(1.0)
    )
    # View-space +Z points back at this camera, which sits on world -Y.
    toward_camera = viewport._view_to_world * glm.vec3(0.0, 0.0, 1.0)
    assert tuple(toward_camera) == pytest.approx((0.0, -1.0, 0.0), abs=1e-5)


@pytest.mark.parametrize("lighting", [True, False])
def test_key_light_follows_view_and_lighting_toggle(viewport, lighting):
    viewport.prog = None
    viewport.settings.enable_lighting = lighting
    viewport.settings.light_azimuth_deg = 0.0
    viewport.settings.light_elevation_deg = 0.0
    del viewport._setup_global_shader_uniforms
    view = glm.lookAt(glm.vec3(0, -10, 0), glm.vec3(0), glm.vec3(0, 0, 1))
    viewport.prog = MagicMock()
    viewport.prog.__contains__.side_effect = lambda name: False
    viewport._setup_global_shader_uniforms(
        glm.mat4(1.0), view, glm.mat4(1.0), glm.mat4(1.0), glm.mat3(1.0)
    )
    light = viewport._key_light_view
    if lighting:
        # Azimuth 0 shines on the front (-Y), which faces this camera.
        assert tuple(light) == pytest.approx((0.0, 0.0, 1.0), abs=1e-5)
    else:
        assert tuple(light) == (0.0, 0.0, 0.0)
