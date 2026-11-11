import json

import pytest

from src.config import AppConfig


@pytest.mark.parametrize(
    "data,expected",
    [
        ({"locked_nifs": ["old.nif", 3, "other.nif"]}, {"1": ["old.nif", "other.nif"]}),
        ({"locked_nifs": ["old.nif"], "locked_presets": {"2": ["new.nif"]}}, {"2": ["new.nif"]}),
        ({"locked_nifs": ["old.nif"], "locked_presets": {}}, {}),
        ({"locked_nifs": "invalid"}, {}),
        ({}, {}),
    ],
)
def test_locked_presets_migrate_and_save_without_legacy_list(tmp_path, data, expected):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    config = AppConfig.load(path, supported_languages=["en"])
    assert config.locked_presets == expected
    config.save(path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "locked_nifs" not in saved
    assert saved["locked_presets"] == expected
    assert AppConfig.load(path, supported_languages=["en"]).locked_presets == expected


def test_removed_smp_settings_are_not_saved(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "smp": {"enabled": False, "collision_xmls": ["Collision.xml"], "show_collisions": True}
    }), encoding="utf-8")
    config = AppConfig.load(path, supported_languages=["en"])
    assert config.smp.enabled is False
    config.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["smp"] == {"enabled": False}


def test_preset_diffuse_textures_validate_and_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    textures = {"0": "diffuse.dds", "1": "second.dds"}
    path.write_text(json.dumps({
        "locked_presets": {"1": ["outfit.nif"], "2": []},
        "locked_preset_diffuse_textures": {
            "1": {"outfit.nif": {**textures, "-1": "bad.dds", "2": 42}, "other.nif": textures},
            "2": {"outfit.nif": textures},
            "3": {"outfit.nif": textures},
        },
    }), encoding="utf-8")
    config = AppConfig.load(path, supported_languages=["en"])
    expected = {"1": {"outfit.nif": textures}}
    assert config.locked_preset_diffuse_textures == expected
    config.save(path)
    assert AppConfig.load(path, supported_languages=["en"]).locked_preset_diffuse_textures == expected
