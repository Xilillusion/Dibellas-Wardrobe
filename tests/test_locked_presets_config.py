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
