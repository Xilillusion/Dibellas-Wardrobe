"""
Configuration manager for Dibella's Wardrobe.

Handles loading and saving of application configuration from/to a JSON file,
including language preference, default mod root folder, lighting, and UI settings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

log = logging.getLogger("dibellas_wardrobe.config")

# Default location for config.json (repository/app root)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
DEFAULT_LOCALES_PATH = DEFAULT_CONFIG_PATH.with_name("locales.json")


def load_locales(path: Optional[Path | str] = None) -> Dict[str, Dict[str, str]]:
    locale_file = Path(path) if path else DEFAULT_LOCALES_PATH
    with locale_file.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Locale file must contain at least one language: {locale_file}")
    if any(
        not isinstance(locale, str)
        or not isinstance(translations, dict)
        or not isinstance(translations.get("language_name"), str)
        for locale, translations in data.items()
    ):
        raise ValueError(
            f"Each locale must be an object with a 'language_name' string: {locale_file}"
        )
    return data


@dataclass
class LightingConfig:
    light_azimuth_deg: float = 35.0
    light_elevation_deg: float = 28.0
    light_intensity: float = 0.85
    ambient_intensity: float = 0.45
    specular_multiplier: float = 1.0
    env_reflection_multiplier: float = 1.0


@dataclass
class CheckboxConfig:
    # Filter checkboxes
    meshes_and_caliente_only: bool = True
    hide_low_weight: bool = True
    hide_gnd_folder: bool = True
    show_mod_dir_filter: bool = True

    # Viewport display checkboxes
    wireframe_overlay: bool = False
    show_ground_grid: bool = True


@dataclass
class AppConfig:
    default_folder: str = ""
    language: str = ""
    lighting: LightingConfig = field(default_factory=LightingConfig)
    checkboxes: CheckboxConfig = field(default_factory=CheckboxConfig)
    locked_nifs: list[str] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        path: Optional[Path | str] = None,
        supported_languages: Optional[Iterable[str]] = None,
    ) -> AppConfig:
        config_file = Path(path) if path else DEFAULT_CONFIG_PATH
        languages = tuple(supported_languages or load_locales().keys())
        default_language = languages[0] if languages else ""
        if not config_file.is_file():
            return cls(language=default_language)

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        except Exception as exc:
            log.warning("Could not read config from %s: %s; using defaults", config_file, exc)
            return cls()

        cfg = cls(language=default_language)

        # default_folder
        if "default_folder" in data and isinstance(data["default_folder"], str):
            cfg.default_folder = data["default_folder"]

        language = data.get("language")
        if language in languages:
            cfg.language = language

        locked_nifs = data.get("locked_nifs", [])
        if isinstance(locked_nifs, list):
            cfg.locked_nifs = [path for path in locked_nifs if isinstance(path, str)]

        # lighting (supports both nested {"lighting": {...}} and flat keys)
        l_data = data.get("lighting", {})
        if not isinstance(l_data, dict):
            l_data = {}
        for key in asdict(cfg.lighting).keys():
            val = l_data.get(key, data.get(key))
            if val is not None:
                try:
                    setattr(cfg.lighting, key, float(val))
                except (ValueError, TypeError):
                    pass

        # checkboxes (supports both nested {"checkboxes": {...}} and flat keys)
        c_data = data.get("checkboxes", {})
        if not isinstance(c_data, dict):
            c_data = {}
        for key in asdict(cfg.checkboxes).keys():
            val = c_data.get(key, data.get(key))
            if val is not None:
                setattr(cfg.checkboxes, key, bool(val))

        return cfg

    def save(self, path: Optional[Path | str] = None) -> None:
        config_file = Path(path) if path else DEFAULT_CONFIG_PATH
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as exc:
            log.warning("Could not save config to %s: %s", config_file, exc)
