"""
Configuration manager for Dibella's Wardrobe.

Handles loading and saving of application configuration from/to a JSON file,
including language preference, default mod root folder, lighting, and UI settings.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .smaa import DEFAULT_SMAA_QUALITY, SMAA_QUALITIES

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
    light_azimuth_deg: float = 110.0
    light_elevation_deg: float = 28.0
    light_intensity: float = 0.85
    ambient_intensity: float = 0.45
    specular_multiplier: float = 1.0
    env_reflection_multiplier: float = 1.0


@dataclass
class ViewportEffectsConfig:
    enable_smaa: bool = True
    smaa_quality: str = DEFAULT_SMAA_QUALITY


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


SOFTNESS_GROUPS = ("breasts", "butt", "belly", "thigh", "other")
SOFTNESS_MIN = 0.0
SOFTNESS_MAX = 10.0


@dataclass
class CbpcConfig:
    enabled: bool = True
    gravity: float = 1.0 / 3.0  # breast gravity factor (0 = Armor, 1 = More Gravity)
    softness: Dict[str, float] = field(default_factory=lambda: dict.fromkeys(SOFTNESS_GROUPS, 1.0))


@dataclass
class SmpConfig:
    enabled: bool = True


@dataclass
class AppConfig:
    default_folder: str = ""
    language: str = ""
    shader_preset: str = "none"
    lighting: LightingConfig = field(default_factory=LightingConfig)
    effects: ViewportEffectsConfig = field(default_factory=ViewportEffectsConfig)
    checkboxes: CheckboxConfig = field(default_factory=CheckboxConfig)
    cbpc: CbpcConfig = field(default_factory=CbpcConfig)
    smp: SmpConfig = field(default_factory=SmpConfig)
    locked_presets: Dict[str, list[str]] = field(default_factory=dict)
    locked_preset_diffuse_textures: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)
    locked_preset_names: Dict[str, str] = field(default_factory=dict)

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

        shader_preset = data.get("shader_preset")
        if isinstance(shader_preset, str):
            cfg.shader_preset = "realistic" if shader_preset == "preset_1" else shader_preset

        locked_presets = data.get("locked_presets")
        if isinstance(locked_presets, dict):
            cfg.locked_presets = {
                page: [path for path in paths if isinstance(path, str)]
                for page, paths in locked_presets.items()
                if isinstance(page, str)
                and page.isdigit()
                and int(page) > 0
                and isinstance(paths, list)
            }
        else:
            legacy_paths = data.get("locked_nifs")
            if isinstance(legacy_paths, list):
                paths = [path for path in legacy_paths if isinstance(path, str)]
                if paths:
                    cfg.locked_presets = {"1": paths}

        textures = data.get("locked_preset_diffuse_textures")
        if isinstance(textures, dict):
            for page, meshes in textures.items():
                if page not in cfg.locked_presets or not isinstance(meshes, dict):
                    continue
                valid_meshes: Dict[str, Dict[str, str]] = {}
                for mesh, shapes in meshes.items():
                    if mesh not in cfg.locked_presets[page] or not isinstance(shapes, dict):
                        continue
                    valid_shapes = {
                        index: path for index, path in shapes.items()
                        if isinstance(index, str) and index.isdecimal()
                        and isinstance(path, str) and path
                    }
                    if valid_shapes:
                        valid_meshes[mesh] = valid_shapes
                if valid_meshes:
                    cfg.locked_preset_diffuse_textures[page] = valid_meshes

        preset_names = data.get("locked_preset_names")
        if isinstance(preset_names, dict):
            cfg.locked_preset_names = {
                page: name.strip()
                for page, name in preset_names.items()
                if isinstance(page, str)
                and page.isdigit()
                and int(page) > 0
                and isinstance(name, str)
                and name.strip()
            }

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

        effects_data = data.get("effects", {})
        if isinstance(effects_data, dict):
            enable_smaa = effects_data.get("enable_smaa")
            if isinstance(enable_smaa, bool):
                cfg.effects.enable_smaa = enable_smaa

            smaa_quality = effects_data.get("smaa_quality")
            if isinstance(smaa_quality, str) and smaa_quality in SMAA_QUALITIES:
                cfg.effects.smaa_quality = smaa_quality

        # checkboxes (supports both nested {"checkboxes": {...}} and flat keys)
        c_data = data.get("checkboxes", {})
        if not isinstance(c_data, dict):
            c_data = {}
        for key in asdict(cfg.checkboxes).keys():
            val = c_data.get(key, data.get(key))
            if val is not None:
                setattr(cfg.checkboxes, key, bool(val))

        cb_data = data.get("cbpc", {})
        if isinstance(cb_data, dict):
            if isinstance(cb_data.get("enabled"), bool):
                cfg.cbpc.enabled = cb_data["enabled"]
            gravity = cb_data.get("gravity")
            if isinstance(gravity, (int, float)) and not isinstance(gravity, bool) and math.isfinite(gravity):
                cfg.cbpc.gravity = max(0.0, min(5.0, float(gravity)))
            softness = cb_data.get("softness")
            if isinstance(softness, dict):
                for group in SOFTNESS_GROUPS:
                    if group not in softness:
                        continue
                    value = softness[group]
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        cfg.cbpc.softness[group] = max(SOFTNESS_MIN, min(SOFTNESS_MAX, float(value)))
                    else:
                        log.warning("Ignoring invalid CBPC softness for %s", group)
                if "thigh" not in softness:
                    cfg.cbpc.softness["thigh"] = cfg.cbpc.softness["other"]
            elif isinstance(softness, (int, float)) and not isinstance(softness, bool) and math.isfinite(softness):
                cfg.cbpc.softness = dict.fromkeys(
                    SOFTNESS_GROUPS, max(SOFTNESS_MIN, min(SOFTNESS_MAX, float(softness)))
                )
            elif "softness" in cb_data:
                log.warning("Ignoring invalid CBPC softness setting")
            if "softness" not in cb_data and "bounce" in cb_data:
                bounce = cb_data["bounce"]
                if isinstance(bounce, (int, float)) and not isinstance(bounce, bool) and math.isfinite(bounce):
                    # Keep the existing inverse mapping for legacy bounce settings.
                    value = min(5.0, 1.0 / min(bounce, 5.0)) if bounce > 0 else 0.0
                    cfg.cbpc.softness = dict.fromkeys(SOFTNESS_GROUPS, value)
                else:
                    log.warning("Ignoring invalid legacy CBPC bounce setting")

        smp_data = data.get("smp", {})
        if isinstance(smp_data, dict):
            if isinstance(smp_data.get("enabled"), bool):
                cfg.smp.enabled = smp_data["enabled"]

        return cfg

    def save(self, path: Optional[Path | str] = None) -> None:
        config_file = Path(path) if path else DEFAULT_CONFIG_PATH
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as exc:
            log.warning("Could not save config to %s: %s", config_file, exc)
