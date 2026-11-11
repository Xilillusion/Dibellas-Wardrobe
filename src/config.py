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
class ViewportEffectsConfig:
    enable_ssr: bool = False
    ssr_step_size: float = 1.0
    ssr_max_steps: int = 32
    ssr_thickness: float = 2.0

    enable_ao: bool = False
    ao_radius: float = 8.0
    ao_intensity: float = 1.0
    ao_bias: float = 0.02

    enable_ssss: bool = False
    ssss_radius: float = 5.0
    ssss_falloff_color: tuple[float, float, float] = (1.0, 0.45, 0.35)


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


SOFTNESS_GROUPS = ("breasts", "butt", "belly", "other")


@dataclass
class CbpcConfig:
    enabled: bool = True
    gravity: float = 1.0 / 3.0  # breast gravity factor (0 = Armor, 1 = More Gravity)
    softness: Dict[str, float] = field(default_factory=lambda: dict.fromkeys(SOFTNESS_GROUPS, 1.0))


@dataclass
class SmpConfig:
    enabled: bool = True
    collision_xmls: list[str] = field(default_factory=list)
    show_collisions: bool = False


@dataclass
class AppConfig:
    default_folder: str = ""
    language: str = ""
    lighting: LightingConfig = field(default_factory=LightingConfig)
    effects: ViewportEffectsConfig = field(default_factory=ViewportEffectsConfig)
    checkboxes: CheckboxConfig = field(default_factory=CheckboxConfig)
    cbpc: CbpcConfig = field(default_factory=CbpcConfig)
    smp: SmpConfig = field(default_factory=SmpConfig)
    locked_nifs: list[str] = field(default_factory=list)
    locked_presets: Dict[str, list[str]] = field(default_factory=dict)
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

        locked_nifs = data.get("locked_nifs", [])
        if isinstance(locked_nifs, list):
            cfg.locked_nifs = [path for path in locked_nifs if isinstance(path, str)]

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
        elif cfg.locked_nifs:
            cfg.locked_presets = {"1": list(cfg.locked_nifs)}

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
            for key in (
                "enable_ssr",
                "ssr_step_size",
                "ssr_max_steps",
                "ssr_thickness",
                "enable_ao",
                "ao_radius",
                "ao_intensity",
                "ao_bias",
                "enable_ssss",
                "ssss_radius",
            ):
                if key not in effects_data:
                    continue
                try:
                    current = getattr(cfg.effects, key)
                    value = effects_data[key]
                    if isinstance(current, bool):
                        if isinstance(value, bool):
                            setattr(cfg.effects, key, value)
                    elif isinstance(current, int):
                        setattr(cfg.effects, key, int(value))
                    else:
                        setattr(cfg.effects, key, float(value))
                except (ValueError, TypeError, OverflowError):
                    log.warning("Ignoring invalid viewport effect setting %s", key)

            falloff_color = effects_data.get("ssss_falloff_color")
            if (
                isinstance(falloff_color, (list, tuple))
                and len(falloff_color) == 3
            ):
                try:
                    cfg.effects.ssss_falloff_color = tuple(
                        max(0.0, min(1.0, float(channel)))
                        for channel in falloff_color
                    )
                except (ValueError, TypeError, OverflowError):
                    log.warning("Ignoring invalid SSSS falloff color")

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
                        cfg.cbpc.softness[group] = max(0.0, min(5.0, float(value)))
                    else:
                        log.warning("Ignoring invalid CBPC softness for %s", group)
            elif isinstance(softness, (int, float)) and not isinstance(softness, bool) and math.isfinite(softness):
                cfg.cbpc.softness = dict.fromkeys(SOFTNESS_GROUPS, max(0.0, min(5.0, float(softness))))
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
            if isinstance(smp_data.get("show_collisions"), bool):
                cfg.smp.show_collisions = smp_data["show_collisions"]
            paths = smp_data.get("collision_xmls", [])
            if isinstance(paths, list) and all(isinstance(path, str) for path in paths):
                cfg.smp.collision_xmls = paths
            else:
                log.warning("Ignoring invalid SMP collision XML paths")

        return cfg

    def save(self, path: Optional[Path | str] = None) -> None:
        config_file = Path(path) if path else DEFAULT_CONFIG_PATH
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as exc:
            log.warning("Could not save config to %s: %s", config_file, exc)
