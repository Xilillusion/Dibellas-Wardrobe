"""
Per-preset post-processing configuration.

Each outfit shader preset (`shaders/<name>.frag`) may ship a sidecar
`shaders/<name>.json` whose `"postprocess"` object selects which screen-space
effects run before the preset and how they are tuned::

    {
      "postprocess": {
        "enable_gtao": true, "gtao_radius": 8.0, "gtao_intensity": 0.6,
        "enable_ssss": true, "sss_radius": 4.0, "sss_strength": 0.2
      }
    }

Omitted fields keep the defaults of `PostProcessProfile`. A preset without a
sidecar, like the built-in "none" look, runs neither effect.

With `"ssss_after_preset": true` the preset instead renders a split output:
`layout(location = 0)` diffuse and `layout(location = 1)` specular (RGB) with
an SSSS mask (alpha). The built-in SSSS pass then runs after the preset,
scattering only the masked diffuse, and adds the specular back unblurred.
GTAO still runs before the preset.

Two more optional top-level keys are read independently of "postprocess":
`"label"` replaces the selector name derived from the file name, and
`"uniforms"` maps preset `float`/`vecN` uniform names to a number or a list of
2-4 numbers, set each frame the preset renders::

    {"label": "Glossy (Oiled)", "uniforms": {"u_gloss_intensity": 1.0}}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


@dataclass(frozen=True)
class PostProcessProfile:
    enable_gtao: bool = False
    gtao_radius: float = 8.0
    gtao_intensity: float = 1.0
    gtao_bias: float = 0.02

    enable_ssss: bool = False
    sss_radius: float = 5.0
    sss_strength: float = 0.35
    sss_falloff_color: Tuple[float, float, float] = (1.0, 0.45, 0.35)
    ssss_after_preset: bool = False


NO_POSTPROCESS = PostProcessProfile()

_FLOAT_RANGES: Dict[str, Tuple[float, float]] = {
    "gtao_radius": (1.0, 40.0),
    "gtao_intensity": (0.0, 3.0),
    "gtao_bias": (0.0, 0.1),
    "sss_radius": (1.0, 24.0),
    "sss_strength": (0.0, 1.0),
}


def preset_metadata_path(fragment_path: Path) -> Path:
    return fragment_path.with_suffix(".json")


UniformValue = Union[float, Tuple[float, ...]]

# Uniforms the viewport binds itself; a sidecar may not override them.
_RESERVED_UNIFORMS = frozenset({
    "u_scene_color", "u_scene_depth", "u_scene_normal", "u_material_normal",
    "u_material_env_mask", "u_material_subsurface", "u_material_flags",
    "u_viewport_size", "u_inverse_projection", "u_key_light",
    "u_studio_cubemap", "u_view_to_world",
})
_MAX_LABEL_LENGTH = 64


def _read_metadata(fragment_path: Path) -> Dict[str, Any]:
    """The sidecar's top-level object, or {} without one; raises ValueError if unreadable."""
    metadata_path = preset_metadata_path(fragment_path)
    if not metadata_path.is_file():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{metadata_path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{metadata_path.name}: expected a JSON object")
    return data


def load_preset_label(fragment_path: Path) -> Optional[str]:
    """The sidecar's display label, or None to use the file name; raises ValueError if invalid."""
    label = _read_metadata(fragment_path).get("label")
    if label is None:
        return None
    if not isinstance(label, str) or not label.strip() or len(label.strip()) > _MAX_LABEL_LENGTH:
        raise ValueError(
            f"{preset_metadata_path(fragment_path).name}: \"label\" must be a non-empty "
            f"string of at most {_MAX_LABEL_LENGTH} characters"
        )
    return label.strip()


def load_preset_uniforms(fragment_path: Path) -> Dict[str, UniformValue]:
    """The sidecar's uniform overrides; raises ValueError if any are invalid."""
    name = preset_metadata_path(fragment_path).name
    section = _read_metadata(fragment_path).get("uniforms", {})
    if not isinstance(section, dict):
        raise ValueError(f"{name}: \"uniforms\" must be an object")
    uniforms: Dict[str, UniformValue] = {}
    for uniform, value in section.items():
        if not uniform.startswith("u_") or not uniform[2:].isidentifier():
            raise ValueError(f"{name}: uniform name {uniform!r} must look like u_name")
        if uniform in _RESERVED_UNIFORMS:
            raise ValueError(f"{name}: uniform {uniform} is set by the viewport")
        if _is_number(value):
            uniforms[uniform] = float(value)
        elif (
            isinstance(value, list)
            and 2 <= len(value) <= 4
            and all(_is_number(component) for component in value)
        ):
            uniforms[uniform] = tuple(float(component) for component in value)
        else:
            raise ValueError(
                f"{name}: uniform {uniform} must be a number or a list of 2-4 numbers"
            )
    return uniforms


def load_postprocess_profile(fragment_path: Path) -> PostProcessProfile:
    """Read a preset's post-processing profile; raises ValueError if it is invalid."""
    metadata_path = preset_metadata_path(fragment_path)
    data = _read_metadata(fragment_path)
    if not data:
        return NO_POSTPROCESS
    section = data.get("postprocess", {})
    if not isinstance(section, dict):
        raise ValueError(f"{metadata_path.name}: \"postprocess\" must be an object")
    try:
        return parse_postprocess_profile(section)
    except ValueError as exc:
        raise ValueError(f"{metadata_path.name}: {exc}") from exc


def parse_postprocess_profile(section: Dict[str, Any]) -> PostProcessProfile:
    known = {item.name for item in fields(PostProcessProfile)}
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"unknown post-process setting(s): {', '.join(unknown)}")

    values: Dict[str, Any] = {}
    for name, value in section.items():
        if name.startswith("enable_") or name == "ssss_after_preset":
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false")
            values[name] = value
        elif name == "sss_falloff_color":
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 3
                or not all(_is_number(channel) for channel in value)
            ):
                raise ValueError(f"{name} must be a list of three numbers")
            values[name] = tuple(_clamp(float(channel), 0.0, 1.0) for channel in value)
        else:
            if not _is_number(value):
                raise ValueError(f"{name} must be a number")
            low, high = _FLOAT_RANGES[name]
            values[name] = _clamp(float(value), low, high)
    return PostProcessProfile(**values)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
