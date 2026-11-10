"""
Asset discovery, background search, and file filtering utilities for Skyrim NIF and HKX assets.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from .mod_scanner import ModScanner, is_animation_hkx

log = logging.getLogger("dibellas_wardrobe.asset_scanner")


def is_filtered_nif_filename(
    filename: str, hide_low_weight: bool = True, hide_gnd: bool = False
) -> bool:
    """Return True if `filename` should be hidden by the active smart NIF filters."""
    lower = filename.lower()
    if not lower.endswith(".nif"):
        return True
    stem = lower[:-4]
    return (hide_low_weight and stem.endswith("_0")) or (
        hide_gnd and stem.endswith("_gnd")
    )


def is_filtered_folder_name(folder_name: str, hide_gnd_folders: bool = True) -> bool:
    """Return True if `folder_name` should be hidden by smart folder filters."""
    if not hide_gnd_folders:
        return False
    lower = folder_name.lower()
    return (
        lower in ("gnd", "_gnd", "ground")
        or lower.endswith("_gnd")
        or lower.startswith("gnd")
    )


@lru_cache(maxsize=16384)
def directory_contains_nif_or_hkx(directory: str) -> bool:
    """Return whether a directory contains a NIF or HKX anywhere below it."""
    def log_walk_error(error: OSError) -> None:
        log.warning("Could not inspect directory %s: %s", error.filename, error)

    for _root, _dirs, files in os.walk(directory, onerror=log_walk_error):
        if any(Path(name).suffix.casefold() in {".nif", ".hkx"} for name in files):
            return True
    return False


def normalize_mods_root(folder: Path) -> Path:
    """Use MO2's mods subdirectory when the user selected the MO2 instance root."""
    if (
        folder.name.casefold() != "mods"
        and (folder / "mods").is_dir()
        and (folder / "overwrite").is_dir()
    ):
        return folder / "mods"
    return folder


def resolve_mod_root_for_nif(
    nif_key: str,
    mod_scanner: Optional[ModScanner],
    current_root_dir: Optional[Path],
) -> Optional[Path]:
    """Resolve the owning mod directory for a given NIF file path."""
    owner = mod_scanner.owner_for_nif(nif_key) if mod_scanner else None
    if owner is not None:
        return owner
    if (
        mod_scanner
        and mod_scanner.fallback_target_for_nif(nif_key) is not None
        and mod_scanner.overwrite_root is not None
    ):
        return mod_scanner.overwrite_root
    if current_root_dir is not None:
        is_mod_dir = any(
            (current_root_dir / folder).is_dir()
            for folder in ("meshes", "textures", "Data/meshes", "Data/textures")
        )
        if is_mod_dir:
            return current_root_dir
        try:
            relative = Path(nif_key).resolve().relative_to(current_root_dir.resolve())
            if len(relative.parts) > 1:
                return current_root_dir / relative.parts[0]
        except ValueError:
            pass
    return Path(nif_key).resolve().parent


def get_mod_name_for_nif(
    nif_key: str,
    mod_scanner: Optional[ModScanner],
    current_root_dir: Optional[Path],
) -> str:
    """Return the display mod name for a given NIF file path."""
    owner = mod_scanner.owner_for_nif(nif_key) if mod_scanner else None
    if owner is not None:
        return owner.name
    if mod_scanner and mod_scanner.fallback_target_for_nif(nif_key) is not None:
        return "[Overwrite]"
    mod_root = resolve_mod_root_for_nif(nif_key, mod_scanner, current_root_dir)
    if mod_root is not None:
        return mod_root.name
    return Path(nif_key).parent.name


def discover_mod_textures_for_nif(
    nif_key: str,
    mod_scanner: Optional[ModScanner],
    current_root_dir: Optional[Path],
) -> List[tuple[str, Path]]:
    """Scan mod textures directory for available diffuse DDS files."""
    mod_root = resolve_mod_root_for_nif(nif_key, mod_scanner, current_root_dir)
    if mod_root is None:
        mod_root = Path(nif_key).resolve().parent

    texture_roots = [
        mod_root / "textures",
        mod_root / "Data" / "textures",
    ]
    results_by_label: Dict[str, tuple[str, Path]] = {}

    for texture_root in texture_roots:
        if not texture_root.is_dir():
            continue
        for root, dirs, files in os.walk(texture_root):
            dirs.sort(key=str.casefold)
            root_path = Path(root)
            for file_name in sorted(files, key=str.casefold):
                if not file_name.casefold().endswith(".dds"):
                    continue
                full_path = (root_path / file_name).resolve()
                try:
                    rel_label = full_path.relative_to(texture_root).as_posix()
                except ValueError:
                    rel_label = full_path.name
                label_parts = Path(rel_label).parts
                if label_parts and label_parts[0].casefold() == mod_root.name.casefold():
                    rel_label = (
                        Path(*label_parts[1:]).as_posix()
                        if len(label_parts) > 1
                        else full_path.name
                    )
                label_key = rel_label.casefold()
                if label_key not in results_by_label:
                    results_by_label[label_key] = (rel_label, full_path)

    return list(results_by_label.values())


class NifSearchThread(QThread):
    """
    Background worker that recursively searches `root_dir` for .nif and .hkx files
    matching the search query without blocking the PySide6 UI thread.
    """

    matchFound = Signal(str, str, str, str)  # (mod name, relative directory, file name, path)

    def __init__(
        self,
        root_dir: Path,
        query: str,
        hide_0: bool,
        hide_gnd_folder: bool = True,
        meshes_and_caliente_only: bool = True,
        mod_scanner: Optional[ModScanner] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.root_dir = root_dir
        self.query = query.lower()
        self.hide_0 = hide_0
        self.hide_gnd_folder = hide_gnd_folder
        self.meshes_and_caliente_only = meshes_and_caliente_only
        self.mod_scanner = mod_scanner
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            for root, dirs, files in os.walk(self.root_dir):
                if self._cancelled:
                    return

                if self.hide_gnd_folder:
                    dirs[:] = [d for d in dirs if not is_filtered_folder_name(d, True)]

                root_path = Path(root)
                if root_path == self.root_dir:
                    if self.query not in self.root_dir.name.lower():
                        dirs[:] = [d for d in dirs if self.query in d.lower()]
                    if self.meshes_and_caliente_only:
                        dirs[:] = [
                            d for d in dirs
                            if directory_contains_nif_or_hkx(str(root_path / d))
                        ]

                try:
                    rel_parts = root_path.relative_to(self.root_dir).parts
                    group_name = rel_parts[0] if rel_parts else root_path.name
                except Exception:
                    group_name = root_path.name

                if self.query not in group_name.lower():
                    continue

                for fname in files:
                    if self._cancelled:
                        return
                    lower_fname = fname.casefold()
                    if lower_fname.endswith(".nif"):
                        if is_filtered_nif_filename(fname, self.hide_0, self.hide_gnd_folder):
                            continue
                    elif is_animation_hkx(fname, rel_parts if "rel_parts" in locals() else ()):
                        pass
                    else:
                        continue
                    if self.mod_scanner:
                        try:
                            relative_parts = root_path.relative_to(
                                self.root_dir
                            ).parts
                            if relative_parts:
                                mod_path = self.root_dir / relative_parts[0]
                                if self.mod_scanner.is_overridden_mod_file(
                                    mod_path, relative_parts[1:] + (fname,)
                                ):
                                    continue
                        except Exception:
                            pass

                    if self.query in fname.lower() or self.query in group_name.lower():
                        try:
                            rel_dir = str(root_path.relative_to(self.root_dir / group_name))
                            if rel_dir == ".":
                                rel_dir = ""
                        except Exception:
                            rel_dir = ""
                        self.matchFound.emit(
                            group_name,
                            rel_dir,
                            fname,
                            str(root_path / fname),
                        )
        except Exception as exc:
            log.warning("Search thread error on %s: %s", self.root_dir, exc)
