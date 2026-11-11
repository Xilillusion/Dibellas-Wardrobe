"""Eager indexing and mod ownership matching for Mod Organizer 2 overwrite NIFs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

log = logging.getLogger("dibellas_wardrobe.mod_scanner")

_NON_ANIM_HKX_NAMES = frozenset(
    {
        "0_master.hkx",
        "defaultfemale.hkx",
        "defaultmale.hkx",
        "mt_behavior.hkx",
    }
)
_NON_ANIM_HKX_DIRS = frozenset(
    {
        "behaviors",
        "character assets",
        "character assets female",
    }
)


def is_animation_hkx(file_name: str, relative_parts: Iterable[str] = ()) -> bool:
    """Return True if `file_name` is a playable `.hkx` animation (excluding behaviors/skeletons)."""
    lower_name = file_name.casefold()
    if not lower_name.endswith(".hkx"):
        return False
    if lower_name.startswith("skeleton") or lower_name in _NON_ANIM_HKX_NAMES:
        return False
    norm_parts = tuple(p.casefold() for p in relative_parts)
    if any(part in _NON_ANIM_HKX_DIRS for part in norm_parts):
        return False
    if "meshes" in norm_parts:
        return "animations" in norm_parts
    return True


@dataclass(frozen=True)
class OverlayEntry:
    """An overwrite file or virtual directory in a mod tree."""

    name: str
    is_dir: bool
    path: Path


@dataclass(frozen=True)
class ModData:
    """A mod root and its on-disk folder prefix for a normalized asset path."""

    path: Path
    tree_prefix: tuple[str, ...]


@dataclass
class MeshItem:
    """Indexed NIF entry whose source path can be replaced by an overwrite path."""

    filepath: Path
    mod: ModData
    relative_parts: tuple[str, ...]


class ModScanner:
    """Merge whitelisted MO2 overwrite assets into matching mod folders."""

    _CHARGEN_PARTS = ("SKSE", "Plugins", "CharGen")

    def __init__(self, mods_root: Path | str) -> None:
        self.mods_root = Path(mods_root).resolve()
        self.overwrite_root: Optional[Path] = None
        self._mod_directory_paths: Dict[Path, Dict[tuple[str, ...], Path]] = {}
        self.relpath_to_mod: Dict[str, ModData] = {}
        self.relpath_to_mesh: Dict[str, MeshItem] = {}
        self._mesh_sources: Dict[str, Dict[Path, MeshItem]] = {}
        self._ambiguous_mod_paths: set[str] = set()
        self._ambiguous_mesh_paths: set[str] = set()
        self._mod_overrides: Dict[
            tuple[Path, tuple[str, ...]], Dict[str, Path]
        ] = {}
        self._mod_overlay_directories: Dict[
            tuple[Path, tuple[str, ...]], Dict[str, Path]
        ] = {}
        self._nif_owners: Dict[Path, Path] = {}
        self._nif_targets: Dict[Path, tuple[Path, tuple[str, ...]]] = {}
        self._fallback_targets: Dict[Path, tuple[str, ...]] = {}
        self._fallback_files: Dict[tuple[str, ...], Dict[str, Path]] = {}
        self._fallback_directories: Dict[
            tuple[str, ...], Dict[str, Path]
        ] = {}
        self._overwrite_search_entries: list[tuple[str, tuple[str, ...], Path]] = []
        self._build_index()

    @property
    def is_mods_root(self) -> bool:
        return self.mods_root.name.casefold() == "mods"

    @staticmethod
    def _normalized_parts(parts: Iterable[str]) -> tuple[str, ...]:
        return tuple(part.lower() for part in parts)

    @staticmethod
    def _path_key(parts: Iterable[str]) -> str:
        return "/".join(parts).lower()

    def _register_mod_path(self, key: str, mod: ModData) -> None:
        if key in self._ambiguous_mod_paths:
            return
        existing = self.relpath_to_mod.get(key)
        if existing is None:
            self.relpath_to_mod[key] = mod
        elif existing.path != mod.path:
            self.relpath_to_mod.pop(key, None)
            self._ambiguous_mod_paths.add(key)
        elif len(mod.tree_prefix) < len(existing.tree_prefix):
            self.relpath_to_mod[key] = mod

    def _register_mesh(self, key: str, mesh: MeshItem) -> None:
        sources = self._mesh_sources.setdefault(key, {})
        existing_source = sources.get(mesh.mod.path)
        if existing_source is None or len(mesh.mod.tree_prefix) < len(existing_source.mod.tree_prefix):
            sources[mesh.mod.path] = mesh
        if key in self._ambiguous_mesh_paths:
            return
        existing = self.relpath_to_mesh.get(key)
        if existing is None:
            self.relpath_to_mesh[key] = mesh
        elif existing.mod.path != mesh.mod.path:
            self.relpath_to_mesh.pop(key, None)
            self._ambiguous_mesh_paths.add(key)
        elif len(mesh.mod.tree_prefix) < len(existing.mod.tree_prefix):
            self.relpath_to_mesh[key] = mesh

    @staticmethod
    def _log_walk_error(error: OSError) -> None:
        log.warning("Could not index directory %s: %s", error.filename, error)

    def _index_mod_directory(
        self,
        mod_path: Path,
        directory_paths: Dict[tuple[str, ...], Path],
        actual_parts: tuple[str, ...],
        actual_path: Path,
    ) -> tuple[Optional[ModData], tuple[str, ...]]:
        normalized_parts = self._normalized_parts(actual_parts)
        directory_paths[normalized_parts] = actual_path

        visible_parts = actual_parts
        if (
            len(normalized_parts) >= 2
            and normalized_parts[0] == "data"
            and normalized_parts[1] in {"meshes", "textures"}
        ):
            visible_parts = actual_parts[1:]
        if not visible_parts or visible_parts[0].casefold() not in {
            "meshes",
            "textures",
        }:
            return None, ()

        tree_prefix = actual_parts[: len(actual_parts) - len(visible_parts)]
        mod_data = ModData(mod_path, tree_prefix)
        self._register_mod_path(self._path_key(visible_parts), mod_data)
        return mod_data, visible_parts

    def _index_mod_directories(self) -> None:
        try:
            mod_paths = sorted([
                Path(entry.path).resolve()
                for entry in os.scandir(self.mods_root)
                if entry.is_dir()
            ], key=lambda path: str(path).casefold())
        except OSError as exc:
            log.warning("Could not list MO2 mods directory %s: %s", self.mods_root, exc)
            return

        for mod_path in mod_paths:
            directory_paths = {(): mod_path}
            for root, dirs, files in os.walk(
                mod_path, onerror=self._log_walk_error
            ):
                root_path = Path(root)
                relative_parts = root_path.relative_to(mod_path).parts
                normalized_parts = self._normalized_parts(relative_parts)
                if not normalized_parts:
                    dirs[:] = [
                        directory
                        for directory in dirs
                        if directory.casefold() in {"meshes", "textures", "data"}
                    ]
                elif normalized_parts == ("data",):
                    dirs[:] = [
                        directory
                        for directory in dirs
                        if directory.casefold() in {"meshes", "textures"}
                    ]
                elif normalized_parts[0] not in {"meshes", "textures", "data"}:
                    dirs.clear()
                    continue

                mod_data, visible_parts = self._index_mod_directory(
                    mod_path, directory_paths, relative_parts, root_path
                )
                if mod_data and visible_parts[0].casefold() == "meshes":
                    for file_name in files:
                        lower_name = file_name.casefold()
                        if not (
                            lower_name.endswith(".nif")
                            or is_animation_hkx(file_name, visible_parts)
                        ):
                            continue
                        mesh_parts = visible_parts + (file_name,)
                        mesh = MeshItem(
                            root_path / file_name,
                            mod_data,
                            mesh_parts,
                        )
                        self._register_mesh(self._path_key(mesh_parts), mesh)
            self._mod_directory_paths[mod_path] = directory_paths

    def _add_mod_override(
        self, mod_path: Path, target_parts: tuple[str, ...], nif_path: Path
    ) -> None:
        normalized_parts = self._normalized_parts(target_parts)
        parent_parts = normalized_parts[:-1]
        file_name = target_parts[-1]
        key = (mod_path, parent_parts)
        self._mod_overrides.setdefault(key, {})[file_name.casefold()] = nif_path
        for depth in range(1, len(parent_parts) + 1):
            directory_parts = target_parts[:depth]
            directory_key = (mod_path, normalized_parts[: depth - 1])
            name = directory_parts[-1]
            virtual_path = self._mod_directory_paths.get(mod_path, {}).get(
                normalized_parts[:depth],
                mod_path.joinpath(*directory_parts),
            )
            self._mod_overlay_directories.setdefault(directory_key, {})[
                name.casefold()
            ] = virtual_path

    def _add_fallback_file(self, relative_parts: tuple[str, ...], path: Path) -> None:
        target_parts = relative_parts or (path.name,)
        parent_parts = self._normalized_parts(target_parts[:-1])
        key = target_parts[-1].casefold()
        resolved_path = path
        entry_key = resolved_path.as_posix().lower() if not relative_parts else key
        self._fallback_files.setdefault(parent_parts, {})[entry_key] = resolved_path
        self._fallback_targets[resolved_path] = target_parts
        self._overwrite_search_entries.append(("[Overwrite]", target_parts, resolved_path))
        assert self.overwrite_root is not None
        for depth in range(1, len(target_parts)):
            directory_parts = target_parts[:depth]
            normalized_directory_parts = self._normalized_parts(directory_parts)
            self._fallback_directories.setdefault(
                normalized_directory_parts[:-1], {}
            )[directory_parts[-1].casefold()] = self.overwrite_root.joinpath(
                *directory_parts
            )

    def _register_overwrite_mesh(self, nif_path: Path, relative_mesh: Path) -> None:
        relative_parts = ("meshes",) + relative_mesh.parts
        rel_key = self._path_key(relative_parts)
        sources = self._mesh_sources.get(rel_key, {})
        if len(sources) > 1:
            for mesh in sources.values():
                target_parts = mesh.mod.tree_prefix + mesh.relative_parts
                self._add_mod_override(mesh.mod.path, target_parts, nif_path)
                self._nif_targets.setdefault(nif_path, (mesh.mod.path, target_parts))
                self._overwrite_search_entries.append(
                    (mesh.mod.path.name, target_parts, nif_path)
                )
            return
        mesh_item = self.relpath_to_mesh.get(rel_key)
        if mesh_item is not None:
            owner = mesh_item.mod
            mesh_item.filepath = nif_path
            target_parts = owner.tree_prefix + mesh_item.relative_parts
        else:
            parent_key = self._path_key(relative_parts[:-1])
            owner = self.relpath_to_mod.get(parent_key)
            if owner is None:
                self._add_fallback_file(relative_parts, nif_path)
                return
            target_parts = owner.tree_prefix + relative_parts
            self._register_mesh(
                rel_key,
                MeshItem(nif_path, owner, relative_parts),
            )

        self._nif_owners[nif_path] = owner.path
        self._nif_targets[nif_path] = (owner.path, target_parts)
        self._add_mod_override(owner.path, target_parts, nif_path)
        self._overwrite_search_entries.append((owner.path.name, target_parts, nif_path))

    def _build_index(self) -> None:
        """Normalize MO2 roots, then build mod and whitelisted overwrite indexes."""
        overwrite_exists = False
        if (
            self.mods_root.name.casefold() != "mods"
            and (self.mods_root / "mods").is_dir()
        ):
            candidate_overwrite = self.mods_root / "overwrite"
            if candidate_overwrite.is_dir():
                self.mods_root = (self.mods_root / "mods").resolve()
                self.overwrite_root = candidate_overwrite.resolve()
                overwrite_exists = True
        if self.overwrite_root is None:
            if self.mods_root.name.casefold() == "mods":
                self.overwrite_root = self.mods_root.parent / "overwrite"
                overwrite_exists = self.overwrite_root.is_dir()
        self._index_mod_directories()
        if not overwrite_exists or not self.overwrite_root:
            return

        meshes_root = self.overwrite_root / "meshes"
        if meshes_root.is_dir():
            for root, _, files in os.walk(
                meshes_root, onerror=self._log_walk_error
            ):
                root_path = Path(root)
                relative_directory = root_path.relative_to(meshes_root)
                rel_parts = ("meshes",) + relative_directory.parts
                for file_name in files:
                    lower_name = file_name.casefold()
                    if not (
                        lower_name.endswith(".nif")
                        or is_animation_hkx(file_name, rel_parts)
                    ):
                        continue
                    nif_path = (root_path / file_name).resolve()
                    relative_mesh = relative_directory / file_name
                    self._register_overwrite_mesh(nif_path, relative_mesh)

        chargen_root = self.overwrite_root.joinpath(*self._CHARGEN_PARTS)
        if chargen_root.is_dir():
            for root, _, files in os.walk(
                chargen_root, onerror=self._log_walk_error
            ):
                for file_name in files:
                    if file_name.casefold().endswith(".nif"):
                        self._add_fallback_file((), (Path(root) / file_name).resolve())

    def mod_overlay_entries(
        self, mod_path: Path | str, relative_parts: Iterable[str]
    ) -> list[OverlayEntry]:
        mod = Path(mod_path).resolve()
        relative = self._normalized_parts(relative_parts)
        key = (mod, relative)
        entries = [
            OverlayEntry(path.name, True, path)
            for path in self._mod_overlay_directories.get(key, {}).values()
        ]
        entries.extend(
            OverlayEntry(path.name, False, path)
            for path in self._mod_overrides.get(key, {}).values()
        )
        return entries

    def fallback_entries(self, relative_parts: Iterable[str] = ()) -> list[OverlayEntry]:
        relative = self._normalized_parts(relative_parts)
        entries = [
            OverlayEntry(path.name, True, path)
            for path in self._fallback_directories.get(relative, {}).values()
        ]
        entries.extend(
            OverlayEntry(path.name, False, path)
            for path in self._fallback_files.get(relative, {}).values()
        )
        return entries

    def owner_for_nif(self, nif_path: Path | str) -> Optional[Path]:
        return self._nif_owners.get(Path(nif_path).resolve())

    def target_for_nif(
        self, nif_path: Path | str
    ) -> Optional[tuple[Path, tuple[str, ...]]]:
        return self._nif_targets.get(Path(nif_path).resolve())

    def fallback_target_for_nif(
        self, nif_path: Path | str
    ) -> Optional[tuple[str, ...]]:
        return self._fallback_targets.get(Path(nif_path).resolve())

    def is_overwrite_nif(self, nif_path: Path | str) -> bool:
        path = Path(nif_path).resolve()
        return path in self._nif_targets or path in self._fallback_targets

    def has_fallback(self) -> bool:
        return bool(self._fallback_files)

    def has_mod_overlay(self, mod_path: Path | str, relative_parts: Iterable[str] = ()) -> bool:
        key = (Path(mod_path).resolve(), self._normalized_parts(relative_parts))
        return bool(self._mod_overlay_directories.get(key) or self._mod_overrides.get(key))

    def is_overridden_mod_file(
        self, mod_path: Path | str, relative_parts: Iterable[str]
    ) -> bool:
        mod = Path(mod_path).resolve()
        parts = self._normalized_parts(relative_parts)
        if not parts:
            return False
        return parts[-1] in self._mod_overrides.get((mod, parts[:-1]), {})

    def search_entries(self) -> list[tuple[str, tuple[str, ...], Path]]:
        """Return indexed overwrite assets and their visible tree paths."""
        return sorted(
            self._overwrite_search_entries,
            key=lambda entry: (entry[0].casefold(), self._path_key(entry[1]), str(entry[2]).casefold()),
        )