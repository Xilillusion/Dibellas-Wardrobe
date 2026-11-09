"""
Heuristic Texture Path Resolver and DDS Texture Loader for Dibella's Wardrobe.

Handles Skyrim SE/AE relative texture paths (e.g. `textures\\armor\\my_mod\\cuirass.dds`),
searches upwards from the `.nif` directory (and across registered mod roots) to find matching
`textures/` folders case-insensitively, decodes BC1/BC3/BC5/BC7 `.dds` textures into RGBA8
NumPy arrays, and provides graceful fallbacks when vanilla/external textures are absent.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imagecodecs
import numpy as np
from PIL import Image

log = logging.getLogger("dibellas_wardrobe.texture_resolver")

@dataclass
class ResolvedTexture:
    """Represents a resolved texture slot and its decoded RGBA8 image data."""

    slot: str
    raw_nif_path: str
    resolved_path: Optional[Path]
    is_fallback: bool
    width: int
    height: int
    rgba_data: np.ndarray  # uint8 array of shape (H, W, 4)


def normalize_nif_texture_path(raw_path: str) -> str:
    """
    Normalize a texture or CharGen-relative path using OS-neutral forward slashes.
    """
    if not raw_path:
        return ""
    cleaned = raw_path.strip().replace("\\", "/")
    while cleaned.startswith("/"):
        cleaned = cleaned[1:]

    lower = cleaned.lower()
    # Strip leading "data/" if authored as "Data/Textures/..."
    if lower.startswith("data/"):
        cleaned = cleaned[5:]
        lower = cleaned.lower()

    # CharGen exports can reference SKSE assets directly; other relative paths are textures.
    if not lower.startswith(("textures/", "skse/plugins/chargen/")):
        idx = lower.find("textures/")
        if idx != -1:
            cleaned = cleaned[idx:]
        else:
            cleaned = f"textures/{cleaned}"

    if not cleaned.lower().endswith(".dds") and "." not in Path(cleaned).name:
        cleaned += ".dds"

    return cleaned


def _resolve_case_insensitive(base_dir: Path, rel_path: str) -> Optional[Path]:
    """
    Resolve `rel_path` inside `base_dir`, trying direct path existence first (fast on Windows)
    and falling back to component-by-component case-insensitive traversal on POSIX.
    """
    candidate = base_dir / rel_path
    try:
        if candidate.is_file():
            return candidate.resolve()
    except OSError:
        pass

    # On Windows, NTFS/FAT filesystems are already case-insensitive.
    # If candidate.is_file() returned False, the file does not exist on disk.
    if os.name == "nt":
        return None

    current = base_dir
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p and p != "."]
    for part in parts:
        direct = current / part
        try:
            if direct.exists():
                current = direct
                continue
        except OSError:
            pass

        part_lower = part.lower()
        matched = None
        try:
            for entry in os.scandir(current):
                if entry.name.lower() == part_lower:
                    matched = Path(entry.path)
                    break
        except OSError:
            return None

        if matched is None:
            return None
        current = matched

    return current.resolve() if current.is_file() else None


class TextureResolver:
    """
    Resolves relative Skyrim texture paths from a `.nif` file by searching upwards from
    the `.nif` file's folder and across any registered root/mod directories.
    Caches decoded RGBA8 arrays in memory so shared textures are only decoded once.
    """

    def __init__(
        self,
        extra_roots: Optional[List[Path | str]] = None,
        overwrite_root: Optional[Path | str] = None,
    ) -> None:
        self.extra_roots: List[Path] = []
        self.overwrite_root = Path(overwrite_root).resolve() if overwrite_root else None
        self._overwrite_indexed = False
        self._overwrite_path_index: Dict[str, Path] = {}
        self._nif_owners: Dict[Path, Path] = {}
        self._path_cache: Dict[Tuple[str, str], Optional[Path]] = {}
        self._root_path_cache: Dict[str, Optional[Path]] = {}
        self._image_cache: Dict[Path, np.ndarray] = {}
        self._fallback_cache: Dict[str, np.ndarray] = {}
        self._mod_subdirs_cache: Dict[Path, List[Path]] = {}
        if extra_roots:
            for r in extra_roots:
                self.add_search_root(r)

    def configure_mo2_paths(
        self,
        mods_root: Optional[Path | str],
        overwrite_root: Optional[Path | str],
    ) -> None:
        """Set the MO2 roots used for overwrite-first and mod-specific texture lookup."""
        self.overwrite_root = Path(overwrite_root).resolve() if overwrite_root else None
        self._overwrite_indexed = False
        self._overwrite_path_index.clear()
        self.extra_roots.clear()
        self._mod_subdirs_cache.clear()
        if mods_root:
            self.add_search_root(mods_root)
        self._nif_owners.clear()
        self._path_cache.clear()
        self._root_path_cache.clear()

    def register_nif_owner(
        self, nif_path: Path | str, mod_path: Path | str
    ) -> None:
        """Register the owning mod for a NIF physically stored outside its mod folder."""
        self._nif_owners[Path(nif_path).resolve()] = Path(mod_path).resolve()
        self._path_cache.clear()

    def add_search_root(self, root: Path | str) -> None:
        """Register an additional root directory (e.g. selected mod folder or Data folder)."""
        p = Path(root).resolve()
        if p.is_dir() and p not in self.extra_roots:
            self.extra_roots.append(p)
            self._path_cache.clear()
            self._root_path_cache.clear()

    def clear_cache(self) -> None:
        self._path_cache.clear()
        self._root_path_cache.clear()
        self._image_cache.clear()
        self._mod_subdirs_cache.clear()
        self._overwrite_path_index.clear()
        self._overwrite_indexed = False

    def _index_overwrite_textures(self) -> None:
        if self._overwrite_indexed:
            return
        self._overwrite_indexed = True
        if not self.overwrite_root:
            return

        search_roots = (
            self.overwrite_root / "textures",
            self.overwrite_root / "SKSE" / "Plugins" / "CharGen",
        )
        for search_root in search_roots:
            if not search_root.is_dir():
                continue
            for root, dirs, files in os.walk(
                search_root, onerror=self._log_overwrite_walk_error
            ):
                dirs.sort(key=str.casefold)
                root_path = Path(root)
                for file_name in sorted(files, key=str.casefold):
                    if not file_name.casefold().endswith(".dds"):
                        continue
                    path = root_path / file_name
                    try:
                        key = path.relative_to(self.overwrite_root).as_posix().lower()
                    except ValueError:
                        continue
                    self._overwrite_path_index.setdefault(key, path.resolve())

    @staticmethod
    def _log_overwrite_walk_error(error: OSError) -> None:
        log.warning("Could not index overwrite texture directory %s: %s", error.filename, error)

    def resolve_path(
        self, raw_nif_path: str, nif_file_path: Optional[Path | str] = None
    ) -> Optional[Path]:
        """
        Locate the `.dds` file on disk for `raw_nif_path`.

        Search order:
        1. MO2 overwrite root.
        2. The NIF's owning mod.
        3. Nearby NIF directories, then registered roots and their mod folders.
        """
        norm_rel = normalize_nif_texture_path(raw_nif_path)
        if not norm_rel:
            return None

        nif_path = Path(nif_file_path).resolve() if nif_file_path else None
        nif_dir_key = str(nif_path) if nif_path else ""
        cache_key = (nif_dir_key, norm_rel.lower())
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        if self.overwrite_root:
            self._index_overwrite_textures()
            overwrite_key = norm_rel.replace("\\", "/").lower()
            found = self._overwrite_path_index.get(overwrite_key)
            if found:
                self._path_cache[cache_key] = found
                return found

        matched_mod = self._nif_owners.get(nif_path) if nif_path else None
        if matched_mod is None and nif_path:
            for root in self.extra_roots:
                if root.name.casefold() != "mods":
                    continue
                try:
                    mod_name = nif_path.relative_to(root).parts[0]
                except (ValueError, IndexError):
                    continue
                candidate_mod = root / mod_name
                if candidate_mod.is_dir():
                    matched_mod = candidate_mod
                    break
        if matched_mod:
            found = _resolve_case_insensitive(matched_mod, norm_rel)
            if found:
                self._path_cache[cache_key] = found
                return found

        sub_after_textures = (
            norm_rel[len("textures/") :]
            if norm_rel.lower().startswith("textures/")
            else norm_rel
        )

        # 1. Search upward from the NIF file's directory
        if nif_path:
            nif_dir = nif_path.parent
            current = nif_dir
            for _ in range(9):
                # Check <current>/textures/...
                found = _resolve_case_insensitive(current, norm_rel)
                if found:
                    self._path_cache[cache_key] = found
                    return found

                # Check <current>/Data/textures/...
                found_data = _resolve_case_insensitive(current / "Data", norm_rel)
                if found_data:
                    self._path_cache[cache_key] = found_data
                    return found_data

                # If current itself is named "textures", check <current>/<sub_after_textures>
                if current.name.lower() == "textures":
                    found_in_tex = _resolve_case_insensitive(current, sub_after_textures)
                    if found_in_tex:
                        self._path_cache[cache_key] = found_in_tex
                        return found_in_tex

                parent = current.parent
                if parent == current:
                    break
                current = parent

            # 2. Check if the DDS sits right next to the NIF file
            dds_name = Path(norm_rel).name
            same_dir_match = _resolve_case_insensitive(nif_dir, dds_name)
            if same_dir_match:
                self._path_cache[cache_key] = same_dir_match
                return same_dir_match

        root_cache_key = norm_rel.lower()
        found = self._root_path_cache.get(root_cache_key)
        if found is None:
            found = self._resolve_from_roots(norm_rel)
            if found is not None:
                self._root_path_cache[root_cache_key] = found
        self._path_cache[cache_key] = found
        return found

    def _resolve_from_roots(self, norm_rel: str) -> Optional[Path]:
        """Resolve a shared texture once across roots, caching by normalized path."""
        for root in self.extra_roots:
            found = _resolve_case_insensitive(root, norm_rel)
            if found:
                return found

            found = _resolve_case_insensitive(root / "Data", norm_rel)
            if found:
                return found

            for mod_dir in self._get_mod_texture_subdirs(root):
                found = _resolve_case_insensitive(mod_dir, norm_rel)
                if found:
                    return found
        return None

    def _get_mod_texture_subdirs(self, root: Path) -> List[Path]:
        """Cache child folders of root that contain a textures directory."""
        if root in self._mod_subdirs_cache:
            return self._mod_subdirs_cache[root]
        res: List[Path] = []
        try:
            for entry in os.scandir(root):
                if entry.is_dir():
                    p = Path(entry.path)
                    if (p / "textures").is_dir():
                        res.append(p)
        except OSError:
            pass
        self._mod_subdirs_cache[root] = res
        return res

    def decode_dds_file(self, dds_path: Path) -> np.ndarray:
        """
        Decode a `.dds` file (BC1/BC2/BC3/BC4/BC5/BC7 or uncompressed) into an RGBA8
        numpy array of shape (H, W, 4).
        """
        dds_path = dds_path.resolve()
        if dds_path in self._image_cache:
            return self._image_cache[dds_path]

        raw_bytes = dds_path.read_bytes()
        rgba: Optional[np.ndarray] = None

        # Fast path: uncompressed 32-bit RGBA/BGRA DDS (avoids Pillow's 30+ second Python loop on 4K textures)
        if len(raw_bytes) > 128 and raw_bytes[:4] == b"DDS ":
            import struct
            h, w = struct.unpack("<II", raw_bytes[12:20])
            fourcc = raw_bytes[84:88]
            bit_count = struct.unpack("<I", raw_bytes[88:92])[0]
            if fourcc == b"\x00\x00\x00\x00" and bit_count == 32 and w > 0 and h > 0:
                expected_len = 128 + w * h * 4
                if len(raw_bytes) >= expected_len:
                    r_mask = struct.unpack("<I", raw_bytes[92:96])[0]
                    raw_grid = np.frombuffer(raw_bytes[128:expected_len], dtype=np.uint8).reshape((h, w, 4))
                    if r_mask == 0x00FF0000:  # BGRA -> RGBA
                        rgba = raw_grid[:, :, [2, 1, 0, 3]].copy()
                    else:
                        rgba = raw_grid.copy()

        # Primary decoder for BC1-BC7 block-compressed DDS (using imagecodecs)
        if rgba is None:
            try:
                decoded = imagecodecs.dds_decode(raw_bytes)
                rgba = _ensure_rgba8(decoded)
            except Exception as exc_ic:
                # Secondary fallback decoder: Pillow DDS plugin
                try:
                    with Image.open(io.BytesIO(raw_bytes)) as img:
                        rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
                except Exception as exc_pil:
                    raise RuntimeError(
                        f"Failed to decode DDS '{dds_path}': imagecodecs={exc_ic}; Pillow={exc_pil}"
                    ) from exc_pil

        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        self._image_cache[dds_path] = rgba
        return rgba

    def get_fallback_texture(self, slot: str, size: int = 64) -> np.ndarray:
        """
        Generate a procedural fallback RGBA8 texture when a file is not present on disk.
        """
        key = f"{slot}:{size}"
        if key in self._fallback_cache:
            return self._fallback_cache[key]

        arr = np.zeros((size, size, 4), dtype=np.uint8)
        if slot == "Diffuse":
            # Missing diffuse maps should reveal the scene behind the mesh.
            arr[:, :] = (0, 0, 0, 0)
        elif slot == "Normal":
            # Flat tangent-space normal (0, 0, 1) -> (128, 128, 255)
            # Alpha = 64 (~0.25 specular strength in Skyrim's normal map alpha channel)
            arr[:, :] = (128, 128, 255, 64)
        elif slot in ("EnvMask", "EnvMapMask"):
            # Red = env reflection mask, Green = glossiness, Blue = metallic
            arr[:, :] = (128, 0, 0, 255)
        elif slot == "Glow":
            arr[:, :] = (0, 0, 0, 255)
        elif slot == "Specular":
            arr[:, :] = (64, 64, 64, 255)
        else:
            arr[:, :] = (255, 255, 255, 255)

        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        self._fallback_cache[key] = arr
        return arr

    def load_or_fallback(
        self, slot: str, raw_nif_path: str, nif_file_path: Optional[Path | str] = None
    ) -> ResolvedTexture:
        """
        Resolve `raw_nif_path` and decode it; if missing or unreadable, return a graceful
        fallback `ResolvedTexture` without raising.
        """
        resolved = self.resolve_path(raw_nif_path, nif_file_path) if raw_nif_path else None
        if resolved is not None:
            try:
                rgba = self.decode_dds_file(resolved)
                return ResolvedTexture(
                    slot=slot,
                    raw_nif_path=raw_nif_path,
                    resolved_path=resolved,
                    is_fallback=False,
                    width=int(rgba.shape[1]),
                    height=int(rgba.shape[0]),
                    rgba_data=rgba,
                )
            except Exception as exc:
                log.warning("Fallback used for %s (%s): %s", slot, resolved, exc)

        fallback_rgba = self.get_fallback_texture(slot)
        return ResolvedTexture(
            slot=slot,
            raw_nif_path=raw_nif_path,
            resolved_path=None,
            is_fallback=True,
            width=int(fallback_rgba.shape[1]),
            height=int(fallback_rgba.shape[0]),
            rgba_data=fallback_rgba,
        )

    def load_from_file(
        self,
        slot: str,
        dds_path: Path | str,
        raw_nif_path: str = "",
    ) -> ResolvedTexture:
        """
        Decode an explicit `.dds` file on disk for `slot` and return a `ResolvedTexture`.
        """
        resolved = Path(dds_path).resolve()
        rgba = self.decode_dds_file(resolved)
        return ResolvedTexture(
            slot=slot,
            raw_nif_path=raw_nif_path or resolved.name,
            resolved_path=resolved,
            is_fallback=False,
            width=int(rgba.shape[1]),
            height=int(rgba.shape[0]),
            rgba_data=rgba,
        )

    @staticmethod
    def generate_studio_cubemap(face_size: int = 64) -> List[np.ndarray]:
        """
        Generate 6 RGBA8 cubemap faces (+X, -X, +Y, -Y, +Z, -Z) simulating a soft studio
        environment map for Skyrim metallic/environment-mapped armor pieces.
        """
        faces: List[np.ndarray] = []
        lin = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
        u, v = np.meshgrid(lin, -lin)

        # Standard OpenGL cubemap face direction vectors
        face_dirs = [
            (np.ones_like(u), v, -u),   # +X
            (-np.ones_like(u), v, u),   # -X
            (u, np.ones_like(u), -v),   # +Y
            (u, -np.ones_like(u), v),   # -Y
            (u, v, np.ones_like(u)),    # +Z
            (-u, v, -np.ones_like(u)),  # -Z
        ]

        sky_color = np.array([0.82, 0.86, 0.94], dtype=np.float32)
        horizon_color = np.array([0.55, 0.56, 0.60], dtype=np.float32)
        ground_color = np.array([0.18, 0.17, 0.20], dtype=np.float32)
        key_dir = np.array([0.45, 0.75, 0.48], dtype=np.float32)
        key_dir /= np.linalg.norm(key_dir)

        for dx, dy, dz in face_dirs:
            norm = np.sqrt(dx * dx + dy * dy + dz * dz)
            nx, ny, nz = dx / norm, dy / norm, dz / norm

            # Vertical gradient (ny in [-1, 1])
            up_t = np.clip(ny, 0.0, 1.0)[..., None]
            down_t = np.clip(-ny, 0.0, 1.0)[..., None]
            col = (
                horizon_color * (1.0 - up_t) + sky_color * up_t
            ) * (1.0 - down_t) + ground_color * down_t

            # Soft studio specular highlight lobe
            dot_key = np.clip(nx * key_dir[0] + ny * key_dir[1] + nz * key_dir[2], 0.0, 1.0)
            highlight = (dot_key ** 6)[..., None] * np.array([0.35, 0.34, 0.30], dtype=np.float32)
            col = np.clip(col + highlight, 0.0, 1.0)

            rgba = np.empty((face_size, face_size, 4), dtype=np.uint8)
            rgba[..., :3] = (col * 255.0).astype(np.uint8)
            rgba[..., 3] = 255
            faces.append(np.ascontiguousarray(rgba))

        return faces


def _ensure_rgba8(arr: np.ndarray) -> np.ndarray:
    """Convert a decoded image array of arbitrary channel count / dtype into uint8 RGBA."""
    # If cubemap/volume/mipmap stack has extra leading dimensions, take top slice
    while arr.ndim > 3:
        arr = arr[0]

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        elif arr.dtype == np.uint16:
            arr = (arr >> 8).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        h, w = arr.shape
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., 0] = arr
        out[..., 1] = arr
        out[..., 2] = arr
        out[..., 3] = 255
        return out

    h, w, c = arr.shape
    if c == 4:
        return arr
    if c == 3:
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = arr
        out[..., 3] = 255
        return out
    if c == 2:
        # BC5 (RG normal map): reconstruct B in shader or set B=255, A=255
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., 0] = arr[..., 0]
        out[..., 1] = arr[..., 1]
        out[..., 2] = 255
        out[..., 3] = 255
        return out
    if c == 1:
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., 0] = arr[..., 0]
        out[..., 1] = arr[..., 0]
        out[..., 2] = arr[..., 0]
        out[..., 3] = 255
        return out

    raise ValueError(f"Unsupported channel count in DDS array: {arr.shape}")
