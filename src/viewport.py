"""
3D OpenGL Viewport Widget for Dibella's Wardrobe.

Embeds a ModernGL 3.3 Core context inside PySide6's `QOpenGLWidget`, compiles Outfit Studio's
`default.frag` + `default.vert` GLSL shaders, manages GPU buffers/textures in VRAM as `.nif`
files are checked/unchecked, and provides an interactive Z-up orbit/pan/zoom camera.
"""

from __future__ import annotations

import logging
import math
import struct
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import glm
import moderngl
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .anim_engine import AnimEngine, LoadedAnimation
from .cbpc_manager import (
    SOFTNESS_MAX,
    SOFTNESS_MIN,
    GRAVITY_MAX,
    GRAVITY_MIN,
    CbpcManager,
)
from .cbpc_overlay import CbpcOverlay
from .nif_loader import MAX_SHADER_BONES, LoadedNifData, ShapeMeshData
from .shader_presets import (
    NO_POSTPROCESS,
    PostProcessProfile,
    UniformValue,
    load_postprocess_profile,
    load_preset_label,
    load_preset_uniforms,
)
from .smp_bridge import SmpBridgeSolver
from .smaa import DEFAULT_SMAA_QUALITY, SMAA_QUALITIES, SmaaPostProcess
from .texture_resolver import ResolvedTexture, TextureResolver

log = logging.getLogger("dibellas_wardrobe.viewport")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SHADER_DIR = _REPO_ROOT / "shaders"


@dataclass
class GpuShapeItem:
    """VRAM resources and material metadata for a single rendered NIF shape."""

    source_key: str
    shape_data: ShapeMeshData
    vbo: moderngl.Buffer
    ibo: moderngl.Buffer
    vao: moderngl.VertexArray
    textures: Dict[str, moderngl.Texture] = field(default_factory=dict)
    mapped_bone_names: List[str] = field(default_factory=list)
    effective_skin_matrices: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4, 4), dtype=np.float32)
    )

    def release(self) -> None:
        """Release VAO, VBO, and IBO from GPU VRAM."""
        for obj in (self.vao, self.vbo, self.ibo):
            try:
                obj.release()
            except Exception:
                pass


@dataclass
class ViewportRenderSettings:
    """Real-time lighting and shader toggles controlled by the UI."""

    shader_preset: str = "none"
    light_azimuth_deg: float = 110.0
    light_elevation_deg: float = 28.0
    light_intensity: float = 0.85
    ambient_intensity: float = 0.45
    specular_multiplier: float = 1.0
    env_reflection_multiplier: float = 1.0

    enable_lighting: bool = True
    show_textures: bool = True
    enable_normal_map: bool = True
    enable_specular: bool = True
    enable_cubemap: bool = True
    wireframe_overlay: bool = False
    show_ground_grid: bool = True

    enable_smaa: bool = True
    smaa_quality: str = DEFAULT_SMAA_QUALITY


class OutfitViewportWidget(QOpenGLWidget):
    """
    Real-time OpenGL 3.3 viewport rendering stacked Skyrim `.nif` meshes with
    Outfit Studio's `default.frag` shader.
    """

    statsChanged = Signal(int, int, int)  # (nif_count, shape_count, triangle_count)
    shaderPresetError = Signal(str)
    animationTimeChanged = Signal(float, float)  # (current_time, duration)
    animationStateChanged = Signal(bool)  # (is_playing)
    animationStopped = Signal()
    smpStateChanged = Signal(bool, bool, bool)  # (enabled, simulating, native library available)
    BASE_BODY_KEY = "__base_body__"

    def __init__(self, parent=None) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSamples(4)
        fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        super().__init__(parent)
        self.setFormat(fmt)
        self.setMinimumSize(520, 520)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.ctx: Optional[moderngl.Context] = None
        self.prog: Optional[moderngl.Program] = None
        self.bone_ubo: Optional[moderngl.Buffer] = None
        self._bone_palette_scratch = np.repeat(
            np.eye(4, dtype=np.float32)[None, :, :], MAX_SHADER_BONES, axis=0
        )
        self.post_prog: Optional[moderngl.Program] = None
        self.post_vao: Optional[moderngl.VertexArray] = None
        # GTAO/SSSS program variants keyed by
        # (enable_gtao, enable_ssss, ssss_after_preset).
        self._post_variants: Dict[
            Tuple[bool, bool, bool], Tuple[moderngl.Program, moderngl.VertexArray]
        ] = {}
        self._failed_post_variants: set[Tuple[bool, bool, bool]] = set()
        self.preset_prog: Optional[moderngl.Program] = None
        self.preset_vao: Optional[moderngl.VertexArray] = None
        self._loaded_shader_preset = "none"
        self.postprocess_profile: PostProcessProfile = NO_POSTPROCESS
        self.preset_uniforms: Dict[str, UniformValue] = {}
        # View-space key light direction for presets; zero while lighting is off.
        self._key_light_view = glm.vec3(0.0)
        # Rotates view-space directions into the studio cubemap's world space.
        self._view_to_world = glm.mat3(1.0)
        self.shader_presets = self._discover_shader_presets()
        self.scene_color: Optional[moderngl.Texture] = None
        self.scene_depth: Optional[moderngl.Texture] = None
        self.scene_normal: Optional[moderngl.Texture] = None
        self.scene_fbo: Optional[moderngl.Framebuffer] = None
        self.material_fbo: Optional[moderngl.Framebuffer] = None
        self.material_clear_fbo: Optional[moderngl.Framebuffer] = None
        self.material_textures: List[moderngl.Texture] = []
        self.preset_color: Optional[moderngl.Texture] = None
        self.preset_fbo: Optional[moderngl.Framebuffer] = None
        # Diffuse and specular + SSSS mask written by "ssss_after_preset" presets.
        self.split_diffuse: Optional[moderngl.Texture] = None
        self.split_specular: Optional[moderngl.Texture] = None
        self.split_fbo: Optional[moderngl.Framebuffer] = None
        self.smaa: Optional[SmaaPostProcess] = None
        self._smaa_failed_qualities: set[str] = set()
        self._scene_target_size: Tuple[int, int] = (0, 0)
        self.grid_prog: Optional[moderngl.Program] = None
        self.grid_vao: Optional[moderngl.VertexArray] = None
        self.grid_vbo: Optional[moderngl.Buffer] = None
        self.grid_vertex_count: int = 0

        self.default_cubemap: Optional[moderngl.TextureCube] = None
        self.fallback_2d_textures: Dict[str, moderngl.Texture] = {}
        self._shared_tex_cache: Dict[Path, Tuple[moderngl.Texture, int]] = {}

        # Pending CPU LoadedNifData waiting for GL context, and active GPU shapes
        self._pending_nifs: Dict[str, LoadedNifData] = {}
        self._gpu_nifs: Dict[str, List[GpuShapeItem]] = {}
        self._nif_visibility: Dict[str, bool] = {}
        self._base_body_visible: bool = True

        self.settings = ViewportRenderSettings()

        # Headless HKX Animation & Skeleton Engine
        self.anim_engine = AnimEngine()
        self._anim_playing: bool = False
        self._anim_current_time: float = 0.0
        self._anim_last_tick: float = 0.0
        self._current_pose_matrices: Optional[Dict[str, np.ndarray]] = None

        # CBPC body physics (pure NumPy) and its settings controls
        self.cbpc = CbpcManager(self.anim_engine)
        self._nif_bone_names: Dict[str, Tuple[str, ...]] = {}
        self.cbpc_overlay = CbpcOverlay(
            self, GRAVITY_MIN, GRAVITY_MAX, (SOFTNESS_MIN, SOFTNESS_MAX)
        )
        self.cbpc_overlay.hide()
        self.cbpc_overlay.enabledChanged.connect(self.set_cbpc_enabled)
        self.cbpc_overlay.smpEnabledChanged.connect(self.set_smp_enabled)
        self.cbpc_overlay.gravityChanged.connect(self.set_cbpc_gravity)
        self.cbpc_overlay.softnessChanged.connect(self.set_cbpc_softness)
        self._sync_cbpc_overlay()

        # HDT-SMP cloth physics (Native AOT BulletSharp library driven through ctypes)
        self.smp = SmpBridgeSolver(self.anim_engine, self.cbpc)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)  # ~60 FPS render/tick loop
        self._anim_timer.timeout.connect(self._on_animation_tick)

        # Orbit camera state (Skyrim Z-up coordinate system: feet at Z=0, head at Z~120)
        self.cam_target = glm.vec3(0.0, 0.0, 62.0)
        self.cam_yaw_deg: float = 90.0
        self.cam_pitch_deg: float = 8.0
        self.cam_distance: float = 185.0
        self.cam_fov_deg: float = 42.0

        self._last_mouse_pos = QPointF()
        self._active_mouse_button = Qt.MouseButton.NoButton

    # ------------------------------------------------------------------
    # Public Scene & Mesh API
    # ------------------------------------------------------------------

    def set_base_body(self, nif_data: Optional[LoadedNifData], visible: bool = True) -> None:
        """Load or replace the default base mannequin mesh."""
        self._base_body_visible = visible
        if nif_data is None:
            self.remove_nif(self.BASE_BODY_KEY)
            return
        self.add_nif(self.BASE_BODY_KEY, nif_data)

    def set_base_body_visible(self, visible: bool) -> None:
        """Toggle visibility of the base mannequin without unloading it."""
        self._base_body_visible = bool(visible)
        self._refresh_cbpc_sources()
        self._emit_stats()
        self.update()

    def is_base_body_visible(self) -> bool:
        return self._base_body_visible and (
            self.BASE_BODY_KEY in self._gpu_nifs or self.BASE_BODY_KEY in self._pending_nifs
        )

    def add_nif(self, key: str, nif_data: LoadedNifData) -> None:
        """
        Upload all shapes of `nif_data` to GPU VRAM under `key` (or queue if GL is not yet initialized).
        """
        self._nif_visibility.setdefault(key, True)
        self._nif_bone_names[key] = tuple(
            dict.fromkeys(b for sh in nif_data.shapes if sh.is_skinned for b in sh.bone_names)
        )
        self.smp.add_source(key, nif_data.filepath, nif_data.smp_xml_ref, nif_data.nodes, nif_data.shapes)
        self._refresh_cbpc_sources()
        if self.ctx is None:
            self._pending_nifs[key] = nif_data
            return

        self.makeCurrent()
        try:
            if key in self._gpu_nifs:
                self._release_nif_internal(key)
            gpu_items = [self._upload_shape(key, sh) for sh in nif_data.shapes]
            self._gpu_nifs[key] = gpu_items
        finally:
            self.doneCurrent()

        self._emit_stats()
        self.update()

    def remove_nif(self, key: str) -> None:
        """Remove a `.nif` and release its GPU buffers/textures from VRAM."""
        self._pending_nifs.pop(key, None)
        self._nif_visibility.pop(key, None)
        self._nif_bone_names.pop(key, None)
        self.smp.remove_source(key)
        self._refresh_cbpc_sources()
        if self.ctx is not None and key in self._gpu_nifs:
            self.makeCurrent()
            try:
                self._release_nif_internal(key)
            finally:
                self.doneCurrent()
        self._emit_stats()
        self.update()

    def set_nif_visible(self, key: str, visible: bool) -> None:
        """Toggle a loaded NIF's visibility without releasing its scene resources."""
        self._nif_visibility[key] = bool(visible)
        self._refresh_cbpc_sources()
        self._emit_stats()
        self.update()

    def is_nif_visible(self, key: str) -> bool:
        """Return whether a loaded NIF is visible."""
        return self._nif_visibility.get(key, True)

    def update_nif_shape_texture(
        self,
        key: str,
        shape_index: int,
        slot: str,
        res_tex: ResolvedTexture,
    ) -> None:
        """Replace a texture slot on a loaded NIF shape in place without re-uploading geometry."""
        if key in self._pending_nifs:
            shapes = self._pending_nifs[key].shapes
            if 0 <= shape_index < len(shapes):
                mat = shapes[shape_index].shader
                mat.resolved_textures[slot] = res_tex
                self._sync_slot_flags(mat, slot, res_tex)
            return

        gpu_items = self._gpu_nifs.get(key)
        if not gpu_items or not (0 <= shape_index < len(gpu_items)):
            return

        item = gpu_items[shape_index]
        mat = item.shape_data.shader
        old_res_tex = mat.resolved_textures.get(slot)
        mat.resolved_textures[slot] = res_tex
        self._sync_slot_flags(mat, slot, res_tex)

        if self.ctx is not None and slot != "EnvMap":
            self.makeCurrent()
            try:
                old_gpu_tex = item.textures.pop(slot, None)
                if old_gpu_tex is not None:
                    self._release_2d_texture(old_res_tex, old_gpu_tex)
                item.textures[slot] = self._acquire_2d_texture(res_tex)
            finally:
                self.doneCurrent()
        self.update()

    def reset_nif_textures(self, key: str, nif_data: Optional[LoadedNifData] = None) -> None:
        """Restore all shapes of a loaded NIF back to their original default textures."""
        if key in self._pending_nifs:
            for sh in self._pending_nifs[key].shapes:
                self._restore_shader_defaults(sh.shader)
            return

        gpu_items = self._gpu_nifs.get(key)
        if not gpu_items:
            if nif_data is not None:
                for sh in nif_data.shapes:
                    self._restore_shader_defaults(sh.shader)
            return

        if self.ctx is not None:
            self.makeCurrent()
            try:
                for item in gpu_items:
                    mat = item.shape_data.shader
                    for slot, tex in list(item.textures.items()):
                        old_res = mat.resolved_textures.get(slot)
                        self._release_2d_texture(old_res, tex)
                    item.textures.clear()
                    self._restore_shader_defaults(mat)
                    for slot, res_tex in mat.resolved_textures.items():
                        if slot == "EnvMap":
                            continue
                        item.textures[slot] = self._acquire_2d_texture(res_tex)
            finally:
                self.doneCurrent()
        else:
            for item in gpu_items:
                self._restore_shader_defaults(item.shape_data.shader)
        self.update()

    @staticmethod
    def _sync_slot_flags(
        mat,
        slot: str,
        res_tex: ResolvedTexture,
    ) -> None:
        has_tex = not res_tex.is_fallback
        if slot == "EnvMask":
            mat.b_env_mask = has_tex
            if has_tex:
                mat.b_cubemap = True
        elif slot == "Glow":
            mat.b_glowmap = has_tex
            if has_tex:
                mat.b_emissive = True

    @staticmethod
    def _restore_shader_defaults(mat) -> None:
        if mat.default_resolved_textures:
            mat.resolved_textures = dict(mat.default_resolved_textures)
        for flag_name, flag_val in mat.default_flags.items():
            setattr(mat, flag_name, flag_val)

    def clear_outfit_nifs(self) -> None:
        """Remove all loaded outfit `.nif` meshes while keeping the base mannequin."""
        keys = [
            k
            for k in set(self._gpu_nifs) | set(self._pending_nifs)
            if k != self.BASE_BODY_KEY
        ]
        for k in keys:
            self.remove_nif(k)

    def reset_camera(self) -> None:
        """Reset orbit camera to default full-body view."""
        self.cam_target = glm.vec3(0.0, 0.0, 62.0)
        self.cam_yaw_deg = 90.0
        self.cam_pitch_deg = 8.0
        self.cam_distance = 185.0
        self.update()

    # ------------------------------------------------------------------
    # Public HKX Animation Playback API
    # ------------------------------------------------------------------

    @property
    def active_animation(self) -> Optional[LoadedAnimation]:
        return self.anim_engine.active_animation

    @property
    def is_animation_playing(self) -> bool:
        return self._anim_playing

    @property
    def animation_time(self) -> float:
        return self._anim_current_time

    @property
    def animation_duration(self) -> float:
        anim = self.anim_engine.active_animation
        return anim.duration if anim is not None else 0.0

    def load_animation(self, hkx_path: Path | str, auto_play: bool = True) -> LoadedAnimation:
        """Load an `.hkx` animation clip, reset time to 0.0, and optionally start playback."""
        loaded = self.anim_engine.load_animation(hkx_path)
        self._anim_current_time = 0.0
        self._anim_last_tick = time.perf_counter()
        self._update_pose(0.0, 0.0, reset=True)
        if auto_play:
            self._anim_playing = True
            self._anim_timer.start()
        else:
            self._anim_playing = False
            self._anim_timer.stop()
        self.animationStateChanged.emit(self._anim_playing)
        self.animationTimeChanged.emit(self._anim_current_time, loaded.duration)
        self.update()
        return loaded

    def play_animation(self) -> None:
        """Resume or start playback of the loaded `.hkx` animation."""
        if self.anim_engine.active_animation is None:
            return
        if (
            self._anim_current_time >= self.animation_duration
            and self.animation_duration > 0.0
        ):
            self._anim_current_time = 0.0
        self._anim_playing = True
        self._anim_last_tick = time.perf_counter()
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.animationStateChanged.emit(self._anim_playing)
        self.update()

    def pause_animation(self) -> None:
        """Pause the active `.hkx` animation at the current frame."""
        if not self._anim_playing:
            return
        self._anim_playing = False
        self._anim_timer.stop()
        self.animationStateChanged.emit(self._anim_playing)
        self.update()

    def toggle_play_pause(self) -> bool:
        """Toggle between Play and Pause; returns the new `is_playing` state."""
        if self._anim_playing:
            self.pause_animation()
        else:
            self.play_animation()
        return self._anim_playing

    def seek_animation(self, time_sec: float) -> None:
        """Scrub/seek the active animation to `time_sec` seconds and refresh the viewport."""
        anim = self.anim_engine.active_animation
        if anim is None:
            return
        self._anim_current_time = max(0.0, min(float(time_sec), anim.duration))
        self._anim_last_tick = time.perf_counter()
        self._update_pose(self._anim_current_time, 0.0, reset=True)
        self.animationTimeChanged.emit(self._anim_current_time, anim.duration)
        self.update()

    def stop_animation(self) -> None:
        """
        Stop playback, unload the active animation, reset the skeleton and attached
        meshes back to the default bind pose, and emit `animationStopped`.
        """
        self._anim_playing = False
        self._anim_timer.stop()
        self._anim_current_time = 0.0
        self.anim_engine.clear_animation()
        self._current_pose_matrices = None
        self._update_static_pose()
        self.animationStateChanged.emit(False)
        self.animationTimeChanged.emit(0.0, 0.0)
        self.animationStopped.emit()
        self.update()

    def _on_animation_tick(self) -> None:
        anim = self.anim_engine.active_animation
        if anim is None or not self._anim_playing:
            self._anim_timer.stop()
            return

        now = time.perf_counter()
        delta_time = max(0.0, min(0.1, now - self._anim_last_tick))
        self._anim_last_tick = now

        looped = False
        if anim.duration > 0.0:
            next_time = self._anim_current_time + delta_time
            if next_time >= anim.duration:
                next_time = math.fmod(next_time, anim.duration)
                looped = True
            self._anim_current_time = next_time
        else:
            self._anim_current_time = 0.0

        self._update_pose(self._anim_current_time, delta_time, reset=looped)
        self.animationTimeChanged.emit(self._anim_current_time, anim.duration)
        self.update()

    def _update_static_pose(self) -> None:
        """Without an animation, run CBPC and SMP on the bind pose and let them settle (gravity sag)."""
        if self.anim_engine.active_animation is not None:
            return
        if not self.cbpc.is_active and not self.smp.is_active:
            self._current_pose_matrices = None
            return
        pose = self.anim_engine.evaluate_pose(0.0)
        self.cbpc.reset(pose)
        self.smp.reset()
        palette = pose
        for _ in range(60):
            palette = self.smp.update(dict(self.cbpc.update(pose, 1.0 / 30.0)), 1.0 / 60.0, pose)
        self._current_pose_matrices = palette

    def _update_pose(self, time_sec: float, dt: float, reset: bool = False) -> None:
        """Evaluate the animated pose, blend CBPC deltas into it, then merge SMP cloth matrices."""
        pose = self.anim_engine.evaluate_pose(time_sec)
        if reset:
            self.cbpc.reset(pose)
            self.smp.reset()
        self._current_pose_matrices = self.smp.update(self.cbpc.update(pose, dt), dt, pose)

    # ------------------------------------------------------------------
    # CBPC physics control
    # ------------------------------------------------------------------

    def set_cbpc_enabled(self, enabled: bool) -> None:
        self.cbpc.set_enabled(enabled)
        if enabled:
            self._update_pose(self._anim_current_time, 0.0, reset=True)
        self._refresh_pose()

    def set_cbpc_gravity(self, factor: float) -> None:
        """Adjust breast rest position without rebuilding or restarting animation."""
        self.cbpc.set_gravity(factor)
        self._refresh_pose()

    def set_cbpc_softness(self, group: str, factor: float) -> None:
        self.cbpc.set_softness(group, factor)
        self._refresh_pose()

    # ------------------------------------------------------------------
    # SMP cloth physics control
    # ------------------------------------------------------------------

    def set_smp_enabled(self, enabled: bool) -> None:
        """Switch HDT-SMP cloth simulation on or off; cloth falls back to following its skeleton parent."""
        enabled = bool(enabled)
        if enabled == self.smp.enabled:
            self._emit_smp_state()
            return
        self.smp.set_enabled(enabled)
        self.smp.refresh()
        self._refresh_pose(reset=True)

    def _emit_smp_state(self) -> None:
        self.smpStateChanged.emit(self.smp.enabled, self.smp.is_active, self.smp.available)

    def _refresh_pose(self, reset: bool = False) -> None:
        if self.anim_engine.active_animation is not None:
            self._update_pose(self._anim_current_time, 0.0, reset=reset)
        else:
            self._update_static_pose()
        self._sync_cbpc_overlay()
        self._emit_smp_state()
        self.update()

    def _refresh_cbpc_sources(self) -> None:
        """Recompute CBPC bones from the visible meshes (Mesh Bones ∩ CBPC config), then the SMP world."""
        sources = {
            key: names
            for key, names in self._nif_bone_names.items()
            if self._nif_visibility.get(key, True)
            and (key != self.BASE_BODY_KEY or self._base_body_visible)
        }
        before = self.cbpc.active_bones
        self.cbpc.set_mesh_bones_bulk(sources)
        self.smp.set_visible_sources(sources)
        smp_changed = self.smp.refresh()
        if self.cbpc.active_bones != before or smp_changed or self._current_pose_matrices is not None:
            self._refresh_pose(reset=smp_changed)
        else:
            self._sync_cbpc_overlay()
            self._emit_smp_state()

    def _sync_cbpc_overlay(self) -> None:
        self.cbpc_overlay.sync(
            self.cbpc.enabled,
            self.cbpc.has_physics,
            self.cbpc.gravity,
            len(self.cbpc.active_bones),
            self.cbpc.softness,
        )
    def grab_viewport_image(self) -> np.ndarray:
        """Capture the current viewport framebuffer as an RGBA8 numpy array (for verification/export)."""
        qimg = self.grabFramebuffer().convertToFormat(
            self.grabFramebuffer().Format.Format_RGBA8888
        )
        w, h = qimg.width(), qimg.height()
        ptr = qimg.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, qimg.bytesPerLine() // 4, 4))
        return arr[:, :w, :].copy()

    # ------------------------------------------------------------------
    # OpenGL Lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)

        vert_src = (_SHADER_DIR / "default.vert").read_text(encoding="utf-8")
        frag_src = (_SHADER_DIR / "default.frag").read_text(encoding="utf-8")
        self.prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        if "BonePalette" in self.prog:
            self.prog["BonePalette"].binding = 0
        self.bone_ubo = self.ctx.buffer(
            np.ascontiguousarray(self._bone_palette_scratch.swapaxes(1, 2)).tobytes()
        )
        self.bone_ubo.bind_to_uniform_block(0)

        self.post_prog, self.post_vao = self._compile_postprocess_variant(False, False)
        self.smaa = SmaaPostProcess(self.ctx)

        self._init_default_textures()
        self._init_ground_grid()

        # Upload any NIFs that were queued prior to GL context creation
        if self._pending_nifs:
            pending = dict(self._pending_nifs)
            self._pending_nifs.clear()
            for key, nif_data in pending.items():
                self._gpu_nifs[key] = [self._upload_shape(key, sh) for sh in nif_data.shapes]
            self._emit_stats()

    @staticmethod
    def _discover_shader_presets(
        shader_dir: Path = _SHADER_DIR,
    ) -> Tuple[Tuple[str, str], ...]:
        excluded = {"default", "postprocess"}
        presets = []
        for path in sorted(shader_dir.glob("*.frag"), key=lambda item: item.name.casefold()):
            if path.stem.casefold() in excluded:
                continue
            try:
                label = load_preset_label(path)
            except ValueError as exc:
                log.warning("Ignoring label of preset %s: %s", path.stem, exc)
                label = None
            if label is None:
                label = path.stem.replace("_", " ").replace("-", " ").title()
            presets.append((path.stem, label))
        return tuple(presets)

    def resizeGL(self, w: int, h: int) -> None:
        pass

    def paintGL(self) -> None:
        if (
            self.ctx is None
            or self.prog is None
            or self.post_prog is None
            or self.post_vao is None
        ):
            return

        qt_fbo_id = self.defaultFramebufferObject()
        output_fbo = self.ctx.detect_framebuffer(qt_fbo_id)
        dpr = self.devicePixelRatioF()
        vp_w = max(1, int(self.width() * dpr))
        vp_h = max(1, int(self.height() * dpr))
        self.ctx.viewport = (0, 0, vp_w, vp_h)
        self._ensure_scene_target(vp_w, vp_h)
        self._ensure_selected_shader_preset()
        assert self.scene_fbo is not None
        self.scene_fbo.use()
        self.ctx.viewport = (0, 0, vp_w, vp_h)

        # Clear the normal validity channel independently from the scene color.
        self.scene_fbo.color_mask = ((True, True, True, True), (False, False, False, False))
        self.scene_fbo.clear(0.13, 0.14, 0.17, 1.0, depth=1.0)
        self.scene_fbo.color_mask = ((False, False, False, False), (True, True, True, True))
        self.scene_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.scene_fbo.color_mask = ((True, True, True, True), (True, True, True, True))

        aspect = vp_w / float(vp_h)
        proj = glm.perspective(glm.radians(self.cam_fov_deg), aspect, 1.0, 2500.0)
        view = self._compute_view_matrix()
        model = glm.mat4(1.0)
        mv = view * model
        mv_inv = glm.inverse(mv)
        normal_mat = glm.mat3(glm.transpose(glm.inverse(mv)))

        # Collect visible shapes and split into opaque vs alpha-blended passes
        opaque_items: List[GpuShapeItem] = []
        transparent_items: List[GpuShapeItem] = []

        for key, items in self._gpu_nifs.items():
            if key == self.BASE_BODY_KEY and not self._base_body_visible:
                continue
            if not self._nif_visibility.get(key, True):
                continue
            for item in items:
                ap = item.shape_data.alpha_prop
                sh = item.shape_data.shader
                diffuse = sh.resolved_textures.get("Diffuse")
                if diffuse is None or diffuse.is_fallback:
                    continue
                if (
                    (ap.has_alpha_property and ap.alpha_blend)
                    or sh.alpha < 0.999
                ):
                    transparent_items.append(item)
                else:
                    opaque_items.append(item)

        self._setup_global_shader_uniforms(proj, view, model, mv_inv, normal_mat)

        if self._current_pose_matrices is None:
            if self.anim_engine.active_animation is not None:
                self._update_pose(self._anim_current_time, 0.0, reset=True)
            else:
                self._update_static_pose()

        # Pass 1: Opaque & Alpha-Cutout meshes
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.scene_fbo.depth_mask = True
        self.ctx.disable(moderngl.BLEND)
        for item in opaque_items:
            self._render_shape_item(item, wireframe=False)

        # Pass 2: Alpha-Blended meshes
        if transparent_items:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            # Keep depth writes on if alpha test is also enabled (e.g. lace/hair), else depth_mask=False
            for item in transparent_items:
                writes_depth = bool(item.shape_data.alpha_prop.alpha_test)
                self.scene_fbo.depth_mask = writes_depth
                self.scene_fbo.color_mask = (
                    (True, True, True, True),
                    (writes_depth, writes_depth, writes_depth, writes_depth),
                )
                self._render_shape_item(item, wireframe=False)
            self.scene_fbo.depth_mask = True
            self.ctx.disable(moderngl.BLEND)
            self.scene_fbo.color_mask = ((True, True, True, True), (True, True, True, True))

        # Optional Pass 3: Wireframe overlay
        if self.settings.wireframe_overlay:
            self.scene_fbo.color_mask = ((True, True, True, True), (False, False, False, False))
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            for item in opaque_items + transparent_items:
                self._render_shape_item(item, wireframe=True)
            self.ctx.disable(moderngl.BLEND)
            self.scene_fbo.color_mask = ((True, True, True, True), (True, True, True, True))

        smaa_quality = self._prepare_smaa(vp_w, vp_h)
        final_fbo = output_fbo
        if smaa_quality is not None:
            assert self.smaa is not None and self.smaa.input_fbo is not None
            final_fbo = self.smaa.input_fbo

        if self.preset_prog is not None and self.preset_vao is not None:
            if any(name in self.preset_prog for name in (
                "u_material_normal", "u_material_env_mask",
                "u_material_subsurface", "u_material_flags",
            )):
                self._render_preset_materials(opaque_items, transparent_items, vp_w, vp_h)
            profile = self.postprocess_profile
            split = profile.ssss_after_preset
            # A split preset runs GTAO before itself and SSSS after itself.
            pre_profile = (
                replace(profile, enable_ssss=False, ssss_after_preset=False)
                if split else profile
            )
            preset_input = self.scene_color
            if pre_profile.enable_gtao or pre_profile.enable_ssss:
                self._ensure_preset_target(vp_w, vp_h)
                assert self.preset_fbo is not None
                self._composite_postprocess(
                    self.preset_fbo, proj, vp_w, vp_h, pre_profile
                )
                preset_input = self.preset_color
            assert preset_input is not None
            if split:
                self._ensure_split_target(vp_w, vp_h)
                assert self.split_fbo is not None
                # Clear so a preset that skips the specular output adds nothing.
                self.split_fbo.clear(0.0, 0.0, 0.0, 0.0)
                self._composite_shader_preset(
                    self.split_fbo, preset_input, proj, vp_w, vp_h
                )
                self._composite_postprocess(
                    final_fbo, proj, vp_w, vp_h, replace(profile, enable_gtao=False)
                )
            else:
                self._composite_shader_preset(final_fbo, preset_input, proj, vp_w, vp_h)
        else:
            # No preset: present the scene through the copy-only variant.
            self._composite_postprocess(final_fbo, proj, vp_w, vp_h, NO_POSTPROCESS)

        if (
            self.settings.show_ground_grid
            and self.grid_vao is not None
            and self.grid_prog is not None
            and self.scene_depth is not None
        ):
            final_fbo.use()
            self.ctx.viewport = (0, 0, vp_w, vp_h)
            self.ctx.disable(moderngl.DEPTH_TEST)
            final_fbo.depth_mask = False
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.grid_prog["u_mvp"].write(bytes(glm.transpose(proj * view)))
            self.grid_prog["u_viewport_size"].value = (float(vp_w), float(vp_h))
            self.grid_prog["u_depth_test_scene"].value = True
            self.scene_depth.use(location=0)
            self.grid_prog["u_scene_depth"].value = 0
            self.grid_vao.render(moderngl.LINES)
            final_fbo.depth_mask = True
            self.ctx.disable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)

        if smaa_quality is not None:
            assert self.smaa is not None
            self.smaa.apply(output_fbo, smaa_quality, vp_w, vp_h)

    def _prepare_smaa(self, width: int, height: int) -> Optional[str]:
        """Ready SMAA for this frame; returns its quality, or None to render without it."""
        quality = self.settings.smaa_quality
        if quality not in SMAA_QUALITIES:
            quality = DEFAULT_SMAA_QUALITY
        if (
            not self.settings.enable_smaa
            or self.smaa is None
            or quality in self._smaa_failed_qualities
        ):
            return None
        try:
            self.smaa.prepare(quality, width, height)
        except (OSError, ValueError, KeyError, moderngl.Error) as exc:
            # Anti-aliasing is cosmetic: keep drawing the scene without it.
            log.error("Could not initialize SMAA (%s): %s", quality, exc)
            self._smaa_failed_qualities.add(quality)
            return None
        return quality

    def _ensure_scene_target(self, width: int, height: int) -> None:
        assert self.ctx is not None
        if self.scene_fbo is not None and self._scene_target_size == (width, height):
            return

        self._release_scene_target()
        self.scene_color = self.ctx.texture((width, height), 4, dtype="f1")
        self.scene_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_color.repeat_x = False
        self.scene_color.repeat_y = False
        self.scene_depth = self.ctx.depth_texture((width, height))
        self.scene_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.scene_depth.repeat_x = False
        self.scene_depth.repeat_y = False
        self.scene_normal = self.ctx.texture((width, height), 4, dtype="f2")
        self.scene_normal.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.scene_normal.repeat_x = False
        self.scene_normal.repeat_y = False
        self.scene_fbo = self.ctx.framebuffer(
            color_attachments=[self.scene_color, self.scene_normal],
            depth_attachment=self.scene_depth,
        )
        self._scene_target_size = (width, height)

    def _ensure_preset_target(self, width: int, height: int) -> None:
        assert self.ctx is not None
        if self.preset_fbo is not None and self._scene_target_size == (width, height):
            return
        if self.preset_fbo is not None:
            self.preset_fbo.release()
        if self.preset_color is not None:
            self.preset_color.release()
        self.preset_color = self.ctx.texture((width, height), 4, dtype="f1")
        self.preset_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.preset_color.repeat_x = False
        self.preset_color.repeat_y = False
        self.preset_fbo = self.ctx.framebuffer(color_attachments=[self.preset_color])

    def _ensure_split_target(self, width: int, height: int) -> None:
        assert self.ctx is not None
        if self.split_fbo is not None and self._scene_target_size == (width, height):
            return
        for resource in (self.split_fbo, self.split_diffuse, self.split_specular):
            if resource is not None:
                resource.release()
        self.split_diffuse = self.ctx.texture((width, height), 4, dtype="f1")
        self.split_diffuse.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # Half float keeps bright highlights unclipped until the final composite.
        self.split_specular = self.ctx.texture((width, height), 4, dtype="f2")
        self.split_specular.filter = (moderngl.NEAREST, moderngl.NEAREST)
        for texture in (self.split_diffuse, self.split_specular):
            texture.repeat_x = False
            texture.repeat_y = False
        self.split_fbo = self.ctx.framebuffer(
            color_attachments=[self.split_diffuse, self.split_specular]
        )

    def _render_preset_materials(
        self,
        opaque_items: List[GpuShapeItem],
        transparent_items: List[GpuShapeItem],
        width: int,
        height: int,
    ) -> None:
        assert self.ctx is not None and self.scene_depth is not None
        if self.material_fbo is None:
            self.material_textures = [
                self.ctx.texture((width, height), 4, dtype="f1") for _ in range(4)
            ]
            for texture in self.material_textures:
                texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
                texture.repeat_x = False
                texture.repeat_y = False
            self.material_fbo = self.ctx.framebuffer(
                color_attachments=self.material_textures,
                depth_attachment=self.scene_depth,
            )
            self.material_clear_fbo = self.ctx.framebuffer(
                color_attachments=self.material_textures
            )
        assert self.material_clear_fbo is not None
        self.material_clear_fbo.use()
        self.material_clear_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.material_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.material_fbo.depth_mask = False
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "=="
        self._set_uniform("bPresetMaterialPass", True)
        try:
            for item in opaque_items:
                self._render_shape_item(item, wireframe=False)
            for item in transparent_items:
                if item.shape_data.alpha_prop.alpha_test:
                    self._render_shape_item(item, wireframe=False)
        finally:
            self._set_uniform("bPresetMaterialPass", False)
            self.ctx.depth_func = "<"
            self.material_fbo.depth_mask = True

    def _release_scene_target(self) -> None:
        for resource in (
            self.material_fbo,
            self.material_clear_fbo,
            *self.material_textures,
            self.scene_fbo,
            self.scene_color,
            self.scene_depth,
            self.scene_normal,
            self.preset_fbo,
            self.preset_color,
            self.split_fbo,
            self.split_diffuse,
            self.split_specular,
        ):
            if resource is not None:
                resource.release()
        self.scene_fbo = None
        self.material_fbo = None
        self.material_clear_fbo = None
        self.material_textures = []
        self.scene_color = None
        self.scene_depth = None
        self.scene_normal = None
        self.preset_fbo = None
        self.preset_color = None
        self.split_fbo = None
        self.split_diffuse = None
        self.split_specular = None
        self._scene_target_size = (0, 0)
        if self.smaa is not None:
            self.smaa.release_targets()

    def _ensure_selected_shader_preset(self) -> None:
        if self.ctx is None or self._loaded_shader_preset == self.settings.shader_preset:
            return

        if self.preset_vao is not None:
            self.preset_vao.release()
            self.preset_vao = None
        if self.preset_prog is not None:
            self.preset_prog.release()
            self.preset_prog = None
        self._loaded_shader_preset = "none"
        self.postprocess_profile = NO_POSTPROCESS
        self.preset_uniforms = {}

        selected = self.settings.shader_preset
        if selected == "none":
            return
        shader_path = next(
            (path for path, _label in self.shader_presets if path == selected),
            None,
        )
        if shader_path is None:
            self._report_shader_preset_error(
                selected, "The selected shader preset is not available."
            )
            return

        try:
            vertex_source = (_SHADER_DIR / "postprocess.vert").read_text(
                encoding="utf-8"
            )
            fragment_source = (_SHADER_DIR / f"{shader_path}.frag").read_text(
                encoding="utf-8"
            )
            program = self.ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source,
            )
            vertex_array = self.ctx.vertex_array(program, [])
        except (OSError, moderngl.Error) as exc:
            self._report_shader_preset_error(selected, str(exc))
            return

        try:
            profile = load_postprocess_profile(_SHADER_DIR / f"{shader_path}.frag")
        except ValueError as exc:
            # The preset still renders; it just runs without GTAO/SSSS.
            log.warning("Ignoring post-process settings of preset %s: %s", selected, exc)
            profile = NO_POSTPROCESS
        try:
            uniforms = load_preset_uniforms(_SHADER_DIR / f"{shader_path}.frag")
        except ValueError as exc:
            # Shader-side uniform defaults apply instead.
            log.warning("Ignoring uniforms of preset %s: %s", selected, exc)
            uniforms = {}

        self.preset_prog = program
        self.preset_vao = vertex_array
        self.postprocess_profile = profile
        self.preset_uniforms = uniforms
        self._loaded_shader_preset = selected

    def _report_shader_preset_error(self, selected: str, message: str) -> None:
        log.error("Could not load shader preset %s: %s", selected, message)
        self.settings.shader_preset = "none"
        self._loaded_shader_preset = "none"
        self.postprocess_profile = NO_POSTPROCESS
        self.preset_uniforms = {}
        self.shaderPresetError.emit(f"{selected}: {message}")

    def _composite_shader_preset(
        self,
        output_fbo: moderngl.Framebuffer,
        input_color: moderngl.Texture,
        projection: glm.mat4,
        width: int,
        height: int,
    ) -> None:
        assert (
            self.ctx is not None
            and self.preset_prog is not None
            and self.preset_vao is not None
            and self.scene_depth is not None
            and self.scene_normal is not None
        )
        output_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = False

        input_color.use(location=0)
        self.scene_depth.use(location=1)
        self.scene_normal.use(location=2)
        self.preset_prog["u_scene_color"].value = 0
        self.preset_prog["u_scene_depth"].value = 1
        if "u_scene_normal" in self.preset_prog:
            self.preset_prog["u_scene_normal"].value = 2
        for index, name in enumerate((
            "u_material_normal", "u_material_env_mask",
            "u_material_subsurface", "u_material_flags",
        )):
            if name in self.preset_prog:
                self.material_textures[index].use(location=index + 3)
                self.preset_prog[name].value = index + 3
        self.preset_prog["u_viewport_size"].value = (float(width), float(height))
        if "u_key_light" in self.preset_prog:
            self.preset_prog["u_key_light"].value = tuple(self._key_light_view)
        if "u_studio_cubemap" in self.preset_prog and self.default_cubemap is not None:
            self.default_cubemap.use(location=7)
            self.preset_prog["u_studio_cubemap"].value = 7
        if "u_view_to_world" in self.preset_prog:
            self.preset_prog["u_view_to_world"].write(
                bytes(glm.transpose(self._view_to_world))
            )
        self.preset_prog["u_inverse_projection"].write(
            bytes(glm.transpose(glm.inverse(projection)))
        )
        for name, value in list(self.preset_uniforms.items()):
            if name not in self.preset_prog:
                continue
            try:
                self.preset_prog[name].value = value
            except (struct.error, TypeError, ValueError, moderngl.Error) as exc:
                # A value shaped unlike the GLSL uniform; keep the shader's default.
                log.warning("Ignoring preset uniform %s=%r: %s", name, value, exc)
                del self.preset_uniforms[name]
        self.preset_vao.render(moderngl.TRIANGLES, vertices=3)

        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _compile_postprocess_variant(
        self, enable_gtao: bool, enable_ssss: bool, split_composite: bool = False
    ) -> Tuple[moderngl.Program, moderngl.VertexArray]:
        assert self.ctx is not None
        vertex_source = (_SHADER_DIR / "postprocess.vert").read_text(encoding="utf-8")
        fragment_source = (_SHADER_DIR / "postprocess.frag").read_text(encoding="utf-8")
        defines = "".join(
            f"#define {name}\n"
            for name, enabled in (
                ("ENABLE_GTAO", enable_gtao),
                ("ENABLE_SSSS", enable_ssss),
                ("SPLIT_COMPOSITE", split_composite),
            )
            if enabled
        )
        if defines:
            version_line, newline, body = fragment_source.partition("\n")
            fragment_source = f"{version_line}{newline}{defines}{body}"
        program = self.ctx.program(vertex_shader=vertex_source, fragment_shader=fragment_source)
        return program, self.ctx.vertex_array(program, [])

    def _postprocess_variant(
        self, profile: PostProcessProfile
    ) -> Tuple[moderngl.Program, moderngl.VertexArray]:
        assert self.post_prog is not None and self.post_vao is not None
        key = (profile.enable_gtao, profile.enable_ssss, profile.ssss_after_preset)
        if key == (False, False, False) or key in self._failed_post_variants:
            return self.post_prog, self.post_vao
        variant = self._post_variants.get(key)
        if variant is None:
            try:
                variant = self._compile_postprocess_variant(*key)
            except (OSError, moderngl.Error) as exc:
                log.error("Could not compile post-process variant %s: %s", key, exc)
                self._failed_post_variants.add(key)
                return self.post_prog, self.post_vao
            self._post_variants[key] = variant
        return variant

    def _composite_postprocess(
        self,
        output_fbo: moderngl.Framebuffer,
        projection: glm.mat4,
        width: int,
        height: int,
        profile: PostProcessProfile,
    ) -> None:
        assert (
            self.ctx is not None
            and self.scene_color is not None
            and self.scene_depth is not None
        )
        program, vertex_array = self._postprocess_variant(profile)
        output_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = False

        source = self.scene_color
        if profile.ssss_after_preset:
            assert self.split_diffuse is not None and self.split_specular is not None
            source = self.split_diffuse
            if "u_scene_specular" in program:
                self.split_specular.use(location=2)
                program["u_scene_specular"].value = 2
        source.use(location=0)
        program["u_scene_color"].value = 0
        if "u_scene_depth" in program:
            self.scene_depth.use(location=1)
            program["u_scene_depth"].value = 1
            program["u_viewport_size"].value = (float(width), float(height))
            program["u_inverse_projection"].write(
                bytes(glm.transpose(glm.inverse(projection)))
            )
        if "u_ao_radius" in program:
            program["u_ao_radius"].value = profile.gtao_radius
            program["u_ao_intensity"].value = profile.gtao_intensity
            program["u_ao_bias"].value = profile.gtao_bias
        if "u_ssss_radius" in program:
            program["u_ssss_radius"].value = profile.sss_radius
            program["u_ssss_strength"].value = profile.sss_strength
            program["u_ssss_falloff_color"].value = profile.sss_falloff_color
        vertex_array.render(moderngl.TRIANGLES, vertices=3)

        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)

    # ------------------------------------------------------------------
    # Internal Shader & GPU Resource Helpers
    # ------------------------------------------------------------------

    def _init_default_textures(self) -> None:
        assert self.ctx is not None
        resolver = TextureResolver()

        # Studio Cubemap
        faces = resolver.generate_studio_cubemap(face_size=64)
        cube_bytes = b"".join(f.tobytes() for f in faces)
        self.default_cubemap = self.ctx.texture_cube((64, 64), 4, cube_bytes)
        self.default_cubemap.build_mipmaps()
        self.default_cubemap.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        # Fallback 2D textures for unbound slots
        for slot in ("Diffuse", "Normal", "EnvMask", "Specular", "Glow", "White"):
            rgba = resolver.get_fallback_texture(slot, size=16)
            tex = self.ctx.texture((16, 16), 4, rgba.tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.fallback_2d_textures[slot] = tex

    def _init_ground_grid(self) -> None:
        assert self.ctx is not None
        grid_vert = """
        #version 330
        layout(location = 0) in vec3 in_pos;
        layout(location = 1) in vec4 in_col;
        uniform mat4 u_mvp;
        out vec4 v_col;
        void main() {
            gl_Position = u_mvp * vec4(in_pos, 1.0);
            v_col = in_col;
        }
        """
        grid_frag = """
        #version 330
        in vec4 v_col;
        uniform sampler2D u_scene_depth;
        uniform vec2 u_viewport_size;
        uniform bool u_depth_test_scene;
        out vec4 fragColor;
        void main() {
            if (u_depth_test_scene) {
                vec2 uv = gl_FragCoord.xy / u_viewport_size;
                float mesh_depth = texture(u_scene_depth, uv).r;
                if (mesh_depth < 0.999999 && gl_FragCoord.z > mesh_depth + 0.0002) {
                    discard;
                }
            }
            fragColor = v_col;
        }
        """
        self.grid_prog = self.ctx.program(vertex_shader=grid_vert, fragment_shader=grid_frag)

        lines: List[ Tuple[float, float, float, float, float, float, float] ] = []
        extent = 60.0
        step = 10.0
        for i in range(-6, 7):
            coord = i * step
            is_axis = i == 0
            col_x = (0.55, 0.25, 0.28, 0.55) if is_axis else (0.35, 0.38, 0.44, 0.28)
            col_y = (0.25, 0.52, 0.32, 0.55) if is_axis else (0.35, 0.38, 0.44, 0.28)
            # Line parallel to X
            lines.append((-extent, coord, 0.0, *col_x))
            lines.append((extent, coord, 0.0, *col_x))
            # Line parallel to Y
            lines.append((coord, -extent, 0.0, *col_y))
            lines.append((coord, extent, 0.0, *col_y))

        # Pedestal ring at radius 22
        seg_count = 64
        radius = 22.0
        for s in range(seg_count):
            a0 = (s / seg_count) * 2.0 * math.pi
            a1 = ((s + 1) / seg_count) * 2.0 * math.pi
            lines.append((radius * math.cos(a0), radius * math.sin(a0), 0.02, 0.72, 0.60, 0.38, 0.55))
            lines.append((radius * math.cos(a1), radius * math.sin(a1), 0.02, 0.72, 0.60, 0.38, 0.55))

        grid_data = np.asarray(lines, dtype=np.float32)
        self.grid_vertex_count = len(grid_data)
        self.grid_vbo = self.ctx.buffer(grid_data.tobytes())
        self.grid_vao = self.ctx.vertex_array(
            self.grid_prog,
            [(self.grid_vbo, "3f 4f", "in_pos", "in_col")],
        )

    def _acquire_2d_texture(self, res_tex: ResolvedTexture) -> moderngl.Texture:
        assert self.ctx is not None
        if not res_tex.is_fallback and res_tex.resolved_path is not None:
            path = res_tex.resolved_path
            if path in self._shared_tex_cache:
                tex, ref_count = self._shared_tex_cache[path]
                self._shared_tex_cache[path] = (tex, ref_count + 1)
                return tex

            tex = self.ctx.texture((res_tex.width, res_tex.height), 4, res_tex.rgba_data.tobytes())
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 8.0
            self._shared_tex_cache[path] = (tex, 1)
            return tex

        # Procedural fallback texture
        tex = self.ctx.texture((res_tex.width, res_tex.height), 4, res_tex.rgba_data.tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    def _release_2d_texture(self, res_tex: Optional[ResolvedTexture], tex: moderngl.Texture) -> None:
        if res_tex is not None and not res_tex.is_fallback and res_tex.resolved_path is not None:
            path = res_tex.resolved_path
            if path in self._shared_tex_cache:
                cached_tex, ref_count = self._shared_tex_cache[path]
                if ref_count <= 1:
                    del self._shared_tex_cache[path]
                    try:
                        cached_tex.release()
                    except Exception:
                        pass
                else:
                    self._shared_tex_cache[path] = (cached_tex, ref_count - 1)
            return

        try:
            tex.release()
        except Exception:
            pass

    def _upload_shape(self, key: str, sh: ShapeMeshData) -> GpuShapeItem:
        assert self.ctx is not None and self.prog is not None
        num_verts = len(sh.positions)
        v_dtype = np.dtype(
            [
                ("pos", "3f4"),
                ("norm", "3f4"),
                ("uv", "2f4"),
                ("tan", "3f4"),
                ("bitan", "3f4"),
                ("col", "4f4"),
                ("bone_idx", "4i4"),
                ("bone_w", "4f4"),
            ]
        )
        interleaved = np.empty(num_verts, dtype=v_dtype)
        interleaved["pos"] = sh.positions
        interleaved["norm"] = sh.normals
        interleaved["uv"] = sh.uvs
        interleaved["tan"] = sh.tangents
        interleaved["bitan"] = sh.bitangents
        interleaved["col"] = sh.colors
        if len(sh.bone_indices) == num_verts:
            interleaved["bone_idx"] = sh.bone_indices
        else:
            interleaved["bone_idx"] = 0
        if len(sh.bone_weights) == num_verts:
            interleaved["bone_w"] = sh.bone_weights
        else:
            interleaved["bone_w"] = (1.0, 0.0, 0.0, 0.0)

        vbo = self.ctx.buffer(interleaved.tobytes())
        ibo = self.ctx.buffer(sh.triangles.astype(np.uint32).tobytes())
        vao = self.ctx.vertex_array(
            self.prog,
            [
                (
                    vbo,
                    "3f 3f 2f 3f 3f 4f 4i 4f",
                    "in_position",
                    "in_normal",
                    "in_uv",
                    "in_tangent",
                    "in_bitangent",
                    "in_color",
                    "in_bone_indices",
                    "in_bone_weights",
                )
            ],
            index_buffer=ibo,
            index_element_size=4,
        )

        gpu_textures: Dict[str, moderngl.Texture] = {}
        for slot, res_tex in sh.shader.resolved_textures.items():
            if slot == "EnvMap":
                continue  # Handled via cubemap unit
            gpu_textures[slot] = self._acquire_2d_texture(res_tex)

        mapped_bone_names: List[str] = []
        num_bones = min(len(sh.bone_names), MAX_SHADER_BONES)
        effective_skin_mats = np.repeat(
            np.eye(4, dtype=np.float32)[None, :, :], num_bones, axis=0
        )
        if sh.is_skinned and num_bones > 0:
            for b_idx in range(num_bones):
                b_name = sh.bone_names[b_idx]
                resolved = self.anim_engine.resolve_bone_name(b_name)
                if resolved is not None:
                    mapped_bone_names.append(resolved)
                    if b_idx < len(sh.bone_skin_matrices):
                        effective_skin_mats[b_idx] = sh.bone_skin_matrices[b_idx]
                elif b_idx < len(sh.bone_skin_matrices) and self.smp.outfit_bone_matches_skin(
                    b_name, sh.bone_skin_matrices[b_idx]
                ):
                    # Outfit-only bone (cloth/hair chain): skinned through its own matrix, which is its animated
                    # NIF parent chain or, when SMP simulates it, the Bullet result.
                    mapped_bone_names.append(b_name)
                    effective_skin_mats[b_idx] = sh.bone_skin_matrices[b_idx]
                else:
                    bind_pos: Optional[np.ndarray] = None
                    if b_idx < len(sh.bone_skin_matrices):
                        try:
                            bind_pos = np.linalg.inv(sh.bone_skin_matrices[b_idx])[:3, 3]
                        except np.linalg.LinAlgError:
                            bind_pos = None
                    ancestor = self.anim_engine.find_fallback_ancestor(b_name, bind_pos)
                    if ancestor is not None and ancestor in self.anim_engine.node_by_name:
                        mapped_bone_names.append(ancestor)
                        effective_skin_mats[b_idx] = (
                            self.anim_engine.node_by_name[ancestor].inverse_bind_matrix
                        )
                    else:
                        fallback_root = self.anim_engine.nodes[0].name
                        mapped_bone_names.append(fallback_root)

        return GpuShapeItem(
            source_key=key,
            shape_data=sh,
            vbo=vbo,
            ibo=ibo,
            vao=vao,
            textures=gpu_textures,
            mapped_bone_names=mapped_bone_names,
            effective_skin_matrices=effective_skin_mats,
        )

    def _release_nif_internal(self, key: str) -> None:
        items = self._gpu_nifs.pop(key, [])
        for item in items:
            for slot, tex in item.textures.items():
                res_tex = item.shape_data.shader.resolved_textures.get(slot)
                self._release_2d_texture(res_tex, tex)
            item.release()

    def _compute_view_matrix(self) -> glm.mat4:
        yaw_rad = math.radians(self.cam_yaw_deg)
        pitch_rad = math.radians(max(-88.0, min(88.0, self.cam_pitch_deg)))
        cos_p = math.cos(pitch_rad)
        offset = glm.vec3(
            self.cam_distance * cos_p * math.cos(yaw_rad),
            self.cam_distance * cos_p * math.sin(yaw_rad),
            self.cam_distance * math.sin(pitch_rad),
        )
        eye = self.cam_target + offset
        return glm.lookAt(eye, self.cam_target, glm.vec3(0.0, 0.0, 1.0))

    def _set_uniform(self, name: str, value) -> None:
        if self.prog is not None and name in self.prog:
            self.prog[name].value = value

    def _write_matrix(self, name: str, mat: glm.mat4 | glm.mat3) -> None:
        if self.prog is not None and name in self.prog:
            self.prog[name].write(bytes(glm.transpose(mat)))

    def _setup_global_shader_uniforms(
        self,
        proj: glm.mat4,
        view: glm.mat4,
        model: glm.mat4,
        mv_inv: glm.mat4,
        normal_mat: glm.mat3,
    ) -> None:
        assert self.prog is not None
        s = self.settings

        self._write_matrix("matProjection", proj)
        self._write_matrix("matView", view)
        self._write_matrix("matModel", model)
        self._write_matrix("matModelViewInverse", mv_inv)
        self._write_matrix("mv_normalMatrix", normal_mat)

        self._set_uniform("bLightEnabled", bool(s.enable_lighting))
        self._set_uniform("bShowTexture", bool(s.show_textures))
        self._set_uniform("bShowMask", False)
        self._set_uniform("bShowWeight", False)
        self._set_uniform("bComplexMaterial", False)

        self._set_uniform("cubemapMaxLod", 6.0)
        self._set_uniform("cubemapMinLod", 0.0)
        self._set_uniform("cubemapTint", (1.0, 1.0, 1.0))
        self._set_uniform("ambient", float(s.ambient_intensity))

        # Compute world-space directional lights from the user's azimuth & elevation sliders
        az = math.radians(s.light_azimuth_deg)
        el = math.radians(s.light_elevation_deg)
        # Rotate around -Y (front of character) so azimuth=0 shines straight on the front
        key_dir = glm.normalize(
            glm.vec3(
                math.sin(az) * math.cos(el),
                -math.cos(az) * math.cos(el),
                math.sin(el),
            )
        )
        fill_az = az - math.radians(115.0)
        self._key_light_view = (
            glm.normalize(glm.mat3(view) * key_dir) if s.enable_lighting else glm.vec3(0.0)
        )
        self._view_to_world = glm.mat3(model) * glm.inverse(glm.mat3(view))
        fill_dir = glm.normalize(
            glm.vec3(
                math.sin(fill_az) * 0.85,
                -math.cos(fill_az) * 0.85,
                0.25,
            )
        )
        rim_az = az + math.radians(165.0)
        rim_dir = glm.normalize(
            glm.vec3(
                math.sin(rim_az) * 0.8,
                -math.cos(rim_az) * 0.8,
                0.55,
            )
        )

        k_str = float(s.light_intensity)
        self._set_uniform("frontal.direction", (0.0, 0.15, 0.988))
        self._set_uniform("frontal.diffuse", (0.18 * k_str, 0.18 * k_str, 0.20 * k_str))

        self._set_uniform("directional0.direction", (float(key_dir.x), float(key_dir.y), float(key_dir.z)))
        self._set_uniform("directional0.diffuse", (0.85 * k_str, 0.82 * k_str, 0.78 * k_str))

        self._set_uniform("directional1.direction", (float(fill_dir.x), float(fill_dir.y), float(fill_dir.z)))
        self._set_uniform("directional1.diffuse", (0.30 * k_str, 0.34 * k_str, 0.42 * k_str))

        self._set_uniform("directional2.direction", (float(rim_dir.x), float(rim_dir.y), float(rim_dir.z)))
        self._set_uniform("directional2.diffuse", (0.28 * k_str, 0.26 * k_str, 0.32 * k_str))

        # Assign texture sampler units once
        self._set_uniform("texDiffuse", 0)
        self._set_uniform("texNormal", 1)
        self._set_uniform("texCubemap", 2)
        self._set_uniform("texEnvMask", 3)
        self._set_uniform("texSpecular", 4)
        self._set_uniform("texBacklight", 5)
        self._set_uniform("texLightmask", 6)
        self._set_uniform("texGlowmap", 7)
        self._set_uniform("texFaceTint", 8)
        self._set_uniform("texPresetNormal", 9)
        self._set_uniform("texPresetEnvMask", 10)
        self._set_uniform("texPresetSubsurface", 11)

    def _render_shape_item(self, item: GpuShapeItem, wireframe: bool = False) -> None:
        assert self.ctx is not None and self.prog is not None
        s = self.settings
        mat = item.shape_data.shader
        ap = item.shape_data.alpha_prop

        if mat.b_double_sided or wireframe:
            self.ctx.disable(moderngl.CULL_FACE)
        else:
            self.ctx.enable(moderngl.CULL_FACE)

        self.ctx.wireframe = bool(wireframe)
        self._set_uniform("bWireframe", bool(wireframe))

        pose = self._current_pose_matrices
        use_skinning = (
            pose is not None
            and item.shape_data.is_skinned
            and len(item.mapped_bone_names) > 0
            and self.bone_ubo is not None
            and all(name in pose for name in item.mapped_bone_names)
        )
        self._set_uniform("bSkinned", bool(use_skinning))
        if use_skinning and pose is not None and self.bone_ubo is not None:
            pose_stack = np.stack(
                [pose[name] for name in item.mapped_bone_names],
                axis=0,
            )
            skin_mats = pose_stack @ item.effective_skin_matrices
            # Upload column-major matrices for GLSL std140 mat4 array
            col_major = np.ascontiguousarray(skin_mats.swapaxes(1, 2), dtype=np.float32)
            self.bone_ubo.write(col_major.tobytes(), offset=0)
            self.bone_ubo.bind_to_uniform_block(0)

        self._set_uniform("bNormalMap", bool(s.enable_normal_map and mat.b_normal_map))
        self._set_uniform("bModelSpace", bool(mat.b_model_space))
        self._set_uniform("bCubemap", bool(s.enable_cubemap and mat.b_cubemap))
        self._set_uniform("bEnvMask", bool(mat.b_env_mask and "EnvMask" in item.textures))
        self._set_uniform("bSpecular", bool(s.enable_specular and mat.b_specular))
        self._set_uniform("bEmissive", bool(mat.b_emissive))
        self._set_uniform("bBacklight", bool(mat.b_backlight and "Backlight" in item.textures))
        self._set_uniform("bRimlight", bool(mat.b_rimlight))
        self._set_uniform("bSoftlight", bool(mat.b_softlight))
        self._set_uniform("bGlowmap", bool(mat.b_glowmap and "Glow" in item.textures))
        self._set_uniform("bTintColor", bool(mat.b_tint_color))
        self._set_uniform("bFaceTint", False)

        self._set_uniform("prop.uvOffset", mat.uv_offset)
        self._set_uniform("prop.uvScale", mat.uv_scale)
        self._set_uniform("prop.specularColor", mat.specular_color)
        self._set_uniform("prop.specularStrength", float(mat.specular_strength * s.specular_multiplier))
        self._set_uniform("prop.shininess", float(mat.shininess))
        self._set_uniform("prop.envReflection", float(mat.env_reflection * s.env_reflection_multiplier))
        self._set_uniform("prop.emissiveColor", mat.emissive_color)
        self._set_uniform("prop.emissiveMultiple", float(mat.emissive_multiple))
        self._set_uniform("prop.tintColor", mat.tint_color)
        self._set_uniform("prop.alpha", float(mat.alpha))
        self._set_uniform("prop.rimlightPower", float(mat.rimlight_power))
        self._set_uniform("prop.softlighting", float(mat.softlighting))
        self._set_uniform("alphaThreshold", float(ap.shader_alpha_threshold))

        # Bind textures to their respective units
        item.textures.get("Diffuse", self.fallback_2d_textures["Diffuse"]).use(location=0)
        item.textures.get("Normal", self.fallback_2d_textures["Normal"]).use(location=1)
        if self.default_cubemap is not None:
            self.default_cubemap.use(location=2)
        item.textures.get("EnvMask", self.fallback_2d_textures["EnvMask"]).use(location=3)
        item.textures.get("Specular", self.fallback_2d_textures["Specular"]).use(location=4)
        item.textures.get("Backlight", self.fallback_2d_textures["White"]).use(location=5)
        item.textures.get("Lightmask", self.fallback_2d_textures["White"]).use(location=6)
        item.textures.get("Glow", self.fallback_2d_textures["Glow"]).use(location=7)
        item.textures.get("FaceTint", self.fallback_2d_textures["White"]).use(location=8)

        for uniform, primary, secondary, location, fallback in (
            ("bPresetHasNormal", "Normal", "PresetNormal", 9, "Normal"),
            ("bPresetHasEnvMask", "EnvMask", "PresetEnvMask", 10, "EnvMask"),
            ("bPresetHasSubsurface", "Subsurface", "Subsurface", 11, "White"),
        ):
            slot = next((
                candidate for candidate in (primary, secondary)
                if candidate in item.textures
                and candidate in mat.resolved_textures
                and not mat.resolved_textures[candidate].is_fallback
            ), None)
            self._set_uniform(uniform, slot is not None)
            if uniform == "bPresetHasNormal":
                model_space = mat.b_model_space
                if slot == "PresetNormal":
                    normal_path = mat.resolved_textures[slot].raw_nif_path
                    model_space = normal_path.lower().endswith("_msn.dds")
                self._set_uniform("bPresetNormalModelSpace", model_space)
            texture = item.textures[slot] if slot is not None else self.fallback_2d_textures[fallback]
            texture.use(location=location)

        item.vao.render(moderngl.TRIANGLES)
        if wireframe:
            self.ctx.wireframe = False

    def _emit_stats(self) -> None:
        nif_count = 0
        shape_count = 0
        tri_count = 0
        for key, items in self._gpu_nifs.items():
            if key == self.BASE_BODY_KEY and not self._base_body_visible:
                continue
            if not self._nif_visibility.get(key, True):
                continue
            if key != self.BASE_BODY_KEY:
                nif_count += 1
            shape_count += len(items)
            for it in items:
                tri_count += len(it.shape_data.triangles)
        self.statsChanged.emit(nif_count, shape_count, tri_count)

    # ------------------------------------------------------------------
    # Mouse Camera Interaction (Orbit / Pan / Zoom)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse_pos = event.position()
        self._active_mouse_button = event.button()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        dx = float(pos.x() - self._last_mouse_pos.x())
        dy = float(pos.y() - self._last_mouse_pos.y())
        self._last_mouse_pos = pos

        mods = event.modifiers()
        is_pan = (
            self._active_mouse_button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton)
            or (
                self._active_mouse_button == Qt.MouseButton.LeftButton
                and bool(mods & Qt.KeyboardModifier.ShiftModifier)
            )
        )

        if self._active_mouse_button == Qt.MouseButton.LeftButton and not is_pan:
            self.cam_yaw_deg -= dx * 0.45
            self.cam_pitch_deg = max(-85.0, min(85.0, self.cam_pitch_deg + dy * 0.45))
            self.update()
        elif is_pan:
            pan_scale = self.cam_distance * 0.0016
            yaw_rad = math.radians(self.cam_yaw_deg)
            pitch_rad = math.radians(self.cam_pitch_deg)
            forward = glm.normalize(
                glm.vec3(
                    -math.cos(pitch_rad) * math.cos(yaw_rad),
                    -math.cos(pitch_rad) * math.sin(yaw_rad),
                    -math.sin(pitch_rad),
                )
            )
            right = glm.normalize(glm.cross(forward, glm.vec3(0.0, 0.0, 1.0)))
            up = glm.normalize(glm.cross(right, forward))
            self.cam_target -= right * (dx * pan_scale)
            self.cam_target += up * (dy * pan_scale)
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._active_mouse_button = Qt.MouseButton.NoButton
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta != 0:
            factor = math.pow(0.88, delta / 120.0)
            self.cam_distance = max(15.0, min(1500.0, self.cam_distance * factor))
            self.update()
        super().wheelEvent(event)
