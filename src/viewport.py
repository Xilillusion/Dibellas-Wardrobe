"""
3D OpenGL Viewport Widget for Dibella's Wardrobe.

Embeds a ModernGL 3.3 Core context inside PySide6's `QOpenGLWidget`, compiles Outfit Studio's
`default.frag` + `default.vert` GLSL shaders, manages GPU buffers/textures in VRAM as `.nif`
files are checked/unchecked, and provides an interactive Z-up orbit/pan/zoom camera.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import glm
import moderngl
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .nif_loader import LoadedNifData, ShapeMeshData
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

    light_azimuth_deg: float = 35.0
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
    ssss_falloff_color: Tuple[float, float, float] = (1.0, 0.45, 0.35)


class OutfitViewportWidget(QOpenGLWidget):
    """
    Real-time OpenGL 3.3 viewport rendering stacked Skyrim `.nif` meshes with
    Outfit Studio's `default.frag` shader.
    """

    statsChanged = Signal(int, int, int)  # (nif_count, shape_count, triangle_count)

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
        self.post_prog: Optional[moderngl.Program] = None
        self.post_vao: Optional[moderngl.VertexArray] = None
        self.scene_color: Optional[moderngl.Texture] = None
        self.scene_depth: Optional[moderngl.Texture] = None
        self.scene_fbo: Optional[moderngl.Framebuffer] = None
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
        self._emit_stats()
        self.update()

    def is_nif_visible(self, key: str) -> bool:
        """Return whether a loaded NIF is visible."""
        return self._nif_visibility.get(key, True)

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

        post_vert_src = (_SHADER_DIR / "postprocess.vert").read_text(encoding="utf-8")
        post_frag_src = (_SHADER_DIR / "postprocess.frag").read_text(encoding="utf-8")
        self.post_prog = self.ctx.program(
            vertex_shader=post_vert_src,
            fragment_shader=post_frag_src,
        )
        self.post_vao = self.ctx.vertex_array(self.post_prog, [])

        self._init_default_textures()
        self._init_ground_grid()

        # Upload any NIFs that were queued prior to GL context creation
        if self._pending_nifs:
            pending = dict(self._pending_nifs)
            self._pending_nifs.clear()
            for key, nif_data in pending.items():
                self._gpu_nifs[key] = [self._upload_shape(key, sh) for sh in nif_data.shapes]
            self._emit_stats()

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
        assert self.scene_fbo is not None
        self.scene_fbo.use()
        self.ctx.viewport = (0, 0, vp_w, vp_h)

        # Effects use only mesh color and depth; the grid is composited afterward.
        self.scene_fbo.clear(0.13, 0.14, 0.17, 1.0, depth=1.0)

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
                missing_diffuse = (
                    self.settings.show_textures
                    and diffuse is not None
                    and diffuse.is_fallback
                )
                if (
                    (ap.has_alpha_property and ap.alpha_blend)
                    or sh.alpha < 0.999
                    or missing_diffuse
                ):
                    transparent_items.append(item)
                else:
                    opaque_items.append(item)

        self._setup_global_shader_uniforms(proj, view, model, mv_inv, normal_mat)

        # Pass 1: Opaque & Alpha-Cutout meshes
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)
        for item in opaque_items:
            self._render_shape_item(item, wireframe=False)

        # Pass 2: Alpha-Blended meshes
        if transparent_items:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            # Keep depth writes on if alpha test is also enabled (e.g. lace/hair), else depth_mask=False
            for item in transparent_items:
                self.ctx.depth_mask = bool(item.shape_data.alpha_prop.alpha_test)
                self._render_shape_item(item, wireframe=False)
            self.ctx.depth_mask = True
            self.ctx.disable(moderngl.BLEND)

        # Optional Pass 3: Wireframe overlay
        if self.settings.wireframe_overlay:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            for item in opaque_items + transparent_items:
                self._render_shape_item(item, wireframe=True)
            self.ctx.disable(moderngl.BLEND)

        self._composite_postprocess(output_fbo, proj, vp_w, vp_h)

        if (
            self.settings.show_ground_grid
            and self.grid_vao is not None
            and self.grid_prog is not None
            and self.scene_depth is not None
        ):
            output_fbo.use()
            self.ctx.viewport = (0, 0, vp_w, vp_h)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.depth_mask = False
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.grid_prog["u_mvp"].write(bytes(glm.transpose(proj * view)))
            self.grid_prog["u_viewport_size"].value = (float(vp_w), float(vp_h))
            self.grid_prog["u_depth_test_scene"].value = True
            self.scene_depth.use(location=0)
            self.grid_prog["u_scene_depth"].value = 0
            self.grid_vao.render(moderngl.LINES)
            self.ctx.depth_mask = True
            self.ctx.disable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)

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
        self.scene_fbo = self.ctx.framebuffer(
            color_attachments=[self.scene_color],
            depth_attachment=self.scene_depth,
        )
        self._scene_target_size = (width, height)

    def _release_scene_target(self) -> None:
        for resource in (self.scene_fbo, self.scene_color, self.scene_depth):
            if resource is not None:
                resource.release()
        self.scene_fbo = None
        self.scene_color = None
        self.scene_depth = None
        self._scene_target_size = (0, 0)

    def _composite_postprocess(
        self,
        output_fbo: moderngl.Framebuffer,
        projection: glm.mat4,
        width: int,
        height: int,
    ) -> None:
        assert (
            self.ctx is not None
            and self.post_prog is not None
            and self.post_vao is not None
            and self.scene_color is not None
            and self.scene_depth is not None
        )
        output_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = False

        self.scene_color.use(location=0)
        self.scene_depth.use(location=1)
        self.post_prog["u_scene_color"].value = 0
        self.post_prog["u_scene_depth"].value = 1
        self.post_prog["u_viewport_size"].value = (float(width), float(height))
        self.post_prog["u_projection"].write(bytes(glm.transpose(projection)))
        self.post_prog["u_inverse_projection"].write(
            bytes(glm.transpose(glm.inverse(projection)))
        )

        settings = self.settings
        self.post_prog["u_enable_ssr"].value = settings.enable_ssr
        self.post_prog["u_ssr_step_size"].value = max(0.1, settings.ssr_step_size)
        self.post_prog["u_ssr_max_steps"].value = max(
            1, min(128, int(settings.ssr_max_steps))
        )
        self.post_prog["u_ssr_thickness"].value = max(0.01, settings.ssr_thickness)
        self.post_prog["u_enable_ao"].value = settings.enable_ao
        self.post_prog["u_ao_radius"].value = max(0.1, settings.ao_radius)
        self.post_prog["u_ao_intensity"].value = max(0.0, settings.ao_intensity)
        self.post_prog["u_ao_bias"].value = max(0.0, settings.ao_bias)
        self.post_prog["u_enable_ssss"].value = settings.enable_ssss
        self.post_prog["u_ssss_radius"].value = max(1.0, settings.ssss_radius)
        self.post_prog["u_ssss_falloff_color"].value = tuple(
            max(0.0, min(1.0, float(channel)))
            for channel in settings.ssss_falloff_color
        )
        self.post_vao.render(moderngl.TRIANGLES, vertices=3)

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
        interleaved = np.empty((num_verts, 18), dtype=np.float32)
        interleaved[:, 0:3] = sh.positions
        interleaved[:, 3:6] = sh.normals
        interleaved[:, 6:8] = sh.uvs
        interleaved[:, 8:11] = sh.tangents
        interleaved[:, 11:14] = sh.bitangents
        interleaved[:, 14:18] = sh.colors

        vbo = self.ctx.buffer(interleaved.tobytes())
        ibo = self.ctx.buffer(sh.triangles.astype(np.uint32).tobytes())
        vao = self.ctx.vertex_array(
            self.prog,
            [
                (
                    vbo,
                    "3f 3f 2f 3f 3f 4f",
                    "in_position",
                    "in_normal",
                    "in_uv",
                    "in_tangent",
                    "in_bitangent",
                    "in_color",
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

        return GpuShapeItem(
            source_key=key,
            shape_data=sh,
            vbo=vbo,
            ibo=ibo,
            vao=vao,
            textures=gpu_textures,
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
