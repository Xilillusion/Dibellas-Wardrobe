"""
SMAA 1x anti-aliasing for the viewport's final image.

Wraps the reference implementation by Jimenez et al. (shaders/smaa/SMAA.hlsl, MIT
licensed) in its three passes: color edge detection, blending weight calculation and
neighborhood blending.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import moderngl

SMAA_DIR = Path(__file__).resolve().parent.parent / "shaders" / "smaa"
SMAA_QUALITIES = ("low", "medium", "high", "ultra")
DEFAULT_SMAA_QUALITY = "high"

_AREA_TEX_SIZE = (160, 560)
_SEARCH_TEX_SIZE = (64, 16)
_PASS_COUNT = 3


def smaa_shader_sources(
    quality: str, smaa_pass: int, shader_dir: Path = SMAA_DIR
) -> Tuple[str, str]:
    """Return the (vertex, fragment) GLSL sources for one SMAA pass."""
    if quality not in SMAA_QUALITIES:
        raise ValueError(f"Unknown SMAA quality: {quality!r}")
    if not 0 <= smaa_pass < _PASS_COUNT:
        raise ValueError(f"Unknown SMAA pass: {smaa_pass!r}")
    # The vendored file is Latin-1 and only uses non-ASCII in comments, which some
    # GLSL compilers reject outright.
    library = (shader_dir / "SMAA.hlsl").read_bytes().decode("latin-1")
    library = library.encode("ascii", "replace").decode("ascii")
    # ModernGL resolves `#include` lines itself, even the usage examples in comments.
    library = library.replace("#include", "include")
    header = (
        "#version 330\n"
        "#define SMAA_GLSL_3\n"
        f"#define SMAA_PRESET_{quality.upper()}\n"
        "#define SMAA_RT_METRICS u_smaa_rt_metrics\n"
        f"#define SMAA_PASS {smaa_pass}\n"
        "uniform vec4 u_smaa_rt_metrics;\n"
    )
    vertex = (
        header + "#define SMAA_INCLUDE_PS 0\n" + library + "\n"
        + (shader_dir / "smaa.vert").read_text(encoding="utf-8")
    )
    fragment = (
        header + "#define SMAA_INCLUDE_VS 0\n" + library + "\n"
        + (shader_dir / "smaa.frag").read_text(encoding="utf-8")
    )
    return vertex, fragment


class SmaaPostProcess:
    """GPU resources for SMAA; render the image into `input_fbo`, then call `apply`."""

    def __init__(self, ctx: moderngl.Context, shader_dir: Path = SMAA_DIR) -> None:
        self.ctx = ctx
        self.shader_dir = shader_dir
        self._programs: Dict[
            str, Tuple[List[moderngl.Program], List[moderngl.VertexArray]]
        ] = {}
        self._area_tex: Optional[moderngl.Texture] = None
        self._search_tex: Optional[moderngl.Texture] = None
        self.input_color: Optional[moderngl.Texture] = None
        self.input_fbo: Optional[moderngl.Framebuffer] = None
        self._edges: Optional[moderngl.Texture] = None
        self._edges_fbo: Optional[moderngl.Framebuffer] = None
        self._blend: Optional[moderngl.Texture] = None
        self._blend_fbo: Optional[moderngl.Framebuffer] = None
        self._size: Tuple[int, int] = (0, 0)

    def prepare(self, quality: str, width: int, height: int) -> moderngl.Framebuffer:
        """Compile the passes and size the targets; returns the framebuffer to render into."""
        self._load_programs(quality)
        self._load_lookup_textures()
        self._ensure_targets(width, height)
        assert self.input_fbo is not None
        return self.input_fbo

    def apply(
        self, output_fbo: moderngl.Framebuffer, quality: str, width: int, height: int
    ) -> None:
        """Anti-alias `input_color` into `output_fbo`; `prepare` must have been called."""
        programs, arrays = self._programs[quality]
        assert (
            self.input_color is not None
            and self._edges is not None
            and self._edges_fbo is not None
            and self._blend is not None
            and self._blend_fbo is not None
            and self._area_tex is not None
            and self._search_tex is not None
        )
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        metrics = (1.0 / width, 1.0 / height, float(width), float(height))
        for program in programs:
            program["u_smaa_rt_metrics"].value = metrics

        # The edge pass discards edgeless pixels, so stale edges must be cleared.
        self._edges_fbo.use()
        ctx.viewport = (0, 0, width, height)
        self._edges_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.input_color.use(location=0)
        self._bind(programs[0], "u_color", 0)
        arrays[0].render(moderngl.TRIANGLES, vertices=3)

        self._blend_fbo.use()
        ctx.viewport = (0, 0, width, height)
        self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._edges.use(location=0)
        self._area_tex.use(location=1)
        self._search_tex.use(location=2)
        self._bind(programs[1], "u_edges", 0)
        self._bind(programs[1], "u_area", 1)
        self._bind(programs[1], "u_search", 2)
        arrays[1].render(moderngl.TRIANGLES, vertices=3)

        output_fbo.use()
        ctx.viewport = (0, 0, width, height)
        self.input_color.use(location=0)
        self._blend.use(location=1)
        self._bind(programs[2], "u_color", 0)
        self._bind(programs[2], "u_blend", 1)
        arrays[2].render(moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST)

    def release_targets(self) -> None:
        for resource in (
            self.input_fbo, self.input_color,
            self._edges_fbo, self._edges,
            self._blend_fbo, self._blend,
        ):
            if resource is not None:
                resource.release()
        self.input_fbo = None
        self.input_color = None
        self._edges_fbo = None
        self._edges = None
        self._blend_fbo = None
        self._blend = None
        self._size = (0, 0)

    def release(self) -> None:
        self.release_targets()
        for programs, arrays in self._programs.values():
            for resource in (*arrays, *programs):
                resource.release()
        self._programs.clear()
        for texture in (self._area_tex, self._search_tex):
            if texture is not None:
                texture.release()
        self._area_tex = None
        self._search_tex = None

    @staticmethod
    def _bind(program: moderngl.Program, name: str, location: int) -> None:
        if name in program:
            program[name].value = location

    def _load_programs(self, quality: str) -> None:
        if quality in self._programs:
            return
        programs: List[moderngl.Program] = []
        arrays: List[moderngl.VertexArray] = []
        try:
            for smaa_pass in range(_PASS_COUNT):
                vertex, fragment = smaa_shader_sources(quality, smaa_pass, self.shader_dir)
                program = self.ctx.program(vertex_shader=vertex, fragment_shader=fragment)
                programs.append(program)
                arrays.append(self.ctx.vertex_array(program, []))
        except Exception:
            for resource in (*arrays, *programs):
                resource.release()
            raise
        self._programs[quality] = (programs, arrays)

    def _load_lookup_textures(self) -> None:
        if self._area_tex is None:
            data = (self.shader_dir / "AreaTex.bin").read_bytes()
            self._area_tex = self._lookup_texture(_AREA_TEX_SIZE, 2, data)
        if self._search_tex is None:
            data = (self.shader_dir / "SearchTex.bin").read_bytes()
            self._search_tex = self._lookup_texture(_SEARCH_TEX_SIZE, 1, data)

    def _lookup_texture(
        self, size: Tuple[int, int], components: int, data: bytes
    ) -> moderngl.Texture:
        expected = size[0] * size[1] * components
        if len(data) != expected:
            raise ValueError(
                f"SMAA lookup texture has {len(data)} bytes; expected {expected}"
            )
        # Rows are uploaded in file order: GL's v=0 is then the first row, which is
        # exactly how SMAA's D3D-oriented lookup math addresses these tables.
        texture = self.ctx.texture(size, components, data, alignment=1)
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    def _ensure_targets(self, width: int, height: int) -> None:
        if self.input_fbo is not None and self._size == (width, height):
            return
        self.release_targets()
        self.input_color = self._target_texture(width, height, 4)
        self.input_fbo = self.ctx.framebuffer(color_attachments=[self.input_color])
        self._edges = self._target_texture(width, height, 2)
        self._edges_fbo = self.ctx.framebuffer(color_attachments=[self._edges])
        self._blend = self._target_texture(width, height, 4)
        self._blend_fbo = self.ctx.framebuffer(color_attachments=[self._blend])
        self._size = (width, height)

    def _target_texture(self, width: int, height: int, components: int) -> moderngl.Texture:
        texture = self.ctx.texture((width, height), components, dtype="f1")
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture
