// Prepended at load time: #version, SMAA_GLSL_3, the quality preset,
// SMAA_RT_METRICS, SMAA_PASS (0 edges, 1 weights, 2 blending) and SMAA.hlsl.

in vec2 v_texcoord;
#if SMAA_PASS == 0
in vec4 v_offset[3];
#elif SMAA_PASS == 1
in vec2 v_pixcoord;
in vec4 v_offset[3];
#else
in vec4 v_offset;
#endif

uniform sampler2D u_color;
uniform sampler2D u_edges;
uniform sampler2D u_area;
uniform sampler2D u_search;
uniform sampler2D u_blend;

out vec4 fragColor;

void main() {
#if SMAA_PASS == 0
    vec4 offset[3] = v_offset;
    fragColor = vec4(SMAAColorEdgeDetectionPS(v_texcoord, offset, u_color), 0.0, 0.0);
#elif SMAA_PASS == 1
    vec4 offset[3] = v_offset;
    fragColor = SMAABlendingWeightCalculationPS(
        v_texcoord, v_pixcoord, offset, u_edges, u_area, u_search, vec4(0.0)
    );
#else
    fragColor = SMAANeighborhoodBlendingPS(v_texcoord, v_offset, u_color, u_blend);
#endif
}
