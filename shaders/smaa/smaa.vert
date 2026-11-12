// Prepended at load time: #version, SMAA_GLSL_3, the quality preset,
// SMAA_RT_METRICS, SMAA_PASS (0 edges, 1 weights, 2 blending) and SMAA.hlsl.

out vec2 v_texcoord;
#if SMAA_PASS == 0
out vec4 v_offset[3];
#elif SMAA_PASS == 1
out vec2 v_pixcoord;
out vec4 v_offset[3];
#else
out vec4 v_offset;
#endif

void main() {
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );
    vec2 position = positions[gl_VertexID];
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = position * 0.5 + 0.5;

#if SMAA_PASS == 0
    vec4 offset[3];
    SMAAEdgeDetectionVS(v_texcoord, offset);
    v_offset = offset;
#elif SMAA_PASS == 1
    vec4 offset[3];
    vec2 pixcoord;
    SMAABlendingWeightCalculationVS(v_texcoord, pixcoord, offset);
    v_pixcoord = pixcoord;
    v_offset = offset;
#else
    SMAANeighborhoodBlendingVS(v_texcoord, v_offset);
#endif
}
