#version 330

// Anime finish in the style of modern 3D anime games such as Wuthering Waves:
// shape-driven two-tone cel shading with cool shadows and warm skin terminators,
// a depth-offset rim light, coloured ink outlines and crisp highlights.

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;
uniform sampler2D u_scene_normal;
uniform sampler2D u_material_env_mask;
uniform sampler2D u_material_subsurface;
uniform sampler2D u_material_flags;
uniform vec2 u_viewport_size;
uniform mat4 u_inverse_projection;
// View-space direction towards the key light; zero while scene lighting is off.
uniform vec3 u_key_light;

const float BACKGROUND_DEPTH = 0.999999;
// Used when scene lighting is off, so the finish still has a light to shade by.
const vec3 FALLBACK_LIGHT = vec3(-0.3537, 0.5558, 0.7680);
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

// Lit tone is a soft warm white; cloth shadows lean blue-violet; skin shadows
// stay rosy and pick up a saturated red band along the terminator.
const vec3 LIT_TINT = vec3(1.015, 1.0, 0.98);
const vec3 CLOTH_SHADOW = vec3(0.86, 0.87, 0.94);
const vec3 SKIN_SHADOW = vec3(0.97, 0.91, 0.90);
const vec3 SKIN_TERMINATOR = vec3(1.045, 0.93, 0.89);
const float TERMINATOR_OFFSET = 0.08;
const float TERMINATOR_SOFTNESS = 0.035;

const float RIM_WIDTH = 4.0;
const float RIM_STRENGTH = 0.10;
const vec3 INK = vec3(0.30, 0.25, 0.36);

vec3 view_position(vec2 uv, float depth) {
    vec4 clip_position = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 position = u_inverse_projection * clip_position;
    return position.xyz / max(abs(position.w), 0.000001) * sign(position.w);
}

float view_depth(vec2 uv) {
    float depth = texture(u_scene_depth, uv).r;
    return depth >= BACKGROUND_DEPTH ? 1.0e6 : -view_position(uv, depth).z;
}

bool scene_normal(vec2 uv, out vec3 normal) {
    vec4 encoded = texture(u_scene_normal, uv);
    normal = encoded.xyz * 2.0 - 1.0;
    if (encoded.a < 0.5 || dot(normal, normal) < 0.25) {
        return false;
    }
    normal = normalize(normal);
    return true;
}

vec3 depth_normal(vec2 uv, vec3 position) {
    vec2 texel = 1.0 / u_viewport_size;
    vec3 dx = view_position(uv + vec2(texel.x, 0.0), texture(u_scene_depth, uv + vec2(texel.x, 0.0)).r)
        - position;
    vec3 dy = view_position(uv + vec2(0.0, texel.y), texture(u_scene_depth, uv + vec2(0.0, texel.y)).r)
        - position;
    vec3 normal = cross(dx, dy);
    if (dot(normal, normal) < 1.0e-12) {
        return vec3(0.0, 0.0, 1.0);
    }
    normal = normalize(normal);
    return dot(normal, -position) < 0.0 ? -normal : normal;
}

vec3 saturate_color(vec3 color, float amount) {
    return max(mix(vec3(dot(color, LUMA)), color, amount), 0.0);
}

void main() {
    vec4 source = texture(u_scene_color, v_uv);
    float depth = texture(u_scene_depth, v_uv).r;
    if (depth >= BACKGROUND_DEPTH) {
        fragColor = source;
        return;
    }

    vec2 texel = 1.0 / u_viewport_size;
    vec3 position = view_position(v_uv, depth);
    float center_depth = -position.z;
    vec3 view_dir = normalize(-position);

    vec3 detail_normal;
    bool has_normal = scene_normal(v_uv, detail_normal);
    if (!has_normal) {
        detail_normal = depth_normal(v_uv, position);
    }

    // Shape normal: average with continuous neighbours so normal-map grain does
    // not speckle the cel shadow, like shading by smooth mesh normals.
    const vec2 CROSS[4] = vec2[](vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0));
    vec3 shape_normal = detail_normal;
    float ink = 0.0;
    bool behind_step = false;
    float near_depths[4];
    for (int i = 0; i < 4; ++i) {
        vec2 near_uv = v_uv + CROSS[i] * texel;
        float near_depth = view_depth(near_uv);
        near_depths[i] = near_depth;
        // Ink on the nearer side of silhouettes and depth steps.
        if (near_depth - center_depth > max(0.04, center_depth * 0.03)) {
            ink = 1.0;
        }
        behind_step = behind_step
            || center_depth - near_depth > max(0.04, center_depth * 0.03);
        vec3 near_normal;
        if (abs(near_depth - center_depth) <= max(0.02, center_depth * 0.01)
            && scene_normal(near_uv, near_normal)) {
            vec3 far_normal;
            vec2 far_uv = v_uv + CROSS[i] * texel * 3.0;
            if (scene_normal(far_uv, far_normal)
                && abs(view_depth(far_uv) - center_depth) <= max(0.03, center_depth * 0.015)) {
                shape_normal += far_normal;
            }
            shape_normal += near_normal;
        }
    }
    shape_normal = normalize(shape_normal);

    // Interior lines on geometric creases and overlapping layers, found from the
    // depth curvature so normal-map grain never draws lines.
    if (ink < 1.0 && !behind_step) {
        float crease = max(
            abs(near_depths[0] + near_depths[1] - 2.0 * center_depth),
            abs(near_depths[2] + near_depths[3] - 2.0 * center_depth)
        );
        float threshold = max(0.004, center_depth * 0.0015);
        ink = max(ink, 0.85 * smoothstep(threshold, threshold * 2.0, crease));
    }

    vec4 flags = texture(u_material_flags, v_uv);
    float skin = step(0.5, flags.b) * smoothstep(0.02, 0.2, max(
        texture(u_material_subsurface, v_uv).r, texture(u_material_subsurface, v_uv).g
    ));

    // Recover an albedo estimate by removing the viewport's own key lighting,
    // then relight it with a two-tone ramp so shading follows shape, not texture.
    bool lighting_on = dot(u_key_light, u_key_light) > 0.25;
    vec3 light = normalize(lighting_on ? u_key_light : FALLBACK_LIGHT);
    float source_ndl = max(dot(detail_normal, light), 0.0);
    vec3 albedo = lighting_on ? source.rgb / mix(0.65, 1.0, source_ndl) : source.rgb;
    albedo = min(albedo, vec3(1.0));

    float ndl = dot(shape_normal, light);
    float lit = smoothstep(
        -TERMINATOR_SOFTNESS, TERMINATOR_SOFTNESS, ndl + TERMINATOR_OFFSET
    );
    vec3 rich = saturate_color(albedo, 1.06);
    vec3 shadow_tint = mix(CLOTH_SHADOW, SKIN_SHADOW, skin);
    vec3 color = mix(rich * shadow_tint, albedo * LIT_TINT, lit);
    float terminator = 1.0 - smoothstep(0.0, 0.22, abs(ndl + TERMINATOR_OFFSET + 0.04));
    color = mix(color, color * SKIN_TERMINATOR, skin * terminator);

    // Crisp highlight where the environment mask says the surface is glossy.
    if (flags.g > 0.5) {
        float gloss = texture(u_material_env_mask, v_uv).r;
        vec3 half_dir = normalize(light + view_dir);
        float highlight = smoothstep(0.955, 0.97, dot(detail_normal, half_dir));
        color += (albedo * 0.12 + 0.15) * gloss * highlight * lit;
    }

    // Rim light: offset towards the silhouette along the screen-space normal and
    // light pixels whose neighbour there is much farther away or background.
    vec2 rim_dir = shape_normal.xy;
    if (dot(rim_dir, rim_dir) > 0.0025) {
        rim_dir = normalize(rim_dir);
        float rim_depth = view_depth(v_uv + rim_dir * texel * RIM_WIDTH);
        float rim = smoothstep(max(0.05, center_depth * 0.03), max(0.12, center_depth * 0.08),
                               rim_depth - center_depth);
        float facing = 1.0 - max(dot(shape_normal, view_dir), 0.0);
        float light_side = mix(0.35, 1.0, smoothstep(-0.3, 0.6, dot(rim_dir, normalize(light.xy + 1.0e-5))));
        rim *= smoothstep(0.15, 0.5, facing) * light_side;
        color += (albedo * 0.5 + vec3(0.32, 0.34, 0.40)) * rim * RIM_STRENGTH;
    }

    // Clean, slightly high-key grade.
    color = saturate_color(color, 1.03);
    color = color * 0.89 + 0.008;

    // Coloured ink: a darkened, saturated version of the surface colour.
    vec3 ink_color = saturate_color(albedo, 1.3) * INK;
    color = mix(color, ink_color, ink * mix(1.0, 0.6, skin));

    fragColor = vec4(clamp(color, 0.0, 1.0), source.a);
}
