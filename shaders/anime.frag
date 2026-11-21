#version 330

// Anime (Cel): Crisp 2D cel-shaded bands, clean saturated colours,
// stylized specular highlights, and sharp anime rim light.

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragSpecular;

uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;
uniform sampler2D u_scene_normal;
uniform sampler2D u_material_flags;
uniform vec2 u_viewport_size;
uniform mat4 u_inverse_projection;
// View-space direction towards the key light; zero while scene lighting is off.
uniform vec3 u_key_light;
uniform float u_rim_intensity = 1.0;

const float BACKGROUND_DEPTH = 0.999999;
const vec3 LUMINANCE = vec3(0.2126, 0.7152, 0.0722);
const vec3 FALLBACK_LIGHT = vec3(-0.3537, 0.5558, 0.7680);

// Cel lighting & cool/lavender-tinted shadow band for a classic anime look.
const vec3 KEY_COLOR = vec3(1.00, 0.98, 0.96);
const vec3 SHADOW_TINT = vec3(0.70, 0.73, 0.86);
const vec3 SKY_AMBIENT = vec3(0.42, 0.44, 0.52);
const vec3 GROUND_AMBIENT = vec3(0.32, 0.29, 0.34);

// Clearer cel cutoffs: crisp bands on cloth, slightly gentler on skin.
const float SHADOW_THRESHOLD_SKIN = -0.02;
const float SHADOW_SOFTNESS_SKIN = 0.14;
const float SHADOW_THRESHOLD_CLOTH = 0.05;
const float SHADOW_SOFTNESS_CLOTH = 0.03;

// Vibrant, clean anime colour grade & subtle cheek tint.
const float EXPOSURE = 1.10;
const float SATURATION = 1.4;
const float CONTRAST = 0.24;
const vec3 SKIN_BLUSH_TINT = vec3(1.06, 0.91, 0.92);
const float SKIN_BLUSH_STRENGTH = 0.30;

// Stylized cel specular & crisp rim light (no broad greasy oil lobe).
const float SPEC_SKIN_STRENGTH = 0.22;
const float SPEC_CLOTH_STRENGTH = 0.48;
const float SPEC_SHARPNESS = 80.0;
const vec3 RIM_COLOR = vec3(0.96, 0.98, 1.00);
const float RIM_POWER = 3.4;
const float RIM_SKIN = 0.24;
const float RIM_CLOTH = 0.30;

// Silhouette / crease line darkening (0.0 to disable).
const float EDGE_DARKEN_STRENGTH = 0.35;

vec3 view_position(vec2 uv, float depth) {
    vec4 position = u_inverse_projection * vec4(vec3(uv, depth) * 2.0 - 1.0, 1.0);
    return position.xyz / position.w;
}

bool scene_normal(vec2 uv, out vec3 normal) {
    vec4 encoded = texture(u_scene_normal, uv);
    normal = encoded.rgb * 2.0 - 1.0;
    if (encoded.a <= 0.5 || dot(normal, normal) < 0.25) {
        return false;
    }
    normal = normalize(normal);
    return true;
}

vec3 cel_shade(vec3 color, vec3 normal, vec3 light, float skin) {
    float n_dot_l = dot(normal, light);

    float threshold = mix(SHADOW_THRESHOLD_CLOTH, SHADOW_THRESHOLD_SKIN, skin);
    float softness = mix(SHADOW_SOFTNESS_CLOTH, SHADOW_SOFTNESS_SKIN, skin);

    // Primary cel shadow step.
    float lit_step = smoothstep(threshold - softness, threshold + softness, n_dot_l);
    // Subtle mid-tone step on cloth so folds read like layered 2D cels.
    float mid_step = smoothstep(threshold - softness * 3.0, threshold - softness * 1.1, n_dot_l);
    float band = mix(mix(0.50, 0.80, mid_step), 1.0, lit_step);

    vec3 hemi = mix(GROUND_AMBIENT, SKY_AMBIENT, normal.y * 0.5 + 0.5);
    vec3 shadow_col = color * SHADOW_TINT * (0.62 + 0.45 * hemi);
    vec3 lit_col = color * (0.78 + 0.34 * KEY_COLOR + 0.22 * hemi);

    return clamp(mix(shadow_col, lit_col, band), 0.0, 1.0);
}

vec3 cel_specular(vec3 normal, vec3 view, vec3 light, float skin) {
    float n_dot_l = max(dot(normal, light), 0.0);
    if (n_dot_l <= 0.05) {
        return vec3(0.0);
    }
    vec3 half_vec = normalize(light + view);
    float n_dot_h = max(dot(normal, half_vec), 0.0);

    // Hard-edged highlight shape typical of anime hair, eyes, and glossy trim.
    float raw_spec = pow(n_dot_h, SPEC_SHARPNESS);
    float cel_spec = smoothstep(0.45, 0.62, raw_spec);

    float strength = mix(SPEC_CLOTH_STRENGTH, SPEC_SKIN_STRENGTH, skin);
    vec3 spec_tint = mix(KEY_COLOR, vec3(1.00, 0.96, 0.95), skin);
    return spec_tint * cel_spec * strength;
}

vec3 cel_rim(vec3 normal, vec3 view, vec3 light, float strength) {
    float fresnel = pow(max(1.0 - max(dot(normal, view), 0.0), 0.0), RIM_POWER);
    // Crisp band edge so rim light looks like a painted anime highlight line.
    float rim_band = smoothstep(0.48, 0.60, fresnel);
    float key_mask = smoothstep(-0.25, 0.45, dot(normal, light));
    return RIM_COLOR * rim_band * key_mask * strength;
}

float screen_edge_factor(vec2 uv, float depth) {
    if (EDGE_DARKEN_STRENGTH <= 0.0) {
        return 1.0;
    }
    vec2 texel = 1.0 / u_viewport_size;
    float d_l = texture(u_scene_depth, uv + vec2(-texel.x, 0.0)).r;
    float d_r = texture(u_scene_depth, uv + vec2( texel.x, 0.0)).r;
    float d_u = texture(u_scene_depth, uv + vec2(0.0,  texel.y)).r;
    float d_d = texture(u_scene_depth, uv + vec2(0.0, -texel.y)).r;

    float depth_jump = max(
        max(abs(depth - d_l), abs(depth - d_r)),
        max(abs(depth - d_u), abs(depth - d_d))
    );
    float edge = smoothstep(0.0006, 0.0028, depth_jump);
    return 1.0 - edge * EDGE_DARKEN_STRENGTH;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_viewport_size;
    vec4 source = texture(u_scene_color, uv);
    float depth = texture(u_scene_depth, uv).r;
    vec3 center = view_position(uv, depth);

    vec3 dx = dFdx(center);
    vec3 dy = dFdy(center);

    if (depth >= BACKGROUND_DEPTH) {
        fragColor = source;
        fragSpecular = vec4(0.0);
        return;
    }

    vec3 color = source.rgb;
    vec4 flags = texture(u_material_flags, uv);
    float skin = float(flags.b > 0.5);
    bool light_on = dot(u_key_light, u_key_light) > 0.25;
    vec3 light = normalize(light_on ? u_key_light : FALLBACK_LIGHT);
    vec3 view = normalize(-center);

    vec3 normal;
    vec3 shine = vec3(0.0);
    if (scene_normal(uv, normal)) {
        if (dot(normal, view) < 0.0) {
            normal = -normal;
        }

        if (light_on && skin < 0.5) {
            color = cel_shade(color, normal, light, skin);
        }

        // Subtle warm blush along the light terminator.
        float terminator = 1.0 - smoothstep(0.0, 0.65, abs(dot(normal, light) - 0.1));
        color *= mix(vec3(1.0), SKIN_BLUSH_TINT, skin * SKIN_BLUSH_STRENGTH * (0.5 + 0.5 * terminator));

        vec3 geo = cross(dx, dy);
        float geo_len2 = dot(geo, geo);
        vec3 rim_normal = (geo_len2 > 1.0e-12 && max(abs(dx.z), abs(dy.z)) < -center.z * 0.04)
            ? normalize(dot(geo, view) < 0.0 ? -geo : geo)
            : normal;

        shine = cel_specular(normal, view, light, skin)
              + cel_rim(rim_normal, view, light, mix(RIM_CLOTH, RIM_SKIN, skin)) * u_rim_intensity;
    }

    color *= screen_edge_factor(uv, depth);

    // Clean, high-clarity anime colour grade.
    color = max(mix(vec3(dot(color, LUMINANCE)), color, SATURATION), 0.0);
    color *= EXPOSURE;
    float luma = dot(color, LUMINANCE);
    color *= 1.0 + CONTRAST * (luma - 0.5) * (1.0 - luma);
    color = clamp(color, 0.0, 1.0);

    fragColor = vec4(color, source.a);
    fragSpecular = vec4(shine, skin);
}