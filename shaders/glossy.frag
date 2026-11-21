#version 330

// Glossy: Vivid plus oil, the classic Skyrim "Korean-style" oiled skin.
// Shares Vivid's matte diffuse base (soft two-tone cloth relight, warm pink
// skin flush, combined saturation/contrast grade, smooth geometric rim light)
// and adds dual-lobe GGX oil specular, studio cubemap reflection, and wet
// darkening. Relies on pre-pass GTAO and SSSS (single-channel output).

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragSpecular;

uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;
uniform sampler2D u_scene_normal;
uniform sampler2D u_material_normal;
uniform sampler2D u_material_env_mask;
uniform sampler2D u_material_flags;
uniform samplerCube u_studio_cubemap;
uniform mat3 u_view_to_world;
uniform vec2 u_viewport_size;
uniform mat4 u_inverse_projection;
// View-space direction towards the key light; zero while scene lighting is off.
uniform vec3 u_key_light;
uniform float u_gloss_intensity = 1.0;
uniform float u_rim_intensity = 1.0;

const float BACKGROUND_DEPTH = 0.999999;
const vec3 LUMINANCE = vec3(0.2126, 0.7152, 0.0722);
const vec3 FALLBACK_LIGHT = vec3(-0.3537, 0.5558, 0.7680);
const float PI = 3.14159265;

// Vivid base: studio relight for non-skin surfaces (soft two-tone ramp).
const vec3 KEY_COLOR = vec3(0.70, 0.66, 0.62);
const vec3 SKY_AMBIENT = vec3(0.46, 0.46, 0.52);
const vec3 GROUND_AMBIENT = vec3(0.40, 0.34, 0.31);
const float TOON_AMOUNT = 0.35;

// Vivid base: colour grade (combined saturation) and warm pink skin flush.
const float EXPOSURE = 1.12;
const float CONTRAST = 0.45;
const float SATURATION = 1.2;
const vec3 FLUSH_TINT = vec3(1.10, 0.90, 0.90);
const float FLUSH_STRENGTH = 0.55;

// Vivid base: rim light.
const vec3 RIM_COLOR = vec3(1.0, 0.92, 0.93);
const float RIM_POWER = 4.0;
const float RIM_SKIN = 0.05;
const float RIM_CLOTH = 0.15;

// Oil layer.
const vec3 OIL_LIGHT = vec3(0.85, 0.82, 0.78);
const float BROAD_ROUGHNESS = 0.40;
const float BROAD_INTENSITY = 3.2;
const float SHARP_ROUGHNESS = 0.11;
const float SHARP_INTENSITY = 4.0;
const float SHARP_LIMIT = 9.0;
const float DIELECTRIC_F0 = 0.05;
const float SKIN_MASK_FLOOR = 0.6;
const float OIL_RIM_POWER = 3.5;
const float RIM_SHARP_BOOST = 4.0;
const float REFLECTION_STRENGTH = 1.0;
const float REFLECTION_FLOOR = 0.2;
const float REFLECTION_LOD = 2.0;
const float BREAKUP_AMOUNT = 0.3;
const float OIL_SCALE = 1.5;

// Skin-specific oil attenuation (1.0 = full oil, 0.25 = subtle sheen, 0.0 = matte skin).
const float SKIN_OIL_COVERAGE_SCALE = 0.12;
const float SKIN_OIL_INTENSITY_SCALE = 0.12;

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

vec3 relight(vec3 color, vec3 normal, vec3 light) {
    float n_dot_l = dot(normal, light);
    vec3 albedo = min(color / mix(0.65, 1.0, max(n_dot_l, 0.0)), vec3(1.0));
    float key = clamp((n_dot_l + 0.08) / 1.08, 0.0, 1.0);
    key = mix(key, smoothstep(0.05, 0.5, key), TOON_AMOUNT);
    vec3 ambient = mix(GROUND_AMBIENT, SKY_AMBIENT, normal.y * 0.5 + 0.5);
    return albedo * (ambient + KEY_COLOR * key);
}

vec3 rim_light(vec3 normal, vec3 view, vec3 light, float strength) {
    float rim = pow(max(1.0 - max(dot(normal, view), 0.0), 0.0), RIM_POWER);
    float lit = smoothstep(-0.2, 0.4, dot(normal, light));
    return RIM_COLOR * rim * lit * strength;
}

// Texture valleys (pores, weave gaps) catch less light than surrounding peaks.
float specular_breakup(vec2 uv, vec3 color) {
    vec2 texel = 1.0 / u_viewport_size;
    float neighbours = dot(
        texture(u_scene_color, uv + vec2(1.5, 0.5) * texel).rgb +
        texture(u_scene_color, uv + vec2(-0.5, 1.5) * texel).rgb +
        texture(u_scene_color, uv + vec2(-1.5, -0.5) * texel).rgb +
        texture(u_scene_color, uv + vec2(0.5, -1.5) * texel).rgb, LUMINANCE) * 0.25;
    float center = dot(color, LUMINANCE);
    return clamp(1.0 + (center - neighbours) / max(center, 0.05) * 2.5, 0.35, 1.5);
}

float ggx(float n_dot_h, float roughness) {
    float alpha2 = pow(roughness, 4.0);
    float denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denominator * denominator);
}

// Dual-lobe oil specular plus a grazing studio reflection.
vec3 oil(vec3 view, vec3 normal, vec3 light, vec3 f0, vec3 tint, float metalness) {
    vec3 half_vector = normalize(light + view);
    float n_dot_h = max(dot(normal, half_vector), 0.0);
    float n_dot_v = max(dot(normal, view), 0.0);
    float v_dot_h = max(dot(view, half_vector), 0.0);
    vec3 fresnel = f0 + (1.0 - f0) * pow(max(1.0 - v_dot_h, 0.0), 5.0);
    vec3 common_term = PI * fresnel * (0.25 / max(v_dot_h * v_dot_h, 0.05))
        * max(dot(normal, light), 0.0) * OIL_LIGHT;
    vec3 broad = common_term * ggx(n_dot_h, BROAD_ROUGHNESS) * BROAD_INTENSITY;
    vec3 sharp = min(common_term * ggx(n_dot_h, SHARP_ROUGHNESS) * SHARP_INTENSITY,
                     vec3(SHARP_LIMIT));
    float rim = pow(max(1.0 - n_dot_v, 0.0), OIL_RIM_POWER);
    sharp *= 1.0 + RIM_SHARP_BOOST * rim;
    vec3 studio = textureLod(u_studio_cubemap, u_view_to_world * reflect(-view, normal),
                             REFLECTION_LOD).rgb * mix(vec3(1.0), tint, metalness);
    return broad + sharp + studio * mix(REFLECTION_FLOOR, 1.0, rim) * REFLECTION_STRENGTH;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_viewport_size;
    vec4 source = texture(u_scene_color, uv);
    float depth = texture(u_scene_depth, uv).r;
    vec3 center = view_position(uv, depth);

    // Compute screen-space derivatives in uniform control flow for smooth rim normals.
    vec3 dx = dFdx(center);
    vec3 dy = dFdy(center);

    if (depth >= BACKGROUND_DEPTH) {
        fragSpecular = vec4(0.0);
        fragColor = source;
        return;
    }

    vec3 color = source.rgb;
    vec4 flags = texture(u_material_flags, uv);
    float skin = float(flags.b > 0.5);
    bool light_on = dot(u_key_light, u_key_light) > 0.25;
    vec3 light = normalize(light_on ? u_key_light : FALLBACK_LIGHT);
    vec3 view = normalize(-center);

    // Oil coverage: env mask, else tangent normal alpha; floored on skin.
    vec4 mask = flags.g > 0.5 ? texture(u_material_env_mask, uv) : vec4(0.0);
    bool tangent_normal = flags.r > 0.5 && flags.a <= 0.5;
    float normal_alpha = tangent_normal ? texture(u_material_normal, uv).a : 0.0;
    float spec_mask = flags.g > 0.5 ? mask.r : (tangent_normal ? normal_alpha : 0.5);
    float gloss = flags.g > 0.5 ? mask.r : normal_alpha * 0.8;
    float skin_coverage = mix(SKIN_MASK_FLOOR, 1.0, spec_mask) * SKIN_OIL_COVERAGE_SCALE;
    float coverage = clamp(mix(gloss, skin_coverage, skin), 0.0, 1.0);

    vec3 normal;
    vec3 shine = vec3(0.0);
    if (scene_normal(uv, normal)) {
        if (dot(normal, view) < 0.0) {
            normal = -normal;
        }
        if (light_on && skin < 0.5) {
            color = relight(color, normal, light);
        }

        // Warm pink flush on skin, strongest along the light terminator.
        float terminator = 1.0 - smoothstep(0.0, 0.6, abs(dot(normal, light) - 0.1));
        color *= mix(vec3(1.0), FLUSH_TINT, skin * FLUSH_STRENGTH * (0.6 + 0.4 * terminator));

        // Smooth geometric normal for rim light to avoid normal-map compression blotches.
        vec3 geo = cross(dx, dy);
        float geo_len2 = dot(geo, geo);
        vec3 rim_normal = (geo_len2 > 1.0e-12 && max(abs(dx.z), abs(dy.z)) < -center.z * 0.04)
            ? normalize(dot(geo, view) < 0.0 ? -geo : geo)
            : normal;
        shine = rim_light(rim_normal, view, light, mix(RIM_CLOTH, RIM_SKIN, skin))
            * u_rim_intensity;

        // Dual-lobe oil specular and studio reflection.
        if (coverage > 0.004) {
            float metalness = flags.g > 0.5
                && max(abs(mask.g - mask.r), abs(mask.b - mask.r)) > 0.04
                ? mask.b * (1.0 - skin) : 0.0;
            vec3 albedo_tint = source.rgb / max(max(source.r, max(source.g, source.b)), 1.0e-4);
            vec3 f0 = mix(vec3(DIELECTRIC_F0), albedo_tint * 0.8, metalness);
            float breakup = mix(1.0, specular_breakup(uv, source.rgb), BREAKUP_AMOUNT);
            float skin_oil_scale = mix(1.0, SKIN_OIL_INTENSITY_SCALE, skin);
            shine += oil(view, normal, light, f0, albedo_tint, metalness)
                * coverage * breakup * OIL_SCALE * skin_oil_scale * u_gloss_intensity;
        }
    }
    // Vivid colour grade (combined saturation, exposure, S-curve contrast).
    color = max(mix(vec3(dot(color, LUMINANCE)), color, SATURATION), 0.0);
    color *= EXPOSURE;
    float luminance = dot(color, LUMINANCE);
    color *= 1.0 + CONTRAST * (luminance - 0.5) * (1.0 - luminance);
    color = clamp(color, 0.0, 1.0);

    fragColor = vec4(color, source.a);
    fragSpecular = vec4(shine, float(skin));
}