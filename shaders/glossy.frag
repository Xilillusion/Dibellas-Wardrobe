#version 330

// Glossy: 3D Hentai plus oil, the classic Skyrim "Korean-style" oiled skin.
// The diffuse is 3D Hentai's (soft two-tone key, smoothed pink-flushed skin,
// saturated grade) under a wet film, its rim light is kept, and its skin gloss
// is replaced by oil: dual-lobe GGX specular (broad sheen + sharp wet
// pinpoints) and a grazing studio reflection. glossy.json sets
// "ssss_after_preset", so this writes a split output: diffuse goes through the
// built-in SSSS pass, while oil and rim are added back afterwards, unblurred.

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;
// RGB: specular added after SSSS. A: SSSS mask.
layout(location = 1) out vec4 fragSpecular;

uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;
uniform sampler2D u_scene_normal;
uniform sampler2D u_material_normal;
uniform sampler2D u_material_env_mask;
uniform sampler2D u_material_subsurface;
uniform sampler2D u_material_flags;
uniform samplerCube u_studio_cubemap;
uniform mat3 u_view_to_world;
uniform vec2 u_viewport_size;
uniform mat4 u_inverse_projection;
// View-space direction towards the key light; zero while scene lighting is off.
uniform vec3 u_key_light;
// Multiplier on the oil specular and studio reflection.
uniform float u_gloss_intensity = 1.0;
uniform float u_rim_intensity = 1.0;

const float BACKGROUND_DEPTH = 0.999999;
const vec3 LUMINANCE = vec3(0.2126, 0.7152, 0.0722);
// Upper-left key light in view space, used while scene lighting is off.
const vec3 FALLBACK_LIGHT = vec3(-0.3537, 0.5558, 0.7680);
const float PI = 3.14159265;

// 3D Hentai: Standard with a slightly softer, warmer studio light.
const vec3 KEY_COLOR = vec3(0.70, 0.66, 0.62);
const vec3 SKY_AMBIENT = vec3(0.46, 0.46, 0.52);
const vec3 GROUND_AMBIENT = vec3(0.40, 0.34, 0.31);
const vec3 SKIN_WRAP = vec3(0.50, 0.26, 0.20);
const float EXPOSURE = 1.12;
const float CONTRAST = 0.45;
const float SATURATION = 1.15;

// Share of the light ramp pulled towards a soft two-tone band.
const float TOON_AMOUNT = 0.35;
// Skin keeps this share of its normal-map detail, for smooth, doll-like skin.
const float SKIN_NORMAL_DETAIL = 0.4;
// Pink flush on skin, strongest along the light terminator.
const vec3 FLUSH_TINT = vec3(1.08, 0.86, 0.90);
const float FLUSH_STRENGTH = 0.55;

// Oil.
const vec3 OIL_LIGHT = vec3(0.85, 0.82, 0.78);
const float BROAD_ROUGHNESS = 0.40;
const float BROAD_INTENSITY = 3.2;
const float SHARP_ROUGHNESS = 0.11;
const float SHARP_INTENSITY = 4.0;
const float SHARP_LIMIT = 9.0;
// Slightly above skin's real 0.028: an oil film reads glossier than bare skin.
const float DIELECTRIC_F0 = 0.05;
// Skin keeps a continuous sheen even where its specular mask is low.
const float SKIN_MASK_FLOOR = 0.6;
const float OIL_RIM_POWER = 3.5;
const float RIM_SHARP_BOOST = 4.0;
const float REFLECTION_STRENGTH = 1.0;
// Facing-on reflection share; a wet film mirrors the studio even head-on.
const float REFLECTION_FLOOR = 0.2;
const float REFLECTION_LOD = 2.0;
// Wet surfaces look darker and richer: the film cuts diffuse back-scatter.
const float WET_DARKENING = 0.18;
const float WET_SATURATION = 1.25;
const vec3 WARM_SKIN = vec3(1.04, 0.99, 0.96);
// Share of Standard's texture-driven highlight breakup kept on the oil film.
const float BREAKUP_AMOUNT = 0.3;
const float OIL_SCALE = 0.88;

// Rim light: a bright, slightly pink Fresnel edge on the lit side.
const vec3 RIM_COLOR = vec3(1.0, 0.90, 0.92);
const float RIM_POWER = 3.0;
const float RIM_SKIN = 0.40;
const float RIM_CLOTH = 0.15;

vec3 view_position(vec2 uv, float depth) {
    vec4 position = u_inverse_projection * vec4(vec3(uv, depth) * 2.0 - 1.0, 1.0);
    return position.xyz / position.w;
}

// View position of a neighbouring surface pixel; false off-screen or on background.
bool surface_at(vec2 uv, out vec3 position) {
    float depth = texture(u_scene_depth, uv).r;
    if (any(lessThanEqual(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0))) ||
        depth >= BACKGROUND_DEPTH) {
        return false;
    }
    position = view_position(uv, depth);
    return true;
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

// Geometric normal from depth, using the side nearest in depth so it does not
// span a silhouette. pixel_size is the view-space width of one pixel.
bool depth_normal(vec2 uv, vec3 center, out vec3 normal, out float pixel_size) {
    vec2 texel = 1.0 / u_viewport_size;
    vec3 left, right, down, up;
    if (!surface_at(uv - vec2(texel.x, 0.0), left) || !surface_at(uv + vec2(texel.x, 0.0), right) ||
        !surface_at(uv - vec2(0.0, texel.y), down) || !surface_at(uv + vec2(0.0, texel.y), up)) {
        return false;
    }
    vec3 dx = abs(left.z - center.z) < abs(right.z - center.z) ? center - left : right - center;
    vec3 dy = abs(down.z - center.z) < abs(up.z - center.z) ? center - down : up - center;
    pixel_size = max(length(dx), length(dy));
    float step_limit = max(-center.z * 0.02, pixel_size * 0.8);
    vec3 n = cross(dx, dy);
    if (pixel_size < 1.0e-6 || abs(dx.z) > step_limit || abs(dy.z) > step_limit ||
        dot(n, n) < 1.0e-12) {
        return false;
    }
    normal = normalize(n) * sign(dot(n, -center));
    return true;
}

// Darkening where nearby geometry rises above the surface (creases, contacts).
// Horizons are measured against the true surface, so a tilted normal map on a
// flat surface does not occlude itself; the shading normal only reweights them.
float contact_occlusion(vec2 uv, vec3 center, vec3 normal, vec3 shading_normal,
                        float pixel_size) {
    const vec2 offsets[8] = vec2[](
        vec2(3.0, 0.0), vec2(-3.0, 0.0), vec2(0.0, 3.0), vec2(0.0, -3.0),
        vec2(5.66, 5.66), vec2(-5.66, 5.66), vec2(5.66, -5.66), vec2(-5.66, -5.66)
    );
    vec2 texel = 1.0 / u_viewport_size;
    float radius = pixel_size * 12.0;
    float occlusion = 0.0;
    for (int i = 0; i < 8; ++i) {
        vec3 sample_position;
        if (!surface_at(uv + offsets[i] * texel, sample_position)) {
            continue;
        }
        vec3 delta = sample_position - center;
        float distance_to_sample = length(delta);
        if (distance_to_sample > 1.0e-6) {
            vec3 direction = delta / distance_to_sample;
            float facing = dot(normal, direction);
            float horizon = max(facing - 0.08, 0.0);
            float normal_factor = clamp(
                (dot(shading_normal, direction) + 0.4) / (facing + 0.4), 0.6, 1.4
            );
            occlusion += horizon * normal_factor
                * (1.0 - smoothstep(radius * 0.25, radius, distance_to_sample));
        }
    }
    return clamp(occlusion * 0.2, 0.0, 0.35);
}

// Replace the viewport's flat Lambert shading: divide out the baked lighting to
// recover albedo, then relight it with the key and sky / ground ambient. The
// key is partly pulled into a soft two-tone ramp.
vec3 relight(vec3 color, vec3 normal, vec3 light, float scatter) {
    float n_dot_l = dot(normal, light);
    vec3 albedo = min(color / mix(0.65, 1.0, max(n_dot_l, 0.0)), vec3(1.0));
    vec3 wrap = vec3(0.08) + SKIN_WRAP * scatter;
    vec3 key = clamp((n_dot_l + wrap) / (1.0 + wrap), 0.0, 1.0);
    key = mix(key, smoothstep(vec3(0.05), vec3(0.5), key), TOON_AMOUNT);
    vec3 ambient = mix(GROUND_AMBIENT, SKY_AMBIENT, normal.y * 0.5 + 0.5);
    return albedo * (ambient + KEY_COLOR * key);
}

// Skin: blend in neighbouring skin pixels that face the same way, and bleed the
// _sk tint into the shadowed side.
vec3 subsurface_scatter(vec2 uv, vec3 center, vec3 color, vec3 normal, float pixel_size,
                        vec3 tint, float strength) {
    const vec2 offsets[8] = vec2[](
        vec2(2.0, 0.0), vec2(-2.0, 0.0), vec2(0.0, 2.0), vec2(0.0, -2.0),
        vec2(3.5, 3.5), vec2(-3.5, 3.5), vec2(3.5, -3.5), vec2(-3.5, -3.5)
    );
    vec2 texel = 1.0 / u_viewport_size;
    vec3 sum = color;
    float weight = 1.0;
    for (int i = 0; i < 8; ++i) {
        // Texel centres, so the filtered colour only mixes vetted neighbours.
        vec2 sample_uv = (floor(uv * u_viewport_size + offsets[i]) + 0.5) * texel;
        vec3 sample_position;
        vec3 sample_normal;
        if (!surface_at(sample_uv, sample_position) ||
            abs(dot(sample_position - center, normal)) > pixel_size * 4.0 ||
            texture(u_material_flags, sample_uv).b <= 0.5) {
            continue;
        }
        float sample_weight = i < 4 ? 1.6 : 0.9;
        if (scene_normal(sample_uv, sample_normal)) {
            sample_weight *= smoothstep(0.55, 0.9, dot(normal, sample_normal));
        }
        sum += texture(u_scene_color, sample_uv).rgb * sample_weight;
        weight += sample_weight;
    }
    vec3 diffused = sum / weight;
    float shadow = 1.0 - smoothstep(0.08, 0.6, dot(color, LUMINANCE));
    vec3 scattered = mix(color, diffused, strength * 0.5)
        + tint * dot(diffused, LUMINANCE) * shadow * 0.14;
    return clamp(scattered, 0.0, 1.0);
}

// Texture valleys (pores, weave gaps) catch less light than the peaks around
// them, so highlights do not read as a smooth plastic coating.
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
vec3 oil(vec3 center, vec3 color, vec3 normal, vec3 light, vec3 f0, vec3 tint, float metalness) {
    vec3 view = normalize(-center);
    vec3 half_vector = normalize(light + view);
    float n_dot_h = max(dot(normal, half_vector), 0.0);
    float n_dot_v = max(dot(normal, view), 0.0);
    float v_dot_h = max(dot(view, half_vector), 0.0);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
    // Kelemen visibility; PI matches the diffuse, which is not divided by PI.
    vec3 common_term = PI * fresnel * (0.25 / max(v_dot_h * v_dot_h, 0.05))
        * max(dot(normal, light), 0.0) * OIL_LIGHT;
    vec3 broad = common_term * ggx(n_dot_h, BROAD_ROUGHNESS) * BROAD_INTENSITY;
    vec3 sharp = min(common_term * ggx(n_dot_h, SHARP_ROUGHNESS) * SHARP_INTENSITY,
                     vec3(SHARP_LIMIT));
    // Grazing glow along contours: boosts the wet lobe and reflects the studio.
    float rim = pow(1.0 - n_dot_v, OIL_RIM_POWER);
    sharp *= 1.0 + RIM_SHARP_BOOST * rim;
    vec3 studio = textureLod(u_studio_cubemap, u_view_to_world * reflect(-view, normal),
                             REFLECTION_LOD).rgb * mix(vec3(1.0), tint, metalness);
    return broad + sharp + studio * mix(REFLECTION_FLOOR, 1.0, rim) * REFLECTION_STRENGTH;
}

// Fresnel rim on the side facing the light, fading into the shadow side.
vec3 rim_light(vec3 normal, vec3 view, vec3 light, float strength) {
    float rim = pow(1.0 - max(dot(normal, view), 0.0), RIM_POWER);
    float lit = smoothstep(-0.2, 0.4, dot(normal, light));
    return RIM_COLOR * rim * lit * strength;
}

void main() {
    vec4 source = texture(u_scene_color, v_uv);
    fragSpecular = vec4(0.0);
    float depth = texture(u_scene_depth, v_uv).r;
    if (depth >= BACKGROUND_DEPTH) {
        fragColor = source;
        return;
    }

    vec3 color = source.rgb;
    vec3 center = view_position(v_uv, depth);
    vec4 flags = texture(u_material_flags, v_uv);
    vec3 tint = flags.b > 0.5 ? texture(u_material_subsurface, v_uv).rgb : vec3(0.0);
    float scatter = clamp(max(tint.r, max(tint.g, tint.b)), 0.0, 1.0);
    float skin = smoothstep(0.02, 0.2, max(tint.r, tint.g));
    bool light_on = dot(u_key_light, u_key_light) > 0.25;
    vec3 light = normalize(light_on ? u_key_light : FALLBACK_LIGHT);

    // 3D Hentai diffuse. Silhouette pixels without a depth normal keep the
    // scene colour, but still get oil and rim from their own shading normal.
    vec3 geometric;
    float pixel_size;
    float occlusion = 0.0;
    vec3 normal;
    bool has_geometry = depth_normal(v_uv, center, geometric, pixel_size);
    if (!scene_normal(v_uv, normal) || (has_geometry && dot(normal, geometric) <= 0.1)) {
        normal = has_geometry ? geometric : vec3(0.0, 0.0, 1.0);
    }
    if (has_geometry) {
        normal = normalize(mix(normal, geometric, skin * (1.0 - SKIN_NORMAL_DETAIL)));
        if (light_on) {
            color = relight(color, normal, light, scatter);
        }
        if (scatter > 0.0) {
            color = subsurface_scatter(v_uv, center, color, normal, pixel_size,
                                       tint / scatter, scatter);
        }
        occlusion = contact_occlusion(v_uv, center, geometric, normal, pixel_size)
            * (1.0 - 0.45 * scatter);
        color *= 1.0 - occlusion;

        // Pink flush, deepest along the terminator where blood shows through.
        float terminator = 1.0 - smoothstep(0.0, 0.6, abs(dot(normal, light) - 0.1));
        color *= mix(vec3(1.0), FLUSH_TINT, skin * FLUSH_STRENGTH * (0.6 + 0.4 * terminator));
    }

    // Oil coverage: the env mask, else the tangent normal map's alpha. Skin is
    // floored, and assumes a mid mask with neither (model-space bodies).
    vec4 mask = flags.g > 0.5 ? texture(u_material_env_mask, v_uv) : vec4(0.0);
    bool tangent_normal = flags.r > 0.5 && flags.a <= 0.5;
    float normal_alpha = tangent_normal ? texture(u_material_normal, v_uv).a : 0.0;
    float spec_mask = flags.g > 0.5 ? mask.r : (tangent_normal ? normal_alpha : 0.5);
    float gloss = flags.g > 0.5 ? mask.r : normal_alpha * 0.8;
    float coverage = clamp(mix(gloss, mix(SKIN_MASK_FLOOR, 1.0, spec_mask), skin), 0.0, 1.0);

    // Wet film on the diffuse: warmer on skin, darker and richer where oiled.
    color *= mix(vec3(1.0), WARM_SKIN, mix(0.5, 1.0, skin));
    vec3 soaked = max(mix(vec3(dot(color, LUMINANCE)), color, WET_SATURATION), 0.0)
        * (1.0 - WET_DARKENING);
    color = mix(color, soaked, coverage);

    // Oil and rim do not catch the key or the studio deep inside creases.
    float visibility = 1.0 - min(occlusion * 2.5, 0.85);
    vec3 view = normalize(-center);
    vec3 shine = rim_light(normal, view, light, mix(RIM_CLOTH, RIM_SKIN, skin)) * visibility
        * u_rim_intensity;
    if (coverage > 0.004) {
        // A Complex Material mask with distinct channels carries metalness in blue.
        float metalness = flags.g > 0.5 && max(abs(mask.g - mask.r), abs(mask.b - mask.r)) > 0.04
            ? mask.b * (1.0 - skin) : 0.0;
        vec3 albedo_tint = source.rgb / max(max(source.r, max(source.g, source.b)), 1.0e-4);
        vec3 f0 = mix(vec3(DIELECTRIC_F0), albedo_tint * 0.8, metalness);
        float breakup = mix(1.0, specular_breakup(v_uv, source.rgb), BREAKUP_AMOUNT);
        shine += oil(center, source.rgb, normal, light, f0, albedo_tint, metalness)
            * coverage * visibility * breakup * OIL_SCALE * u_gloss_intensity;
    }

    color = max(mix(vec3(dot(color, LUMINANCE)), color, SATURATION), 0.0);
    color *= EXPOSURE;
    float luminance = dot(color, LUMINANCE);
    color *= 1.0 + CONTRAST * (luminance - 0.5) * (1.0 - luminance);
    fragColor = vec4(clamp(color, 0.0, 1.0), source.a);
    fragSpecular = vec4(max(shine, 0.0), skin * mix(0.85, 1.0, smoothstep(0.2, 0.8, scatter)));
}
