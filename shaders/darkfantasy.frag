#version 330

// Dark Fantasy: Standard restyled for grimdark, Souls-like armour, leather and
// cloaks. Chiaroscuro key lighting over a near-black cool fill, a slightly
// desaturated albedo graded slate-blue in shadow and warm where key-lit,
// starkly gleaming metal with tinted studio
// reflections, a cold moonlight Fresnel rim, and pale, weathered skin.
// darkfantasy.json sets "ssss_after_preset", so this writes a split output:
// diffuse goes through the built-in SSSS pass, while highlights,
// reflections and rim are added back afterwards, unblurred.

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
// Exponent on the key's N.L: higher drops more of the form into shadow.
uniform float u_contrast_power = 1.35;
uniform float u_rim_intensity = 1.0;
// Multiplier on metal highlights and studio reflections.
uniform float u_metal_boost = 1.0;

const float BACKGROUND_DEPTH = 0.999999;
const vec3 LUMINANCE = vec3(0.2126, 0.7152, 0.0722);
const vec3 LUMA_601 = vec3(0.299, 0.587, 0.114);
// Upper-left key light in view space, used while scene lighting is off.
const vec3 FALLBACK_LIGHT = vec3(-0.3537, 0.5558, 0.7680);
const float PI = 3.14159265;

// Chiaroscuro: a strong warm key over a very low, cool fill.
const vec3 KEY_COLOR = vec3(1.04, 0.97, 0.88);
const vec3 AMBIENT = vec3(0.06, 0.07, 0.10);
const float ALBEDO_SATURATION = 0.82;
const vec3 SHADOW_GRADE = vec3(0.84, 0.91, 1.08);
const vec3 LIT_GRADE = vec3(1.03, 1.0, 0.96);
const float EXPOSURE = 1.05;
const float CONTRAST = 0.5;

// Metal is an env mask of at least this much (Skyrim _m.dds).
const float METAL_MASK = 0.3;
const float METAL_SHARPEN = 1.3;
const float METAL_REFLECTION = 1.4;

// Cold moonlight rim.
const vec3 RIM_COLOR = vec3(0.55, 0.72, 0.95);
const float RIM_POWER = 3.0;

// Pale, weathered skin: a muted subsurface tint and a tight wrap.
const vec3 SSS_TINT = vec3(0.75, 0.32, 0.25);
const float SKIN_WRAP = 0.45;

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
// recover albedo, then relight it with a hard key over a near-black cool fill.
// lit is the key's share, used to split the shadow and highlight grade.
vec3 relight(vec3 color, vec3 normal, vec3 light, float scatter, out float lit) {
    float n_dot_l = dot(normal, light);
    vec3 albedo = min(color / mix(0.65, 1.0, max(n_dot_l, 0.0)), vec3(1.0));
    vec3 wrap = SSS_TINT * SKIN_WRAP * scatter;
    vec3 wrapped = clamp((n_dot_l + wrap) / (1.0 + wrap), 0.0, 1.0);
    vec3 key = pow(wrapped, vec3(max(u_contrast_power, 0.1)));
    lit = dot(key, LUMINANCE);
    return albedo * (AMBIENT + KEY_COLOR * key);
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
            if (dot(sample_normal, normal) < 0.0) {
                sample_normal = -sample_normal;
            }
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

// Highlight and environment reflection from the env mask, or else the normal
// map's specular alpha. Metal (env mask >= 0.3) gets a sharper highlight and
// a stronger studio reflection tinted by its albedo.
vec3 armor_specular(vec2 uv, vec3 center, vec3 color, vec3 normal, vec3 light, vec4 flags,
                    float visibility) {
    vec4 mask = flags.g > 0.5 ? texture(u_material_env_mask, uv) : vec4(0.0);
    float reflection = flags.g > 0.5 ? mask.r
        : (flags.r > 0.5 && flags.a <= 0.5 ? texture(u_material_normal, uv).a * 0.35 : 0.0);
    if (reflection <= 0.004) {
        return vec3(0.0);
    }
    float glossiness = mask.g > 4.0 / 255.0 ? mask.g : 0.25;
    float metal = flags.g > 0.5 && mask.r >= METAL_MASK ? 1.0 : 0.0;

    vec3 view = normalize(-center);
    vec3 half_vector = normalize(light + view);
    vec3 tint = color / max(max(color.r, max(color.g, color.b)), 1.0e-4);
    vec3 f0 = mix(vec3(0.04), tint * 0.8, metal);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(view, half_vector), 0.0), 5.0);

    float power = exp2(mix(3.0, 9.0, glossiness)) * mix(1.0, METAL_SHARPEN, metal);
    float lobe = (power + 8.0) / (8.0 * PI) * pow(max(dot(normal, half_vector), 0.0), power)
        * max(dot(normal, light), 0.0);
    float lit = mix(0.35, 1.0, smoothstep(0.03, 0.45, dot(color, LUMINANCE)));
    vec3 highlight = fresnel * min(lobe, 32.0) * lit * 0.25 * mix(0.45, 1.0, glossiness);

    float grazing = pow(1.0 - max(dot(normal, view), 0.0), 5.0);
    vec3 environment;
    if (metal > 0.5) {
        vec3 studio = textureLod(u_studio_cubemap, u_view_to_world * reflect(-view, normal),
                                 mix(5.0, 1.0, glossiness)).rgb;
        environment = studio * tint * mix(0.8, 1.0, grazing) * METAL_REFLECTION;
    } else {
        // A dim, overcast surrounding: dark ground, grey sky.
        float sky = reflect(-view, normal).y * 0.5 + 0.5;
        environment = vec3(mix(0.06, 0.35, sky)) * mix(f0, vec3(1.0), grazing * glossiness)
            * mix(0.15, 1.0, glossiness * glossiness) * 0.4;
    }

    vec3 result = (highlight + environment) * reflection * mix(1.0, u_metal_boost, metal)
        * visibility * specular_breakup(uv, color);
    // Roll highlights off softly; a hard clamp flattens them into plateaus.
    return result / (1.0 + max(result.r, max(result.g, result.b)) * 1.2);
}

// Cold moonlight Fresnel rim on the side facing the key.
vec3 moon_rim(vec3 normal, vec3 view, vec3 light) {
    float rim = pow(1.0 - max(dot(normal, view), 0.0), RIM_POWER)
        * smoothstep(-0.3, 0.5, dot(normal, light));
    return RIM_COLOR * u_rim_intensity * rim;
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

    // Depth geometry controls contact shading, never the lighting normal.
    vec3 geometric;
    float pixel_size;
    bool has_geometry = depth_normal(v_uv, center, geometric, pixel_size);
    vec3 normal;
    bool has_normal = scene_normal(v_uv, normal);
    if (has_normal && has_geometry) {
        if (dot(normal, geometric) < 0.0) {
            normal = -normal;
        }
        has_normal = dot(normal, geometric) >= 0.15;
    }
    float occlusion = 0.0;
    float lit = smoothstep(0.08, 0.55, dot(color, LUMINANCE));
    if (has_normal) {
        if (light_on) {
            color = relight(color, normal, light, scatter, lit);
        }
        if (scatter > 0.0 && has_geometry) {
            color = subsurface_scatter(v_uv, center, color, normal, pixel_size,
                                       SSS_TINT / SSS_TINT.r, scatter);
        }
        if (has_geometry) {
            occlusion = contact_occlusion(v_uv, center, geometric, normal, pixel_size)
                * (1.0 - 0.45 * scatter);
        }
        color *= 1.0 - occlusion;
    } else if (has_geometry) {
        occlusion = contact_occlusion(v_uv, center, geometric, geometric, pixel_size)
            * (1.0 - 0.45 * scatter);
        color *= 1.0 - occlusion;
    }

    vec3 view = normalize(-center);
    float visibility = 1.0 - min(occlusion * 2.5, 0.85);
    vec3 shine = vec3(0.0);
    if (has_normal) {
        shine = armor_specular(v_uv, center, color, normal, light, flags, visibility)
            + moon_rim(normal, view, light) * visibility;
    }

    color = mix(vec3(dot(color, LUMA_601)), color, ALBEDO_SATURATION);
    color *= mix(SHADOW_GRADE, LIT_GRADE, clamp(lit, 0.0, 1.0));
    color *= EXPOSURE;
    float luminance = dot(color, LUMINANCE);
    color *= 1.0 + CONTRAST * (luminance - 0.5) * (1.0 - luminance);
    fragColor = vec4(clamp(color, 0.0, 1.0), source.a);
    fragSpecular = vec4(max(shine, 0.0), skin);
}
