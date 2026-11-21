#version 330

// Dark Fantasy: Grimdark, Souls-like chiaroscuro lighting with a strong warm
// key over a near-black cool fill, desaturated albedo graded slate-blue in
// shadow and warm where key-lit, gleaming metal with tinted studio reflections,
// and a cold moonlight Fresnel rim. All materials (including skin) share the
// same relighting pass; writes a split output for "ssss_after_preset": true.

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;
// RGB: specular added after SSSS. A: SSSS mask.
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
// Exponent on the key's N.L: higher drops more of the form into shadow.
uniform float u_contrast_power = 1.35;
uniform float u_rim_intensity = 1.0;
// Multiplier on metal highlights and studio reflections.
uniform float u_metal_boost = 1.0;

const float BACKGROUND_DEPTH = 0.999999;
const vec3 LUMINANCE = vec3(0.2126, 0.7152, 0.0722);
const vec3 FALLBACK_LIGHT = vec3(-0.3537, 0.5558, 0.7680);
const float PI = 3.14159265;

// Chiaroscuro: strong warm key over a very low, cool fill.
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

// Unified chiaroscuro relight for all surfaces (skin and non-skin alike).
vec3 relight(vec3 color, vec3 normal, vec3 light, out float lit) {
    float n_dot_l = dot(normal, light);
    vec3 albedo = min(color / mix(0.65, 1.0, max(n_dot_l, 0.0)), vec3(1.0));
    float key = pow(max(n_dot_l, 0.0), max(u_contrast_power, 0.1));
    lit = key;
    return albedo * (AMBIENT + KEY_COLOR * key);
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

// Highlight and environment reflection from env mask or tangent normal alpha.
// Metal (env mask >= 0.3) gets a sharper highlight and tinted studio reflection.
vec3 armor_specular(vec2 uv, vec3 view, vec3 color, vec3 normal, vec3 light, vec4 flags) {
    vec4 mask = flags.g > 0.5 ? texture(u_material_env_mask, uv) : vec4(0.0);
    float reflection = flags.g > 0.5 ? mask.r
        : (flags.r > 0.5 && flags.a <= 0.5 ? texture(u_material_normal, uv).a * 0.35 : 0.0);
    if (reflection <= 0.004) {
        return vec3(0.0);
    }
    float glossiness = mask.g > 4.0 / 255.0 ? mask.g : 0.25;
    float metal = flags.g > 0.5 && mask.r >= METAL_MASK ? 1.0 : 0.0;

    vec3 half_vector = normalize(light + view);
    vec3 tint = color / max(max(color.r, max(color.g, color.b)), 1.0e-4);
    vec3 f0 = mix(vec3(0.04), tint * 0.8, metal);
    vec3 fresnel = f0 + (1.0 - f0) * pow(max(1.0 - max(dot(view, half_vector), 0.0), 0.0), 5.0);

    float power = exp2(mix(3.0, 9.0, glossiness)) * mix(1.0, METAL_SHARPEN, metal);
    float lobe = (power + 8.0) / (8.0 * PI) * pow(max(dot(normal, half_vector), 0.0), power)
        * max(dot(normal, light), 0.0);
    float lit = mix(0.35, 1.0, smoothstep(0.03, 0.45, dot(color, LUMINANCE)));
    vec3 highlight = fresnel * min(lobe, 32.0) * lit * 0.25 * mix(0.45, 1.0, glossiness);

    float grazing = pow(max(1.0 - max(dot(normal, view), 0.0), 0.0), 5.0);
    vec3 environment;
    if (metal > 0.5) {
        vec3 studio = textureLod(u_studio_cubemap, u_view_to_world * reflect(-view, normal),
                                 mix(5.0, 1.0, glossiness)).rgb;
        environment = studio * tint * mix(0.8, 1.0, grazing) * METAL_REFLECTION;
    } else {
        float sky = reflect(-view, normal).y * 0.5 + 0.5;
        environment = vec3(mix(0.06, 0.35, sky)) * mix(f0, vec3(1.0), grazing * glossiness)
            * mix(0.15, 1.0, glossiness * glossiness) * 0.4;
    }

    vec3 result = (highlight + environment) * reflection * mix(1.0, u_metal_boost, metal)
        * specular_breakup(uv, color);
    return result / (1.0 + max(result.r, max(result.g, result.b)) * 1.2);
}

// Cold moonlight Fresnel rim on the side facing the key.
vec3 moon_rim(vec3 normal, vec3 view, vec3 light) {
    float rim = pow(max(1.0 - max(dot(normal, view), 0.0), 0.0), RIM_POWER)
        * smoothstep(-0.3, 0.5, dot(normal, light));
    return RIM_COLOR * u_rim_intensity * rim;
}

void main() {
    vec4 source = texture(u_scene_color, v_uv);
    float depth = texture(u_scene_depth, v_uv).r;
    vec3 center = view_position(v_uv, depth);

    // Compute screen-space derivatives in uniform control flow for smooth skin rim normals.
    vec3 dx = dFdx(center);
    vec3 dy = dFdy(center);

    if (depth >= BACKGROUND_DEPTH) {
        fragColor = source;
        fragSpecular = vec4(0.0);
        return;
    }

    vec3 color = source.rgb;
    vec4 flags = texture(u_material_flags, v_uv);
    float skin = float(flags.b > 0.5);
    bool light_on = dot(u_key_light, u_key_light) > 0.25;
    vec3 light = normalize(light_on ? u_key_light : FALLBACK_LIGHT);
    vec3 view = normalize(-center);

    float lit = smoothstep(0.08, 0.55, dot(color, LUMINANCE));
    vec3 normal;
    vec3 shine = vec3(0.0);
    if (scene_normal(v_uv, normal)) {
        if (dot(normal, view) < 0.0) {
            normal = -normal;
        }
        if (light_on) {
            color = relight(color, normal, light, lit);
        }

        // Keep crisp normal-map rims on armour; use smooth geometric normal on skin.
        vec3 rim_normal = normal;
        if (skin > 0.5) {
            vec3 geo = cross(dx, dy);
            if (dot(geo, geo) > 1.0e-12 && max(abs(dx.z), abs(dy.z)) < -center.z * 0.04) {
                rim_normal = normalize(dot(geo, view) < 0.0 ? -geo : geo);
            }
        }

        shine = armor_specular(v_uv, view, color, normal, light, flags)
            + moon_rim(rim_normal, view, light);
    }

    color = max(mix(vec3(dot(color, LUMINANCE)), color, ALBEDO_SATURATION), 0.0);
    color *= mix(SHADOW_GRADE, LIT_GRADE, clamp(lit, 0.0, 1.0));
    color *= EXPOSURE;
    float luminance = dot(color, LUMINANCE);
    color *= 1.0 + CONTRAST * (luminance - 0.5) * (1.0 - luminance);

    fragColor = vec4(clamp(color, 0.0, 1.0), source.a);
    fragSpecular = vec4(max(shine, 0.0), skin);
}
