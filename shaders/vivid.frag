#version 330

// Vivid: Matte-finish grading with rich saturation, warm pink skin flush,
// soft two-tone cloth lighting, and a clean Fresnel rim light.
// Relies on vivid.json's pre-pass GTAO and SSSS (single-channel output).

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

// Studio relight for non-skin surfaces (soft two-tone ramp).
const vec3 KEY_COLOR = vec3(0.70, 0.66, 0.62);
const vec3 SKY_AMBIENT = vec3(0.46, 0.46, 0.52);
const vec3 GROUND_AMBIENT = vec3(0.40, 0.34, 0.31);
const float TOON_AMOUNT = 0.35;

// Colour grade (combined rich + vivid saturation) and warm pink skin flush.
const float EXPOSURE = 1.12;
const float CONTRAST = 0.45;
const float SATURATION = 1.2;
const vec3 FLUSH_TINT = vec3(1.10, 0.90, 0.90);
const float FLUSH_STRENGTH = 0.55;

// Rim light: soft pink Fresnel edge on the lit side.
const vec3 RIM_COLOR = vec3(1.0, 0.92, 0.93);
const float RIM_POWER = 4.0;
const float RIM_SKIN = 0.05;
const float RIM_CLOTH = 0.15;

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
    float skin = float(texture(u_material_flags, uv).b > 0.5);
    bool light_on = dot(u_key_light, u_key_light) > 0.25;
    vec3 light = normalize(light_on ? u_key_light : FALLBACK_LIGHT);
    vec3 view = normalize(-center);

    vec3 normal;
    vec3 shine = vec3(0.0);
    if (scene_normal(uv, normal)) {
        if (dot(normal, view) < 0.0) {
            normal = -normal;
        }
        if (light_on  && skin < 0.5) {
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