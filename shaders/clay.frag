#version 330

// Clay: Neutral matte white/grey lighting diagnostic preset.
// Strips albedo textures to inspect key-light direction, shadow terminators,
// and normal consistency between skin (_msn) and clothing (_n).

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;

uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;
uniform sampler2D u_scene_normal;
uniform vec2 u_viewport_size;
uniform mat4 u_inverse_projection;
// View-space direction towards the key light; zero while scene lighting is off.
uniform vec3 u_key_light;

const float BACKGROUND_DEPTH = 0.999999;
const vec3 FALLBACK_LIGHT = vec3(-0.3537, 0.5558, 0.7680);

// Light-grey clay avoids clipping at peak N.L while reading as clean white.
const float CLAY_ALBEDO = 0.82;
const float AMBIENT = 0.08;
const float SPEC_STRENGTH = 0.25;
const float SPEC_POWER = 48.0;

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

void main() {
    vec2 uv = gl_FragCoord.xy / u_viewport_size;
    vec4 source = texture(u_scene_color, uv);
    float depth = texture(u_scene_depth, uv).r;
    if (depth >= BACKGROUND_DEPTH) {
        fragColor = source;
        return;
    }

    vec3 normal;
    if (!scene_normal(uv, normal)) {
        fragColor = vec4(vec3(CLAY_ALBEDO * AMBIENT), source.a);
        return;
    }

    vec3 center = view_position(uv, depth);
    vec3 view = normalize(-center);
    bool light_on = dot(u_key_light, u_key_light) > 0.25;
    vec3 light = normalize(light_on ? u_key_light : FALLBACK_LIGHT);

    float n_dot_l = max(dot(normal, light), 0.0);
    vec3 half_vec = normalize(light + view);
    float spec = pow(max(dot(normal, half_vec), 0.0), SPEC_POWER) * step(0.0, dot(normal, light));

    vec3 color = vec3(CLAY_ALBEDO * (AMBIENT + n_dot_l) + spec * SPEC_STRENGTH);

    fragColor = vec4(clamp(color, 0.0, 1.0), source.a);
}