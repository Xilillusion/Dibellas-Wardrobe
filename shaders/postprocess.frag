#version 330

// The viewport inserts ENABLE_GTAO / ENABLE_SSSS defines after #version
// according to the active outfit preset. Without either, this is a copy pass.
// SPLIT_COMPOSITE runs after a preset with "ssss_after_preset": u_scene_color
// is the preset's diffuse, scattered only where u_scene_specular.a masks it,
// and the specular RGB is added afterwards so SSSS never blurs it.

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene_color;
#ifdef SPLIT_COMPOSITE
uniform sampler2D u_scene_specular;
#endif

#if defined(ENABLE_GTAO) || defined(ENABLE_SSSS)
uniform sampler2D u_scene_depth;
uniform vec2 u_viewport_size;
uniform mat4 u_inverse_projection;
#endif

#ifdef ENABLE_GTAO
uniform float u_ao_radius;
uniform float u_ao_intensity;
uniform float u_ao_bias;
#endif

#ifdef ENABLE_SSSS
uniform float u_ssss_radius;
uniform float u_ssss_strength;
uniform vec3 u_ssss_falloff_color;
#endif

#if defined(ENABLE_GTAO) || defined(ENABLE_SSSS)
const float BACKGROUND_DEPTH = 0.999999;

vec3 view_position(vec2 uv, float depth) {
    vec4 clip_position = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 position = u_inverse_projection * clip_position;
    return position.xyz / max(abs(position.w), 0.000001) * sign(position.w);
}

bool surface_normal(vec2 uv, vec3 center, out vec3 normal) {
    vec2 texel = 1.0 / u_viewport_size;
    float center_depth = -center.z;
    float depths[4] = float[](
        texture(u_scene_depth, uv - vec2(texel.x, 0.0)).r,
        texture(u_scene_depth, uv + vec2(texel.x, 0.0)).r,
        texture(u_scene_depth, uv - vec2(0.0, texel.y)).r,
        texture(u_scene_depth, uv + vec2(0.0, texel.y)).r
    );
    for (int i = 0; i < 4; ++i) {
        if (depths[i] >= BACKGROUND_DEPTH) {
            return false;
        }
        float neighbor_depth = -view_position(
            uv + (i == 0 ? vec2(-texel.x, 0.0) :
                  i == 1 ? vec2(texel.x, 0.0) :
                  i == 2 ? vec2(0.0, -texel.y) : vec2(0.0, texel.y)),
            depths[i]
        ).z;
        if (abs(neighbor_depth - center_depth) > max(0.02, center_depth * 0.04)) {
            return false;
        }
    }

    vec3 left_position = view_position(uv - vec2(texel.x, 0.0), depths[0]);
    vec3 right_position = view_position(uv + vec2(texel.x, 0.0), depths[1]);
    vec3 down_position = view_position(uv - vec2(0.0, texel.y), depths[2]);
    vec3 up_position = view_position(uv + vec2(0.0, texel.y), depths[3]);
    normal = normalize(cross(right_position - left_position, up_position - down_position));
    if (dot(normal, -center) < 0.0) {
        normal = -normal;
    }
    return dot(normal, normal) > 0.5;
}
#endif

#ifdef ENABLE_GTAO
float ambient_occlusion(vec2 uv, vec3 position, vec3 normal) {
    const vec2 offsets[8] = vec2[](
        vec2(1.0, 0.0), vec2(-1.0, 0.0),
        vec2(0.0, 1.0), vec2(0.0, -1.0),
        vec2(0.707, 0.707), vec2(-0.707, 0.707),
        vec2(0.707, -0.707), vec2(-0.707, -0.707)
    );
    float pixel_radius = clamp(u_ao_radius * 0.75, 1.0, 24.0);
    float occlusion = 0.0;
    float samples = 0.0;
    for (int i = 0; i < 8; ++i) {
        vec2 sample_uv = uv + offsets[i] * pixel_radius / u_viewport_size;
        if (any(lessThanEqual(sample_uv, vec2(0.0))) ||
            any(greaterThanEqual(sample_uv, vec2(1.0)))) {
            continue;
        }
        float sample_depth = texture(u_scene_depth, sample_uv).r;
        if (sample_depth >= BACKGROUND_DEPTH) {
            continue;
        }
        vec3 delta = view_position(sample_uv, sample_depth) - position;
        float distance_to_sample = length(delta);
        if (distance_to_sample < 0.0001 || distance_to_sample > u_ao_radius) {
            continue;
        }
        float facing = max(dot(normal, delta / distance_to_sample) - u_ao_bias, 0.0);
        occlusion += facing * (1.0 - distance_to_sample / u_ao_radius);
        samples += 1.0;
    }
    return samples > 0.0 ? clamp(occlusion / samples, 0.0, 1.0) : 0.0;
}
#endif

#ifdef ENABLE_SSSS
vec3 subsurface_scatter(
    vec2 uv, vec3 position, vec3 normal, vec3 source_color, float strength
) {
    const vec2 offsets[8] = vec2[](
        vec2(1.0, 0.0), vec2(-1.0, 0.0),
        vec2(0.0, 1.0), vec2(0.0, -1.0),
        vec2(0.707, 0.707), vec2(-0.707, 0.707),
        vec2(0.707, -0.707), vec2(-0.707, -0.707)
    );
    float pixel_radius = clamp(u_ssss_radius, 1.0, 24.0);
    float center_depth = -position.z;
    vec3 scattered = source_color;
    float weight_sum = 1.0;
    for (int i = 0; i < 8; ++i) {
        vec2 sample_uv = uv + offsets[i] * pixel_radius / u_viewport_size;
        if (any(lessThanEqual(sample_uv, vec2(0.0))) ||
            any(greaterThanEqual(sample_uv, vec2(1.0)))) {
            continue;
        }
        float sample_depth = texture(u_scene_depth, sample_uv).r;
        if (sample_depth >= BACKGROUND_DEPTH) {
            continue;
        }
        vec3 sample_position = view_position(sample_uv, sample_depth);
        float depth_gap = abs(-sample_position.z - center_depth);
        if (depth_gap > max(0.02, center_depth * 0.01)) {
            continue;
        }
        vec3 sample_normal;
        if (!surface_normal(sample_uv, sample_position, sample_normal) ||
            dot(normal, sample_normal) < 0.5) {
            continue;
        }
        float weight = 1.0 - float(i) * 0.055;
#ifdef SPLIT_COMPOSITE
        // Only masked neighbours scatter, so unmasked surfaces do not bleed in.
        weight *= texture(u_scene_specular, sample_uv).a;
        if (weight <= 0.0) {
            continue;
        }
#endif
        scattered += texture(u_scene_color, sample_uv).rgb * weight;
        weight_sum += weight;
    }
    vec3 blur_color = scattered / weight_sum;
    vec3 tinted_scatter = mix(blur_color, blur_color * u_ssss_falloff_color, 0.35);
    return mix(source_color, tinted_scatter, strength);
}
#endif

void main() {
    vec4 source = texture(u_scene_color, v_uv);
    vec3 color = source.rgb;
#ifdef SPLIT_COMPOSITE
    vec4 specular = texture(u_scene_specular, v_uv);
    float ssss_mask = specular.a;
#else
    float ssss_mask = 1.0;
#endif
#if defined(ENABLE_GTAO) || defined(ENABLE_SSSS)
    float depth = texture(u_scene_depth, v_uv).r;
    if (depth < BACKGROUND_DEPTH) {
        vec3 position = view_position(v_uv, depth);
        vec3 normal;
        if (surface_normal(v_uv, position, normal)) {
#ifdef ENABLE_SSSS
            if (ssss_mask > 0.0) {
                color = subsurface_scatter(
                    v_uv, position, normal, color, u_ssss_strength * ssss_mask
                );
            }
#endif
#ifdef ENABLE_GTAO
            float occlusion = ambient_occlusion(v_uv, position, normal);
            color *= 1.0 - clamp(occlusion * u_ao_intensity * 0.65, 0.0, 0.85);
#endif
        }
    }
#endif
#ifdef SPLIT_COMPOSITE
    color += specular.rgb;
#endif
    fragColor = vec4(color, source.a);
}
