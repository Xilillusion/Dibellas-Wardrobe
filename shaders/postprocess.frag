#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;
uniform vec2 u_viewport_size;
uniform mat4 u_projection;
uniform mat4 u_inverse_projection;

uniform bool u_enable_ssr;
uniform float u_ssr_step_size;
uniform int u_ssr_max_steps;
uniform float u_ssr_thickness;

uniform bool u_enable_ao;
uniform float u_ao_radius;
uniform float u_ao_intensity;
uniform float u_ao_bias;

uniform bool u_enable_ssss;
uniform float u_ssss_radius;
uniform vec3 u_ssss_falloff_color;

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

vec3 screen_space_reflection(vec2 uv, vec3 position, vec3 normal) {
    vec3 ray_direction = normalize(reflect(normalize(position), normal));
    vec3 ray_position = position + ray_direction * u_ssr_step_size;
    for (int i = 0; i < 128; ++i) {
        if (i >= u_ssr_max_steps) {
            break;
        }
        vec4 projected = u_projection * vec4(ray_position, 1.0);
        if (projected.w <= 0.0) {
            break;
        }
        vec2 sample_uv = projected.xy / projected.w * 0.5 + 0.5;
        if (any(lessThanEqual(sample_uv, vec2(0.0))) ||
            any(greaterThanEqual(sample_uv, vec2(1.0)))) {
            break;
        }

        float sample_depth = texture(u_scene_depth, sample_uv).r;
        if (sample_depth < BACKGROUND_DEPTH && i > 1) {
            vec3 scene_position = view_position(sample_uv, sample_depth);
            float ray_depth = -ray_position.z;
            float scene_depth = -scene_position.z;
            float depth_delta = ray_depth - scene_depth;
            if (depth_delta >= 0.0 && depth_delta <= u_ssr_thickness) {
                return texture(u_scene_color, sample_uv).rgb;
            }
        }
        ray_position += ray_direction * u_ssr_step_size;
    }
    return vec3(-1.0);
}

vec3 subsurface_scatter(vec2 uv, vec3 position, vec3 normal, vec3 source_color) {
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
        scattered += texture(u_scene_color, sample_uv).rgb * weight;
        weight_sum += weight;
    }
    vec3 blur_color = scattered / weight_sum;
    vec3 tinted_scatter = mix(blur_color, blur_color * u_ssss_falloff_color, 0.35);
    return mix(source_color, tinted_scatter, 0.35);
}

void main() {
    vec4 source = texture(u_scene_color, v_uv);
    float depth = texture(u_scene_depth, v_uv).r;
    if (depth >= BACKGROUND_DEPTH) {
        fragColor = source;
        return;
    }

    vec3 position = view_position(v_uv, depth);
    vec3 normal;
    if (!surface_normal(v_uv, position, normal)) {
        fragColor = source;
        return;
    }

    vec3 color = source.rgb;
    if (u_enable_ssr) {
        vec3 reflected_color = screen_space_reflection(v_uv, position, normal);
        if (reflected_color.r >= 0.0) {
            float facing = 1.0 - max(dot(normal, normalize(-position)), 0.0);
            color = mix(color, reflected_color, 0.2 + 0.2 * facing);
        }
    }
    if (u_enable_ssss) {
        color = subsurface_scatter(v_uv, position, normal, color);
    }
    if (u_enable_ao) {
        float occlusion = ambient_occlusion(v_uv, position, normal);
        color *= 1.0 - clamp(occlusion * u_ao_intensity * 0.65, 0.0, 0.85);
    }
    fragColor = vec4(color, source.a);
}
