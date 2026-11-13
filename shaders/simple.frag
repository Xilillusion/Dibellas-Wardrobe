#version 330

// Keep the viewport's material and lighting; GTAO and SSSS run before this pass.
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;
uniform sampler2D u_scene_normal;
uniform vec2 u_viewport_size;
uniform mat4 u_inverse_projection;
uniform vec3 u_key_light;

const float BACKGROUND_DEPTH = 0.999999;
const vec3 LUMINANCE = vec3(0.2126, 0.7152, 0.0722);
const float EXPOSURE = 1.12;
const float CONTRAST = 0.45;
const float SATURATION = 1.15;

void main() {
    vec4 source = texture(u_scene_color, v_uv);
    float depth = texture(u_scene_depth, v_uv).r;
    if (depth >= BACKGROUND_DEPTH) {
        fragColor = source;
        return;
    }

    vec3 color = source.rgb;
    vec4 encoded = texture(u_scene_normal, v_uv);
    vec3 normal = encoded.rgb * 2.0 - 1.0;
    if (encoded.a > 0.5 && dot(normal, normal) > 0.25) {
        normal = normalize(normal);
        vec2 half_texel = 0.5 / u_viewport_size;
        vec2 uv = clamp(v_uv, half_texel, 1.0 - half_texel);
        vec4 position = u_inverse_projection * vec4(vec3(uv, depth) * 2.0 - 1.0, 1.0);
        vec3 view_vector = -position.xyz / position.w;
        if (dot(view_vector, view_vector) > 1.0e-8) {
            float grazing = pow(1.0 - max(dot(normal, normalize(view_vector)), 0.0), 3.0);
            color *= 1.0 + 0.02 * grazing;
        }
        if (dot(u_key_light, u_key_light) > 0.25) {
            float light = max(dot(normal, normalize(u_key_light)), 0.0);
            color *= mix(vec3(0.98, 0.99, 1.0), vec3(1.02, 1.01, 1.0), light);
        }
    }

    color = max(mix(vec3(dot(color, LUMINANCE)), color, SATURATION), 0.0);
    color *= EXPOSURE;
    float luminance = dot(color, LUMINANCE);
    color *= 1.0 + CONTRAST * (luminance - 0.5) * (1.0 - luminance);
    fragColor = vec4(clamp(color, 0.0, 1.0), source.a);
}
