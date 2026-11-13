#version 330

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;
uniform sampler2D u_scene_normal;
uniform sampler2D u_material_normal;
uniform sampler2D u_material_env_mask;
uniform sampler2D u_material_subsurface;
uniform sampler2D u_material_flags;
uniform vec2 u_viewport_size;
uniform mat4 u_inverse_projection;
// View-space direction towards the key light; zero while scene lighting is off.
uniform vec3 u_key_light;

const float BACKGROUND_DEPTH = 0.999999;
const vec3 LUMINANCE = vec3(0.2126, 0.7152, 0.0722);
// Normalized vec3(-0.35, 0.55, 0.76): an upper-left key light in view space,
// used for highlights while scene lighting is off.
const vec3 FALLBACK_LIGHT = vec3(-0.3537, 0.5558, 0.7680);
const float PI = 3.14159265;

// Relighting: a warm key over cool sky / warm ground ambient. A surface facing
// the key at the horizon sums to about 1, so lit areas keep their exposure.
const vec3 KEY_COLOR = vec3(0.68, 0.66, 0.62);
const vec3 SKY_AMBIENT = vec3(0.44, 0.47, 0.54);
const vec3 GROUND_AMBIENT = vec3(0.36, 0.33, 0.29);
// Light wraps a little past the terminator; skin wraps further, red the most,
// so its shadow edge glows warm like light scattered under the surface.
const vec3 SKIN_WRAP = vec3(0.45, 0.24, 0.16);
// Overall brightness applied before the final contrast curve.
const float EXPOSURE = 1.1;

struct Material {
    bool has_normal;
    bool has_env_mask;
    bool has_subsurface;
    bool model_space_normal;
    vec4 normal_map;
    vec4 env_mask;
    vec4 subsurface;
};

Material material_at(vec2 uv) {
    vec4 flags = texture(u_material_flags, uv);
    Material material;
    material.has_normal = flags.r > 0.5;
    material.has_env_mask = flags.g > 0.5;
    material.has_subsurface = flags.b > 0.5;
    material.model_space_normal = flags.a > 0.5;
    material.normal_map = material.has_normal ? texture(u_material_normal, uv) : vec4(0.0);
    material.env_mask = material.has_env_mask ? texture(u_material_env_mask, uv) : vec4(0.0);
    material.subsurface = material.has_subsurface ? texture(u_material_subsurface, uv) : vec4(0.0);
    return material;
}

float subsurface_strength(Material material) {
    if (!material.has_subsurface) {
        return 0.0;
    }
    vec3 tint = material.subsurface.rgb;
    return clamp(max(tint.r, max(tint.g, tint.b)), 0.0, 1.0);
}

vec3 view_position(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 position = u_inverse_projection * clip;
    return position.xyz / position.w;
}

bool valid_uv(vec2 uv) {
    return all(greaterThan(uv, vec2(0.0))) &&
           all(lessThan(uv, vec2(1.0)));
}

bool neighbor_position(vec2 uv, out vec3 position) {
    if (!valid_uv(uv)) {
        return false;
    }
    float depth = texture(u_scene_depth, uv).r;
    if (depth >= BACKGROUND_DEPTH) {
        return false;
    }
    position = view_position(uv, depth);
    return true;
}

bool shading_normal_at(vec2 uv, out vec3 normal) {
    vec4 encoded = texture(u_scene_normal, uv);
    vec3 decoded = encoded.rgb * 2.0 - 1.0;
    float length_squared = dot(decoded, decoded);
    if (encoded.a <= 0.5 || length_squared < 0.25) {
        return false;
    }
    normal = decoded * inversesqrt(length_squared);
    return true;
}

vec3 matching_hemisphere(vec3 normal, vec3 reference) {
    return dot(normal, reference) < 0.0 ? -normal : normal;
}

bool surface_basis(vec2 uv, vec3 center, out vec3 normal, out float pixel_size) {
    vec2 texel = 1.0 / u_viewport_size;
    vec3 left, right, down, up;
    if (!neighbor_position(uv - vec2(texel.x, 0.0), left) ||
        !neighbor_position(uv + vec2(texel.x, 0.0), right) ||
        !neighbor_position(uv - vec2(0.0, texel.y), down) ||
        !neighbor_position(uv + vec2(0.0, texel.y), up)) {
        return false;
    }

    // Choose the side nearest in depth to avoid normals spanning a silhouette.
    vec3 dx = abs(left.z - center.z) < abs(right.z - center.z)
        ? center - left : right - center;
    vec3 dy = abs(down.z - center.z) < abs(up.z - center.z)
        ? center - down : up - center;
    pixel_size = max(length(dx), length(dy));
    if (pixel_size < 0.000001 ||
        abs(dx.z) > max(-center.z * 0.02, pixel_size * 0.8) ||
        abs(dy.z) > max(-center.z * 0.02, pixel_size * 0.8)) {
        return false;
    }

    vec3 n = cross(dx, dy);
    if (dot(n, n) < 0.000000000001) {
        return false;
    }
    normal = normalize(n);
    if (dot(normal, -center) < 0.0) {
        normal = -normal;
    }
    return true;
}

float contact_occlusion(
    vec2 uv, vec3 center, vec3 geometric_normal, vec3 shading_normal, float pixel_size
) {
    // Cardinal taps cover the near ring (3 px) and diagonal taps the far one (8 px).
    const vec2 offsets[8] = vec2[](
        vec2(3.0, 0.0), vec2(-3.0, 0.0),
        vec2(0.0, 3.0), vec2(0.0, -3.0),
        vec2(5.656854, 5.656854), vec2(-5.656854, 5.656854),
        vec2(5.656854, -5.656854), vec2(-5.656854, -5.656854)
    );
    vec2 texel = 1.0 / u_viewport_size;
    float radius = pixel_size * 12.0;
    float occlusion = 0.0;

    for (int i = 0; i < 8; ++i) {
        vec3 sample_position;
        if (!neighbor_position(uv + offsets[i] * texel, sample_position)) {
            continue;
        }
        vec3 delta = sample_position - center;
        float distance_to_sample = length(delta);
        if (distance_to_sample < 0.000001 || distance_to_sample > radius) {
            continue;
        }
        vec3 direction = delta / distance_to_sample;
        float geometric_facing = dot(geometric_normal, direction);
        float horizon = max(geometric_facing - 0.08, 0.0);
        if (horizon <= 0.0) {
            continue;
        }
        float normal_factor = clamp(
            (dot(shading_normal, direction) + 0.4) / (geometric_facing + 0.4),
            0.6, 1.4
        );
        float range_weight = 1.0 - smoothstep(radius * 0.25, radius, distance_to_sample);
        occlusion += horizon * normal_factor * range_weight;
    }
    return clamp(occlusion / 8.0 * 1.6, 0.0, 0.35);
}

float normal_cavity(
    vec2 uv, vec3 center, vec3 shading_normal, float pixel_size
) {
    const vec2 offsets[4] = vec2[](
        vec2(1.0, 0.0), vec2(-1.0, 0.0),
        vec2(0.0, 1.0), vec2(0.0, -1.0)
    );
    vec2 texel = 1.0 / u_viewport_size;
    float curvature = 0.0;
    float samples = 0.0;
    for (int i = 0; i < 4; ++i) {
        vec2 sample_uv = uv + offsets[i] * texel;
        vec3 sample_position;
        vec3 sample_normal;
        if (!neighbor_position(sample_uv, sample_position) ||
            abs(sample_position.z - center.z) > pixel_size * 0.8 ||
            !shading_normal_at(sample_uv, sample_normal) ||
            abs(dot(sample_normal, shading_normal)) < 0.5) {
            continue;
        }
        sample_normal = matching_hemisphere(sample_normal, shading_normal);
        vec3 delta = sample_position - center;
        vec3 tangent = delta - shading_normal * dot(delta, shading_normal);
        float tangent_length = length(tangent);
        if (tangent_length < 0.000001) {
            continue;
        }
        curvature += dot(shading_normal - sample_normal, tangent / tangent_length);
        samples += 1.0;
    }
    if (samples < 3.0) {
        return 0.0;
    }
    return clamp((curvature / samples - 0.015) * 0.14, 0.0, 0.08);
}

vec3 surface_detail(
    vec2 uv, vec3 center, vec3 color, float pixel_size, vec3 normal, bool has_normal,
    float amount
) {
    const vec2 offsets[4] = vec2[](
        vec2(1.0, 0.0), vec2(-1.0, 0.0),
        vec2(0.0, 1.0), vec2(0.0, -1.0)
    );
    vec2 texel = 1.0 / u_viewport_size;
    vec3 sum = color;
    float weight = 1.0;
    for (int i = 0; i < 4; ++i) {
        vec2 sample_uv = uv + offsets[i] * texel;
        vec3 sample_position;
        if (!neighbor_position(sample_uv, sample_position) ||
            abs(sample_position.z - center.z) > pixel_size * 0.8) {
            continue;
        }
        float sample_weight = 1.0;
        vec3 sample_normal;
        if (has_normal && shading_normal_at(sample_uv, sample_normal)) {
            sample_normal = matching_hemisphere(sample_normal, normal);
            sample_weight = smoothstep(0.75, 0.95, dot(normal, sample_normal));
        }
        sum += texture(u_scene_color, sample_uv).rgb * sample_weight;
        weight += sample_weight;
    }
    vec3 detail = clamp(color - sum / weight, vec3(-0.12), vec3(0.12));
    return clamp(color + detail * amount, 0.0, 1.0);
}

// Rough surfaces scatter their highlight: neighbouring normals that disagree
// (fine normal-map detail) widen the lobe, in the spirit of Toksvig filtering.
float normal_variance(vec2 uv, vec3 normal) {
    const vec2 offsets[4] = vec2[](
        vec2(1.0, 0.0), vec2(-1.0, 0.0),
        vec2(0.0, 1.0), vec2(0.0, -1.0)
    );
    vec2 texel = 1.0 / u_viewport_size;
    float agreement = 0.0;
    float samples = 0.0;
    for (int i = 0; i < 4; ++i) {
        vec3 sample_normal;
        if (shading_normal_at(uv + offsets[i] * texel, sample_normal)) {
            sample_normal = matching_hemisphere(sample_normal, normal);
            agreement += max(dot(normal, sample_normal), 0.0);
            samples += 1.0;
        }
    }
    return samples > 0.0 ? 1.0 - agreement / samples : 0.0;
}

// Light entering skin or similar materials exits nearby: blur lighting across
// the same subsurface surface and bleed the _sk tint into the shadowed side.
vec3 subsurface_scatter(
    vec2 uv, vec3 center, vec3 color, vec3 normal,
    float pixel_size, Material material, float strength
) {
    // xy: pixel offset, z: tap distance, w: weight. Cardinal taps cover the near
    // ring and diagonal taps the far one; the doubled weights keep the blur as
    // strong as sampling both rings in every direction.
    const vec4 taps[8] = vec4[](
        vec4(2.0, 0.0, 2.0, 1.6), vec4(-2.0, 0.0, 2.0, 1.6),
        vec4(0.0, 2.0, 2.0, 1.6), vec4(0.0, -2.0, 2.0, 1.6),
        vec4(3.535534, 3.535534, 5.0, 0.9), vec4(-3.535534, 3.535534, 5.0, 0.9),
        vec4(3.535534, -3.535534, 5.0, 0.9), vec4(-3.535534, -3.535534, 5.0, 0.9)
    );
    vec2 texel = 1.0 / u_viewport_size;
    vec3 sum = color;
    float weight = 1.0;
    for (int i = 0; i < 8; ++i) {
        // Snap to texel centres so the linearly filtered colour cannot blend
        // in a neighbour the nearest-filtered depth and flags did not vet.
        vec2 sample_uv = (floor(uv * u_viewport_size + taps[i].xy) + 0.5) * texel;
        vec3 sample_position;
        if (!neighbor_position(sample_uv, sample_position) ||
            abs(dot(sample_position - center, normal)) >
                pixel_size * taps[i].z * 0.8 ||
            texture(u_material_flags, sample_uv).b <= 0.5) {
            continue;
        }
        float sample_weight = taps[i].w;
        vec3 sample_normal;
        if (shading_normal_at(sample_uv, sample_normal)) {
            sample_normal = matching_hemisphere(sample_normal, normal);
            sample_weight *= smoothstep(0.55, 0.9, dot(normal, sample_normal));
        }
        sum += texture(u_scene_color, sample_uv).rgb * sample_weight;
        weight += sample_weight;
    }
    vec3 diffused = sum / weight;
    vec3 scattered = mix(color, diffused, strength * 0.5);

    vec3 tint = material.subsurface.rgb / max(strength, 0.0001);
    float lit = dot(diffused, LUMINANCE);
    float shadow = 1.0 - smoothstep(0.08, 0.6, dot(color, LUMINANCE));
    scattered += tint * lit * shadow * strength * 0.14;
    return clamp(scattered, 0.0, 1.0);
}

bool scene_light_on() {
    return dot(u_key_light, u_key_light) > 0.25;
}

vec3 key_light_direction() {
    return normalize(scene_light_on() ? u_key_light : FALLBACK_LIGHT);
}

vec3 key_lighting(vec3 normal, vec3 light, float scatter) {
    vec3 wrap = vec3(0.08) + SKIN_WRAP * scatter;
    vec3 diffuse = clamp((dot(normal, light) + wrap) / (1.0 + wrap), 0.0, 1.0);
    vec3 ambient = mix(GROUND_AMBIENT, SKY_AMBIENT, normal.y * 0.5 + 0.5);
    return ambient + KEY_COLOR * diffuse;
}

// Replace the viewport's flat Lambert shading with the key-light model: divide
// out an estimate of the baked lighting to recover albedo, then relight it.
vec3 relight(vec3 color, vec3 normal, vec3 light, float scatter) {
    float baked = mix(0.65, 1.0, max(dot(normal, light), 0.0));
    vec3 albedo = min(color / baked, vec3(1.0));
    return albedo * key_lighting(normal, light, scatter);
}

// Real surfaces reflect unevenly: texture valleys (pores, weave gaps, scratches)
// catch less light than the peaks around them. Without this, highlights read as
// a smooth plastic coating.
float specular_breakup(vec2 uv, vec3 color) {
    const vec2 offsets[4] = vec2[](
        vec2(1.5, 0.5), vec2(-0.5, 1.5), vec2(-1.5, -0.5), vec2(0.5, -1.5)
    );
    vec2 texel = 1.0 / u_viewport_size;
    float center = dot(color, LUMINANCE);
    float sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum += dot(texture(u_scene_color, uv + offsets[i] * texel).rgb, LUMINANCE);
    }
    float relative = (center - sum * 0.25) / max(center, 0.05);
    return clamp(1.0 + relative * 2.5, 0.35, 1.5);
}

// Highlight and Fresnel reflection the environment mask (or, failing that, the
// normal map's specular alpha) calls for. The scene normal already carries the
// normal map, so it is not reapplied here.
vec3 material_specular(
    vec2 uv, vec3 center, vec3 color, vec3 normal, vec3 light, Material material,
    float occlusion
) {
    float reflection = 0.0;
    if (material.has_env_mask) {
        reflection = material.env_mask.r;
    } else if (material.has_normal && !material.model_space_normal) {
        // Normal alpha is only a specular map when the material enables it;
        // weight it down since that switch is not visible here.
        reflection = material.normal_map.a * 0.35;
    }
    if (reflection <= 0.004) {
        return vec3(0.0);
    }

    float glossiness = 0.25;
    float metalness = 0.0;
    if (material.has_env_mask && material.env_mask.g > 4.0 / 255.0) {
        glossiness = material.env_mask.g;
        // A plain greyscale mask repeats red in every channel; only a mask with
        // distinct channels is a Complex Material carrying metalness in blue.
        vec3 mask = material.env_mask.rgb;
        if (max(abs(mask.g - mask.r), abs(mask.b - mask.r)) > 0.04) {
            metalness = mask.b;
        }
    }
    glossiness *= 1.0 - 0.6 * smoothstep(0.01, 0.2, normal_variance(uv, normal));

    vec3 view = normalize(-center);
    vec3 half_vector = normalize(light + view);
    float n_dot_l = max(dot(normal, light), 0.0);
    float n_dot_h = max(dot(normal, half_vector), 0.0);
    float n_dot_v = max(dot(normal, view), 0.0);
    float v_dot_h = max(dot(view, half_vector), 0.0);

    float albedo_luminance = dot(color, LUMINANCE);
    vec3 albedo_tint = color / max(max(color.r, max(color.g, color.b)), 0.0001);
    vec3 f0 = mix(vec3(0.04), albedo_tint * 0.8, metalness);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);

    float power = exp2(mix(3.0, 9.0, glossiness));
    float lobe = (power + 8.0) / (8.0 * PI) * pow(n_dot_h, power) * n_dot_l;
    // Highlights stay muted where the baked lighting is already in shadow.
    float baked_light = mix(0.35, 1.0, smoothstep(0.03, 0.45, albedo_luminance));
    vec3 highlight = fresnel * min(lobe, 24.0) * baked_light * 0.25
        * mix(0.45, 1.0, glossiness);

    vec3 reflected = reflect(-view, normal);
    vec3 surroundings = vec3(mix(0.16, 0.55, reflected.y * 0.5 + 0.5));
    surroundings *= mix(vec3(1.0), albedo_tint, metalness);
    float grazing = pow(1.0 - n_dot_v, 5.0);
    // Rough surfaces scatter their grazing reflection away; only smooth ones
    // keep a mirror-like edge.
    vec3 sheen = surroundings * mix(f0, vec3(1.0), grazing * glossiness)
        * mix(0.15, 1.0, glossiness * glossiness) * 0.4;

    float visibility = 1.0 - clamp(occlusion * 2.5, 0.0, 0.85);
    vec3 specular = (highlight + sheen) * reflection * visibility
        * specular_breakup(uv, color);
    // Roll highlights off softly; a hard clamp flattens them into plateaus.
    float peak = max(specular.r, max(specular.g, specular.b));
    return specular / (1.0 + peak * 1.6);
}

void main() {
    vec4 source = texture(u_scene_color, v_uv);
    float depth = texture(u_scene_depth, v_uv).r;
    if (depth >= BACKGROUND_DEPTH) {
        fragColor = source;
        return;
    }

    vec3 color = source.rgb;
    vec3 center = view_position(v_uv, depth);
    vec3 geometric;
    float pixel_size;
    bool has_geometry = surface_basis(v_uv, center, geometric, pixel_size);
    vec3 shading_normal;
    bool has_normal = shading_normal_at(v_uv, shading_normal);
    if (has_normal && has_geometry) {
        if (dot(shading_normal, geometric) < 0.0) {
            shading_normal = -shading_normal;
        }
        has_normal = dot(shading_normal, geometric) >= 0.15;
    }
    if (has_normal) {
        Material material = material_at(v_uv);
        float scatter = subsurface_strength(material);
        if (has_geometry) {
            color *= 1.0 - normal_cavity(v_uv, center, shading_normal, pixel_size)
                * (1.0 - 0.6 * scatter);
        }
        vec3 light = key_light_direction();
        if (scene_light_on()) {
            color = relight(color, shading_normal, light, scatter);
        }
        if (has_geometry) {
            color = surface_detail(
                v_uv, center, color, pixel_size, shading_normal, true,
                0.18 * (1.0 - 0.45 * scatter)
            );
        }
        if (scatter > 0.0 && has_geometry) {
            color = subsurface_scatter(
                v_uv, center, color, shading_normal, pixel_size, material, scatter
            );
        }
        float occlusion = has_geometry
            ? contact_occlusion(v_uv, center, geometric, shading_normal, pixel_size)
                * (1.0 - 0.45 * scatter)
            : 0.0;
        color *= 1.0 - occlusion;
        color += material_specular(
            v_uv, center, color, shading_normal, light, material, occlusion
        );
    } else if (has_geometry) {
        color = surface_detail(v_uv, center, color, pixel_size, vec3(0.0), false, 0.18);
        color *= 1.0 - contact_occlusion(v_uv, center, geometric, geometric, pixel_size);
    }

    color *= EXPOSURE;
    float luminance = dot(color, LUMINANCE);
    float contrast = 1.0 + 0.40 * (luminance - 0.5) * (1.0 - luminance);
    color *= contrast;
    fragColor = vec4(clamp(color, 0.0, 1.0), source.a);
}
