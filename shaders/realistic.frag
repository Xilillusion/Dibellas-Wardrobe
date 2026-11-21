#version 330

// Realistic: Physically-based studio portrait & armour rendering.
// Works in linear light space with RGB subsurface wrap for skin, true-IOR
// dielectric/metallic GGX specular, studio cubemap IBL, and ACES Filmic
// tone mapping. Writes a split output for "ssss_after_preset": true.

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;
// RGB: unblurred specular added after SSSS. A: SSSS mask.
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

const float BACKGROUND_DEPTH = 0.999999;
const vec3 FALLBACK_LIGHT = vec3(-0.3537, 0.5558, 0.7680);
const float PI = 3.14159265;

const vec3 LUMINANCE = vec3(0.2126, 0.7152, 0.0722);

// Studio lighting (linear space): lower key prevents highlight blowout while
// keeping fill high enough to stay above ACES's dark shadow toe.
const vec3 KEY_COLOR = vec3(0.92, 0.87, 0.82);
const vec3 FILL_SKY = vec3(0.14, 0.16, 0.20);
const vec3 FILL_GROUND = vec3(0.11, 0.09, 0.08);
const float IBL_DIFFUSE = 0.09;
const float EXPOSURE = 0.86;
const float SATURATION = 1.20;

// Skin: red scatters deeper than green/blue across the shadow terminator.
const vec3 SKIN_WRAP = vec3(0.42, 0.18, 0.12);
const float SKIN_F0 = 0.028;

vec3 to_linear(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
vec3 to_srgb(vec3 c)   { return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2)); }

// ACES Filmic tone mapping (Narkowicz approximation).
vec3 aces_tonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec3 grade(vec3 lin) {
    vec3 mapped = aces_tonemap(max(lin * EXPOSURE, vec3(0.0)));
    return to_srgb(mapped);
}

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

float ggx_d(float n_dot_h, float roughness) {
    float a2 = pow(max(roughness, 0.04), 4.0);
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float smith_g(float n_dot_v, float n_dot_l, float roughness) {
    float k = (roughness + 1.0);
    k = (k * k) * 0.125;
    float gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    float gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

vec3 fresnel_schlick(float cos_theta, vec3 f0, float roughness) {
    vec3 max_f = max(vec3(1.0 - roughness), f0);
    return f0 + (max_f - f0) * pow(max(1.0 - cos_theta, 0.0), 5.0);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_viewport_size;
    vec4 source = texture(u_scene_color, uv);
    float depth = texture(u_scene_depth, uv).r;
    vec3 center = view_position(uv, depth);

    // Screen-space derivatives in uniform control flow for smooth skin grazing normals.
    vec3 dx = dFdx(center);
    vec3 dy = dFdy(center);

    if (depth >= BACKGROUND_DEPTH) {
        fragColor = source;
        fragSpecular = vec4(0.0);
        return;
    }

    vec4 flags = texture(u_material_flags, uv);
    float skin = float(flags.b > 0.5);
    bool light_on = dot(u_key_light, u_key_light) > 0.25;
    vec3 light = normalize(light_on ? u_key_light : FALLBACK_LIGHT);
    vec3 view = normalize(-center);

    vec3 normal;
    if (!scene_normal(uv, normal)) {
        fragColor = source;
        fragSpecular = vec4(0.0, 0.0, 0.0, skin);
        return;
    }
    if (dot(normal, view) < 0.0) {
        normal = -normal;
    }

    // Recover approximate albedo by dividing out viewport Lambert, then move to linear space.
    float n_dot_l_raw = dot(normal, light);
    vec3 albedo_srgb = light_on
        ? min(source.rgb / mix(0.65, 1.0, max(n_dot_l_raw, 0.0)), vec3(1.0))
        : source.rgb;
    vec3 albedo = to_linear(albedo_srgb);

    // Material parameters from env mask / Complex Material / normal alpha.
    vec4 mask = flags.g > 0.5 ? texture(u_material_env_mask, uv) : vec4(0.0);
    bool tangent_normal = flags.r > 0.5 && flags.a <= 0.5;
    float normal_alpha = tangent_normal ? texture(u_material_normal, uv).a : 0.0;
    float spec_weight = flags.g > 0.5 ? mask.r : (tangent_normal ? normal_alpha : 0.35);

    bool complex_mat = flags.g > 0.5 && max(abs(mask.g - mask.r), abs(mask.b - mask.r)) > 0.04;
    float metalness = (1.0 - skin) * (complex_mat ? mask.b : smoothstep(0.45, 0.75, mask.r) * 0.85);
    float glossiness = complex_mat ? mask.g : mix(0.32, 0.65, spec_weight);
    float roughness = mix(clamp(1.0 - glossiness, 0.14, 0.88), 0.46, skin);

    // Diffuse lighting: RGB subsurface wrap on skin, Lambert on non-skin, plus studio IBL.
    vec3 world_n = u_view_to_world * normal;
    vec3 ibl_diffuse = to_linear(textureLod(u_studio_cubemap, world_n, 6.0).rgb);
    vec3 hemi = mix(FILL_GROUND, FILL_SKY, normal.y * 0.5 + 0.5);
    vec3 ambient = hemi + ibl_diffuse * IBL_DIFFUSE;

    vec3 diff_term = mix(
        vec3(max(n_dot_l_raw, 0.0)),
        clamp((n_dot_l_raw + SKIN_WRAP) / (1.0 + SKIN_WRAP), 0.0, 1.0),
        skin
    );
    vec3 diffuse_lin = albedo * (1.0 - metalness) * (ambient + KEY_COLOR * diff_term);

    // Specular lighting: physically-based GGX direct + studio cubemap IBL.
    vec3 half_vec = normalize(light + view);
    float n_dot_v = max(dot(normal, view), 0.001);
    float n_dot_l = max(n_dot_l_raw, 0.0);
    float n_dot_h = max(dot(normal, half_vec), 0.0);
    float v_dot_h = max(dot(view, half_vec), 0.0);

    vec3 f0 = mix(vec3(0.04), albedo, metalness);
    f0 = mix(f0, vec3(SKIN_F0), skin);

    vec3 fresnel_dir = f0 + (1.0 - f0) * pow(max(1.0 - v_dot_h, 0.0), 5.0);
    vec3 spec_direct;
    if (skin > 0.5) {
        // Dual-lobe realistic skin sheen (soft broad + subtle pore highlight).
        float d = ggx_d(n_dot_h, 0.48) * 0.75 + ggx_d(n_dot_h, 0.24) * 0.25;
        float g = smith_g(n_dot_v, n_dot_l, 0.42);
        spec_direct = fresnel_dir * (d * g / (4.0 * n_dot_v)) * KEY_COLOR
            * mix(0.45, 1.0, spec_weight) * 0.65;
    } else {
        float d = ggx_d(n_dot_h, roughness);
        float g = smith_g(n_dot_v, n_dot_l, roughness);
        spec_direct = fresnel_dir * (d * g / (4.0 * n_dot_v)) * KEY_COLOR
            * mix(0.25, 1.0, max(spec_weight, metalness));
    }

    // Studio cubemap reflection (sharp on polished metal, soft on leather/skin).
    vec3 refl_dir = u_view_to_world * reflect(-view, normal);
    vec3 env_spec = to_linear(textureLod(u_studio_cubemap, refl_dir, roughness * 6.0).rgb);
    vec3 fresnel_ibl = fresnel_schlick(n_dot_v, f0, roughness);
    float ibl_scale = mix(mix(0.25, 0.95, metalness) * max(spec_weight, metalness), 0.32, skin);
    vec3 spec_ibl = env_spec * fresnel_ibl * ibl_scale;

    // Subtle peach-fuzz / velvet grazing sheen using smooth geometric normal on skin.
    vec3 sheen_n = normal;
    if (skin > 0.5) {
        vec3 geo = cross(dx, dy);
        if (dot(geo, geo) > 1.0e-12 && max(abs(dx.z), abs(dy.z)) < -center.z * 0.04) {
            sheen_n = normalize(dot(geo, view) < 0.0 ? -geo : geo);
        }
    }
    float grazing = pow(max(1.0 - max(dot(sheen_n, view), 0.0), 0.0), 4.0);
    vec3 sheen = mix(vec3(0.08), vec3(0.11, 0.09, 0.085), skin)
        * grazing * smoothstep(-0.2, 0.5, n_dot_l_raw);

    // ACES Filmic tone mapping: compute combined and diffuse-only so the split
    // specular delta adds back in display space without ever clipping to flat white.
    vec3 spec_lin = max(spec_direct + spec_ibl + sheen, vec3(0.0));
    vec3 diff_graded = grade(diffuse_lin);
    float diff_luma = dot(diff_graded, LUMINANCE);
    diff_graded = clamp(mix(vec3(diff_luma), diff_graded, SATURATION), 0.0, 1.0);

    vec3 total_graded = grade(diffuse_lin + spec_lin);
    vec3 spec_delta = max(total_graded - diff_graded, vec3(0.0));

    fragColor = vec4(diff_graded, source.a);
    fragSpecular = vec4(spec_delta, skin);
}