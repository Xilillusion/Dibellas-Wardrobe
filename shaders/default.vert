#version 330

/*
 * Vertex Shader for Dibella's Wardrobe
 * Paired with Outfit Studio's default.frag (#version 330)
 */

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_tangent;
layout(location = 4) in vec3 in_bitangent;
layout(location = 5) in vec4 in_color;

uniform mat4 matProjection;
uniform mat4 matModel;
uniform mat4 matView;
uniform mat3 mv_normalMatrix;
uniform bool bModelSpace;

struct DirectionalLight
{
	vec3 diffuse;
	vec3 direction;
};

uniform DirectionalLight frontal;
uniform DirectionalLight directional0;
uniform DirectionalLight directional1;
uniform DirectionalLight directional2;

out vec3 lightFrontal;
out vec3 lightDirectional0;
out vec3 lightDirectional1;
out vec3 lightDirectional2;

out vec3 viewDir;
out vec3 n;
out mat3 mv_tbn;

out float maskFactor;
out vec3 weightColor;

out vec4 vColor;
out vec2 vUV;

void main(void)
{
	vec4 mvPos = matView * matModel * vec4(in_position, 1.0);
	gl_Position = matProjection * mvPos;

	// Camera is at (0,0,0) in view space
	viewDir = normalize(-mvPos.xyz);

	// Object-space normal (transformed by mv_normalMatrix in default.frag when normal map is off)
	n = in_normal;

	if (bModelSpace)
	{
		// Skyrim _msn.dds stores normals in object space (with R flipped in default.frag)
		mv_tbn = mv_normalMatrix;
	}
	else
	{
		vec3 T = normalize(mv_normalMatrix * in_tangent);
		vec3 B = normalize(mv_normalMatrix * in_bitangent);
		vec3 N = normalize(mv_normalMatrix * in_normal);
		mv_tbn = mat3(T, B, N);
	}

	// Frontal light moves with camera (view space); directional0..2 are in world space
	lightFrontal = normalize(frontal.direction);
	mat3 viewRot = mat3(matView);
	lightDirectional0 = normalize(viewRot * directional0.direction);
	lightDirectional1 = normalize(viewRot * directional1.direction);
	lightDirectional2 = normalize(viewRot * directional2.direction);

	maskFactor = 1.0;
	weightColor = vec3(1.0);
	vColor = in_color;
	vUV = in_uv;
}
