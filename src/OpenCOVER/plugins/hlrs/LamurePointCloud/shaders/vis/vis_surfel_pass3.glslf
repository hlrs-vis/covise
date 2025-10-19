#version 420 core

layout(binding = 0) uniform sampler2D in_color_texture;       // rgba: sum(color*w), a: sum w
layout(binding = 1) uniform sampler2D in_normal_texture;      // rgb:  sum(normal*w)
layout(binding = 2) uniform sampler2D in_vs_position_texture; // rgb:  sum(pos_vs*w)
layout(binding = 3) uniform sampler2D in_depth_texture;       // depth from pass 1

layout(location = 0) out vec4 out_color;
uniform bool  lighting;
uniform vec3  point_light_pos_vs;
uniform float point_light_intensity;
uniform float ambient_intensity;
uniform float specular_intensity;
uniform float shininess;
uniform float gamma;
uniform bool  use_tone_mapping;

in VsOut {
    vec2 uv;  // 0..1
} fs_in;

INCLUDE ../common/shading/lighting.glsl

void main()
{
    vec2 t = fs_in.uv;

    vec4 accumulated_color  = texture(in_color_texture,       t);
    vec3 accumulated_normal = texture(in_normal_texture,      t).rgb;
    vec3 accumulated_pos_vs = texture(in_vs_position_texture, t).rgb;

    if (accumulated_color.a <= 0.0)
        discard; // no surfel contribution

    float sum_w = accumulated_color.a; // sum weights

    // Weighted mean reconstruction
    vec3 albedo    = accumulated_color.rgb  / max(sum_w, 1e-8);
    vec3 normal_vs = normalize(accumulated_normal / max(sum_w, 1e-8));
    vec3 pos_vs    = accumulated_pos_vs     / max(sum_w, 1e-8);

    LightingParams lighting_params = LightingParams(
        point_light_pos_vs,
        point_light_intensity,
        ambient_intensity,
        specular_intensity,
        shininess,
        gamma,
        use_tone_mapping
    );

    // Lighting can be disabled for debug views
    vec3 shaded = lighting ? shade_blinn_phong(pos_vs, normal_vs, albedo, lighting_params)
                           : albedo;

    // Premultiplied resolve output, also provide depth
    float depth = texture(in_depth_texture, t).r;
    gl_FragDepth = depth;
    out_color = vec4(shaded, 1.0);
}
