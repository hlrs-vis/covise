// ===================== Fragment Shader =====================
#version 420 core

in FS_IN {
    vec3  pass_point_color;
    vec2  pass_uv_coords;
    vec3  pass_world_pos;
    vec3  pass_vs_pos;
    vec3  pass_vs_normal;
    float pass_radius_ws;
    float pass_screen_size;
} fsIn;

layout(location = 0) out vec4 out_color;

uniform bool  show_normals;
uniform bool  show_accuracy;
uniform bool  show_radius_deviation;
uniform bool  show_output_sensitivity;
uniform float accuracy;
uniform float average_radius;

uniform vec3  point_light_pos_vs;
uniform float point_light_intensity;
uniform float ambient_intensity;
uniform float specular_intensity;
uniform float shininess;
uniform float gamma;
uniform bool  use_tone_mapping;

INCLUDE vis_color_no_prov.glsl
INCLUDE ../common/shading/lighting.glsl

void main() {
    if (length(fsIn.pass_uv_coords) > 1.0) discard;

    ColorDebugParams dbg = ColorDebugParams(
        show_normals,
        show_accuracy,
        show_radius_deviation,
        show_output_sensitivity,
        accuracy,
        average_radius
    );

    LightingParams lighting = LightingParams(
        point_light_pos_vs,
        point_light_intensity,
        ambient_intensity,
        specular_intensity,
        shininess,
        gamma,
        use_tone_mapping
    );

    vec3 baseColor = get_color(
        fsIn.pass_world_pos,
        fsIn.pass_vs_normal,
        fsIn.pass_point_color,
        fsIn.pass_radius_ws,
        fsIn.pass_screen_size,
        dbg
    );

    vec3 finalColor = shade_blinn_phong(
        fsIn.pass_vs_pos,
        fsIn.pass_vs_normal,
        baseColor,
        lighting
    );

    out_color = vec4(finalColor, 1.0);
}
