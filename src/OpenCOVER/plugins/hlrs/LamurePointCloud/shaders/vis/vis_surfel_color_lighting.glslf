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

INCLUDE ../common/shading/lighting.glsl

void main() {
    if (length(fsIn.pass_uv_coords) > 1.0) discard;

    vec3 finalColor = shade_blinn_phong(
            fsIn.pass_vs_pos,
            fsIn.pass_vs_normal,
            fsIn.pass_point_color
    );

    out_color = vec4(finalColor, 1.0);
}
