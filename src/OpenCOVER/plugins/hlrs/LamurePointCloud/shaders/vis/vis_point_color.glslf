#version 420 core
in VertexData {
    vec3  pass_point_color;
    vec3  pass_world_pos;
    vec3  pass_vs_pos;
    vec3  pass_vs_normal;
    float pass_radius_ws;
    float pass_screen_size;
} fsIn;

layout(location = 0) out vec4 out_color;

INCLUDE vis_color_no_prov.glsl

void main() {
    vec3 finalColor = get_color(
        fsIn.pass_world_pos,
        fsIn.pass_vs_normal,
        fsIn.pass_point_color,
        fsIn.pass_radius_ws,
        fsIn.pass_screen_size
    );

    out_color = vec4(finalColor, 1.0);
}