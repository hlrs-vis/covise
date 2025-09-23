// ===================== Fragment Shader =====================
#version 420 core

in FS_IN {
    vec3  pass_point_color;
    vec2  pass_uv_coords;
    vec3  pass_world_pos;
    vec3  pass_normal_ws;
    vec3  pass_vs_pos;
    vec3  pass_vs_normal;
    float pass_radius_ws;    // Durchmesser (WS) nach Pixel-Clamp
    float pass_screen_size;  // Durchmesser in Pixeln
} fsIn;

layout(location = 0) out vec4 out_color;

INCLUDE vis_color_no_prov.glsl

void main() {
    // Kreisscheibe maskieren
    if (length(fsIn.pass_uv_coords) > 1.0) discard;

    vec3 finalColor = get_color(
        fsIn.pass_world_pos,
        fsIn.pass_vs_normal,
        fsIn.pass_point_color,
        fsIn.pass_radius_ws,
        fsIn.pass_screen_size
    );

    out_color = vec4(finalColor, 1.0);
}