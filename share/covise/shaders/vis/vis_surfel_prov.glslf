#version 420 core

in FS_IN {
    vec3  pass_point_color;
    vec2  pass_uv_coords;
    vec3  pass_world_pos;
    vec3  pass_normal_ws;
    float pass_radius_ws;     // Radius (WS) nach Clamp
    float pass_prov1;
    float pass_prov2;
    float pass_prov3;
    float pass_prov4;
    float pass_prov5;
    float pass_prov6;
    float pass_screen_size;   // Pixel-Radius nach Clamp
} fsIn;

layout(location = 0) out vec4 out_color;

uniform bool  show_normals;
uniform bool  show_accuracy;
uniform bool  show_radius_deviation;
uniform bool  show_output_sensitivity;
uniform float accuracy;
uniform float average_radius;
uniform int   channel;
uniform bool  heatmap;
uniform float heatmap_min;
uniform float heatmap_max;
uniform vec3  heatmap_min_color;
uniform vec3  heatmap_max_color;

INCLUDE vis_color_prov.glsl

void main() {
    // Kreismaske
    if (length(fsIn.pass_uv_coords) > 1.0) discard;

    ColorProvDebugParams dbg = ColorProvDebugParams(
        show_normals,
        show_accuracy,
        show_radius_deviation,
        show_output_sensitivity,
        accuracy,
        average_radius,
        channel,
        heatmap,
        heatmap_min,
        heatmap_max,
        heatmap_min_color,
        heatmap_max_color
    );

    vec3 col = get_color(
        fsIn.pass_world_pos,
        fsIn.pass_normal_ws,
        fsIn.pass_point_color,
        fsIn.pass_radius_ws,
        fsIn.pass_screen_size,
        fsIn.pass_prov1, fsIn.pass_prov2, fsIn.pass_prov3,
        fsIn.pass_prov4, fsIn.pass_prov5, fsIn.pass_prov6,
        dbg
    );
    out_color = vec4(col, 1.0);
}
