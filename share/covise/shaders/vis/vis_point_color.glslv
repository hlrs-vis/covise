#version 420 core
INCLUDE vis_point_util.glsl
INCLUDE vis_clip_util.glsl

uniform int clip_plane_count;
uniform vec4 clip_planes[LAMURE_MAX_CLIP_PLANES];
uniform mat4 model_matrix;

layout(location = 0) in vec3  in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 4) in float empty;
layout(location = 5) in float in_radius;   // Roh-Radius
layout(location = 6) in vec3  in_normal;

uniform mat4 mvp_matrix;
uniform mat4 view_matrix;
uniform mat3 normal_matrix;

uniform float min_radius;          // Weltmaß-Clamp (nach Shaping/Scale)
uniform float max_radius;          // Weltmaß-Clamp (nach Shaping/Scale)
uniform float min_screen_size;     // Pixel-Clamp
uniform float max_screen_size;     // Pixel-Clamp
uniform float scale_radius;        // Roh -> Weltmaß
uniform float scale_projection;    // Weltmaß -> Pixel
uniform float scale_radius_gamma;  // Gamma fürs Shaping (z.B. 0.5)
uniform float max_radius_cut;      // Weltmaß-Cut (wirkt VOR Skalierung)
uniform bool  use_aniso;
uniform vec4  Pcol0;
uniform vec4  Pcol1;
uniform float viewport_half_y;
uniform float aniso_normalize;

out VertexData {
    vec3  pass_point_color;
    vec3  pass_world_pos;
    vec3  pass_vs_pos;
    vec3  pass_vs_normal;
    float pass_radius_ws;
    float pass_screen_size;
} VertexOut;

void main() {
    const float EPS = 1e-6;

    if (clip_plane_count > 0)
    {
        vec4 clipWorldPos = model_matrix * vec4(in_position, 1.0);
        lamure_apply_clip_planes(clipWorldPos, clip_plane_count, clip_planes);
    }
    else
    {
        lamure_apply_clip_planes(vec4(0.0), 0, clip_planes);
    }

    vec4 worldPos4 = vec4(in_position, 1.0);
    vec4 vsPos4    = view_matrix * worldPos4;
    vec3 nVS       = normalize(normal_matrix * in_normal);

    vec4 clipPos   = mvp_matrix * worldPos4;
    gl_Position    = clipPos;

    float r_raw = max(0.0, in_radius);
    if (max_radius_cut > 0.0 && r_raw > max_radius_cut) {
        gl_PointSize               = 0.0;
        gl_Position                = vec4(2e9, 2e9, 2e9, 1.0);
        VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
        VertexOut.pass_world_pos   = worldPos4.xyz;
        VertexOut.pass_vs_pos      = vsPos4.xyz;
        VertexOut.pass_vs_normal   = nVS;
        VertexOut.pass_radius_ws   = 0.0;
        VertexOut.pass_screen_size = 0.0;
        return;
    }

    float r_ws = calc_world_radius(
        r_raw, max_radius_cut, scale_radius_gamma, scale_radius, min_radius, max_radius);

    if (abs(clipPos.w) <= EPS || r_ws <= EPS) {
        gl_PointSize               = 0.0;
        VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
        VertexOut.pass_world_pos   = worldPos4.xyz;
        VertexOut.pass_vs_pos      = vsPos4.xyz;
        VertexOut.pass_vs_normal   = nVS;
        VertexOut.pass_radius_ws   = r_ws;
        VertexOut.pass_screen_size = 0.0;
        return;
    }

    float diameter_px = use_aniso
        ? calc_diameter_px_aniso(clipPos, r_ws, Pcol0, Pcol1, viewport_half_y, aniso_normalize,
                                 min_screen_size, max_screen_size)
        : calc_diameter_px_iso(clipPos, r_ws, scale_projection, min_screen_size, max_screen_size);

    if (diameter_px <= EPS) {
        gl_PointSize               = 0.0;
        VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
        VertexOut.pass_world_pos   = worldPos4.xyz;
        VertexOut.pass_vs_pos      = vsPos4.xyz;
        VertexOut.pass_vs_normal   = nVS;
        VertexOut.pass_radius_ws   = r_ws;
        VertexOut.pass_screen_size = 0.0;
        return;
    }

    gl_PointSize = diameter_px;

    VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
    VertexOut.pass_world_pos   = worldPos4.xyz;
    VertexOut.pass_vs_pos      = vsPos4.xyz;
    VertexOut.pass_vs_normal   = nVS;
    VertexOut.pass_radius_ws   = r_ws;
    VertexOut.pass_screen_size = diameter_px;
}
